import os
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime
from io import StringIO, BytesIO

import azure.functions as func
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
STORAGE_ACCOUNT_URL = os.getenv("STORAGE_ACCOUNT_URL")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")
BLOB_MODEL_NAME = os.getenv("BLOB_MODEL_NAME")
OUTPUT_CONTAINER_NAME = os.getenv("OUTPUT_CONTAINER_NAME", BLOB_CONTAINER_NAME)
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "predictions")
DEFAULT_MASTER_FILE = os.getenv("DEFAULT_MASTER_FILE", "sla_output_results.csv")

LOCAL_MODEL_PATH = Path(r"C:\temp\Model\catboost_model.cbm")
MODEL: CatBoostRegressor | None = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _get_credential() -> DefaultAzureCredential:
    client_id = os.getenv("AZURE_CLIENT_ID") or os.getenv("MANAGED_IDENTITY_CLIENT_ID")
    if not client_id and os.getenv("WEBSITE_INSTANCE_ID"):
        client_id = "89ab0e3a-79c6-4737-8946-e6e7bbda01c0"
    return DefaultAzureCredential(managed_identity_client_id=client_id)


def _get_blob_client(container: str, blob: str):
    credential = _get_credential()
    svc = BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=credential)
    return svc.get_blob_client(container=container, blob=blob)


def load_model_if_needed() -> None:
    global MODEL
    if MODEL is not None: return

    try:
        temp_path = Path(tempfile.gettempdir()) / (BLOB_MODEL_NAME or "model.cbm")
        blob_client = _get_blob_client(BLOB_CONTAINER_NAME, BLOB_MODEL_NAME)

        with open(temp_path, "wb") as f:
            blob_client.download_blob().readinto(f)

        MODEL = CatBoostRegressor()
        MODEL.load_model(str(temp_path))
        temp_path.unlink(missing_ok=True)
    except Exception as e:
        logging.warning(f"Blob load failed: {e}. Trying local fallback.")
        MODEL = CatBoostRegressor()
        MODEL.load_model(str(LOCAL_MODEL_PATH))


def adjust_for_weekend(dt: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(dt) or not hasattr(dt, "weekday"): return dt
    wd = dt.weekday()
    if wd == 5: return dt + pd.Timedelta(days=2)
    if wd == 6: return dt + pd.Timedelta(days=1)
    return dt


def ensure_features(df: pd.DataFrame, required_features: List[str], cat_indices: List[int]) -> pd.DataFrame:
    cat_set = {required_features[i] for i in cat_indices if 0 <= i < len(required_features)}
    for col in required_features:
        if col not in df.columns:
            df[col] = "Unknown" if col in cat_set else 0
        if col in cat_set:
            df[col] = df[col].astype(str).fillna("Unknown")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df[required_features]


def save_results_to_blob(df_to_save: pd.DataFrame, blob_path: str, deduplicate_column: str = 'TicketNumber') -> Dict[
    str, Any]:
    try:
        blob_client = _get_blob_client(OUTPUT_CONTAINER_NAME, blob_path)
        existing_df = pd.DataFrame()
        operation = "created"

        if blob_client.exists():
            existing_data = blob_client.download_blob().readall()
            existing_df = pd.read_csv(BytesIO(existing_data))
            operation = "appended"

        if not existing_df.empty:
            final_df = pd.concat([existing_df, df_to_save], ignore_index=True)
            if deduplicate_column in final_df.columns:
                final_df = final_df.drop_duplicates(subset=[deduplicate_column], keep='last')
        else:
            final_df = df_to_save

        final_df['Last Updated'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        blob_client.upload_blob(final_df.to_csv(index=False), overwrite=True)
        return {"success": True, "operation": operation, "total_records": len(final_df)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def prepare_final_output(df: pd.DataFrame, predictions_minutes: np.ndarray, predicted_dates: pd.Series) -> pd.DataFrame:
    results = pd.DataFrame()
    results["TicketNumber"] = df["TicketNumber"] if "TicketNumber" in df.columns else range(1, len(df) + 1)

    if "msdyn_receiveddate" in df.columns:
        results["Received Date"] = pd.to_datetime(df["msdyn_receiveddate"], errors="coerce").dt.strftime(
            "%m/%d/%Y %H:%M:%S")

    results["Predicted Resolution Date"] = pd.to_datetime(predicted_dates, errors="coerce").dt.strftime(
        "%m/%d/%Y %H:%M:%S")
    results["Predicted Duration (Mins)"] = predictions_minutes.astype(int)

    # Automatically map your 11+ additional business columns
    additional_columns = [
        "msdyn_caserocname", "msdyn_businessfunctionidname", "msdyn_lineofbusinessidname",
        "msdyn_programidname", "msdyn_casetypeidname", "msdyn_casesubtypeidname",
        "msdyn_casereasonidname", "msdyn_casesubreasonidname", "prioritycodename",
        "msdyn_countrysubmitteridname", "msdyn_countryprocessedidname"
    ]
    for col in additional_columns:
        results[col] = df[col].values if col in df.columns else None

    results["SLA Status"] = 'Pending'
    return results


# -----------------------------------------------------------------------------
# Azure Function Endpoints
# -----------------------------------------------------------------------------
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


@app.function_name(name="upload_local_to_blob")
@app.route(route="upload_local_to_blob", methods=["POST"])
def storage_upload(req: func.HttpRequest) -> func.HttpResponse:
    """Relay endpoint: Local PC -> Azure Function -> Blob Storage"""
    try:
        blob_name = req.params.get('blob')
        if not blob_name:
            return func.HttpResponse("Error: URL must contain ?blob=filename.csv", status_code=400)

        file_data = req.get_body()
        blob_client = _get_blob_client(BLOB_CONTAINER_NAME, blob_name)
        blob_client.upload_blob(file_data, overwrite=True)

        return func.HttpResponse(json.dumps({"success": True, "path": blob_name}), status_code=200)
    except Exception as e:
        return func.HttpResponse(str(e), status_code=500)


@app.function_name(name="predict")
@app.route(route="predict", methods=["POST"])
def predict(req: func.HttpRequest) -> func.HttpResponse:
    try:
        load_model_if_needed()
        payload = req.get_json()
        records = payload.get("data", payload)
        if not isinstance(records, list): records = [records]

        df = pd.DataFrame(records)
        df["received_dt"] = pd.to_datetime(df["msdyn_receiveddate"], errors="coerce")
        df["hour_of_day"] = df["received_dt"].dt.hour.fillna(0).astype(int)
        df["day_of_week"] = df["received_dt"].dt.dayofweek.fillna(0).astype(int)
        df["daily_volume"] = 1

        df_model = ensure_features(df, MODEL.feature_names_, MODEL.get_cat_feature_indices())
        preds = MODEL.predict(Pool(data=df_model, cat_features=MODEL.get_cat_feature_indices()))

        preds_minutes = np.maximum(np.expm1(preds), 0).round(0)
        pred_resolved_str = (df["received_dt"] + pd.to_timedelta(preds_minutes, unit="m")).apply(
            adjust_for_weekend).dt.strftime("%Y-%m-%d %H:%M:%S")

        final_results = prepare_final_output(df, preds_minutes, pred_resolved_str)

        if payload.get("save_to_blob", True):
            save_results_to_blob(final_results, f"{OUTPUT_FOLDER}/{payload.get('filename', DEFAULT_MASTER_FILE)}")

        return func.HttpResponse(json.dumps({"success": True, "results": final_results.to_dict(orient="records")}),
                                 status_code=200)
    except Exception as e:
        return func.HttpResponse(str(e), status_code=500)

