import os
import io
import json
import yaml
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime
from io import StringIO, BytesIO

import numpy as np
import pandas as pd

import azure.functions as func
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Azure Functions App (Python v2 programming model)
# =============================================================================
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# =============================================================================
# Shared Configuration
# =============================================================================
STORAGE_ACCOUNT_URL = os.getenv("STORAGE_ACCOUNT_URL")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")

# Model blob path/name
BLOB_MODEL_NAME = os.getenv("BLOB_MODEL_NAME", "model/catboost_model_deploy.cbm")

# Inference output
OUTPUT_CONTAINER_NAME = os.getenv("OUTPUT_CONTAINER_NAME", BLOB_CONTAINER_NAME)
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "predictions")
DEFAULT_MASTER_FILE = os.getenv("DEFAULT_MASTER_FILE", "sla_output_results.csv")

# Global model cache
MODEL: CatBoostRegressor | None = None

# =============================================================================
# Identity / Storage helpers
# =============================================================================
def _get_credential() -> DefaultAzureCredential:
    client_id = os.getenv("AZURE_CLIENT_ID") or os.getenv("MANAGED_IDENTITY_CLIENT_ID")
    return DefaultAzureCredential(managed_identity_client_id=client_id)

def _get_blob_service_client() -> BlobServiceClient:
    return BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=_get_credential())

def _get_container_client(container_name: str):
    return _get_blob_service_client().get_container_client(container_name)

def _get_blob_client(container: str, blob: str):
    return _get_blob_service_client().get_blob_client(container=container, blob=blob)

# =============================================================================
# Model load helpers
# =============================================================================
def load_model_if_needed() -> None:
    global MODEL
    if MODEL is not None:
        return

    try:
        temp_path = Path(tempfile.gettempdir()) / "model_temp.cbm"
        logging.info(f"Downloading model from blob: {BLOB_MODEL_NAME}")
        
        blob_client = _get_blob_client(BLOB_CONTAINER_NAME, BLOB_MODEL_NAME)
        with open(temp_path, "wb") as f:
            blob_client.download_blob().readinto(f)

        MODEL = CatBoostRegressor()
        MODEL.load_model(str(temp_path))
        
        if temp_path.exists():
            temp_path.unlink()
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Could not load model from Blob: {e}")

# =============================================================================
# Inference utilities
# =============================================================================
def adjust_for_weekend(dt: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(dt) or not hasattr(dt, "weekday"):
        return dt
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

def save_results_to_blob(df_to_save: pd.DataFrame, blob_path: str, deduplicate_column: str = "TicketNumber") -> Dict[str, Any]:
    try:
        blob_client = _get_blob_client(OUTPUT_CONTAINER_NAME, blob_path)
        existing_df = pd.DataFrame()
        
        try:
            if blob_client.exists():
                data = blob_client.download_blob().readall()
                existing_df = pd.read_csv(BytesIO(data))
        except:
            pass

        final_df = pd.concat([existing_df, df_to_save], ignore_index=True) if not existing_df.empty else df_to_save
        if deduplicate_column in final_df.columns:
            final_df = final_df.drop_duplicates(subset=[deduplicate_column], keep="last")

        final_df["Last Updated"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        blob_client.upload_blob(final_df.to_csv(index=False), overwrite=True)
        return {"success": True, "blob_url": blob_client.url}
    except Exception as e:
        return {"success": False, "error": str(e)}

def prepare_final_output(df: pd.DataFrame, predictions_minutes: np.ndarray, predicted_dates: pd.Series) -> pd.DataFrame:
    results = pd.DataFrame()
    results["TicketNumber"] = df["TicketNumber"] if "TicketNumber" in df.columns else range(1, len(df) + 1)
    results["Predicted Resolution Date"] = pd.to_datetime(predicted_dates).dt.strftime("%m/%d/%Y %H:%M:%S")
    results["Predicted Duration (Mins)"] = predictions_minutes.astype(int)
    
    # Add category info for context
    cols = ["msdyn_caserocname", "msdyn_casereasonidname", "prioritycodename"]
    for c in cols:
        if c in df.columns: results[c] = df[c].values
    return results

# =============================================================================
# Training helpers
# =============================================================================
def load_blob_to_obj(blob_name: str):
    # This function is now the ONLY way data is loaded
    container_client = _get_container_client(BLOB_CONTAINER_NAME)
    blob_client = container_client.get_blob_client(blob_name)
    stream = blob_client.download_blob().readall()

    if blob_name.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(stream), engine="openpyxl")
    if blob_name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(stream))
    return stream

def clean_and_engineer(df, quantile_limit=0.75):
    df["received_dt"] = pd.to_datetime(df["msdyn_receiveddate"], errors="coerce")
    df["resolved_dt"] = pd.to_datetime(df["msdyn_resolveddate"], errors="coerce")
    df = df.dropna(subset=["received_dt", "resolved_dt"])
    df["target_minutes"] = (df["resolved_dt"] - df["received_dt"]).dt.total_seconds() / 60
    df = df[(df["target_minutes"] > 30) & (df["target_minutes"] < 43200)]
    
    df = df.sort_values("received_dt").set_index("received_dt")
    df["backlog_4h"] = df["msdyn_caserocname"].rolling(window="4h").count() - 1
    df = df.reset_index()
    
    df["hour_of_day"] = df["received_dt"].dt.hour.astype(str)
    df["day_of_week"] = df["received_dt"].dt.dayofweek.astype(str)
    
    upper_limit = df["target_minutes"].quantile(quantile_limit)
    df_clean = df[df["target_minutes"] < upper_limit].copy()
    median_map = df_clean.groupby("msdyn_casereasonidname")["target_minutes"].median()
    df_clean["reason_median_speed"] = df_clean["msdyn_casereasonidname"].map(median_map)
    return df_clean, median_map

# =============================================================================
# Main Functions
# =============================================================================

@app.function_name(name="catboost_infer")
@app.route(route="catboost_infer", methods=["POST"])
def catboost_infer(req: func.HttpRequest) -> func.HttpResponse:
    try:
        load_model_if_needed()
        payload = req.get_json()
        records = payload.get("data", [payload])
        df = pd.DataFrame(records)

        if "msdyn_receiveddate" in df.columns:
            df["received_dt"] = pd.to_datetime(df["msdyn_receiveddate"], errors="coerce")
            df["hour_of_day"] = df["received_dt"].dt.hour.fillna(0).astype(int)
            df["day_of_week"] = df["received_dt"].dt.dayofweek.fillna(0).astype(int)
        
        df["daily_volume"] = 1
        
        required_features = MODEL.feature_names_
        cat_indices = MODEL.get_cat_feature_indices()
        df_model = ensure_features(df, required_features, cat_indices)

        preds = MODEL.predict(Pool(data=df_model, cat_features=cat_indices))
        preds_minutes = np.expm1(preds)

        # Output Prep
        raw_finish = df["received_dt"] + pd.to_timedelta(preds_minutes, unit="m")
        pred_resolved = raw_finish.apply(adjust_for_weekend)
        
        final_results = prepare_final_output(df, preds_minutes, pred_resolved)
        
        if payload.get("save_to_blob", True):
            filename = payload.get("filename", DEFAULT_MASTER_FILE)
            save_results_to_blob(final_results, f"{OUTPUT_FOLDER}/{filename}")

        return func.HttpResponse(json.dumps({"success": True, "results": final_results.to_dict(orient="records")}), status_code=200)
    except Exception as e:
        return func.HttpResponse(json.dumps({"success": False, "error": str(e)}), status_code=500)

@app.function_name(name="catboost_train")
@app.route(route="catboost_train", methods=["POST"])
def catboost_train(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Training pipeline started.")
    try:
        # POINT TO BLOB PATHS ONLY (No local data/ folder)
        config = {
            "paths": {
                "raw_data": "raw/Case_06Feb2026.xlsx", # Path inside your blob container
                "plot_folder": "visualization/figures"
            },
            "params": {
                "target_col": "target_minutes",
                "candidate_features": ["msdyn_caserocname", "msdyn_casereasonidname", "hour_of_day", "day_of_week", "reason_median_speed"],
                "categorical_cols": ["msdyn_caserocname", "msdyn_casereasonidname", "hour_of_day", "day_of_week"]
            }
        }

        # 1. Load from Blob
        df_raw = load_blob_to_obj(config["paths"]["raw_data"])
        df_raw = df_raw.rename(columns={"createdon": "msdyn_receiveddate"})
        
        # 2. Process
        df_clean, median_map = clean_and_engineer(df_raw)
        
        # 3. Train
        X = df_clean[config["params"]["candidate_features"]]
        y = np.log1p(df_clean[config["params"]["target_col"]])
        
        final_model = CatBoostRegressor(iterations=500, loss_function="MAE", verbose=0)
        final_model.fit(X, y, cat_features=config["params"]["categorical_cols"])

        # 4. Upload Model to Blob
        temp_model_path = Path(tempfile.gettempdir()) / "new_model.cbm"
        final_model.save_model(str(temp_model_path))
        
        with open(temp_model_path, "rb") as f:
            _get_container_client(BLOB_CONTAINER_NAME).upload_blob(name=BLOB_MODEL_NAME, data=f, overwrite=True)

        # Reset global cache
        global MODEL
        MODEL = None

        return func.HttpResponse(json.dumps({"success": True, "model_path": BLOB_MODEL_NAME}), status_code=200)
    except Exception as e:
        logging.exception("Training failed")
        return func.HttpResponse(json.dumps({"success": False, "error": str(e)}), status_code=500)

