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
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Ensure these are set in Azure Function App Settings
STORAGE_ACCOUNT_URL = os.getenv("STORAGE_ACCOUNT_URL")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")
BLOB_MODEL_NAME = os.getenv("BLOB_MODEL_NAME")
OUTPUT_CONTAINER_NAME = os.getenv("OUTPUT_CONTAINER_NAME", BLOB_CONTAINER_NAME)
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "predictions")

# Default master file name (can be overridden in request)
DEFAULT_MASTER_FILE = os.getenv("DEFAULT_MASTER_FILE", "sla_output_results.csv")

# Hard-coded local fallback
LOCAL_MODEL_PATH = Path(r"C:\temp\Model\catboost_model.cbm")

# Global model (lazy-loaded for performance)
MODEL: CatBoostRegressor | None = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _get_credential() -> DefaultAzureCredential:
    """Managed identity selection for POC; uses explicit client_id when present."""
    client_id = os.getenv("AZURE_CLIENT_ID") or os.getenv("MANAGED_IDENTITY_CLIENT_ID")
    if not client_id and os.getenv("WEBSITE_INSTANCE_ID"):
        client_id = "89ab0e3a-79c6-4737-8946-e6e7bbda01c0"
    return DefaultAzureCredential(managed_identity_client_id=client_id)


def _load_model_from_path(p: Path) -> CatBoostRegressor:
    m = CatBoostRegressor()
    m.load_model(str(p))
    return m


def _get_blob_client(container: str, blob: str):
    """Get a blob client for a specific blob"""
    credential = _get_credential()
    svc = BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=credential)
    return svc.get_blob_client(container=container, blob=blob)


def _get_blob_service_client() -> BlobServiceClient:
    """Get the blob service client"""
    credential = _get_credential()
    return BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=credential)


def load_model_if_needed() -> None:
    """
    POC loader:
      1) Try to load model from Azure Blob (download -> temp -> load).
      2) If blob missing/failed, load from local hard-coded path: C:\\temp\\Model\\catboost_model.cbm
    """
    global MODEL
    if MODEL is not None:
        logging.info("MODEL already loaded; skipping reload.")
        return

    # ---------- Try Blob first ----------
    blob_attempted = False
    try:
        if not all([STORAGE_ACCOUNT_URL, BLOB_CONTAINER_NAME, BLOB_MODEL_NAME]):
            logging.info("Blob env vars missing; will skip Blob and try local path.")
            raise ValueError("Missing blob config")

        temp_path = Path(tempfile.gettempdir()) / (BLOB_MODEL_NAME or "model.cbm")
        logging.info(f"Trying to load model from Blob: container='{BLOB_CONTAINER_NAME}', blob='{BLOB_MODEL_NAME}'")
        blob_attempted = True

        blob_client = _get_blob_client(BLOB_CONTAINER_NAME, BLOB_MODEL_NAME)

        if hasattr(blob_client, "exists") and not blob_client.exists():
            logging.info("Blob does not exist; will try local path.")
            raise ResourceNotFoundError("Blob not found")

        with open(temp_path, "wb") as f:
            blob_client.download_blob().readinto(f)
        logging.info(f"Downloaded model blob to temp file: {temp_path}")

        MODEL = _load_model_from_path(temp_path)
        logging.info("CatBoost model loaded from Blob successfully.")

        try:
            temp_path.unlink()
            logging.info("Temp file deleted after successful blob load.")
        except Exception:
            logging.info("Temp file delete failed (non-blocking).")

        return

    except ResourceNotFoundError:
        if blob_attempted:
            logging.info("Blob not found (ResourceNotFound). Will try local fallback next.")
    except Exception as e:
        if blob_attempted:
            logging.info(f"Blob load failed with error: {e}. Will try local fallback next.")

    # ---------- Fallback: Local hard-coded path ----------
    try:
        logging.info(f"Trying to load model from local path: {LOCAL_MODEL_PATH}")
        MODEL = _load_model_from_path(LOCAL_MODEL_PATH)
        logging.info("CatBoost model loaded from local path successfully.")
        return
    except Exception as e:
        logging.info(f"Local load failed: {e}")

    logging.info("Model load failed from both Blob and local path.")
    raise RuntimeError("Could not load model from Blob or local path.")


def adjust_for_weekend(dt: pd.Timestamp) -> pd.Timestamp:
    """Adjust date to next business day if it falls on weekend"""
    if pd.isna(dt) or not hasattr(dt, "weekday"):
        return dt
    wd = dt.weekday()
    if wd == 5:
        return dt + pd.Timedelta(days=2)  # Sat -> Mon
    if wd == 6:
        return dt + pd.Timedelta(days=1)  # Sun -> Mon
    return dt


def ensure_features(df: pd.DataFrame, required_features: List[str], cat_indices: List[int]) -> pd.DataFrame:
    """Ensure all required features are present in the dataframe"""
    cat_set = {required_features[i] for i in cat_indices if 0 <= i < len(required_features)}
    for col in required_features:
        if col not in df.columns:
            df[col] = "Unknown" if col in cat_set else 0
        if col in cat_set:
            df[col] = df[col].astype(str).fillna("Unknown")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df[required_features]


def calculate_sla_status(actual_duration: float, predicted_duration: float, threshold_percent: float = 10) -> str:
    """
    Calculate SLA status based on prediction accuracy

    Parameters:
    -----------
    actual_duration : float
        Actual resolution duration in minutes
    predicted_duration : float
        Predicted resolution duration in minutes
    threshold_percent : float
        Threshold percentage for determining if prediction is within SLA (default 10%)

    Returns:
    --------
    str : "Met" if prediction is within threshold, "Breached" otherwise, "Unknown" if actual is missing
    """
    if pd.isna(actual_duration) or actual_duration <= 0:
        return "Unknown"

    if pd.isna(predicted_duration) or predicted_duration <= 0:
        return "Unknown"

    # Calculate percentage difference
    diff_percent = abs(actual_duration - predicted_duration) / actual_duration * 100

    if diff_percent <= threshold_percent:
        return "Met"
    else:
        return "Breached"


def read_existing_predictions(blob_path: str) -> pd.DataFrame:
    """
    Read existing predictions CSV from blob storage

    Parameters:
    -----------
    blob_path : str
        Path to the blob file

    Returns:
    --------
    pd.DataFrame : Existing predictions or empty DataFrame with correct schema
    """
    try:
        blob_client = _get_blob_client(OUTPUT_CONTAINER_NAME, blob_path)

        # Check if blob exists
        if hasattr(blob_client, "exists") and blob_client.exists():
            # Download and read CSV
            blob_data = blob_client.download_blob()
            csv_string = blob_data.readall().decode('utf-8')
            existing_df = pd.read_csv(StringIO(csv_string))
            logging.info(f"Loaded {len(existing_df)} existing records from {blob_path}")
            return existing_df
        else:
            logging.info(f"Blob {blob_path} does not exist. Will create new file.")
            return pd.DataFrame()

    except ResourceNotFoundError:
        logging.info(f"Blob {blob_path} not found. Will create new file.")
        return pd.DataFrame()
    except Exception as e:
        logging.warning(f"Error reading existing predictions: {e}. Will create new file.")
        return pd.DataFrame()


def save_results_to_blob(df_to_save: pd.DataFrame, blob_path: str, deduplicate_column: str = 'TicketNumber') -> Dict[
    str, Any]:
    """
    Save or append results to Azure Blob Storage CSV file

    Parameters:
    -----------
    df_to_save : pd.DataFrame
        New records to save
    blob_path : str
        Full blob path (e.g., "predictions/sla_output_results.csv")
    deduplicate_column : str
        Column to use for deduplication (default: 'TicketNumber')

    Returns:
    --------
    dict : Information about the save operation
    """
    try:
        blob_client = _get_blob_client(OUTPUT_CONTAINER_NAME, blob_path)

        # Check if file exists
        existing_df = pd.DataFrame()
        operation = "created"

        try:
            if hasattr(blob_client, "exists") and blob_client.exists():
                # Download existing data
                existing_data = blob_client.download_blob().readall()
                existing_df = pd.read_csv(BytesIO(existing_data))
                operation = "appended"
                logging.info(f"Found existing file with {len(existing_df)} records")
        except Exception as e:
            logging.info(f"No existing file found or error reading: {e}. Creating new file.")

        # Combine datasets
        if not existing_df.empty:
            # Append new records to existing
            final_df = pd.concat([existing_df, df_to_save], ignore_index=True)

            # Deduplicate by specified column (keep latest)
            if deduplicate_column in final_df.columns:
                original_count = len(final_df)
                final_df = final_df.drop_duplicates(subset=[deduplicate_column], keep='last')
                duplicates_removed = original_count - len(final_df)
                logging.info(f"Removed {duplicates_removed} duplicate records based on {deduplicate_column}")
        else:
            final_df = df_to_save

        # Add timestamp for tracking
        final_df['Last Updated'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

        # Convert to CSV and upload
        csv_data = final_df.to_csv(index=False)
        blob_client.upload_blob(csv_data, overwrite=True)

        logging.info(f"Successfully saved {len(df_to_save)} new records. Total records: {len(final_df)}")

        return {
            "success": True,
            "blob_url": blob_client.url,
            "blob_path": blob_path,
            "operation": operation,
            "new_records": len(df_to_save),
            "total_records": len(final_df),
            "duplicates_removed": len(existing_df) + len(df_to_save) - len(final_df) if not existing_df.empty else 0
        }

    except Exception as e:
        logging.error(f"Failed to save to Azure Blob: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def prepare_final_output(df: pd.DataFrame, predictions_minutes: np.ndarray,
                         predicted_dates: pd.Series) -> pd.DataFrame:
    """
    Prepare final output dataframe with all required columns

    Columns include:
    - TicketNumber
    - Received Date
    - Predicted Resolution Date
    - Actual Resolution Date
    - Actual Duration (Mins)
    - Predicted Duration (Mins)
    - Delta (Mins)
    - SLA_Status
    - Additional business columns
    """
    results = pd.DataFrame()

    # Ticket ID/Number
    if "TicketNumber" in df.columns:
        results["TicketNumber"] = df["TicketNumber"]
    else:
        results["TicketNumber"] = range(1, len(df) + 1)

    # Received Date
    if "msdyn_receiveddate" in df.columns:
        received_dt = pd.to_datetime(df["msdyn_receiveddate"], errors="coerce")
        results["Received Date"] = received_dt.dt.strftime("%m/%d/%Y %H:%M:%S")
    else:
        results["Received Date"] = None

    # Predicted Resolution Date
    pred_dt = pd.to_datetime(predicted_dates, errors="coerce")
    results["Predicted Resolution Date"] = pred_dt.dt.strftime("%m/%d/%Y %H:%M:%S")

    # Actual Resolution Date
    actual_res_date_col = None
    for col_name in ["msdyn_resolveddate", "resolved_date", "actual_resolved_date", "modifiedon"]:
        if col_name in df.columns:
            actual_res_date_col = col_name
            break

    has_actual_date = False
    if actual_res_date_col:
        actual_dt = pd.to_datetime(df[actual_res_date_col], errors="coerce")
        if actual_dt.notna().any():
            has_actual_date = True
            results["Actual Resolution Date"] = actual_dt.dt.strftime("%m/%d/%Y %H:%M:%S")
        else:
            results["Actual Resolution Date"] = None
    else:
        results["Actual Resolution Date"] = None

    # Predicted Duration (Mins)
    results["Predicted Duration (Mins)"] = predictions_minutes.round(0).astype(int)

    # Actual Duration (Mins)
    if actual_res_date_col and "msdyn_receiveddate" in df.columns:
        received_dt = pd.to_datetime(df["msdyn_receiveddate"], errors="coerce")
        resolved_dt = pd.to_datetime(df[actual_res_date_col], errors="coerce")
        actual_duration = (resolved_dt - received_dt).dt.total_seconds() / 60
        results["Actual Duration (Mins)"] = actual_duration.round(0).astype('Int64')
    elif "actual_duration" in df.columns:
        results["Actual Duration (Mins)"] = pd.to_numeric(df["actual_duration"], errors="coerce").round(0).astype(
            'Int64')
    else:
        results["Actual Duration (Mins)"] = None

    # Delta (Mins)
    if results["Actual Duration (Mins)"].notna().any():
        results["Delta (Mins)"] = (results["Actual Duration (Mins)"] - results["Predicted Duration (Mins)"]).round(0)
    else:
        results["Delta (Mins)"] = None

    # SLA Status - Enhanced logic similar to app.py
    if has_actual_date and "Actual Resolution Date" in results.columns:
        # Calculate SLA status based on delta between predicted and actual
        delta_seconds = (actual_dt - pred_dt).dt.total_seconds()
        delta_minutes = delta_seconds / 60
        results["SLA Status"] = delta_minutes.apply(
            lambda x: 'Delay' if x > 0 else ('Early' if x < 0 else 'On Time')
        )
    else:
        results["SLA Status"] = 'Pending'

    # ========================================================================
    # ADDITIONAL BUSINESS COLUMNS (from app.py requirements)
    # ========================================================================
    additional_columns = [
        "msdyn_caserocname",
        "msdyn_businessfunctionidname",
        "msdyn_lineofbusinessidname",
        "msdyn_programidname",
        "msdyn_casetypeidname",
        "msdyn_casesubtypeidname",
        "msdyn_casereasonidname",
        "msdyn_casesubreasonidname",
        "prioritycodename",
        "msdyn_countrysubmitteridname",
        "msdyn_countryprocessedidname"
    ]

    for col in additional_columns:
        if col in df.columns:
            results[col] = df[col].values
        else:
            results[col] = None

    return results


# -----------------------------------------------------------------------------
# Azure Function
# -----------------------------------------------------------------------------
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


@app.function_name(name="predict")
@app.route(route="predict", methods=["POST"])
def predict(req: func.HttpRequest) -> func.HttpResponse:
    """
    Main prediction endpoint that accepts JSON input, makes predictions,
    and automatically saves results to blob storage

    Expected JSON format (single record or array):
    {
        "TicketNumber": "12345",
        "msdyn_receiveddate": "2026-01-15T10:30:00",
        "prioritycodename": "High",
        ... other features ...
    }

    OR

    [
        { "TicketNumber": "12345", ... },
        { "TicketNumber": "12346", ... }
    ]

    Optional parameters can be included:
    {
        "data": [...],                    # Records to process
        "save_to_blob": true,             # Default: true
        "filename": "custom.csv",         # Default: sla_output_results.csv
        "deduplicate_column": "TicketNumber"  # Default: TicketNumber
    }
    """
    try:
        # Load model
        load_model_if_needed()

        # Parse request
        try:
            payload = req.get_json()
            logging.info(f"Received prediction request")
        except ValueError:
            return func.HttpResponse(
                json.dumps({"success": False, "error": "Invalid JSON"}),
                status_code=400,
                mimetype="application/json"
            )

        # Extract parameters
        save_to_blob = payload.get("save_to_blob", True)
        custom_filename = payload.get("filename", DEFAULT_MASTER_FILE)
        deduplicate_column = payload.get("deduplicate_column", "TicketNumber")

        # Get data - support both formats
        if "data" in payload:
            records = payload["data"]
        else:
            # Assume the entire payload is the data
            # Filter out control parameters
            control_params = {"save_to_blob", "filename", "deduplicate_column"}
            records = {k: v for k, v in payload.items() if k not in control_params}

        # Support both single object and list of objects
        if not isinstance(records, list):
            records = [records]

        df = pd.DataFrame(records)
        logging.info(f"Processing {len(df)} records")

        # Basic Feature Engineering
        if "msdyn_receiveddate" in df.columns:
            df["received_dt"] = pd.to_datetime(df["msdyn_receiveddate"], errors="coerce")
        else:
            df["received_dt"] = pd.NaT

        df["hour_of_day"] = df["received_dt"].dt.hour.fillna(0).astype(int)
        df["day_of_week"] = df["received_dt"].dt.dayofweek.fillna(0).astype(int)
        df["daily_volume"] = 1

        # Model Inference
        required_features = MODEL.feature_names_
        cat_indices = MODEL.get_cat_feature_indices()
        df_model = ensure_features(df, required_features, cat_indices)

        pool = Pool(data=df_model, cat_features=cat_indices if cat_indices else None)
        preds = MODEL.predict(pool)

        # Resolution Logic
        preds_minutes = np.expm1(preds)
        preds_minutes = np.maximum(preds_minutes, 0).round(0)

        # Calculate predicted resolution dates
        if not df["received_dt"].isna().all():
            raw_finish = df["received_dt"] + pd.to_timedelta(preds_minutes, unit="m")
            pred_resolved = raw_finish.apply(adjust_for_weekend)
            pred_resolved_str = pred_resolved.dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            pred_resolved_str = pd.Series([None] * len(df))

        # Prepare final output (matching app.py format)
        final_results = prepare_final_output(df, preds_minutes, pred_resolved_str)

        # Prepare response
        response_data = {
            "success": True,
            "records_processed": len(df),
            "results": final_results.to_dict(orient="records")
        }

        # Save to blob storage (automatic update/append)
        if save_to_blob:
            try:
                # Ensure filename has .csv extension
                if not custom_filename.endswith('.csv'):
                    custom_filename = f"{custom_filename}.csv"

                # Create full blob path
                blob_path = f"{OUTPUT_FOLDER}/{custom_filename}"

                # Save/update results
                save_info = save_results_to_blob(final_results, blob_path, deduplicate_column)

                if save_info["success"]:
                    response_data.update({
                        "blob_saved": True,
                        "blob_url": save_info["blob_url"],
                        "blob_path": save_info["blob_path"],
                        "operation": save_info["operation"],
                        "new_records": save_info["new_records"],
                        "total_records": save_info["total_records"],
                        "duplicates_removed": save_info.get("duplicates_removed", 0)
                    })
                    logging.info(
                        f"Results saved: {save_info['operation']} - {save_info['new_records']} new, {save_info['total_records']} total")
                else:
                    response_data["blob_saved"] = False
                    response_data["blob_error"] = save_info.get("error", "Unknown error")

            except Exception as e:
                logging.error(f"Failed to save to blob: {e}")
                response_data["blob_saved"] = False
                response_data["blob_error"] = str(e)

        return func.HttpResponse(
            json.dumps(response_data, indent=2),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.exception("Unhandled error in predict endpoint.")
        return func.HttpResponse(
            json.dumps({"success": False, "error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )


@app.function_name(name="health")
@app.route(route="health", methods=["GET"])
def health(req: func.HttpRequest) -> func.HttpResponse:
    """Simple health check endpoint"""
    try:
        load_model_if_needed()
        return func.HttpResponse(
            json.dumps({
                "status": "healthy",
                "model_loaded": MODEL is not None,
                "timestamp": datetime.utcnow().isoformat(),
                "output_container": OUTPUT_CONTAINER_NAME,
                "output_folder": OUTPUT_FOLDER,
                "default_file": DEFAULT_MASTER_FILE
            }),
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        return func.HttpResponse(
            json.dumps({
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }),
            status_code=500,
            mimetype="application/json"
        )

