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
# Shared Configuration (set these in Azure Function App -> Configuration)
# =============================================================================
STORAGE_ACCOUNT_URL = os.getenv("STORAGE_ACCOUNT_URL")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")

# Model blob path/name (used by both train + infer)
BLOB_MODEL_NAME = os.getenv("BLOB_MODEL_NAME", "model/catboost_model_deploy.cbm")

# Inference output
OUTPUT_CONTAINER_NAME = os.getenv("OUTPUT_CONTAINER_NAME", BLOB_CONTAINER_NAME)
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "predictions")
DEFAULT_MASTER_FILE = os.getenv("DEFAULT_MASTER_FILE", "sla_output_results.csv")

# Local fallback (mainly for local dev). In Azure, you typically rely on Blob.
LOCAL_MODEL_PATH = Path(os.getenv("LOCAL_MODEL_PATH", r"C:\temp\Model\catboost_model.cbm"))

# Global model cache
MODEL: CatBoostRegressor | None = None


# =============================================================================
# Identity / Storage helpers
# =============================================================================
def _get_credential() -> DefaultAzureCredential:
    """
    Uses Managed Identity in Azure (recommended).
    If AZURE_CLIENT_ID / MANAGED_IDENTITY_CLIENT_ID is set, uses that identity.
    """
    client_id = os.getenv("AZURE_CLIENT_ID") or os.getenv("MANAGED_IDENTITY_CLIENT_ID")
    return DefaultAzureCredential(managed_identity_client_id=client_id)


def _get_blob_service_client() -> BlobServiceClient:
    credential = _get_credential()
    return BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=credential)


def _get_container_client(container_name: str):
    svc = _get_blob_service_client()
    return svc.get_container_client(container_name)


def _get_blob_client(container: str, blob: str):
    svc = _get_blob_service_client()
    return svc.get_blob_client(container=container, blob=blob)


# =============================================================================
# Model load helpers (Inference)
# =============================================================================
def _load_model_from_path(p: Path) -> CatBoostRegressor:
    m = CatBoostRegressor()
    m.load_model(str(p))
    return m


def load_model_if_needed() -> None:
    """
    Lazy load model:
      1) Download from Blob (BLOB_CONTAINER_NAME + BLOB_MODEL_NAME) into temp file, load.
      2) Fallback to LOCAL_MODEL_PATH.
    """
    global MODEL

    if MODEL is not None:
        logging.info("MODEL already loaded; skipping reload.")
        return

    if not STORAGE_ACCOUNT_URL or not BLOB_CONTAINER_NAME or not BLOB_MODEL_NAME:
        logging.warning("Blob config missing (STORAGE_ACCOUNT_URL/BLOB_CONTAINER_NAME/BLOB_MODEL_NAME).")

    # --- Try Blob first ---
    blob_attempted = False
    try:
        if not (STORAGE_ACCOUNT_URL and BLOB_CONTAINER_NAME and BLOB_MODEL_NAME):
            raise ValueError("Missing blob config")

        blob_attempted = True
        temp_path = Path(tempfile.gettempdir()) / Path(BLOB_MODEL_NAME).name
        logging.info(
            "Trying to load model from Blob: container='%s', blob='%s'",
            BLOB_CONTAINER_NAME,
            BLOB_MODEL_NAME,
        )
        blob_client = _get_blob_client(BLOB_CONTAINER_NAME, BLOB_MODEL_NAME)

        # exists() is supported on newer SDKs; keep it defensive
        if hasattr(blob_client, "exists") and not blob_client.exists():
            raise ResourceNotFoundError("Model blob not found")

        with open(temp_path, "wb") as f:
            blob_client.download_blob().readinto(f)

        MODEL = _load_model_from_path(temp_path)
        logging.info("CatBoost model loaded from Blob successfully.")

        try:
            temp_path.unlink()
        except Exception:
            pass
        return

    except ResourceNotFoundError:
        if blob_attempted:
            logging.warning("Model blob not found. Will try local fallback.")
    except Exception as e:
        if blob_attempted:
            logging.warning("Blob model load failed (%s). Will try local fallback.", e)

    # --- Fallback local ---
    try:
        logging.info("Trying to load model from local path: %s", LOCAL_MODEL_PATH)
        MODEL = _load_model_from_path(LOCAL_MODEL_PATH)
        logging.info("CatBoost model loaded from local path successfully.")
        return
    except Exception as e:
        logging.error("Local model load failed: %s", e)

    raise RuntimeError("Could not load model from Blob or local path.")


# =============================================================================
# Inference utilities
# =============================================================================
def adjust_for_weekend(dt: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(dt) or not hasattr(dt, "weekday"):
        return dt
    wd = dt.weekday()
    if wd == 5:
        return dt + pd.Timedelta(days=2)  # Sat -> Mon
    if wd == 6:
        return dt + pd.Timedelta(days=1)  # Sun -> Mon
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


def read_existing_predictions(blob_path: str) -> pd.DataFrame:
    try:
        blob_client = _get_blob_client(OUTPUT_CONTAINER_NAME, blob_path)
        if hasattr(blob_client, "exists") and blob_client.exists():
            blob_data = blob_client.download_blob()
            csv_string = blob_data.readall().decode("utf-8")
            return pd.read_csv(StringIO(csv_string))
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def save_results_to_blob(
    df_to_save: pd.DataFrame,
    blob_path: str,
    deduplicate_column: str = "TicketNumber",
) -> Dict[str, Any]:
    try:
        blob_client = _get_blob_client(OUTPUT_CONTAINER_NAME, blob_path)

        existing_df = pd.DataFrame()
        operation = "created"
        try:
            if hasattr(blob_client, "exists") and blob_client.exists():
                existing_data = blob_client.download_blob().readall()
                existing_df = pd.read_csv(BytesIO(existing_data))
                operation = "appended"
        except Exception:
            existing_df = pd.DataFrame()

        if not existing_df.empty:
            final_df = pd.concat([existing_df, df_to_save], ignore_index=True)
            if deduplicate_column in final_df.columns:
                original_count = len(final_df)
                final_df = final_df.drop_duplicates(subset=[deduplicate_column], keep="last")
                duplicates_removed = original_count - len(final_df)
            else:
                duplicates_removed = 0
        else:
            final_df = df_to_save
            duplicates_removed = 0

        final_df["Last Updated"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        csv_data = final_df.to_csv(index=False)
        blob_client.upload_blob(csv_data, overwrite=True)

        return {
            "success": True,
            "blob_url": blob_client.url,
            "blob_path": blob_path,
            "operation": operation,
            "new_records": len(df_to_save),
            "total_records": len(final_df),
            "duplicates_removed": duplicates_removed,
        }
    except Exception as e:
        logging.exception("Failed to save predictions to blob")
        return {"success": False, "error": str(e)}


def prepare_final_output(df: pd.DataFrame, predictions_minutes: np.ndarray, predicted_dates: pd.Series) -> pd.DataFrame:
    results = pd.DataFrame()

    results["TicketNumber"] = df["TicketNumber"] if "TicketNumber" in df.columns else range(1, len(df) + 1)

    if "msdyn_receiveddate" in df.columns:
        received_dt = pd.to_datetime(df["msdyn_receiveddate"], errors="coerce")
        results["Received Date"] = received_dt.dt.strftime("%m/%d/%Y %H:%M:%S")
    else:
        results["Received Date"] = None

    pred_dt = pd.to_datetime(predicted_dates, errors="coerce")
    results["Predicted Resolution Date"] = pred_dt.dt.strftime("%m/%d/%Y %H:%M:%S")

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

    results["Predicted Duration (Mins)"] = predictions_minutes.round(0).astype(int)

    if actual_res_date_col and "msdyn_receiveddate" in df.columns:
        received_dt = pd.to_datetime(df["msdyn_receiveddate"], errors="coerce")
        resolved_dt = pd.to_datetime(df[actual_res_date_col], errors="coerce")
        actual_duration = (resolved_dt - received_dt).dt.total_seconds() / 60
        results["Actual Duration (Mins)"] = actual_duration.round(0).astype("Int64")
    elif "actual_duration" in df.columns:
        results["Actual Duration (Mins)"] = pd.to_numeric(df["actual_duration"], errors="coerce").round(0).astype("Int64")
    else:
        results["Actual Duration (Mins)"] = None

    if results["Actual Duration (Mins)"].notna().any():
        results["Delta (Mins)"] = (results["Actual Duration (Mins)"] - results["Predicted Duration (Mins)"]).round(0)
    else:
        results["Delta (Mins)"] = None

    if has_actual_date and "Actual Resolution Date" in results.columns:
        # "Delay/Early/On Time" based on whether actual is after/before predicted
        delta_seconds = (actual_dt - pred_dt).dt.total_seconds()
        delta_minutes = delta_seconds / 60
        results["SLA Status"] = delta_minutes.apply(lambda x: "Delay" if x > 0 else ("Early" if x < 0 else "On Time"))
    else:
        results["SLA Status"] = "Pending"

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
        "msdyn_countryprocessedidname",
    ]
    for col in additional_columns:
        results[col] = df[col].values if col in df.columns else None

    return results


# =============================================================================
# Training helpers
# =============================================================================
def load_blob_to_obj(blob_name: str):
    container_client = _get_container_client(BLOB_CONTAINER_NAME)
    blob_client = container_client.get_blob_client(blob_name)
    stream = blob_client.download_blob().readall()

    if blob_name.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(stream), engine="openpyxl")
    if blob_name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(stream))
    if blob_name.endswith((".yaml", ".yml")):
        return yaml.safe_load(stream)
    if blob_name.endswith(".json"):
        return json.loads(stream.decode("utf-8"))
    return stream


def save_bytes_to_blob(blob_name: str, data: bytes, container_name: str | None = None):
    container = container_name or BLOB_CONTAINER_NAME
    container_client = _get_container_client(container)
    container_client.upload_blob(name=blob_name, data=data, overwrite=True)


def save_json_to_blob(blob_name: str, obj: Any, container_name: str | None = None):
    data = json.dumps(obj, indent=2).encode("utf-8")
    save_bytes_to_blob(blob_name, data, container_name=container_name)


def clean_and_engineer(df, quantile_limit=0.75):
    df["received_dt"] = pd.to_datetime(df["msdyn_receiveddate"], errors="coerce")
    df["resolved_dt"] = pd.to_datetime(df["msdyn_resolveddate"], errors="coerce")
    df = df.dropna(subset=["received_dt", "resolved_dt"])

    df["target_minutes"] = (df["resolved_dt"] - df["received_dt"]).dt.total_seconds() / 60
    df = df[(df["target_minutes"] > 30) & (df["target_minutes"] < 43200)]

    df = df.sort_values("received_dt").set_index("received_dt")
    df["backlog_4h"] = df["msdyn_caserocname"].rolling(window="4h").count() - 1
    df["backlog_24h"] = df["msdyn_caserocname"].rolling(window="24h").count() - 1
    df = df.reset_index()

    df["daily_volume"] = df.groupby(df["received_dt"].dt.date)["msdyn_caserocname"].transform("count")
    df["hour_of_day"] = df["received_dt"].dt.hour.astype(str)
    df["day_of_week"] = df["received_dt"].dt.dayofweek.astype(str)

    upper_limit = df["target_minutes"].quantile(quantile_limit)
    df_clean = df[df["target_minutes"] < upper_limit].copy()
    median_map = df_clean.groupby("msdyn_casereasonidname")["target_minutes"].median()
    df_clean["reason_median_speed"] = df_clean["msdyn_casereasonidname"].map(median_map)
    return df_clean, median_map


def clean_categorical_strings(df, cat_cols):
    for col in cat_cols:
        df[col] = df[col].map(
            lambda x: str(int(x)) if pd.notnull(x) and isinstance(x, (float, int)) else str(x)
        ).replace("nan", "Unknown")
    return df


def select_top_features(df, candidate_features, categorical_cols, target_col, top_n=8):
    X_temp = df[candidate_features]
    y_temp = np.log1p(df[target_col])
    selector_model = CatBoostRegressor(iterations=300, cat_features=categorical_cols, verbose=0)
    selector_model.fit(X_temp, y_temp)
    feat_imp = pd.Series(selector_model.get_feature_importance(), index=candidate_features).sort_values(ascending=False)
    return feat_imp.head(top_n).index.tolist()


# =============================================================================
# HTTP Function: Inference
# =============================================================================
@app.function_name(name="catboost_infer")
@app.route(route="catboost_infer", methods=["POST"])
def catboost_infer(req: func.HttpRequest) -> func.HttpResponse:
    try:
        load_model_if_needed()

        try:
            payload = req.get_json()
        except ValueError:
            return func.HttpResponse(
                json.dumps({"success": False, "error": "Invalid JSON"}),
                status_code=400,
                mimetype="application/json",
            )

        save_to_blob = payload.get("save_to_blob", True)
        custom_filename = payload.get("filename", DEFAULT_MASTER_FILE)
        deduplicate_column = payload.get("deduplicate_column", "TicketNumber")

        if "data" in payload:
            records = payload["data"]
        else:
            control_params = {"save_to_blob", "filename", "deduplicate_column"}
            records = {k: v for k, v in payload.items() if k not in control_params}

        if not isinstance(records, list):
            records = [records]

        df = pd.DataFrame(records)

        # Feature engineering consistent with your inference code
        if "msdyn_receiveddate" in df.columns:
            df["received_dt"] = pd.to_datetime(df["msdyn_receiveddate"], errors="coerce")
        else:
            df["received_dt"] = pd.NaT

        df["hour_of_day"] = df["received_dt"].dt.hour.fillna(0).astype(int)
        df["day_of_week"] = df["received_dt"].dt.dayofweek.fillna(0).astype(int)
        df["daily_volume"] = 1

        required_features = MODEL.feature_names_
        cat_indices = MODEL.get_cat_feature_indices()
        df_model = ensure_features(df, required_features, cat_indices)

        pool = Pool(data=df_model, cat_features=cat_indices if cat_indices else None)
        preds = MODEL.predict(pool)

        preds_minutes = np.expm1(preds)
        preds_minutes = np.maximum(preds_minutes, 0).round(0)

        if not df["received_dt"].isna().all():
            raw_finish = df["received_dt"] + pd.to_timedelta(preds_minutes, unit="m")
            pred_resolved = raw_finish.apply(adjust_for_weekend)
            pred_resolved_str = pred_resolved.dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            pred_resolved_str = pd.Series([None] * len(df))

        final_results = prepare_final_output(df, preds_minutes, pred_resolved_str)

        response_data: Dict[str, Any] = {
            "success": True,
            "records_processed": len(df),
            "results": final_results.to_dict(orient="records"),
        }

        if save_to_blob:
            if not custom_filename.endswith(".csv"):
                custom_filename = f"{custom_filename}.csv"
            blob_path = f"{OUTPUT_FOLDER}/{custom_filename}"
            save_info = save_results_to_blob(final_results, blob_path, deduplicate_column)

            if save_info.get("success"):
                response_data.update(
                    {
                        "blob_saved": True,
                        "blob_url": save_info["blob_url"],
                        "blob_path": save_info["blob_path"],
                        "operation": save_info["operation"],
                        "new_records": save_info["new_records"],
                        "total_records": save_info["total_records"],
                        "duplicates_removed": save_info.get("duplicates_removed", 0),
                    }
                )
            else:
                response_data["blob_saved"] = False
                response_data["blob_error"] = save_info.get("error", "Unknown error")

        return func.HttpResponse(json.dumps(response_data, indent=2), status_code=200, mimetype="application/json")

    except Exception as e:
        logging.exception("Unhandled error in catboost_infer.")
        return func.HttpResponse(
            json.dumps({"success": False, "error": str(e)}),
            status_code=500,
            mimetype="application/json",
        )

# =============================================================================
# HTTP Function: Training trigger
# =============================================================================
@app.function_name(name="catboost_train")
@app.route(route="catboost_train", methods=["POST", "GET"])
def catboost_train(req: func.HttpRequest) -> func.HttpResponse:
    """
    Expects config.yaml in blob container (BLOB_CONTAINER_NAME).
    Trains and uploads model to BLOB_MODEL_NAME.
    """
    logging.info("Manual ML Pipeline execution started.")

    try:
        if not STORAGE_ACCOUNT_URL or not BLOB_CONTAINER_NAME:
            raise ValueError("Missing STORAGE_ACCOUNT_URL or BLOB_CONTAINER_NAME app setting.")

        # ---------------------------------------------------------------------
        # CONFIG.YAML replacement (hardcoded config dict in code)
        # ---------------------------------------------------------------------
        config = {
            "paths": {
                "raw_data": "data/raw/Case_06Feb2026.xlsx",
                "processed_data": "data/processed/cleaned_data.csv",
                "plot_folder": "visualization/figures/",
                "model_path": "models/catboost_model_feb2026.cbm",
                "demo_results": "data/processed/predictions_feb2026.csv",
            },
            "params": {
                "target_col": "target_minutes",
                "test_size": 0.3,
                "random_state": 40,
                "outlier_quantile": 0.75,
                "candidate_features": [
                    "msdyn_caserocname",
                    "msdyn_casereasonidname",
                    "msdyn_casesubreasonidname",
                    "msdyn_casesubtypeidname",
                    "msdyn_casetypeidname",
                    "msdyn_countrysubmitteridname",
                    "hour_of_day",
                    "day_of_week",
                    "daily_volume",
                    "reason_median_speed",
                    "msdyn_businessfunctionidname",
                    "msdyn_lineofbusinessidname",
                    "msdyn_programidname",
                    "backlog_4h",
                    "backlog_24h",
                ],
                "categorical_cols": [
                    "msdyn_caserocname",
                    "msdyn_casereasonidname",
                    "msdyn_casesubreasonidname",
                    "msdyn_casesubtypeidname",
                    "msdyn_casetypeidname",
                    "msdyn_countrysubmitteridname",
                    "hour_of_day",
                    "day_of_week",
                    "msdyn_businessfunctionidname",
                    "msdyn_lineofbusinessidname",
                    "msdyn_programidname",
                ],
            },
        }
        # ---------------------------------------------------------------------

        plot_folder = config["paths"]["plot_folder"].strip("/")

        # Load raw data
        df_raw_loaded = load_blob_to_obj(config["paths"]["raw_data"])
        df_raw = df_raw_loaded.rename(columns={"createdon": "msdyn_receiveddate"})
        df_raw["msdyn_receiveddate"] = pd.to_datetime(df_raw["msdyn_receiveddate"], errors="coerce")

        # Filter for relevant data
        df_raw = df_raw[df_raw["msdyn_receiveddate"].dt.year >= 2025].copy()
        df_raw = df_raw[
            (df_raw["msdyn_resolveddate"].notna())
            & (df_raw["msdyn_caserocname"].notna())
            & ((df_raw.get("onholdtime", 0) == 0) | (df_raw.get("onholdtime").isna()))
        ].copy()

        df_clean, median_map = clean_and_engineer(df_raw, config["params"]["outlier_quantile"])
        df_clean = clean_categorical_strings(df_clean, config["params"]["categorical_cols"])

        # EDA plot -> upload
        plt.figure(figsize=(12, 6))
        sns.histplot(df_clean[config["params"]["target_col"]], bins=40, kde=True, color="purple")
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png", bbox_inches="tight")
        plt.close()
        img_buf.seek(0)
        save_bytes_to_blob(f"{plot_folder}/target_distribution_y.png", img_buf.getvalue())

        # Feature selection and training
        top_features = select_top_features(
            df_clean,
            config["params"]["candidate_features"],
            config["params"]["categorical_cols"],
            config["params"]["target_col"],
        )

        final_cat_features = [f for f in top_features if f in config["params"]["categorical_cols"]]

        X = df_clean[top_features]
        y = np.log1p(df_clean[config["params"]["target_col"]])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

        final_model = CatBoostRegressor(
            iterations=3000,
            learning_rate=0.015,
            depth=10,
            loss_function="MAE",
            cat_features=final_cat_features,
            verbose=100,
        )
        final_model.fit(X_train, y_train, eval_set=(X_test, y_test))

        # Save model to temp, upload to blob
        temp_model_path = Path(tempfile.gettempdir()) / "catboost_model.cbm"
        final_model.save_model(str(temp_model_path))

        container_client = _get_container_client(BLOB_CONTAINER_NAME)
        with open(temp_model_path, "rb") as f:
            container_client.upload_blob(name=BLOB_MODEL_NAME, data=f, overwrite=True)

        # Save metadata
        global_median = df_clean[config["params"]["target_col"]].median()
        metadata = {
            "global_median": float(global_median),
            "median_map": median_map.to_dict(),
            "features": top_features,
            "trained_at": datetime.utcnow().isoformat(),
            "model_blob": BLOB_MODEL_NAME,
        }
        metadata_blob_path = f"{os.path.dirname(BLOB_MODEL_NAME)}/median_map.json"
        save_json_to_blob(metadata_blob_path, metadata)

        # IMPORTANT: clear cached model so next infer loads the newly trained one
        global MODEL
        MODEL = None

        return func.HttpResponse(
            json.dumps(
                {
                    "success": True,
                    "message": f"Model saved to {BLOB_MODEL_NAME}",
                    "features_used": top_features,
                    "metadata_saved_to": metadata_blob_path,
                },
                indent=2,
            ),
            status_code=200,
            mimetype="application/json",
        )

    except Exception as e:
        logging.exception("Error during training execution")
        return func.HttpResponse(
            json.dumps({"success": False, "error": str(e)}, indent=2),
            status_code=500,
            mimetype="application/json",
        )

