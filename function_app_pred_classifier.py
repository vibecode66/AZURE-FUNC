import os
import io
import json
import yaml
import logging
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from io import StringIO, BytesIO

import numpy as np
import pandas as pd

import azure.functions as func
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
)

import optuna
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# =============================================================================
# Azure Functions App (Python v2 programming model)
# =============================================================================
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# =============================================================================
# Shared Configuration (set these in Azure Function App -> Configuration)
# =============================================================================
STORAGE_ACCOUNT_URL = os.getenv("STORAGE_ACCOUNT_URL")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")

# ── Regression model blob path/name (used by regression train + infer) ──
BLOB_MODEL_NAME = os.getenv("BLOB_MODEL_NAME", "model/catboost_model_deploy.cbm")

# ── Classifier model blob paths ──
BLOB_CLASSIFIER_MODEL_NAME = os.getenv(
    "BLOB_CLASSIFIER_MODEL_NAME", "model/catboost_hold_classifier.cbm"
)
BLOB_CLASSIFIER_FEATURE_META = os.getenv(
    "BLOB_CLASSIFIER_FEATURE_META", "model/feature_metadata.json"
)
BLOB_CLASSIFIER_METRICS = os.getenv(
    "BLOB_CLASSIFIER_METRICS", "model/classifier_metrics.json"
)

# Inference output
OUTPUT_CONTAINER_NAME = os.getenv("OUTPUT_CONTAINER_NAME", BLOB_CONTAINER_NAME)
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "predictions")
DEFAULT_MASTER_FILE = os.getenv("DEFAULT_MASTER_FILE", "sla_output_results.csv")
DEFAULT_CLASSIFIER_FILE = os.getenv(
    "DEFAULT_CLASSIFIER_FILE", "hold_predictions.csv"
)

# Local fallback (mainly for local dev)
LOCAL_MODEL_PATH = Path(
    os.getenv("LOCAL_MODEL_PATH", r"C:\temp\Model\catboost_model.cbm")
)
LOCAL_CLASSIFIER_MODEL_PATH = Path(
    os.getenv("LOCAL_CLASSIFIER_MODEL_PATH", r"C:\temp\Model\catboost_hold_classifier.cbm")
)
LOCAL_CLASSIFIER_META_PATH = Path(
    os.getenv("LOCAL_CLASSIFIER_META_PATH", r"C:\temp\Model\feature_metadata.json")
)
LOCAL_CLASSIFIER_METRICS_PATH = Path(
    os.getenv("LOCAL_CLASSIFIER_METRICS_PATH", r"C:\temp\Model\classifier_metrics.json")
)

# Global model caches
MODEL: CatBoostRegressor | None = None  # Regression model
CLASSIFIER_MODEL: CatBoostClassifier | None = None  # Classifier model
CLASSIFIER_META: dict | None = None  # feature_metadata.json contents
CLASSIFIER_METRICS: dict | None = None  # metrics.json contents
CLASSIFIER_THRESHOLD: float | None = None  # decision threshold

# =============================================================================
# Classifier constants (match training exactly)
# =============================================================================
LEAKAGE_COLUMNS = [
    "HoldTimeInMinutes",
    "ActualCaseTimeinMinutes",
    "CaseTotalDurationinMinutes",
    "msdyn_resolveddate",
]

RAW_CATEGORICAL_COLUMNS = [
    "msdyn_caserocname",
    "msdyn_businessfunctionidname",
    "msdyn_lineofbusinessidname",
    "msdyn_programidname",
    "msdyn_casetypeidname",
    "msdyn_casesubtypeidname",
    "msdyn_casereasonidname",
    "msdyn_casesubreasonidname",
    "prioritycodename",
    "msdyn_complexityname",
    "msdyn_countrysubmitteridname",
    "msdyn_countryprocessedidname",
    "statuscodename",
]

DERIVED_CATEGORICAL_COLUMNS = [
    "CaseHierarchy",
    "ReasonGroup",
    "LOBProgram",
]

ALL_CATEGORICAL_COLUMNS = RAW_CATEGORICAL_COLUMNS + DERIVED_CATEGORICAL_COLUMNS

DERIVED_NUMERIC_COLUMNS = [
    "ReasonHoldRate",
    "SubTypeHoldRate",
    "ProgramHoldRate",
    "IsSameCountry",
    "created_hour",
    "created_dayofweek",
    "created_is_weekend",
    "created_month",
]

CLASSIFIER_FEATURE_COLUMNS = (
    RAW_CATEGORICAL_COLUMNS + DERIVED_CATEGORICAL_COLUMNS + DERIVED_NUMERIC_COLUMNS
)

CLASSIFIER_TARGET_COLUMN = "IsHold"

# Smoothing parameters for target encoding (Bayesian smoothing)
SMOOTHING_MIN_COUNT = 30
SMOOTHING_STRENGTH = 10

# Optuna settings (can be overridden via request payload)
DEFAULT_OPTUNA_N_TRIALS = 50
DEFAULT_OPTUNA_TIMEOUT = None
OPTUNA_SEED = 42


# =============================================================================
# Identity / Storage helpers
# =============================================================================
def _get_credential() -> DefaultAzureCredential:
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
# Generic blob I/O helpers
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


def load_blob_bytes(blob_name: str, container_name: str | None = None) -> bytes:
    container = container_name or BLOB_CONTAINER_NAME
    container_client = _get_container_client(container)
    blob_client = container_client.get_blob_client(blob_name)
    return blob_client.download_blob().readall()


def save_bytes_to_blob(
    blob_name: str, data: bytes, container_name: str | None = None
):
    container = container_name or BLOB_CONTAINER_NAME
    container_client = _get_container_client(container)
    container_client.upload_blob(name=blob_name, data=data, overwrite=True)


def save_json_to_blob(
    blob_name: str, obj: Any, container_name: str | None = None
):
    data = json.dumps(obj, indent=2).encode("utf-8")
    save_bytes_to_blob(blob_name, data, container_name=container_name)


# =============================================================================
# Regression model load helpers
# =============================================================================
def _load_regressor_from_path(p: Path) -> CatBoostRegressor:
    m = CatBoostRegressor()
    m.load_model(str(p))
    return m


def load_regression_model_if_needed() -> None:
    global MODEL

    if MODEL is not None:
        logging.info("Regression MODEL already loaded; skipping reload.")
        return

    if not STORAGE_ACCOUNT_URL or not BLOB_CONTAINER_NAME or not BLOB_MODEL_NAME:
        logging.warning(
            "Blob config missing for regression model "
            "(STORAGE_ACCOUNT_URL/BLOB_CONTAINER_NAME/BLOB_MODEL_NAME)."
        )

    # --- Try Blob first ---
    blob_attempted = False
    try:
        if not (STORAGE_ACCOUNT_URL and BLOB_CONTAINER_NAME and BLOB_MODEL_NAME):
            raise ValueError("Missing blob config")

        blob_attempted = True
        temp_path = Path(tempfile.gettempdir()) / Path(BLOB_MODEL_NAME).name
        logging.info(
            "Trying to load regression model from Blob: container='%s', blob='%s'",
            BLOB_CONTAINER_NAME,
            BLOB_MODEL_NAME,
        )
        blob_client = _get_blob_client(BLOB_CONTAINER_NAME, BLOB_MODEL_NAME)

        if hasattr(blob_client, "exists") and not blob_client.exists():
            raise ResourceNotFoundError("Regression model blob not found")

        with open(temp_path, "wb") as f:
            blob_client.download_blob().readinto(f)

        MODEL = _load_regressor_from_path(temp_path)
        logging.info("CatBoost regression model loaded from Blob successfully.")

        try:
            temp_path.unlink()
        except Exception:
            pass
        return

    except ResourceNotFoundError:
        if blob_attempted:
            logging.warning("Regression model blob not found. Will try local fallback.")
    except Exception as e:
        if blob_attempted:
            logging.warning(
                "Blob regression model load failed (%s). Will try local fallback.", e
            )

    # --- Fallback local ---
    try:
        logging.info(
            "Trying to load regression model from local path: %s", LOCAL_MODEL_PATH
        )
        MODEL = _load_regressor_from_path(LOCAL_MODEL_PATH)
        logging.info("CatBoost regression model loaded from local path successfully.")
        return
    except Exception as e:
        logging.error("Local regression model load failed: %s", e)

    raise RuntimeError("Could not load regression model from Blob or local path.")


# =============================================================================
# Classifier model load helpers
# =============================================================================
def _load_classifier_from_path(p: Path) -> CatBoostClassifier:
    m = CatBoostClassifier()
    m.load_model(str(p))
    return m


def _load_json_file(p: Path) -> dict:
    with open(p, "r") as f:
        return json.load(f)


def load_classifier_model_if_needed() -> None:
    global CLASSIFIER_MODEL, CLASSIFIER_META, CLASSIFIER_METRICS, CLASSIFIER_THRESHOLD

    if CLASSIFIER_MODEL is not None and CLASSIFIER_META is not None:
        logging.info("Classifier MODEL already loaded; skipping reload.")
        return

    blob_ok = bool(STORAGE_ACCOUNT_URL and BLOB_CONTAINER_NAME)

    # ── Load model (.cbm) ────────────────────────────────────────────
    model_loaded = False
    if blob_ok:
        try:
            temp_path = Path(tempfile.gettempdir()) / Path(BLOB_CLASSIFIER_MODEL_NAME).name
            logging.info(
                "Loading classifier model from Blob: container='%s', blob='%s'",
                BLOB_CONTAINER_NAME,
                BLOB_CLASSIFIER_MODEL_NAME,
            )
            blob_client = _get_blob_client(BLOB_CONTAINER_NAME, BLOB_CLASSIFIER_MODEL_NAME)
            if hasattr(blob_client, "exists") and not blob_client.exists():
                raise ResourceNotFoundError("Classifier model blob not found")
            with open(temp_path, "wb") as f:
                blob_client.download_blob().readinto(f)
            CLASSIFIER_MODEL = _load_classifier_from_path(temp_path)
            model_loaded = True
            logging.info("Classifier model loaded from Blob.")
            try:
                temp_path.unlink()
            except Exception:
                pass
        except Exception as e:
            logging.warning("Blob classifier model load failed (%s). Trying local.", e)

    if not model_loaded:
        try:
            CLASSIFIER_MODEL = _load_classifier_from_path(LOCAL_CLASSIFIER_MODEL_PATH)
            logging.info("Classifier model loaded from local path.")
        except Exception as e:
            raise RuntimeError(
                f"Could not load classifier model from Blob or local: {e}"
            )

    # ── Load feature metadata (.json) ────────────────────────────────
    meta_loaded = False
    if blob_ok:
        try:
            raw = load_blob_bytes(BLOB_CLASSIFIER_FEATURE_META)
            CLASSIFIER_META = json.loads(raw.decode("utf-8"))
            meta_loaded = True
            logging.info("Classifier feature metadata loaded from Blob.")
        except Exception as e:
            logging.warning("Blob classifier meta load failed (%s). Trying local.", e)

    if not meta_loaded:
        try:
            CLASSIFIER_META = _load_json_file(LOCAL_CLASSIFIER_META_PATH)
            logging.info("Classifier feature metadata loaded from local path.")
        except Exception as e:
            raise RuntimeError(
                f"Could not load classifier feature metadata from Blob or local: {e}"
            )

    # ── Load metrics (.json) — optional ──────────────────────────────
    if blob_ok:
        try:
            raw = load_blob_bytes(BLOB_CLASSIFIER_METRICS)
            CLASSIFIER_METRICS = json.loads(raw.decode("utf-8"))
            logging.info("Classifier metrics loaded from Blob.")
        except Exception:
            CLASSIFIER_METRICS = {}
    else:
        try:
            CLASSIFIER_METRICS = _load_json_file(LOCAL_CLASSIFIER_METRICS_PATH)
        except Exception:
            CLASSIFIER_METRICS = {}

    # ── Determine threshold ──────────────────────────────────────────
    if CLASSIFIER_METRICS:
        CLASSIFIER_THRESHOLD = float(
            CLASSIFIER_METRICS.get("recommended_threshold", 0.5)
        )
    else:
        CLASSIFIER_THRESHOLD = 0.5
    logging.info("Classifier decision threshold: %.5f", CLASSIFIER_THRESHOLD)


# =============================================================================
# Regression inference utilities
# =============================================================================
def adjust_for_weekend(dt: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(dt) or not hasattr(dt, "weekday"):
        return dt
    wd = dt.weekday()
    if wd == 5:
        return dt + pd.Timedelta(days=2)
    if wd == 6:
        return dt + pd.Timedelta(days=1)
    return dt


def ensure_features(
    df: pd.DataFrame, required_features: List[str], cat_indices: List[int]
) -> pd.DataFrame:
    cat_set = {
        required_features[i] for i in cat_indices if 0 <= i < len(required_features)
    }
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
                final_df = final_df.drop_duplicates(
                    subset=[deduplicate_column], keep="last"
                )
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


def prepare_final_output(
    df: pd.DataFrame,
    predictions_minutes: np.ndarray,
    predicted_dates: pd.Series,
) -> pd.DataFrame:
    results = pd.DataFrame()

    results["TicketNumber"] = (
        df["TicketNumber"] if "TicketNumber" in df.columns else range(1, len(df) + 1)
    )

    if "msdyn_receiveddate" in df.columns:
        received_dt = pd.to_datetime(df["msdyn_receiveddate"], errors="coerce")
        results["Received Date"] = received_dt.dt.strftime("%m/%d/%Y %H:%M:%S")
    else:
        results["Received Date"] = None

    pred_dt = pd.to_datetime(predicted_dates, errors="coerce")
    results["Predicted Resolution Date"] = pred_dt.dt.strftime("%m/%d/%Y %H:%M:%S")

    actual_res_date_col = None
    for col_name in [
        "msdyn_resolveddate",
        "resolved_date",
        "actual_resolved_date",
        "modifiedon",
    ]:
        if col_name in df.columns:
            actual_res_date_col = col_name
            break

    has_actual_date = False
    if actual_res_date_col:
        actual_dt = pd.to_datetime(df[actual_res_date_col], errors="coerce")
        if actual_dt.notna().any():
            has_actual_date = True
            results["Actual Resolution Date"] = actual_dt.dt.strftime(
                "%m/%d/%Y %H:%M:%S"
            )
        else:
            results["Actual Resolution Date"] = None
    else:
        results["Actual Resolution Date"] = None

    results["Predicted Duration (Mins)"] = predictions_minutes.round(0).astype(int)

    if actual_res_date_col and "msdyn_receiveddate" in df.columns:
        received_dt = pd.to_datetime(df["msdyn_receiveddate"], errors="coerce")
        resolved_dt = pd.to_datetime(df[actual_res_date_col], errors="coerce")
        actual_duration = (resolved_dt - received_dt).dt.total_seconds() / 60
        results["Actual Duration (Mins)"] = (
            actual_duration.round(0).astype("Int64")
        )
    elif "actual_duration" in df.columns:
        results["Actual Duration (Mins)"] = (
            pd.to_numeric(df["actual_duration"], errors="coerce")
            .round(0)
            .astype("Int64")
        )
    else:
        results["Actual Duration (Mins)"] = None

    if results["Actual Duration (Mins)"].notna().any():
        results["Delta (Mins)"] = (
            results["Actual Duration (Mins)"] - results["Predicted Duration (Mins)"]
        ).round(0)
    else:
        results["Delta (Mins)"] = None

    if has_actual_date and "Actual Resolution Date" in results.columns:
        delta_seconds = (actual_dt - pred_dt).dt.total_seconds()
        delta_minutes = delta_seconds / 60
        results["SLA Status"] = delta_minutes.apply(
            lambda x: "Delay" if x > 0 else ("Early" if x < 0 else "On Time")
        )
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
# Regression training helpers
# =============================================================================
def clean_and_engineer(df, quantile_limit=0.75):
    df["received_dt"] = pd.to_datetime(df["msdyn_receiveddate"], errors="coerce")
    df["resolved_dt"] = pd.to_datetime(df["msdyn_resolveddate"], errors="coerce")
    df = df.dropna(subset=["received_dt", "resolved_dt"])

    df["target_minutes"] = (
        (df["resolved_dt"] - df["received_dt"]).dt.total_seconds() / 60
    )
    df = df[(df["target_minutes"] > 30) & (df["target_minutes"] < 43200)]

    df = df.sort_values("received_dt").set_index("received_dt")
    df["backlog_4h"] = df["msdyn_caserocname"].rolling(window="4h").count() - 1
    df["backlog_24h"] = df["msdyn_caserocname"].rolling(window="24h").count() - 1
    df = df.reset_index()

    df["daily_volume"] = df.groupby(df["received_dt"].dt.date)[
        "msdyn_caserocname"
    ].transform("count")
    df["hour_of_day"] = df["received_dt"].dt.hour.astype(str)
    df["day_of_week"] = df["received_dt"].dt.dayofweek.astype(str)

    upper_limit = df["target_minutes"].quantile(quantile_limit)
    df_clean = df[df["target_minutes"] < upper_limit].copy()
    median_map = df_clean.groupby("msdyn_casereasonidname")["target_minutes"].median()
    df_clean["reason_median_speed"] = df_clean["msdyn_casereasonidname"].map(
        median_map
    )
    return df_clean, median_map


def clean_categorical_strings(df, cat_cols):
    for col in cat_cols:
        df[col] = (
            df[col]
            .map(
                lambda x: str(int(x))
                if pd.notnull(x) and isinstance(x, (float, int))
                else str(x)
            )
            .replace("nan", "Unknown")
        )
    return df


def select_top_features(df, candidate_features, categorical_cols, target_col, top_n=8):
    X_temp = df[candidate_features]
    y_temp = np.log1p(df[target_col])
    selector_model = CatBoostRegressor(
        iterations=300,
        cat_features=categorical_cols,
        verbose=0,
        train_dir=os.path.join(tempfile.gettempdir(), "catboost_info_select"),
    )
    selector_model.fit(X_temp, y_temp)
    feat_imp = pd.Series(
        selector_model.get_feature_importance(), index=candidate_features
    ).sort_values(ascending=False)
    return feat_imp.head(top_n).index.tolist()


# =============================================================================
# Classifier training helpers
# =============================================================================
def classifier_make_target(df: pd.DataFrame) -> pd.DataFrame:
    df["HoldTimeInMinutes"] = pd.to_numeric(
        df["HoldTimeInMinutes"], errors="coerce"
    ).fillna(0)
    df[CLASSIFIER_TARGET_COLUMN] = (df["HoldTimeInMinutes"] > 0).astype(int)
    return df


def classifier_build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["createdon"] = pd.to_datetime(df["createdon"], errors="coerce")

    for col in RAW_CATEGORICAL_COLUMNS:
        df[col] = df[col].fillna("__MISSING__").astype(str)

    df["created_hour"] = df["createdon"].dt.hour
    df["created_dayofweek"] = df["createdon"].dt.dayofweek
    df["created_is_weekend"] = (df["created_dayofweek"] >= 5).astype(int)
    df["created_month"] = df["createdon"].dt.month

    df["IsSameCountry"] = (
        df["msdyn_countrysubmitteridname"] == df["msdyn_countryprocessedidname"]
    ).astype(int)

    df["CaseHierarchy"] = (
        df["msdyn_casetypeidname"].astype(str)
        + "_"
        + df["msdyn_casesubtypeidname"].astype(str)
    )
    df["ReasonGroup"] = (
        df["msdyn_casereasonidname"].astype(str)
        + "_"
        + df["msdyn_casesubreasonidname"].astype(str)
    )
    df["LOBProgram"] = (
        df["msdyn_lineofbusinessidname"].astype(str)
        + "_"
        + df["msdyn_programidname"].astype(str)
    )

    for col in DERIVED_CATEGORICAL_COLUMNS:
        df[col] = df[col].astype(str)

    return df


def _compute_smoothed_rate(
    group_series: pd.Series,
    target_series: pd.Series,
    global_rate: float,
) -> dict:
    stats = pd.DataFrame({"group": group_series, "target": target_series})
    agg = stats.groupby("group")["target"].agg(["mean", "count"])
    agg["smoothed"] = (
        agg["count"] * agg["mean"] + SMOOTHING_STRENGTH * global_rate
    ) / (agg["count"] + SMOOTHING_STRENGTH)
    agg.loc[agg["count"] < SMOOTHING_MIN_COUNT, "smoothed"] = global_rate
    return agg["smoothed"].to_dict()


def classifier_fit_target_encoders(
    train_df: pd.DataFrame, y_train: pd.Series
) -> tuple:
    global_rate = float(y_train.mean())

    mappings = {}
    mappings["ReasonHoldRate"] = _compute_smoothed_rate(
        train_df["ReasonGroup"], y_train, global_rate
    )
    mappings["SubTypeHoldRate"] = _compute_smoothed_rate(
        train_df["msdyn_casesubtypeidname"], y_train, global_rate
    )
    mappings["ProgramHoldRate"] = _compute_smoothed_rate(
        train_df["msdyn_programidname"], y_train, global_rate
    )

    return mappings, global_rate


def classifier_apply_target_encoders(
    df: pd.DataFrame, mappings: dict, global_rate: float
) -> pd.DataFrame:
    df["ReasonHoldRate"] = (
        df["ReasonGroup"]
        .map(mappings["ReasonHoldRate"])
        .fillna(global_rate)
        .astype(float)
    )
    df["SubTypeHoldRate"] = (
        df["msdyn_casesubtypeidname"]
        .map(mappings["SubTypeHoldRate"])
        .fillna(global_rate)
        .astype(float)
    )
    df["ProgramHoldRate"] = (
        df["msdyn_programidname"]
        .map(mappings["ProgramHoldRate"])
        .fillna(global_rate)
        .astype(float)
    )
    return df


def classifier_evaluate(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    cat_cols: list,
    threshold: float = 0.5,
    set_name: str = "val",
) -> dict:
    for col in cat_cols:
        X[col] = X[col].astype(str)

    pool = Pool(X, cat_features=cat_cols)
    y_prob = model.predict_proba(pool)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = roc_auc_score(y, y_prob)
    pr_auc = average_precision_score(y, y_prob)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred).tolist()

    precisions, recalls, thresholds = precision_recall_curve(y, y_prob)
    f1_scores = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls),
        0,
    )
    best_idx = np.argmax(f1_scores)
    best_threshold = (
        float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    )
    best_f1 = float(f1_scores[best_idx])

    y_pred_best = (y_prob >= best_threshold).astype(int)
    precision_best = precision_score(y, y_pred_best, zero_division=0)
    recall_best = recall_score(y, y_pred_best, zero_division=0)
    f1_best = f1_score(y, y_pred_best, zero_division=0)
    cm_best = confusion_matrix(y, y_pred_best).tolist()

    metrics = {
        f"{set_name}_roc_auc": round(roc_auc, 5),
        f"{set_name}_pr_auc": round(pr_auc, 5),
        f"{set_name}_precision_at_0.5": round(precision, 5),
        f"{set_name}_recall_at_0.5": round(recall, 5),
        f"{set_name}_f1_at_0.5": round(f1, 5),
        f"{set_name}_confusion_matrix_at_0.5": cm,
        f"{set_name}_best_f1_threshold": round(best_threshold, 5),
        f"{set_name}_best_f1": round(best_f1, 5),
        f"{set_name}_precision_at_best_threshold": round(precision_best, 5),
        f"{set_name}_recall_at_best_threshold": round(recall_best, 5),
        f"{set_name}_f1_at_best_threshold": round(f1_best, 5),
        f"{set_name}_confusion_matrix_at_best_threshold": cm_best,
    }

    logging.info(
        "[%s] ROC-AUC: %.4f | PR-AUC: %.4f | F1@0.5: %.4f | Best-F1: %.4f @%.4f",
        set_name.upper(),
        roc_auc,
        pr_auc,
        f1,
        best_f1,
        best_threshold,
    )

    return metrics


def classifier_run_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cat_cols: list,
    n_trials: int = DEFAULT_OPTUNA_N_TRIALS,
    timeout: int | None = DEFAULT_OPTUNA_TIMEOUT,
) -> dict:
    for col in cat_cols:
        X_train[col] = X_train[col].astype(str)
        X_val[col] = X_val[col].astype(str)

    train_pool = Pool(X_train, label=y_train, cat_features=cat_cols)
    val_pool = Pool(X_val, label=y_val, cat_features=cat_cols)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 500, 5000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature", 0.0, 10.0
            ),
            "auto_class_weights": "Balanced",
            "eval_metric": "AUC",
            "random_seed": 42,
            "early_stopping_rounds": 100,
            "verbose": 0,
            "use_best_model": True,
            "train_dir": os.path.join(tempfile.gettempdir(), "catboost_optuna"),
        }
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool)
        y_prob = model.predict_proba(val_pool)[:, 1]
        return roc_auc_score(y_val, y_prob)

    sampler = optuna.samplers.TPESampler(seed=OPTUNA_SEED)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="catboost_hold_hpo",
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    logging.info(
        "Optuna best trial #%d — Val ROC-AUC: %.5f",
        study.best_trial.number,
        study.best_value,
    )
    return study.best_params


# =============================================================================
# Classifier inference helpers (used by the HTTP endpoint)
# =============================================================================
def classifier_prepare_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing for classifier inference using loaded metadata."""
    # Leakage guard
    leak = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    if leak:
        logging.info("Dropping leakage columns: %s", leak)
        df = df.drop(columns=leak)

    df = classifier_build_base_features(df)

    # Apply target encodings from saved metadata
    meta = CLASSIFIER_META
    global_rate = meta["global_hold_rate_train"]
    te_mappings = meta["target_encoding_mappings"]

    df["ReasonHoldRate"] = (
        df["ReasonGroup"]
        .map(te_mappings["ReasonHoldRate"])
        .fillna(global_rate)
        .astype(float)
    )
    df["SubTypeHoldRate"] = (
        df["msdyn_casesubtypeidname"]
        .map(te_mappings["SubTypeHoldRate"])
        .fillna(global_rate)
        .astype(float)
    )
    df["ProgramHoldRate"] = (
        df["msdyn_programidname"]
        .map(te_mappings["ProgramHoldRate"])
        .fillna(global_rate)
        .astype(float)
    )

    feature_columns = meta["feature_columns"]
    categorical_columns = meta["categorical_columns"]

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Required feature columns missing from input: {missing}")

    X = df[feature_columns].copy()
    for col in categorical_columns:
        if col in X.columns:
            X[col] = X[col].astype(str)
    return X


# =============================================================================
# HTTP Function 1: Regression Inference
# =============================================================================
@app.function_name(name="catboost_infer")
@app.route(route="catboost_infer", methods=["POST"])
def catboost_infer(req: func.HttpRequest) -> func.HttpResponse:
    try:
        load_regression_model_if_needed()

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

        if "msdyn_receiveddate" in df.columns:
            df["received_dt"] = pd.to_datetime(
                df["msdyn_receiveddate"], errors="coerce"
            )
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
            save_info = save_results_to_blob(
                final_results, blob_path, deduplicate_column
            )

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

        return func.HttpResponse(
            json.dumps(response_data, indent=2),
            status_code=200,
            mimetype="application/json",
        )

    except Exception as e:
        logging.exception("Unhandled error in catboost_infer.")
        return func.HttpResponse(
            json.dumps({"success": False, "error": str(e)}),
            status_code=500,
            mimetype="application/json",
        )


# =============================================================================
# HTTP Function 2: Regression Training
# =============================================================================
@app.function_name(name="catboost_train")
@app.route(route="catboost_train", methods=["POST", "GET"])
def catboost_train(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Regression ML Pipeline execution started.")

    try:
        if not STORAGE_ACCOUNT_URL or not BLOB_CONTAINER_NAME:
            raise ValueError(
                "Missing STORAGE_ACCOUNT_URL or BLOB_CONTAINER_NAME app setting."
            )

        config = {
            "paths": {
                "raw_data": "data/training_raw_dataset_catboost.xlsx",
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

        plot_folder = config["paths"]["plot_folder"].strip("/")

        df_raw_loaded = load_blob_to_obj(config["paths"]["raw_data"])
        df_raw = df_raw_loaded.rename(columns={"createdon": "msdyn_receiveddate"})
        df_raw["msdyn_receiveddate"] = pd.to_datetime(
            df_raw["msdyn_receiveddate"], errors="coerce"
        )

        df_raw = df_raw[df_raw["msdyn_receiveddate"].dt.year >= 2025].copy()
        df_raw = df_raw[
            (df_raw["msdyn_resolveddate"].notna())
            & (df_raw["msdyn_caserocname"].notna())
            & (
                (df_raw.get("onholdtime", 0) == 0)
                | (df_raw.get("onholdtime").isna())
            )
        ].copy()

        df_clean, median_map = clean_and_engineer(
            df_raw, config["params"]["outlier_quantile"]
        )
        df_clean = clean_categorical_strings(
            df_clean, config["params"]["categorical_cols"]
        )

        plt.figure(figsize=(12, 6))
        sns.histplot(
            df_clean[config["params"]["target_col"]], bins=40, kde=True, color="purple"
        )
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png", bbox_inches="tight")
        plt.close()
        img_buf.seek(0)
        save_bytes_to_blob(
            f"{plot_folder}/target_distribution_y.png", img_buf.getvalue()
        )

        top_features = select_top_features(
            df_clean,
            config["params"]["candidate_features"],
            config["params"]["categorical_cols"],
            config["params"]["target_col"],
        )

        final_cat_features = [
            f for f in top_features if f in config["params"]["categorical_cols"]
        ]

        X = df_clean[top_features]
        y = np.log1p(df_clean[config["params"]["target_col"]])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=40
        )

        final_model = CatBoostRegressor(
            iterations=3000,
            learning_rate=0.015,
            depth=10,
            loss_function="MAE",
            cat_features=final_cat_features,
            verbose=100,
            train_dir=os.path.join(tempfile.gettempdir(), "catboost_info"),
        )

        final_model.fit(X_train, y_train, eval_set=(X_test, y_test))

        temp_model_path = Path(tempfile.gettempdir()) / "catboost_model.cbm"
        final_model.save_model(str(temp_model_path))

        container_client = _get_container_client(BLOB_CONTAINER_NAME)
        with open(temp_model_path, "rb") as f:
            container_client.upload_blob(name=BLOB_MODEL_NAME, data=f, overwrite=True)

        global_median = df_clean[config["params"]["target_col"]].median()
        metadata = {
            "global_median": float(global_median),
            "median_map": median_map.to_dict(),
            "features": top_features,
            "trained_at": datetime.utcnow().isoformat(),
            "model_blob": BLOB_MODEL_NAME,
        }
        metadata_blob_path = (
            f"{os.path.dirname(BLOB_MODEL_NAME)}/median_map.json"
        )
        save_json_to_blob(metadata_blob_path, metadata)

        # Clear cached model so next infer loads the newly trained one
        global MODEL
        MODEL = None

        return func.HttpResponse(
            json.dumps(
                {
                    "success": True,
                    "message": f"Regression model saved to {BLOB_MODEL_NAME}",
                    "features_used": top_features,
                    "metadata_saved_to": metadata_blob_path,
                },
                indent=2,
            ),
            status_code=200,
            mimetype="application/json",
        )

    except Exception as e:
        logging.exception("Error during regression training execution")
        return func.HttpResponse(
            json.dumps({"success": False, "error": str(e)}, indent=2),
            status_code=500,
            mimetype="application/json",
        )


# =============================================================================
# HTTP Function 3: Classifier Training
# =============================================================================
@app.function_name(name="classifier_train")
@app.route(route="classifier_train", methods=["POST", "GET"])
def classifier_train(req: func.HttpRequest) -> func.HttpResponse:
    """
    Trains the Hold / No-Hold CatBoost classifier with Optuna HPO.

    Optional JSON body parameters:
      - raw_data_blob: str  (default: "data/catboost_train_dataset_raw.xlsx")
      - optuna_n_trials: int (default: 50)
      - optuna_timeout: int|null (default: null)
    """
    logging.info("Classifier training pipeline execution started.")

    try:
        if not STORAGE_ACCOUNT_URL or not BLOB_CONTAINER_NAME:
            raise ValueError(
                "Missing STORAGE_ACCOUNT_URL or BLOB_CONTAINER_NAME app setting."
            )

        # Parse optional parameters from request
        try:
            payload = req.get_json()
        except (ValueError, Exception):
            payload = {}

        raw_data_blob = payload.get(
            "raw_data_blob", "data/catboost_train_dataset_raw.xlsx"
        )
        n_trials = int(payload.get("optuna_n_trials", DEFAULT_OPTUNA_N_TRIALS))
        optuna_timeout = payload.get("optuna_timeout", DEFAULT_OPTUNA_TIMEOUT)
        if optuna_timeout is not None:
            optuna_timeout = int(optuna_timeout)

        # ── Step 1: Load data ────────────────────────────────────────
        logging.info("Loading classifier training data from blob: %s", raw_data_blob)
        df = load_blob_to_obj(raw_data_blob)
        logging.info("Raw shape: %s", df.shape)

        df["createdon"] = pd.to_datetime(df["createdon"], errors="coerce")
        df["msdyn_resolveddate"] = pd.to_datetime(
            df["msdyn_resolveddate"], errors="coerce"
        )
        before = len(df)
        df = df.dropna(subset=["createdon"]).reset_index(drop=True)
        logging.info(
            "Dropped %d rows with missing 'createdon'. Shape: %s",
            before - len(df),
            df.shape,
        )

        # ── Step 2: Create target ────────────────────────────────────
        df = classifier_make_target(df)
        hold_count = int(df[CLASSIFIER_TARGET_COLUMN].sum())
        total = len(df)
        logging.info(
            "IsHold distribution: 1=%d (%.1f%%), 0=%d (%.1f%%)",
            hold_count,
            hold_count / total * 100,
            total - hold_count,
            (total - hold_count) / total * 100,
        )

        # ── Step 3: Build base features ──────────────────────────────
        df = classifier_build_base_features(df)

        # ── Step 4: Split (BEFORE target encoding) ───────────────────
        y = df[CLASSIFIER_TARGET_COLUMN]
        train_df, temp_df = train_test_split(
            df, test_size=0.30, random_state=42, stratify=y
        )
        y_temp = temp_df[CLASSIFIER_TARGET_COLUMN]
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, random_state=42, stratify=y_temp
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        logging.info(
            "Split — Train: %d | Val: %d | Test: %d",
            len(train_df),
            len(val_df),
            len(test_df),
        )

        # ── Step 5: Fit target encoders on TRAINING data only ────────
        y_train = train_df[CLASSIFIER_TARGET_COLUMN]
        mappings, global_rate = classifier_fit_target_encoders(train_df, y_train)

        train_df = classifier_apply_target_encoders(train_df, mappings, global_rate)
        val_df = classifier_apply_target_encoders(val_df, mappings, global_rate)
        test_df = classifier_apply_target_encoders(test_df, mappings, global_rate)

        # Leakage check
        for col in LEAKAGE_COLUMNS:
            assert col not in CLASSIFIER_FEATURE_COLUMNS, (
                f"LEAKAGE DETECTED: {col} in feature list!"
            )

        X_train = train_df[CLASSIFIER_FEATURE_COLUMNS].copy()
        y_train = train_df[CLASSIFIER_TARGET_COLUMN].copy()
        X_val = val_df[CLASSIFIER_FEATURE_COLUMNS].copy()
        y_val = val_df[CLASSIFIER_TARGET_COLUMN].copy()
        X_test = test_df[CLASSIFIER_FEATURE_COLUMNS].copy()
        y_test = test_df[CLASSIFIER_TARGET_COLUMN].copy()

        cat_cols_in_features = [
            c for c in ALL_CATEGORICAL_COLUMNS if c in CLASSIFIER_FEATURE_COLUMNS
        ]

        # ── Step 6: Optuna HPO ───────────────────────────────────────
        logging.info("Starting Optuna HPO with %d trials...", n_trials)
        best_params = classifier_run_optuna(
            X_train.copy(),
            y_train,
            X_val.copy(),
            y_val,
            cat_cols_in_features,
            n_trials=n_trials,
            timeout=optuna_timeout,
        )

        # ── Step 7: Train final model ────────────────────────────────
        logging.info("Training final classifier with best hyperparameters...")
        for col in cat_cols_in_features:
            X_train[col] = X_train[col].astype(str)
            X_val[col] = X_val[col].astype(str)

        train_pool = Pool(X_train, label=y_train, cat_features=cat_cols_in_features)
        val_pool = Pool(X_val, label=y_val, cat_features=cat_cols_in_features)

        final_params = {
            **best_params,
            "auto_class_weights": "Balanced",
            "eval_metric": "AUC",
            "random_seed": 42,
            "early_stopping_rounds": 100,
            "verbose": 200,
            "use_best_model": True,
            "train_dir": os.path.join(tempfile.gettempdir(), "catboost_clf_final"),
        }

        model = CatBoostClassifier(**final_params)
        model.fit(train_pool, eval_set=val_pool)
        best_iteration = model.get_best_iteration()
        logging.info("Best iteration: %d", best_iteration)

        # ── Step 8: Evaluate ─────────────────────────────────────────
        all_metrics = {}
        val_metrics = classifier_evaluate(
            model, X_val.copy(), y_val, cat_cols_in_features, 0.5, "val"
        )
        all_metrics.update(val_metrics)

        test_metrics = classifier_evaluate(
            model, X_test.copy(), y_test, cat_cols_in_features, 0.5, "test"
        )
        all_metrics.update(test_metrics)

        train_metrics = classifier_evaluate(
            model, X_train.copy(), y_train, cat_cols_in_features, 0.5, "train"
        )
        all_metrics.update(train_metrics)

        all_metrics["default_threshold"] = 0.5
        all_metrics["recommended_threshold"] = val_metrics.get(
            "val_best_f1_threshold", 0.5
        )
        all_metrics["optuna_best_params"] = best_params

        # ── Step 9: Save artifacts to Blob ───────────────────────────
        # Save model
        temp_model_path = Path(tempfile.gettempdir()) / "catboost_hold_classifier.cbm"
        model.save_model(str(temp_model_path))

        with open(temp_model_path, "rb") as f:
            save_bytes_to_blob(BLOB_CLASSIFIER_MODEL_NAME, f.read())
        logging.info("Classifier model saved to blob: %s", BLOB_CLASSIFIER_MODEL_NAME)

        # Save feature metadata
        serializable_mappings = {}
        for feat_name, mapping in mappings.items():
            serializable_mappings[feat_name] = {
                str(k): float(v) for k, v in mapping.items()
            }

        feature_meta = {
            "feature_columns": CLASSIFIER_FEATURE_COLUMNS,
            "categorical_columns": cat_cols_in_features,
            "leakage_columns": LEAKAGE_COLUMNS,
            "global_hold_rate_train": round(global_rate, 6),
            "target_encoding_mappings": serializable_mappings,
            "smoothing_min_count": SMOOTHING_MIN_COUNT,
            "smoothing_strength": SMOOTHING_STRENGTH,
            "best_hyperparameters": best_params,
            "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_json_to_blob(BLOB_CLASSIFIER_FEATURE_META, feature_meta)
        logging.info(
            "Feature metadata saved to blob: %s", BLOB_CLASSIFIER_FEATURE_META
        )

        # Save metrics
        save_json_to_blob(BLOB_CLASSIFIER_METRICS, all_metrics)
        logging.info("Metrics saved to blob: %s", BLOB_CLASSIFIER_METRICS)

        # Feature importance (top 20)
        importance = model.get_feature_importance()
        feat_imp = (
            pd.DataFrame(
                {"feature": CLASSIFIER_FEATURE_COLUMNS, "importance": importance}
            )
            .sort_values("importance", ascending=False)
            .head(20)
        )
        top_features_list = feat_imp.to_dict(orient="records")

        # Clear cached classifier so next infer loads the newly trained one
        global CLASSIFIER_MODEL, CLASSIFIER_META, CLASSIFIER_METRICS, CLASSIFIER_THRESHOLD
        CLASSIFIER_MODEL = None
        CLASSIFIER_META = None
        CLASSIFIER_METRICS = None
        CLASSIFIER_THRESHOLD = None

        return func.HttpResponse(
            json.dumps(
                {
                    "success": True,
                    "message": "Classifier training complete.",
                    "model_blob": BLOB_CLASSIFIER_MODEL_NAME,
                    "feature_meta_blob": BLOB_CLASSIFIER_FEATURE_META,
                    "metrics_blob": BLOB_CLASSIFIER_METRICS,
                    "best_iteration": best_iteration,
                    "optuna_best_params": best_params,
                    "recommended_threshold": all_metrics["recommended_threshold"],
                    "val_roc_auc": all_metrics.get("val_roc_auc"),
                    "val_best_f1": all_metrics.get("val_best_f1"),
                    "test_roc_auc": all_metrics.get("test_roc_auc"),
                    "test_best_f1": all_metrics.get("test_best_f1"),
                    "top_20_features": top_features_list,
                    "train_samples": len(train_df),
                    "val_samples": len(val_df),
                    "test_samples": len(test_df),
                },
                indent=2,
            ),
            status_code=200,
            mimetype="application/json",
        )

    except Exception as e:
        logging.exception("Error during classifier training execution")
        return func.HttpResponse(
            json.dumps({"success": False, "error": str(e)}, indent=2),
            status_code=500,
            mimetype="application/json",
        )


# =============================================================================
# HTTP Function 4: Classifier Inference
# =============================================================================
@app.function_name(name="classifier_infer")
@app.route(route="classifier_infer", methods=["POST"])
def classifier_infer(req: func.HttpRequest) -> func.HttpResponse:
    """
    Hold / No-Hold classification inference.

    JSON body:
      - data: list[dict] — case records (required)
      - threshold: float (optional, overrides recommended_threshold)
      - save_to_blob: bool (default: true)
      - filename: str (default: DEFAULT_CLASSIFIER_FILE)
      - deduplicate_column: str (default: "TicketNumber")
      - include_input: bool (default: false) — include original fields in output
    """
    try:
        load_classifier_model_if_needed()

        try:
            payload = req.get_json()
        except ValueError:
            return func.HttpResponse(
                json.dumps({"success": False, "error": "Invalid JSON"}),
                status_code=400,
                mimetype="application/json",
            )

        save_to_blob = payload.get("save_to_blob", True)
        custom_filename = payload.get("filename", DEFAULT_CLASSIFIER_FILE)
        deduplicate_column = payload.get("deduplicate_column", "TicketNumber")
        include_input = payload.get("include_input", False)
        threshold_override = payload.get("threshold", None)

        threshold = (
            float(threshold_override)
            if threshold_override is not None
            else CLASSIFIER_THRESHOLD
        )

        if "data" in payload:
            records = payload["data"]
        else:
            control_params = {
                "save_to_blob",
                "filename",
                "deduplicate_column",
                "include_input",
                "threshold",
            }
            records = {k: v for k, v in payload.items() if k not in control_params}

        if not isinstance(records, list):
            records = [records]

        df = pd.DataFrame(records)

        # Prepare features
        X = classifier_prepare_inference(df)

        # Predict
        cat_cols = CLASSIFIER_META["categorical_columns"]
        pool = Pool(X, cat_features=cat_cols)
        probas = CLASSIFIER_MODEL.predict_proba(pool)[:, 1]
        preds = (probas >= threshold).astype(int)
        labels = np.where(preds == 1, "Hold", "No Hold")

        result = pd.DataFrame(
            {
                "hold_probability": np.round(probas, 6),
                "prediction": preds,
                "prediction_label": labels,
            }
        )

        # Add ticket number if available
        if "TicketNumber" in df.columns:
            result.insert(0, "TicketNumber", df["TicketNumber"].values)

        if include_input:
            # Prefix original columns to avoid name collisions
            result = pd.concat([df.reset_index(drop=True), result], axis=1)

        response_data: Dict[str, Any] = {
            "success": True,
            "records_processed": len(df),
            "threshold_used": threshold,
            "hold_count": int(preds.sum()),
            "no_hold_count": int((preds == 0).sum()),
            "results": result.to_dict(orient="records"),
        }

        if save_to_blob:
            if not custom_filename.endswith(".csv"):
                custom_filename = f"{custom_filename}.csv"
            blob_path = f"{OUTPUT_FOLDER}/{custom_filename}"
            save_info = save_results_to_blob(result, blob_path, deduplicate_column)

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

        return func.HttpResponse(
            json.dumps(response_data, indent=2, default=str),
            status_code=200,
            mimetype="application/json",
        )

    except Exception as e:
        logging.exception("Unhandled error in classifier_infer.")
        return func.HttpResponse(
            json.dumps({"success": False, "error": str(e)}),
            status_code=500,
            mimetype="application/json",
        )