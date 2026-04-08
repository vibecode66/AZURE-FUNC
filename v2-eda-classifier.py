import os
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- LOCAL CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_INPUT_DIR = SCRIPT_DIR / "data"

BASE_DIR = SCRIPT_DIR / "classifier_output"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DATA_DIR = BASE_DIR / "predictions"

# Create directories if they don't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Local File Paths
LOCAL_MODEL_PATH = MODEL_DIR / "catboost_hold_classifier.cbm"
LOCAL_FEATURE_META_PATH = MODEL_DIR / "feature_metadata.json"
LOCAL_PREDICTIONS_PATH = OUTPUT_DATA_DIR / "hold_predictions.csv"

# --- CONSTANTS ---
RAW_CATEGORICAL_COLUMNS = [
    "msdyn_caserocname", "msdyn_businessfunctionidname", "msdyn_lineofbusinessidname",
    "msdyn_programidname", "msdyn_casetypeidname", "msdyn_casesubtypeidname",
    "msdyn_casereasonidname", "msdyn_casesubreasonidname", "prioritycodename",
    "msdyn_complexityname", "msdyn_countrysubmitteridname", "msdyn_countryprocessedidname", "statuscodename"
]
DERIVED_CATEGORICAL_COLUMNS = ["CaseHierarchy", "ReasonGroup", "LOBProgram"]
ALL_CATEGORICAL_COLUMNS = RAW_CATEGORICAL_COLUMNS + DERIVED_CATEGORICAL_COLUMNS
DERIVED_NUMERIC_COLUMNS = [
    "ReasonHoldRate", "SubTypeHoldRate", "ProgramHoldRate", "IsSameCountry",
    "created_hour", "created_dayofweek", "created_is_weekend", "created_month"
]
CLASSIFIER_FEATURE_COLUMNS = RAW_CATEGORICAL_COLUMNS + DERIVED_CATEGORICAL_COLUMNS + DERIVED_NUMERIC_COLUMNS
CLASSIFIER_TARGET_COLUMN = "IsHold"
SMOOTHING_MIN_COUNT = 30
SMOOTHING_STRENGTH = 10


# --- FEATURE ENGINEERING LOGIC ---

def classifier_make_target(df: pd.DataFrame) -> pd.DataFrame:
    df["HoldTimeInMinutes"] = pd.to_numeric(df["HoldTimeInMinutes"], errors="coerce").fillna(0)
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
    df["IsSameCountry"] = (df["msdyn_countrysubmitteridname"] == df["msdyn_countryprocessedidname"]).astype(int)
    df["CaseHierarchy"] = df["msdyn_casetypeidname"].astype(str) + "_" + df["msdyn_casesubtypeidname"].astype(str)
    df["ReasonGroup"] = df["msdyn_casereasonidname"].astype(str) + "_" + df["msdyn_casesubreasonidname"].astype(str)
    df["LOBProgram"] = df["msdyn_lineofbusinessidname"].astype(str) + "_" + df["msdyn_programidname"].astype(str)

    for col in DERIVED_CATEGORICAL_COLUMNS:
        df[col] = df[col].astype(str)
    return df


def _compute_smoothed_rate(group_series: pd.Series, target_series: pd.Series, global_rate: float) -> dict:
    stats = pd.DataFrame({"group": group_series, "target": target_series})
    agg = stats.groupby("group")["target"].agg(["mean", "count"])
    agg["smoothed"] = (agg["count"] * agg["mean"] + SMOOTHING_STRENGTH * global_rate) / (
                agg["count"] + SMOOTHING_STRENGTH)
    agg.loc[agg["count"] < SMOOTHING_MIN_COUNT, "smoothed"] = global_rate
    return agg["smoothed"].to_dict()


def classifier_fit_target_encoders(train_df: pd.DataFrame, y_train: pd.Series) -> tuple:
    global_rate = float(y_train.mean())
    mappings = {
        "ReasonHoldRate": _compute_smoothed_rate(train_df["ReasonGroup"], y_train, global_rate),
        "SubTypeHoldRate": _compute_smoothed_rate(train_df["msdyn_casesubtypeidname"], y_train, global_rate),
        "ProgramHoldRate": _compute_smoothed_rate(train_df["msdyn_programidname"], y_train, global_rate)
    }
    return mappings, global_rate


def classifier_apply_target_encoders(df: pd.DataFrame, mappings: dict, global_rate: float) -> pd.DataFrame:
    df["ReasonHoldRate"] = df["ReasonGroup"].map(mappings["ReasonHoldRate"]).fillna(global_rate).astype(float)
    df["SubTypeHoldRate"] = df["msdyn_casesubtypeidname"].map(mappings["SubTypeHoldRate"]).fillna(global_rate).astype(
        float)
    df["ProgramHoldRate"] = df["msdyn_programidname"].map(mappings["ProgramHoldRate"]).fillna(global_rate).astype(float)
    return df


# --- TRAINING SECTION WITH TUNED GRID SEARCH ---

def train_classifier_local(input_csv_path: str):
    logging.info(f"Loading local data from {input_csv_path}")
    df = pd.read_csv(input_csv_path) if input_csv_path.endswith('.csv') else pd.read_excel(input_csv_path)

    df["createdon"] = pd.to_datetime(df["createdon"], errors="coerce")
    df = df.dropna(subset=["createdon"]).reset_index(drop=True)
    df = classifier_make_target(df)
    df = classifier_build_base_features(df)

    y = df[CLASSIFIER_TARGET_COLUMN]
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=y)
    val_df, _ = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df[CLASSIFIER_TARGET_COLUMN])

    mappings, global_rate = classifier_fit_target_encoders(train_df, train_df[CLASSIFIER_TARGET_COLUMN])
    train_df = classifier_apply_target_encoders(train_df, mappings, global_rate)
    val_df = classifier_apply_target_encoders(val_df, mappings, global_rate)

    cat_cols_in_features = [c for c in ALL_CATEGORICAL_COLUMNS if c in CLASSIFIER_FEATURE_COLUMNS]
    X_train, y_train = train_df[CLASSIFIER_FEATURE_COLUMNS], train_df[CLASSIFIER_TARGET_COLUMN]
    X_val, y_val = val_df[CLASSIFIER_FEATURE_COLUMNS], val_df[CLASSIFIER_TARGET_COLUMN]

    # Initialize Model for Grid Search
    # Added early_stopping_rounds to handle the high iteration counts safely
    model = CatBoostClassifier(
        eval_metric="AUC",
        auto_class_weights="Balanced",
        early_stopping_rounds=100,
        verbose=False,
        allow_writing_files=False
    )

    # UPDATED PARAMETER GRID with your requested iterations
    # Included a very small learning rate (0.01) to support 5000 iterations
    param_grid = {
        'iterations': [1000, 3000, 5000],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [4, 6, 8],
    }

    logging.info(f"Starting Grid Search (Caution: Iterations up to 5000 may take time)...")
    grid_search_results = model.grid_search(
        param_grid,
        Pool(X_train, y_train, cat_features=cat_cols_in_features),
        cv=3,
        partition_random_seed=42,
        plot=False,
        verbose=False
    )

    logging.info(f"Best parameters selected: {grid_search_results['params']}")

    # Final Fit with best params
    model.fit(
        Pool(X_train, y_train, cat_features=cat_cols_in_features),
        eval_set=Pool(X_val, y_val, cat_features=cat_cols_in_features),
        verbose=100
    )

    model.save_model(str(LOCAL_MODEL_PATH))

    feature_meta = {
        "feature_columns": CLASSIFIER_FEATURE_COLUMNS,
        "categorical_columns": cat_cols_in_features,
        "global_hold_rate_train": global_rate,
        "target_encoding_mappings": {k: {str(key): float(val) for key, val in v.items()} for k, v in mappings.items()},
        "best_params": grid_search_results['params']
    }
    with open(LOCAL_FEATURE_META_PATH, "w") as f:
        json.dump(feature_meta, f, indent=2)

    logging.info(f"Final Model saved to {MODEL_DIR}")


# --- INFERENCE SECTION ---

def run_inference_local(input_file_path: str):
    model = CatBoostClassifier()
    model.load_model(str(LOCAL_MODEL_PATH))
    with open(LOCAL_FEATURE_META_PATH, "r") as f:
        meta = json.load(f)

    df = pd.read_csv(input_file_path) if input_file_path.endswith('.csv') else pd.read_excel(input_file_path)
    df_proc = classifier_build_base_features(df)
    df_proc = classifier_apply_target_encoders(df_proc, meta["target_encoding_mappings"],
                                               meta["global_hold_rate_train"])

    X = df_proc[meta["feature_columns"]]
    for col in meta["categorical_columns"]:
        X[col] = X[col].astype(str)

    pool = Pool(X, cat_features=meta["categorical_columns"])
    probas = model.predict_proba(pool)[:, 1]

    df["hold_probability"] = np.round(probas, 4)
    df["prediction"] = (probas >= 0.5).astype(int)
    df["prediction_label"] = np.where(df["prediction"] == 1, "Hold", "No Hold")

    df.to_csv(LOCAL_PREDICTIONS_PATH, index=False)
    logging.info(f"Inference complete. Results saved to {LOCAL_PREDICTIONS_PATH}")


if __name__ == "__main__":
    # Ensure file is in the 'data' folder
    FILENAME = "catboost_train_dataset_raw.xlsx"
    input_data_file = DATA_INPUT_DIR / FILENAME

    if input_data_file.exists():
        train_classifier_local(str(input_data_file))
    else:
        logging.error(f"Could not find file: {input_data_file}")

