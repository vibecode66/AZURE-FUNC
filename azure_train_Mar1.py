import azure.functions as func
import os
import io
import yaml
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from pathlib import Path
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# Configuration (Matching Inference Settings)
# -----------------------------------------------------------------------------
STORAGE_ACCOUNT_URL = os.getenv("STORAGE_ACCOUNT_URL")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")  # Main container for data/config
MODEL_PATH_OUT = os.getenv("BLOB_MODEL_NAME", "model/catboost_model_deploy.cbm")

app = func.FunctionApp()


# --- UTILITIES (Updated for Managed Identity) ---

def _get_credential() -> DefaultAzureCredential:
    """Managed identity selection; uses explicit client_id when present in Azure."""
    client_id = os.getenv("AZURE_CLIENT_ID") or os.getenv("MANAGED_IDENTITY_CLIENT_ID")
    return DefaultAzureCredential(managed_identity_client_id=client_id)


def _get_container_client():
    """Get a container client using Managed Identity"""
    credential = _get_credential()
    svc = BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=credential)
    return svc.get_container_client(BLOB_CONTAINER_NAME)


def load_blob_to_df(blob_name):
    container_client = _get_container_client()
    blob_client = container_client.get_blob_client(blob_name)
    stream = blob_client.download_blob().readall()

    if blob_name.endswith('.xlsx'):
        return pd.read_excel(io.BytesIO(stream), engine='openpyxl')
    elif blob_name.endswith('.csv'):
        return pd.read_csv(io.BytesIO(stream))
    elif blob_name.endswith(('.yaml', '.yml')):
        return yaml.safe_load(stream)
    return None


def save_df_to_blob(df, blob_name):
    container_client = _get_container_client()
    output = io.BytesIO()
    if blob_name.endswith('.csv'):
        df.to_csv(output, index=False)
    elif blob_name.endswith('.json'):
        data = df if isinstance(df, dict) else df.to_dict(orient='records')
        output.write(json.dumps(data, indent=4).encode('utf-8'))

    output.seek(0)
    container_client.upload_blob(name=blob_name, data=output, overwrite=True)


def save_plt_to_blob(plt_obj, blob_name):
    container_client = _get_container_client()
    img_data = io.BytesIO()
    plt_obj.savefig(img_data, format='png', bbox_inches='tight')
    img_data.seek(0)
    container_client.upload_blob(name=blob_name, data=img_data, overwrite=True)
    plt_obj.clf()
    plt_obj.close()


# --- DATA ENGINEERING FUNCTIONS (Kept from your original logic) ---

def clean_and_engineer(df, quantile_limit=0.75):
    df['received_dt'] = pd.to_datetime(df['msdyn_receiveddate'], errors='coerce')
    df['resolved_dt'] = pd.to_datetime(df['msdyn_resolveddate'], errors='coerce')
    df = df.dropna(subset=['received_dt', 'resolved_dt'])
    df['target_minutes'] = (df['resolved_dt'] - df['received_dt']).dt.total_seconds() / 60
    df = df[(df['target_minutes'] > 30) & (df['target_minutes'] < 43200)]
    df = df.sort_values('received_dt').set_index('received_dt')
    df['backlog_4h'] = df['msdyn_caserocname'].rolling(window='4h').count() - 1
    df['backlog_24h'] = df['msdyn_caserocname'].rolling(window='24h').count() - 1
    df = df.reset_index()
    df['daily_volume'] = df.groupby(df['received_dt'].dt.date)['msdyn_caserocname'].transform('count')
    df['hour_of_day'] = df['received_dt'].dt.hour.astype(str)
    df['day_of_week'] = df['received_dt'].dt.dayofweek.astype(str)
    upper_limit = df['target_minutes'].quantile(quantile_limit)
    df_clean = df[df['target_minutes'] < upper_limit].copy()
    median_map = df_clean.groupby('msdyn_casereasonidname')['target_minutes'].median()
    df_clean['reason_median_speed'] = df_clean['msdyn_casereasonidname'].map(median_map)
    return df_clean, median_map


def clean_categorical_strings(df, cat_cols):
    for col in cat_cols:
        df[col] = df[col].map(
            lambda x: str(int(x)) if pd.notnull(x) and isinstance(x, (float, int)) else str(x)).replace('nan',
                                                                                                        'Unknown')
    return df


def select_top_features(df, candidate_features, categorical_cols, target_col, top_n=8):
    X_temp = df[candidate_features]
    y_temp = np.log1p(df[target_col])
    selector_model = CatBoostRegressor(iterations=300, cat_features=categorical_cols, verbose=0)
    selector_model.fit(X_temp, y_temp)
    feat_imp = pd.Series(selector_model.get_feature_importance(), index=candidate_features).sort_values(ascending=False)
    top_features = feat_imp.head(top_n).index.tolist()
    return top_features


# --- AZURE FUNCTION: TRAINING TRIGGER ---

@app.route(route="catboost_train", methods=["POST", "GET"])
def ManualTrainingPipeline(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Manual ML Pipeline execution started.')

    try:
        # 1. SETUP & CONFIG (Assuming config.yaml is in the root of the container)
        config = load_blob_to_df("config.yaml")
        if not config:
            raise ValueError("Could not find config.yaml in storage.")

        plot_folder = config['paths']['plot_folder'].strip("/")

        # 2. LOAD & CLEAN
        df_raw_loaded = load_blob_to_df(config['paths']['raw_data'])
        df_raw = df_raw_loaded.rename(columns={'createdon': 'msdyn_receiveddate'})
        df_raw['msdyn_receiveddate'] = pd.to_datetime(df_raw['msdyn_receiveddate'], errors='coerce')

        # Filtering for relevant data
        df_raw = df_raw[df_raw['msdyn_receiveddate'].dt.year >= 2025].copy()
        df_raw = df_raw[(df_raw['msdyn_resolveddate'].notna()) &
                        (df_raw['msdyn_caserocname'].notna()) &
                        ((df_raw['onholdtime'] == 0) | (df_raw['onholdtime'].isna()))].copy()

        df_clean, median_map = clean_and_engineer(df_raw, config['params']['outlier_quantile'])
        df_clean = clean_categorical_strings(df_clean, config['params']['categorical_cols'])

        # 3. EDA PLOTS
        plt.figure(figsize=(12, 6))
        sns.histplot(df_clean[config['params']['target_col']], bins=40, kde=True, color='purple')
        save_plt_to_blob(plt, f"{plot_folder}/target_distribution_y.png")

        # 4. FEATURE SELECTION & TRAINING
        top_features = select_top_features(df_clean, config['params']['candidate_features'],
                                           config['params']['categorical_cols'], config['params']['target_col'])

        final_cat_features = [f for f in top_features if f in config['params']['categorical_cols']]

        X = df_clean[top_features]
        y = np.log1p(df_clean[config['params']['target_col']])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

        final_model = CatBoostRegressor(iterations=3000, learning_rate=0.015, depth=10, loss_function='MAE',
                                        cat_features=final_cat_features, verbose=100)
        final_model.fit(X_train, y_train, eval_set=(X_test, y_test))

        # 5. SAVING MODEL (Using Tempfile for cross-platform Azure compatibility)
        temp_dir = Path(tempfile.gettempdir())
        temp_model_path = temp_dir / "temp_model.cbm"
        final_model.save_model(str(temp_model_path))

        container_client = _get_container_client()
        with open(temp_model_path, "rb") as f:
            container_client.upload_blob(name=MODEL_PATH_OUT, data=f, overwrite=True)

        # 6. SAVE METADATA
        global_median = df_clean[config['params']['target_col']].median()
        metadata = {
            "global_median": float(global_median),
            "median_map": median_map.to_dict(),
            "features": top_features,
            "trained_at": datetime.utcnow().isoformat()
        }
        metadata_blob_path = f"{os.path.dirname(MODEL_PATH_OUT)}/median_map.json"
        save_df_to_blob(metadata, metadata_blob_path)

        return func.HttpResponse(
            json.dumps({
                "success": True,
                "message": f"Model saved to {MODEL_PATH_OUT}",
                "features_used": top_features
            }),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.exception("Error during training execution")
        return func.HttpResponse(
            json.dumps({"success": False, "error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

