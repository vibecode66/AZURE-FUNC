import os
import io
import re
import yaml
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import fill
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# --- AZURE UTILITIES ---
# Replace with your actual connection string or use environment variables
AZURE_CONNECTION_STRING = "your_connection_string_here"
CONTAINER_NAME = "your-container-name"

blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)


def load_blob_to_df(blob_name):
    blob_client = container_client.get_blob_client(blob_name)
    stream = blob_client.download_blob().readall()
    if blob_name.endswith('.xlsx'):
        return pd.read_excel(io.BytesIO(stream), engine='openpyxl')
    elif blob_name.endswith('.csv'):
        return pd.read_csv(io.BytesIO(stream))
    elif blob_name.endswith('.yaml') or blob_name.endswith('.yml'):
        return yaml.safe_load(stream)


def save_df_to_blob(df, blob_name):
    output = io.BytesIO()
    if blob_name.endswith('.csv'):
        df.to_csv(output, index=False)
    elif blob_name.endswith('.json'):
        output.write(json.dumps(df, indent=4).encode('utf-8'))

    output.seek(0)
    container_client.upload_blob(name=blob_name, data=output, overwrite=True)


def save_plt_to_blob(plt_obj, blob_name):
    img_data = io.BytesIO()
    plt_obj.savefig(img_data, format='png', bbox_inches='tight')
    img_data.seek(0)
    container_client.upload_blob(name=blob_name, data=img_data, overwrite=True)
    plt_obj.clf()
    plt_obj.close()


# --- REFACTORED FUNCTIONS FROM YOUR MODULES ---

def adjust_for_weekend(dt):
    if dt.weekday() == 5: return dt + pd.Timedelta(days=2)
    if dt.weekday() == 6: return dt + pd.Timedelta(days=1)
    return dt


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
    final_cat_features = [f for f in top_features if f in categorical_cols]
    return top_features, final_cat_features


# --- MAIN EXECUTION PIPELINE ---

def run_pipeline():
    # 1. SETUP & CONFIG (Assuming config.yaml is also in Blob)
    config = load_blob_to_df("config.yaml")
    plot_folder = config['paths']['plot_folder'].strip("/")

    # 2. LOAD & CLEAN
    df_raw_loaded = load_blob_to_df(config['paths']['raw_data'])
    df_raw = df_raw_loaded.rename(columns={'createdon': 'msdyn_receiveddate'})
    df_raw['msdyn_receiveddate'] = pd.to_datetime(df_raw['msdyn_receiveddate'], errors='coerce')
    df_raw = df_raw[df_raw['msdyn_receiveddate'].dt.year == 2025].copy()

    df_raw = df_raw[(df_raw['msdyn_resolveddate'].notna()) & (df_raw['msdyn_caserocname'].notna()) &
                    ((df_raw['onholdtime'] == 0) | (df_raw['onholdtime'].isna()))].copy()

    df_clean, median_map = clean_and_engineer(df_raw, config['params']['outlier_quantile'])
    df_clean = clean_categorical_strings(df_clean, config['params']['categorical_cols'])

    # 3. EDA PLOTS
    print("Generating EDA plots to Blob...")
    # A. Target Distribution
    plt.figure(figsize=(12, 6))
    stats = pd.Series(df_clean[config['params']['target_col']]).describe()
    stats_str = "--- Statistics ---\n" + "\n".join([f"{idx:<8}: {val:>10.2f}" for idx, val in stats.items()])
    sns.histplot(df_clean[config['params']['target_col']], bins=40, kde=True, color='purple')
    plt.annotate(stats_str, xy=(1.02, 0.5), xycoords='axes fraction', family='monospace',
                 bbox=dict(boxstyle="round", fc="white"))
    save_plt_to_blob(plt, f"{plot_folder}/target_distribution_y.png")

    # B. Unique Counts
    unique_counts = pd.Series(
        {col: df_clean[col].nunique() for col in config['params']['categorical_cols']}).sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=unique_counts.values, y=unique_counts.index, palette='viridis')
    save_plt_to_blob(plt, f"{plot_folder}/categorical_unique_counts.png")

    # 4. SAVE DEMO DATA
    demo_sample = df_clean.sample(n=min(100, len(df_clean)), random_state=config['params']['random_state'])
    save_df_to_blob(demo_sample, "data/processed/Inference.csv")

    # 5. FEATURE SELECTION & TRAINING
    top_features, top_cats = select_top_features(df_clean, config['params']['candidate_features'],
                                                 config['params']['categorical_cols'], config['params']['target_col'])

    X = df_clean[top_features]
    y = np.log1p(df_clean[config['params']['target_col']])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

    final_model = CatBoostRegressor(iterations=3000, learning_rate=0.015, depth=10, loss_function='MAE',
                                    cat_features=top_cats, verbose=100)
    final_model.fit(X_train, y_train, eval_set=(X_test, y_test))

    # 6. TRAINING PLOTS
    imps = final_model.get_feature_importance()
    labels = [fill(lbl, width=22) for lbl in top_features]
    plt.figure(figsize=(12, 8))
    sns.barplot(x=imps, y=labels, palette='coolwarm')
    save_plt_to_blob(plt, f"{plot_folder}/optimized_feature_importance.png")

    # 7. PREDICTION ON DEMO
    demo_sample['received_dt'] = pd.to_datetime(demo_sample['msdyn_receiveddate'])
    demo_sample = demo_sample.sort_values('received_dt').set_index('received_dt')
    demo_sample['backlog_4h'] = demo_sample['msdyn_caserocname'].rolling(window='4h').count() - 1
    demo_sample['backlog_24h'] = demo_sample['msdyn_caserocname'].rolling(window='24h').count() - 1
    demo_sample = demo_sample.reset_index()
    demo_sample['hour_of_day'] = demo_sample['received_dt'].dt.hour.astype(str)
    demo_sample['day_of_week'] = demo_sample['received_dt'].dt.dayofweek.astype(str)
    demo_sample['daily_volume'] = demo_sample.groupby(demo_sample['received_dt'].dt.date)[
        'msdyn_caserocname'].transform('count')

    global_median = df_clean[config['params']['target_col']].median()
    demo_sample['reason_median_speed'] = demo_sample['msdyn_casereasonidname'].map(median_map).fillna(global_median)

    demo_sample = clean_categorical_strings(demo_sample, top_cats)
    preds_log = final_model.predict(demo_sample[top_features])
    demo_sample['predicted_resolution_minutes'] = np.expm1(preds_log)

    # 8. DEMO PLOTS & SAVE RESULTS
    plt.figure(figsize=(12, 7))
    sns.scatterplot(x='target_minutes', y='predicted_resolution_minutes', data=demo_sample)
    save_plt_to_blob(plt, f"{plot_folder}/demo_actual_vs_pred_colored.png")

    save_df_to_blob(demo_sample, config['paths']['demo_results'])

    # 9. SAVE MODEL & METADATA
    # Note: CatBoost model saving to bytes requires a local temp file or specific buffer handling
    final_model.save_model("temp_model.cbm")
    with open("temp_model.cbm", "rb") as f:
        container_client.upload_blob(name=config['paths']['model_path'], data=f, overwrite=True)

    metadata = {
        "global_median": float(global_median),
        "median_map": median_map.to_dict()
    }
    save_df_to_blob(metadata, os.path.dirname(config['paths']['model_path']) + "/median_map.json")

    print("Pipeline executed successfully. All files uploaded to Azure Blob Storage.")


if __name__ == "__main__":
    run_pipeline()

