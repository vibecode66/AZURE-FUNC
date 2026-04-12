import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import yaml
import json
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_absolute_error, r2_score

# 0. Load Configuration
with open("config.yaml", "r") as f:
    full_config = yaml.safe_load(f)['regression_training']

paths = full_config['paths']
schema = full_config['data_schema']
prep = full_config['preprocessing']

# 1. Load Data
df = pd.read_excel(paths['raw_data'])

# 2. Create Target Variable & Handle Nulls/Outliers
df['target'] = df[schema['target_col']] - df[schema['hold_time_col']].fillna(0)
df = df.dropna(subset=['target']).copy()

upper_limit = df['target'].quantile(prep['outlier_quantile'])
df = df[(df['target'] > 0) & (df['target'] <= upper_limit)].copy()

# Log Transformation
df['target_log'] = np.log1p(df['target'])

# --- FEATURE ENGINEERING ---
date_col = schema['date_col']
if date_col in df.columns:
    df[date_col] = pd.to_datetime(df[date_col])
    df['resolved_hour'] = df[date_col].dt.hour
    df['resolved_day_of_week'] = df[date_col].dt.dayofweek
    df['is_weekend'] = df['resolved_day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_business_hour'] = df['resolved_hour'].between(
        prep['business_hour_start'],
        prep['business_hour_end']
    ).astype(int)

# --- ADDED: Median of Subreason Logic ---
# Identifying the subreason column from schema or assuming standard naming
subreason_col = 'msdyn_casesubreasonidname' # Ensure this matches your excel column name
if subreason_col in df.columns:
    # Calculate median per subreason based on the target
    subreason_medians = df.groupby(subreason_col)['target'].median().to_dict()
    # Map the median values back to the dataframe as a new feature
    df['subreason_median_target'] = df[subreason_col].map(subreason_medians)
    # Fill any potential NaNs (for new subreasons) with the global median
    df['subreason_median_target'] = df['subreason_median_target'].fillna(df['target'].median())

# Text Length Features
for col in schema['text_cols']:
    if col in df.columns:
        df[f'{col}_len'] = df[col].astype(str).apply(lambda x: len(x) if x != 'nan' else 0)

# 3. Clean Categorical Features
internal_drops = ['target', 'target_log']
all_drop_cols = schema['drop_cols'] + internal_drops

cat_features = list(df.select_dtypes(include=['object', 'category']).columns)
cat_features = [c for c in cat_features if c in df.columns and c not in all_drop_cols]

for col in cat_features:
    df[col] = df[col].astype(str).replace('nan', 'Unknown')

# 4. Feature Selection
# 'subreason_median_target' is now included in X because it is not in 'all_drop_cols'
X = df.drop(columns=[c for c in all_drop_cols if c in df.columns])
y = df['target_log']

# Capture Metadata for regressor_infer.py
feature_names = X.columns.tolist()
cat_indices = [i for i, col in enumerate(feature_names) if col in cat_features]

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=prep['test_size'],
    random_state=prep['random_state']
)

# 6. Grid Search
param_grid = full_config['grid_search_params']
best_score = -np.inf
best_params, best_model = None, None

print("Starting Manual Grid Search...")
for params in ParameterGrid(param_grid):
    model = CatBoostRegressor(
        **params,
        loss_function='RMSE',
        random_seed=prep['random_state'],
        verbose=False,
        cat_features=cat_features
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)

    if score > best_score:
        best_score, best_params, best_model = score, params, model

# 7. Final Evaluation
log_preds = best_model.predict(X_test)
actual_preds = np.expm1(log_preds)
actual_y_test = np.expm1(y_test)

# 8. SAVE MODEL & INFER METADATA
best_model.save_model(paths['model_output'])

# Save metadata json for the inference script (updated with new feature names)
metadata = {
    "features": feature_names,
    "cat_indices": cat_indices,
    "target": schema['target_col'],
    "subreason_medians": subreason_medians if subreason_col in df.columns else {}
}
with open("catboost_regressor_metadata.json", "w") as f:
    json.dump(metadata, f)

# 9. SIMULATED INFERENCE
print("Running Inference on Test Set...")
preds_minutes = np.expm1(best_model.predict(X_test))

infer_results = []
for i in range(len(X_test)):
    arrival = pd.to_datetime(X_test.iloc[i].get(schema['date_col'], pd.Timestamp.now()))
    pred_min = float(preds_minutes[i])

    raw_completion = arrival + pd.Timedelta(minutes=pred_min)

    adj_completion = raw_completion
    if raw_completion.weekday() >= 5:
        adj_completion += pd.Timedelta(days=2)

    infer_results.append({
        "predicted_minutes": pred_min,
        "expected_resolution_date": adj_completion.isoformat()
    })

df_final_predictions = pd.DataFrame(infer_results)
df_final_predictions.to_csv("inference_output_results.csv", index=False)

# 10. VISUALIZATIONS
viz_folder = paths['viz_folder']
os.makedirs(viz_folder, exist_ok=True)

fea_imp = pd.DataFrame({'imp': best_model.feature_importances_, 'col': X.columns}).sort_values(by='imp', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x='imp', y='col', data=fea_imp, palette='viridis')
plt.title("Top 10 Features (Including Subreason Median)")
plt.savefig(f'{viz_folder}/feature_importance.png')

print(f"Workflow complete. Results saved to 'inference_output_results.csv'.")

