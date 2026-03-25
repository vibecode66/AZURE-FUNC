import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load Data
df = pd.read_excel('data/raw/catboost_train_dataset_raw.xlsx')

# 2. Create Target Variable & Handle Nulls/Outliers
df['target'] = df['CaseTotalDurationinMinutes'] - df['HoldTimeInMinutes'].fillna(0)
df = df.dropna(subset=['target']).copy()

upper_limit = df['target'].quantile(0.70)
df = df[(df['target'] > 0) & (df['target'] <= upper_limit)].copy()

# Log Transformation
df['target_log'] = np.log1p(df['target'])

# --- NEW FEATURES: DATE & TEXT ---
if 'msdyn_resolveddate' in df.columns:
    df['msdyn_resolveddate'] = pd.to_datetime(df['msdyn_resolveddate'])
    df['resolved_hour'] = df['msdyn_resolveddate'].dt.hour
    df['resolved_day_of_week'] = df['msdyn_resolveddate'].dt.dayofweek
    df['is_weekend'] = df['resolved_day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_business_hour'] = df['resolved_hour'].between(9, 17).astype(int)

# Text Length Features
for col in ['title', 'description', 'subject']:
    if col in df.columns:
        df[f'{col}_len'] = df[col].astype(str).apply(lambda x: len(x) if x != 'nan' else 0)

# 3. Clean Categorical Features
cat_features = list(df.select_dtypes(include=['object', 'category']).columns)
cols_to_drop = ['target', 'target_log', 'CaseTotalDurationinMinutes', 'HoldTimeInMinutes',
                'ActualCaseTimeinMinutes', 'msdyn_resolveddate']

cat_features = [c for c in cat_features if c in df.columns and c not in cols_to_drop]
for col in cat_features:
    df[col] = df[col].astype(str).replace('nan', 'Unknown')

# 4. Feature Selection
X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
y = df['target_log']

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. MANUAL GRID SEARCH (Bypassing SKLearn's incompatible engine)
param_grid = {
    'iterations': [2000, 3500,5000],
    'learning_rate': [0.03, 0.05],
    'depth': [6, 8,10],
    'l2_leaf_reg': [3, 5]
}

best_score = -np.inf
best_params = None
best_model = None

print("Starting Manual Grid Search to ensure compatibility...")
for params in ParameterGrid(param_grid):
    print(f"Testing: {params}")
    model = CatBoostRegressor(
        **params,
        loss_function='RMSE',
        random_seed=42,
        verbose=False,
        cat_features=cat_features
    )
    model.fit(X_train, y_train)

    # Evaluate on Test Set
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)

    if score > best_score:
        best_score = score
        best_params = params
        best_model = model

print(f"\nOptimization Finished!")
print(f"Best Parameters: {best_params}")

# 7. Final Evaluation (Converting Log back to Minutes)
log_preds = best_model.predict(X_test)
actual_preds = np.expm1(log_preds)
actual_y_test = np.expm1(y_test)

mae = mean_absolute_error(actual_y_test, actual_preds)
r2 = r2_score(actual_y_test, actual_preds)

print(f"\n--- Results ---")
print(f"Mean Absolute Error: {mae:.2f} minutes")
print(f"R-Squared Score: {r2:.4f}")

# 8. SAVE MODEL & RESULTS
best_model.save_model('catboost_final_model.cbm')

results_summary = pd.DataFrame({
    'Metric': ['MAE', 'R2_Score', 'Best_Depth', 'Best_Iterations', 'Best_LR'],
    'Value': [mae, r2, best_params['depth'], best_params['iterations'], best_params['learning_rate']]
})
results_summary.to_csv('final_model_performance.csv', index=False)
print("Files 'catboost_final_model.cbm' and 'final_model_performance.csv' saved.")

# 9. Visualization (Top 10 Features)
fea_imp = pd.DataFrame({'imp': best_model.feature_importances_, 'col': X.columns})
fea_imp = fea_imp.sort_values(by='imp', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x='imp', y='col', data=fea_imp, palette='viridis')
plt.title("Top 10 Features")
plt.show()

# Actual vs Predicted Plot
plt.figure(figsize=(8, 8))
plt.scatter(actual_y_test, actual_preds, alpha=0.3, color='teal')
plt.plot([actual_y_test.min(), actual_y_test.max()], [actual_y_test.min(), actual_y_test.max()], 'r--')
plt.xlabel("Actual Minutes")
plt.ylabel("Predicted Minutes")
plt.title("Actual vs Predicted")
plt.show()

