import yaml
import os
import pandas as pd
import numpy as np
from src.data_loader import load_raw_data
from src.preprocess import clean_and_engineer, clean_categorical_strings
from src.train import select_top_features, train_final_model
from src.predict import generate_demo_predictions
from src.visualization import save_eda_plots, plot_training_results, plot_demo_performance

# 1. SETUP & CONFIG
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Ensure all output directories exist (derived from YAML paths)
folders_to_ensure = [
    config['paths']['plot_folder'],
    os.path.dirname(config['paths']['processed_data']),
    os.path.dirname(config['paths']['model_path']),
    os.path.dirname(config['paths']['demo_results'])
]

for folder in folders_to_ensure:
    if folder: # Ensure the string isn't empty
        os.makedirs(folder, exist_ok=True)

# 2. LOAD & CLEAN
df_raw = load_raw_data(config['paths']['raw_data'])
df_clean, median_map = clean_and_engineer(df_raw, config['params']['outlier_quantile'])
df_clean = clean_categorical_strings(df_clean, config['params']['categorical_cols'])

# 3. EDA PLOTS
print("Generating EDA plots...")
save_eda_plots(
    df_clean,
    df_clean[config['params']['candidate_features']],
    df_clean[config['params']['target_col']],
    config['params']['categorical_cols'],
    config['paths']['plot_folder']
)

# 4. SAVE DEMO DATA
demo_sample = df_clean.sample(n=min(100, len(df_clean)), random_state=config['params']['random_state'])
demo_sample.to_csv("data/processed/Inference.csv", index=False)

# 5. FEATURE SELECTION & TRAINING
top_features, top_cats = select_top_features(
    df_clean, config['params']['candidate_features'],
    config['params']['categorical_cols'], config['params']['target_col']
)


final_model, X_test, y_test = train_final_model(
    df_clean, top_features, top_cats, config['params']['target_col']
)

# 6. TRAINING PLOTS (Residuals/Importance)
preds_actual = np.expm1(final_model.predict(X_test))
y_test_actual = np.expm1(y_test)
plot_training_results(final_model, top_features, y_test_actual, preds_actual, config['paths']['plot_folder'])

# # 7. PREDICTION ON DEMO
# demo_results = generate_demo_predictions(
#     final_model, demo_sample, top_features, top_cats,
#     median_map, df_clean[config['params']['target_col']].median()
# )
# demo_results.to_csv(config['paths']['demo_results'], index=False)

# 7. PREDICTION ON DEMO
# NOTE: This function now prints R2/MAE and SAVES the CSV automatically using the YAML config
demo_results = generate_demo_predictions(
    final_model,
    demo_sample,
    top_features,
    top_cats,
    median_map,
    df_clean[config['params']['target_col']].median(),
    config_path='config.yaml'
)

# 8. DEMO PLOTS
plot_demo_performance(demo_results, config['paths']['plot_folder'])

# 9. SAVE MODEL
final_model.save_model(config['paths']['model_path'])

print("-" * 30)
print("Pipeline executed successfully.")
print(f"Predictions saved to: {config['paths']['demo_results']}")
print(f"Plots saved to:       {config['paths']['plot_folder']}")
print("-" * 30)

print("Pipeline executed successfully. Check /reports/figures for results.")
