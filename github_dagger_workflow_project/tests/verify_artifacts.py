import os
import sys


artifacts_dir = "/pipeline/github_dagger_workflow_project/artifacts"
files_to_check = [
    "cat_missing_impute.csv",
    "columns_drift.json",
    "columns_list.json",
    "date_limits.json",
    "lead_model_lr.pkl",
    "lead_model_xgboost.json",
    "model_results.json",
    "outlier_summary.csv",
    "raw_data.csv",
    "scaler.pkl",
    "train_data_gold.csv",
    "training_data.csv",
    "best_model.pkl",
    "best_experiment.pkl",
]

missing_files = [f for f in files_to_check if not os.path.isfile(os.path.join(artifacts_dir, f))]

if missing_files:
    print(f"Missing files: {', '.join(missing_files)}")
    sys.exit(1)  # Exit with non-zero status to indicate failure
else:
    print("All required files are present.")
    sys.exit(0)  # Exit with zero status to indicate success