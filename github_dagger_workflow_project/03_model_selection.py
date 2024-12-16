import datetime
import shutil
import mlflow

# Constants used:
current_date = datetime.datetime.now().strftime("%Y_%B_%d")
artifact_path = "model"
model_name = "lead_model"
experiment_name = current_date

# Selecting the best model
experiment_ids = [mlflow.get_experiment_by_name(experiment_name).experiment_id]

experiment_best = mlflow.search_runs(
    experiment_ids=experiment_ids, order_by=["metrics.f1_score DESC"], max_results=1
).iloc[0]
best_model_type = experiment_best["params.model_type"]

if best_model_type == "XGBoost":
    best_model_artifact = "lead_model_xgboost.pkl"
elif best_model_type == "LogisticRegression":
    best_model_artifact = "lead_model_lr.pkl"


original_file_path = f"./artifacts/{best_model_artifact}"
new_file_path = "./artifacts/best_model.pkl"

shutil.copyfile(original_file_path, new_file_path)

experiment_best.to_pickle("./artifacts/best_experiment.pkl")
