import datetime
import shutil
import mlflow
from github_dagger_workflow_project.config import BEST_EXPERIMENT_PATH, BEST_MODEL_PATH


def select_best_model(experiment_name):
    experiment_ids = [mlflow.get_experiment_by_name(experiment_name).experiment_id]
    experiment_best = mlflow.search_runs(
        experiment_ids=experiment_ids, order_by=["metrics.f1_score DESC"], max_results=1
    ).iloc[0]
    return experiment_best


def save_best_model(experiment_best, best_model_type) -> None:
    # store the best experiment (model) metadata
    experiment_best.to_pickle(BEST_EXPERIMENT_PATH)

    if best_model_type == "XGBoost":
        best_model_artifact = "lead_model_xgboost.pkl"
    elif best_model_type == "LogisticRegression":
        best_model_artifact = "lead_model_lr.pkl"

    original_file_path = f"./artifacts/{best_model_artifact}"
    new_file_path = BEST_MODEL_PATH

    shutil.copyfile(original_file_path, new_file_path)


# Constants used:
current_date = datetime.datetime.now().strftime("%Y_%B_%d")
experiment_name = current_date

experiment_best = select_best_model(experiment_name)
# Save best model
# Currently we pick only LR, no matter what the best model is.
# When ready, uncomment to consider xgboost.
# best_model_type = experiment_best["params.model_type"]
# and comment below code
best_model_type = "LogisticRegression"
save_best_model(experiment_best, best_model_type)
