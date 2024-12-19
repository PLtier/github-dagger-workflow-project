import mlflow
import pandas as pd
from github_dagger_workflow_project import utils
from github_dagger_workflow_project.config import PROD_BEST_EXPERIMENT_PATH
from github_dagger_workflow_project.mlflow_client import client


def get_production_model(model_name):
    return [
        model
        for model in client.search_model_versions(f"name='{model_name}'")
        if dict(model)["current_stage"] == "Production"
    ]


def get_model_score(run_id):
    data, _ = mlflow.get_run(run_id)
    return data[1]["metrics.f1_score"]


def register_and_wait_model(run_id, artifact_path, model_name):
    model_uri = f"runs:/{run_id}/{artifact_path}"
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    utils.wait_until_ready(model_details.name, model_details.version)
    return dict(model_details)


artifact_path = "model"
model_name = "lead_model"
experiment_best = pd.read_pickle(PROD_BEST_EXPERIMENT_PATH)


train_model_score = experiment_best["metrics.f1_score"]
run_id = None
prod_model = get_production_model(model_name)
prod_model_exists = len(prod_model) > 0

if prod_model_exists:
    prod_model_run_id = dict(prod_model[0])["run_id"]
    prod_model_score = get_model_score(prod_model_run_id)

    if train_model_score > prod_model_score:
        run_id = experiment_best["run_id"]
else:
    run_id = experiment_best["run_id"]

if run_id is not None:
    model_details = register_and_wait_model(run_id, artifact_path, model_name)
