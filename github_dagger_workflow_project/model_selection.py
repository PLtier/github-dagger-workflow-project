import datetime
import json
import shutil

import mlflow
from mlflow.tracking.client import MlflowClient
import pandas as pd

import utils

# Constants used:
current_date = datetime.datetime.now().strftime("%Y_%B_%d")
artifact_path = "model"
model_name = "lead_model"
experiment_name = current_date

# Selecting the best model
experiment_ids = [mlflow.get_experiment_by_name(experiment_name).experiment_id]

experiment_best = mlflow.search_runs(
    experiment_ids=experiment_ids,
    order_by=["metrics.f1_score DESC"],
    max_results=1
).iloc[0]

with open("./artifacts/model_results.json", "r") as f:
    model_results = json.load(f)
results_df = pd.DataFrame({model: val["weighted avg"] for model, val in model_results.items()}).T

best_model = results_df.sort_values("f1-score", ascending=False).iloc[0].name

client = MlflowClient()
prod_model = [model for model in client.search_model_versions(f"name='{model_name}'") if dict(model)['current_stage']=='Production']
prod_model_exists = len(prod_model)>0

if prod_model_exists:
    prod_model_version = dict(prod_model[0])['version']
    prod_model_run_id = dict(prod_model[0])['run_id']

train_model_score = experiment_best["metrics.f1_score"]
model_details = {}
model_status = {}
run_id = None

if prod_model_exists:
    data, details = mlflow.get_run(prod_model_run_id)
    prod_model_score = data[1]["metrics.f1_score"]

    model_status["current"] = train_model_score
    model_status["prod"] = prod_model_score

    if train_model_score > prod_model_score:
        run_id = experiment_best["run_id"]
else:
    run_id = experiment_best["run_id"]

if run_id is not None:
    model_uri = "runs:/{run_id}/{artifact_path}".format(
        run_id=run_id,
        artifact_path=artifact_path
    )
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    utils.wait_until_ready(model_details.name, model_details.version)
    model_details = dict(model_details)

best_model_type = experiment_best["params.model_type"]

if best_model_type == "XGBoost":
    best_model_artifact = "lead_model_xgboost.pkl"
elif best_model_type == "LogisticRegression":
    best_model_artifact = "lead_model_lr.pkl"
    

original_file_path = f"./artifacts/{best_model_artifact}"
new_file_path = "./artifacts/best_model.pkl"

shutil.copyfile(original_file_path, new_file_path)
