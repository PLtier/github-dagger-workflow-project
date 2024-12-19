import pandas as pd
from github_dagger_workflow_project import pipeline_utils as pu
from github_dagger_workflow_project.config import (
    PROD_BEST_EXPERIMENT_PATH,
    ARTIFACT_PATH,
    MODEL_NAME,
)


experiment_best = pd.read_pickle(PROD_BEST_EXPERIMENT_PATH)
train_model_score = experiment_best["metrics.f1_score"]
run_id = None

prod_model = pu.get_production_model(MODEL_NAME)
prod_model_exists = len(prod_model) > 0

if prod_model_exists:
    prod_model_run_id = dict(prod_model[0])["run_id"]
    prod_model_score = pu.get_model_score(prod_model_run_id)

    if train_model_score > prod_model_score:
        run_id = experiment_best["run_id"]
else:
    run_id = experiment_best["run_id"]

if run_id is not None:
    model_details = pu.register_and_wait_model(run_id, ARTIFACT_PATH, MODEL_NAME)
