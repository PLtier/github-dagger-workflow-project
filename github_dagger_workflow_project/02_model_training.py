import os

import mlflow
import mlflow.pyfunc
import pandas as pd

from github_dagger_workflow_project import pipeline_utils as pu
from github_dagger_workflow_project.config import (
    DATA_GOLD_PATH,
    EXPERIMENT_NAME,
)


# Create directories
os.makedirs("artifacts", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)
os.makedirs("mlruns/.trash", exist_ok=True)

# Set mlflow experiment
mlflow.set_experiment(EXPERIMENT_NAME)

data = pd.read_csv(DATA_GOLD_PATH)

X_train, X_test, y_train, y_test = pu.prepare_data(data)

pu.save_column_list(X_train)

mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

model_results = {}
xgboost_cr = pu.train_xgboost(X_train, X_test, y_train, y_test, experiment_id)
lr_cr = pu.train_linear_regression(X_train, X_test, y_train, y_test, experiment_id)

model_results.update(xgboost_cr)
model_results.update(lr_cr)

pu.save_model_results(model_results)
