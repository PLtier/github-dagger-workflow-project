import datetime
import json
import os

import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd
from scipy.stats import uniform, randint
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRFClassifier

from github_dagger_workflow_project import utils


# Constants used:
current_date = datetime.datetime.now().strftime("%Y_%B_%d")
data_gold_path = "./artifacts/train_data_gold.csv"
data_version = "00000"
experiment_name = current_date

# Create directories
os.makedirs("artifacts", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)
os.makedirs("mlruns/.trash", exist_ok=True)

# Set mlflow experiment
mlflow.set_experiment(experiment_name)

data = pd.read_csv(data_gold_path)


def prepare_data(data: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Drops unnecessary columns,
    creates dummy variables for categorical features,
    sets float as common type for all features,
    then splits the data into train and test sets.
    """

    data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)

    cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
    cat_vars = data[cat_cols]
    for col in cat_vars:
        cat_vars[col] = cat_vars[col].astype("category")
        cat_vars = utils.create_dummy_cols(cat_vars, col)

    other_vars = data.drop(cat_cols, axis=1)
    data = pd.concat([other_vars, cat_vars], axis=1)
    for col in data:
        data[col] = data[col].astype("float64")

    y = data["lead_indicator"]
    X = data.drop(["lead_indicator"], axis=1)

    return train_test_split(X, y, random_state=42, test_size=0.15, stratify=y)


def save_column_list(X_train: pd.DataFrame) -> None:
    """
    Saves the list of columns to a json file.
    """
    column_list_path = "./artifacts/columns_list.json"
    with open(column_list_path, "w+") as columns_file:
        columns = {"column_names": list(X_train.columns)}
        json.dump(columns, columns_file)


X_train, X_test, y_train, y_test = prepare_data(data)

save_column_list(X_train)

mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# mlflow xgboost experiments
with mlflow.start_run(experiment_id=experiment_id) as run:
    model = XGBRFClassifier(random_state=42)
    xgboost_model_path = "./artifacts/lead_model_xgboost.pkl"
    params = {
        "learning_rate": uniform(1e-2, 3e-1),
        "min_split_loss": uniform(0, 10),
        "max_depth": randint(3, 10),
        "subsample": uniform(0, 1),
        "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
        "eval_metric": ["aucpr", "error"],
    }
    model_grid = RandomizedSearchCV(
        model, param_distributions=params, n_jobs=-1, verbose=3, n_iter=10, cv=10
    )

    model_grid.fit(X_train, y_train)

    y_pred_train = model_grid.predict(X_train)
    y_pred_test = model_grid.predict(X_test)

    xgboost_model = model_grid.best_estimator_

    # log artifacts
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred_test, average="binary"))
    mlflow.log_artifacts("artifacts", artifact_path="model")
    mlflow.log_param("data_version", "00000")
    mlflow.log_param("model_type", "XGBoost")

    # store model for model interpretability
    joblib.dump(value=xgboost_model, filename=xgboost_model_path)

    # Custom python model for predicting probability
    mlflow.pyfunc.log_model("model", python_model=utils.ProbaModelWrapper(xgboost_model))

# Save lead xgboost model as artifact
xgboost_model_path = "./artifacts/lead_model_xgboost.json"
xgboost_model.save_model(xgboost_model_path)

# Defining model results dict
model_results = {
    xgboost_model_path: classification_report(y_train, y_pred_train, output_dict=True)
}

# mlflow logistic regression experiments
with mlflow.start_run(experiment_id=experiment_id) as run:
    model = LogisticRegression()
    lr_model_path = "./artifacts/lead_model_lr.pkl"

    params = {
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "penalty": ["none", "l1", "l2", "elasticnet"],
        "C": [100, 10, 1.0, 0.1, 0.01],
    }
    model_grid = RandomizedSearchCV(model, param_distributions=params, verbose=3, n_iter=10, cv=3)
    model_grid.fit(X_train, y_train)

    best_model = model_grid.best_estimator_

    y_pred_train = model_grid.predict(X_train)
    y_pred_test = model_grid.predict(X_test)

    # log artifacts
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred_test, average="binary"))
    mlflow.log_artifacts("artifacts", artifact_path="model")
    mlflow.log_param("data_version", "00000")
    mlflow.log_param("model_type", "LogisticRegression")

    # store model for model interpretability
    joblib.dump(value=best_model, filename=lr_model_path)

    # Custom python model for predicting probability
    mlflow.pyfunc.log_model("model", python_model=utils.ProbaModelWrapper(best_model))

# Testing model and storing the columns and model results
model_classification_report = classification_report(y_test, y_pred_test, output_dict=True)

best_model_lr_params = model_grid.best_params_

model_results[lr_model_path] = model_classification_report

column_list_path = "./artifacts/columns_list.json"
with open(column_list_path, "w+") as columns_file:
    columns = {"column_names": list(X_train.columns)}
    json.dump(columns, columns_file)

model_results_path = "./artifacts/model_results.json"
with open(model_results_path, "w+") as results_file:
    json.dump(model_results, results_file)
