import datetime
import joblib
import json
import numpy as np
import pandas as pd

import joblib
import mlflow
import shutil
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import uniform, randint
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRFClassifier

from github_dagger_workflow_project import utils
from github_dagger_workflow_project.mlflow_client import client
from github_dagger_workflow_project.config import (
    OUTLIER_SUMMARY_PATH,
    CAT_MISSING_IMPUTE_PATH,
    SCALER_PATH,
    COLUMNS_DRIFT_PATH,
    TRAINING_DATA_PATH,
    TRAIN_DATA_GOLD_PATH,
    COLUMNS_LIST_PATH,
    XGBOOST_MODEL_PATH,
    XGBOOST_MODEL_JSON_PATH,
    LR_MODEL_PATH,
    MODEL_RESULTS_PATH,
    BEST_EXPERIMENT_PATH,
    BEST_MODEL_PATH,
)


def initialize_dates(max_date_str, min_date_str):
    """
    Initialize min and max dates for filtering the data.
    """
    if not max_date_str:
        max_date = pd.to_datetime(datetime.datetime.now().date()).date()
    else:
        max_date = pd.to_datetime(max_date_str).date()

    min_date = pd.to_datetime(min_date_str).date()
    return min_date, max_date


def load_data(file_path):
    return pd.read_csv(file_path)


def filter_date_range(data, min_date, max_date):
    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    return data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]


def save_date_limits(data, file_path):
    min_date = data["date_part"].min()
    max_date = data["date_part"].max()
    date_limits = {"min_date": str(min_date), "max_date": str(max_date)}
    with open(file_path, "w") as f:
        json.dump(date_limits, f)


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses data by:
    Drops unnecessary columns.
    Replaces empty strings with NaN in specific columns.
    Removes rows with missing values in critical columns.
    Filters rows based on the 'source' column being 'signup'.
    """
    columns_to_drop = [
        "is_active",
        "marketing_consent",
        "first_booking",
        "existing_customer",
        "last_seen",
        "domain",
        "country",
        "visited_learn_more_before_booking",
        "visited_faq",
    ]
    data = data.drop(columns=columns_to_drop, axis=1)

    columns_to_clean = ["lead_indicator", "lead_id", "customer_code"]
    data[columns_to_clean] = data[columns_to_clean].replace("", np.nan)

    data = data.dropna(axis=0, subset=["lead_indicator", "lead_id"])
    data = data[data.source == "signup"]

    return data


def process_and_save_artifacts(data):
    vars = [
        "lead_id",
        "lead_indicator",
        "customer_group",
        "onboarding",
        "source",
        "customer_code",
    ]
    for col in vars:
        data[col] = data[col].astype("object")

    cont_vars = data.loc[:, ((data.dtypes == "float64") | (data.dtypes == "int64"))]
    cat_vars = data.loc[:, (data.dtypes == "object")]

    cont_vars = cont_vars.apply(
        lambda x: x.clip(lower=(x.mean() - 2 * x.std()), upper=(x.mean() + 2 * x.std()))
    )

    outlier_summary = cont_vars.apply(utils.describe_numeric_col).T
    outlier_summary.to_csv(OUTLIER_SUMMARY_PATH)

    cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
    cat_missing_impute.to_csv(CAT_MISSING_IMPUTE_PATH)

    cont_vars = cont_vars.apply(utils.impute_missing_values)
    cont_vars.apply(utils.describe_numeric_col).T

    cat_vars.loc[cat_vars["customer_code"].isna(), "customer_code"] = "None"
    cat_vars = cat_vars.apply(utils.impute_missing_values)
    cat_vars.apply(
        lambda x: pd.Series([x.count(), x.isnull().sum()], index=["Count", "Missing"])
    ).T

    scaler = MinMaxScaler()
    scaler.fit(cont_vars)
    joblib.dump(value=scaler, filename=SCALER_PATH)

    cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)
    cont_vars = cont_vars.reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)

    data = pd.concat([cat_vars, cont_vars], axis=1)

    data_columns = list(data.columns)
    with open(COLUMNS_DRIFT_PATH, "w+") as f:
        json.dump(data_columns, f)

    data.to_csv(TRAINING_DATA_PATH, index=False)

    data["bin_source"] = data["source"]
    values_list = ["li", "organic", "signup", "fb"]
    data.loc[~data["source"].isin(values_list), "bin_source"] = "Others"
    mapping = {"li": "socials", "fb": "socials", "organic": "group1", "signup": "group1"}
    data["bin_source"] = data["source"].map(mapping)

    data.to_csv(TRAIN_DATA_GOLD_PATH, index=False)


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
    with open(COLUMNS_LIST_PATH, "w+") as columns_file:
        columns = {"column_names": list(X_train.columns)}
        json.dump(columns, columns_file)


def train_xgboost(X_train, X_test, y_train, y_test, experiment_id):
    with mlflow.start_run(experiment_id=experiment_id):
        model = XGBRFClassifier(random_state=42)
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

        best_model_xgboost = model_grid.best_estimator_

        y_pred_test = model_grid.predict(X_test)

        # log artifacts
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred_test, average="binary"))
        mlflow.log_artifacts("artifacts", artifact_path="model")
        mlflow.log_param("data_version", "00000")
        mlflow.log_param("model_type", "XGBoost")
        # Custom python model for predicting probability
        mlflow.pyfunc.log_model("model", python_model=utils.ProbaModelWrapper(best_model_xgboost))

    xgboost_model_path = XGBOOST_MODEL_PATH
    joblib.dump(value=best_model_xgboost, filename=xgboost_model_path)
    # Save lead xgboost model as artifact
    xgboost_model_path = XGBOOST_MODEL_JSON_PATH
    best_model_xgboost.save_model(xgboost_model_path)

    # Defining model results dict
    xgboost_cr = {xgboost_model_path: classification_report(y_test, y_pred_test, output_dict=True)}

    return xgboost_cr


# mlflow logistic regression experiments
def train_linear_regression(X_train, X_test, y_train, y_test, experiment_id):
    with mlflow.start_run(experiment_id=experiment_id):
        model = LogisticRegression()

        params = {
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            "penalty": ["none", "l1", "l2", "elasticnet"],
            "C": [100, 10, 1.0, 0.1, 0.01],
        }
        model_grid = RandomizedSearchCV(
            model, param_distributions=params, verbose=3, n_iter=10, cv=3
        )
        model_grid.fit(X_train, y_train)

        best_lr_model = model_grid.best_estimator_

        y_pred_test = model_grid.predict(X_test)

        # log artifacts
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred_test, average="binary"))
        mlflow.log_artifacts("artifacts", artifact_path="model")
        mlflow.log_param("data_version", "00000")
        mlflow.log_param("model_type", "LogisticRegression")

        # Custom python model for predicting probability
        mlflow.pyfunc.log_model("model", python_model=utils.ProbaModelWrapper(best_lr_model))

    # store model for model interpretability
    lr_model_path = LR_MODEL_PATH
    joblib.dump(value=best_lr_model, filename=lr_model_path)

    # Testing model and storing the columns and model results
    lr_cr = {lr_model_path: classification_report(y_test, y_pred_test, output_dict=True)}

    return lr_cr


def save_model_results(model_results):
    with open(MODEL_RESULTS_PATH, "w+") as results_file:
        json.dump(model_results, results_file)


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
