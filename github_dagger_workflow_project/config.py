import datetime

ARTIFACTS_DIR = "artifacts"
RAW_DATA_PATH = "./artifacts/raw_data.csv"
DATE_LIMITS_PATH = "./artifacts/date_limits.json"
OUTLIER_SUMMARY_PATH = "./artifacts/outlier_summary.csv"
CAT_MISSING_IMPUTE_PATH = "./artifacts/cat_missing_impute.csv"
SCALER_PATH = "./artifacts/scaler.pkl"
COLUMNS_DRIFT_PATH = "./artifacts/columns_drift.json"
TRAINING_DATA_PATH = "./artifacts/training_data.csv"
TRAIN_DATA_GOLD_PATH = "./artifacts/train_data_gold.csv"
COLUMNS_LIST_PATH = "./artifacts/columns_list.json"
XGBOOST_MODEL_PATH = "./artifacts/lead_model_xgboost.pkl"
XGBOOST_MODEL_JSON_PATH = "./artifacts/lead_model_xgboost.json"
LR_MODEL_PATH = "./artifacts/lead_model_lr.pkl"
MODEL_RESULTS_PATH = "./artifacts/model_results.json"
BEST_EXPERIMENT_PATH = "./artifacts/best_experiment.pkl"
BEST_MODEL_PATH = "./artifacts/best_model.pkl"
PROD_BEST_EXPERIMENT_PATH = "./artifacts/best_experiment.pkl"
MAX_DATE_STR = "2024-01-31"
MIN_DATE_STR = "2024-01-01"
CURRENT_DATE = datetime.datetime.now().strftime("%Y_%B_%d")
EXPERIMENT_NAME = CURRENT_DATE
DATA_GOLD_PATH = TRAIN_DATA_GOLD_PATH
ARTIFACT_PATH = "model"
MODEL_NAME = "lead_model"
MODEL_VERSION = 1
