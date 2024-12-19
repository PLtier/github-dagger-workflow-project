import os

from github_dagger_workflow_project import pipeline_utils as pu
from github_dagger_workflow_project.config import (
    ARTIFACTS_DIR,
    RAW_DATA_PATH,
    DATE_LIMITS_PATH,
    MAX_DATE_STR,
    MIN_DATE_STR,
)


# Define min and max date
min_date, max_date = pu.initialize_dates(MAX_DATE_STR, MIN_DATE_STR)

# Create artifacts folder
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

data = pu.load_data(RAW_DATA_PATH)
data = pu.filter_date_range(data, min_date, max_date)
pu.save_date_limits(data, DATE_LIMITS_PATH)
data = pu.preprocess_data(data)
pu.process_and_save_artifacts(data)
