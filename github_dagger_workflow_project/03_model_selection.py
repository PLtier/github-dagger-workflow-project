from github_dagger_workflow_project import pipeline_utils as pu
from github_dagger_workflow_project.config import (
    EXPERIMENT_NAME,
)


experiment_best = pu.select_best_model(EXPERIMENT_NAME)

# Save best model
# Currently we pick only LR, no matter what the best model is.
# When ready, uncomment to consider xgboost.
# best_model_type = experiment_best["params.model_type"]
# and comment below code
best_model_type = "LogisticRegression"
pu.save_best_model(experiment_best, best_model_type)
