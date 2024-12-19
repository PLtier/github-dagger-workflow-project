from github_dagger_workflow_project import utils
from github_dagger_workflow_project.mlflow_client import client
from github_dagger_workflow_project.config import (
    MODEL_NAME,
    MODEL_VERSION,
)


model_status = True

model_version_details = dict(client.get_model_version(name=MODEL_NAME, version=MODEL_VERSION))

if model_version_details["current_stage"] != "Staging":
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=MODEL_VERSION,
        stage="Staging",
        archive_existing_versions=True,
    )
    model_status = utils.wait_for_deployment(MODEL_NAME, MODEL_VERSION, client, "Staging")
