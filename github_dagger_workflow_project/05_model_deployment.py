from github_dagger_workflow_project import utils
from github_dagger_workflow_project.mlflow_client import client

model_version = 1
model_name = "lead_model"
model_status = True


model_version_details = dict(client.get_model_version(name=model_name, version=model_version))

if model_version_details["current_stage"] != "Staging":
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Staging",
        archive_existing_versions=True,
    )
    model_status = utils.wait_for_deployment(model_name, model_version, client, "Staging")
