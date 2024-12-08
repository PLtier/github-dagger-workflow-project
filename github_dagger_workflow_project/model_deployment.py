import utils
from mlflow.tracking import MlflowClient

model_version = 1
model_name = "lead_model"

client = MlflowClient()

model_version_details = dict(client.get_model_version(name=model_name,version=model_version))
model_status = True
if model_version_details['current_stage'] != 'Staging':
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,stage="Staging", 
        archive_existing_versions=True
    )
    model_status = utils.wait_for_deployment(model_name, model_version, 'Staging')
else:
    print('Model already in staging')