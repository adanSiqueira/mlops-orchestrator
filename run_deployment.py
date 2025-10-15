import click
from typing import cast
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from pipelines.deployment_pipeline import continuous_deployment_pipeline
from zenml.client import Client

client = Client()
DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"

@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Choose to deploy the model, make predictions, or both. Default is both")

@click.option(
    "--min-accuracy",
    default=0.92,
    help="Minimum accuracy threshold for deployment.")

def run_deployment(config: str, min_accuracy: float = 0.92):
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT
    if deploy:
        continuous_deployment_pipeline(
            min_accuracy=min_accuracy,
            workers=3,
            timeout=60)
    if predict:
        # inference_pipeline()
        pass

    existing_services = []
    for s in client.list_services():
        labels = s.labels or {}
        if (
            labels.get("pipeline_name") == "continuous_deployment_pipeline"
            and labels.get("pipeline_step_name") == "mlflow_model_deployer_step"
            and labels.get("model_name") == "model"
        ):
            existing_services.append(s)

if __name__ == "__main__":
    run_deployment()