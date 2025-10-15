import numpy as np
import pandas as pd
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer

from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.mlflow.services import MLFlowDeploymentService

from steps.ingesting_data import ingest_data_step
from steps.cleaning_data import clean_data_step
from steps.training_model import train_model_step
from steps.evaluating_model import evaluate_model_step

docker_settings = DockerSettings(required_integrations=[MLFLOW])

# class DeploymentTriggerConfig():
#     """Configuration for the deployment trigger step."""
#     min_accuracy: float = 0.92

@step
def deployment_trigger(
    accuracy: float,
    # config: DeploymentTriggerConfig
):
    """A step that triggers deployment based on accuracy."""
    # return accuracy >= config.min_accuracy
    return accuracy >= 0.92

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(min_accuracy: float = 0.92, workers: int = 1,timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT):
              
        """A continuous deployment pipeline that deploys a model if it meets the accuracy threshold."""

        data_path = "data\olist_customers_dataset.csv"
        raw_data = ingest_data_step(data_path)
        X_train, X_test, y_train, y_test = clean_data_step(data=raw_data)
        model = train_model_step(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        R2, RMSE, MSE = evaluate_model_step(model=model, X_test=X_test, y_test=y_test)
        deployment_decision = deployment_trigger(accuracy=R2)

        mlflow_model_deployer_step(
            model=model,
            # deployment_decision=deployment_decision,
            workers=workers,
            timeout=timeout)