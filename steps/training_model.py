import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def train_model_step(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame, 
    y_train: pd.Series,
    y_test: pd.Series,
    config: str = "LinearRegression"
) -> RegressorMixin:
    """
    Trains the model on ingested data.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The target values.
    """
    try:
        model = None
        if config == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model {config} is not supported.")
    except Exception as e:
        logging.error(f"An error occurred in train_model_step: {e}")
        raise e