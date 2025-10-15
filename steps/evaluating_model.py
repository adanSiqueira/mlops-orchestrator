import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from src.evaluate_model import Evaluation, MSE, R2Score, RMSE
from typing import Tuple
from typing_extensions import Annotated
from zenml.client import Client
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model_step(model: RegressorMixin,
                        X_test: pd.DataFrame,
                        y_test: pd.DataFrame) -> Tuple[
    Annotated[float, "R² Score"],
    Annotated[float, "Root Mean Squared Error (RMSE)"],
    Annotated[float, "Mean Squared Error (MSE)"]
    ]:
    """
    Evaluates the trained model on test data.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to evaluate.
    """
    try:
        
        logging.info("Evaluating the model...")

        predictions = model.predict(X_test)

        mse_evaluator = MSE()
        mse = mse_evaluator.calculate_scores(y_test, predictions)
        mlflow.log_metric("mse", mse)

        r2_evaluator = R2Score()
        r2 = r2_evaluator.calculate_scores(y_test, predictions)
        mlflow.log_metric("r2", r2)
        
        rmse_evaluator = RMSE()
        rmse = rmse_evaluator.calculate_scores(y_test, predictions)
        mlflow.log_metric("rmse", rmse)

        logging.info(f"Mean Squared Error (MSE): {mse}")
        logging.info(f"R² Score: {r2}")
        logging.info(f"Root Mean Squared Error (RMSE): {rmse}")
        logging.info("Model evaluation completed.")

        return r2, rmse, mse
    
    except Exception as e:
        logging.error(f"An error occurred in evaluate_model_step: {e}")
        raise e