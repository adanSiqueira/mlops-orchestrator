import logging
from abc import ABC, abstractmethod
from zenml import step
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract class defyning strategy for model evaluation.
    
    This class defines the interface for all evaluation strategies, ensuring they implement
    the evaluate method.
    """
    @abstractmethod
    def calculate_scores(self, y_test: np.array, y_pred: np.array) -> dict:
        """
        Evaluate the model on the provided test data.
        
        Parameters:
        model: Trained machine learning model
            The model to be evaluated.
        X_test: array-like, shape (n_samples, n_features)
            The input data for testing.
        y_test: array-like, shape (n_samples,)
            The target values for testing.
        """
        pass

class MSE(Evaluation):
    """
    Mean Squared Error evaluation strategy.
    """
    def calculate_scores(self, y_test: np.array, y_pred: np.array):
        """       
        Returns:
        dict: A dictionary containing the MSE score.
        """
        try:
            logging.info("Calculating Mean Squared Error (MSE)...")
            mse = mean_squared_error(y_test, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"An error occurred during MSE calculation: {e}")
            raise e
        
class R2Score(Evaluation):
    """
    R² (R-squared) evaluation strategy.
    """
    def calculate_scores(self, y_test: np.array, y_pred: np.array):
        """       
        Returns:
        dict: A dictionary containing the R² score.
        """
        try:
            logging.info("Calculating R² Score...")
            r2 = r2_score(y_test, y_pred)
            logging.info(f"R² Score: {r2}")
            return r2
        except Exception as e:
            logging.error(f"An error occurred during R² calculation: {e}")
            raise e
        
class RMSE(Evaluation):
    """
    Root Mean Squared Error evaluation strategy.
    """
    def calculate_scores(self, y_test: np.array, y_pred: np.array):
        """       
        Returns:
        dict: A dictionary containing the RMSE score.
        """
        try:
            logging.info("Calculating Root Mean Squared Error (RMSE)...")
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            logging.info(f"RMSE: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"An error occurred during RMSE calculation: {e}")
            raise e