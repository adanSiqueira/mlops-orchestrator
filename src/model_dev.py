import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression 

class Model(ABC):
    """
    Abstract base class for machine learning models.
    
    This class defines the interface for all models, ensuring they implement
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model on the provided training data.
        
        Parameters:
        X_train: array-like, shape (n_samples, n_features)
            The input data for training.
        y_train: array-like, shape (n_samples,)
            The target values for training.
        """
        pass

class LinearRegressionModel(Model):
    """
    Linear Regression model implementation.
    """

    def train (self, X_train, y_train, **kwargs):
        """
        Train the Linear Regression model using the provided training data.
        
        Parameters:
        X_train: array-like, shape (n_samples, n_features)
            The input data for training.
        y_train: array-like, shape (n_samples,)
            The target values for training.
        kwargs: additional keyword arguments
            Additional parameters for training.
        """
        try:
            logging.info("Training Linear Regression model...")
            logging.info(f"X_train shape: {X_train.shape}")
            logging.info(f"y_train shape: {y_train.shape}")

            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training completed.")

            return reg
        
        except Exception as e:
            logging.error(f"An error occurred during model training: {e}")
            raise

