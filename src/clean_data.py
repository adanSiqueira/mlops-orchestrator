import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """ Abstract base class for data cleaning strategies.
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    """ Concrete strategy for preprocessing data.
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data
        """
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ], axis=1
            )
            data["product_weight_g"] = data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"] = data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"] = data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"] = data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            data["review_comment_message"] = data["review_comment_message"].fillna("No comment", inplace=True)

            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ['customer_zip_code_prefix', 'order_item_id']
            data = data.drop(columns=cols_to_drop, axis=1)
            return data
        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")
            raise e
        
class DataSplitStrategy(DataStrategy):
    """ Concrete strategy for splitting data.
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Split the data into training and testing sets
        """
        try:
            X = data.drop("review_score", axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in splitting data: {e}")
            raise e

class DataCleaning:
    """
    Class for cleanin data which processes the data and divides it into train and test
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data using the specified strategy
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e

if __name__ == "__main__":
    data = pd.read_csv("data/olist_order_reviews_dataset.csv")
    data_cleaning = DataCleaning(data, DataPreProcessStrategy())
    data_cleaning.handle_data()