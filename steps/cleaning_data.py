import logging
import pandas as pd
from zenml import step
from src.clean_data import DataCleaning, DataPreProcessStrategy, DataSplitStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_data_step(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """
    ZenML step to clean the input data by handling missing values.
    
    Args:
        data (pd.DataFrame): Input data as a pandas DataFrame.
    Returns:
        pd.DataFrame: Cleaned data with missing values handled.
    """
    logger = logging.getLogger(__name__)
    try:
        logger.info("Starting data cleaning process")
        
        process_strategy = DataPreProcessStrategy()
        divide_strategy = DataSplitStrategy()

        data_preprocess = DataCleaning(data, process_strategy)
        cleaned_data = data_preprocess.handle_data()

        data_split = DataCleaning(cleaned_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_split.handle_data()
        logger.info("Data cleaning completed successfully")

        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")
        raise e