import logging
import pandas as pd
from zenml import step

class IngestData():
    """Class to handle data ingestion."""

    def __init__(self, data_path: str):
        """
        Args:
            data_path: Path to the CSV file containing the data.
        """
        self.file_path = data_path
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> pd.DataFrame:
        """Load data from a CSV file into a pandas DataFrame."""
        try:
            self.logger.info(f"Loading data from {self.file_path}")
            data = pd.read_csv(self.file_path)
            self.logger.info("Data loaded successfully")
            return data
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

@step
def ingest_data_step(data_path: str) -> pd.DataFrame:
    """
    ZenML step to ingest data from the data_path.
    
    Args:
        data_path (str): Path to the CSV file containing the data.
    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.q
    """
    ingestor = IngestData(data_path)
    df = ingestor.load_data()
    return df