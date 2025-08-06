import pandas as pd
import numpy as np
import os
import sys
import logging


class DataIngestion:
    def __init__(self):
        # Configuring logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            filename='logs/data_ingestion.log', filemode='a')
        self.logger = logging.getLogger(__name__)
        self.file_path = os.path.join("data", "raw", "SkateData.csv")
        self.data = None
        self.logger.info("Data Ingestion has been initialized at %s",
                         os.path.abspath(self.file_path))

    def load_data(self):
        try:
            self.logger.info(f"Attempting to load data from {self.file_path}")
            self.data = pd.read_csv(self.file_path)
            self.logger.info(f"Data loaded successfully. Shape:{self.data.shape}")
        except FileNotFoundError:
            self.logger.error(f"File not found at {self.file_path}. Please check path")
            raise
        except pd.errors.EmptyDataError:
            self.logger.error(f"Empty Data at {self.file_path}. Please check path")
            raise
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while tryng to load data: {str(e)}.")
            raise


if __name__ == "__main__":
    ingestor = DataIngestion()
    ingestor.load_data()
