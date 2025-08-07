import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from Utils import logging_config
from data_ingestion import DataIngestion
import joblib
import os


class DataPreprocessor:
    def __init__(self, data,
                 output_dir='data/processed'):
        # Configuring logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            filename='logs/preprocessing.log', filemode='a')
        self.logger = logging.getLogger(__name__)

        self.file_path = os.path.join("data", "raw", "SkateData.csv")
        self.data = data
        self.logger.info("Data Ingestion has been initialized at %s",
                         os.path.abspath(self.file_path))
        self.output_dir = output_dir
        self.target = 'time_to_land_trick'

        # Define column types
        self.numerical_cols = ['age', 'practice_hours_per_week', 'confidence_level', 'motivation_level']
        self.categorical_cols = ['gender', 'experience_level', 'favorite_trick',
                                 'skateboard_type', 'learning_method', 'previous_injuries']

        # Create transformer
        self.preprocessor = ColumnTransformer(transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), self.numerical_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), self.categorical_cols)
        ])
        self.logger.info("Data processor has been initialized at %s",
                         os.path.abspath(self.file_path))

    def preprocess(self):
        logging.info("Attempting to load data")
        df = self.data
        X = df.drop(columns=[self.target])
        y = df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Fitting and transforming training data...")
        X_train_processed = self.preprocessor.fit_transform(X_train)
        print("Transforming test data...")
        X_test_processed = self.preprocessor.transform(X_test)

        self.save_data(X_train_processed, X_test_processed, y_train, y_test)

    def save_data(self, X_train, X_test, y_train, y_test):
        logging.info("Saving data")
        os.makedirs(self.output_dir, exist_ok=True)
        pd.DataFrame(X_train).to_csv(os.path.join(self.output_dir, 'X_train.csv'), index=False)
        pd.DataFrame(X_test).to_csv(os.path.join(self.output_dir, 'X_test.csv'), index=False)
        pd.DataFrame(y_train).to_csv(os.path.join(self.output_dir, 'y_train.csv'), index=False)
        pd.DataFrame(y_test).to_csv(os.path.join(self.output_dir, 'y_test.csv'), index=False)
        joblib.dump(self.preprocessor, 'models/preprocessor.pkl')
        print("Preprocessing complete. Files saved to 'data/processed/'")
        logging.info(f"Data has been saved in: {self.output_dir}")

    def run(self):
        self.preprocess()


if __name__ == "__main__":
    ingestor = DataIngestion()
    ingestor.load_data()
    preprocessor = DataPreprocessor(data=ingestor.data)
    preprocessor.run()
