import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

import joblib
import os
import logging


class ModelTrainer:
    """ Class to train and evaluate the machine learning model"""

    def __init__(self, data_dir='data\\processed', model_dir='models'):
        # Configuring logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            filename='logs/training.log', filemode='a')
        self.logger = logging.getLogger(__name__)

        self.data_dir = data_dir
        self.model_dir = model_dir
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'linear regression': LinearRegression(),
            'DT': DecisionTreeRegressor(random_state=42),
            'svr': SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'),
            'xgboost': XGBRegressor(objective='reg:squarederror', random_state=42),
            'catboost': CatBoostRegressor(random_state=42, verbose=0)
        }
        self.logger.info('Models initialized')

        self.best_model = None
        self.best_r2 = -float('inf')
        self.best_model_name = " "

    def load_data(self):
        """ Loading the preprocessed data"""
        self.logger.info(f'Working in directory :{os.getcwd()}')
        self.logger.info(
            f'Attempting to load X_train data from: {os.path.abspath(os.path.join(self.data_dir, 'X_train.csv'))}')
        try:
            self.X_train = pd.read_csv(os.path.join(self.data_dir, 'X_train.csv'))
            self.X_test = pd.read_csv(os.path.join(self.data_dir, 'X_test.csv'))
            self.y_train = pd.read_csv(os.path.join(self.data_dir, 'y_train.csv'))
            self.y_test = pd.read_csv(os.path.join(self.data_dir, 'y_test.csv'))
            self.logger.info('Loaded train data successfully')
        except Exception as e:
            print(f"Error: Preprocessed data not found in {self.data_dir}. Details:{e}")
            self.logger.warning(f'Preprocessed data not found in {self.data_dir}. Details:{e}')
            exit(1)

    def train_evaluate(self):
        """ Training and evaluating the model"""
        self.logger.info(f'Training the models')
        print("Training the models....")
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            self.logger.info(f'Model trained successfully')
            y_pred = model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            print(f"\n{name} Results: ")
            print(f"MSE: {mse} ")
            print(f"R2: {r2}")
            if r2 > self.best_r2:
                self.best_r2 = r2
                self.best_model = model
                self.best_model_name = name

        print("\nBest Model: ", self.best_model_name)
        print("Best R2: ", self.best_r2)
        self.logger.info(f'Best Model is selected as: {self.best_model_name}, with R2 score of: {self.best_r2}')

        # Saving the best model

    def save_model(self):
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, f"Best_model_{self.best_model_name.replace(' ', '_')}.pkl")
        joblib.dump(self.best_model, model_path)
        print(f"Best model saved to {model_path}")
        self.logger.info(f'Best model saved to {model_path}')

    def run(self):
        """Run the full training pipeline."""
        self.load_data()
        self.train_evaluate()
        self.save_model()


if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.run()
