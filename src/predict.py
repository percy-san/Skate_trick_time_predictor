import joblib
import pandas as pd
import os
import logging
from data_ingestion import DataIngestion

# Configuring logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='logs/prediction.log', filemode='a')
logger = logging.getLogger(__name__)


class ModelPredictor:
    def __init__(self):
        self.preprocessor = self.load_preprocessor()
        self.model = self.load_model()
        self.gender_options = None
        self.experience_level_options = None
        self.favorite_trick_options = None
        self.skateboard_type_options = None
        self.learning_method_options = None
        self.previous_injuries_options = None
        self.load_dropdown_options()
        logger.info("ModelPredictor initialized")

    def load_preprocessor(self):
        preprocessor_path = 'models/preprocessor.pkl'
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            logger.info(f"Preprocessor loaded from {preprocessor_path}")
            return preprocessor
        else:
            logger.error(f"Preprocessor not found at {preprocessor_path}")
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")

    def load_model(self):
        model_dir = 'models'
        model_files = [f for f in os.listdir(model_dir) if f.startswith('Best_model_') and f.endswith('.pkl')]
        if model_files:
            model_path = os.path.join(model_dir, model_files[0])
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return model
        else:
            logger.error("No best model file found in 'models/' directory")
            raise FileNotFoundError("No best model file found in 'models/' directory")

    def load_dropdown_options(self):
        ingestor = DataIngestion()
        ingestor.load_data()
        data = ingestor.data
        self.gender_options = sorted(data['gender'].unique().tolist())
        self.experience_level_options = sorted(data['experience_level'].unique().tolist())
        self.favorite_trick_options = sorted(data['favorite_trick'].unique().tolist())
        self.skateboard_type_options = sorted(data['skateboard_type'].unique().tolist())
        self.learning_method_options = sorted(data['learning_method'].unique().tolist())
        self.previous_injuries_options = sorted(data['previous_injuries'].unique().tolist())
        logger.info("Dropdown options loaded from data")

    def predict(self, age, practice_hours_per_week, confidence_level, motivation_level,
                gender, experience_level, favorite_trick, skateboard_type,
                learning_method, previous_injuries):
        try:
            input_data = pd.DataFrame({
                'age': [age],
                'practice_hours_per_week': [practice_hours_per_week],
                'confidence_level': [confidence_level],
                'motivation_level': [motivation_level],
                'gender': [gender],
                'experience_level': [experience_level],
                'favorite_trick': [favorite_trick],
                'skateboard_type': [skateboard_type],
                'learning_method': [learning_method],
                'previous_injuries': [previous_injuries]
            })

            processed_data = self.preprocessor.transform(input_data)
            prediction = self.model.predict(processed_data)[0]
            logger.info("Prediction made successfully")
            return f"Predicted time to land the trick: {prediction:.2f}"
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return f"Error: {str(e)}"


if __name__ == "__main__":
    predictor = ModelPredictor()
