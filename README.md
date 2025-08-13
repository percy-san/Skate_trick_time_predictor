## Skate Trick Learning Time Predictor

Predict how long it will take a skater to land a new trick using a machine learning model and a simple Gradio UI.

### What this project does
- **Preprocesses** a raw skate dataset into model-ready features and saves a reusable **preprocessor**.
- **Trains and evaluates** several regressors (Linear Regression, Decision Tree, Random Forest, SVR, XGBoost, CatBoost) and saves the **best model**.
- **Serves** an interactive **Gradio app** to make predictions from user inputs.

## Project structure
```
Skateboarding Trick time/
  data/
    raw/             # Put SkateData.csv here
    processed/       # Generated: X_train.csv, X_test.csv, y_train.csv, y_test.csv
  models/            # Generated: preprocessor.pkl, Best_model_<name>.pkl
  logs/              # Log files written by each stage
  src/
    app.py           # Gradio UI
    data_ingestion.py
    data_preprocessing.py
    model.py         # Training and model selection
    predict.py       # ModelPredictor (used by app)
    Utils.py
```

## Requirements
- Python 3.9+ recommended
- See `requirements.txt`

Install dependencies (Windows PowerShell):
```bash
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data expectations
Place your dataset at `data/raw/SkateData.csv` with at least these columns:
```csv
age,practice_hours_per_week,confidence_level,motivation_level,gender,experience_level,favorite_trick,skateboard_type,learning_method,previous_injuries,time_to_land_trick
```
- **Target**: `time_to_land_trick` (numeric)
- **Numerical features**: `age`, `practice_hours_per_week`, `confidence_level`, `motivation_level`
- **Categorical features**: `gender`, `experience_level`, `favorite_trick`, `skateboard_type`, `learning_method`, `previous_injuries`

Notes:
- Missing values are imputed (median for numeric, most frequent for categorical).
- Categorical variables are one-hot encoded with `handle_unknown='ignore'`.

## End-to-end usage

1) Preprocess data (creates `data/processed/*` and `models/preprocessor.pkl`):
```bash
python src/data_preprocessing.py
```

2) Train models and save the best one to `models/`:
```bash
python src/model.py
```
This evaluates multiple models and saves the best as `Best_model_<ModelName>.pkl` (for example, `Best_model_catboost.pkl`).

3) Launch the Gradio app:
```bash
python src/app.py
```
You’ll get a local URL and a share URL (because `share=True`).

## Using the app
- Fill in the input fields (age, practice hours, confidence, motivation, etc.).
- Dropdown options are auto-populated from `data/raw/SkateData.csv`.
- Click **Predict** to see the predicted time to land the trick.

## How it works (brief)
- `src/data_preprocessing.py`: builds a `ColumnTransformer` pipeline, splits data, fits the preprocessor on train, transforms train/test, saves processed CSVs and `models/preprocessor.pkl`.
- `src/model.py`: loads processed data, trains several regressors, tracks R², and saves the best model.
- `src/predict.py`: loads `preprocessor.pkl` and the best model file; exposes a `ModelPredictor.predict(...)` method.
- `src/app.py`: Gradio UI consuming `ModelPredictor`.

## Re-training with new data
1) Replace/update `data/raw/SkateData.csv`.
2) Re-run preprocessing and training:
```bash
python src/data_preprocessing.py
python src/model.py
```
3) Restart the app:
```bash
python src/app.py
```

## Troubleshooting
- **File not found: models/preprocessor.pkl**
  - Run preprocessing first: `python src/data_preprocessing.py`.
- **No best model file found in 'models/' directory**
  - Run training: `python src/model.py`.
- **Dropdowns are empty or weird**
  - Ensure `data/raw/SkateData.csv` has the categorical columns with valid values.
- **Shapes or dtype errors during training**
  - Verify `time_to_land_trick` exists and is numeric; remove unexpected columns.
- **Permission issues writing logs or models**
  - Ensure the `logs/`, `models/`, and `data/processed/` folders are writable.

## Logging
Logs are written under `logs/` by each stage (`data_ingestion.log`, `preprocessing.log`, `training.log`, `prediction.log`, `app.log`).

## License
Add your preferred license file (e.g., MIT) at the project root.


