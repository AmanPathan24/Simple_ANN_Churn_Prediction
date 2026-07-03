# Simple ANN Churn & Salary Analytics

A Streamlit application for predicting customer churn and estimated salary with trained artificial neural networks. The project includes training notebooks, a prediction notebook, exported model artifacts, and a polished inference UI for interactive classification and regression scoring.

### Deployed Link: `https://simple-ann-churn-prediction.streamlit.app/`

## What’s included

### Core Application & Inference
- `app.py` - Streamlit app featuring a sidebar selector to toggle between:
  - **Classification**: Predicts customer churn risk (using `EstimatedSalary` as an input feature).
  - **Regression**: Predicts estimated salary (using `Exited` status as an input feature).
- `prediction.ipynb` - Inference notebook demonstrating classification model loading and prediction.

### Classification Artifacts
- `dl-3-simple-ann.ipynb` - Notebook to preprocess and train the customer churn classification model.
- `models/` - Classification pipeline artifacts:
  - `model.h5` - Saved classification ANN model.
  - `label_encoder_gender.pkl` - Saved gender encoder.
  - `onehot_encoder_geo.pkl` - Saved geography encoder.
  - `scaler.pkl` - Saved standard scaler.

### Regression Artifacts
- `dl-4-simple-ann-regression.ipynb` - Notebook to preprocess and train the estimated salary regression model.
- `models/regression/` - Regression pipeline artifacts:
  - `model.h5` - Saved regression ANN model.
  - `label_encoder_gender.pkl` - Saved gender encoder.
  - `onehot_encoder_geo.pkl` - Saved geography encoder.
  - `scaler.pkl` - Saved standard scaler.

## Project flow

1. The training notebooks prepare the churn dataset and partition model inputs.
   - For **Classification**, `EstimatedSalary` is kept as a feature to predict customer churn (`Exited`).
   - For **Regression**, `Exited` (churn status) is kept as a feature to predict the customer's `EstimatedSalary`.
2. Categorical data is encoded with `LabelEncoder` for gender and `OneHotEncoder` for geography.
3. Numerical features are standardized with `StandardScaler`.
4. Feed-forward ANN models are trained with dense layers:
   - Classification uses binary cross-entropy loss and sigmoid output.
   - Regression uses mean absolute error (MAE) loss and linear output activation.
5. Streamlit loads selected artifacts dynamically on demand, performs matching pre-processing, and generates predictions.

## Run the app

Install the dependencies and start Streamlit from the project root:

```bash
streamlit run app.py
```

## Work with the notebooks

- Open `dl-3-simple-ann.ipynb` to review the classification training workflow.
- Open `dl-4-simple-ann-regression.ipynb` to review the regression training workflow.
- Open `prediction.ipynb` to see a classification inference notebook-based example.

## Notes

- The Streamlit UI is configured for a dark-only presentation.
- Keep the files in `models/` and `models/regression/` together, since the app depends on all saved artifacts at startup.
- If you retrain any model, regenerate the encoders and scaler so the app and notebook stay in sync.

