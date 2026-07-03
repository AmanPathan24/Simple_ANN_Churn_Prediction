# Simple ANN Churn Data

A Streamlit application for predicting customer churn with a trained artificial neural network. The project includes the training notebook, a separate prediction notebook, the exported model artifacts, and a polished inference UI for interactive scoring.

### Deployed Link: `https://simple-ann-churn-prediction.streamlit.app/`

## What’s included

- `app.py` - Streamlit app for entering a customer profile and getting a churn probability.
- `dl-3-simple-ann.ipynb` - Training notebook that builds the ANN pipeline, preprocesses the data, trains the model, and saves the fitted artifacts.
- `prediction.ipynb` - Inference notebook that reloads the saved model, encoders, and scaler, then runs a sample prediction.
- `models/model.h5` - Saved ANN model.
- `models/label_encoder_gender.pkl` - Saved gender encoder.
- `models/onehot_encoder_geo.pkl` - Saved geography encoder.
- `models/scaler.pkl` - Saved standard scaler.

## Project flow

1. The training notebook prepares the churn dataset and selects the model inputs.
2. Categorical data is encoded with `LabelEncoder` for gender and `OneHotEncoder` for geography.
3. Numerical features are standardized with `StandardScaler`.
4. A feed-forward ANN is trained with dense layers and binary cross-entropy loss.
5. The trained model and preprocessing artifacts are saved to the `models/` directory.
6. The Streamlit app reloads those artifacts and reproduces the same preprocessing pipeline at inference time.

## Model details

The training notebook uses a compact ANN with two hidden dense layers and a sigmoid output layer for binary churn prediction. The notebook also uses early stopping and TensorBoard logging. Training reached roughly the mid-0.80s in validation accuracy in the saved run.

## Inference inputs

The app expects the same customer features used during training:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

## Run the app

Install the dependencies and start Streamlit from the project root:

```bash
streamlit run app.py
```

## Work with the notebooks

Open `dl-3-simple-ann.ipynb` to review the full training workflow and `prediction.ipynb` to see a notebook-based inference example. The notebooks are useful if you want to retrain the model, inspect preprocessing, or compare notebook predictions with the Streamlit app.

## Notes

- The Streamlit UI is configured for a dark-only presentation.
- Keep the files in `models/` together, since the app depends on all saved artifacts at startup.
- If you retrain the model, regenerate the encoders and scaler so the app and notebook stay in sync.
