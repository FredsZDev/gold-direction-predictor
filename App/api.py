from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException

from app.features import FEATURE_COLUMNS, create_features


# =========================================================
# APP CONFIGURATION
# =========================================================

app = FastAPI(
    title="Gold Direction Predictor API",
    description=(
        "API for predicting the next-day direction of Gold "
        "using a machine learning classification model."
    ),
    version="1.0.0",
)


# =========================================================
# PATHS
# =========================================================

MODEL_PATH = Path("models/gold_direction_model.pkl")
DATA_PATH = Path("data/raw/xauusd.csv")


# =========================================================
# LOAD MODEL
# =========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():
    """
    Check if the API is running correctly.
    """

    return {
        "status": "ok",
        "model_loaded": True,
    }


# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.get("/predict")
def predict_gold_direction():
    """
    Generate a prediction for the next trading day's
    Gold price direction.
    """

    if not DATA_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Market data file not found.",
        )

    # Load historical market data
    data = pd.read_csv(
        DATA_PATH,
        index_col="Date",
        parse_dates=True,
    )

    # Create predictive features
    dataset = create_features(data)

    if dataset.empty:
        raise HTTPException(
            status_code=500,
            detail="Could not create features from market data.",
        )

    # Get the latest available row
    latest_data = dataset.iloc[[-1]]

    # Select only the features used by the model
    X_latest = latest_data[FEATURE_COLUMNS]

    # Generate prediction
    prediction = model.predict(X_latest)[0]

    # Generate prediction probabilities
    probabilities = model.predict_proba(X_latest)[0]

    probability_down = float(probabilities[0])
    probability_up = float(probabilities[1])

    # Convert numerical prediction to human-readable label
    direction = "UP" if prediction == 1 else "DOWN"

    return {
        "date": latest_data.index[0].strftime("%Y-%m-%d"),
        "close": float(latest_data["Close"].iloc[0]),
        "prediction": direction,
        "probability_up": round(
            probability_up,
            4,
        ),
        "probability_down": round(
            probability_down,
            4,
        ),
    }