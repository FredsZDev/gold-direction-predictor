from pathlib import Path

import joblib
import pandas as pd

from features import FEATURE_COLUMNS, create_features


# =========================================================
# 1. CONFIGURATION
# =========================================================

# Path to the trained machine learning model
MODEL_PATH = Path("models/gold_direction_model.pkl")

# Path to the raw gold price dataset
DATA_PATH = Path("data/raw/xauusd.csv")


# =========================================================
# 2. LOAD DATA
# =========================================================

def load_data():
    """
    Load the historical gold price data from the CSV file.
    """

    data = pd.read_csv(
        DATA_PATH,
        index_col="Date",
        parse_dates=True,
    )

    return data


# =========================================================
# 3. PREPARE FEATURES
# =========================================================

def prepare_features(data):
    """
    Create the same predictive features used during model training.

    The model must receive the same features during prediction
    that were used during training.
    """

    dataset = create_features(data)

    # Select only the columns expected by the model
    X = dataset[FEATURE_COLUMNS]

    return dataset, X


# =========================================================
# 4. MAKE PREDICTION
# =========================================================

def predict_direction():
    """
    Load the trained model and predict the direction
    of the next gold price movement.
    """

    print("=" * 40)
    print("GOLD DIRECTION PREDICTION")
    print("=" * 40)

    # -----------------------------------------------------
    # Load the trained machine learning model
    # -----------------------------------------------------

    print("\nLoading trained model...")

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully.")

    # -----------------------------------------------------
    # Load historical data
    # -----------------------------------------------------

    print("\nLoading market data...")

    data = load_data()

    print(
        f"Loaded {len(data)} rows from {DATA_PATH}"
    )

    # -----------------------------------------------------
    # Create features
    # -----------------------------------------------------

    dataset, X = prepare_features(data)

    # -----------------------------------------------------
    # Get the latest available market observation
    # -----------------------------------------------------

    latest_date = X.index[-1]

    latest_features = X.iloc[[-1]]

    latest_close = dataset["Close"].iloc[-1]

    # -----------------------------------------------------
    # Generate prediction
    # -----------------------------------------------------

    prediction = model.predict(
        latest_features
    )[0]

    # -----------------------------------------------------
    # Generate prediction probabilities
    # -----------------------------------------------------

    probabilities = model.predict_proba(
        latest_features
    )[0]

    probability_down = probabilities[0]

    probability_up = probabilities[1]

    # -----------------------------------------------------
    # Convert numerical prediction to human-readable text
    # -----------------------------------------------------

    if prediction == 1:
        direction = "UP"
    else:
        direction = "DOWN"

    # =====================================================
    # 5. DISPLAY RESULTS
    # =====================================================

    print("\n" + "=" * 40)
    print("PREDICTION RESULT")
    print("=" * 40)

    print(
        f"\nLatest date: "
        f"{latest_date.date()}"
    )

    print(
        f"Latest close: "
        f"{latest_close:.2f}"
    )

    print(
        f"\nPrediction: "
        f"{direction}"
    )

    print(
        f"Probability of UP: "
        f"{probability_up:.2%}"
    )

    print(
        f"Probability of DOWN: "
        f"{probability_down:.2%}"
    )

    print("\nFeatures used:")

    for feature in FEATURE_COLUMNS:
        print(
            f"{feature}: "
            f"{latest_features[feature].iloc[0]:.6f}"
        )

    print("\n" + "=" * 40)


# =========================================================
# 6. SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":
    predict_direction()