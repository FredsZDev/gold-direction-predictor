from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS, create_features


# =========================================================
# 1. CONFIGURATION
# =========================================================

# Path to the raw historical gold dataset
DATA_PATH = Path("data/raw/xauusd.csv")

# Path where the trained model will be saved
MODEL_PATH = Path("models/gold_direction_model.pkl")


# =========================================================
# 2. TRAIN MODEL
# =========================================================

def train_model():
    """
    Load historical gold data, create predictive features,
    create the target variable, split the data chronologically,
    train a Logistic Regression model and evaluate its performance.
    """

    # =====================================================
    # LOAD DATA
    # =====================================================

    data = pd.read_csv(
        DATA_PATH,
        index_col="Date",
        parse_dates=True,
    )

    print(
        f"Loaded {len(data)} rows from {DATA_PATH}"
    )

    # =====================================================
    # CREATE FEATURES
    # =====================================================

    dataset = create_features(data)

    print(
        f"Dataset after feature engineering: "
        f"{len(dataset)} rows"
    )

    # =====================================================
    # CREATE TARGET
    # =====================================================

    # Target definition:
    #
    # 1 = the next day's closing price is higher
    # 0 = the next day's closing price is equal or lower

    dataset["target"] = (
        dataset["Close"].shift(-1)
        > dataset["Close"]
    ).astype(int)

    # The final row does not have a future day
    # to compare against, so we remove it.
    dataset = dataset.dropna(
        subset=["target"]
    )

    # =====================================================
    # DEFINE FEATURES AND TARGET
    # =====================================================

    X = dataset[
        FEATURE_COLUMNS
    ]

    y = dataset[
        "target"
    ]

    print("\n" + "=" * 40)
    print("FEATURES USED BY THE MODEL")
    print("=" * 40)

    print(FEATURE_COLUMNS)

    print("\nTarget distribution:")
    print(y.value_counts())

    # =====================================================
    # TIME-BASED TRAIN / TEST SPLIT
    # =====================================================

    # We use chronological splitting instead of random
    # splitting because financial data is time-dependent.
    #
    # The model trains on the past and tests on the future.

    split_index = int(
        len(dataset) * 0.8
    )

    X_train = X.iloc[
        :split_index
    ]

    X_test = X.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]

    print("\n" + "=" * 40)
    print("TIME-BASED TRAIN/TEST SPLIT")
    print("=" * 40)

    print(
        f"Training samples: "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test)}"
    )

    print(
        f"Training period: "
        f"{X_train.index.min().date()} "
        f"to "
        f"{X_train.index.max().date()}"
    )

    print(
        f"Testing period: "
        f"{X_test.index.min().date()} "
        f"to "
        f"{X_test.index.max().date()}"
    )

    # =====================================================
    # CREATE MODEL
    # =====================================================

    # Logistic Regression was the best-performing model
    # among the models tested in experiment.py.
    #
    # StandardScaler normalizes the feature values before
    # they are passed to Logistic Regression.

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                ),
            ),
        ]
    )

    # =====================================================
    # TRAIN MODEL
    # =====================================================

    print("\nTraining Logistic Regression model...")

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Model training completed."
    )

    # =====================================================
    # MAKE PREDICTIONS
    # =====================================================

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # =====================================================
    # EVALUATE MODEL
    # =====================================================

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    print("\n" + "=" * 40)
    print("MODEL EVALUATION")
    print("=" * 40)

    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{roc_auc:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    # =====================================================
    # BASELINE COMPARISON
    # =====================================================

    # The baseline always predicts the majority class
    # observed in the training dataset.

    majority_class = y_train.mode()[0]

    baseline_predictions = [
        majority_class
    ] * len(y_test)

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions,
    )

    print("\n" + "=" * 40)
    print("BASELINE COMPARISON")
    print("=" * 40)

    print(
        f"Baseline strategy: "
        f"Always predict {majority_class}"
    )

    print(
        f"Baseline accuracy: "
        f"{baseline_accuracy:.4f}"
    )

    print(
        f"Model accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Improvement over baseline: "
        f"{accuracy - baseline_accuracy:.4f}"
    )

    # =====================================================
    # SAVE MODEL
    # =====================================================

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(
        f"\nModel saved to: "
        f"{MODEL_PATH}"
    )


# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":
    train_model()