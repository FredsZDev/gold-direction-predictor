from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from features import create_features


def train_model():
    """
    Load historical gold data, create features, split the dataset
    chronologically, train a Random Forest classifier, evaluate it,
    compare it against a simple baseline, and save the trained model.
    """

    # =========================================================
    # 1. LOAD RAW DATA
    # =========================================================

    data_path = Path("data/raw/xauusd.csv")

    data = pd.read_csv(
        data_path,
        index_col="Date",
        parse_dates=True,
    )

    print(f"Loaded {len(data)} rows from {data_path}")

    # =========================================================
    # 2. CREATE FEATURES AND TARGET
    # =========================================================

    dataset = create_features(data)

    print(
        f"Dataset after feature engineering: "
        f"{len(dataset)} rows"
    )

    # =========================================================
    # 3. DEFINE FEATURES
    # =========================================================

    feature_columns = [
        "return_1d",
        "return_5d",
        "price_vs_sma_10",
        "price_vs_sma_30",
        "volatility_10",
        "volume_change",
    ]

    X = dataset[feature_columns]
    y = dataset["target"]

    print("\nFeatures used by the model:")
    print(feature_columns)

    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    # =========================================================
    # 4. TIME-BASED TRAIN / TEST SPLIT
    # =========================================================

    # We use the first 80% of the chronological data
    # for training and the final 20% for testing.
    #
    # This is important for financial time series because
    # we must not train the model using future data.

    split_index = int(len(dataset) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("\n" + "=" * 40)
    print("TIME-BASED TRAIN/TEST SPLIT")
    print("=" * 40)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print(
        f"\nTraining period: "
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

    # =========================================================
    # 5. CREATE RANDOM FOREST MODEL
    # =========================================================

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    # =========================================================
    # 6. TRAIN MODEL
    # =========================================================

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train,
    )

    print("Model training completed.")

    # =========================================================
    # 7. GENERATE PREDICTIONS
    # =========================================================

    predictions = model.predict(X_test)

    # Probability that the next candle will be class 1
    probabilities = model.predict_proba(X_test)[:, 1]

    # =========================================================
    # 8. CHECK PREDICTION DISTRIBUTION
    # =========================================================

    print("\n" + "=" * 40)
    print("PREDICTION DISTRIBUTION")
    print("=" * 40)

    print("\nModel predictions:")
    print(
        pd.Series(predictions)
        .value_counts()
        .sort_index()
    )

    print("\nActual test distribution:")
    print(
        y_test.value_counts()
        .sort_index()
    )

    # =========================================================
    # 9. BASELINE COMPARISON
    # =========================================================

    # A baseline is a very simple strategy that we use
    # as a reference point.
    #
    # Here, we always predict the majority class
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

    # =========================================================
    # 10. MODEL EVALUATION
    # =========================================================

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
        f"Model accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{roc_auc:.4f}"
    )

    print(
        f"Accuracy improvement over baseline: "
        f"{accuracy - baseline_accuracy:.4f}"
    )

    # =========================================================
    # 11. CLASSIFICATION REPORT
    # =========================================================

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    # =========================================================
    # 12. CONFUSION MATRIX
    # =========================================================

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    # =========================================================
    # 13. FEATURE IMPORTANCE
    # =========================================================

    feature_importance = pd.Series(
        model.feature_importances_,
        index=feature_columns,
    ).sort_values(
        ascending=False
    )

    print("\nFeature Importance:")

    print(feature_importance)

    # =========================================================
    # 14. SAVE MODEL
    # =========================================================

    models_path = Path("models")

    models_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_file = (
        models_path
        / "gold_direction_model.pkl"
    )

    joblib.dump(
        model,
        model_file,
    )

    print(
        f"\nModel saved to: "
        f"{model_file}"
    )


if __name__ == "__main__":
    train_model()