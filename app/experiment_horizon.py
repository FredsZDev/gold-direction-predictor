from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import create_features


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load the raw Gold price dataset.
    """

    data_path = Path(
        "data/raw/xauusd.csv"
    )

    data = pd.read_csv(
        data_path,
        index_col="Date",
        parse_dates=True,
    )

    print(
        f"Loaded {len(data)} rows from {data_path}"
    )

    return data


# ============================================================
# CREATE HORIZON TARGET
# ============================================================

def create_horizon_target(
    data,
    horizon,
):
    """
    Create a target based on the future price
    after a specified number of days.

    horizon = 1
        Predict the next trading day.

    horizon = 3
        Predict the price direction 3 trading days ahead.

    horizon = 5
        Predict the price direction 5 trading days ahead.
    """

    df = data.copy()

    # Future closing price
    future_close = df[
        "Close"
    ].shift(-horizon)

    # Target:
    # 1 = future price is higher
    # 0 = future price is lower or equal
    df["target"] = (
        future_close > df["Close"]
    ).astype(int)

    return df


# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_experiment(
    horizon,
    data,
):
    """
    Train and evaluate Logistic Regression
    for a specific prediction horizon.
    """

    print("\n" + "=" * 60)

    print(
        f"PREDICTION HORIZON: "
        f"{horizon} DAY(S)"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # 1. Create target
    # --------------------------------------------------------

    dataset = create_features(
    data
)

    dataset = create_horizon_target(
    dataset,
    horizon,
)

    # --------------------------------------------------------
    # 3. Remove rows with invalid values
    # --------------------------------------------------------

    dataset = dataset.replace(
        [
            float("inf"),
            float("-inf"),
        ],
        pd.NA,
    )

    dataset = dataset.dropna()

    # --------------------------------------------------------
    # 4. Define model features
    # --------------------------------------------------------

    feature_columns = [
        "return_1d",
        "return_5d",
        "price_vs_sma_10",
        "price_vs_sma_30",
        "volatility_10",
        "volume_change",
    ]

    X = dataset[
        feature_columns
    ]

    y = dataset[
        "target"
    ]

    # --------------------------------------------------------
    # 5. Time-based train/test split
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 6. Target distribution
    # --------------------------------------------------------

    print(
        "\nTarget distribution:"
    )

    print(
        y.value_counts()
    )

    # --------------------------------------------------------
    # 7. Create Logistic Regression pipeline
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 8. Train model
    # --------------------------------------------------------

    print(
        "\nTraining model..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Model training completed."
    )

    # --------------------------------------------------------
    # 9. Generate predictions
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # 10. Calculate metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    # --------------------------------------------------------
    # 11. Baseline
    # --------------------------------------------------------

    majority_class = y_train.mode()[0]

    baseline_predictions = [
        majority_class
    ] * len(y_test)

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions,
    )

    # --------------------------------------------------------
    # 12. Print results
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        f"RESULTS - {horizon} DAY HORIZON"
    )

    print(
        "=" * 60
    )

    print(
        f"Baseline Accuracy: "
        f"{baseline_accuracy:.4f}"
    )

    print(
        f"Model Accuracy:    "
        f"{accuracy:.4f}"
    )

    print(
        f"Improvement:       "
        f"{accuracy - baseline_accuracy:+.4f}"
    )

    print(
        f"ROC-AUC:           "
        f"{roc_auc:.4f}"
    )

    print(
        f"Precision:         "
        f"{precision:.4f}"
    )

    print(
        f"Recall:            "
        f"{recall:.4f}"
    )

    print(
        f"F1-score:          "
        f"{f1:.4f}"
    )

    return {
        "horizon": horizon,
        "baseline_accuracy": baseline_accuracy,
        "accuracy": accuracy,
        "improvement": (
            accuracy - baseline_accuracy
        ),
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Load raw dataset
    data = load_data()

    # Horizons we want to test
    horizons = [
        1,
        3,
        5,
    ]

    results = []

    # Run experiment for each horizon
    for horizon in horizons:

        result = run_experiment(
            horizon,
            data,
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "HORIZON COMPARISON"
    )

    print(
        "=" * 60
    )

    print(
        results_df.to_string(
            index=False
        )
    )