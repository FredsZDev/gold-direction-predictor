from pathlib import Path

import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from features import FEATURE_COLUMNS, create_features


def load_dataset():
    """
    Load the raw gold dataset, create predictive features,
    and create the target variable.

    The target represents the direction of the next trading day:
    1 = next day's closing price is higher
    0 = next day's closing price is equal or lower
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

    # =========================================================
    # 2. CREATE FEATURES
    # =========================================================

    dataset = create_features(data)

    # =========================================================
    # 3. CREATE TARGET
    # =========================================================

    # Compare today's closing price with tomorrow's closing price.
    #
    # shift(-1) moves the next day's Close value
    # to the current row.
    #
    # Example:
    #
    # Today's Close = 4000
    # Tomorrow's Close = 4050
    #
    # 4050 > 4000
    # target = 1

    dataset["target"] = (
        dataset["Close"].shift(-1)
        > dataset["Close"]
    ).astype(int)

    # The final row does not have a future closing price,
    # so its target cannot be calculated.
    dataset = dataset.dropna(
        subset=["target"]
    )

    return dataset


def evaluate_model(
    model,
    model_name,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Train and evaluate a machine learning model.
    """

    print("\n" + "=" * 50)
    print(f"MODEL: {model_name}")
    print("=" * 50)

    # =========================================================
    # 1. TRAIN MODEL
    # =========================================================

    model.fit(
        X_train,
        y_train,
    )

    print("Model training completed.")

    # =========================================================
    # 2. GENERATE PREDICTIONS
    # =========================================================

    predictions = model.predict(
        X_test
    )

    # Probability of the positive class (1)
    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # =========================================================
    # 3. CALCULATE METRICS
    # =========================================================

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

    # =========================================================
    # 4. PRINT RESULTS
    # =========================================================

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1-score:  {f1:.4f}"
    )

    return {
        "model": model_name,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run_experiment():

    # =========================================================
    # 1. LOAD DATASET
    # =========================================================

    dataset = load_dataset()

    print(
        f"Dataset loaded: "
        f"{len(dataset)} rows"
    )

    # =========================================================
    # 2. DEFINE FEATURES AND TARGET
    # =========================================================

    # FEATURE_COLUMNS comes directly from features.py.
    #
    # This means that if we add or remove features in
    # features.py, the experiment automatically uses
    # the updated feature set.

    X = dataset[
        FEATURE_COLUMNS
    ]

    y = dataset[
        "target"
    ]

    print("\n" + "=" * 50)
    print("FEATURES USED")
    print("=" * 50)

    print(
        FEATURE_COLUMNS
    )

    print("\nTarget distribution:")

    print(
        y.value_counts()
    )

    # =========================================================
    # 3. TIME-BASED TRAIN / TEST SPLIT
    # =========================================================

    # We use chronological splitting instead of random splitting.
    #
    # The model trains on older data and is tested on newer data.
    #
    # This better represents a real-world financial prediction
    # scenario.

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

    print("\n" + "=" * 50)
    print("TIME-BASED DATA SPLIT")
    print("=" * 50)

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

    # =========================================================
    # 4. DEFINE MODELS
    # =========================================================

    models = {

        # Logistic Regression
        "Logistic Regression": Pipeline(
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
        ),

        # Random Forest
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),

        # Gradient Boosting
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    # =========================================================
    # 5. RUN MODEL EXPERIMENTS
    # =========================================================

    results = []

    for model_name, model in models.items():

        result = evaluate_model(
            model,
            model_name,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        results.append(
            result
        )

    # =========================================================
    # 6. BASELINE
    # =========================================================

    # The baseline always predicts the most common
    # class found in the training dataset.

    majority_class = y_train.mode()[0]

    baseline_predictions = [
        majority_class
    ] * len(y_test)

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions,
    )

    print("\n" + "=" * 50)
    print("BASELINE")
    print("=" * 50)

    print(
        f"Baseline strategy: "
        f"Always predict {majority_class}"
    )

    print(
        f"Baseline accuracy: "
        f"{baseline_accuracy:.4f}"
    )

    results.append(
        {
            "model": "Baseline",
            "accuracy": baseline_accuracy,
            "roc_auc": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }
    )

    # =========================================================
    # 7. RESULTS TABLE
    # =========================================================

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)

    print(
        results_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    run_experiment()