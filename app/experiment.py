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

from features import create_features


def load_dataset():
    """
    Load the raw gold dataset and create the features
    used by the machine learning models.
    """

    data_path = Path("data/raw/xauusd.csv")

    data = pd.read_csv(
        data_path,
        index_col="Date",
        parse_dates=True,
    )

    dataset = create_features(data)

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

    # Train the model
    model.fit(
        X_train,
        y_train,
    )

    # Generate class predictions
    predictions = model.predict(X_test)

    # Generate probability predictions
    probabilities = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
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

    # Print results
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

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
    # 2. DEFINE FEATURES
    # =========================================================

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

    y = dataset["target"]

    # =========================================================
    # 3. TIME-BASED TRAIN / TEST SPLIT
    # =========================================================

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
    # 5. RUN EXPERIMENTS
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

        results.append(result)

    # =========================================================
    # 6. BASELINE
    # =========================================================

    majority_class = y_train.mode()[0]

    baseline_predictions = [
        majority_class
    ] * len(y_test)

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions,
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