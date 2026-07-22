import pandas as pd


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create predictive features and target variable.

    All features are calculated using only current and past data
    to avoid look-ahead leakage.
    """

    # Create a copy so we don't modify the original DataFrame
    df = data.copy()

    # =========================================================
    # 1. DAILY RETURN
    # =========================================================

    # Measures the percentage change from the previous candle
    df["return_1d"] = df["Close"].pct_change()

    # =========================================================
    # 2. 5-DAY RETURN
    # =========================================================

    # Measures the price momentum over the last 5 periods
    df["return_5d"] = df["Close"].pct_change(5)

    # =========================================================
    # 3. MOVING AVERAGES
    # =========================================================

    # Calculate short-term and medium-term moving averages
    sma_10 = df["Close"].rolling(window=10).mean()
    sma_30 = df["Close"].rolling(window=30).mean()

    # =========================================================
    # 4. PRICE RELATIVE TO SMA 10
    # =========================================================

    # Measures how far the current price is from the 10-period
    # moving average.
    #
    # Example:
    # Close = 105
    # SMA10  = 100
    #
    # Result = 0.05
    #
    # This means the price is 5% above the SMA10.

    df["price_vs_sma_10"] = (
        df["Close"] / sma_10 - 1
    )

    # =========================================================
    # 5. PRICE RELATIVE TO SMA 30
    # =========================================================

    # Measures how far the current price is from the 30-period
    # moving average.

    df["price_vs_sma_30"] = (
        df["Close"] / sma_30 - 1
    )

    # =========================================================
    # 6. ROLLING VOLATILITY
    # =========================================================

    # Measures the standard deviation of recent daily returns.
    # Higher values indicate higher recent price volatility.

    df["volatility_10"] = (
        df["return_1d"]
        .rolling(window=10)
        .std()
    )

    # =========================================================
    # 7. VOLUME CHANGE
    # =========================================================

    # Measures the percentage change in trading volume.

    df["volume_change"] = (
        df["Volume"]
        .pct_change()
    )

    # =========================================================
    # 8. TARGET
    # =========================================================

    # Target:
    #
    # 1 = The next candle closes higher than the current candle
    # 0 = The next candle closes lower or equal to the current candle
    #
    # shift(-1) looks at the NEXT available candle.
    #
    # This is our target variable, not a feature.
    # The model will never receive this column as input.

    df["target"] = (
        df["Close"].shift(-1) > df["Close"]
    ).astype(int)

    # =========================================================
    # 9. HANDLE INFINITE VALUES
    # =========================================================

    # Percentage calculations can generate infinity if
    # the previous value is zero.
    #
    # We replace infinity with NaN before removing invalid rows.

    df = df.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    # =========================================================
    # 10. REMOVE MISSING VALUES
    # =========================================================

    # Rolling calculations and percentage changes naturally
    # generate NaN values at the beginning of the dataset.

    df = df.dropna()

    return df


if __name__ == "__main__":

    # =========================================================
    # LOAD RAW DATA
    # =========================================================

    data = pd.read_csv(
        "data/raw/xauusd.csv",
        index_col="Date",
        parse_dates=True
    )

    # =========================================================
    # CREATE FEATURES
    # =========================================================

    dataset = create_features(data)

    # =========================================================
    # PRINT REPORT
    # =========================================================

    print("\n" + "=" * 40)
    print("FEATURE ENGINEERING REPORT")
    print("=" * 40)

    print(
        f"Original rows: "
        f"{len(data)}"
    )

    print(
        f"Rows after feature engineering: "
        f"{len(dataset)}"
    )

    print("\nFeatures used by the model:")

    feature_columns = [
        "return_1d",
        "return_5d",
        "price_vs_sma_10",
        "price_vs_sma_30",
        "volatility_10",
        "volume_change",
    ]

    print(feature_columns)

    print("\nFeatures and target:")

    print(
        dataset[
            feature_columns + ["target"]
        ].head(10)
    )

    print("\nTarget distribution:")

    print(
        dataset["target"]
        .value_counts()
        .sort_index()
    )

    print("\nInfinite values:")

    print(
        dataset[
            feature_columns
        ]
        .isin(
            [
                float("inf"),
                float("-inf")
            ]
        )
        .sum()
    )