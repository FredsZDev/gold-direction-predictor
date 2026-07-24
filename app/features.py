import pandas as pd


# Features that will be used by the machine learning models
FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "price_vs_sma_10",
    "price_vs_sma_30",
    "sma_10_vs_sma_30",
    "volatility_10",
    "volatility_20",
    "volume_change",
    "price_position_20",
]


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create predictive features from historical Gold price data.

    The target variable is not created here.
    Each experiment defines its own prediction horizon.
    """

    # Create a copy to avoid modifying the original DataFrame
    df = data.copy()

    # ========================================================
    # 1. Daily return
    # ========================================================

    # Measures the percentage change from the previous trading day
    df["return_1d"] = (
        df["Close"].pct_change()
    )

    # ========================================================
    # 2. Five-day return
    # ========================================================

    # Measures the percentage change over approximately one week
    df["return_5d"] = (
        df["Close"].pct_change(periods=5)
    )

    # ========================================================
    # 3. Ten-day return
    # ========================================================

    # Measures medium-term price momentum
    df["return_10d"] = (
        df["Close"].pct_change(periods=10)
    )

    # ========================================================
    # 4. Short-term moving average
    # ========================================================

    sma_10 = (
        df["Close"]
        .rolling(window=10)
        .mean()
    )

    # Measures how far the current price is from
    # its 10-day moving average
    df["price_vs_sma_10"] = (
        df["Close"] / sma_10 - 1
    )

    # ========================================================
    # 5. Medium-term moving average
    # ========================================================

    sma_30 = (
        df["Close"]
        .rolling(window=30)
        .mean()
    )

    # Measures how far the current price is from
    # its 30-day moving average
    df["price_vs_sma_30"] = (
        df["Close"] / sma_30 - 1
    )

    # ========================================================
    # 6. Moving average relationship
    # ========================================================

    # Compares short-term and medium-term trends
    df["sma_10_vs_sma_30"] = (
        sma_10 / sma_30 - 1
    )

    # ========================================================
    # 7. Short-term volatility
    # ========================================================

    # Measures how much daily returns fluctuate
    # over the last 10 trading days
    df["volatility_10"] = (
        df["return_1d"]
        .rolling(window=10)
        .std()
    )

    # ========================================================
    # 8. Medium-term volatility
    # ========================================================

    # Measures price fluctuation over a longer period
    df["volatility_20"] = (
        df["return_1d"]
        .rolling(window=20)
        .std()
    )

    # ========================================================
    # 9. Volume change
    # ========================================================

    # Measures the percentage change in trading volume
    df["volume_change"] = (
        df["Volume"].pct_change()
    )

    # ========================================================
    # 10. Price position within 20-day range
    # ========================================================

    # Highest price during the last 20 trading days
    rolling_high_20 = (
        df["High"]
        .rolling(window=20)
        .max()
    )

    # Lowest price during the last 20 trading days
    rolling_low_20 = (
        df["Low"]
        .rolling(window=20)
        .min()
    )

    # Determines where the current closing price
    # is positioned inside the recent 20-day price range
    df["price_position_20"] = (
        (df["Close"] - rolling_low_20)
        /
        (rolling_high_20 - rolling_low_20)
    )

    # ========================================================
    # 11. Remove infinite values
    # ========================================================

    # Prevents infinite values from being passed
    # to machine learning models
    df = df.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    # ========================================================
    # 12. Remove rows with missing values
    # ========================================================

    # Rolling calculations and percentage changes
    # generate missing values at the beginning of the dataset
    df = df.dropna()

    return df


if __name__ == "__main__":

    # Load the raw dataset
    data = pd.read_csv(
        "data/raw/xauusd.csv",
        index_col="Date",
        parse_dates=True,
    )

    # Create the features
    dataset = create_features(data)

    print("\n" + "=" * 40)
    print("FEATURE ENGINEERING V3 REPORT")
    print("=" * 40)

    print(f"Original rows: {len(data)}")
    print(
        f"Rows after feature engineering: {len(dataset)}"
    )

    print("\nFeatures used by the model:")

    for feature in FEATURE_COLUMNS:
        print(f"- {feature}")

    print("\nFeature preview:")

    print(
        dataset[FEATURE_COLUMNS]
        .head(10)
    )

    print("\nInfinite values:")

    print(
        dataset[FEATURE_COLUMNS]
        .isin([float("inf"), float("-inf")])
        .sum()
    )