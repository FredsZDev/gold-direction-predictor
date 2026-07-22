import pandas as pd


def create_features(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Create predictive features.

    The target variable is intentionally not created here.
    Each experiment or training pipeline is responsible
    for defining its own prediction target.
    """

    df = data.copy()

    # ========================================================
    # 1. Daily return
    # ========================================================

    df["return_1d"] = (
        df["Close"].pct_change()
    )

    # ========================================================
    # 2. Five-day return
    # ========================================================

    df["return_5d"] = (
        df["Close"].pct_change(
            periods=5
        )
    )

    # ========================================================
    # 3. Short-term price position
    # ========================================================

    sma_10 = (
        df["Close"]
        .rolling(window=10)
        .mean()
    )

    df["price_vs_sma_10"] = (
        df["Close"] / sma_10 - 1
    )

    # ========================================================
    # 4. Medium-term price position
    # ========================================================

    sma_30 = (
        df["Close"]
        .rolling(window=30)
        .mean()
    )

    df["price_vs_sma_30"] = (
        df["Close"] / sma_30 - 1
    )

    # ========================================================
    # 5. Rolling volatility
    # ========================================================

    df["volatility_10"] = (
        df["return_1d"]
        .rolling(window=10)
        .std()
    )

    # ========================================================
    # 6. Volume change
    # ========================================================

    df["volume_change"] = (
        df["Volume"].pct_change()
    )

    # ========================================================
    # 7. Remove infinite values
    # ========================================================

    df = df.replace(
        [
            float("inf"),
            float("-inf"),
        ],
        pd.NA,
    )

    # ========================================================
    # 8. Remove rows with missing values
    # ========================================================

    df = df.dropna()

    return df