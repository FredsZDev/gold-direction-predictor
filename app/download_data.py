from pathlib import Path

# External libraries
import pandas as pd
import yfinance as yf


def download_gold_data():
    """
    Download historical OHLC data for Gold from Yahoo Finance.

    We use the Gold Futures ticker (GC=F) as a proxy for gold price data,
    since Yahoo Finance does not provide XAUUSD directly through this ticker.
    """

    ticker = "GC=F"

    # Download historical data
    data = yf.download(
        ticker,
        start="2015-01-01",
        progress=False,
        auto_adjust=False
    )

    # Handle case where download returns None
    if data is None or data.empty:
        raise ValueError("No data was found/downloaded.")

    # Flatten yfinance MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    # Make sure the index has a proper name
    data.index.name = "Date"

    def validate_dataset(data: pd.DataFrame) -> None:
        """
        Validate the downloaded dataset.
        """

        print("\n" + "=" * 40)
        print("DATASET VALIDATION REPORT")
        print("=" * 40)

        rows, cols = data.shape

        print(f"Rows: {rows}")
        print(f"Columns: {cols}")

        missing_values = data.isnull().sum()

        print("\nMissing values:")
        print(missing_values)

        if data.empty:
            raise ValueError("No data was found/downloaded.")

    # Validate downloaded data
    validate_dataset(data)

    print(
        f"\nDownloaded {len(data)} rows of data "
        f"for {ticker} from Yahoo Finance."
    )

    print("\nFirst rows:")
    print(data.head())

    print("\nLast rows:")
    print(data.tail())

    # Save the data to a CSV file
    output_path = Path("data/raw")
    output_path.mkdir(parents=True, exist_ok=True)

    csv_file = output_path / "xauusd.csv"

    data.to_csv(csv_file)

    print(f"\nDataset saved to {csv_file}")


if __name__ == "__main__":
    print("SCRIPT STARTED")

    download_gold_data()