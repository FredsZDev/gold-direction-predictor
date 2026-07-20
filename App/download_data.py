from pathlib import Path

#external libraries imports
import pandas as pd
import yfinance as yf
#----------------------------


def download_gold_data():
    """
    Download historical OHLC data for Gold
    from Yahoo Finance. It's good to note that i choose Yahoo Finance because it is simpler  to use
    """

    ticker = "GC=F" #Yahoo Finance does not makes available directly XAUUSD, so we have to use the Gold Futures instead 

    data = yf.download(  
    ticker,
    start="2015-01-01",
    progress=False,
    auto_adjust=False
    # 10 years of data is enough for our analysis
    )

    if data.empty:
        raise ValueError("No data was found/downloaded.")
    print(f"Downloaded {len(data)} rows of data for {ticker} from Yahoo Finance.")
    print(data.head())
    print(data.tail())

    # Save the data to a CSV file
    output_path = Path("data/raw")
    output_path.mkdir(parents=True, exist_ok=True)

    csv_file = output_path / "xauusd.csv"

    data.to_csv(csv_file)
    print(f"\nDataset saved to {csv_file}")

    if  __name_ == "_main_":
        download_gold_data()