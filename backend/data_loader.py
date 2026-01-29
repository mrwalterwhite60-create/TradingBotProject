import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

class InstitutionalDataLoader:
    """
    Production-grade data loader for retrieving financial time series.
    Implements local caching to minimize API latency and respect rate limits.
    """
    def __init__(self, data_dir="data", cache_expiry_hours=1):

        self.data_dir = data_dir
        self.cache_expiry = timedelta(hours=cache_expiry_hours)
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"[SYSTEM] Data Pipeline Initialized. Storage: {self.data_dir}")

    def get_data(self, ticker: str, period="2y", interval="1d", force_refresh=False) -> pd.DataFrame:
        """
        Fetches OHLCV data for a given ticker.
        Checks local cache first, updates if expired or forced.
        """
        file_path = os.path.join(self.data_dir, f"{ticker}_{period}_{interval}.parquet")
        
        # Check Cache
        if not force_refresh and os.path.exists(file_path):
            modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if datetime.now() - modified_time < self.cache_expiry:
                # print(f"[CACHE] Loading {ticker} from local storage.")
                return pd.read_parquet(file_path)

        # Fetch Live Data
        print(f"[NETWORK] Fetching live data for {ticker} via Yahoo Finance API...")
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if df.empty:
                raise ValueError(f"No data found for {ticker}")
            
            # Flatten MultiIndex Columns if they exist (yfinance v0.2.x quirk)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Save to Parquet
            df.to_parquet(file_path)
            print(f"[SYSTEM] Cached {len(df)} rows for {ticker}.")
            return df
        except Exception as e:
            print(f"[ERROR] Failed to fetch {ticker}: {e}")
            return pd.DataFrame()

    def get_multiple(self, tickers: list, period="2y") -> dict:
        return {t: self.get_data(t, period=period) for t in tickers}

if __name__ == "__main__":
    # Smoke Test
    loader = InstitutionalDataLoader()
    df = loader.get_data("MSFT")
    print(df.tail())
