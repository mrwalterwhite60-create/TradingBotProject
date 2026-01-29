import pandas as pd
import ta
import numpy as np

class AlphaGenerator:
    """
    Quantitative Feature Engineering Engine.
    Generates alpha signals from raw OHLCV data using Technical Analysis.
    """
    
    @staticmethod
    def add_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Enriches the dataframe with technical indicators.
        """
        if df.empty:
            return df
            
        df = df.copy()
        
        # Flatten MultiIndex Columns if they exist
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # -- Indicators (Same as before) --
        df['rsi'] = ta.momentum.RSIIndicator(close=df['Close']).rsi()
        stoch = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        macd = ta.trend.MACD(close=df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        df['ema_50'] = ta.trend.EMAIndicator(close=df['Close'], window=50).ema_indicator()
        df['ema_200'] = ta.trend.EMAIndicator(close=df['Close'], window=200).ema_indicator()
        bb = ta.volatility.BollingerBands(close=df['Close'])
        df['bb_width'] = bb.bollinger_wband()
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()

        # -- Target Variable --
        if is_training:
            df['target_close'] = df['Close'].shift(-1)
            # Drop NaN rows (first 200 for indicators + last 1 for target)
            df.dropna(inplace=True)
        else:
            # During inference, we don't drop the last row!
            # We only drop the first 200 rows that have NaN indicators
            # We can use a simpler approach: drop based on 'ema_200' only
            df.dropna(subset=['ema_200'], inplace=True)
            # Ensure the relative last columns are filled if needed
        
        return df

if __name__ == "__main__":
    # Test connection with DataLoader
    try:
        from data_loader import InstitutionalDataLoader
        loader = InstitutionalDataLoader(data_dir="../data") # Adjusted path for execution in same dir
        df = loader.get_data("GOOGL")
        df_rich = AlphaGenerator.add_features(df)
        print(f"Generated {len(df_rich.columns)} features.")
        print(df_rich[['Close', 'rsi', 'macd', 'ichimoku_a']].tail())
    except ImportError:
        print("Please run this from the backend directory to test imports.")
