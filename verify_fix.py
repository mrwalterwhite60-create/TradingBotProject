from backend.inference import QuantInference
from backend.data_loader import InstitutionalDataLoader
from backend.indicators import AlphaGenerator
import os
import pandas as pd

def debug_full_pipeline(ticker):
    print(f"--- Debugging Full Pipeline for {ticker} ---")
    loader = InstitutionalDataLoader(data_dir="data")
    df = loader.get_data(ticker, period="1y", force_refresh=True)
    print(f"Initial rows: {len(df)}")
    
    df_rich = AlphaGenerator.add_features(df)
    print(f"Rows after indicators: {len(df_rich)}")
    
    if len(df_rich) > 0:
        print("Last 3 rows after indicators:")
        print(df_rich[['Close', 'rsi', 'ema_200', 'target_close']].tail(3))
    
    inf = QuantInference()
    try:
        res = inf.predict(ticker)
        print(f"\nFinal Result for {ticker}:")
        print(f"Live Price: ${res['current_price']:.2f}")
        print(f"Predicted: ${res['predicted_price']:.2f}")
    except Exception as e:
        print(f"\nInference Error: {e}")

if __name__ == "__main__":
    debug_full_pipeline("MSFT")
