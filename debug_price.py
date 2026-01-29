import yfinance as yf
import pandas as pd

def debug_price(ticker):
    print(f"--- Debugging {ticker} ---")
    try:
        # Method 1: yf.download 1m
        print("Method 1: yf.download(period='1d', interval='1m')")
        data = yf.download(ticker, period="1d", interval="1m", progress=False)
        print(f"Index: {data.index[-1]}")
        print(f"Columns: {data.columns}")
        # Show last few rows
        print(data.tail(2))
        
        # Method 2: yf.Ticker fast_info
        print("\nMethod 2: yf.Ticker.fast_info")
        t = yf.Ticker(ticker)
        print(f"Last Price: {t.fast_info['last_price']}")
        
        # Method 3: yf.download daily
        print("\nMethod 3: yf.download(period='1y')")
        df = yf.download(ticker, period="1y", progress=False)
        print(f"Last Daily Close: {df['Close'].iloc[-1]}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_price("MSFT")
