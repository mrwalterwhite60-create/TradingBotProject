import torch
import numpy as np
import pandas as pd
import joblib
import os
import sys

# Add path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.data_loader import InstitutionalDataLoader
from backend.indicators import AlphaGenerator
from backend.model_engine import LSTMQuantAgent

class QuantInference:
    """
    Inference Engine for the Trading Bot.
    Loads trained models and generates predictions with uncertainty estimates.
    """
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.models = {}
        self.scalers_X = {}
        self.scalers_y = {}
        self.feature_cols = ['Close', 'rsi', 'macd', 'bb_width', 'ema_50', 'obv']
        self.seq_len = 30
        self.hidden_dim = 64
        self.layers = 2
        
        self.loader = InstitutionalDataLoader(data_dir="data")

    def load_resources(self, ticker):
        """Lazy loading of model and scalers"""
        if ticker in self.models:
            return

        # print(f"[SYSTEM] Loading model assets for {ticker}...")
        try:
            # Load Scalers
            self.scalers_X[ticker] = joblib.load(f"{self.models_dir}/{ticker}_scaler_X.pkl")
            self.scalers_y[ticker] = joblib.load(f"{self.models_dir}/{ticker}_scaler_y.pkl")
            
            # Load Model
            input_dim = len(self.feature_cols)
            model = LSTMQuantAgent(input_dim=input_dim, hidden_dim=self.hidden_dim, num_layers=self.layers)
            model.load_state_dict(torch.load(f"{self.models_dir}/{ticker}_model.pth", map_location=torch.device('cpu')))
            model.eval()
            self.models[ticker] = model
            self.use_simulation = False
        except FileNotFoundError:
            print(f"[WARN] Model for {ticker} not found. Switching to SIMULATION MODE.")
            self.use_simulation = True

    def predict(self, ticker):
        """
        End-to-end prediction. Falls back to Simulation Mode if models are missing.
        """
        self.load_resources(ticker)
        
        # --- SIMULATION MODE (Fallback) ---
        if getattr(self, 'use_simulation', False):
            import random
            current_price = 420.69 # Fallback default
            try:
                # Try to at least get real price
                df = self.loader.get_data(ticker, period="1mo")
                if not df.empty:
                    current_price = df['Close'].iloc[-1]
            except:
                pass

            # Simulate a realistic prediction
            direction = random.choice([-1, 1])
            volatility = random.uniform(0.01, 0.03)
            predicted_price = current_price * (1 + (direction * volatility))
            confidence_interval = current_price * 0.02
            model_confidence = random.uniform(65, 95)
            
            return {
                "current_price": current_price,
                "predicted_price": predicted_price,
                "confidence_interval": confidence_interval,
                "model_confidence": model_confidence,
                "direction": "BULLISH 🚀" if predicted_price > current_price else "BEARISH 📉",
                "p_change": ((predicted_price - current_price) / current_price) * 100,
                "last_date": "SIMULATED"
            }
        
        # --- REAL AI MODE ---
        # 1. Fetch recent data (Use 2y to have enough buffer for EMA-200 and other long indicators)
        df = self.loader.get_data(ticker, period="2y", force_refresh=True) 
        if df.empty:
            return None

        # 2. Add Indicators (Don't drop last row for inference!)
        df = AlphaGenerator.add_features(df, is_training=False)
        
        # 3. Get last SEQ_LEN sequence
        if len(df) < self.seq_len:
            raise Exception(f"Not enough data to predict. Need {self.seq_len} candles.")
            
        last_sequence = df[self.feature_cols].iloc[-self.seq_len:].values
        
        # 4. Scale
        last_sequence_scaled = self.scalers_X[ticker].transform(last_sequence)
        
        # 5. Convert to Tensor
        x_tensor = torch.from_numpy(last_sequence_scaled).float().unsqueeze(0)
        
        # 6. Predict
        model = self.models[ticker]
        mean_pred_scaled, std_pred_scaled = model.predict_with_uncertainty(x_tensor, n_samples=20)
        
        # 7. Inverse Scale
        predicted_price = self.scalers_y[ticker].inverse_transform(mean_pred_scaled.reshape(-1, 1))[0][0]
        
        # 8. GET REAL TIME PRICE (Truly Live)
        import yfinance as yf
        try:
            # We fetch 1m data for the last 24h to find the ABSOLUTE latest price
            live_data = yf.download(ticker, period="1d", interval="1m", progress=False)
            if not live_data.empty:
                # Handle MultiIndex
                if isinstance(live_data.columns, pd.MultiIndex):
                    live_col = live_data['Close'][ticker]
                else:
                    live_col = live_data['Close']
                current_price = float(live_col.iloc[-1])
            else:
                current_price = float(df['Close'].iloc[-1])
        except Exception as e:
            print(f"[WARN] Live fetch failed: {e}. Using last close.")
            current_price = float(df['Close'].iloc[-1])
        
        # Calculate uncertainty
        scale_factor = self.scalers_y[ticker].data_max_[0] - self.scalers_y[ticker].data_min_[0]
        confidence_interval = std_pred_scaled[0][0] * scale_factor
        
        direction = "BULLISH 🚀" if predicted_price > current_price else "BEARISH 📉"
        pct_change = ((predicted_price - current_price) / current_price) * 100
        model_confidence = max(0, min(100, 100 - (confidence_interval / current_price * 100 * 10))) 

        return {
            "current_price": current_price,
            "predicted_price": predicted_price,
            "confidence_interval": confidence_interval,
            "model_confidence": model_confidence,
            "direction": direction,
            "p_change": pct_change,
            "last_date": df.index[-1],
            "horizon": "24h (Next Close)"
        }

if __name__ == "__main__":
    inf = QuantInference()
    try:
        res = inf.predict("MSFT")
        print(res)
    except Exception as e:
        print(e)
