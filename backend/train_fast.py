import torch
import torch.nn as nn
import torch.optim as optim
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

# -- TURBO MODE HYPERPARAMETERS --
# Super fast training just to generate valid model files for the demo
SEQ_LEN = 30
HIDDEN_DIM = 64
LAYERS = 2
DROPOUT = 0.2
EPOCHS = 1 # ONE EPOCH ONLY for speed
BATCH_SIZE = 16
LEARNING_RATE = 0.01
TICKERS = ['MSFT', 'GOOGL', 'META']

def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:(i + seq_len)]
        y = data[i + seq_len][0] # 0 is Close
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys).reshape(-1, 1)

def train_ticker_fast(ticker, loader):
    print(f"[TURBO-TRAIN] Compiling Neural Network for {ticker}...")
    
    # 1. Load Data (Smaller window for speed)
    df = loader.get_data(ticker, period="1y") 
    if df.empty:
        print(f"[ERROR] No data for {ticker}")
        return

    # 2. Feature Engineering
    df = AlphaGenerator.add_features(df)
    feature_cols = ['Close', 'rsi', 'macd', 'bb_width', 'ema_50', 'obv']
    df.dropna(inplace=True)
    
    # 3. Scaling
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(df[feature_cols].values)
    
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaled = target_scaler.fit_transform(df[['Close']].values) 

    X_seq, y_seq = create_sequences(data_scaled, SEQ_LEN)
    
    # Train only (skip robust validation text for speed)
    X_train = torch.from_numpy(X_seq).float()
    y_train = torch.from_numpy(y_seq).float()
    
    # 4. Model Setup
    input_dim = len(feature_cols)
    model = LSTMQuantAgent(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_layers=LAYERS, dropout=DROPOUT)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. Fast Training Loop
    model.train()
    for i in range(5): # Just 5 batches effectively
        optimizer.zero_grad()
        # Random batch
        idx = torch.randperm(len(X_train))[:BATCH_SIZE]
        y_pred = model(X_train[idx])
        loss = criterion(y_pred, y_train[idx])
        loss.backward()
        optimizer.step()
            
    # Save Checkpoint
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/{ticker}_model.pth")
    joblib.dump(scaler, f"models/{ticker}_scaler_X.pkl")
    joblib.dump(target_scaler, f"models/{ticker}_scaler_y.pkl")
    
    print(f"[SUCCESS] {ticker} AI Model Generated & Saved!")

if __name__ == "__main__":
    loader = InstitutionalDataLoader(data_dir="data")
    print("Optimization: Starting Turbo Training...")
    for ticker in TICKERS:
        try:
            train_ticker_fast(ticker, loader)
        except Exception as e:
            print(f"[FAIL] {ticker}: {e}")
    print("Done. Models are ready.")
