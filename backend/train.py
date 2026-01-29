import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Import our custom modules
# Note: When running as script, these imports depend on sys.path or running from root
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.data_loader import InstitutionalDataLoader
from backend.indicators import AlphaGenerator
from backend.model_engine import LSTMQuantAgent

# -- Hyperparameters --
SEQ_LEN = 60 # Lookback window (e.g. 60 days)
HIDDEN_DIM = 64
LAYERS = 2
DROPOUT = 0.2
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TICKERS = ['MSFT', 'GOOGL', 'META']

def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:(i + seq_len)]
        y = data[i + seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_ticker(ticker, loader):
    print(f"\n[TRAINING] Starting pipeline for {ticker}...")
    
    # 1. Load Data
    df = loader.get_data(ticker, period="5y") # More data for deep learning
    if df.empty:
        print(f"[ERROR] No data for {ticker}")
        return

    # 2. Feature Engineering
    df = AlphaGenerator.add_features(df)
    
    # Select features for model
    # We predict 'Close' based on ['Close', 'rsi', 'macd', 'volatility', etc]
    feature_cols = ['Close', 'rsi', 'macd', 'bb_width', 'ema_50', 'obv']
    target_col = 'target_close' # We created this in indicators.py (shifted close)
    
    # Drop rows with NaN (from indicators)
    df.dropna(inplace=True)
    
    # 3. Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(df[feature_cols].values)
    
    # The target is likely the 'Close' price (index 0 of features) shifted. 
    # But wait, 'target_close' is the actual price. Let's scale targets separately or use the Close column.
    # To simplify inference, let's train to predict the *next Close* from the *current sequence of features*.
    # So Y is just the 'Close' column shifted.
    
    # Better approach for stability: Predict the Scaled Close.
    # x: [0..59] features
    # y: [60] Close price (scaled)
    
    # We need a scaler just for the target to inverse transform later
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaled = target_scaler.fit_transform(df[['Close']].values) # Use Current Close as proxy for scale
    
    # X data: all features
    X_data = data_scaled
    # Y data: target_close (shifted)
    # We already have 'target_close' in df, let's scale it using target_scaler
    y_data = target_scaler.transform(df[['target_close']].values)

    X_seq, y_seq = create_sequences(X_data, SEQ_LEN)
    
    # Train/Test Split (Time series split, no random shuffle!)
    train_size = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:train_size], X_seq[train_size:]
    y_train, y_test = y_seq[:train_size], y_seq[train_size:]
    
    # To Tensor
    train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    test_data = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)
    
    # 4. Model Setup
    input_dim = len(feature_cols)
    model = LSTMQuantAgent(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_layers=LAYERS, dropout=DROPOUT)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. Training Loop
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()
        
        val_loss /= len(test_loader)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss {epoch_loss/len(train_loader):.5f}, Val Loss {val_loss:.5f}")
            
        if val_loss < best_loss:
            best_loss = val_loss
            # Save Checkpoint
            torch.save(model.state_dict(), f"models/{ticker}_model.pth")
    
    # Save Scalers
    joblib.dump(scaler, f"models/{ticker}_scaler_X.pkl")
    joblib.dump(target_scaler, f"models/{ticker}_scaler_y.pkl")
    
    print(f"[SUCCESS] Trained {ticker}. Best Val Loss: {best_loss:.5f}")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    loader = InstitutionalDataLoader(data_dir="data")
    
    for ticker in TICKERS:
        try:
            train_ticker(ticker, loader)
        except Exception as e:
            print(f"[FAIL] Error training {ticker}: {e}")
