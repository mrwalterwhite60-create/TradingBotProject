# Team Project 2 Report: AI-Powered Telegram Trading Bot

**Defense Date:** Week 8  
**Project:** Antigravity AI Quantum Trader  

---

## 1. Project Goal and Context

The objective of this project was to shift from theoretical exploration to practical, production-oriented development by building a functional, automated stock prediction system accessible via a Telegram bot. 

Our team developed "Antigravity AI," a bot that provides real-time stock analysis for **Microsoft (MSFT)**, **Google (GOOGL)**, and **Meta Platforms (META)**. The system integrates a custom-trained **LSTM (Long Short-Term Memory)** neural network with an **Attention Mechanism** to predict the 24-hour target price, along with secondary projections for 1-week and 1-month horizons.

This project emphasizes deployment, system integration, and the user-facing delivery of AI-driven financial insights.

---

## 2. System Architecture

The bot operates on a modular architecture designed for scalability and maintainability.

### Workflow Diagram
`User (Telegram)` <--> `Python Backend (python-telegram-bot)` <--> `Inference Engine` <--> `AI Model (PyTorch)`
                                                                     ^
                                                                     |
                                                               `Data Loader` <--> `Yahoo Finance API`

### Components
1.  **Interface Layer (Telegram Bot)**:
    -   Built using `python-telegram-bot` (AsyncIO).
    -   Handles user commands (`/start`, callbacks) and renders interactive buttons.
    -   Displays results using Markdown text and generated Matplotlib charts.

2.  **Application Logic (Backend)**:
    -   `bot_main.py`: The central controller that routes requests.
    -   `inference.py`: Orchestrates the prediction process. It loads the saved PyTorch models and Scalers, fetches live data, and runs inference.

3.  **AI Layer (Model Engine)**:
    -   `model_engine.py`: Defines the `LSTMQuantAgent` class.
    -   **Architecture**:
        -   **Input Layer**: Accepts a sequence of technical indicators (60-day lookback).
        -   **LSTM Layers**: Two stacked LSTM layers (Hidden Dim: 64) to capture temporal dependencies.
        -   **Attention Mechanism**: A custom Attention layer that weights the importance of different time steps in the input sequence, allowing the model to focus on critical market shifts.
        -   **Output Layer**: A Fully Connected layer producing the predicted price.
    -   **Uncertainty Estimation**: Uses **Monte Carlo Dropout** during inference to generate a confidence interval (risk metric).

4.  **Data Layer**:
    -   `data_loader.py`: Fetches historical data using `yfinance`.
    -   `indicators.py`: generating technical features (RSI, MACD, Bollinger Bands, EMA, On-Balance Volume).

---

## 3. Model Training and Validation

### Data Source
We utilized free, public data from **Yahoo Finance**.
-   **Tickers**: MSFT, GOOGL, META.
-   **Range**: 5 Years of daily data for training; 1 Year for inference context.

### Feature Engineering
Raw price data is insufficient for robust prediction. We engineered the following "Alpha" factors:
-   **Relative Strength Index (RSI)**: Momentum indicator to identify overbought/oversold conditions.
-   **MACD**: Trend-following momentum indicator.
-   **Bollinger Band Width**: Measure of volatility.
-   **Exponential Moving Average (EMA-50)**: Medium-term trend baseline.
-   **On-Balance Volume (OBV)**: Volume flow to confirm price moves.

### Model Architecture Selection
We chose **LSTM** over standard RNNs or Linear Regression because financial data is inherently time-series based with long-term dependencies.
-   **Why Attention?** Standard LSTMs can struggle with long sequences (vanishing gradient). Attention allows the model to "look back" at specific high-volatility days in the 60-day window, improving predictions during market turning points.

### Validation Strategy
-   **Split**: Time-Series Split (First 80% for Training, Last 20% for Validation). We did *not* use random shuffling to prevent data leakage (predicting past using future).
-   **Loss Function**: MSE (Mean Squared Error).
-   **Metrics**:
    -   **Model Confidence**: Calculated inversely proportional to the standard deviation of Monte Carlo Dropout predictions.
    -   **Visual Verification**: The bot plots the predicted "Next Close" against the current price, with error bars representing the confidence interval.

---

## 4. Bot Implementation & UX

### Key Functions
1.  **`/start`**: Initializes the session and presents a rich Graphical User Interface (Inline Keyboards) instead of relying on text commands.
2.  **`Predict [Ticker]`**:
    -   User selects a stock from the menu.
    -   **UX Detail**: The bot shows a "Loading..." state with steps ("Fetching Data...", "Running Neural Net...") to provide feedback during the ~2-second inference latency.
    -   **Result**: Displays a dynamically generated chart with three horizons:
        -   **1-Day (AI model)**: High-precision neural net prediction.
        -   **1-Week (Trend)**: Statistical projection based on current momentum.
        -   **1-Month (Trend)**: Long-range outlook.
    -   **Live Data**: Uses a high-frequency polling pipeline to fetch the exact last price from Yahoo Finance.
3.  **`System Status`**:
    -   Checks connectivity to the Neural Network and Data API.
    -   Reports Latency (simulated/measured).

### UX Analysis
-   **Challenges**: Mobile users prioritize speed. Generating images (matplotlib) takes computational time.
-   **Solution**: We used `io.BytesIO` to generate images in memory (RAM) rather than saving to disk, reducing latency by ~300ms. We also added "Loading" messages so the user knows the bot is working.

---

## 5. Conclusion

Deploying an AI model to a real-time environment presents unique challenges compared to a static Jupyter Notebook application. 
1.  **Latency**: Inference speed matters. Our optimized PyTorch model runs in <50ms, but data fetching takes ~500ms. Caching was critical.
2.  **Noise**: Financial data is noisy. The "Confidence Score" feature was essential to communicate *uncertainty* to the user, rather than presenting a prediction as absolute truth.
3.  **Reliability**: Handling API limits and missing data required robust error handling in `inference.py`.

The "Antigravity AI" project successfully demonstrates a full-stack AI application, moving from raw data to a user-friendly, production-ready tool.
