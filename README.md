# Antigravity AI: Quantum Trading Bot (Project 2)

**Defense Week 8** | **Team Antigravity**

A production-ready Telegram Bot that uses a custom **LSTM-Attention Neural Network** to predict stock prices for **MSFT**, **GOOGL**, and **META**.

## Features
-   **Real-time Inference**: Queries live market data.
-   **Deep Learning Model**: PyTorch-based LSTM with Attention mechanism.
-   **Uncertainty Estimation**: Monte Carlo Dropout to provide Confidence Scores.
-   **Interactive UI**: Inline keyboards and generated charts.

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Set up your `.env` file with `TELEGRAM_BOT_TOKEN_PROJ2`.

## Usage

1.  **Train the Models** (First time only):
    ```bash
    python backend/train.py
    ```
2.  **Run the Bot**:
    ```bash
    python backend/bot_main.py
    ```

## Project Structure
-   `backend/`: Python source code (Bot, Model, Training).
-   `models/`: Saved PyTorch models (`.pth`) and Scalers.
-   `data/`: Cached stock data.

## Deliverables
-   [Report](PROJECT_REPORT.md)
-   [Roles & Logbook](ROLES_AND_LOGBOOK.md)
