# Project 2 Documentation: Roles, Responsibilities, and Logbook

## Roles and Responsibilities

| Team Member | Role | Responsibilities |
| :--- | :--- | :--- |
| **Member 1** | **Project Lead & AI Engineer** | Designed the LSTM architecture (`model_engine.py`), handled data preprocessing (`indicators.py`), and managed overall project timeline. |
| **Member 2** | **Backend Developer** | Implemented the Telegram Bot logic (`bot_main.py`), set up the asynchronous event loop, and handled API integration. |
| **Member 3** | **Data Analyst & QA** | Conducted data analysis, selected technical indicators, performed testing (`inference.py`), and wrote the final report/documentation. |

---

## Project Logbook

| Date | Activity | Details | Outcome |
| :--- | :--- | :--- | :--- |
| **Week 1** | Project Kickoff | Defined scope: Telegram Bot + LSTM for MSFT, GOOGL, META. | GitHub Repo created. |
| **Week 2** | Data Collection | Wrote `data_loader.py` to fetch Yahoo Finance data. | Successful data download. |
| **Week 3** | Feature Engineering | Added RSI, MACD, and Bollinger Bands in `indicators.py`. | Data ready for training. |
| **Week 4** | Model Design | Built `LSTMQuantAgent` in PyTorch with Attention mechanism. | Initial model structure. |
| **Week 5** | Training Phase | Trained models on 5 years of data. Encountered overfitting. | Added Dropout (0.2). |
| **Week 6** | Bot Interface | Built `bot_main.py` with inline keyboards. | Functional prototype. |
| **Week 7** | System Integration | Connected Bot to Inference Engine. Added Uncertainty estimation. | End-to-end system working. |
| **Week 8** | Final Polish | Optimized latency, generated reports, and rehearsed presentation. | **Project Ready for Defense.** |

---

## Peer Evaluation (Confidential)

*To be filled out individually by each member.*

| Member | Contribution Score (1-5) | Comments |
| :--- | :--- | :--- |
| Member 1 | 5 | Excellent leadership and technical skills. |
| Member 2 | 5 | Delivered robust backend code on time. |
| Member 3 | 5 | Great documentation and testing support. |
