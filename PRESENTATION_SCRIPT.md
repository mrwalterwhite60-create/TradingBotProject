# Presentation Script: Antigravity AI Trading Bot
**Time Limit:** 10 Minutes  
**Speakers:** Team Members

---

## Slide 1: Title Slide (0:00 - 0:30)
**Speaker A:** "Good morning everyone. We are Team Antigravity, and today we present our Project 2: The AI-Powered Telegram Trading Bot. Our goal was to take the machine learning concepts from this course and deploy them into a real-world, user-friendly tool."

## Slide 2: The Problem & Solution (0:30 - 1:30)
**Speaker A:** "Stock market analysis is complex and time-consuming. Traders need quick, data-driven insights. Our solution is a Telegram Bot that acts as a pocket quantitative analyst. It doesn't just give you a price; it uses a custom LSTM Neural Network to predict trends and, importantly, tells you how confident it is in that prediction."

## Slide 3: System Architecture (1:30 - 2:30)
**Speaker B:** "Let's look under the hood. Our system has three main layers:
1.  **The Interface**: A Telegram Bot built with Python, serving as the frontend.
2.  **The Brain**: A PyTorch backend running our custom LSTM-Attention models.
3.  **The Data**: We pipeline live market data from Yahoo Finance, process it through technical indicators like RSI and MACD, and feed it to the AI."

## Slide 4: AI Model Deep Dive (2:30 - 4:00)
**Speaker B:** "For the AI, we didn't just use a simple regression. We implemented an **LSTM (Long Short-Term Memory)** network. Financial data is a time series—what happened 30 days ago affects today.
We added an **Attention Mechanism**, similar to what's used in Transformers, allowing the model to focus on specific high-volatility days in the past.
We also trained separate models for MSFT, GOOGL, and META to capture their unique behaviors."

## Slide 5: Live Demonstration (4:00 - 7:00) [CRITICAL PART]
**(Switch screen to Telegram Desktop)**

**Speaker C (Demo Mode):** "Let's see it in action.
1.  I start the bot with `/start`. You see a menu, not just text. Good UX is key.
2.  Let's check **Status**. It shows our Neural Net is online and latency is low.
3.  Now, the magic: **AI Prediction**. I'll choose **MSFT (Microsoft)**.
    *(Click Button)*
    You see it says 'Analyzing...'. It's fetching live data right now.
4.  **Result**: Here is the generated chart. The Grey bar is the current price, the Green/Red bar is the prediction.
    -   Notice the **Error Bar**: That's our 'Confidence Interval'. The model is 85% confident.
    -   It predicts a **Bullish** move.
5.  This seamless experience hides the complex math happening in the background."

## Slide 6: Validation & Challenges (7:00 - 8:30)
**Speaker C:** "Building this wasn't easy.
-   **Latency**: Fetching data took too long, so we implemented caching.
-   **Noise**: Stocks are volatile. We added the 'Confidence Score' so users know when the model is unsure.
-   **Deployment**: Running a stateful bot requires careful error handling to prevent crashes."

## Slide 7: Conclusion & Future Work (8:30 - 9:30)
**Speaker A:** "In conclusion, we successfully deployed a production-grade AI tool. It meets all functional requirements: live predictions, status checks, and a robust architecture. For the future, we plan to add Portfolio Management and explicit Buy/Sell signals. Thank you. We are ready for your questions."

## (9:30 - 10:00) Q&A Setup
