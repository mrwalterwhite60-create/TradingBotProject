# ☁️ How to Deploy Your Bot 24/7 (Free & Easy)

To keep your bot running after you turn off your computer, follow these simple steps to host it on **Render.com**.

## Step 1: Create a GitHub Account
If you don't have one, go to [github.com](https://github.com) and sign up. It's where you will store your code.

## Step 2: Upload Your Code (Automated)
I have created a script that handles all technical steps for you.
1. Open the folder: `c:\Users\G\.gemini\antigravity\scratch\chatbot_kaz\TeamProject2\`
2. Double-click the file **`PUSH_TO_GITHUB.bat`**.
3. It will automatically configure Git and push your code. If a browser window opens, just log in.

## Step 3: Connect to Render.com (FREE Tier)
1. Go to [Render.com](https://dashboard.render.com).
2. Click **New +** and select **Web Service** (NOT Background Worker).
3. Connect your GitHub account and select your `TradingBotProject` repo.
4. **Settings**:
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python backend/bot_main.py`
   - **Instance Type**: Select **Free** ($0/month).

## Step 4: Add Environment Variables (IMPORTANT)
Go to the **Environment** tab on Render and add:
- `TELEGRAM_BOT_TOKEN_PROJ2` = (your bot token)
- `GEMINI_API_KEY` = (your gemini key)

**Done!** Render will give you a URL (e.g., `trading-bot.onrender.com`). Your bot is now live. 🚀
> [!TIP]
> On the Free tier, Render puts the bot to sleep after 15 minutes of inactivity. Don't worry! If you want to use it for the defense, just open your bot's Render URL in a browser — it will wake up instantly.
