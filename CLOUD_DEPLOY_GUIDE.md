# ☁️ How to Deploy Your Bot 24/7 (Free & Easy)

To keep your bot running after you turn off your computer, follow these simple steps to host it on **Render.com**.

## Step 1: Create a GitHub Account
If you don't have one, go to [github.com](https://github.com) and sign up. It's where you will store your code.

## Step 2: Upload Your Code (Automated)
I have created a script that handles all technical steps for you.
1. Open the folder: `c:\Users\G\.gemini\antigravity\scratch\chatbot_kaz\TeamProject2\`
2. Double-click the file **`PUSH_TO_GITHUB.bat`**.
3. It will automatically configure Git and push your code. If a browser window opens, just log in.

## Step 3: Connect to Render.com
1. Go to [Render.com](https://dashboard.render.com).
2. Click **New +** and select **Background Worker**.
3. Connect your GitHub account and select your `TradingBotProject` repo.
4. **Settings**:
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r TeamProject2/requirements.txt`
   - **Start Command**: `python TeamProject2/backend/bot_main.py`

## Step 4: Add Environment Variables (IMPORTANT)
Go to the **Environment** tab on Render and add:
- `TELEGRAM_BOT_TOKEN_PROJ2` = (your bot token)
- `GEMINI_API_KEY` = (your gemini key)

**That's it!** Render will run the bot automatically. You can turn off your PC and use the bot from your phone. 🚀
