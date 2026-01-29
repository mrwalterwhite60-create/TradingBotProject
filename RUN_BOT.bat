@echo off
echo [SYSTEM] Launching Antigravity Quant Bot...
set PYTHONPATH=%~dp0
:: Load .env variables (simple parser)
for /f "tokens=1,2 delims==" %%a in (TeamProject2\.env) do (
    if "%%a"=="TELEGRAM_BOT_TOKEN_PROJ2" set TELEGRAM_BOT_TOKEN_PROJ2=%%b
)

python TeamProject2/backend/bot_main.py
pause
