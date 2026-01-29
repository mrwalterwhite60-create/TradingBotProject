@echo off
color 0A
echo ===================================================
echo   ANTIGRAVITY DEPLOYMENT SYSTEM (GITHUB LINK)
echo ===================================================
echo.
echo [1/5] Initializing Repository...
cd /d "%~dp0"
git init

echo [2/5] Adding Files...
git add .

echo [3/5] Committing Code...
git commit -m "Final Deploy: AI Trading Bot v3.0 (Cloud Ready)"

echo [4/5] Linking to Remote Server...
git branch -M main
:: Remove origin if it exists to avoid errors on re-run
git remote remove origin 2>nul
git remote add origin https://github.com/mrwalterwhite60-create/OracleAlgo.git

echo [5/5] PUSHING TO GITHUB (Force Update)...
git push -u origin main --force


echo.
echo ===================================================
echo   DONE! Check your GitHub Repository now.
echo ===================================================
pause
