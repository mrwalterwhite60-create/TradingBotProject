@echo off
setlocal
echo ==========================================
echo   Trading Bot GitHub Deployment Helper
echo ==========================================

:: 1. Navigate to the project directory
cd /d "%~dp0"

echo [SYSTEM] Working directory: %CD%

:: 2. Check for index.lock and remove if exists
if exist ".git\index.lock" (
    echo [FIX] Removing git lock file...
    del ".git\index.lock"
)

:: 3. Initialize Git if not already done
if not exist ".git" (
    echo [GIT] Initializing new repository...
    git init
)

:: 4. Set Identity if not set
git config user.email >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [GIT] Setting default identity...
    git config user.email "student@example.com"
    git config user.name "AI Student"
)

:: 5. Add files
echo [GIT] Adding files...
git add .

:: 6. Commit
echo [GIT] Committing changes...
git commit -m "Deploy Trading Bot"

:: 6. Set branch to main
git branch -M main

:: 7. Handle remote
:: Try to add, if fails (exists), set instead
echo [GIT] Setting remote URL...
git remote add origin https://github.com/mrwalterwhite60-create/TradingBotProject.git 2>nul
if %ERRORLEVEL% neq 0 (
    git remote set-url origin https://github.com/mrwalterwhite60-create/TradingBotProject.git
)

:: 8. Push
echo [GIT] Pushing to GitHub...
echo [!] A browser window may open for login.
git push -u origin main

echo ==========================================
echo   Done! Check your GitHub repository.
echo ==========================================
pause
