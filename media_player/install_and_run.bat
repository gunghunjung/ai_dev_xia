@echo off
chcp 65001 > nul
echo ========================================
echo   PyPlayer - Install and Run
echo ========================================
echo.
echo [1/2] Installing packages...
pip install PyQt6 python-vlc -q
if errorlevel 1 (
    echo [ERROR] Package install failed. Check internet connection.
    pause
    exit /b 1
)
echo [2/2] Starting PyPlayer...
echo.
python main.py %*
pause
