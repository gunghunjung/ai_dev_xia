@echo off
chcp 65001 > nul
echo ========================================
echo   PyPlayer EXE Build
echo ========================================
echo.
echo [NOTE] VLC Player must be installed.
echo        https://www.videolan.org (64-bit)
echo.
pip install PyQt6 python-vlc pyinstaller -q
python build_exe.py
echo.
echo Done! Run: dist\PyPlayer\PyPlayer.exe
pause
