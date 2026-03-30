@echo off
chcp 65001 >nul
echo ====================================================
echo   야유회 기획 도우미 - EXE 빌드 (tkinter, 소형)
echo   Python: C:\Users\HS1\anaconda3\python.exe
echo ====================================================
set PYTHON=C:\Users\HS1\anaconda3\python.exe

%PYTHON% -m pip show pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo PyInstaller 설치 중...
    %PYTHON% -m pip install pyinstaller
)

echo 빌드 시작...
%PYTHON% -m PyInstaller ^
    --onefile --windowed ^
    --name "야유회기획도우미" ^
    --exclude-module PyQt5 --exclude-module numpy --exclude-module pandas ^
    --exclude-module scipy --exclude-module matplotlib --exclude-module PIL ^
    --exclude-module cv2 --exclude-module sklearn --exclude-module torch ^
    --exclude-module tensorflow --exclude-module IPython ^
    --exclude-module PySide2 --exclude-module PySide6 --exclude-module PyQt6 ^
    retreat_planner_tk.py

if exist "dist\야유회기획도우미.exe" (
    echo ====================================================
    echo   빌드 완료!  dist\야유회기획도우미.exe
    echo ====================================================
    explorer dist
) else (
    echo 빌드 실패.
)
pause
