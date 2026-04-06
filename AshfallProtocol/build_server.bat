@echo off
echo ============================================================
echo   Ashfall Protocol  SERVER BUILD
echo   dist\AshfallProtocol_Server.exe
echo ============================================================
echo.

python -m PyInstaller --version
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] PyInstaller not found. Installing...
    pip install pyinstaller
)

echo.
echo [1/3] Cleaning...
if exist build\AshfallProtocol_Server rmdir /s /q build\AshfallProtocol_Server
if exist dist\AshfallProtocol_Server.exe del /f /q dist\AshfallProtocol_Server.exe

echo [2/3] Building...
echo.
pyinstaller AshfallProtocol_Server.spec --clean --noconfirm

echo.
if %ERRORLEVEL% NEQ 0 goto :fail
if not exist dist\AshfallProtocol_Server.exe goto :fail

echo [3/3] Done!
echo.
echo ============================================================
echo   SUCCESS ^> dist\AshfallProtocol_Server.exe
echo ============================================================
echo.
pause
exit /b 0

:fail
echo.
echo ============================================================
echo   SERVER BUILD FAILED
echo ============================================================
echo.
pause
exit /b 1
