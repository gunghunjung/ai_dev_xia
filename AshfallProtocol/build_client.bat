@echo off
echo ============================================================
echo   Ashfall Protocol  CLIENT BUILD
echo   dist\AshfallProtocol.exe
echo ============================================================
echo.

python -m PyInstaller --version
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] PyInstaller not found. Installing...
    pip install pyinstaller
)

echo.
echo [1/3] Cleaning...
if exist build\AshfallProtocol rmdir /s /q build\AshfallProtocol
if exist dist\AshfallProtocol.exe del /f /q dist\AshfallProtocol.exe

echo [2/3] Building...
echo.
pyinstaller AshfallProtocol_Client.spec --clean --noconfirm

echo.
if %ERRORLEVEL% NEQ 0 goto :fail
if not exist dist\AshfallProtocol.exe goto :fail

echo [3/3] Done!
echo.
echo ============================================================
echo   SUCCESS ^> dist\AshfallProtocol.exe
echo ============================================================
echo.
pause
exit /b 0

:fail
echo.
echo ============================================================
echo   CLIENT BUILD FAILED
echo ============================================================
echo.
pause
exit /b 1
