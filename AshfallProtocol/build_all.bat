@echo off
echo ============================================================
echo   Ashfall Protocol  FULL BUILD  (Client + Server)
echo ============================================================
echo.

python -m PyInstaller --version
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] PyInstaller not found. Installing...
    pip install pyinstaller
)

echo.
echo === STEP 1/2 : CLIENT =====================================
echo.
if exist build\AshfallProtocol rmdir /s /q build\AshfallProtocol
if exist dist\AshfallProtocol.exe del /f /q dist\AshfallProtocol.exe

pyinstaller AshfallProtocol_Client.spec --clean --noconfirm
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Client build failed.
    goto :fail
)
if not exist dist\AshfallProtocol.exe (
    echo [ERROR] Client exe missing.
    goto :fail
)
echo [OK] Client done.

echo.
echo === STEP 2/2 : SERVER =====================================
echo.
if exist build\AshfallProtocol_Server rmdir /s /q build\AshfallProtocol_Server
if exist dist\AshfallProtocol_Server.exe del /f /q dist\AshfallProtocol_Server.exe

pyinstaller AshfallProtocol_Server.spec --clean --noconfirm
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Server build failed.
    goto :fail
)
if not exist dist\AshfallProtocol_Server.exe (
    echo [ERROR] Server exe missing.
    goto :fail
)
echo [OK] Server done.

echo.
echo ============================================================
echo   ALL BUILDS SUCCESSFUL
echo   dist\AshfallProtocol.exe         - Game client
echo   dist\AshfallProtocol_Server.exe  - Server manager
echo   No Python needed to run either exe.
echo ============================================================
echo.
pause
exit /b 0

:fail
echo.
echo ============================================================
echo   BUILD FAILED - see errors above
echo ============================================================
echo.
pause
exit /b 1
