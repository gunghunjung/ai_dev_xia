@echo off
echo ========================================
echo  Ashfall Protocol - Build Executable
echo ========================================

pyinstaller AshfallProtocol.spec --clean

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Build successful!
    echo Executable: dist\AshfallProtocol.exe
) else (
    echo.
    echo Build failed. Check error messages above.
)
pause
