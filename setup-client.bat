@echo off
REM Setup script for Docker Model Runner client dependencies

echo 📦 Installing Docker Model Runner client dependencies...
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.11+ and add it to your PATH
    pause
    exit /b 1
)

echo ✅ Python is available

REM Check if pip is available
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip is not available
    echo Please ensure pip is installed with Python
    pause
    exit /b 1
)

echo ✅ pip is available

REM Install client dependencies
echo.
echo Installing client dependencies...
pip install -r client-requirements.txt

if %errorlevel% eq 0 (
    echo.
    echo ✅ Client dependencies installed successfully!
    echo.
    echo You can now run:
    echo   python client.py --mode test
    echo   python client.py --mode interactive
    echo   python test_llama.py
) else (
    echo.
    echo ❌ Failed to install client dependencies
    echo Please check the error messages above
)

echo.
pause
