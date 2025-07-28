@echo off
REM Docker Model Runner with Llama 3.2 - Windows Batch Script
echo.
echo 🦙 Docker Model Runner with Llama 3.2 - Quick Start
echo ========================================================

REM Check if Docker is available
echo Checking Docker...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed or not running
    echo Please install Docker Desktop and make sure it's running
    pause
    exit /b 1
)
echo ✅ Docker is available

REM Check if Python is available
echo Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.11+ and add it to your PATH
    pause
    exit /b 1
)
echo ✅ Python is available

REM Clean up existing container
echo.
echo 🧹 Cleaning up existing containers...
docker stop model-runner-test >nul 2>&1
docker rm model-runner-test >nul 2>&1
echo ✅ Cleanup completed

REM Build the Docker image
echo.
echo 🐳 Building Docker image (this may take several minutes)...
docker build -t model-runner .
if %errorlevel% neq 0 (
    echo ❌ Docker build failed
    pause
    exit /b 1
)
echo ✅ Docker image built successfully

REM Start the container
echo.
echo 🚀 Starting the container...
docker run -d -p 8000:8000 --name model-runner-test model-runner
if %errorlevel% neq 0 (
    echo ❌ Failed to start container
    pause
    exit /b 1
)
echo ✅ Container started successfully

REM Wait for server to start
echo.
echo ⏳ Waiting for server to start (Llama model loading may take a few minutes)...
echo This will take approximately 45 seconds...
timeout /t 45 /nobreak >nul

REM Check server health
echo.
echo 🏥 Checking server health...
curl -f http://localhost:8000/health >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Server health check failed
    echo Checking container logs...
    docker logs model-runner-test --tail 20
    pause
    exit /b 1
)
echo ✅ Server is healthy!

REM Install client dependencies
echo.
echo 📦 Installing client dependencies...
if exist client-requirements.txt (
    pip install -r client-requirements.txt -q
    echo ✅ Client dependencies installed
) else (
    echo ⚠️ client-requirements.txt not found, continuing anyway...
)

REM Run tests
echo.
echo 🧪 Running API tests...
python client.py --mode test --url http://localhost:8000
if %errorlevel% neq 0 (
    echo ⚠️ Some API tests may have failed, but continuing...
)

echo.
echo 🦙 Running Llama-specific tests...
if exist test_llama.py (
    python test_llama.py
    if %errorlevel% neq 0 (
        echo ⚠️ Llama tests may have failed
    )
) else (
    echo ⚠️ test_llama.py not found, skipping Llama tests
)

REM Success message
echo.
echo ========================================================
echo 🎉 SETUP COMPLETE!
echo ========================================================
echo.
echo 🌐 API Documentation: http://localhost:8000/docs
echo 💓 Health Check: http://localhost:8000/health
echo 📋 Models List: http://localhost:8000/models
echo.
echo Quick test commands:
echo   python client.py --mode interactive
echo   python test_llama.py
echo.
echo To stop the container:
echo   docker stop model-runner-test
echo   docker rm model-runner-test
echo.
pause
