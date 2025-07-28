# Docker Model Runner with Llama 3.2 - Quick Start Script

param(
    [switch]$SkipBuild,
    [switch]$SkipTest,
    [int]$WaitTime = 45
)

function Write-Step {
    param([string]$Message, [string]$Color = "Cyan")
    Write-Host "`n� $Message" -ForegroundColor $Color
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

# Main script
try {
    Write-Host "🦙 Docker Model Runner with Llama 3.2 - Quick Start" -ForegroundColor Magenta
    Write-Host "=" * 60 -ForegroundColor Magenta

    # Check prerequisites
    Write-Step "Checking prerequisites..."
    
    # Check Docker
    try {
        $dockerVersion = docker --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Docker is available: $dockerVersion"
        } else {
            throw "Docker not found"
        }
    } catch {
        Write-Error "Docker is not installed or not running"
        Write-Host "Please install Docker Desktop and make sure it's running"
        exit 1
    }

    # Check Python
    try {
        $pythonVersion = python --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Python is available: $pythonVersion"
        } else {
            throw "Python not found"
        }
    } catch {
        Write-Error "Python is not installed or not in PATH"
        Write-Host "Please install Python 3.11+ and add it to your PATH"
        exit 1
    }

    # Clean up any existing container
    Write-Step "Cleaning up existing containers..."
    docker stop model-runner-test 2>$null | Out-Null
    docker rm model-runner-test 2>$null | Out-Null
    Write-Success "Cleanup completed"

    # Build the Docker image
    if (-not $SkipBuild) {
        Write-Step "Building Docker image (this may take several minutes)..."
        $buildResult = docker build -t model-runner . 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Docker image built successfully"
        } else {
            Write-Error "Docker build failed"
            Write-Host $buildResult
            exit 1
        }
    } else {
        Write-Warning "Skipping build (using existing image)"
    }

    # Start the container
    Write-Step "Starting the container..."
    $runResult = docker run -d -p 8000:8000 --name model-runner-test model-runner 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Container started successfully"
        Write-Host "Container ID: $runResult"
    } else {
        Write-Error "Failed to start container"
        Write-Host $runResult
        exit 1
    }

    # Wait for server to start
    Write-Step "Waiting for server to start (Llama model loading may take a few minutes)..."
    Write-Host "⏳ This will take approximately $WaitTime seconds..." -ForegroundColor Yellow
    
    $progressParams = @{
        Activity = "Starting Model Runner"
        Status = "Loading Llama 3.2 model..."
        PercentComplete = 0
    }
    
    for ($i = 0; $i -lt $WaitTime; $i++) {
        $progressParams.PercentComplete = [math]::Round(($i / $WaitTime) * 100)
        $progressParams.Status = "Loading Llama 3.2 model... ($i/$WaitTime seconds)"
        Write-Progress @progressParams
        Start-Sleep 1
    }
    Write-Progress -Activity "Starting Model Runner" -Completed

    # Check if server is responding
    Write-Step "Checking server health..."
    try {
        $healthCheck = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get -TimeoutSec 10
        Write-Success "Server is healthy!"
        Write-Host "   Status: $($healthCheck.status)"
        Write-Host "   Models loaded: $($healthCheck.models_loaded)"
        Write-Host "   Uptime: $([math]::Round($healthCheck.uptime_seconds, 2)) seconds"
    } catch {
        Write-Error "Server health check failed"
        Write-Host "Checking container logs..."
        docker logs model-runner-test --tail 20
        exit 1
    }

    # Install client dependencies if needed
    Write-Step "Installing client dependencies..."
    if (Test-Path "client-requirements.txt") {
        try {
            # Check if aiohttp is already installed
            $aioHttpCheck = python -c "import aiohttp; print('OK')" 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Client dependencies already installed"
            } else {
                Write-Host "Installing missing dependencies..." -ForegroundColor Yellow
                pip install -r client-requirements.txt
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "Client dependencies installed"
                } else {
                    Write-Warning "Failed to install client dependencies"
                    Write-Host "You can install them manually by running:" -ForegroundColor Yellow
                    Write-Host "  pip install -r client-requirements.txt" -ForegroundColor White
                }
            }
        } catch {
            Write-Warning "Failed to install client dependencies, continuing anyway..."
        }
    } else {
        Write-Warning "client-requirements.txt not found, continuing anyway..."
    }

    # Run tests
    if (-not $SkipTest) {
        Write-Step "Running API tests..."
        try {
            python client.py --mode test --url http://localhost:8000
            if ($LASTEXITCODE -eq 0) {
                Write-Success "API tests completed successfully"
            } else {
                Write-Warning "Some API tests may have failed, but continuing..."
            }
        } catch {
            Write-Warning "API tests failed, but continuing..."
        }

        Write-Step "Running Llama-specific tests..."
        try {
            if (Test-Path "test_llama.py") {
                python test_llama.py
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "Llama tests completed successfully"
                } else {
                    Write-Warning "Llama tests may have failed"
                }
            } else {
                Write-Warning "test_llama.py not found, skipping Llama tests"
            }
        } catch {
            Write-Warning "Llama tests failed, but server should still be running"
        }
    } else {
        Write-Warning "Skipping tests"
    }

    # Success message
    Write-Host "`n" + "=" * 60 -ForegroundColor Green
    Write-Host "🎉 SETUP COMPLETE!" -ForegroundColor Green
    Write-Host "=" * 60 -ForegroundColor Green
    Write-Host ""
    Write-Host "🌐 API Documentation: " -NoNewline
    Write-Host "http://localhost:8000/docs" -ForegroundColor Cyan
    Write-Host "💓 Health Check: " -NoNewline  
    Write-Host "http://localhost:8000/health" -ForegroundColor Cyan
    Write-Host "📋 Models List: " -NoNewline
    Write-Host "http://localhost:8000/models" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Quick test commands:" -ForegroundColor Yellow
    Write-Host "  python client.py --mode interactive" -ForegroundColor White
    Write-Host "  python test_llama.py" -ForegroundColor White
    Write-Host ""
    Write-Host "To stop the container:" -ForegroundColor Yellow
    Write-Host "  docker stop model-runner-test" -ForegroundColor White
    Write-Host "  docker rm model-runner-test" -ForegroundColor White

} catch {
    Write-Error "Script failed with error: $($_.Exception.Message)"
    Write-Host "Cleaning up..."
    docker stop model-runner-test 2>$null | Out-Null
    docker rm model-runner-test 2>$null | Out-Null
    exit 1
}

# Usage help
Write-Host "`nScript options:" -ForegroundColor Gray
Write-Host "  -SkipBuild    Skip Docker image building" -ForegroundColor Gray
Write-Host "  -SkipTest     Skip running tests" -ForegroundColor Gray
Write-Host "  -WaitTime N   Wait N seconds for startup (default: 45)" -ForegroundColor Gray
