# Setup script for Docker Model Runner client dependencies

Write-Host "📦 Installing Docker Model Runner client dependencies..." -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Python is available: $pythonVersion" -ForegroundColor Green
    } else {
        throw "Python not found"
    }
} catch {
    Write-Host "❌ Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.11+ and add it to your PATH" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if pip is available
try {
    $pipVersion = pip --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ pip is available: $pipVersion" -ForegroundColor Green
    } else {
        throw "pip not found"
    }
} catch {
    Write-Host "❌ pip is not available" -ForegroundColor Red
    Write-Host "Please ensure pip is installed with Python" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Install client dependencies
Write-Host ""
Write-Host "Installing client dependencies..." -ForegroundColor Yellow

try {
    pip install -r client-requirements.txt
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ Client dependencies installed successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "You can now run:" -ForegroundColor Cyan
        Write-Host "  python client.py --mode test" -ForegroundColor White
        Write-Host "  python client.py --mode interactive" -ForegroundColor White
        Write-Host "  python test_llama.py" -ForegroundColor White
    } else {
        throw "pip install failed"
    }
} catch {
    Write-Host ""
    Write-Host "❌ Failed to install client dependencies" -ForegroundColor Red
    Write-Host "Please check the error messages above" -ForegroundColor Yellow
}

Write-Host ""
Read-Host "Press Enter to continue"
