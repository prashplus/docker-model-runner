#!/bin/bash
# Setup script for Docker Model Runner client dependencies

echo "📦 Installing Docker Model Runner client dependencies..."
echo ""

# Check if Python is available
if command -v python3 &> /dev/null; then
    echo "✅ Python is available: $(python3 --version)"
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    echo "✅ Python is available: $(python --version)"
    PYTHON_CMD="python"
else
    echo "❌ Python is not installed or not in PATH"
    echo "Please install Python 3.11+ and add it to your PATH"
    exit 1
fi

# Check if pip is available
if command -v pip3 &> /dev/null; then
    echo "✅ pip is available: $(pip3 --version)"
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    echo "✅ pip is available: $(pip --version)"
    PIP_CMD="pip"
else
    echo "❌ pip is not available"
    echo "Please ensure pip is installed with Python"
    exit 1
fi

# Install client dependencies
echo ""
echo "Installing client dependencies..."

if $PIP_CMD install -r client-requirements.txt; then
    echo ""
    echo "✅ Client dependencies installed successfully!"
    echo ""
    echo "You can now run:"
    echo "  $PYTHON_CMD client.py --mode test"
    echo "  $PYTHON_CMD client.py --mode interactive" 
    echo "  $PYTHON_CMD test_llama.py"
else
    echo ""
    echo "❌ Failed to install client dependencies"
    echo "Please check the error messages above"
    exit 1
fi

echo ""
