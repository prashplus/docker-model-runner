# Docker Model Runner with Llama 3.2

Write-Host "🐳 Building Docker Model Runner with Llama 3.2..." -ForegroundColor Green

# Build the Docker image
docker build -t model-runner .

Write-Host "🚀 Starting the container..." -ForegroundColor Green

# Run the container
docker run -d -p 8000:8000 --name model-runner-test model-runner

Write-Host "⏳ Waiting for server to start (Llama model loading may take a few minutes)..." -ForegroundColor Yellow
Start-Sleep 30

Write-Host "🧪 Testing the API..." -ForegroundColor Green

# Test the API
python client.py --mode test

Write-Host "🦙 Quick Llama test..." -ForegroundColor Green
python test_llama.py

Write-Host "✅ Setup complete! Visit http://localhost:8000/docs for API documentation" -ForegroundColor Green
