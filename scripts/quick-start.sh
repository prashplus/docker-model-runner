#!/bin/bash
# Docker Model Runner with Llama 3.2

echo "🐳 Building Docker Model Runner with Llama 3.2..."

# Build the Docker image
docker build -t model-runner .

echo "🚀 Starting the container..."

# Run the container in background
docker run -d -p 8000:8000 --name model-runner-test model-runner

echo "⏳ Waiting for server to start (Llama model loading may take a few minutes)..."
sleep 30

echo "🧪 Testing the API..."

# Test the API
python client.py --mode test

echo "🦙 Quick Llama test..."
python test_llama.py

echo "✅ Setup complete! Visit http://localhost:8000/docs for API documentation"
