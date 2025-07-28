# Docker Model Runner

A containerized machine learning model serving platform that demonstrates how to deploy and manage ML models using Docker. This project provides a FastAPI-based REST API for model inference, with support for multiple models including **Llama 3.2**, health monitoring, and easy deployment.

## Features

- 🐳 **Docker-based deployment** - Easy containerization and deployment
- 🚀 **FastAPI REST API** - High-performance async API server
- 🦙 **Llama 3.2 Integration** - Support for Meta's latest Llama 3.2 1B Instruct model
- 🔄 **Dynamic model management** - Load/unload models at runtime
- 📊 **Health monitoring** - Built-in health checks and metrics
- 🧪 **Demo models included** - Ready-to-use classification model + Llama
- 🔧 **Client library** - Python client for easy testing and integration
- 📈 **Performance testing** - Built-in performance benchmarking
- 🛡️ **Security** - Non-root container execution
- 🔍 **CORS support** - Cross-origin resource sharing enabled
- 🤖 **Mixed model support** - Traditional ML models + Large Language Models

## Quick Start

### Prerequisites

- Docker and Docker Compose (with at least 8GB RAM available)
- Python 3.11+ (for client testing)
- **Note**: Llama 3.2 models require significant memory. Ensure your system has at least 8GB RAM.

### 1. Clone the Repository

```bash
git clone https://github.com/prashplus/docker-model-runner.git
cd docker-model-runner
```

### 2. Build and Run with Docker Compose

```bash
# Build and start the service
docker-compose up --build

# Or run in background
docker-compose up --build -d
```

### 3. Test the API

Install client dependencies:
```bash
pip install -r client-requirements.txt
```

Run automated tests:
```bash
python client.py --mode test
```

Or try interactive mode:
```bash
python client.py --mode interactive
```

## Manual Docker Setup

### Build the Image

```bash
docker build -t model-runner .
```

### Run the Container

```bash
# Basic run
docker run -p 8000:8000 model-runner

# With volume mounting for persistent models
docker run -p 8000:8000 -v $(pwd)/models:/app/models model-runner

# With environment variables
docker run -p 8000:8000 -e HOST=0.0.0.0 -e PORT=8000 model-runner
```

## API Documentation

Once the server is running, visit:
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Model List**: http://localhost:8000/models

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check with metrics |
| GET | `/models` | List all available models |
| POST | `/predict` | Make predictions (ML models) or generate text (Llama) |
| POST | `/generate` | Generate text using Llama models |
| POST | `/models/{name}/load` | Load a specific model |
| DELETE | `/models/{name}` | Unload a model |
| POST | `/models/upload` | Upload a new model file |

### Example API Usage

#### Make a Traditional ML Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "data": [[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]],
       "model_name": "default"
     }'
```

#### Generate Text with Llama 3.2

```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Explain artificial intelligence in simple terms",
       "model_name": "llama3.2",
       "max_tokens": 150,
       "temperature": 0.7
     }'
```

#### Make Llama Prediction via Predict Endpoint

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "data": "What is Docker and how does it work?",
       "model_name": "llama3.2"
     }'
```

#### Check Health

```bash
curl http://localhost:8000/health
```

#### List Models

```bash
curl http://localhost:8000/models
```

## Client Usage

The included Python client provides easy programmatic access to the API.

### Basic Usage

```python
import asyncio
from client import ModelRunnerClient

async def main():
    async with ModelRunnerClient("http://localhost:8000") as client:
        # Check health
        health = await client.health_check()
        print(f"Server status: {health['status']}")
        
        # Make traditional ML prediction
        data = [[1.0, 2.0, 3.0, 4.0]]
        result = await client.predict(data)
        print(f"ML Prediction: {result['predictions']}")
        
        # Generate text with Llama
        text_result = await client.generate_text("What is machine learning?")
        print(f"Llama Response: {text_result['generated_text']}")

asyncio.run(main())
```

### Command Line Usage

```bash
# Run automated tests
python client.py --mode test --url http://localhost:8000

# Interactive mode
python client.py --mode interactive --url http://localhost:8000
```

## Working with Llama 3.2

### Model Information

This project includes Meta's Llama 3.2 1B Instruct model, which provides:
- **1 billion parameters** - Compact yet powerful
- **Instruction following** - Optimized for chat and instruction tasks
- **Efficient inference** - Suitable for containerized deployment
- **Broad language support** - Multilingual capabilities

### Text Generation Examples

#### Using the Generate Endpoint

```bash
# Simple question answering
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Explain Docker containers in 2 sentences",
       "max_tokens": 100,
       "temperature": 0.5
     }'

# Creative writing
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Write a haiku about machine learning",
       "max_tokens": 50,
       "temperature": 0.8
     }'
```

#### Using Python Client

```python
async with ModelRunnerClient() as client:
    # Technical explanation
    result = await client.generate_text(
        "How do neural networks learn?",
        max_tokens=200,
        temperature=0.6
    )
    print(result['generated_text'])
    
    # Code explanation
    result = await client.generate_text(
        "Explain this Python code: def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        max_tokens=150
    )
    print(result['generated_text'])
```

## Adding Custom Models

### Traditional ML Models

### 1. Prepare Your Model

Save your trained scikit-learn model as a pickle file:

```python
import pickle
from sklearn.ensemble import RandomForestClassifier

# Train your model
model = RandomForestClassifier()
# ... training code ...

# Save the model
with open('my_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### 2. Add to Container

#### Option A: Build-time inclusion
Place your `.pkl` files in the `models/` directory before building:

```bash
cp my_model.pkl models/
docker-compose up --build
```

#### Option B: Runtime upload
Use the upload endpoint:

```bash
curl -X POST "http://localhost:8000/models/upload" \
     -F "file=@my_model.pkl"
```

#### Option C: Volume mounting
Mount your models directory:

```bash
docker run -p 8000:8000 -v /path/to/your/models:/app/models model-runner
```

### 3. Load and Use

```bash
# Load the model
curl -X POST "http://localhost:8000/models/my_model/load"

# Make predictions
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"data": [[1,2,3,4]], "model_name": "my_model"}'
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |

### Docker Compose Configuration

Modify `docker-compose.yml` to customize:

- Port mappings
- Volume mounts
- Environment variables
- Resource limits
- Network settings

Example with resource limits:

```yaml
services:
  model-runner:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

## Development

### Project Structure

```
docker-model-runner/
├── src/
│   ├── server.py           # FastAPI application
│   └── model_manager.py    # Model management logic
├── models/                 # Model storage directory
├── scripts/               # Helper scripts
│   ├── quick-start.sh     # Linux/Mac quick start
│   └── quick-start.ps1    # Windows quick start
├── client.py              # Python client library
├── Dockerfile             # Container definition
├── docker-compose.yml     # Multi-container setup
├── requirements.txt       # Python dependencies
├── client-requirements.txt # Client dependencies
└── README.md              # This file
```

### Running in Development Mode

For development with auto-reload:

```bash
# Install dependencies locally
pip install -r requirements.txt

# Run with uvicorn directly
cd src
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### Adding New Model Types

The current implementation supports scikit-learn models and Llama models. To add support for other frameworks:

1. Extend `ModelManager` class in `src/model_manager.py`
2. Add framework-specific loading logic
3. Handle different prediction methods
4. Update model info metadata

Example for TensorFlow:

```python
# In model_manager.py
import tensorflow as tf

async def load_tensorflow_model(self, model_path: str, model_name: str):
    model = tf.keras.models.load_model(model_path)
    self.models[model_name] = model
    # ... update model_info
```

Example for other Hugging Face models:

```python
# In model_manager.py
async def load_custom_transformer(self, model_name: str, hf_model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(hf_model_name)
    
    self.models[model_name] = model
    self.tokenizers[model_name] = tokenizer
    # ... update model_info
```

## Testing

### Automated Testing

The client includes comprehensive tests:

```bash
python client.py --mode test
```

Test categories:
- Basic functionality (health, model listing)
- Traditional ML prediction accuracy and performance
- Llama text generation capabilities
- Mixed model testing (ML + LLM)
- Model management (load/unload)
- Error handling
- Performance benchmarking

### Manual Testing

Use interactive mode for manual exploration:

```bash
python client.py --mode interactive
```

Available commands:
- `health` - Check server health
- `models` - List available models
- `predict` - Make a prediction with sample data
- `generate <prompt>` - Generate text with Llama
- `llama` - Quick Llama test
- `load <model>` - Load a specific model
- `unload <model>` - Unload a model
- `quit` - Exit interactive mode

### Performance Testing

The client includes performance benchmarking:

```python
# Run 100 prediction requests
python -c "
import asyncio
from client import ModelRunnerClient

async def test():
    async with ModelRunnerClient() as client:
        await test_performance(client, num_requests=100)

asyncio.run(test())
"
```

## Monitoring and Observability

### Health Checks

The service includes built-in health checks:

```bash
# Docker health check
docker ps  # Shows health status

# Manual health check
curl http://localhost:8000/health
```

Health response includes:
- Service status
- Number of loaded models
- Uptime in seconds
- Timestamp

### Logging

Logs are written to stdout and can be viewed with:

```bash
# Docker Compose logs
docker-compose logs -f

# Docker logs
docker logs <container_id> -f
```

Log levels can be configured via environment variables.

### Metrics

Basic metrics are available through the health endpoint. For production monitoring, consider integrating:

- Prometheus metrics
- Application Performance Monitoring (APM)
- Custom logging solutions

## Production Deployment

### Security Considerations

1. **Non-root execution**: Container runs as non-root user
2. **Resource limits**: Set appropriate CPU/memory limits
3. **Network security**: Use proper firewall rules
4. **Model validation**: Validate uploaded models
5. **Input sanitization**: Validate API inputs

### Scaling

#### Horizontal Scaling

```yaml
# docker-compose.yml
services:
  model-runner:
    # ... configuration ...
    deploy:
      replicas: 3
```

#### Load Balancing

Use a reverse proxy like nginx:

```nginx
upstream model_runners {
    server localhost:8001;
    server localhost:8002;
    server localhost:8003;
}

server {
    listen 80;
    location / {
        proxy_pass http://model_runners;
    }
}
```

### Persistent Storage

For production, use persistent volumes:

```yaml
services:
  model-runner:
    volumes:
      - model_data:/app/models
      - log_data:/app/logs

volumes:
  model_data:
  log_data:
```

## Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Kill the process or use different port
docker run -p 8001:8000 model-runner
```

#### Model Loading Failures
```bash
# Check model files
docker exec -it <container> ls -la /app/models/

# Check logs
docker logs <container>

# Validate model format
python -c "import pickle; pickle.load(open('model.pkl', 'rb'))"
```

#### Memory Issues
```bash
# Check container resource usage
docker stats

# Increase memory limits for Llama models
docker run -m 8g model-runner

# For Docker Compose, edit docker-compose.yml:
# deploy:
#   resources:
#     limits:
#       memory: 8G
```

#### Llama Model Loading Issues
```bash
# Check if model is downloading
docker logs <container> | grep -i llama

# Check available disk space for model cache
docker exec -it <container> df -h /app/cache

# Manual model loading test
docker exec -it <container> python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')
print('Llama model accessible')
"
```

### Debug Mode

Enable debug logging:

```bash
docker run -e LOG_LEVEL=DEBUG -p 8000:8000 model-runner
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Ensure all tests pass: `python client.py --mode test`
5. Submit a pull request

### Development Setup

```bash
# Clone and setup
git clone <your-fork>
cd docker-model-runner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r client-requirements.txt

# Run tests
python client.py --mode test
```

## License

This project is open source. Please check the LICENSE file for details.

**Note on Llama 3.2**: The Llama 3.2 model is subject to Meta's custom license. Please review the [Llama 2 Community License Agreement](https://ai.meta.com/llama/license/) before commercial use.

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation at `/docs`
- For Llama-specific issues, check the model loading logs

## Performance Notes

### Resource Requirements
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended for Llama**: 8GB RAM, 4 CPU cores
- **GPU**: Optional but recommended for Llama inference
- **Storage**: ~2-3GB for Llama model cache

### First-time Setup
1. First container start may take 5-10 minutes (Llama model download)
2. Model files are cached in Docker volume for subsequent runs
3. Use `docker logs <container>` to monitor loading progress

### Quick Test Commands
```bash
# Quick Llama test
python test_llama.py

# Interactive mode with Llama commands
python client.py --mode interactive
# Then try: llama, generate Hello!, etc.

# Full test suite
python client.py --mode test
```

---

**Happy Model Serving with Llama 3.2! 🚀🐳🦙**

## Authors

* **Prashant Piprotar** - - [Prash+](https://github.com/prashplus)

Visit my blog for more Tech Stuff
### http://prashplus.blogspot.com
