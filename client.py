"""
Docker Model Runner Client - Test script to interact with the model server
"""

import asyncio
import aiohttp
import json
import time
import numpy as np
from typing import List, Dict, Any, Union
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRunnerClient:
    """Client for interacting with Docker Model Runner API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check server health"""
        async with self.session.get(f"{self.base_url}/health") as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Health check failed: {response.status}")
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models"""
        async with self.session.get(f"{self.base_url}/models") as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Failed to list models: {response.status}")
    
    async def predict(self, data: Union[List[List[float]], str], model_name: str = "default") -> Dict[str, Any]:
        """Make prediction"""
        payload = {
            "data": data,
            "model_name": model_name
        }
        
        async with self.session.post(
            f"{self.base_url}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Prediction failed: {response.status} - {error_text}")
    
    async def generate_text(self, prompt: str, model_name: str = "llama3.2", max_tokens: int = 256, temperature: float = 0.7) -> Dict[str, Any]:
        """Generate text using Llama model"""
        payload = {
            "prompt": prompt,
            "model_name": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        async with self.session.post(
            f"{self.base_url}/generate",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Text generation failed: {response.status} - {error_text}")
    
    async def load_model(self, model_name: str) -> Dict[str, Any]:
        """Load a model"""
        async with self.session.post(f"{self.base_url}/models/{model_name}/load") as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Model loading failed: {response.status} - {error_text}")
    
    async def unload_model(self, model_name: str) -> Dict[str, Any]:
        """Unload a model"""
        async with self.session.delete(f"{self.base_url}/models/{model_name}") as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Model unloading failed: {response.status} - {error_text}")

async def test_basic_functionality(client: ModelRunnerClient):
    """Test basic server functionality"""
    logger.info("=" * 50)
    logger.info("Testing Basic Functionality")
    logger.info("=" * 50)
    
    # Health check
    logger.info("1. Checking server health...")
    health = await client.health_check()
    logger.info(f"   Server status: {health['status']}")
    logger.info(f"   Models loaded: {health['models_loaded']}")
    logger.info(f"   Uptime: {health['uptime_seconds']:.2f} seconds")
    
    # List models
    logger.info("\n2. Listing available models...")
    models = await client.list_models()
    for model in models:
        logger.info(f"   - {model['name']} ({model['type']}) - Loaded: {model['loaded']}")

async def test_predictions(client: ModelRunnerClient):
    """Test prediction functionality"""
    logger.info("\n" + "=" * 50)
    logger.info("Testing Predictions")
    logger.info("=" * 50)
    
    # Generate test data
    np.random.seed(42)
    test_data = np.random.randn(5, 4).tolist()  # 5 samples, 4 features
    
    logger.info(f"Making prediction with test data: {len(test_data)} samples")
    
    start_time = time.time()
    result = await client.predict(test_data)
    end_time = time.time()
    
    logger.info(f"Prediction completed in {(end_time - start_time) * 1000:.2f} ms")
    logger.info(f"Server processing time: {result['processing_time_ms']:.2f} ms")
    logger.info(f"Model used: {result['model_name']}")
    logger.info(f"Predictions: {result['predictions']}")

async def test_model_management(client: ModelRunnerClient):
    """Test model loading/unloading"""
    logger.info("\n" + "=" * 50)
    logger.info("Testing Model Management")
    logger.info("=" * 50)
    
    # Try to unload default model
    logger.info("1. Unloading default model...")
    try:
        result = await client.unload_model("default")
        logger.info(f"   {result['message']}")
    except Exception as e:
        logger.error(f"   Failed: {e}")
    
    # List models after unloading
    logger.info("\n2. Listing models after unloading...")
    models = await client.list_models()
    for model in models:
        logger.info(f"   - {model['name']} - Loaded: {model['loaded']}")
    
    # Try to reload model
    logger.info("\n3. Reloading default model...")
    try:
        result = await client.load_model("default")
        logger.info(f"   {result['message']}")
    except Exception as e:
        logger.error(f"   Failed: {e}")

async def test_performance(client: ModelRunnerClient, num_requests: int = 10):
    """Test performance with multiple requests"""
    logger.info("\n" + "=" * 50)
    logger.info(f"Testing Performance ({num_requests} requests)")
    logger.info("=" * 50)
    
    # Generate test data
    np.random.seed(42)
    test_data = np.random.randn(10, 4).tolist()  # 10 samples, 4 features
    
    times = []
    
    for i in range(num_requests):
        start_time = time.time()
        try:
            result = await client.predict(test_data)
            end_time = time.time()
            request_time = (end_time - start_time) * 1000
            times.append(request_time)
            
            if i % 5 == 0:
                logger.info(f"Request {i+1}: {request_time:.2f} ms")
                
        except Exception as e:
            logger.error(f"Request {i+1} failed: {e}")
    
    if times:
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        logger.info(f"\nPerformance Summary:")
        logger.info(f"   Average time: {avg_time:.2f} ms")
        logger.info(f"   Min time: {min_time:.2f} ms")
        logger.info(f"   Max time: {max_time:.2f} ms")
        logger.info(f"   Successful requests: {len(times)}/{num_requests}")

async def test_llama_generation(client: ModelRunnerClient):
    """Test Llama text generation functionality"""
    logger.info("\n" + "=" * 50)
    logger.info("Testing Llama Text Generation")
    logger.info("=" * 50)
    
    # Check if Llama model is available
    models = await client.list_models()
    llama_model = next((model for model in models if model['name'] == 'llama3.2'), None)
    
    if not llama_model:
        logger.warning("Llama 3.2 model not found in available models")
        logger.info("Available models:", [m['name'] for m in models])
        
        # Try to load the model
        logger.info("Attempting to load Llama 3.2 model...")
        try:
            await client.load_model("llama3.2")
            logger.info("Llama 3.2 model load request sent, checking again...")
            
            # Wait a bit for loading
            await asyncio.sleep(5)
            models = await client.list_models()
            llama_model = next((model for model in models if model['name'] == 'llama3.2'), None)
            
        except Exception as e:
            logger.error(f"Failed to load Llama model: {e}")
    
    if not llama_model or not llama_model.get('loaded', False):
        logger.warning("Llama 3.2 model not available or not loaded, skipping text generation tests")
        logger.info("This might be due to:")
        logger.info("  - Insufficient memory (Llama needs ~4-6GB RAM)")
        logger.info("  - Model still downloading/loading (first time can take 5-10 minutes)")
        logger.info("  - Container resource limits")
        return
    
    test_prompts = [
        "What is artificial intelligence?",
        "Explain machine learning in simple terms.",
        "Write a short poem about technology.",
        "How does Docker work?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        logger.info(f"{i}. Testing prompt: '{prompt[:50]}...'")
        
        start_time = time.time()
        try:
            result = await client.generate_text(prompt, model_name="llama3.2")
            end_time = time.time()
            
            logger.info(f"   Generated in {(end_time - start_time) * 1000:.2f} ms")
            logger.info(f"   Server processing: {result['processing_time_ms']:.2f} ms")
            logger.info(f"   Response: {result['generated_text'][:100]}...")
            
        except Exception as e:
            logger.error(f"   Failed: {e}")

async def test_mixed_predictions(client: ModelRunnerClient):
    """Test both traditional ML and Llama predictions"""
    logger.info("\n" + "=" * 50)
    logger.info("Testing Mixed Model Predictions")
    logger.info("=" * 50)
    
    # Test traditional ML model
    logger.info("1. Testing traditional ML model...")
    np.random.seed(42)
    ml_data = np.random.randn(3, 4).tolist()
    
    try:
        result = await client.predict(ml_data, model_name="default")
        logger.info(f"   ML Predictions: {result['predictions']}")
    except Exception as e:
        logger.error(f"   ML Prediction failed: {e}")
    
    # Test Llama model
    logger.info("\n2. Testing Llama model...")
    try:
        result = await client.predict(
            "Explain the concept of containerization in one sentence.",
            model_name="llama3.2"
        )
        logger.info(f"   Llama Response: {result['predictions'][:100]}...")
    except Exception as e:
        logger.error(f"   Llama Prediction failed: {e}")

async def test_error_handling(client: ModelRunnerClient):
    """Test error handling"""
    logger.info("\n" + "=" * 50)
    logger.info("Testing Error Handling")
    logger.info("=" * 50)
    
    # Test with invalid model
    logger.info("1. Testing with non-existent model...")
    try:
        result = await client.predict([[1, 2, 3, 4]], model_name="nonexistent")
        logger.warning(f"   Unexpected success: {result}")
    except Exception as e:
        logger.info(f"   Expected error caught: {e}")
    
    # Test with invalid data format
    logger.info("\n2. Testing with empty data...")
    try:
        result = await client.predict([])
        logger.warning(f"   Unexpected success: {result}")
    except Exception as e:
        logger.info(f"   Expected error caught: {e}")

async def run_all_tests(base_url: str):
    """Run all tests"""
    logger.info(f"Starting tests for Docker Model Runner at {base_url}")
    
    async with ModelRunnerClient(base_url) as client:
        try:
            await test_basic_functionality(client)
            await test_predictions(client)
            await test_llama_generation(client)
            await test_mixed_predictions(client)
            await test_model_management(client)
            await test_performance(client, num_requests=10)
            await test_error_handling(client)
            
            logger.info("\n" + "=" * 50)
            logger.info("All tests completed successfully!")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise

async def interactive_mode(base_url: str):
    """Interactive mode for manual testing"""
    logger.info("Starting interactive mode...")
    logger.info("Commands: health, models, predict, generate <prompt>, llama, load <model>, unload <model>, quit")
    
    async with ModelRunnerClient(base_url) as client:
        while True:
            try:
                command = input("\nEnter command: ").strip().lower()
                
                if command == "quit":
                    break
                elif command == "health":
                    health = await client.health_check()
                    print(json.dumps(health, indent=2))
                elif command == "models":
                    models = await client.list_models()
                    print(json.dumps(models, indent=2))
                elif command == "predict":
                    # Use sample data
                    data = [[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]]
                    result = await client.predict(data)
                    print(json.dumps(result, indent=2))
                elif command.startswith("generate "):
                    prompt = command.split(" ", 1)[1]
                    result = await client.generate_text(prompt)
                    print(json.dumps(result, indent=2))
                elif command == "llama":
                    # Quick Llama test
                    result = await client.generate_text("Hello! How are you?")
                    print(json.dumps(result, indent=2))
                elif command.startswith("load "):
                    model_name = command.split(" ", 1)[1]
                    result = await client.load_model(model_name)
                    print(json.dumps(result, indent=2))
                elif command.startswith("unload "):
                    model_name = command.split(" ", 1)[1]
                    result = await client.unload_model(model_name)
                    print(json.dumps(result, indent=2))
                else:
                    print("Invalid command. Available: health, models, predict, generate <prompt>, llama, load <model>, unload <model>, quit")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Command failed: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Docker Model Runner Client")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the model server (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--mode",
        choices=["test", "interactive"],
        default="test",
        help="Run mode: test (automated tests) or interactive (manual commands)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "test":
        asyncio.run(run_all_tests(args.url))
    else:
        asyncio.run(interactive_mode(args.url))

if __name__ == "__main__":
    main()
