#!/usr/bin/env python3
"""
Python-based quick start script for Docker Model Runner
Alternative to PowerShell/Batch scripts
"""

import subprocess
import sys
import time
import requests
import os

def run_command(cmd, shell=True, check=True):
    """Run a command and return the result"""
    try:
        result = subprocess.run(cmd, shell=shell, check=check, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def print_step(message, color="cyan"):
    """Print a step message"""
    colors = {
        "green": "\033[92m",
        "red": "\033[91m", 
        "yellow": "\033[93m",
        "cyan": "\033[96m",
        "reset": "\033[0m"
    }
    print(f"\n🔹 {colors.get(color, '')}{message}{colors['reset']}")

def print_success(message):
    print(f"✅ \033[92m{message}\033[0m")

def print_error(message):
    print(f"❌ \033[91m{message}\033[0m")

def print_warning(message):
    print(f"⚠️  \033[93m{message}\033[0m")

def main():
    print("\033[95m🦙 Docker Model Runner with Llama 3.2 - Python Quick Start\033[0m")
    print("=" * 65)
    
    # Check prerequisites
    print_step("Checking prerequisites...")
    
    # Check Docker
    success, stdout, stderr = run_command("docker --version")
    if success:
        print_success(f"Docker is available: {stdout.strip()}")
    else:
        print_error("Docker is not installed or not running")
        print("Please install Docker Desktop and make sure it's running")
        return 1
    
    # Check Python
    print_success(f"Python is available: {sys.version}")
    
    # Clean up existing containers
    print_step("Cleaning up existing containers...")
    run_command("docker stop model-runner-test", check=False)
    run_command("docker rm model-runner-test", check=False)
    print_success("Cleanup completed")
    
    # Install client dependencies
    print_step("Installing client dependencies...")
    success, stdout, stderr = run_command("pip install -r client-requirements.txt")
    if success:
        print_success("Client dependencies installed")
    else:
        print_warning("Failed to install client dependencies, continuing anyway...")
        print(f"Error: {stderr}")
    
    # Build Docker image
    print_step("Building Docker image (this may take several minutes)...")
    success, stdout, stderr = run_command("docker build -t model-runner .")
    if success:
        print_success("Docker image built successfully")
    else:
        print_error("Docker build failed")
        print(f"Error: {stderr}")
        return 1
    
    # Start container
    print_step("Starting the container...")
    success, stdout, stderr = run_command("docker run -d -p 8000:8000 --name model-runner-test model-runner")
    if success:
        print_success("Container started successfully")
        print(f"Container ID: {stdout.strip()}")
    else:
        print_error("Failed to start container")
        print(f"Error: {stderr}")
        return 1
    
    # Wait for server to start
    print_step("Waiting for server to start (Llama model loading may take a few minutes)...")
    print("⏳ This will take approximately 45 seconds...")
    
    for i in range(45):
        if i % 10 == 0:
            print(f"   Waiting... {i}/45 seconds")
        time.sleep(1)
    
    # Check server health and load Llama model if needed
    print_step("Checking server health and models...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print_success("Server is healthy!")
            print(f"   Status: {health_data['status']}")
            print(f"   Models loaded: {health_data['models_loaded']}")
            print(f"   Uptime: {health_data['uptime_seconds']:.2f} seconds")
            
            # Check if Llama model is loaded
            models_response = requests.get("http://localhost:8000/models", timeout=10)
            if models_response.status_code == 200:
                models = models_response.json()
                llama_model = next((m for m in models if m['name'] == 'llama3.2'), None)
                
                if not llama_model or not llama_model.get('loaded', False):
                    print_warning("Llama 3.2 model not loaded yet, attempting to load...")
                    try:
                        load_response = requests.post("http://localhost:8000/models/llama3.2/load", timeout=60)
                        if load_response.status_code == 200:
                            print_success("Llama 3.2 model loaded successfully!")
                        else:
                            print_warning("Failed to load Llama 3.2 model - may need more time or memory")
                    except Exception as load_e:
                        print_warning(f"Failed to load Llama 3.2 model: {str(load_e)}")
                else:
                    print_success("Llama 3.2 model is already loaded!")
            
        else:
            print_error(f"Server health check failed: {response.status_code}")
            return 1
    except Exception as e:
        print_error(f"Server health check failed: {str(e)}")
        print("Checking container logs...")
        success, stdout, stderr = run_command("docker logs model-runner-test --tail 20")
        if success:
            print(stdout)
        return 1
    
    # Run tests
    print_step("Running API tests...")
    success, stdout, stderr = run_command("python client.py --mode test --url http://localhost:8000")
    if success:
        print_success("API tests completed successfully")
    else:
        print_warning("Some API tests may have failed, but continuing...")
        print(f"Output: {stdout}")
        print(f"Error: {stderr}")
    
    # Run Llama tests
    print_step("Running Llama-specific tests...")
    if os.path.exists("test_llama.py"):
        success, stdout, stderr = run_command("python test_llama.py")
        if success:
            print_success("Llama tests completed successfully")
        else:
            print_warning("Llama tests may have failed")
            print(f"Output: {stdout}")
            print(f"Error: {stderr}")
    else:
        print_warning("test_llama.py not found, skipping Llama tests")
    
    # Success message
    print("\n" + "=" * 65)
    print("\033[92m🎉 SETUP COMPLETE!\033[0m")
    print("=" * 65)
    print()
    print("🌐 API Documentation: \033[96mhttp://localhost:8000/docs\033[0m")
    print("💓 Health Check: \033[96mhttp://localhost:8000/health\033[0m")
    print("📋 Models List: \033[96mhttp://localhost:8000/models\033[0m")
    print()
    print("\033[93mQuick test commands:\033[0m")
    print("  python client.py --mode interactive")
    print("  python test_llama.py")
    print()
    print("\033[93mTo stop the container:\033[0m")
    print("  docker stop model-runner-test")
    print("  docker rm model-runner-test")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Setup interrupted by user.")
        print("Cleaning up...")
        run_command("docker stop model-runner-test", check=False)
        run_command("docker rm model-runner-test", check=False)
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        sys.exit(1)
