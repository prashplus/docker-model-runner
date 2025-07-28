"""
Quick Llama 3.2 Test Script
Run this script to quickly test Llama 3.2 functionality after starting the server
"""

import asyncio
import aiohttp
import json
import time

async def test_llama_quick():
    """Quick test of Llama 3.2 functionality"""
    base_url = "http://localhost:8000"
    
    print("LLAMA Quick Llama 3.2 Test")
    print("=" * 40)
    
    async with aiohttp.ClientSession() as session:
        # Test health
        print("1. Checking server health...")
        try:
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    health = await response.json()
                    print(f"   [OK] Server healthy, {health['models_loaded']} models loaded")
                else:
                    print(f"   [ERROR] Health check failed: {response.status}")
                    return
        except Exception as e:
            print(f"   [ERROR] Cannot connect to server: {e}")
            print("   Make sure the server is running: docker-compose up")
            return
        
        # List models
        print("\n2. Checking available models...")
        try:
            async with session.get(f"{base_url}/models") as response:
                models = await response.json()
                llama_available = any(m['name'] == 'llama3.2' for m in models)
                print(f"   Models: {[m['name'] for m in models]}")
                if not llama_available:
                    print("   [WARNING] Llama 3.2 not loaded yet (may still be loading)")
        except Exception as e:
            print(f"   [ERROR] Failed to list models: {e}")
        
        # Test Llama generation
        print("\n3. Testing Llama text generation...")
        test_prompts = [
            "Hello! How are you?",
            "Explain machine learning briefly.",
            "What is Docker?"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n   Test {i}: '{prompt}'")
            start_time = time.time()
            
            try:
                payload = {
                    "prompt": prompt,
                    "model_name": "llama3.2",
                    "max_tokens": 100,
                    "temperature": 0.7
                }
                
                async with session.post(
                    f"{base_url}/generate",
                    json=payload,
                    timeout=120  # 2 minutes timeout for first load
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        elapsed = time.time() - start_time
                        
                        print(f"   [OK] Generated in {elapsed:.1f}s")
                        print(f"   Response: {result['generated_text'][:150]}...")
                        if len(result['generated_text']) > 150:
                            print("   ...")
                    else:
                        error = await response.text()
                        print(f"   [ERROR] Generation failed ({response.status}): {error}")
                        
            except asyncio.TimeoutError:
                print("   [TIMEOUT] Timeout - model may still be loading")
            except Exception as e:
                print(f"   [ERROR] Error: {e}")
        
        # Test predict endpoint with text
        print("\n4. Testing predict endpoint with text...")
        try:
            payload = {
                "data": "Write a short poem about AI",
                "model_name": "llama3.2"
            }
            
            start_time = time.time()
            async with session.post(
                f"{base_url}/predict",
                json=payload,
                timeout=60
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    elapsed = time.time() - start_time
                    print(f"   [OK] Predict endpoint works ({elapsed:.1f}s)")
                    print(f"   Response: {result['predictions'][:100]}...")
                else:
                    error = await response.text()
                    print(f"   [ERROR] Predict failed ({response.status}): {error}")
                    
        except Exception as e:
            print(f"   [ERROR] Predict test error: {e}")
        
    print("\n" + "=" * 40)
    print("LLAMA Llama 3.2 test completed!")
    print("\nFor interactive testing, run:")
    print("python client.py --mode interactive")
    print("\nFor full test suite, run:")
    print("python client.py --mode test")

if __name__ == "__main__":
    asyncio.run(test_llama_quick())
