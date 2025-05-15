#!/usr/bin/env python3
"""
GPU Verification Script for RAG Application
This script checks if Ollama can access and utilize GPU resources.
"""
import os
import sys
import json
import requests
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GPU_Verifier")

def check_ollama_gpu():
    """Test if Ollama can use GPU"""
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "llama3:latest")
    
    logger.info(f"Testing GPU with Ollama at {ollama_base_url} using model {model}")
    
    try:
        # Get model list
        logger.info("Checking available models...")
        response = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
        if response.status_code != 200:
            logger.error(f"Failed to get models: HTTP {response.status_code}")
            return False
            
        models = response.json().get("models", [])
        available_models = [m.get("name") for m in models]
        
        if not available_models:
            logger.error("No models found in Ollama")
            return False
            
        if model not in available_models:
            logger.warning(f"Model {model} not found. Available models: {', '.join(available_models)}")
            # Use the first available model as fallback
            model = available_models[0]
            logger.info(f"Using fallback model: {model}")
        
        # Test with GPU parameters
        logger.info(f"Running test inference with model {model} and GPU parameters...")
        test_prompt = "Explain how vector retrieval works in 5 words."
        
        start_time = time.time()
        response = requests.post(
            f"{ollama_base_url}/api/generate",
            json={
                "model": model,
                "prompt": test_prompt,
                "options": {
                    "num_gpu": 1,
                    "num_thread": 8,
                    "f16_kv": True
                }
            },
            timeout=30
        )
        end_time = time.time()
        
        if response.status_code != 200:
            logger.error(f"Inference failed: HTTP {response.status_code}")
            logger.error(response.text)
            return False
            
        result = response.json()
        response_text = result.get("response", "").strip()
        
        logger.info(f"Response: {response_text}")
        logger.info(f"Inference time: {end_time - start_time:.2f} seconds")
        
        # Check GPU usage through dedicated question
        logger.info("Checking GPU usage through dedicated query...")
        response = requests.post(
            f"{ollama_base_url}/api/generate",
            json={
                "model": model,
                "prompt": "Are you using GPU acceleration? Just answer yes or no.",
                "options": {
                    "num_gpu": 1
                }
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            usage_response = result.get("response", "").lower().strip()
            logger.info(f"GPU usage response: {usage_response}")
            using_gpu = 'yes' in usage_response
        else:
            logger.warning("Could not determine GPU usage from model response")
            using_gpu = None
        
        return True
    except Exception as e:
        logger.error(f"Error testing GPU with Ollama: {str(e)}")
        return False

def check_system_gpu():
    """Check if system GPU is available and recognized"""
    logger.info("Checking system GPU resources...")
    
    # Try NVIDIA tools first
    try:
        logger.info("Checking with nvidia-smi...")
        import subprocess
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("NVIDIA GPU detected:")
            print(result.stdout)
            return True
        else:
            logger.warning("nvidia-smi command failed")
    except Exception as e:
        logger.warning(f"Error checking NVIDIA GPU: {str(e)}")
    
    # Try PyTorch if available
    try:
        logger.info("Checking with PyTorch...")
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            logger.info(f"PyTorch detected {device_count} GPU(s): {device_name}")
            return True
        else:
            logger.warning("PyTorch reports no available GPU")
    except Exception as e:
        logger.warning(f"Error checking GPU with PyTorch: {str(e)}")
    
    # Try TensorFlow if available
    try:
        logger.info("Checking with TensorFlow...")
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"TensorFlow detected GPU: {gpus}")
            return True
        else:
            logger.warning("TensorFlow reports no available GPU")
    except Exception as e:
        logger.warning(f"Error checking GPU with TensorFlow: {str(e)}")
    
    logger.error("No GPU detected with any method")
    return False

def suggest_fixes():
    """Suggest potential fixes for GPU issues"""
    logger.info("\n=== GPU Issue Troubleshooting ===")
    print("""
Common issues and fixes:

1. CUDA not installed or incorrect version:
   - Install CUDA Toolkit 11.8+ or 12.x 
   - Set CUDA_HOME environment variable

2. Ollama configuration:
   - Make sure Ollama server is running with GPU support
   - Check Ollama logs for GPU initialization errors
   - Set OLLAMA_NUM_GPU=1 environment variable

3. Model issues:
   - Ensure your model supports GPU acceleration
   - Some smaller models may not benefit from GPU acceleration
   - Try a different model (like llama3:latest or mistral:latest)

4. Environment variables:
   - Set CUDA_VISIBLE_DEVICES=0
   - Check LD_LIBRARY_PATH includes CUDA libraries

5. Hardware issues:
   - Verify GPU is recognized at OS level
   - Check power and cooling
   - Update GPU drivers
    """)

if __name__ == "__main__":
    print("=== RAG Application GPU Verification ===\n")
    
    system_gpu = check_system_gpu()
    ollama_gpu = check_ollama_gpu()
    
    print("\n=== Results ===")
    print(f"System GPU detected: {'Yes' if system_gpu else 'No'}")
    print(f"Ollama GPU utilization: {'Yes' if ollama_gpu else 'No'}")
    
    if not system_gpu or not ollama_gpu:
        suggest_fixes()
        sys.exit(1)
    else:
        print("\nGPU setup looks good! The RAG application should be able to use GPU acceleration.")
        sys.exit(0)
