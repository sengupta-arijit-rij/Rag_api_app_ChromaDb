#!/usr/bin/env python3
"""
Enhanced GPU Diagnostic for RAG Application
This script provides a comprehensive diagnosis of GPU issues with Ollama integration.
"""
import os
import sys
import json
import requests
import time
import logging
import subprocess
import platform
from typing import Dict, Any, List, Optional

# Setup colorful logging for better readability
try:
    import coloredlogs
    coloredlogs.install(
        level=logging.INFO,
        fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        level_styles={
            'debug': {'color': 'green'},
            'info': {'color': 'blue'},
            'warning': {'color': 'yellow', 'bold': True},
            'error': {'color': 'red', 'bold': True},
            'critical': {'color': 'red', 'bold': True, 'background': 'white'}
        }
    )
except ImportError:
    # Fall back to standard logging if coloredlogs is not available
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

logger = logging.getLogger("GPU_Diagnostic")

class GPUDiagnostic:
    def __init__(self):
        self.ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.environ.get("OLLAMA_MODEL", "llama3:latest")
        self.system_info = self._get_system_info()
        self.env_vars = self._get_environment_variables()
        self.gpu_info = {}
        
    def _get_system_info(self) -> Dict[str, str]:
        """Get basic system information"""
        return {
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor()
        }
        
    def _get_environment_variables(self) -> Dict[str, str]:
        """Get relevant environment variables"""
        env_vars = {}
        important_vars = [
            "OLLAMA_BASE_URL", "OLLAMA_MODEL", "OLLAMA_NUM_GPU",
            "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "PATH",
            "LD_LIBRARY_PATH", "NVIDIA_DRIVER_CAPABILITIES",
            "NVIDIA_VISIBLE_DEVICES"
        ]
        
        for var in important_vars:
            env_vars[var] = os.environ.get(var, "Not set")
            
        return env_vars
        
    def check_nvidia_toolkit(self) -> Dict[str, Any]:
        """Check NVIDIA CUDA toolkit installation"""
        result = {"installed": False, "version": None, "details": []}
        
        # Check for nvcc (CUDA compiler)
        try:
            nvcc_output = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
            if nvcc_output.returncode == 0:
                result["installed"] = True
                version_line = [line for line in nvcc_output.stdout.split('\n') if "release" in line.lower()]
                if version_line:
                    result["version"] = version_line[0].strip()
                    result["details"].append(f"CUDA compiler: {version_line[0].strip()}")
        except Exception as e:
            result["details"].append(f"NVCC not found: {str(e)}")
            
        # Check CUDA_HOME
        cuda_home = os.environ.get("CUDA_HOME")
        if cuda_home:
            result["details"].append(f"CUDA_HOME set to: {cuda_home}")
            if os.path.exists(cuda_home):
                result["details"].append("CUDA_HOME path exists")
            else:
                result["details"].append("WARNING: CUDA_HOME path does not exist")
        else:
            result["details"].append("WARNING: CUDA_HOME environment variable not set")
            
        return result
        
    def check_system_gpu(self) -> Dict[str, Any]:
        """Check if system GPU is available and recognized"""
        result = {"detected": False, "details": [], "device_info": []}
        
        # Try NVIDIA tools first
        try:
            logger.info("Checking with nvidia-smi...")
            nvidia_smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if nvidia_smi.returncode == 0:
                result["detected"] = True
                result["details"].append("nvidia-smi command successful")
                
                # Extract GPU information
                gpu_info_cmd = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version,cuda_version", "--format=csv,noheader"],
                    capture_output=True, text=True
                )
                
                if gpu_info_cmd.returncode == 0:
                    for i, line in enumerate(gpu_info_cmd.stdout.strip().split('\n')):
                        parts = [part.strip() for part in line.split(',')]
                        if len(parts) >= 5:
                            gpu_dict = {
                                "id": i,
                                "name": parts[0],
                                "memory_total": parts[1],
                                "memory_free": parts[2],
                                "driver_version": parts[3],
                                "cuda_version": parts[4]
                            }
                            result["device_info"].append(gpu_dict)
                
                # Add full output for reference
                result["nvidia_smi_output"] = nvidia_smi.stdout
            else:
                result["details"].append(f"nvidia-smi command failed with error: {nvidia_smi.stderr}")
        except Exception as e:
            result["details"].append(f"Error checking NVIDIA GPU: {str(e)}")
        
        # Try PyTorch if available
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                result["detected"] = True
                device_count = torch.cuda.device_count()
                result["details"].append(f"PyTorch detected {device_count} GPU(s)")
                
                for i in range(device_count):
                    device_name = torch.cuda.get_device_name(i)
                    result["details"].append(f"PyTorch GPU {i}: {device_name}")
                    
                    # Check if this device is already in our list
                    found = False
                    for gpu in result["device_info"]:
                        if gpu.get("name") == device_name:
                            gpu["pytorch_available"] = True
                            found = True
                            break
                            
                    if not found:
                        result["device_info"].append({
                            "id": i,
                            "name": device_name,
                            "pytorch_available": True
                        })
            else:
                result["details"].append("PyTorch reports no available GPU")
        except ImportError:
            result["details"].append("PyTorch not installed")
        except Exception as e:
            result["details"].append(f"Error checking GPU with PyTorch: {str(e)}")
            
        return result
        
    def check_ollama_models(self) -> Dict[str, Any]:
        """Check available Ollama models"""
        result = {"status": "error", "models": [], "details": []}
        
        try:
            logger.info(f"Checking Ollama models at {self.ollama_base_url}...")
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                result["status"] = "success"
                data = response.json()
                models = data.get("models", [])
                result["models"] = [m.get("name") for m in models]
                result["model_details"] = models
                
                if self.model in result["models"]:
                    result["details"].append(f"Specified model '{self.model}' is available")
                else:
                    result["details"].append(f"WARNING: Specified model '{self.model}' not found in available models")
                    
                # Get models with quantization info
                gpu_compatible_models = []
                cpu_only_models = []
                
                for model in models:
                    name = model.get("name", "")
                    is_gpu_compatible = False
                    
                    # Check for quantization in name (heuristic)
                    if any(q in name.lower() for q in ["f16", "q4", "q5", "q6", "q8"]):
                        is_gpu_compatible = True
                        gpu_compatible_models.append(name)
                    else:
                        cpu_only_models.append(name)
                        
                result["gpu_compatible_models"] = gpu_compatible_models
                result["cpu_only_models"] = cpu_only_models
                
                if gpu_compatible_models:
                    result["details"].append(f"Found {len(gpu_compatible_models)} models that may support GPU acceleration")
                else:
                    result["details"].append("WARNING: No models found with GPU quantization markers")
            else:
                result["details"].append(f"Failed to get models: HTTP {response.status_code}")
        except requests.exceptions.ConnectionError:
            result["details"].append(f"Connection error: Could not connect to Ollama at {self.ollama_base_url}")
        except Exception as e:
            result["details"].append(f"Error checking Ollama models: {str(e)}")
            
        return result
        
    def check_ollama_gpu(self) -> Dict[str, Any]:
        """Test if Ollama can use GPU with detailed diagnostics"""
        result = {
            "status": "error",
            "using_gpu": False,
            "inference_time": None,
            "temperature": None,
            "response": None,
            "details": []
        }
        
        logger.info(f"Testing GPU with Ollama at {self.ollama_base_url} using model {self.model}")
        
        try:
            # First check if Ollama is running
            try:
                health_check = requests.get(f"{self.ollama_base_url}/api/tags", timeout=2)
                if health_check.status_code != 200:
                    result["details"].append(f"Ollama not responding: HTTP {health_check.status_code}")
                    return result
            except requests.exceptions.ConnectionError:
                result["details"].append(f"Connection error: Could not connect to Ollama at {self.ollama_base_url}")
                return result
                
            # Check if model exists
            models_check = self.check_ollama_models()
            available_models = models_check.get("models", [])
            
            if not available_models:
                result["details"].append("No models found in Ollama")
                return result
                
            if self.model not in available_models:
                result["details"].append(f"Model {self.model} not found. Available models: {', '.join(available_models)}")
                # Use the first available model as fallback
                fallback_model = models_check.get("gpu_compatible_models", [])
                if fallback_model:
                    self.model = fallback_model[0]
                    result["details"].append(f"Using fallback model with GPU compatibility: {self.model}")
                else:
                    self.model = available_models[0]
                    result["details"].append(f"Using fallback model (may not support GPU): {self.model}")
                    
            # Test with various GPU parameters
            gpu_config_tests = [
                {
                    "name": "High GPU utilization",
                    "options": {
                        "num_gpu": 99,  # Request all available GPUs
                        "num_thread": 8,
                        "f16_kv": True,
                        "temperature": 0.1  # Low temperature for more deterministic results
                    }
                },
                {
                    "name": "Standard GPU utilization",
                    "options": {
                        "num_gpu": 1,
                        "num_thread": 4
                    }
                },
                {
                    "name": "Minimal GPU options",
                    "options": {
                        "num_gpu": 1
                    }
                }
            ]
            
            fastest_config = None
            fastest_time = float('inf')
            
            for config in gpu_config_tests:
                logger.info(f"Testing {config['name']} configuration...")
                test_prompt = "Explain how vector embedding works in RAG applications in 5 words."
                
                try:
                    start_time = time.time()
                    response = requests.post(
                        f"{self.ollama_base_url}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": test_prompt,
                            "options": config["options"]
                        },
                        timeout=60
                    )
                    end_time = time.time()
                    inference_time = end_time - start_time
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        response_text = response_data.get("response", "").strip()
                        
                        config_result = {
                            "name": config["name"],
                            "time": inference_time,
                            "response": response_text
                        }
                        
                        result["details"].append(f"{config['name']} completed in {inference_time:.2f}s: {response_text}")
                        
                        if inference_time < fastest_time:
                            fastest_time = inference_time
                            fastest_config = config["name"]
                            
                    else:
                        result["details"].append(f"{config['name']} failed: HTTP {response.status_code}")
                except Exception as e:
                    result["details"].append(f"{config['name']} error: {str(e)}")
                    
            if fastest_config:
                result["details"].append(f"Fastest configuration: {fastest_config} ({fastest_time:.2f}s)")
                
            # Check GPU usage through dedicated question
            try:
                logger.info("Checking GPU usage through dedicated query...")
                gpu_check_response = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": "Are you currently executing on a GPU? Just answer yes or no and nothing else.",
                        "options": {
                            "num_gpu": 1
                        }
                    },
                    timeout=15
                )
                
                if gpu_check_response.status_code == 200:
                    gpu_check_data = gpu_check_response.json()
                    usage_response = gpu_check_data.get("response", "").lower().strip()
                    result["response"] = usage_response
                    result["using_gpu"] = 'yes' in usage_response
                    result["details"].append(f"GPU usage self-report: '{usage_response}'")
                    
                    if 'yes' in usage_response:
                        result["status"] = "success"
                    else:
                        result["status"] = "warning"
                        result["details"].append("Model reports it is NOT using GPU")
                else:
                    result["details"].append(f"GPU usage check failed: HTTP {gpu_check_response.status_code}")
            except Exception as e:
                result["details"].append(f"GPU usage check error: {str(e)}")
                
            # If we got this far but couldn't confirm GPU usage
            if result["status"] == "error" and fastest_time < float('inf'):
                result["status"] = "warning"  # At least something is working
                result["inference_time"] = fastest_time
                
            return result
        except Exception as e:
            result["details"].append(f"Error testing GPU with Ollama: {str(e)}")
            return result
            
    def check_rag_engine_config(self) -> Dict[str, Any]:
        """Analyze the RAG engine configuration"""
        result = {"issues": [], "recommendations": []}
        
        # Check for common RAG engine configuration issues
        with open("rag_engine.py", "r") as f:
            engine_code = f.read()
            
        # Check if GPU options are properly passed to Ollama
        if "num_gpu" not in engine_code:
            result["issues"].append("RAG engine does not set num_gpu parameter for Ollama")
            result["recommendations"].append("Add num_gpu parameter to Ollama configuration in rag_engine.py")
            
        # Check for CUDA-specific settings
        if "embed_batch_size" in engine_code:
            # Good: Batch size is configured which helps GPU performance
            current_batch_size = None
            import re
            batch_match = re.search(r'embed_batch_size=(\d+)', engine_code)
            if batch_match:
                current_batch_size = int(batch_match.group(1))
                
            if current_batch_size and current_batch_size < 8:
                result["recommendations"].append(f"Current embed_batch_size={current_batch_size} may be too small for efficient GPU usage. Consider increasing to 8-16.")
                
        # Other checks can be added here
        
        return result
        
    def generate_fix_script(self) -> str:
        """Generate a fix script based on diagnostic results"""
        script = "#!/bin/bash\n\n"
        script += "# GPU Fix Script for RAG Application\n"
        script += "# Generated by GPU Diagnostic Tool\n\n"
        
        # Check diagnostic results to determine necessary fixes
        gpu_check = self.check_system_gpu()
        ollama_check = self.check_ollama_gpu()
        nvidia_check = self.check_nvidia_toolkit()
        rag_config = self.check_rag_engine_config()
        
        # Environment variable fixes
        script += "# Set environment variables\n"
        script += "export OLLAMA_NUM_GPU=1\n"
        
        if not nvidia_check["installed"]:
            script += "\necho 'CUDA toolkit not found. Installing CUDA dependencies...'\n"
            script += "# You may need to modify this for your specific OS\n"
            if "ubuntu" in self.system_info["os"].lower() or "debian" in self.system_info["os"].lower():
                script += "sudo apt-get update\n"
                script += "sudo apt-get install -y nvidia-cuda-toolkit\n"
            elif "centos" in self.system_info["os"].lower() or "rhel" in self.system_info["os"].lower():
                script += "sudo yum install -y nvidia-cuda-toolkit\n"
            else:
                script += "# Please install CUDA toolkit manually for your OS\n"
                
        # RAG engine modifications
        if rag_config["recommendations"]:
            script += "\n# Modify RAG Engine\n"
            script += "echo 'Updating RAG engine configuration...'\n\n"
            
            # Create a patch file to modify rag_engine.py
            script += "cat > rag_engine_gpu.patch << 'EOL'\n"
            script += "--- rag_engine.py\t2023-01-01 00:00:00.000000000 +0000\n"
            script += "+++ rag_engine_gpu.py\t2023-01-01 00:00:00.000000000 +0000\n"
            
            # Add num_gpu parameter
            if "RAG engine does not set num_gpu parameter" in '\n'.join(rag_config["issues"]):
                script += "@@ -62,6 +62,7 @@\n"
                script += "             self.embed_model = OllamaEmbedding(\n"
                script += "                 model_name=self.ollama_model,\n"
                script += "                 base_url=self.ollama_base_url,\n"
                script += "+                ollama_additional_kwargs={\"num_gpu\": 1},\n"
                script += "                 embed_batch_size=10,  # Reduce batch size for stability\n"
                script += "                 ollama_additional_kwargs={\"mirostat\": 0},\n"
                script += "             )\n"
                
                script += "@@ -73,6 +74,7 @@\n"
                script += "             self.llm = Ollama(\n"
                script += "                 model=self.ollama_model, \n"
                script += "                 base_url=self.ollama_base_url,\n"
                script += "+                additional_kwargs={\"num_gpu\": 1},\n"
                script += "                 request_timeout=180.0  # Increased timeout for longer responses\n"
                script += "             )\n"
                
            script += "EOL\n\n"
            script += "# Apply the patch\n"
            script += "patch -p0 < rag_engine_gpu.patch\n"
        
        # Ollama model recommendations
        if ollama_check.get("status") != "success":
            models_check = self.check_ollama_models()
            gpu_models = models_check.get("gpu_compatible_models", [])
            
            if gpu_models:
                script += "\n# Download a GPU-compatible model\n"
                script += f"echo 'Downloading GPU-compatible model: {gpu_models[0]}...'\n"
                script += f"ollama pull {gpu_models[0]}\n"
                script += f"export OLLAMA_MODEL={gpu_models[0]}\n"
            else:
                script += "\n# Download a model with GPU support\n"
                script += "echo 'Downloading llama3:latest which has good GPU support...'\n"
                script += "ollama pull llama3:latest\n"
                script += "export OLLAMA_MODEL=llama3:latest\n"
        
        # Add final validation
        script += "\n# Verify GPU setup\n"
        script += "echo 'Running GPU verification...'\n"
        script += "python verify_gpu.py\n\n"
        script += "echo 'Setup complete. If verification succeeded, your RAG application should now use GPU acceleration.'\n"
        
        return script
        
    def run_full_diagnostic(self) -> Dict[str, Any]:
        """Run full diagnostic and return results"""
        results = {
            "timestamp": time.time(),
            "system_info": self.system_info,
            "environment": self.env_vars,
            "nvidia_toolkit": self.check_nvidia_toolkit(),
            "system_gpu": self.check_system_gpu(),
            "ollama_models": self.check_ollama_models(),
            "ollama_gpu": self.check_ollama_gpu(),
            "rag_config": self.check_rag_engine_config()
        }
        
        return results
        
    def print_diagnostic_report(self, results: Dict[str, Any] = None):
        """Print a formatted diagnostic report"""
        if results is None:
            results = self.run_full_diagnostic()
            
        print("\n" + "="*80)
        print(" "*30 + "GPU DIAGNOSTIC REPORT")
        print("="*80)
        
        # System Information
        print("\n" + "-"*80)
        print("SYSTEM INFORMATION")
        print("-"*80)
        for key, value in results["system_info"].items():
            print(f"{key}: {value}")
            
        # Environment Variables
        print("\n" + "-"*80)
        print("ENVIRONMENT VARIABLES")
        print("-"*80)
        for key, value in results["environment"].items():
            if key in ["PATH", "LD_LIBRARY_PATH"]:
                print(f"{key}: [Long path value]")
            else:
                print(f"{key}: {value}")
                
        # NVIDIA Toolkit
        print("\n" + "-"*80)
        print("NVIDIA CUDA TOOLKIT")
        print("-"*80)
        if results["nvidia_toolkit"]["installed"]:
            print(f"✅ INSTALLED: {results['nvidia_toolkit']['version']}")
        else:
            print("❌ NOT INSTALLED")
        for detail in results["nvidia_toolkit"]["details"]:
            print(f"  - {detail}")
            
        # System GPU
        print("\n" + "-"*80)
        print("SYSTEM GPU STATUS")
        print("-"*80)
        if results["system_gpu"]["detected"]:
            print("✅ GPU DETECTED")
            for gpu in results["system_gpu"]["device_info"]:
                print(f"  Device {gpu.get('id', 'N/A')}: {gpu.get('name', 'Unknown')}")
                if "memory_total" in gpu:
                    print(f"    - Memory: {gpu.get('memory_total', 'Unknown')}")
                if "driver_version" in gpu:
                    print(f"    - Driver: {gpu.get('driver_version', 'Unknown')}")
                if "cuda_version" in gpu:
                    print(f"    - CUDA: {gpu.get('cuda_version', 'Unknown')}")
        else:
            print("❌ NO GPU DETECTED")
        for detail in results["system_gpu"]["details"]:
            print(f"  - {detail}")
            
        # Ollama Models
        print("\n" + "-"*80)
        print("OLLAMA MODELS")
        print("-"*80)
        if results["ollama_models"]["status"] == "success":
            print(f"✅ CONNECTED: Found {len(results['ollama_models']['models'])} models")
            print("  GPU-compatible models:")
            for model in results["ollama_models"].get("gpu_compatible_models", []):
                print(f"  - {model}")
        else:
            print("❌ FAILED TO CONNECT TO OLLAMA")
        for detail in results["ollama_models"]["details"]:
            print(f"  - {detail}")
            
        # Ollama GPU
        print("\n" + "-"*80)
        print("OLLAMA GPU USAGE")
        print("-"*80)
        if results["ollama_gpu"]["status"] == "success":
            print("✅ USING GPU: Confirmed")
        elif results["ollama_gpu"]["status"] == "warning":
            print("⚠️ AMBIGUOUS: GPU usage could not be confirmed")
        else:
            print("❌ NOT USING GPU")
        
        if results["ollama_gpu"]["inference_time"]:
            print(f"  - Inference time: {results['ollama_gpu']['inference_time']:.2f}s")
        
        for detail in results["ollama_gpu"]["details"]:
            print(f"  - {detail}")
            
        # RAG Configuration
        print("\n" + "-"*80)
        print("RAG ENGINE CONFIGURATION")
        print("-"*80)
        if not results["rag_config"]["issues"]:
            print("✅ No major issues detected")
        else:
            print("⚠️ Issues detected:")
            for issue in results["rag_config"]["issues"]:
                print(f"  - {issue}")
                
        if results["rag_config"]["recommendations"]:
            print("\nRecommendations:")
            for rec in results["rag_config"]["recommendations"]:
                print(f"  - {rec}")
                
        # Overall Assessment
        print("\n" + "-"*80)
        print("OVERALL ASSESSMENT")
        print("-"*80)
        
        # Determine if GPU is working
        gpu_status = "⚠️ UNKNOWN"
        if results["system_gpu"]["detected"] and results["ollama_gpu"]["status"] == "success":
            gpu_status = "✅ WORKING"
        elif not results["system_gpu"]["detected"]:
            gpu_status = "❌ NO GPU DETECTED"
        elif results["system_gpu"]["detected"] and results["ollama_gpu"]["status"] != "success":
            gpu_status = "❌ GPU DETECTED BUT NOT USED BY OLLAMA"
            
        print(f"GPU Status: {gpu_status}")
        
        # Generate fix script
        print("\n" + "-"*80)
        print("RECOMMENDED FIXES")
        print("-"*80)
        
        fix_script = self.generate_fix_script()
        print("A fix script has been generated. Run the following to apply fixes:")
        print("\n1. Save the following content to 'fix_gpu.sh':")
        print("2. Run 'chmod +x fix_gpu.sh'")
        print("3. Execute './fix_gpu.sh'")
        
        print("\n" + "="*80)

if __name__ == "__main__":
    print("="*80)
    print(" "*25 + "RAG APPLICATION GPU DIAGNOSTIC")
    print("="*80)
    
    diagnostic = GPUDiagnostic()
    results = diagnostic.run_full_diagnostic()
    diagnostic.print_diagnostic_report(results)
    
    # Generate and save fix script
    fix_script = diagnostic.generate_fix_script()
    with open("fix_gpu.sh", "w") as f:
        f.write(fix_script)
    
    # Make script executable
    try:
        os.chmod("fix_gpu.sh", 0o755)
        print("\n✅ Fix script saved to 'fix_gpu.sh' and made executable.")
        print("   Run './fix_gpu.sh' to apply the fixes.")
    except Exception as e:
        print(f"\n⚠️ Fix script saved to 'fix_gpu.sh' but couldn't make it executable: {str(e)}")
        print("   Run 'chmod +x fix_gpu.sh && ./fix_gpu.sh' to apply the fixes.")
    
    # Exit with appropriate code
    if results["system_gpu"]["detected"] and results["ollama_gpu"]["status"] == "success":
        print("\n✅ GPU appears to be working correctly.")
        sys.exit(0)
    else:
        print("\n⚠️ GPU issues detected. Apply the suggested fixes.")
        sys.exit(1)
