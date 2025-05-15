#!/bin/bash
# launch_rag_gpu.sh - Configure and launch RAG application with GPU support

set -e

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Load environment variables
if [ -f .env ]; then
    echo "Loading environment from .env file"
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found, using defaults"
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# GPU Check
echo "Checking GPU availability..."
if command_exists nvidia-smi; then
    echo "NVIDIA GPU detected:"
    nvidia-smi
    export CUDA_VISIBLE_DEVICES=0
else
    echo "Warning: nvidia-smi not found, GPU may not be available"
fi

# Ensure Ollama is running
echo "Checking if Ollama is running..."
OLLAMA_RUNNING=false
if command_exists curl; then
    if curl -s -o /dev/null -w "%{http_code}" "${OLLAMA_BASE_URL:-http://localhost:11434}/api/tags" | grep -q "200"; then
        OLLAMA_RUNNING=true
        echo "Ollama is running at ${OLLAMA_BASE_URL:-http://localhost:11434}"
    fi
fi

if [ "$OLLAMA_RUNNING" = false ]; then
    echo "Warning: Ollama does not appear to be running. Start it with:"
    echo "  OLLAMA_NUM_GPU=1 ollama serve"
    
    read -p "Do you want to try starting Ollama with GPU support now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Starting Ollama with GPU support..."
        if command_exists screen; then
            screen -dmS ollama bash -c "OLLAMA_NUM_GPU=1 ollama serve"
            echo "Ollama started in a screen session. Attach with: screen -r ollama"
            sleep 3  # Give Ollama time to start
        else
            echo "Screen command not found, starting Ollama directly..."
            OLLAMA_NUM_GPU=1 ollama serve &
            OLLAMA_PID=$!
            echo "Ollama started with PID: $OLLAMA_PID"
            sleep 3  # Give Ollama time to start
        fi
    fi
fi

# Verify GPU setup
echo "Verifying GPU setup..."
if command_exists python3; then
    if [ -f "verify_gpu.py" ]; then
        python3 verify_gpu.py || true  # Continue even if verification fails
    else
        echo "Warning: verify_gpu.py not found, skipping verification"
    fi
else
    echo "Warning: python3 not found, skipping GPU verification"
fi

# Set environment variables for GPU
export OLLAMA_NUM_GPU=1
export CUDA_VISIBLE_DEVICES=0

# Print configuration
echo 
echo "=== RAG Application Configuration ==="
echo "Ollama URL: ${OLLAMA_BASE_URL:-http://localhost:11434}"
echo "Ollama Model: ${OLLAMA_MODEL:-llama3:latest}"
echo "GPU Settings: OLLAMA_NUM_GPU=${OLLAMA_NUM_GPU}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Port: ${PORT:-5001}"
echo "Debug: ${DEBUG:-False}"
echo "Vector Store: ${CHROMA_PERSIST_DIR:-./chroma_db}"
echo "Document Dir: ${DOCUMENT_DIR:-./DocumentDir}"
echo "======================================"
echo

# Launch application
echo "Starting RAG application with GPU acceleration..."
python3 app.py

# If we get here, the application has stopped
echo "RAG application has stopped."
