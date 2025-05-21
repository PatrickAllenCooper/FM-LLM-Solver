#!/bin/bash
# FM-LLM-Solver Inference Runner for RTX 3080
# This script runs inference with the optimized fine-tuned model

echo "=== FM-LLM-Solver Barrier Certificate Inference ==="
echo "This script will run inference with the fine-tuned model on a system example"

# Set a test system example if one isn't provided as an argument
TEST_SYSTEM="System Dynamics: dx/dt = -x**3 - y, dy/dt = x - y**3. Initial Set: x**2+y**2 <= 0.1. Unsafe Set: x >= 1.5"

# If an argument was provided, use it as the test system
if [ $# -gt 0 ]; then
    TEST_SYSTEM="$1"
fi

echo
echo "=== VRAM Optimization ==="
echo "1. Setting PyTorch to release memory aggressively"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "2. Limiting CUDA visible devices to single GPU"
export CUDA_VISIBLE_DEVICES=0

echo "3. Setting memory growth for better utilization"
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo
echo "=== Running Inference ==="
echo "System: $TEST_SYSTEM"
echo

# Check if the adapter exists
if [ ! -d "output/finetuning_results/final_adapter" ]; then
    echo "Error: Fine-tuned adapter not found at output/finetuning_results/final_adapter"
    echo "Please run fine-tuning first with run_optimized_experiments.sh"
    exit 1
fi

# Run inference with the test system
python inference/generate_certificate.py "$TEST_SYSTEM" --config config.yaml

# Check if inference was successful
if [ $? -ne 0 ]; then
    echo
    echo "Inference failed with error code $?"
    echo "Check error messages above for details."
    exit 1
fi

echo
echo "=== Done! ==="
echo "Inference completed successfully."
echo

echo "=== Additional Examples ==="
echo "You can run this script with a custom system description:"
echo "./run_inference.sh \"System Dynamics: dx/dt = -x - y, dy/dt = 3x - y. Initial Set: x^2+y^2 <= 0.5. Unsafe Set: x^2+y^2 >= 4\""
echo

# Make script executable
chmod +x "$0" 