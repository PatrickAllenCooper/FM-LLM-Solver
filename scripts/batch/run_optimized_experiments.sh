#!/bin/bash
# FM-LLM-Solver Optimized Experiment Runner for RTX 3080
# This script runs the fine-tuning and evaluation pipeline with optimized settings

echo "=== FM-LLM-Solver Optimized Experiment Runner ==="
echo "This script will run experiments with settings optimized for 15B models on RTX 3080 (10GB VRAM)"

# Create necessary directories
mkdir -p output/knowledge_base
mkdir -p output/finetuning_results

# Check if knowledge base exists
if [ -f "paper_index_mathpix.faiss" ]; then
    echo "Knowledge base found, continuing with fine-tuning"
else
    echo "Knowledge base files not found. Please run knowledge base creation first."
    echo "You can build the knowledge base using: python run_kb_build.py"
    exit 1
fi

echo
echo "=== VRAM Optimization ==="
echo "1. Setting PyTorch to release memory aggressively"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "2. Limiting CUDA visible devices to single GPU"
export CUDA_VISIBLE_DEVICES=0

echo "3. Setting TF memory growth (for potential TF dependencies)"
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo
echo "=== Starting Fine-tuning with Optimized Settings ==="
echo "This will use the configuration from config.yaml with 4-bit quantization,"
echo "LoRA with rank 8, and gradient accumulation steps of 8."

# Run fine-tuning directly to skip unnecessary steps in the pipeline
python fine_tuning/finetune_llm.py --config config.yaml

# Check if fine-tuning was successful
if [ $? -ne 0 ]; then
    echo
    echo "Fine-tuning failed with error code $?"
    echo "Check error messages above for details."
    echo
    echo "Troubleshooting tips:"
    echo "1. If you're getting CUDA out of memory errors, try editing config.yaml to:"
    echo "   - Reduce lora.r (currently 8, try 4)"
    echo "   - Increase gradient_accumulation_steps (currently 8, try 16)"
    echo "   - Decrease max_seq_length (currently 1024, try 768)"
    echo "2. Make sure your GPU drivers are up to date"
    echo "3. Close other applications using the GPU"
    exit 1
fi

echo
echo "=== Starting Evaluation and Verification ==="
echo "Running evaluation on the fine-tuned model"

# Run evaluation pipeline
python evaluation/evaluate_pipeline.py --config config.yaml

# Check if evaluation was successful
if [ $? -ne 0 ]; then
    echo
    echo "Evaluation failed with error code $?"
    echo "Check error messages above for details."
    exit 1
fi

echo
echo "=== All Done! ==="
echo "Experiment completed successfully."
echo "Results can be found in the output directory."
echo

# Make script executable
chmod +x "$0" 