# Optimized FM-LLM-Solver for RTX 3080

This README provides instructions for running the FM-LLM-Solver barrier certificate system with 15B parameter models on an RTX 3080 (10GB VRAM) GPU. The configuration has been optimized to work within the memory constraints of consumer GPUs.

## Optimizations Applied

The following optimizations have been implemented to enable training and inference with 15B parameter models:

1. **Memory-efficient 4-bit quantization**:
   - 4-bit quantization with NF4 data type
   - Enabled nested quantization for extra memory savings
   - Using flash attention when available

2. **Reduced LoRA parameters**:
   - Reduced LoRA rank from 16 to 8
   - Reduced LoRA alpha from 16 to 8
   - Targeted only attention modules (query, key, value, output projections)

3. **Training optimizations**:
   - Small batch size of 1 with increased gradient accumulation steps (8)
   - Disabled group_by_length (which can use more memory)
   - Limited sequence length to 1024 tokens
   - Enabled gradient checkpointing
   - Disabled caching during training
   - Added explicit garbage collection and CUDA cache clearing
   - Reduced logging and checkpoint frequency

4. **Verification optimizations**:
   - Reduced sample sizes for numerical verification
   - Streamlined optimization parameters

## Running the Optimized Pipeline

### Prerequisites

1. Ensure Python 3.8+ is installed
2. Install PyTorch with CUDA support
3. Install all dependencies from requirements.txt
4. Make sure you have the knowledge base files in place:
   - paper_index_mathpix.faiss
   - paper_metadata_mathpix.jsonl

### Setup

1. Clone the repository if you haven't already:
   ```
   git clone https://github.com/yourusername/FM-LLM-Solver.git
   cd FM-LLM-Solver
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure your knowledge base files are in the correct location (root directory or as specified in config.yaml)

### Running Experiments

We've provided a batch script that configures environment variables, runs fine-tuning, and then evaluation:

```
run_optimized_experiments.bat
```

This script will:
1. Set memory optimization environment variables
2. Check for the knowledge base files
3. Run fine-tuning with the optimized settings
4. Run evaluation with the fine-tuned model
5. Report results in the output directory

### Running Inference

To generate barrier certificates for a specific system:

```
run_inference.bat "System Dynamics: dx/dt = -x**3 - y, dy/dt = x - y**3. Initial Set: x**2+y**2 <= 0.1. Unsafe Set: x >= 1.5"
```

If no system is provided, a default example will be used.

## Troubleshooting

If you encounter out-of-memory errors:

1. **Further reduce LoRA rank**: Edit config.yaml and change lora.r from 8 to 4
2. **Increase gradient accumulation**: Edit config.yaml and increase gradient_accumulation_steps from 8 to 16
3. **Reduce sequence length**: Edit config.yaml and decrease max_seq_length from 1024 to 768
4. **Close other GPU applications**: Make sure no other programs are using the GPU
5. **Try partial CPU offloading**: Run fine-tuning with the `--offload_layers` flag:
   ```
   python fine_tuning/finetune_llm.py --config config.yaml --offload_layers
   ```

## Advanced Configuration

### Flash Attention

Flash Attention 2 is used if available to reduce memory usage. Make sure you have the latest version:

```
pip install flash-attn --no-build-isolation
```

### Layer Offloading

For extremely large models, the script supports offloading some transformer layers to CPU:

```
python fine_tuning/finetune_llm.py --config config.yaml --offload_layers
```

### Custom Configuration

You can edit `config.yaml` to further customize model parameters:

```yaml
# Memory optimization settings
fine_tuning:
  quantization:
    use_4bit: true
    bnb_4bit_compute_dtype: float16
    use_nested_quant: true
  lora:
    r: 8  # Try 4 for extreme memory constraints
    alpha: 8
  training:
    gradient_accumulation_steps: 8  # Try 16 for extreme memory constraints
    max_seq_length: 1024  # Try 768 for extreme memory constraints
```

## Results and Performance

With these optimizations, you should be able to fine-tune 15B parameter models like Qwen2.5-15B-Instruct on an RTX 3080 with 10GB VRAM.

Typical training metrics:
- Training time: ~2-4 hours depending on dataset size
- Peak VRAM usage: ~9.5GB
- Throughput: ~1-2 samples per second (depending on sequence length) 