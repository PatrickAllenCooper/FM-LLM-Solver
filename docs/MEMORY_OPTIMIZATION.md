# Memory Optimization for Large Language Models

This guide provides detailed information on optimizing memory usage when fine-tuning large language models (LLMs) on consumer-grade GPUs such as the RTX 3080.

## Table of Contents

- [Memory Requirements](#memory-requirements)
- [Optimization Techniques](#optimization-techniques)
- [Using the Diagnostic Tool](#using-the-diagnostic-tool)
- [Model Size Recommendations](#model-size-recommendations)
- [Troubleshooting Common Issues](#troubleshooting-common-issues)

## Memory Requirements

Different model sizes require different amounts of GPU memory:

| Model Size | Unoptimized Memory | Optimized Memory | Recommended GPU |
|------------|-------------------|------------------|-----------------|
| 7B         | ~14GB             | ~8GB             | RTX 3080 (10GB) or higher |
| 14B/15B    | ~28GB             | ~12GB            | RTX 3090 (24GB) or higher |
| 70B/72B    | ~140GB            | ~50GB            | A100 (80GB) or multiple GPUs |

The FM-LLM-Solver project includes optimizations that allow running even 14B models on consumer GPUs with 10-12GB of VRAM.

## Optimization Techniques

Our system implements several key memory optimization techniques:

### 1. Quantization

**4-bit Quantization**: Reduces model precision from FP16/BF16 to 4-bit integers.

Configuration in `config.yaml`:
```yaml
quantization:
  use_4bit: true
  bnb_4bit_compute_dtype: float16
  bnb_4bit_quant_type: nf4
  use_nested_quant: true
```

**Nested Quantization**: Further compresses weights by quantizing the quantization constants.

### 2. LoRA (Low-Rank Adaptation)

LoRA reduces memory by training only small adapter matrices instead of full model weights.

Key parameters:
- **Rank (r)**: Lower values (4-8) use less memory, higher values (16-64) potentially capture more information
- **Alpha**: Typically set to same value as rank or 2x rank

Optimized configuration:
```yaml
lora:
  r: 8           # Reduced rank for memory savings
  alpha: 8       # Matching alpha
  dropout: 0.1
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
```

### 3. Gradient Accumulation

Allows simulating larger batch sizes while keeping memory usage low.

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8  # Increased for memory efficiency
```

### 4. Sequence Length Limitation

Longer sequences require more memory. Limiting sequence length helps manage memory usage.

```yaml
training:
  max_seq_length: 1024  # Limit sequence length to save memory
```

### 5. Environment Variables

Setting these environment variables helps manage memory:

```
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
CUDA_VISIBLE_DEVICES=0
TF_FORCE_GPU_ALLOW_GROWTH=true
```

### 6. Gradient Checkpointing

Trades computation for memory by discarding intermediate activations and recomputing them during the backward pass.

```yaml
training:
  gradient_checkpointing: true
```

## Using the Diagnostic Tool

The FM-LLM-Solver includes a diagnostic tool that helps identify and fix memory-related issues.

### Running the Diagnostic Tool

```bash
# Basic diagnostics only
scripts\optimization\diagnose_finetune.bat

# Apply automatic fixes to config.yaml
scripts\optimization\diagnose_finetune.bat --fix

# Run a minimal fine-tuning test
scripts\optimization\diagnose_finetune.bat --test

# Both fix and test
scripts\optimization\diagnose_finetune.bat --all
```

### What the Diagnostic Tool Does

1. Checks GPU configuration and available memory
2. Verifies model availability on Hugging Face
3. Analyzes config compatibility with your GPU
4. Examines error logs for known issues
5. Optionally applies fixes to config.yaml
6. Optionally runs a minimal fine-tuning test

## Model Size Recommendations

Based on available GPU memory:

- **8GB VRAM (RTX 3070)**: 7B models with heavy optimization
- **10GB VRAM (RTX 3080)**: 7B models with standard optimization, 14B with extreme optimization
- **12GB VRAM (RTX 3060)**: 7B models with minimal optimization, 14B with heavy optimization
- **24GB VRAM (RTX 3090/4090)**: 14B models with standard optimization
- **40GB+ VRAM (A100, A6000)**: 70B models with optimization

## Running Optimized Scripts

We provide optimized batch scripts for common scenarios:

### For 7B Models

```bash
scripts\experiments\run_7b_optimized.bat
```

This script:
1. Backs up your config
2. Modifies config.yaml to use Qwen2.5-7B-Instruct
3. Applies memory optimizations
4. Runs fine-tuning and evaluation
5. Restores original config

### For 14B Models

```bash
scripts\experiments\run_optimized_experiments.bat
```

## Troubleshooting Common Issues

### CUDA Out of Memory Errors

If you encounter "CUDA out of memory" errors:

1. Reduce LoRA rank (r) to 4 or 8
2. Increase gradient_accumulation_steps to 8 or 16
3. Reduce max_seq_length to 768 or 512
4. Enable nested quantization
5. Try a smaller model (7B instead of 14B)

### Model Loading Errors

If the model fails to load:

1. Check that the model name is correct (e.g., "Qwen/Qwen2.5-14B-Instruct" not "Qwen/Qwen2.5-15B-Instruct")
2. Verify you have internet access for downloading
3. Check if the model requires authentication
4. Run the diagnostic tool to verify model availability:
   ```
   scripts\optimization\diagnose_finetune.bat
   ```

### Slow Training

If training is very slow:

1. Check that PyTorch is using CUDA (should see GPU usage in Task Manager)
2. Verify you're not running in CPU-only mode
3. Increase per_device_train_batch_size if memory allows

## Advanced Techniques

For extreme cases where even these optimizations aren't enough:

1. **CPU Offloading**: Move some layers to CPU memory
   ```python
   device_map = "auto"  # Let transformers decide best distribution
   ```

2. **Flash Attention**: Use optimized attention implementation
   ```python
   attn_implementation="flash_attention_2"
   ```

3. **DeepSpeed ZeRO**: Partition model across devices, useful for multi-GPU setups

## Checking Your Results

After implementing optimizations, verify that:

1. Training progresses without memory errors
2. Results are still reasonable (check loss values)
3. Generation quality remains acceptable

---

For more information or support, please refer to the main project documentation or file an issue on GitHub. 