# Optimization Tools for FM-LLM-Solver

This directory contains scripts and tools for optimizing the FM-LLM-Solver system, particularly for fine-tuning large language models on consumer-grade hardware.

## Available Tools

### 1. Fine-tuning Diagnostic Tool

- **diagnose_and_fix_finetune.py**: Python script that diagnoses and fixes common issues with fine-tuning
- **diagnose_finetune.bat**: Windows batch script wrapper for the diagnostic tool

#### Features

- Check GPU configuration and available memory
- Verify model availability on Hugging Face Hub
- Analyze configuration compatibility with hardware
- Fix common issues automatically (incorrect model names, memory parameters)
- Run minimal fine-tuning tests to verify system functionality

#### Usage

```bash
# Run basic diagnostics:
scripts\optimization\diagnose_finetune.bat

# Apply automatic fixes:
scripts\optimization\diagnose_finetune.bat --fix

# Run diagnostic test:
scripts\optimization\diagnose_finetune.bat --test

# Apply fixes and run test:
scripts\optimization\diagnose_finetune.bat --all
```

### 2. Memory Optimization

For detailed information about memory optimization techniques, see:
- [Memory Optimization Guide](../../docs/MEMORY_OPTIMIZATION.md)

## Common Issues Addressed

These optimization tools help address the following common issues:

1. **Memory Management**: Optimize settings for limited VRAM (10GB on RTX 3080)
2. **Model Configuration**: Ensure correct model names and parameters
3. **Training Stability**: Improve reliability of fine-tuning process
4. **Performance**: Enhance training speed and efficiency

## Requirements

- Python 3.8 or higher
- PyTorch with CUDA support
- NVIDIA GPU with at least 8GB VRAM
- huggingface_hub and psutil Python packages

## Additional Resources

See the [run_7b_optimized.bat](../experiments/run_7b_optimized.bat) and [run_optimized_experiments.bat](../experiments/run_optimized_experiments.bat) scripts for ready-to-use optimized fine-tuning pipelines. 