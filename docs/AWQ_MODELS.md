# Using AWQ-Quantized Models with FM-LLM-Solver

This guide explains how to use AWQ-quantized Qwen2.5 models for fine-tuning and evaluation in the FM-LLM-Solver project.

## What is AWQ?

AWQ (Activation-aware Weight Quantization) is an advanced quantization technique that enables running large language models with significantly reduced memory requirements. Key benefits include:

- **Memory Efficiency**: Reduces model memory footprint by up to 75% compared to FP16
- **Speed**: Often faster than other quantization methods
- **Quality Preservation**: Minimal impact on model quality compared to other 4-bit methods

## Available AWQ-Quantized Qwen2.5 Models

| Model | Parameters | Size | Min VRAM for Inference | Min VRAM for Fine-tuning |
|-------|------------|------|------------------------|---------------------------|
| [Qwen/Qwen2.5-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ) | 7B | ~2GB | ~6GB | ~8GB |
| [Qwen/Qwen2.5-14B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ) | 14B | ~4GB | ~10GB | ~14GB |
| [Qwen/Qwen2.5-72B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-AWQ) | 72B | ~12GB | ~20GB | Not recommended |

## Setup and Requirements

Before using AWQ-quantized models, you need to install the necessary library:

```bash
pip install autoawq
```

For compatibility with Hugging Face Transformers, ensure you have the latest version:

```bash
pip install --upgrade transformers
```

## Using the AWQ-Quantized Model Script

We provide a dedicated script for working with AWQ-quantized models:

```bash
scripts\experiments\run_quantized_qwen.bat
```

This script:
1. Checks if your GPU supports AWQ
2. Installs necessary dependencies if missing
3. Provides a menu to select the model size based on your hardware
4. Configures the system appropriately for the selected model
5. Runs fine-tuning (if applicable) and evaluation

### Running the Script

1. Open a command prompt
2. Navigate to the FM-LLM-Solver directory
3. Run: `scripts\experiments\run_quantized_qwen.bat`
4. Choose the model size when prompted:
   - Option 1: 7B (recommended for most GPUs)
   - Option 2: 14B (requires 16GB+ VRAM)
   - Option 3: 72B (inference only, requires high-end GPU)

## Memory Requirements

### For RTX 3080 (10GB VRAM)

- **Recommended**: Use the Qwen2.5-7B-Instruct-AWQ model
- Fine-tuning the 14B model is possible but may require extreme optimization
- The 72B model can only be used for inference, not fine-tuning

### For RTX 3090/4090 (24GB VRAM)

- Can fine-tune Qwen2.5-14B-Instruct-AWQ
- Can run inference on Qwen2.5-72B-Instruct-AWQ 
- Can fine-tune Qwen2.5-7B-Instruct-AWQ with larger batch sizes

### For RTX 4060/4070 (8-12GB VRAM)

- Can fine-tune Qwen2.5-7B-Instruct-AWQ
- May run inference on Qwen2.5-14B-Instruct-AWQ

## Configuration Options

You can customize the AWQ model configuration in `config.yaml` by adding the following:

```yaml
fine_tuning:
  base_model_name: "Qwen/Qwen2.5-7B-Instruct-AWQ"  # Change to desired model
  quantization:
    quantization_method: "awq"                      # Specify AWQ method
    use_4bit: true
    use_nested_quant: true
  lora:
    r: 8            # Use 8 or 4 for memory savings
    alpha: 8        # Usually matches r value
  training:
    gradient_accumulation_steps: 8  # Increase for memory savings
    max_seq_length: 1024            # Limit input length
```

## Advanced: Manual AWQ Configuration

If you need more control over the AWQ process, you can manually create an AWQ-quantized model from a base model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from awq import AutoAWQForCausalLM

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Load full precision model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", 
                                          device_map="auto")

# Quantize model with AWQ
awq_model = AutoAWQForCausalLM.from_pretrained(
    model,
    tokenizer=tokenizer,
    quant_config={"zero_point": True, "q_group_size": 128}
)

# Save the quantized model
awq_model.save_pretrained("./Qwen2.5-7B-Instruct-AWQ")
tokenizer.save_pretrained("./Qwen2.5-7B-Instruct-AWQ")
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Use a smaller model size (7B instead of 14B)
   - Reduce LoRA rank (try 4 instead of 8)
   - Increase gradient accumulation steps (try 16)
   - Reduce sequence length (try 768 or 512)

2. **Model Loading Errors**:
   - Ensure you have the latest transformers library
   - Make sure AutoAWQ is installed (`pip install autoawq`)
   - Try using a non-AWQ model if issues persist

3. **Slow Performance**:
   - Set environment variables for better memory allocation:
     ```
     PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
     ```
   - Ensure other GPU-intensive applications are closed

## Further Resources

- [AutoAWQ GitHub Repository](https://github.com/casper-hansen/AutoAWQ)
- [AWQ Paper](https://arxiv.org/abs/2306.00978)
- [Qwen2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-AWQ)

For additional help, run the diagnostic tool:
```bash
scripts\optimization\diagnose_finetune.bat --all
``` 