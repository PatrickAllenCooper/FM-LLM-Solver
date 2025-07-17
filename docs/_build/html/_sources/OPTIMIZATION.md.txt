# Optimization Guide

Tips for running FM-LLM Solver on limited hardware.

## GPU Memory Management

### Quick Settings

For limited VRAM (8-12GB), add to `config.yaml`:

```yaml
model:
  quantization: "4bit"  # or "8bit"
  use_gradient_checkpointing: true

training:
  batch_size: 1
  gradient_accumulation_steps: 8
  max_length: 512
```

### Model Selection by VRAM

| VRAM | Recommended Model | Settings |
|------|------------------|----------|
| 6-8GB | Qwen2.5-7B | 4-bit quantization |
| 10-12GB | Qwen2.5-14B | 4-bit + gradient checkpointing |
| 16GB+ | Qwen2.5-14B | 8-bit or full precision |
| 24GB+ | Larger models | Full precision |

### Inference Optimization

```python
# Reduce memory usage
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    ),
    device_map="auto"
)
```

## CPU Optimization

For CPU-only systems:

1. Use smaller models (7B or less)
2. Disable GPU-specific features:
   ```yaml
   model:
     device: "cpu"
     torch_dtype: "float32"
   ```
3. Increase generation time limits

## Batch Processing

For multiple queries:

```python
# Process in batches to save memory
results = []
for batch in chunks(queries, batch_size=4):
    batch_results = process_batch(batch)
    results.extend(batch_results)
    torch.cuda.empty_cache()  # Clear GPU memory
```

## AWQ Models

For maximum compression with minimal quality loss:

```yaml
model:
  provider: "qwen_awq"
  name: "Qwen/Qwen2.5-14B-Instruct-AWQ"
  awq_config:
    version: "gemm"  # or "gemv" for smaller batch sizes
```

Benefits:
- 4x compression (14B → 3.5GB)
- 2-3x faster inference
- Minimal accuracy loss

## Knowledge Base Optimization

For faster KB building:

```yaml
paths:
  chunk_size: 512  # Smaller chunks
  chunk_overlap: 50
  
knowledge_base:
  batch_size: 100  # Process PDFs in batches
  use_gpu: false   # CPU processing if GPU needed elsewhere
```

## Monitoring Performance

```bash
# Check GPU usage
nvidia-smi -l 1

# Profile memory
python -m torch.utils.bottleneck your_script.py
```

## Common Issues

### OOM Errors
1. Reduce batch size
2. Enable gradient checkpointing
3. Use stronger quantization
4. Clear cache: `torch.cuda.empty_cache()`

### Slow Generation
1. Reduce `max_new_tokens`
2. Adjust `temperature` and `top_p`
3. Disable RAG if not needed
4. Use AWQ models

### Quality vs Speed Trade-offs
- 4-bit: Fastest, slight quality loss
- 8-bit: Good balance
- 16-bit: Best quality, more memory
- AWQ: Best compression, good quality 