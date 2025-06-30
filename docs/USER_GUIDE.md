# User Guide

This guide covers all aspects of using FM-LLM Solver, from basic certificate generation to advanced features.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Building Knowledge Base](#building-knowledge-base)
3. [Generating Certificates](#generating-certificates)
4. [Fine-Tuning](#fine-tuning)
5. [Web Interface](#web-interface)
6. [Evaluation & Benchmarking](#evaluation--benchmarking)
7. [Advanced Features](#advanced-features)

## Getting Started

### Configuration

The main configuration file is `config/config.yaml`. Key settings:

```yaml
model:
  provider: "qwen"
  name: "Qwen/Qwen2.5-14B-Instruct"
  use_finetuned: true  # Use fine-tuned model if available

rag:
  enabled: true
  k_retrieved: 3  # Number of context documents

deployment:
  mode: "local"  # local, hybrid, or cloud
```

### Quick Test

Generate your first barrier certificate:

```bash
python inference/generate_certificate.py \
  "System: dx/dt = -x^3 - y, dy/dt = x - y^3. Initial: x^2+y^2 <= 0.1. Unsafe: x >= 1.5"
```

## Building Knowledge Base

### 1. Fetch Papers

```bash
# Set environment variable
export UNPAYWALL_EMAIL='your-email@example.com'

# Fetch papers (uses data/user_ids.csv for author list)
python data_fetching/paper_fetcher.py
```

### 2. Build Knowledge Base

```bash
# Set Mathpix credentials
export MATHPIX_APP_ID='your_app_id'
export MATHPIX_APP_KEY='your_app_key'

# Build knowledge base
python knowledge_base/knowledge_base_builder.py
```

The knowledge base will be saved to `kb_data/` or `output/knowledge_base/`.

### 3. Test Knowledge Base

```bash
python knowledge_base/test_knowledge_base.py "What is a barrier certificate?" -k 3
```

## Generating Certificates

### Command Line Interface

Basic usage:
```bash
python inference/generate_certificate.py "System description"
```

With options:
```bash
python inference/generate_certificate.py \
  "System: dx/dt = -x + y, dy/dt = -x - y" \
  --model_config base \
  --rag_k 5 \
  --kb_path kb_data/
```

### Python API

```python
from inference.generate_certificate import generate_barrier_certificate

result = generate_barrier_certificate(
    system_description="dx/dt = -x^3 - y, dy/dt = x - y^3",
    model_config="finetuned",
    rag_k=3
)

print(f"Certificate: {result['certificate']}")
print(f"Confidence: {result['confidence']}")
```

### System Description Format

Include these components:
- **System Dynamics**: The differential equations
- **Initial Set**: Where trajectories start
- **Unsafe Set**: Region to avoid

Example:
```
System Dynamics: dx/dt = -x^3 - y, dy/dt = x - y^3
Initial Set: x^2 + y^2 <= 0.1
Unsafe Set: x >= 1.5 or y >= 1.5
```

## Fine-Tuning

### Prepare Training Data

1. **Manual creation**:
   ```bash
   python fine_tuning/create_finetuning_data.py
   ```

2. **Extract from papers**:
   ```bash
   python fine_tuning/extract_from_papers.py
   ```

3. **Combine datasets**:
   ```bash
   python fine_tuning/combine_datasets.py
   ```

### Train Model

```bash
# Ensure CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Fine-tune
python fine_tuning/finetune_llm.py
```

Training parameters can be adjusted in `config.yaml`:
```yaml
training:
  num_epochs: 3
  batch_size: 4
  learning_rate: 2e-4
  gradient_accumulation_steps: 4
```

## Web Interface

### Start the Web Server

```bash
# Initialize security (first time only)
python scripts/init_security.py

# Start server
python run_web_interface.py
```

Access at http://localhost:5000

### Features

- **Interactive Chat**: Conversational interface for certificate generation
- **History**: View past generations and conversations
- **Authentication**: Secure login with rate limiting
- **API Access**: RESTful endpoints for programmatic use
- **Monitoring**: Dashboard with usage statistics

### API Endpoints

Generate certificate:
```bash
curl -X POST http://localhost:5000/api/generate \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "system_description": "dx/dt = -x^3 - y, dy/dt = x - y^3",
    "model_config": "finetuned"
  }'
```

Check status:
```bash
curl http://localhost:5000/api/status
```

## Evaluation & Benchmarking

### Run Benchmark Suite

```bash
# Evaluate on all benchmark systems
python evaluation/evaluate_pipeline.py

# With custom parameters
python evaluation/evaluate_pipeline.py -k 5 --output results.csv
```

### Add Custom Benchmarks

Edit `data/benchmark_systems.json`:
```json
{
  "systems": [
    {
      "name": "My System",
      "dynamics": "dx/dt = -x + u, du/dt = -u",
      "initial_set": "x^2 + u^2 <= 1",
      "unsafe_set": "x >= 2",
      "expected_certificate": "V(x,u) = x^2 + u^2"
    }
  ]
}
```

### Model Comparison

Compare base vs fine-tuned model:
```bash
python scripts/analysis/compare_models.py
```

## Advanced Features

### Domain Bounds

Specify validity regions for certificates:
```python
result = generate_barrier_certificate(
    system_description="...",
    domain_bounds={
        "x": [-2, 2],
        "y": [-2, 2]
    }
)
```

### Discrete-Time Systems

For discrete-time systems, use appropriate syntax:
```
System: x[k+1] = 0.9*x[k] + 0.1*u[k], u[k+1] = -0.1*x[k] + 0.9*u[k]
```

### Stochastic Systems

Include noise terms in the dynamics:
```
System: dx = (-x^3 - y)dt + 0.1*dW1, dy = (x - y^3)dt + 0.1*dW2
```

### Custom Verification

Use the verification module directly:
```python
from evaluation.verify_certificate import verify_barrier_certificate

result = verify_barrier_certificate(
    dynamics={"x": "-x**3 - y", "y": "x - y**3"},
    certificate="x**4 + y**4",
    initial_set="x**2 + y**2 <= 0.1",
    unsafe_set="x >= 1.5"
)
```

## Tips & Best Practices

1. **System Description**: Be precise and use standard mathematical notation
2. **RAG Context**: Start with k=3, increase if results are poor
3. **Verification**: Always verify generated certificates before use
4. **Fine-Tuning**: Use domain-specific examples for best results
5. **Memory**: Enable quantization for large models on limited GPUs

## Troubleshooting

### Common Issues

- **GPU Memory**: Reduce batch size or enable quantization
- **Slow Generation**: Disable RAG or reduce k_retrieved
- **Poor Results**: Ensure knowledge base is built correctly
- **API Errors**: Check credentials and rate limits

### Getting Help

- Check [API Reference](API_REFERENCE.md) for detailed documentation
- Review [Development Guide](DEVELOPMENT.md) for technical details
- Submit issues on GitHub for bugs or feature requests 