# Installation Guide

This guide provides detailed instructions for setting up FM-LLM Solver on your system.

## System Requirements

### Hardware
- **CPU**: Modern multi-core processor (Intel/AMD)
- **GPU**: NVIDIA GPU with CUDA support (GTX 1060 or better)
  - Minimum 6GB VRAM for inference
  - 12GB+ VRAM recommended for fine-tuning
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB free space

### Software
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows 10/11
- **Python**: 3.10 or 3.11 (3.12 also supported)
- **CUDA**: 11.7 or 11.8 (for GPU support)
- **Git**: For repository cloning

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/PatrickAllenCooper/FM-LLM-Solver.git
cd FM-LLM-Solver
```

### 2. Set Up Python Environment

Using Conda (recommended):
```bash
conda create -n fmllm python=3.10
conda activate fmllm
```

Using venv:
```bash
python -m venv fmllm-env
source fmllm-env/bin/activate  # Linux/macOS
# or
fmllm-env\Scripts\activate  # Windows
```

### 3. Install Dependencies

**Automated installation with CUDA support:**
```bash
python scripts/setup/setup_environment.py
```

This script will:
- Detect your CUDA version
- Install PyTorch with appropriate CUDA support
- Install all required dependencies
- Verify the installation

**Manual installation:**
```bash
# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements/requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file or export these variables:

```bash
# Required for knowledge base building
export MATHPIX_APP_ID='your_mathpix_app_id'
export MATHPIX_APP_KEY='your_mathpix_app_key'

# Required for paper fetching
export UNPAYWALL_EMAIL='your-email@example.com'

# Optional
export SEMANTIC_SCHOLAR_API_KEY='your_api_key'  # For enhanced paper fetching
export OPENAI_API_KEY='your_openai_key'         # If using OpenAI models
```

### 5. Verify Installation

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Test basic functionality
python knowledge_base/test_knowledge_base.py "What is a barrier certificate?" -k 3
```

## Deployment Options

### Local Deployment

Default configuration works out of the box for local use.

### Hybrid Deployment (Recommended for Production)

For cost-effective cloud deployment, configure `config/config.yaml`:

```yaml
deployment:
  mode: hybrid  # local, hybrid, or cloud
  inference_api_url: "https://your-gpu-instance.com/api"
```

See [Hybrid Deployment Guide](../HYBRID_DEPLOYMENT.md) for details.

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t fm-llm-solver .
docker run -p 5000:5000 --gpus all fm-llm-solver
```

## Troubleshooting

### CUDA Not Available

If `torch.cuda.is_available()` returns False:

1. Check NVIDIA drivers:
   ```bash
   nvidia-smi
   ```

2. Reinstall PyTorch with correct CUDA version:
   ```bash
   python scripts/setup/setup_environment.py --force-reinstall
   ```

3. For WSL2 users, ensure CUDA support is properly configured.

### Memory Issues

For large models on limited GPU memory:

1. Enable gradient checkpointing in `config.yaml`:
   ```yaml
   model:
     use_gradient_checkpointing: true
   ```

2. Use quantization:
   ```yaml
   model:
     quantization: "4bit"  # or "8bit"
   ```

3. Reduce batch size:
   ```yaml
   training:
     batch_size: 1
   ```

### API Key Issues

- **Mathpix**: Sign up at https://mathpix.com/ocr-api
- **Unpaywall**: Use your academic email at https://unpaywall.org/products/api

## Next Steps

After installation:
1. [Build the knowledge base](USER_GUIDE.md#building-knowledge-base)
2. [Fine-tune the model](USER_GUIDE.md#fine-tuning) (optional)
3. [Generate barrier certificates](USER_GUIDE.md#generating-certificates) 