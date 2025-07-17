# 🚀 Deployment Guide

**Get FM-LLM Solver running in under 5 minutes!**

## ⚡ One-Command Deployment (Recommended)

### Prerequisites
- **Docker** + **Docker Compose** + **NVIDIA Container Toolkit**
- **CUDA-capable GPU** (8GB+ VRAM recommended)
- **Linux/WSL2** (Windows users use WSL2)

### Quick Setup
```bash
# Clone repository
git clone https://github.com/PatrickAllenCooper/FM-LLM-Solver.git
cd FM-LLM-Solver

# One-command deployment
chmod +x deploy_simple.sh
./deploy_simple.sh
```

**That's it!** 🎉 Access at http://localhost:5000

---

## 🐳 Manual Docker Deployment

### 1. Build and Run
```bash
# Build image
docker-compose -f docker-compose.simple.yml build

# Start service
docker-compose -f docker-compose.simple.yml up -d

# Check status
docker-compose -f docker-compose.simple.yml ps
```

### 2. Test GPU Pipeline
```bash
# Test real LLM GPU inference
docker exec fm-llm-solver python3 quick_gpu_test.py

# Run comprehensive tests
docker exec fm-llm-solver python3 tests/gpu_real_llm_tests.py
```

### 3. Management Commands
```bash
# View logs
docker-compose -f docker-compose.simple.yml logs -f

# Stop service
docker-compose -f docker-compose.simple.yml down

# Restart
docker-compose -f docker-compose.simple.yml restart
```

---

## 🖥️ Local Development Setup

### Prerequisites
- **Python 3.8+**
- **CUDA-capable GPU** with drivers
- **Git**

### Setup
```bash
# Clone and install
git clone https://github.com/PatrickAllenCooper/FM-LLM-Solver.git
cd FM-LLM-Solver
pip install -r requirements.txt

# Test immediately
python quick_gpu_test.py

# Start web interface
python run_web_interface.py
```

---

## ⚙️ Configuration

### Basic Config (`config.yaml`)
```yaml
model:
  provider: qwen
  name: Qwen/Qwen2.5-7B-Instruct
  device: cuda          # Use 'cpu' if no GPU
  quantization: 4bit     # For 8GB GPUs
  temperature: 0.1

verification:
  numerical_samples: 200
  tolerance: 0.1
```

### GPU Memory Optimization
| GPU Memory | Model Size | Quantization |
|------------|------------|--------------|
| 8GB | 7B | 4bit |
| 16GB | 7B | 8bit or fp16 |
| 24GB+ | 14B | 4bit |

---

## 🔧 Troubleshooting

### GPU Issues
```bash
# Check GPU availability
nvidia-smi

# Test CUDA in Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Common Fixes
**Out of Memory**:
- Use `quantization: 4bit` in config
- Reduce `numerical_samples: 100`
- Use smaller model: `Qwen/Qwen2.5-3B-Instruct`

**Model Download Fails**:
- Check internet connection
- Clear cache: `rm -rf ~/.cache/huggingface/`

**Web Interface Not Loading**:
- Check logs: `docker-compose -f docker-compose.simple.yml logs`
- Try different port: Edit `docker-compose.simple.yml`

---

## 📊 Verification

### Expected Results
```bash
# Quick test should show:
✅ GPU detected: NVIDIA GeForce RTX 4070
✅ Model loaded in 18-27s
✅ Generation completed in 9-10s
✅ Extracted: 'x**2 + y**2 - 1.5'
🎉 SUCCESS: Real LLM pipeline working!
```

### Performance Metrics
- **Model Load**: 18-27s (first time)
- **Generation**: 9-10s per certificate
- **Memory**: ~5.6GB GPU usage
- **Success Rate**: 20% end-to-end

---

## 🎯 Quick Commands Reference

```bash
# One-command deploy
./deploy_simple.sh

# Test GPU pipeline
python quick_gpu_test.py

# Web interface
python run_web_interface.py

# Comprehensive testing
python tests/gpu_real_llm_tests.py

# Docker management
docker-compose -f docker-compose.simple.yml up -d
docker-compose -f docker-compose.simple.yml down
```

---

## 🆘 Need Help?

1. **Check logs**: `docker-compose -f docker-compose.simple.yml logs`
2. **Test basics**: `python quick_gpu_test.py`
3. **Verify GPU**: `nvidia-smi`
4. **Check documentation**: See `docs/` directory

**Most issues are GPU/CUDA related** - ensure NVIDIA drivers and Docker GPU support are properly installed.

---

*🎉 You're ready to generate barrier certificates with real LLM GPU inference!* 