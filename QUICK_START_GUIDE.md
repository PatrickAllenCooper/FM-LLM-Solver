# Quick Start Guide

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: 8GB+ VRAM)
- Git

## Installation

```bash
git clone https://github.com/PatrickAllenCooper/FM-LLM-Solver.git
cd FM-LLM-Solver
pip install -r requirements.txt
```

## 30-Second Test

```bash
# Test real LLM GPU pipeline
python quick_gpu_test.py
```

Expected output:
```
✅ GPU detected: NVIDIA GeForce RTX 4070
✅ Model loaded in 18.7s
✅ Generation completed in 9.4s
✅ Extracted: 'x**2 + y**2 - 1.5'
🎉 SUCCESS: Real LLM pipeline working!
```

## Usage Options

### 1. Web Interface
```bash
python run_web_interface.py
# Open http://localhost:5000
```

### 2. Python API
```python
from fm_llm_solver import CertificateGenerator

generator = CertificateGenerator.from_config()
result = generator.generate(system_description)
```

### 3. Comprehensive Testing
```bash
# Test multiple systems
python tests/gpu_real_llm_tests.py

# Compare mock vs real
python tests/compare_mock_vs_real_llm.py
```

## Configuration

Create `config.yaml`:
```yaml
model:
  provider: qwen
  name: Qwen/Qwen2.5-7B-Instruct
  device: cuda
  quantization: 4bit
```

## Troubleshooting

### GPU Issues
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues
- Use 4-bit quantization for 8GB GPUs
- Use CPU mode if GPU insufficient: `device: cpu`

### Model Loading
- First load takes 30-60 seconds
- Models cached after first download

## Next Steps

1. ✅ **Basic Testing**: Run `quick_gpu_test.py`
2. 📊 **Performance**: Check `tests/gpu_real_llm_tests.py` results  
3. 🌐 **Web Interface**: Launch `run_web_interface.py`
4. 📚 **Documentation**: See `docs/` for detailed guides

---
*For complete documentation, see the main [README](README.md).* 