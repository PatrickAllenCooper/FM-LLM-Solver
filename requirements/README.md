# Requirements Structure

This directory contains modular requirements files for different use cases:

## Core Files

- **`base.txt`** - Core dependencies required by all components
- **`web.txt`** - Web interface and Flask dependencies  
- **`inference.txt`** - Machine learning and LLM dependencies
- **`dev.txt`** - Development, testing, and code quality tools
- **`production.txt`** - Production monitoring and deployment tools

## Installation Examples

### Web Interface Only
```bash
pip install -r requirements/base.txt -r requirements/web.txt
```

### Full ML/Inference Setup
```bash
pip install -r requirements/base.txt -r requirements/inference.txt
```

### Development Environment  
```bash
pip install -r requirements/base.txt -r requirements/dev.txt
```

### Production Deployment
```bash
pip install -r requirements/base.txt -r requirements/web.txt -r requirements/production.txt
```

### Everything (Development + ML)
```bash
pip install -r requirements/base.txt -r requirements/web.txt -r requirements/inference.txt -r requirements/dev.txt
```

## Notes

- Always install `base.txt` first as it contains core dependencies
- For GPU support, install PyTorch separately with CUDA support
- See individual files for specific version requirements and notes 