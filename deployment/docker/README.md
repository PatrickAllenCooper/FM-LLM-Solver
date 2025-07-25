# Docker Deployment Guide

This directory contains consolidated, optimized Docker configurations for FM-LLM Solver.

## 🏗️ Available Images

### 1. **Main Dockerfile** - Multi-Stage Full Stack
```bash
# Full production stack (web + inference)
docker build --target production -t fm-llm-solver:full .

# Web interface only (CPU optimized)
docker build --target web -t fm-llm-solver:web .

# Inference API only (GPU optimized)
docker build --target inference -t fm-llm-solver:inference .
```

### 2. **Dockerfile.web** - Web Interface Only
```bash
# Lightweight web interface (CPU only)
docker build -f Dockerfile.web -t fm-llm-solver:web-only .
```

### 3. **Dockerfile.inference** - ML/Inference Only
```bash
# GPU-optimized inference container
docker build -f Dockerfile.inference -t fm-llm-solver:inference-only .
```

### 4. **Dockerfile.dev** - Development Environment
```bash
# Full development environment with tools
docker build -f Dockerfile.dev -t fm-llm-solver:dev .
```

## 🚀 Quick Start

### Option 1: Full Stack (Production)
```bash
# Build and run complete system
docker build --target production -t fm-llm-solver:full .
docker run -p 5000:5000 -p 8000:8000 fm-llm-solver:full
```

### Option 2: Web Interface Only
```bash
# Build and run web interface only
docker build --target web -t fm-llm-solver:web .
docker run -p 5000:5000 fm-llm-solver:web
```

### Option 3: Inference API Only
```bash
# Build and run inference API only (requires GPU)
docker build --target inference -t fm-llm-solver:inference .
docker run --gpus all -p 8000:8000 fm-llm-solver:inference
```

### Option 4: Development Mode
```bash
# Build and run development environment
docker build -f Dockerfile.dev -t fm-llm-solver:dev .

# Interactive development shell
docker run -it --gpus all -v $(pwd):/app fm-llm-solver:dev

# Run specific development tasks
docker run -it fm-llm-solver:dev dev web       # Web dev mode
docker run -it fm-llm-solver:dev dev inference # Inference dev mode
docker run -it fm-llm-solver:dev dev test      # Run tests
docker run -it fm-llm-solver:dev dev jupyter   # Jupyter notebook
```

## 🐳 Docker Compose

Use the provided compose files for complete orchestration:

```bash
# Local development
cd deployment/docker
docker-compose -f docker-compose.simple.yml up

# Production deployment
docker-compose -f docker-compose.yml up

# Hybrid deployment (web local, inference remote)
docker-compose -f docker-compose.hybrid.yml up
```

## 🎯 Entrypoint Modes

The unified entrypoint script supports different modes:

| Mode | Description | Ports | Use Case |
|------|-------------|--------|----------|
| `web` | Web interface only | 5000 | Frontend deployment |
| `inference` | ML API only | 8000 | GPU-optimized inference |
| `both` | Full stack | 5000, 8000 | Complete system |
| `dev` | Development mode | 5000, 8000, 8888 | Development and testing |

### Examples:
```bash
# Default web mode
docker run fm-llm-solver:full

# Explicit inference mode
docker run fm-llm-solver:full inference

# Development with specific command
docker run fm-llm-solver:dev dev test --verbose

# Development Jupyter notebook
docker run -p 8888:8888 fm-llm-solver:dev dev jupyter
```

## 🔧 Environment Variables

### Common Variables
```bash
# Application mode
FM_LLM_ENV=production          # production, development, testing

# Database
DATABASE_URL=postgresql://...  # PostgreSQL connection string
SQLALCHEMY_DATABASE_URI=...    # Alternative database config

# Redis (optional)
REDIS_URL=redis://...          # Redis connection string

# Security
SECRET_KEY=...                 # Flask secret key
JWT_SECRET_KEY=...             # JWT signing key

# API Configuration
INFERENCE_API_URL=...          # External inference API URL
```

### GPU Configuration
```bash
# CUDA settings
CUDA_VISIBLE_DEVICES=0                    # GPU device selection
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Memory management
```

### Web Interface
```bash
# Web server
GUNICORN_WORKERS=4             # Number of worker processes
GUNICORN_TIMEOUT=120           # Request timeout (seconds)

# Features
ENABLE_USER_REGISTRATION=true  # Allow new user signup
ENABLE_API_KEYS=true          # Enable API key authentication
```

### Development
```bash
# Development settings
FLASK_DEBUG=1                  # Enable debug mode
FLASK_ENV=development         # Flask environment
```

## 📦 Multi-Architecture Builds

Build for different architectures:

```bash
# Intel/AMD64
docker build --platform linux/amd64 -t fm-llm-solver:amd64 .

# ARM64 (Apple Silicon, etc.)
docker build --platform linux/arm64 -t fm-llm-solver:arm64 .

# Multi-platform build
docker buildx build --platform linux/amd64,linux/arm64 -t fm-llm-solver:multi .
```

## 🚨 GPU Requirements

For inference containers:

### NVIDIA GPU Support
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

apt-get update && apt-get install -y nvidia-docker2
systemctl restart docker

# Run with GPU support
docker run --gpus all fm-llm-solver:inference
```

### Resource Requirements
- **Web Only**: 2GB RAM, 1 CPU core
- **Inference**: 8GB+ GPU VRAM, 16GB+ RAM, 4+ CPU cores
- **Full Stack**: 8GB+ GPU VRAM, 24GB+ RAM, 6+ CPU cores

## 🔍 Troubleshooting

### Common Issues

1. **Port conflicts**
   ```bash
   # Check what's using ports
   lsof -i :5000
   lsof -i :8000
   
   # Use different ports
   docker run -p 5001:5000 fm-llm-solver:web
   ```

2. **GPU not detected**
   ```bash
   # Verify GPU access
   docker run --gpus all nvidia/cuda:11.8-runtime-ubuntu22.04 nvidia-smi
   
   # Check CUDA in container
   docker exec container_name nvidia-smi
   ```

3. **Memory issues**
   ```bash
   # Monitor memory usage
   docker stats
   
   # Limit memory usage
   docker run --memory=16g fm-llm-solver:inference
   ```

4. **Permission issues**
   ```bash
   # Fix file permissions
   docker run --user $(id -u):$(id -g) fm-llm-solver:web
   ```

### Logs and Debugging
```bash
# View container logs
docker logs container_name

# Interactive debugging
docker exec -it container_name /bin/bash

# Check health status
docker inspect container_name | grep Health -A 10
```

## 📈 Performance Optimization

### Production Recommendations
```bash
# Optimize for production
docker run \
  --memory=16g \
  --cpus=4 \
  --restart=unless-stopped \
  --health-cmd="curl -f http://localhost:5000/health || exit 1" \
  --health-interval=30s \
  --health-retries=3 \
  fm-llm-solver:full
```

### Development Tips
```bash
# Use volume mounts for hot reload
docker run -v $(pwd):/app fm-llm-solver:dev dev web

# Enable development features
docker run -e FLASK_DEBUG=1 -e FLASK_ENV=development fm-llm-solver:dev
```

## 🔒 Security

### Production Security
- Always use custom `SECRET_KEY` and `JWT_SECRET_KEY`
- Enable HTTPS/SSL in production
- Use secure database connections
- Regularly update base images
- Scan images for vulnerabilities

### Example Secure Deployment
```bash
docker run \
  -e SECRET_KEY="$(openssl rand -base64 32)" \
  -e JWT_SECRET_KEY="$(openssl rand -base64 32)" \
  -e DATABASE_URL="postgresql://user:pass@host:5432/db?sslmode=require" \
  --read-only \
  --tmpfs /tmp \
  --security-opt no-new-privileges \
  fm-llm-solver:full
``` 