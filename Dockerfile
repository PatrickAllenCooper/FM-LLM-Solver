# Multi-stage Dockerfile for FM-LLM Solver
# Supports both web interface and inference components

# Base stage with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements/requirements.txt requirements/web_requirements.txt ./requirements/
RUN pip3 install --no-cache-dir -r requirements/requirements.txt && \
    pip3 install --no-cache-dir flask flask-sqlalchemy gunicorn

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Web interface stage
FROM base AS web

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/web_interface/instance \
    /app/output/finetuning_results \
    /app/kb_data \
    /app/logs

# Set permissions
RUN chmod -R 755 /app

# Expose web port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/api/stats || exit 1

# Default command for web interface
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--timeout", "300", "--access-logfile", "-", "web_interface.app:app"]

# Inference API stage
FROM base AS inference

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/output/finetuning_results \
    /app/kb_data \
    /app/logs \
    /app/model_cache

# Install additional inference dependencies
RUN pip3 install --no-cache-dir fastapi uvicorn pydantic redis

# Expose inference API port
EXPOSE 8000

# Health check for inference service
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for inference API
CMD ["uvicorn", "inference_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Development stage with all tools
FROM base AS development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    tmux \
    htop \
    nvidia-utils-525 \
    && rm -rf /var/lib/apt/lists/*

# Copy everything
COPY . /app/

# Install all requirements including dev tools
RUN pip3 install --no-cache-dir pytest pytest-cov black flake8 ipython

# Create all necessary directories
RUN mkdir -p /app/web_interface/instance \
    /app/output/finetuning_results \
    /app/kb_data \
    /app/logs \
    /app/model_cache

# Set up development environment
ENV FLASK_ENV=development \
    FLASK_DEBUG=1

# Expose all ports
EXPOSE 5000 8000 9090

# Default to bash for development
CMD ["/bin/bash"] 