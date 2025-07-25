# Multi-stage Dockerfile for FM-LLM Solver
# Supports both web interface and inference components
# Usage:
#   Full stack: docker build --target production .
#   Web only:   docker build --target web .
#   Inference:  docker build --target inference .

# ============================================================================
# Base Stage - Common dependencies
# ============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app \
    FM_LLM_ENV=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# Create app directory and user
WORKDIR /app
RUN groupadd -r fmllm && useradd -r -g fmllm fmllm

# Copy requirements files
COPY requirements/ ./requirements/
COPY requirements.txt ./

# Install base requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements/base.txt

# ============================================================================
# Web Stage - Web interface optimized (CPU)
# ============================================================================
FROM base AS web

# Install web-specific requirements
RUN pip install --no-cache-dir -r requirements/web.txt

# Copy application code
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p /app/instance /app/logs /app/cache /app/uploads && \
    chown -R fmllm:fmllm /app

# Copy entrypoint
COPY deployment/docker/docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER fmllm
EXPOSE 5000

CMD ["/entrypoint.sh", "web"]

# ============================================================================
# Inference Stage - ML/AI optimized (GPU)
# ============================================================================
FROM base AS inference

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install inference-specific requirements
RUN pip install --no-cache-dir -r requirements/inference.txt

# Copy application code
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p /app/logs /app/cache /app/models && \
    chown -R fmllm:fmllm /app

# Copy entrypoint
COPY deployment/docker/docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER fmllm
EXPOSE 8000

CMD ["/entrypoint.sh", "inference"]

# ============================================================================
# Production Stage - Full stack (both web + inference)
# ============================================================================
FROM base AS production

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install all requirements
RUN pip install --no-cache-dir \
    -r requirements/web.txt \
    -r requirements/inference.txt \
    -r requirements/production.txt

# Copy application code
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p /app/instance /app/logs /app/cache /app/uploads /app/models && \
    chown -R fmllm:fmllm /app

# Copy production configuration
COPY config/config-production.yaml /app/config/config.yaml

# Copy entrypoint
COPY deployment/docker/docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER fmllm
EXPOSE 5000 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["/entrypoint.sh", "both"] 