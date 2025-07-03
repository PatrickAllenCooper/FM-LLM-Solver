# Multi-stage Dockerfile for FM-LLM Solver
# Production-ready container with comprehensive service support

# Base stage with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    FM_LLM_ENV=production \
    PYTHONPATH=/app

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
    && rm -rf /var/lib/apt/lists/*

# Create app directory and user
WORKDIR /app
RUN groupadd -r fmllm && useradd -r -g fmllm fmllm

# Install Python dependencies
COPY requirements.txt requirements/requirements.txt requirements/web_requirements.txt ./requirements/
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir -r requirements/requirements.txt && \
    pip3 install --no-cache-dir -r requirements/web_requirements.txt

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional production dependencies
RUN pip3 install --no-cache-dir \
    gunicorn \
    uvicorn \
    psycopg2-binary \
    redis \
    prometheus-client \
    cryptography

# Web interface stage
FROM base AS web

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p \
    /app/logs \
    /app/kb_data \
    /app/output \
    /app/data \
    /app/instance \
    /app/migrations

# Set permissions
RUN chown -R fmllm:fmllm /app && \
    chmod -R 755 /app

# Switch to non-root user
USER fmllm

# Expose web port
EXPOSE 5000

# Health check using our monitoring endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command for web interface using our new app factory
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--timeout", "300", "--access-logfile", "-", "--error-logfile", "-", "fm_llm_solver.web.app:create_app()"]

# CLI tools stage
FROM base AS cli

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p \
    /app/logs \
    /app/kb_data \
    /app/output \
    /app/data \
    /app/migrations

# Set permissions
RUN chown -R fmllm:fmllm /app && \
    chmod -R 755 /app && \
    chmod +x /app/scripts/fm-llm

# Switch to non-root user
USER fmllm

# Default command is our CLI tool
ENTRYPOINT ["/app/scripts/fm-llm"]

# Inference API stage (legacy support)
FROM base AS inference

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p \
    /app/logs \
    /app/kb_data \
    /app/output \
    /app/model_cache

# Set permissions
RUN chown -R fmllm:fmllm /app && \
    chmod -R 755 /app

# Switch to non-root user
USER fmllm

# Expose inference API port
EXPOSE 8000

# Health check for inference service
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for inference API (legacy)
CMD ["python", "-m", "inference_api.main"]

# Development stage with all tools
FROM base AS development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    nano \
    tmux \
    htop \
    tree \
    jq \
    git \
    ssh \
    && rm -rf /var/lib/apt/lists/*

# Copy everything
COPY . /app/

# Install all requirements including dev tools
RUN pip3 install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    flake8 \
    mypy \
    ipython \
    jupyter \
    pre-commit \
    coverage

# Create all necessary directories
RUN mkdir -p \
    /app/logs \
    /app/kb_data \
    /app/output \
    /app/data \
    /app/instance \
    /app/migrations \
    /app/model_cache \
    /app/.pytest_cache

# Set permissions for development
RUN chown -R fmllm:fmllm /app && \
    chmod -R 755 /app && \
    chmod +x /app/scripts/fm-llm

# Set up development environment
ENV FM_LLM_ENV=development \
    FLASK_DEBUG=1 \
    PYTHONPATH=/app

# Switch to non-root user
USER fmllm

# Expose all ports (web, api, monitoring, jupyter)
EXPOSE 5000 8000 9090 8888

# Create entrypoint script for development
USER root
RUN echo '#!/bin/bash\n\
if [ "$1" = "test" ]; then\n\
    exec pytest "${@:2}"\n\
elif [ "$1" = "web" ]; then\n\
    exec python -m fm_llm_solver.web.app\n\
elif [ "$1" = "cli" ]; then\n\
    exec /app/scripts/fm-llm "${@:2}"\n\
elif [ "$1" = "jupyter" ]; then\n\
    exec jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root\n\
else\n\
    exec "$@"\n\
fi' > /app/entrypoint-dev.sh && \
chmod +x /app/entrypoint-dev.sh

USER fmllm

# Default to bash for development
ENTRYPOINT ["/app/entrypoint-dev.sh"]
CMD ["/bin/bash"] 