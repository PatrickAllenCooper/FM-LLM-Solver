# FM-LLM Solver Hybrid Deployment Implementation

This document describes the hybrid deployment architecture implemented for the FM-LLM Solver, enabling cost-effective cloud deployment while maintaining local development capabilities.

## Overview

The hybrid deployment architecture separates the web interface from the GPU-intensive inference service, allowing you to:

- Run the web interface on cheap or free hosting (Vercel, Railway, local server)
- Deploy inference API only when needed on GPU cloud providers (RunPod, Modal, Vast.ai)
- Save 80-95% on costs compared to running dedicated GPU instances 24/7

## Architecture Changes

### 1. **Deployment Configuration**

Added comprehensive deployment configuration to `config/config.yaml`:

```yaml
deployment:
  mode: "local"  # Options: "local", "hybrid", "cloud"
  
  services:
    inference:
      type: "local"  # Options: "local", "modal", "runpod", "vastai"
      timeout: 300
      
  performance:
    enable_caching: true
    cache_ttl: 3600
```

### 2. **Separated Inference API**

Created a standalone FastAPI service in `inference_api/`:

- **`main.py`**: High-performance inference API with caching
- Supports batch processing
- Health checks and monitoring
- Redis caching support
- Model preloading on startup

### 3. **Enhanced Web Interface**

Updated `web_interface/certificate_generator.py` to support:

- Local mode (original functionality)
- Remote mode (calls inference API)
- Automatic mode detection based on configuration

### 4. **Containerization**

Created comprehensive Docker setup:

- **Multi-stage Dockerfile** for optimized images
- **docker-compose.yml** with service orchestration
- Support for development, production, and monitoring profiles

### 5. **Deployment Automation**

Created deployment scripts in `deployment/`:

- **`deploy.py`**: Unified deployment to multiple providers
- **`test_deployment.py`**: Comprehensive testing suite
- **Provider-specific configurations** for RunPod, Modal, Vast.ai, GCP

## Quick Start

### 1. **Local Development (No Changes)**

Run exactly as before:
```bash
python run_web_interface.py
```

### 2. **Local Docker Deployment**

```bash
./deploy.sh local
```

This runs both web and inference locally using Docker.

### 3. **Hybrid Deployment (Recommended)**

```bash
# Deploy inference to Modal (cheapest for intermittent use)
./deploy.sh hybrid --provider modal

# Or deploy to RunPod (good for consistent use)
./deploy.sh hybrid --provider runpod
```

### 4. **Full Cloud Deployment**

```bash
./deploy.sh cloud --provider gcp
```

## Cost Analysis

### Weekly Cost Estimates

| Usage Level | Requests/Day | Hybrid (Modal) | Hybrid (RunPod) | Dedicated GPU |
|------------|--------------|----------------|-----------------|---------------|
| Light      | 10           | $3.86          | $1.55           | $75          |
| Moderate   | 50           | $12.88         | $5.16           | $75          |
| Heavy      | 200          | $38.51         | $15.41          | $75          |

### Cost Optimization Features

1. **Result Caching**: Avoid recomputing identical requests
2. **Batch Processing**: Process multiple requests in one GPU session
3. **Auto-scaling**: Modal scales to zero when not in use
4. **Spot Instances**: Use cheaper spot/preemptible instances

## Configuration Guide

### Environment Variables

Create `.env` file from template:
```bash
cp config/env.example .env
```

Key variables:
```bash
# Deployment mode
DEPLOYMENT_MODE=hybrid

# For hybrid/cloud mode
INFERENCE_API_URL=https://your-api.modal.run
DATABASE_URL=postgresql://user:pass@host/db

# GPU settings
CUDA_VISIBLE_DEVICES=0
MODEL_CACHE_DIR=/tmp/models
```

### Provider-Specific Setup

#### Modal (Serverless GPU)

1. Install Modal CLI: `pip install modal`
2. Authenticate: `modal token new`
3. Deploy: `python deployment/deploy.py modal`

#### RunPod (Dedicated GPU Pods)

1. Get API key from RunPod dashboard
2. Set: `export RUNPOD_API_KEY=your-key`
3. Deploy: `python deployment/deploy.py runpod`

#### Vast.ai (GPU Marketplace)

1. Install CLI: `pip install vastai`
2. Set API key: `vastai set api-key YOUR_KEY`
3. Find instance: `vastai search offers 'RTX 3090'`
4. Deploy: `python deployment/deploy.py vastai`

## Monitoring

### Health Checks

- Web interface: `http://localhost:5000/api/stats`
- Inference API: `http://localhost:8000/health`

### Enable Monitoring Stack

```bash
docker compose --profile monitoring up -d
```

Access:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## Testing

Run comprehensive tests:
```bash
python deployment/test_deployment.py
```

This verifies:
- Configuration validity
- Docker setup
- API structure
- Deployment scripts
- Web interface compatibility

## Troubleshooting

### Common Issues

1. **"Cannot connect to inference API"**
   - Check `INFERENCE_API_URL` is set correctly
   - Verify the inference service is running
   - Test with: `curl $INFERENCE_API_URL/health`

2. **"Model loading failed"**
   - Ensure models are in `output/finetuning_results/`
   - Check GPU memory availability
   - Verify CUDA installation

3. **"Docker compose not found"**
   - Use `docker compose` (with space) for newer Docker
   - Or install docker-compose separately

### Debug Commands

```bash
# View logs
./deploy.sh logs

# Check GPU in container
docker exec fm-llm-inference nvidia-smi

# Test inference directly
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"system_description": "test", "model_config": "base"}'
```

## Migration Guide

### From Existing Installation

No changes required! The system maintains full backward compatibility:

1. Original scripts work unchanged
2. Local mode is the default
3. All existing functionality preserved
4. New features are opt-in via configuration

### To Hybrid Deployment

1. Update configuration:
   ```yaml
   deployment:
     mode: "hybrid"
   ```

2. Set inference API URL:
   ```bash
   export INFERENCE_API_URL=https://your-api-endpoint
   ```

3. Deploy inference service to cloud

4. Run web interface locally or deploy to free hosting

## Production Checklist

Before deploying to production:

- [ ] Set secure `SECRET_KEY`
- [ ] Configure PostgreSQL instead of SQLite
- [ ] Enable HTTPS/SSL
- [ ] Set up monitoring alerts
- [ ] Configure backups
- [ ] Test failover procedures
- [ ] Document API endpoints
- [ ] Enable rate limiting
- [ ] Set up CI/CD pipeline

## Support

For deployment issues:
1. Check deployment logs: `./deploy.sh logs`
2. Run tests: `./deploy.sh test`
3. See `deployment/README.md` for detailed provider guides
4. Check GitHub issues for common problems 