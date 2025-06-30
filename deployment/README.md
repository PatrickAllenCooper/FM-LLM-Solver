# FM-LLM Solver Deployment Guide

This guide provides comprehensive instructions for deploying the FM-LLM Solver in various environments, from local development to production cloud deployments.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Deployment Modes](#deployment-modes)
3. [Local Development](#local-development)
4. [Cloud Deployments](#cloud-deployments)
   - [RunPod](#runpod-deployment)
   - [Modal](#modal-deployment)
   - [Vast.ai](#vastai-deployment)
   - [Google Cloud Platform](#gcp-deployment)
5. [Configuration](#configuration)
6. [Monitoring](#monitoring)
7. [Cost Optimization](#cost-optimization)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Python 3.10+
- GPU with CUDA support (for inference)
- Cloud provider accounts (for cloud deployments)

### Basic Deployment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/FM-LLM-Solver.git
   cd FM-LLM-Solver
   ```

2. **Copy and configure environment variables:**
   ```bash
   cp config/env.example .env
   # Edit .env with your configurations
   ```

3. **Deploy locally:**
   ```bash
   python deployment/deploy.py local
   ```

## Deployment Modes

### 1. Local Mode
- All services run on local machine
- Requires GPU for inference
- Best for development and testing

### 2. Hybrid Mode
- Web interface runs locally or on lightweight server
- Inference API runs on GPU cloud
- Cost-effective for intermittent use

### 3. Cloud Mode
- All services run in cloud
- Scalable and production-ready
- Higher cost but better reliability

## Local Development

### Using Docker Compose

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Without Docker (Native)

```bash
# Start web interface
python run_web_interface.py

# In another terminal, start inference API
cd inference_api
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Cloud Deployments

### RunPod Deployment

RunPod offers affordable GPU instances with good availability.

1. **Set up RunPod API key:**
   ```bash
   export RUNPOD_API_KEY="your-api-key"
   ```

2. **Deploy:**
   ```bash
   python deployment/deploy.py runpod
   ```

3. **Manual deployment via RunPod UI:**
   - Go to [RunPod Console](https://runpod.io)
   - Create new pod with:
     - GPU: RTX 3090 (24GB)
     - Container: `fm-llm-solver:inference`
     - Ports: 8000

### Modal Deployment

Modal provides serverless GPU computing with automatic scaling.

1. **Install Modal CLI:**
   ```bash
   pip install modal
   modal token new
   ```

2. **Deploy:**
   ```bash
   python deployment/deploy.py modal
   ```

3. **Check deployment:**
   ```bash
   modal app list
   ```

### Vast.ai Deployment

Vast.ai offers the cheapest GPU rates through a marketplace model.

1. **Install Vast CLI:**
   ```bash
   pip install vastai
   vastai set api-key YOUR_API_KEY
   ```

2. **Find suitable instance:**
   ```bash
   vastai search offers 'RTX 3090' --disk 50 --inet_down 100
   ```

3. **Deploy:**
   ```bash
   # Note the instance ID from search
   python deployment/deploy.py vastai --instance-id <ID>
   ```

### GCP Deployment

For enterprise deployments with Google Cloud Platform.

1. **Prerequisites:**
   - GCP account with billing enabled
   - `gcloud` CLI installed and configured
   - Kubernetes cluster with GPU nodes

2. **Build and push images:**
   ```bash
   # Build images
   docker build -t gcr.io/YOUR_PROJECT/fm-llm-solver:web --target web .
   docker build -t gcr.io/YOUR_PROJECT/fm-llm-solver:inference --target inference .
   
   # Push to GCR
   docker push gcr.io/YOUR_PROJECT/fm-llm-solver:web
   docker push gcr.io/YOUR_PROJECT/fm-llm-solver:inference
   ```

3. **Deploy to GKE:**
   ```bash
   # Update project ID in k8s configs
   sed -i 's/PROJECT_ID/YOUR_PROJECT/g' deployment/k8s/*.yaml
   
   # Apply configurations
   kubectl apply -f deployment/k8s/
   ```

## Configuration

### Environment Variables

Key environment variables for deployment:

```bash
# Deployment mode
DEPLOYMENT_MODE=hybrid  # local, hybrid, cloud

# Service URLs
INFERENCE_API_URL=https://your-inference-api.modal.run
DATABASE_URL=postgresql://user:pass@host:5432/db

# GPU settings
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Performance
CACHE_TTL=3600
BATCH_SIZE=5
```

### Model Storage

For cloud deployments, models can be stored in:

1. **Cloud Storage (Recommended):**
   ```yaml
   deployment:
     services:
       storage:
         type: "s3"  # or "r2", "gcs"
         bucket: "fm-llm-models"
   ```

2. **Persistent Volumes:**
   - Configured automatically in Kubernetes deployments
   - Manual setup required for other providers

### SSL/TLS Configuration

For production deployments:

1. **Using Cloudflare (Recommended):**
   - Add your domain to Cloudflare
   - Enable SSL/TLS in Cloudflare dashboard
   - Update `INFERENCE_API_URL` to use HTTPS

2. **Using Let's Encrypt:**
   ```bash
   # Add to docker-compose.yml
   certbot:
     image: certbot/certbot
     volumes:
       - ./certbot/conf:/etc/letsencrypt
   ```

## Monitoring

### Prometheus + Grafana

Enable monitoring in docker-compose:

```bash
docker-compose --profile monitoring up -d
```

Access dashboards:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

### Health Checks

All services expose health endpoints:

```bash
# Web interface
curl http://localhost:5000/api/stats

# Inference API
curl http://localhost:8000/health
```

## Cost Optimization

### Estimated Weekly Costs

| Provider | GPU Type | Cost/Hour | Weekly (50 gen/day) |
|----------|----------|-----------|---------------------|
| RunPod   | RTX 3090 | $0.44     | ~$5.16             |
| Modal    | A10G     | $1.10     | ~$12.88            |
| Vast.ai  | RTX 3090 | $0.20-0.40| ~$2.32-4.64        |

### Cost Reduction Strategies

1. **Use Spot/Preemptible Instances**
   ```python
   # In deployment config
   "spot_instance": True,
   "max_bid": 0.30  # Maximum hourly bid
   ```

2. **Enable Caching**
   ```yaml
   deployment:
     performance:
       enable_caching: true
       cache_ttl: 7200  # 2 hours
   ```

3. **Batch Processing**
   ```python
   # Process multiple requests together
   POST /generate_batch
   {
     "requests": [
       {"system_description": "..."},
       {"system_description": "..."}
     ]
   }
   ```

4. **Auto-scaling**
   - Modal: Automatic (scales to zero)
   - Kubernetes: Configure HPA
   - Others: Use external autoscalers

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use smaller model (7B instead of 14B)
   - Enable gradient checkpointing

2. **Connection to Inference API Failed**
   - Check `INFERENCE_API_URL` is correct
   - Verify firewall/security group rules
   - Test with curl: `curl -X GET $INFERENCE_API_URL/health`

3. **Model Loading Errors**
   - Ensure models are downloaded/cached
   - Check storage permissions
   - Verify CUDA compatibility

4. **Slow Inference**
   - Enable model caching
   - Use faster GPU (A100 > V100 > RTX 3090)
   - Reduce context size (lower `rag_k`)

### Debug Commands

```bash
# Check GPU availability
docker exec fm-llm-inference nvidia-smi

# View container logs
docker logs fm-llm-inference -f

# Test inference directly
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "system_description": "Test system",
    "model_config": "base",
    "rag_k": 0
  }'
```

### Support

For deployment issues:
1. Check logs in `logs/` directory
2. Review [GitHub Issues](https://github.com/your-repo/issues)
3. Contact support with deployment logs

## Production Checklist

Before going to production:

- [ ] Set secure `SECRET_KEY`
- [ ] Enable HTTPS/SSL
- [ ] Configure proper database (PostgreSQL)
- [ ] Set up monitoring and alerts
- [ ] Configure backups
- [ ] Test disaster recovery
- [ ] Document API endpoints
- [ ] Set up CI/CD pipeline
- [ ] Configure rate limiting
- [ ] Enable access logging 