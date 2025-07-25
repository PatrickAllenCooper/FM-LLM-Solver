# Deployment Structure

This directory contains all deployment-related files organized by environment and technology.

## Directory Structure

```
deployment/
├── environments/
│   ├── local/          # Local development configurations
│   ├── staging/        # Staging/testing deployments
│   └── production/     # Production deployments
├── docker/             # Docker configurations
│   ├── Dockerfile*     # Various Docker images
│   ├── docker-compose* # Compose configurations
│   └── docker-entrypoint.sh
├── kubernetes/         # Kubernetes manifests
│   └── gke-ingress.yaml
├── cloudbuild*.yaml    # Google Cloud Build configs
└── deploy*.sh          # Deployment scripts
```

## Quick Start

### Local Development
```bash
cd deployment/docker
docker-compose -f docker-compose.simple.yml up
```

### Production Deployment
```bash
cd deployment/environments/production
# Review and customize deployment files
kubectl apply -f complete-production-deployment.yaml
```

### Staging Environment
```bash
cd deployment/environments/staging
# Use staging configurations for testing
```

## Environment Guide

- **Local**: Use for development with minimal resources
- **Staging**: Test production-like deployments before going live  
- **Production**: Full production configurations with monitoring

## Docker Images

- `Dockerfile` - Main unified image
- `Dockerfile.web` - Web interface only
- `Dockerfile.inference` - ML/inference only
- `Dockerfile.production` - Production optimized

Choose the appropriate Dockerfile based on your deployment needs. 