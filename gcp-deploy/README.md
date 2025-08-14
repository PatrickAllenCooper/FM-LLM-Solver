# FM-LLM Solver GCP Deployment

This directory contains the configuration and scripts needed to deploy the FM-LLM Solver to Google Cloud Platform, specifically to the **fmgen.net** domain using Cloud Run services.

## Architecture Overview

- **Frontend (fmgen-ui)**: React SPA served via Nginx on Cloud Run → `fmgen.net`
- **Backend (fmgen-api)**: Node.js API on Cloud Run → `api.fmgen.net`
- **Database**: Cloud SQL PostgreSQL
- **Cache**: Cloud Memory Store (Redis)
- **Secrets**: Google Secret Manager
- **SSL**: Automatically managed certificates

## Prerequisites

1. **Google Cloud SDK** installed and authenticated
   ```bash
   # Install gcloud CLI
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   
   # Authenticate
   gcloud auth login
   gcloud auth application-default login
   ```

2. **Domain ownership** of `fmgen.net`

3. **Anthropic API Key** for LLM integration

## Quick Deployment

### 1. Deploy Infrastructure and Services

```bash
# Make scripts executable
chmod +x gcp-deploy/scripts/*.sh

# Deploy everything (replace with your GCP project ID)
./gcp-deploy/scripts/deploy.sh YOUR_PROJECT_ID us-central1
```

This script will:
- Enable required GCP APIs
- Create Cloud SQL PostgreSQL instance
- Create Redis instance  
- Set up secrets management
- Build and deploy Docker images
- Deploy Cloud Run services

### 2. Add Anthropic API Key

```bash
# Add your Anthropic API key to secrets
echo 'YOUR_ANTHROPIC_API_KEY' | gcloud secrets versions add anthropic-api-key --data-file=-
```

### 3. Configure Custom Domain

```bash
# Set up fmgen.net domain mapping
./gcp-deploy/scripts/setup-domain.sh YOUR_PROJECT_ID us-central1
```

Follow the DNS configuration instructions provided by the script.

## Manual Step-by-Step Deployment

### 1. Set Up GCP Project

```bash
# Set project
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Enable APIs
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    sql-component.googleapis.com \
    sqladmin.googleapis.com \
    secretmanager.googleapis.com \
    domains.googleapis.com
```

### 2. Create Database

```bash
# Create Cloud SQL instance
gcloud sql instances create fmgen-postgres \
    --database-version=POSTGRES_14 \
    --tier=db-f1-micro \
    --region=us-central1

# Create database
gcloud sql databases create fm_llm_solver --instance=fmgen-postgres

# Create user
gcloud sql users create fmgen-user \
    --instance=fmgen-postgres \
    --password="SECURE_PASSWORD"
```

### 3. Set Up Secrets

```bash
# Create secrets
echo "SECURE_PASSWORD" | gcloud secrets create db-password --data-file=-
echo "fmgen-user" | gcloud secrets create db-user --data-file=-
echo "$(openssl rand -base64 64)" | gcloud secrets create jwt-secret --data-file=-
echo "YOUR_ANTHROPIC_API_KEY" | gcloud secrets create anthropic-api-key --data-file=-
```

### 4. Build and Deploy

```bash
# Build backend
gcloud builds submit . --config=gcp-deploy/cloudbuild-backend.yaml

# Build frontend  
gcloud builds submit . --config=gcp-deploy/cloudbuild-frontend.yaml

# Deploy services
gcloud run services replace gcp-deploy/cloud-run/fmgen-api.yaml --region=us-central1
gcloud run services replace gcp-deploy/cloud-run/fmgen-ui.yaml --region=us-central1

# Allow public access
gcloud run services add-iam-policy-binding fmgen-api \
    --region=us-central1 \
    --member="allUsers" \
    --role="roles/run.invoker"
    
gcloud run services add-iam-policy-binding fmgen-ui \
    --region=us-central1 \
    --member="allUsers" \
    --role="roles/run.invoker"
```

## File Structure

```
gcp-deploy/
├── README.md                    # This file
├── cloud-run/
│   ├── fmgen-api.yaml          # Backend service config
│   └── fmgen-ui.yaml           # Frontend service config
├── scripts/
│   ├── deploy.sh               # Main deployment script
│   ├── setup-domain.sh         # Domain configuration
│   └── health-check.sh         # Health check script
├── Dockerfile.backend.prod     # Production backend Dockerfile
├── Dockerfile.frontend.prod    # Production frontend Dockerfile
├── nginx.conf                  # Nginx configuration for frontend
├── cloudbuild-backend.yaml     # Cloud Build config for backend
├── cloudbuild-frontend.yaml    # Cloud Build config for frontend
└── frontend.env.production     # Production environment variables
```

## Environment Variables

### Backend (fmgen-api)
- `NODE_ENV=production`
- `PORT=3000`
- `DB_HOST=/cloudsql/PROJECT_ID:REGION:fmgen-postgres`
- `DB_NAME=fm_llm_solver`
- `DB_USER` (from secret)
- `DB_PASSWORD` (from secret)
- `JWT_SECRET` (from secret)
- `ANTHROPIC_API_KEY` (from secret)
- `CORS_ORIGIN=https://fmgen.net,https://www.fmgen.net`

### Frontend (fmgen-ui)
- `VITE_API_URL=https://api.fmgen.net`

## Monitoring and Logs

```bash
# View service logs
gcloud logs read "resource.type=cloud_run_revision AND resource.labels.service_name=fmgen-api" --limit=50

gcloud logs read "resource.type=cloud_run_revision AND resource.labels.service_name=fmgen-ui" --limit=50

# Check service status
gcloud run services describe fmgen-api --region=us-central1
gcloud run services describe fmgen-ui --region=us-central1

# Monitor domain mapping
gcloud run domain-mappings describe --domain=fmgen.net --region=us-central1
```

## Scaling and Performance

### Backend Configuration
- **CPU**: 2 vCPU
- **Memory**: 2GB
- **Concurrency**: 80 requests per instance
- **Min instances**: 1 (always warm)
- **Max instances**: 10

### Frontend Configuration  
- **CPU**: 1 vCPU
- **Memory**: 1GB
- **Concurrency**: 100 requests per instance
- **Min instances**: 1
- **Max instances**: 10

## Security Features

- **HTTPS only** with automatic SSL certificates
- **Security headers** configured in Nginx
- **Secrets management** via Google Secret Manager
- **Private database** access via Cloud SQL Proxy
- **CORS** configured for fmgen.net domains only
- **Non-root containers** for security

## Troubleshooting

### Common Issues

1. **Build failures**: Check Cloud Build logs
   ```bash
   gcloud builds list
   gcloud builds log BUILD_ID
   ```

2. **Service deployment issues**: Check service events
   ```bash
   gcloud run services describe SERVICE_NAME --region=us-central1
   ```

3. **Database connection issues**: Verify secrets and Cloud SQL setup
   ```bash
   gcloud sql instances describe fmgen-postgres
   gcloud secrets versions list SECRET_NAME
   ```

4. **Domain mapping issues**: Verify domain ownership and DNS
   ```bash
   gcloud domains list-user-verified-domains
   gcloud run domain-mappings describe --domain=fmgen.net --region=us-central1
   ```

### Debug Commands

```bash
# Test backend health
curl https://api.fmgen.net/health

# Test frontend
curl https://fmgen.net/health

# Check service URLs
gcloud run services list --region=us-central1

# View recent deployments
gcloud run revisions list --service=fmgen-api --region=us-central1
```

## Cost Optimization

- Uses **Cloud Run** (pay-per-request) instead of always-on instances
- **f1-micro** database tier for cost efficiency
- **Automatic scaling** to zero when not in use
- **Efficient Docker images** with multi-stage builds

## Support

For deployment issues:
1. Check the troubleshooting section above
2. Review Cloud Console logs
3. Verify all prerequisites are met
4. Ensure domain DNS is properly configured
