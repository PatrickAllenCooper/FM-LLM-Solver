# Efficient Cloud Deployment Process

This document outlines the streamlined deployment process for FM-LLM Solver to Google Cloud Platform, optimized for rapid iteration and reliable deployments.

## Quick Deployment Workflow

### Prerequisites
- `gcloud` CLI authenticated and configured
- Project ID: `fmgen-net-production`
- Region: `us-central1`

### 1. Local Development & Testing
```bash
# Test builds locally before deployment
cd /Users/patrickcooper/code/FM-LLM-Solver
npm run build  # Tests both frontend and backend builds

# Fix any TypeScript/build errors
# Commit changes with descriptive messages
git add .
git commit -m "🔧 Feature description"
git push
```

### 2. Automated Build & Deploy (2-command deployment)
```bash
# Build and push both services using Cloud Build
gcloud builds submit . --config=gcp-deploy/cloudbuild-backend.yaml --substitutions=_PROJECT_ID=fmgen-net-production
gcloud builds submit . --config=gcp-deploy/cloudbuild-frontend.yaml --substitutions=_PROJECT_ID=fmgen-net-production

# Deploy services with configuration substitution
sed "s/PROJECT_ID/fmgen-net-production/g; s/REGION/us-central1/g" gcp-deploy/cloud-run/fmgen-api.yaml | gcloud run services replace - --region=us-central1
sed "s/PROJECT_ID/fmgen-net-production/g" gcp-deploy/cloud-run/fmgen-ui.yaml | gcloud run services replace - --region=us-central1
```

### 3. Verification
```bash
# Verify deployment success
gcloud run services list --region=us-central1
curl https://fmgen-api-610214208348.us-central1.run.app/health
curl -I https://fmgen.net
```

## Key Efficiency Improvements

### Automated Build Pipeline
- **Cloud Build Integration**: Automatically builds Docker images from source
- **Multi-stage Builds**: Optimized for production with security hardening
- **Parallel Processing**: Frontend and backend build simultaneously
- **Cache Optimization**: Layer caching reduces build times to ~1-2 minutes

### Configuration Management
- **Template Substitution**: Single YAML files with PROJECT_ID/REGION variables
- **Environment Separation**: Production configs isolated from development
- **Secret Integration**: Automatic secret mounting from Google Secret Manager
- **Zero-downtime Deployment**: Rolling updates with health checks

### Reliability Features
- **Startup Probes**: Enhanced Cloud Run startup configuration prevents timeout failures
- **Health Monitoring**: Automatic rollback on deployment failures
- **Build Verification**: TypeScript compilation and tests run before deployment
- **Artifact Tracking**: Container images tagged with Git commit SHAs

## Typical Deployment Timeline
- **Local testing**: 30 seconds (npm run build)
- **Cloud Build**: 60-90 seconds per service
- **Service deployment**: 30-60 seconds per service
- **Total time**: ~4-6 minutes from code to live deployment

## Emergency Rollback
```bash
# Quick rollback to previous revision
gcloud run services update-traffic fmgen-api --to-revisions=PREVIOUS_REVISION=100 --region=us-central1
gcloud run services update-traffic fmgen-ui --to-revisions=PREVIOUS_REVISION=100 --region=us-central1
```

This process enables rapid iteration while maintaining production reliability and complete audit trails.
