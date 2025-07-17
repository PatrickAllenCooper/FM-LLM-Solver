#!/bin/bash
set -e

# Cloud Build Deployment for FM-LLM Solver
# Builds and deploys images using Google Cloud Build

echo "☁️ Deploying FM-LLM Solver via Google Cloud Build..."

# Configuration
PROJECT_ID=${PROJECT_ID:-"fmgen-net-production"}
REGION=${REGION:-"us-central1"}
ZONE=${ZONE:-"us-central1-b"}

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Set project
gcloud config set project $PROJECT_ID

echo ""
echo "🏗️ Building images with Cloud Build..."

# Submit build job for web interface
print_info "Building web interface..."
gcloud builds submit . \
    --config=cloudbuild-web.yaml \
    --substitutions=_LOCATION=$REGION,_REPOSITORY=fm-llm-repo

# Submit build job for inference API  
print_info "Building inference API..."
gcloud builds submit . \
    --config=cloudbuild-inference.yaml \
    --substitutions=_LOCATION=$REGION,_REPOSITORY=fm-llm-repo

print_status "Images built and pushed via Cloud Build"

echo ""
echo "🚀 Deploying to Kubernetes..."

# Get cluster credentials
gcloud container clusters get-credentials fm-llm-cluster --zone=$ZONE

# Set image names
export WEB_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/fm-llm-repo/fm-llm-web:latest"
export INFERENCE_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/fm-llm-repo/fm-llm-inference:latest"

# Apply Kubernetes configuration
envsubst < deployment/kubernetes/fm-llm-full-stack.yaml | kubectl apply -f -

print_status "Application deployed to Kubernetes"

echo ""
echo "🎯 Deployment Complete!"
echo "========================="
echo "Web Interface: https://fmgen.net"
echo "Monitor: kubectl get pods -n fm-llm-prod -w"
echo ""
echo "🎮 GPU nodes will auto-scale when inference API receives requests"
echo "💰 Cost: ~$0.35/hour only when GPU nodes are active"

print_status "FM-LLM Solver with GPU inference is ready!" 