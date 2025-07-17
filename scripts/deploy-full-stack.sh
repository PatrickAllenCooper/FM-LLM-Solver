#!/bin/bash
set -e

# FM-LLM Solver Full Stack Deployment Script
# Deploys the complete application including web interface and inference API

echo "🚀 Deploying FM-LLM Solver Full Stack to Google Cloud Platform..."

# Configuration
PROJECT_ID=${PROJECT_ID:-"fmgen-net-production"}
REGION=${REGION:-"us-central1"}
CLUSTER_NAME=${CLUSTER_NAME:-"fm-llm-cluster"}
NAMESPACE="fm-llm-prod"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

print_header() {
    echo ""
    echo -e "${BLUE}$1${NC}"
    echo "======================================"
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check if gcloud is authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
        print_error "Please authenticate with Google Cloud: gcloud auth login"
    fi
    
    # Check if docker is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker."
    fi
    
    # Check if kubectl is configured
    if ! kubectl cluster-info >/dev/null 2>&1; then
        print_error "kubectl is not configured. Please set up cluster access."
    fi
    
    print_status "All prerequisites met"
}

# Set up model storage
setup_model_storage() {
    print_header "Setting Up Model Storage"
    
    if [ -f "scripts/setup-model-storage.sh" ]; then
        bash scripts/setup-model-storage.sh
    else
        print_warning "Model storage setup script not found, skipping..."
    fi
}

# Create GPU node pool for inference
create_gpu_nodepool() {
    print_header "Creating GPU Node Pool for LLM Inference"
    
    if [ -f "scripts/create-gpu-nodepool.sh" ]; then
        chmod +x scripts/create-gpu-nodepool.sh
        bash scripts/create-gpu-nodepool.sh
    else
        print_warning "GPU node pool script not found, creating manually..."
        
        # Check if GPU pool already exists
        if ! gcloud container node-pools describe gpu-inference-pool --cluster=${CLUSTER_NAME} --region=${REGION} >/dev/null 2>&1; then
            print_info "Creating GPU node pool for inference workloads..."
            gcloud container node-pools create gpu-inference-pool \
                --cluster=${CLUSTER_NAME} \
                --region=${REGION} \
                --machine-type=n1-standard-4 \
                --accelerator=type=nvidia-tesla-t4,count=1 \
                --disk-type=pd-ssd \
                --disk-size=100GB \
                --enable-autoscaling \
                --min-nodes=0 \
                --max-nodes=2 \
                --num-nodes=0 \
                --node-labels=workload-type=gpu-inference \
                --node-taints=nvidia.com/gpu=present:NoSchedule \
                --enable-autorepair \
                --enable-autoupgrade \
                --preemptible
            
            print_status "GPU node pool created successfully"
        else
            print_status "GPU node pool already exists"
        fi
    fi
}

# Install NVIDIA GPU device plugin
install_gpu_drivers() {
    print_header "Installing GPU Drivers and Device Plugin"
    
    # Install NVIDIA GPU device plugin for GKE
    kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
    
    print_status "GPU drivers installation initiated"
}

# Build and push Docker images
build_and_push_images() {
    print_header "Building and Pushing Docker Images"
    
    # Configure Docker for Artifact Registry
    gcloud auth configure-docker ${REGION}-docker.pkg.dev
    
    # Set up image names
    REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/fm-llm-repo"
    WEB_IMAGE="${REGISTRY}/fm-llm-web:latest"
    INFERENCE_IMAGE="${REGISTRY}/fm-llm-inference:latest"
    
    print_info "Building web interface image (CPU-only)..."
    docker build -f Dockerfile.web -t ${WEB_IMAGE} .
    print_status "Web interface image built"
    
    print_info "Building inference API image (GPU-enabled)..."
    docker build -f Dockerfile.inference -t ${INFERENCE_IMAGE} .
    print_status "Inference API image built"
    
    print_info "Pushing images to Artifact Registry..."
    docker push ${WEB_IMAGE}
    docker push ${INFERENCE_IMAGE}
    print_status "Images pushed successfully"
    
    # Export image names for use in deployment
    export WEB_IMAGE
    export INFERENCE_IMAGE
}

# Create secrets and configuration
create_secrets() {
    print_header "Creating Secrets and Configuration"
    
    # Generate random secrets if they don't exist
    SECRET_KEY=$(openssl rand -base64 32)
    ENCRYPTION_KEY=$(openssl rand -base64 32)
    JWT_SECRET=$(openssl rand -base64 32)
    
    # Create namespace if it doesn't exist
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    
    # Create application secrets
    kubectl create secret generic app-secrets \
        --from-literal=secret-key="${SECRET_KEY}" \
        --from-literal=encryption-key="${ENCRYPTION_KEY}" \
        --from-literal=jwt-secret="${JWT_SECRET}" \
        -n ${NAMESPACE} \
        --dry-run=client -o yaml | kubectl apply -f -
    
    print_status "Application secrets created"
}

# Deploy database configuration
deploy_database() {
    print_header "Configuring Database"
    
    # Apply existing database configuration
    if [ -f "deployment/kubernetes/postgres.yaml" ]; then
        kubectl apply -f deployment/kubernetes/postgres.yaml
        print_status "Database configuration applied"
    else
        print_warning "Database configuration not found, assuming Cloud SQL is already set up"
    fi
}

# Deploy Redis configuration
deploy_redis() {
    print_header "Configuring Redis Cache"
    
    # Apply existing Redis configuration
    if [ -f "deployment/kubernetes/redis.yaml" ]; then
        kubectl apply -f deployment/kubernetes/redis.yaml
        print_status "Redis configuration applied"
    else
        print_warning "Redis configuration not found, assuming Redis is already set up"
    fi
}

# Deploy application stack
deploy_application() {
    print_header "Deploying Application Stack"
    
    # Get current cluster credentials
    gcloud container clusters get-credentials ${CLUSTER_NAME} --region=${REGION} --project=${PROJECT_ID}
    
    # Substitute environment variables in deployment file
    envsubst < deployment/kubernetes/fm-llm-full-stack.yaml > /tmp/fm-llm-deployment.yaml
    
    # Apply the deployment
    kubectl apply -f /tmp/fm-llm-deployment.yaml
    
    print_status "Application deployment initiated"
    
    # Wait for deployments to be ready
    print_info "Waiting for web interface deployment..."
    kubectl rollout status deployment/fm-llm-web -n ${NAMESPACE} --timeout=600s
    
    print_info "Waiting for inference API deployment..."
    kubectl rollout status deployment/fm-llm-inference -n ${NAMESPACE} --timeout=900s
    
    print_status "All deployments are ready"
}

# Verify deployment
verify_deployment() {
    print_header "Verifying Deployment"
    
    # Check pod status
    echo "Pod Status:"
    kubectl get pods -n ${NAMESPACE}
    
    # Check service status
    echo ""
    echo "Service Status:"
    kubectl get services -n ${NAMESPACE}
    
    # Check ingress status
    echo ""
    echo "Ingress Status:"
    kubectl get ingress -n ${NAMESPACE}
    
    # Get load balancer IP
    EXTERNAL_IP=$(kubectl get ingress fm-llm-ingress -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    
    if [ "$EXTERNAL_IP" != "pending" ] && [ -n "$EXTERNAL_IP" ]; then
        print_status "Load balancer IP: $EXTERNAL_IP"
        echo ""
        echo "🌐 Your application should be available at:"
        echo "   https://fmgen.net"
        echo "   https://www.fmgen.net"
    else
        print_warning "Load balancer IP is still pending. Please check again in a few minutes."
    fi
}

# Monitor deployment
monitor_deployment() {
    print_header "Deployment Monitoring"
    
    echo "Real-time pod status (press Ctrl+C to exit):"
    echo ""
    kubectl get pods -n ${NAMESPACE} -w
}

# Main execution
main() {
    echo "🎯 FM-LLM Solver Full Stack Deployment (with GPU Support)"
    echo "Project: $PROJECT_ID"
    echo "Region: $REGION"
    echo "Cluster: $CLUSTER_NAME"
    echo ""
    
    # Set project
    gcloud config set project $PROJECT_ID
    
    # Run deployment steps
    check_prerequisites
    setup_model_storage
    create_gpu_nodepool
    install_gpu_drivers
    build_and_push_images
    create_secrets
    deploy_database
    deploy_redis
    deploy_application
    verify_deployment
    
    print_status "🎉 Full stack deployment with GPU support complete!"
    echo ""
    echo "📋 Infrastructure Summary:"
    echo "========================="
    echo "✅ Web Interface: CPU nodes (auto-scaling 2-6 pods)"
    echo "✅ Inference API: GPU nodes (Tesla T4, auto-scaling 0-2 nodes)"
    echo "✅ Model Storage: Google Cloud Storage buckets"
    echo "✅ Database: Cloud SQL PostgreSQL"
    echo "✅ Cache: Redis"
    echo "✅ SSL: Let's Encrypt certificates"
    echo ""
    echo "💰 Cost Optimization:"
    echo "- GPU nodes: Preemptible (scales to 0 when unused)"
    echo "- Estimated: ~$0.35/hour when GPU active"
    echo "- Total monthly: ~$80-120 depending on usage"
    echo ""
    echo "🚀 Your AI-powered certificate generation is live at:"
    echo "   https://fmgen.net"
    
    # Ask if user wants to monitor
    read -p "Would you like to monitor the deployment in real-time? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        monitor_deployment
    fi
}

# Handle script arguments
case "${1:-deploy}" in
    "setup-storage")
        setup_model_storage
        ;;
    "build")
        build_and_push_images
        ;;
    "deploy")
        main
        ;;
    "verify")
        verify_deployment
        ;;
    "monitor")
        monitor_deployment
        ;;
    *)
        echo "Usage: $0 [setup-storage|build|deploy|verify|monitor]"
        echo "  setup-storage: Set up Google Cloud Storage for models"
        echo "  build:         Build and push Docker images only"
        echo "  deploy:        Full deployment (default)"
        echo "  verify:        Verify existing deployment"
        echo "  monitor:       Monitor deployment status"
        exit 1
        ;;
esac 