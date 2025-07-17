#!/bin/bash
# FM-LLM Solver Production Deployment Script for US-West1
# This script handles the complete deployment to the new us-west1 cluster

set -e

echo "🚀 FM-LLM Solver Production Deployment to US-West1"
echo "=================================================="

# Configuration
PROJECT_ID="fmgen-net-production"
CLUSTER_NAME="fm-llm-cluster-west"
ZONE="us-west1-b"
NAMESPACE="fm-llm-prod"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI not found. Please install Google Cloud SDK."
        exit 1
    fi
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    
    # Check if logged into correct project
    current_project=$(gcloud config get-value project 2>/dev/null)
    if [ "$current_project" != "$PROJECT_ID" ]; then
        log_error "Not logged into correct project. Current: $current_project, Expected: $PROJECT_ID"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Wait for cluster to be ready
wait_for_cluster() {
    log_info "Waiting for cluster to be ready..."
    
    while true; do
        status=$(gcloud container clusters describe $CLUSTER_NAME --zone=$ZONE --format="value(status)" 2>/dev/null || echo "NOT_FOUND")
        echo "Cluster status: $status"
        
        if [ "$status" = "RUNNING" ]; then
            log_success "Cluster is ready!"
            break
        elif [ "$status" = "ERROR" ] || [ "$status" = "DEGRADED" ]; then
            log_error "Cluster failed to provision: $status"
            exit 1
        elif [ "$status" = "NOT_FOUND" ]; then
            log_error "Cluster not found. Please create the cluster first."
            exit 1
        fi
        
        log_info "Cluster still provisioning, waiting 30 seconds..."
        sleep 30
    done
}

# Configure kubectl context
configure_kubectl() {
    log_info "Configuring kubectl context..."
    gcloud container clusters get-credentials $CLUSTER_NAME --zone=$ZONE
    log_success "kubectl configured for $CLUSTER_NAME"
}

# Create GPU node pool
create_gpu_node_pool() {
    log_info "Creating GPU node pool..."
    
    # Check if GPU node pool already exists
    if gcloud container node-pools describe gpu-inference-pool --cluster=$CLUSTER_NAME --zone=$ZONE &>/dev/null; then
        log_warning "GPU node pool already exists, skipping creation"
        return
    fi
    
    gcloud container node-pools create gpu-inference-pool \
        --cluster=$CLUSTER_NAME \
        --zone=$ZONE \
        --machine-type=n1-standard-2 \
        --accelerator=type=nvidia-tesla-t4,count=1 \
        --num-nodes=0 \
        --enable-autoscaling \
        --min-nodes=0 \
        --max-nodes=2 \
        --preemptible \
        --node-taints=nvidia.com/gpu=present:NoSchedule \
        --node-labels=workload-type=gpu-inference \
        --disk-size=50 \
        --enable-autorepair
        
    log_success "GPU node pool created"
}

# Deploy NVIDIA GPU drivers
deploy_gpu_drivers() {
    log_info "Deploying NVIDIA GPU drivers..."
    
    kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
    
    log_info "Waiting for GPU drivers to be ready..."
    kubectl rollout status daemonset nvidia-driver-installer -n kube-system --timeout=300s
    
    log_success "GPU drivers deployed"
}

# Deploy application
deploy_application() {
    log_info "Deploying FM-LLM Solver application..."
    
    # Apply the deployment configuration
    kubectl apply -f deployment/kubernetes/fm-llm-west-production.yaml
    
    log_info "Waiting for deployments to be ready..."
    
    # Wait for web deployment
    log_info "Waiting for web deployment..."
    kubectl rollout status deployment/fm-llm-web -n $NAMESPACE --timeout=300s
    
    # Wait for inference deployment (may take longer due to model downloads)
    log_info "Waiting for inference deployment..."
    kubectl rollout status deployment/fm-llm-inference -n $NAMESPACE --timeout=600s
    
    log_success "Application deployed successfully"
}

# Check deployment status
check_deployment_status() {
    log_info "Checking deployment status..."
    
    echo ""
    echo "=== Pods Status ==="
    kubectl get pods -n $NAMESPACE -o wide
    
    echo ""
    echo "=== Services Status ==="
    kubectl get services -n $NAMESPACE
    
    echo ""
    echo "=== Ingress Status ==="
    kubectl get ingress -n $NAMESPACE
    
    echo ""
    echo "=== HPA Status ==="
    kubectl get hpa -n $NAMESPACE
}

# Update DNS (instructions)
update_dns_instructions() {
    static_ip=$(gcloud compute addresses describe fm-llm-static-ip-west --global --format="value(address)")
    
    echo ""
    echo "🌐 DNS Update Required"
    echo "====================="
    echo "Please update your DNS settings:"
    echo "  Domain: fmgen.net"
    echo "  Type: A Record"
    echo "  Value: $static_ip"
    echo ""
    echo "After DNS propagation (5-60 minutes), your application will be available at:"
    echo "  https://fmgen.net"
    echo ""
}

# Test deployment
test_deployment() {
    log_info "Testing deployment..."
    
    # Get the static IP
    static_ip=$(gcloud compute addresses describe fm-llm-static-ip-west --global --format="value(address)")
    
    # Test web interface
    log_info "Testing web interface at IP: $static_ip"
    if curl -I -H "Host: fmgen.net" "http://$static_ip/" &>/dev/null; then
        log_success "Web interface is responding"
    else
        log_warning "Web interface not responding yet (may take a few minutes for SSL certificate)"
    fi
    
    # Check SSL certificate status
    log_info "Checking SSL certificate status..."
    cert_status=$(kubectl get managedcertificate fm-llm-ssl-cert -n $NAMESPACE -o jsonpath='{.status.certificateStatus}' 2>/dev/null || echo "Not found")
    echo "SSL Certificate Status: $cert_status"
}

# Main deployment process
main() {
    echo "Starting deployment process..."
    
    check_prerequisites
    wait_for_cluster
    configure_kubectl
    create_gpu_node_pool
    deploy_gpu_drivers
    deploy_application
    check_deployment_status
    update_dns_instructions
    test_deployment
    
    echo ""
    log_success "🎉 Production deployment completed!"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Update DNS to point fmgen.net to $static_ip"
    echo "2. Wait for SSL certificate provisioning (5-10 minutes)"
    echo "3. Test the application at https://fmgen.net"
    echo "4. Monitor the deployment with: kubectl get pods -n $NAMESPACE -w"
    echo ""
    echo "🔍 Useful Commands:"
    echo "  Check pods: kubectl get pods -n $NAMESPACE"
    echo "  Check logs: kubectl logs -f deployment/fm-llm-web -n $NAMESPACE"
    echo "  Check GPU nodes: kubectl get nodes -l workload-type=gpu-inference"
    echo "  Scale inference: kubectl scale deployment fm-llm-inference --replicas=1 -n $NAMESPACE"
    echo ""
}

# Run main function
main "$@" 