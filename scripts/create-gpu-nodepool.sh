#!/bin/bash
set -e

# GPU Node Pool Creation for FM-LLM Solver
# Adds GPU nodes to existing cluster for LLM inference

echo "🎮 Creating GPU Node Pool for LLM Inference..."

# Configuration
PROJECT_ID=${PROJECT_ID:-"fmgen-net-production"}
REGION=${REGION:-"us-central1"}
CLUSTER_NAME=${CLUSTER_NAME:-"fm-llm-cluster"}
GPU_POOL_NAME="gpu-inference-pool"

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

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    print_error "Please authenticate with Google Cloud: gcloud auth login"
fi

# Set project
gcloud config set project $PROJECT_ID
print_info "Using project: $PROJECT_ID"

# Check if cluster exists
if ! gcloud container clusters describe $CLUSTER_NAME --region=$REGION >/dev/null 2>&1; then
    print_error "Cluster $CLUSTER_NAME not found in region $REGION"
fi

print_info "Adding GPU node pool to cluster: $CLUSTER_NAME"

# Check if GPU pool already exists
if gcloud container node-pools describe $GPU_POOL_NAME --cluster=$CLUSTER_NAME --region=$REGION >/dev/null 2>&1; then
    print_warning "GPU node pool already exists: $GPU_POOL_NAME"
    echo "To recreate, delete it first:"
    echo "gcloud container node-pools delete $GPU_POOL_NAME --cluster=$CLUSTER_NAME --region=$REGION"
    exit 0
fi

echo ""
echo "🎮 Creating GPU Node Pool Configuration:"
echo "Name: $GPU_POOL_NAME"
echo "Machine Type: n1-standard-4"
echo "GPU Type: nvidia-tesla-t4"
echo "GPU Count: 1 per node"
echo "Disk: 100GB SSD"
echo "Nodes: 0-2 (auto-scaling)"
echo ""

# Create GPU node pool
gcloud container node-pools create $GPU_POOL_NAME \
    --cluster=$CLUSTER_NAME \
    --region=$REGION \
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

echo ""
echo "📋 GPU Node Pool Summary:"
echo "========================"
echo "Pool Name: $GPU_POOL_NAME"
echo "Cluster: $CLUSTER_NAME"
echo "Machine Type: n1-standard-4 (4 vCPU, 15GB RAM)"
echo "GPU: NVIDIA Tesla T4 (16GB VRAM)"
echo "Scaling: 0-2 nodes (preemptible for cost savings)"
echo "Cost: ~$0.35/hour per node when active"

echo ""
echo "🎯 GPU Node Pool is ready for LLM inference workloads!"
echo ""
echo "Next steps:"
echo "1. Deploy inference API with GPU requirements"
echo "2. Pods will automatically scale up GPU nodes when needed"
echo "3. Nodes will scale down to 0 when not in use (cost optimization)"

print_status "GPU infrastructure setup complete!" 