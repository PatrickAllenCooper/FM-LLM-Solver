#!/bin/bash
# Streamlined FM-LLM Solver Deployment
# One-command setup for real LLM GPU testing

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() { echo -e "${GREEN}✅ $1${NC}"; }
print_warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

echo "🚀 FM-LLM Solver - Streamlined Deployment"
echo "=========================================="

# Check prerequisites
print_info "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    print_error "Docker not found. Install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose not found. Install Docker Compose first."
    exit 1
fi

# Check for NVIDIA Docker
if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    print_warn "NVIDIA Docker not working. GPU acceleration may not be available."
    print_warn "Install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

print_info "Prerequisites OK"

# Create config if it doesn't exist
if [ ! -f "config.yaml" ]; then
    print_info "Creating default config.yaml..."
    cat > config.yaml << 'EOF'
model:
  provider: qwen
  name: Qwen/Qwen2.5-7B-Instruct
  device: cuda
  quantization: 4bit
  temperature: 0.1
  max_tokens: 512

verification:
  numerical_samples: 200
  tolerance: 0.1

logging:
  level: INFO
  file: logs/fm_llm_solver.log
EOF
    print_info "Created config.yaml with default settings"
fi

# Build and deploy
print_info "Building Docker image..."
docker-compose -f docker-compose.simple.yml build

print_info "Starting FM-LLM Solver..."
docker-compose -f docker-compose.simple.yml up -d

# Wait for startup
print_info "Waiting for service to start..."
sleep 10

# Check if it's running
if docker-compose -f docker-compose.simple.yml ps | grep -q "Up"; then
    print_info "Deployment successful!"
    echo ""
    echo "🎉 FM-LLM Solver is running!"
    echo "   🌐 Web Interface: http://localhost:5000"
    echo ""
    echo "📋 Quick Commands:"
    echo "   • Test GPU: docker exec fm-llm-solver python3 quick_gpu_test.py"
    echo "   • View logs: docker-compose -f docker-compose.simple.yml logs -f"
    echo "   • Stop: docker-compose -f docker-compose.simple.yml down"
    echo ""
else
    print_error "Deployment failed. Check logs:"
    docker-compose -f docker-compose.simple.yml logs
    exit 1
fi 