#!/bin/bash
# FM-LLM Solver Deployment Script
# Simple wrapper for deployment operations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.10+."
        exit 1
    fi
    
    # Check if .env exists
    if [ ! -f ".env" ]; then
        print_warn ".env file not found. Creating from template..."
        cp config/env.example .env
        print_info "Please edit .env file with your configuration before deploying."
        exit 0
    fi
    
    print_info "All prerequisites met!"
}

# Function to show usage
show_usage() {
    echo "FM-LLM Solver Deployment Script"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  local       Deploy locally using Docker Compose"
    echo "  hybrid      Deploy web locally, inference on cloud"
    echo "  cloud       Deploy everything to cloud"
    echo "  test        Run deployment tests"
    echo "  stop        Stop all services"
    echo "  logs        Show logs"
    echo "  clean       Clean up containers and volumes"
    echo ""
    echo "Cloud Provider Options (for cloud/hybrid):"
    echo "  --provider [runpod|modal|vastai|gcp]"
    echo ""
    echo "Examples:"
    echo "  $0 local                    # Deploy everything locally"
    echo "  $0 hybrid --provider modal  # Web local, inference on Modal"
    echo "  $0 cloud --provider runpod  # Everything on RunPod"
    echo "  $0 test                     # Run deployment tests"
}

# Main script logic
case "$1" in
    local)
        check_prerequisites
        print_info "Starting local deployment..."
        
        # Set deployment mode
        export DEPLOYMENT_MODE=local
        
        # Build and start services
        print_info "Building Docker images..."
        docker compose build
        
        print_info "Starting services..."
        docker compose up -d
        
        print_info "Local deployment complete!"
        print_info "Web interface: http://localhost:5000"
        print_info "Inference API: http://localhost:8000"
        ;;
        
    hybrid)
        check_prerequisites
        PROVIDER=${3:-modal}
        
        print_info "Starting hybrid deployment..."
        print_info "Provider: $PROVIDER"
        
        # Set deployment mode
        export DEPLOYMENT_MODE=hybrid
        
        # Deploy inference to cloud
        print_info "Deploying inference API to $PROVIDER..."
        python deployment/deploy.py $PROVIDER
        
        # Start web interface locally
        print_info "Starting web interface locally..."
        docker compose up -d web
        
        print_info "Hybrid deployment complete!"
        print_info "Web interface: http://localhost:5000"
        print_info "Check $PROVIDER dashboard for inference API endpoint"
        ;;
        
    cloud)
        check_prerequisites
        PROVIDER=${3:-runpod}
        
        print_info "Starting cloud deployment..."
        print_info "Provider: $PROVIDER"
        
        # Set deployment mode
        export DEPLOYMENT_MODE=cloud
        
        # Deploy everything to cloud
        python deployment/deploy.py $PROVIDER
        
        print_info "Cloud deployment initiated!"
        print_info "Check $PROVIDER dashboard for service endpoints"
        ;;
        
    test)
        print_info "Running deployment tests..."
        python deployment/test_deployment.py
        ;;
        
    stop)
        print_info "Stopping all services..."
        docker compose down
        print_info "Services stopped."
        ;;
        
    logs)
        SERVICE=${2:-}
        if [ -z "$SERVICE" ]; then
            docker compose logs -f
        else
            docker compose logs -f $SERVICE
        fi
        ;;
        
    clean)
        print_warn "This will remove all containers, volumes, and images."
        read -p "Are you sure? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Cleaning up..."
            docker compose down -v --rmi all
            print_info "Cleanup complete."
        fi
        ;;
        
    *)
        show_usage
        exit 1
        ;;
esac 