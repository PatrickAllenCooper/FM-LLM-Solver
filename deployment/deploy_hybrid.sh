#!/bin/bash

# FM-LLM-Solver Hybrid Deployment Script
# Deploys inference to Modal and web interface locally

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if .env file exists and has required variables
check_env() {
    if [[ ! -f ".env" ]]; then
        error ".env file not found!"
        exit 1
    fi
    
    source .env
    
    if [[ -z "$SECRET_KEY" ]]; then
        error "SECRET_KEY not set in .env file"
        exit 1
    fi
    
    log "✅ Environment configuration validated"
}

# Check if Modal is authenticated
check_modal_auth() {
    if ! modal token current >/dev/null 2>&1; then
        warn "Modal authentication required"
        echo "Please run: modal token new"
        echo "Then re-run this script"
        exit 1
    fi
    
    log "✅ Modal authentication verified"
}

# Deploy inference to Modal
deploy_inference() {
    log "🚀 Deploying inference service to Modal..."
    
    if [[ ! -f "modal_inference_app.py" ]]; then
        error "modal_inference_app.py not found!"
        exit 1
    fi
    
    # Deploy to Modal
    modal deploy modal_inference_app.py
    
    # Get the deployed URL
    log "Getting Modal deployment URL..."
    MODAL_URL=$(modal app list | grep "fm-llm-solver-inference" | awk '{print $3}' | head -1)
    
    if [[ -z "$MODAL_URL" ]]; then
        warn "Could not automatically detect Modal URL"
        echo "Please check your Modal dashboard and update INFERENCE_API_URL in .env"
        echo "The URL should look like: https://your-app--web-generate.modal.run"
    else
        log "✅ Modal deployment complete: $MODAL_URL"
        
        # Update .env with inference URL
        if grep -q "INFERENCE_API_URL=" .env; then
            sed -i "s|INFERENCE_API_URL=.*|INFERENCE_API_URL=${MODAL_URL}/generate|" .env
        else
            echo "INFERENCE_API_URL=${MODAL_URL}/generate" >> .env
        fi
        log "✅ Updated INFERENCE_API_URL in .env"
    fi
}

# Build and start local services
start_local_services() {
    log "🏗️ Building Docker images..."
    docker-compose -f docker-compose.hybrid.yml build --no-cache
    
    log "🚀 Starting local services (Redis + Web Interface)..."
    docker-compose -f docker-compose.hybrid.yml up -d
    
    # Wait for services to be ready
    log "⏳ Waiting for services to start..."
    sleep 30
    
    # Check health
    if curl -f http://localhost:5000/health >/dev/null 2>&1; then
        log "✅ Web interface is healthy"
    else
        warn "Web interface health check failed - checking logs..."
        docker-compose -f docker-compose.hybrid.yml logs web
    fi
}

# Test the deployment
test_deployment() {
    log "🧪 Testing hybrid deployment..."
    
    # Test web interface
    if curl -f http://localhost:5000/ >/dev/null 2>&1; then
        log "✅ Web interface accessible at http://localhost:5000"
    else
        error "❌ Web interface not accessible"
        return 1
    fi
    
    # Test inference API if URL is set
    source .env
    if [[ -n "$INFERENCE_API_URL" ]]; then
        if curl -f "${INFERENCE_API_URL%/generate}/health" >/dev/null 2>&1; then
            log "✅ Inference API is healthy"
        else
            warn "⚠️ Inference API health check failed"
        fi
    fi
}

# Show deployment summary
show_summary() {
    echo ""
    echo "🎉 Hybrid deployment complete!"
    echo ""
    echo "📊 Services:"
    echo "   🌐 Web Interface: http://localhost:5000"
    echo "   🗄️ Redis Cache: localhost:6379"
    echo "   🚀 Inference API: Modal (serverless)"
    echo ""
    echo "💰 Cost estimate: ~$3-5/week (based on usage)"
    echo ""
    echo "🔧 Useful commands:"
    echo "   View logs: docker-compose -f docker-compose.hybrid.yml logs -f"
    echo "   Stop services: docker-compose -f docker-compose.hybrid.yml down"
    echo "   Restart: ./deploy_hybrid.sh"
    echo ""
    echo "🔗 Test your deployment:"
    echo "   curl http://localhost:5000/health"
    echo ""
}

# Main deployment process
main() {
    log "🚀 Starting FM-LLM-Solver Hybrid Deployment"
    echo ""
    
    # Pre-flight checks
    check_env
    check_modal_auth
    
    # Deploy inference to Modal
    deploy_inference
    
    # Start local services
    start_local_services
    
    # Test deployment
    test_deployment
    
    # Show summary
    show_summary
}

# Handle errors
trap 'error "Deployment failed at line $LINENO"' ERR

# Run main function
main "$@" 