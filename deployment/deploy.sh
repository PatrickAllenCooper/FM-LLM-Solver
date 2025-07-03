#!/bin/bash

# FM-LLM-Solver Deployment Script
# Usage: ./deploy.sh [environment] [options]

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
KUBE_CONFIGS_DIR="${SCRIPT_DIR}/kubernetes"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT=""
NAMESPACE="fm-llm-solver"
IMAGE_TAG="latest"
DRY_RUN=false
VERBOSE=false
SKIP_TESTS=false
WAIT_TIMEOUT=300

# Functions
log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

debug() {
    if [[ "$VERBOSE" == true ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

usage() {
    cat << EOF
FM-LLM-Solver Deployment Script

Usage: $0 [ENVIRONMENT] [OPTIONS]

ENVIRONMENTS:
    local       Deploy locally using docker-compose
    staging     Deploy to staging Kubernetes cluster
    production  Deploy to production Kubernetes cluster

OPTIONS:
    -t, --tag TAG           Docker image tag to deploy (default: latest)
    -n, --namespace NS      Kubernetes namespace (default: fm-llm-solver)
    --dry-run              Show what would be deployed without actually deploying
    --skip-tests           Skip pre-deployment tests
    --wait-timeout SEC     Wait timeout for deployment (default: 300)
    -v, --verbose          Enable verbose output
    -h, --help             Show this help message

EXAMPLES:
    $0 local                        # Deploy locally
    $0 staging -t v1.2.3            # Deploy to staging with specific tag
    $0 production --dry-run         # Show what would be deployed to production
    $0 local --skip-tests -v        # Deploy locally, skip tests, verbose output

PREREQUISITES:
    - Docker and docker-compose (for local deployment)
    - kubectl configured for your cluster (for Kubernetes deployment)
    - Appropriate secrets configured in your cluster

EOF
}

check_prerequisites() {
    debug "Checking prerequisites for $ENVIRONMENT deployment"
    
    case $ENVIRONMENT in
        local)
            if ! command -v docker >/dev/null 2>&1; then
                error "Docker is not installed or not in PATH"
                exit 1
            fi
            if ! command -v docker-compose >/dev/null 2>&1; then
                error "Docker Compose is not installed or not in PATH"
                exit 1
            fi
            ;;
        staging|production)
            if ! command -v kubectl >/dev/null 2>&1; then
                error "kubectl is not installed or not in PATH"
                exit 1
            fi
            if ! kubectl cluster-info >/dev/null 2>&1; then
                error "kubectl is not configured or cluster is not accessible"
                exit 1
            fi
            ;;
        *)
            error "Invalid environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
}

run_tests() {
    if [[ "$SKIP_TESTS" == true ]]; then
        warn "Skipping pre-deployment tests as requested"
        return 0
    fi
    
    log "Running pre-deployment tests..."
    
    cd "$PROJECT_ROOT"
    
    # Run unit tests
    debug "Running unit tests"
    if ! python -m pytest tests/unit/ -v --tb=short; then
        error "Unit tests failed"
        exit 1
    fi
    
    # Run integration tests (if available)
    if [[ -d "tests/integration" ]]; then
        debug "Running integration tests"
        if ! python -m pytest tests/integration/ -v --tb=short; then
            error "Integration tests failed"
            exit 1
        fi
    fi
    
    log "All tests passed"
}

deploy_local() {
    log "Deploying to local environment using Docker Compose"
    
    cd "$PROJECT_ROOT"
    
    # Build images
    log "Building Docker images..."
    if [[ "$DRY_RUN" == true ]]; then
        debug "DRY RUN: Would build Docker images"
    else
        docker-compose build
    fi
    
    # Deploy services
    log "Starting services..."
    if [[ "$DRY_RUN" == true ]]; then
        debug "DRY RUN: Would start services with docker-compose up -d"
    else
        docker-compose up -d
        
        # Wait for services to be ready
        log "Waiting for services to be ready..."
        sleep 10
        
        # Check health
        max_attempts=30
        attempt=0
        while [[ $attempt -lt $max_attempts ]]; do
            if curl -f http://localhost:5000/health >/dev/null 2>&1; then
                log "Application is ready!"
                break
            fi
            attempt=$((attempt + 1))
            debug "Health check attempt $attempt/$max_attempts"
            sleep 5
        done
        
        if [[ $attempt -eq $max_attempts ]]; then
            error "Application failed to become ready"
            exit 1
        fi
    fi
    
    log "Local deployment completed successfully"
    log "Application available at: http://localhost:5000"
}

deploy_kubernetes() {
    log "Deploying to $ENVIRONMENT Kubernetes cluster"
    
    cd "$KUBE_CONFIGS_DIR"
    
    # Update image tags in manifests
    log "Updating image tags to $IMAGE_TAG"
    if [[ "$DRY_RUN" == true ]]; then
        debug "DRY RUN: Would update image tags in manifests"
    else
        # Create temporary manifests with updated image tags
        temp_dir=$(mktemp -d)
        cp -r . "$temp_dir/"
        
        # Update web app image tag
        sed -i.bak "s|image: fm-llm-solver:web|image: ghcr.io/patrickalllencooper/fm-llm-solver:$IMAGE_TAG|g" "$temp_dir/web-app.yaml"
        
        cd "$temp_dir"
    fi
    
    # Apply manifests
    log "Applying Kubernetes manifests..."
    
    manifests=(
        "namespace.yaml"
        "secrets.yaml"
        "configmap.yaml"
        "postgres.yaml"
        "redis.yaml"
        "web-app.yaml"
        "ingress.yaml"
    )
    
    for manifest in "${manifests[@]}"; do
        log "Applying $manifest..."
        if [[ "$DRY_RUN" == true ]]; then
            debug "DRY RUN: Would apply $manifest"
            kubectl apply -f "$manifest" --dry-run=client -o yaml
        else
            kubectl apply -f "$manifest"
        fi
    done
    
    if [[ "$DRY_RUN" == false ]]; then
        # Wait for deployment to be ready
        log "Waiting for deployment to be ready..."
        if ! kubectl wait --for=condition=available --timeout="${WAIT_TIMEOUT}s" deployment/fm-llm-solver-web -n "$NAMESPACE"; then
            error "Deployment failed to become ready within ${WAIT_TIMEOUT} seconds"
            exit 1
        fi
        
        # Check pod status
        log "Checking pod status..."
        kubectl get pods -n "$NAMESPACE"
        
        # Check service endpoints
        log "Checking service endpoints..."
        kubectl get services -n "$NAMESPACE"
        
        # Cleanup temporary directory
        rm -rf "$temp_dir"
    fi
    
    log "$ENVIRONMENT deployment completed successfully"
}

rollback_deployment() {
    case $ENVIRONMENT in
        local)
            log "Rolling back local deployment..."
            docker-compose down
            ;;
        staging|production)
            log "Rolling back $ENVIRONMENT deployment..."
            kubectl rollout undo deployment/fm-llm-solver-web -n "$NAMESPACE"
            ;;
    esac
}

cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        error "Deployment failed with exit code $exit_code"
        
        read -p "Do you want to rollback? (y/n): " -r
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rollback_deployment
        fi
    fi
    exit $exit_code
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        local|staging|production)
            ENVIRONMENT="$1"
            shift
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --wait-timeout)
            WAIT_TIMEOUT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate environment
if [[ -z "$ENVIRONMENT" ]]; then
    error "Environment must be specified"
    usage
    exit 1
fi

# Set up signal handlers
trap cleanup EXIT

# Main deployment flow
log "Starting deployment to $ENVIRONMENT environment"
debug "Image tag: $IMAGE_TAG"
debug "Namespace: $NAMESPACE"
debug "Dry run: $DRY_RUN"

# Check prerequisites
check_prerequisites

# Run tests
run_tests

# Deploy based on environment
case $ENVIRONMENT in
    local)
        deploy_local
        ;;
    staging|production)
        deploy_kubernetes
        ;;
esac

log "Deployment completed successfully!" 