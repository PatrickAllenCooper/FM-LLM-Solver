#!/bin/bash

# Professional GCP Deployment Script for FM-LLM-Solver
# Cost-controlled deployment under $100/month with user quotas

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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
        log_error "gcloud CLI is not installed. Please install Google Cloud SDK first."
        exit 1
    fi
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Installing..."
        gcloud components install kubectl
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Configuration
setup_config() {
    log_info "Setting up configuration..."
    
    # Default values
    export PROJECT_ID="${PROJECT_ID:-fmgen-net-production}"
    export REGION="${REGION:-us-central1}"
    export ZONE="${ZONE:-us-central1-b}"
    export CLUSTER_NAME="${CLUSTER_NAME:-fm-llm-cluster}"
    export DOMAIN="${DOMAIN:-fmgen.net}"
    
    # Prompt for billing account if not set
    if [ -z "$BILLING_ACCOUNT" ]; then
        echo "Available billing accounts:"
        gcloud billing accounts list
        echo ""
        read -p "Enter your billing account ID: " BILLING_ACCOUNT
        export BILLING_ACCOUNT
    fi
    
    # Prompt for email for SSL certificates
    if [ -z "$SSL_EMAIL" ]; then
        read -p "Enter your email for SSL certificates: " SSL_EMAIL
        export SSL_EMAIL
    fi
    
    log_success "Configuration complete"
    log_info "Project ID: $PROJECT_ID"
    log_info "Region: $REGION"
    log_info "Domain: $DOMAIN"
}

# Create GCP project
create_project() {
    log_info "Creating GCP project..."
    
    # Check if project exists
    if gcloud projects describe $PROJECT_ID &>/dev/null; then
        log_warning "Project $PROJECT_ID already exists, skipping creation"
    else
        gcloud projects create $PROJECT_ID --name="FMGen Production"
        log_success "Project created"
    fi
    
    # Set active project
    gcloud config set project $PROJECT_ID
    
    # Link billing account
    gcloud billing projects link $PROJECT_ID --billing-account=$BILLING_ACCOUNT
    
    log_success "Project setup complete"
}

# Enable required APIs
enable_apis() {
    log_info "Enabling required APIs..."
    
    gcloud services enable \
        container.googleapis.com \
        sql-component.googleapis.com \
        sqladmin.googleapis.com \
        redis.googleapis.com \
        storage.googleapis.com \
        artifactregistry.googleapis.com \
        run.googleapis.com \
        cloudbuild.googleapis.com \
        compute.googleapis.com \
        secretmanager.googleapis.com \
        monitoring.googleapis.com \
        --quiet
    
    log_success "APIs enabled"
}

# Create service account
create_service_account() {
    log_info "Creating service account..."
    
    export SERVICE_ACCOUNT="fm-llm-deployer@${PROJECT_ID}.iam.gserviceaccount.com"
    
    # Check if service account exists
    if gcloud iam service-accounts describe $SERVICE_ACCOUNT &>/dev/null; then
        log_warning "Service account already exists, skipping creation"
    else
        gcloud iam service-accounts create fm-llm-deployer \
            --display-name="FM-LLM Deployer Service Account"
    fi
    
    # Grant roles
    local roles=(
        "roles/container.developer"
        "roles/cloudsql.client"
        "roles/redis.editor"
        "roles/run.developer"
        "roles/storage.admin"
        "roles/secretmanager.admin"
    )
    
    for role in "${roles[@]}"; do
        gcloud projects add-iam-policy-binding $PROJECT_ID \
            --member="serviceAccount:${SERVICE_ACCOUNT}" \
            --role="$role" --quiet
    done
    
    # Create service account key
    if [ ! -f ~/fm-llm-gcp-key.json ]; then
        gcloud iam service-accounts keys create ~/fm-llm-gcp-key.json \
            --iam-account=$SERVICE_ACCOUNT
    fi
    
    log_success "Service account configured"
}

# Create GKE cluster
create_gke_cluster() {
    log_info "Creating GKE cluster..."
    
    # Check if cluster exists
    if gcloud container clusters describe $CLUSTER_NAME --zone=$ZONE &>/dev/null; then
        log_warning "Cluster already exists, skipping creation"
    else
        gcloud container clusters create $CLUSTER_NAME \
            --zone=$ZONE \
            --machine-type=e2-small \
            --num-nodes=2 \
            --enable-autoscaling \
            --min-nodes=1 \
            --max-nodes=4 \
            --preemptible \
            --disk-size=20GB \
            --disk-type=pd-standard \
            --enable-autorepair \
            --enable-autoupgrade \
            --quiet
    fi
    
    # Get credentials
    gcloud container clusters get-credentials $CLUSTER_NAME --zone=$ZONE
    
    # Install ingress controller
    kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml
    
    log_success "GKE cluster created and configured"
}

# Create Cloud SQL instance
create_cloud_sql() {
    log_info "Creating Cloud SQL instance..."
    
    # Check if instance exists
    if gcloud sql instances describe fm-llm-postgres &>/dev/null; then
        log_warning "Cloud SQL instance already exists, skipping creation"
    else
        gcloud sql instances create fm-llm-postgres \
            --database-version=POSTGRES_15 \
            --tier=db-f1-micro \
            --region=$REGION \
            --storage-type=HDD \
            --storage-size=20GB \
            --storage-auto-increase \
            --backup-start-time=03:00 \
            --quiet
    fi
    
    # Generate password and create database
    export DB_PASSWORD=$(openssl rand -base64 24)
    echo "Database password: $DB_PASSWORD" > ~/fm-llm-db-password.txt
    
    # Create database and user
    gcloud sql databases create fmllm --instance=fm-llm-postgres || true
    gcloud sql users create fmllm \
        --instance=fm-llm-postgres \
        --password=$DB_PASSWORD || true
    
    # Get connection name
    export SQL_CONNECTION_NAME=$(gcloud sql instances describe fm-llm-postgres --format="value(connectionName)")
    
    log_success "Cloud SQL instance created"
}

# Create Redis instance
create_redis() {
    log_info "Creating Redis instance..."
    
    # Check if instance exists
    if gcloud redis instances describe fm-llm-redis --region=$REGION &>/dev/null; then
        log_warning "Redis instance already exists, skipping creation"
    else
        gcloud redis instances create fm-llm-redis \
            --size=1 \
            --region=$REGION \
            --tier=basic \
            --redis-version=redis_7_0 \
            --quiet
    fi
    
    export REDIS_HOST=$(gcloud redis instances describe fm-llm-redis --region=$REGION --format="value(host)")
    export REDIS_PORT=$(gcloud redis instances describe fm-llm-redis --region=$REGION --format="value(port)")
    
    log_success "Redis instance created"
}

# Create Artifact Registry
create_artifact_registry() {
    log_info "Creating Artifact Registry..."
    
    # Check if repository exists
    if gcloud artifacts repositories describe fm-llm-repo --location=$REGION &>/dev/null; then
        log_warning "Artifact Registry already exists, skipping creation"
    else
        gcloud artifacts repositories create fm-llm-repo \
            --repository-format=docker \
            --location=$REGION \
            --description="FM-LLM Solver container images"
    fi
    
    # Configure Docker authentication
    gcloud auth configure-docker ${REGION}-docker.pkg.dev
    
    log_success "Artifact Registry configured"
}

# Build and push images
build_and_push_images() {
    log_info "Building and pushing container images..."
    
    export WEB_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/fm-llm-repo/fm-llm-web:latest"
    
    # Build web application image
    docker build -t $WEB_IMAGE .
    docker push $WEB_IMAGE
    
    log_success "Container images built and pushed"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Create namespace
    kubectl create namespace fm-llm-prod || true
    
    # Create secrets
    kubectl create secret generic db-credentials \
        --from-literal=username=fmllm \
        --from-literal=password=$DB_PASSWORD \
        --from-literal=database=fmllm \
        --from-literal=host=127.0.0.1 \
        --from-literal=port=5432 \
        -n fm-llm-prod --dry-run=client -o yaml | kubectl apply -f -
    
    kubectl create secret generic app-secrets \
        --from-literal=secret-key=$(openssl rand -base64 32) \
        --from-literal=encryption-key=$(openssl rand -base64 32) \
        --from-literal=jwt-secret=$(openssl rand -base64 32) \
        -n fm-llm-prod --dry-run=client -o yaml | kubectl apply -f -
    
    kubectl create secret generic cloudsql-key \
        --from-file=key.json=~/fm-llm-gcp-key.json \
        -n fm-llm-prod --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply deployment
    envsubst < deployment/kubernetes/gcp-production.yaml | kubectl apply -f -
    
    log_success "Application deployed to Kubernetes"
}

# Setup SSL certificates
setup_ssl() {
    log_info "Setting up SSL certificates..."
    
    # Install cert-manager
    kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.2/cert-manager.yaml
    
    # Wait for cert-manager
    kubectl wait --for=condition=ready pod -l app=cert-manager -n cert-manager --timeout=300s
    
    # Create Let's Encrypt issuer
    cat << EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: $SSL_EMAIL
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
    
    log_success "SSL certificates configured"
}

# Setup monitoring and budgets
setup_monitoring() {
    log_info "Setting up monitoring and cost controls..."
    
    # Create budget
    gcloud billing budgets create \
        --billing-account=$BILLING_ACCOUNT \
        --display-name="FM-LLM Monthly Budget" \
        --budget-amount=100USD \
        --threshold-rules-percent=0.5,0.8,0.9,1.0 \
        --threshold-rules-spend-basis=CURRENT_SPEND \
        --all-projects-scope --quiet || true
    
    log_success "Monitoring and budgets configured"
}

# Get deployment information
get_deployment_info() {
    log_info "Getting deployment information..."
    
    # Get external IP
    sleep 60  # Wait for load balancer
    export EXTERNAL_IP=$(kubectl get ingress fm-llm-ingress -n fm-llm-prod -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "Pending...")
    
    log_success "Deployment complete!"
    echo ""
    echo "🎉 FM-LLM-Solver Professional GCP Deployment Complete!"
    echo ""
    echo "📊 Deployment Details:"
    echo "   Project ID: $PROJECT_ID"
    echo "   Region: $REGION"
    echo "   Cluster: $CLUSTER_NAME"
    echo "   Domain: $DOMAIN"
    echo "   External IP: $EXTERNAL_IP"
    echo ""
    echo "🌐 DNS Configuration:"
    echo "   1. Go to your Squarespace DNS settings"
    echo "   2. Add A record: @ → $EXTERNAL_IP"
    echo "   3. Add A record: www → $EXTERNAL_IP"
    echo "   4. Wait 5-60 minutes for propagation"
    echo ""
    echo "💰 Cost Management:"
    echo "   • Monthly budget set to \$100"
    echo "   • Preemptible nodes for cost savings"
    echo "   • Auto-scaling enabled"
    echo "   • Database password saved to: ~/fm-llm-db-password.txt"
    echo ""
    echo "🔧 Management Commands:"
    echo "   • View pods: kubectl get pods -n fm-llm-prod"
    echo "   • View logs: kubectl logs -f deployment/fm-llm-web -n fm-llm-prod"
    echo "   • Scale up: kubectl scale deployment fm-llm-web --replicas=3 -n fm-llm-prod"
    echo "   • Scale down: kubectl scale deployment fm-llm-web --replicas=1 -n fm-llm-prod"
}

# Main execution
main() {
    echo "🚀 FM-LLM-Solver Professional GCP Deployment"
    echo "Target: <\$100/month with professional infrastructure"
    echo ""
    
    check_prerequisites
    setup_config
    create_project
    enable_apis
    create_service_account
    create_gke_cluster
    create_cloud_sql
    create_redis
    create_artifact_registry
    build_and_push_images
    deploy_to_kubernetes
    setup_ssl
    setup_monitoring
    get_deployment_info
}

# Handle script arguments
case "${1:-}" in
    "cleanup")
        log_warning "Cleaning up GCP resources..."
        gcloud container clusters delete $CLUSTER_NAME --zone=$ZONE --quiet || true
        gcloud sql instances delete fm-llm-postgres --quiet || true
        gcloud redis instances delete fm-llm-redis --region=$REGION --quiet || true
        gcloud projects delete $PROJECT_ID --quiet || true
        log_success "Cleanup complete"
        ;;
    "status")
        kubectl get all -n fm-llm-prod
        ;;
    "logs")
        kubectl logs -f deployment/fm-llm-web -n fm-llm-prod
        ;;
    "scale")
        if [ -z "$2" ]; then
            echo "Usage: $0 scale <replicas>"
            exit 1
        fi
        kubectl scale deployment fm-llm-web --replicas=$2 -n fm-llm-prod
        ;;
    *)
        main
        ;;
esac 