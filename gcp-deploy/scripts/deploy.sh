#!/bin/bash

# FM-LLM Solver GCP Deployment Script
# Deploys to fmgen.net domain using Cloud Run

set -e

# Configuration
PROJECT_ID=${1:-"fm-llm-solver"}
REGION=${2:-"us-central1"}
DOMAIN="fmgen.net"
API_DOMAIN="api.fmgen.net"

echo "🚀 Starting FM-LLM Solver deployment to GCP"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Domain: $DOMAIN"

# Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI is not installed"
    echo "Please install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Error: No active gcloud authentication found"
    echo "Please run: gcloud auth login"
    exit 1
fi

# Set project
echo "📝 Setting GCP project..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "⚙️ Enabling required GCP APIs..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    sql-component.googleapis.com \
    sqladmin.googleapis.com \
    secretmanager.googleapis.com \
    domains.googleapis.com \
    certificatemanager.googleapis.com \
    compute.googleapis.com \
    redis.googleapis.com

# Create Cloud SQL instance
echo "🗄️ Creating Cloud SQL PostgreSQL instance..."
if ! gcloud sql instances describe fmgen-postgres --quiet 2>/dev/null; then
    gcloud sql instances create fmgen-postgres \
        --database-version=POSTGRES_14 \
        --tier=db-f1-micro \
        --region=$REGION \
        --storage-type=SSD \
        --storage-size=20GB \
        --enable-bin-log=false \
        --backup-start-time=03:00 \
        --maintenance-window-day=SUN \
        --maintenance-window-hour=04
    
    echo "⏳ Waiting for Cloud SQL instance to be ready..."
    gcloud sql instances patch fmgen-postgres --database-flags=max_connections=100
fi

# Create database
echo "📊 Creating database..."
if ! gcloud sql databases describe fm_llm_solver --instance=fmgen-postgres --quiet 2>/dev/null; then
    gcloud sql databases create fm_llm_solver --instance=fmgen-postgres
fi

# Create database user
echo "👤 Creating database user..."
DB_PASSWORD=$(openssl rand -base64 32)
if ! gcloud sql users describe fmgen-user --instance=fmgen-postgres --quiet 2>/dev/null; then
    gcloud sql users create fmgen-user \
        --instance=fmgen-postgres \
        --password="$DB_PASSWORD"
fi

# Create Redis instance
echo "🔴 Creating Redis instance..."
if ! gcloud redis instances describe fmgen-redis --region=$REGION --quiet 2>/dev/null; then
    gcloud redis instances create fmgen-redis \
        --region=$REGION \
        --memory-size=1GB \
        --network=default
fi

# Get Redis connection string
REDIS_HOST=$(gcloud redis instances describe fmgen-redis --region=$REGION --format="value(host)")
REDIS_PORT=$(gcloud redis instances describe fmgen-redis --region=$REGION --format="value(port)")
REDIS_URL="redis://$REDIS_HOST:$REDIS_PORT"

# Create secrets
echo "🔐 Creating secrets..."
JWT_SECRET=$(openssl rand -base64 64)

# Check if secrets exist, create if not
if ! gcloud secrets describe fmgen-secrets --quiet 2>/dev/null; then
    # Create secrets
    echo "$DB_PASSWORD" | gcloud secrets create db-password --data-file=-
    echo "fmgen-user" | gcloud secrets create db-user --data-file=-
    echo "$JWT_SECRET" | gcloud secrets create jwt-secret --data-file=-
    echo "$REDIS_URL" | gcloud secrets create redis-url --data-file=-
    
    # Create combined secret for easy access
    cat <<EOF | gcloud secrets create fmgen-secrets --data-file=-
{
  "db-user": "fmgen-user",
  "db-password": "$DB_PASSWORD",
  "jwt-secret": "$JWT_SECRET",
  "redis-url": "$REDIS_URL"
}
EOF
    
    echo "⚠️  IMPORTANT: Please add your Anthropic API key:"
    echo "echo 'YOUR_ANTHROPIC_API_KEY' | gcloud secrets create anthropic-api-key --data-file=-"
fi

# Build and push backend image
echo "🏗️ Building backend image..."
cd "$(dirname "$0")/../.."
gcloud builds submit . \
    --config=gcp-deploy/cloudbuild-backend.yaml \
    --substitutions=_PROJECT_ID=$PROJECT_ID

# Build and push frontend image
echo "🎨 Building frontend image..."
gcloud builds submit . \
    --config=gcp-deploy/cloudbuild-frontend.yaml \
    --substitutions=_PROJECT_ID=$PROJECT_ID

# Deploy backend service
echo "🚀 Deploying backend service..."
sed "s/PROJECT_ID/$PROJECT_ID/g; s/REGION/$REGION/g" gcp-deploy/cloud-run/fmgen-api.yaml | \
    gcloud run services replace - --region=$REGION

# Deploy frontend service
echo "🌐 Deploying frontend service..."
sed "s/PROJECT_ID/$PROJECT_ID/g" gcp-deploy/cloud-run/fmgen-ui.yaml | \
    gcloud run services replace - --region=$REGION

# Allow unauthenticated access
echo "🔓 Configuring public access..."
gcloud run services add-iam-policy-binding fmgen-api \
    --region=$REGION \
    --member="allUsers" \
    --role="roles/run.invoker"

gcloud run services add-iam-policy-binding fmgen-ui \
    --region=$REGION \
    --member="allUsers" \
    --role="roles/run.invoker"

# Get service URLs
API_URL=$(gcloud run services describe fmgen-api --region=$REGION --format="value(status.url)")
UI_URL=$(gcloud run services describe fmgen-ui --region=$REGION --format="value(status.url)")

echo "✅ Deployment completed!"
echo ""
echo "🔗 Service URLs:"
echo "Backend API: $API_URL"
echo "Frontend UI: $UI_URL"
echo ""
echo "🌐 Next steps for custom domain:"
echo "1. Configure domain mapping:"
echo "   gcloud run domain-mappings create --service=fmgen-ui --domain=$DOMAIN --region=$REGION"
echo "   gcloud run domain-mappings create --service=fmgen-api --domain=$API_DOMAIN --region=$REGION"
echo ""
echo "2. Update DNS records to point to the Cloud Run services"
echo "3. SSL certificates will be automatically provisioned"
echo ""
echo "📊 Database connection string:"
echo "Host: /cloudsql/$PROJECT_ID:$REGION:fmgen-postgres"
echo "Database: fm_llm_solver"
echo "User: fmgen-user"
echo ""
echo "🔐 Don't forget to set your Anthropic API key:"
echo "echo 'YOUR_API_KEY' | gcloud secrets versions add anthropic-api-key --data-file=-"

