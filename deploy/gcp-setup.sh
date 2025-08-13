#!/bin/bash

# FM-LLM Solver GCP Deployment Setup Script
# This script sets up the complete GCP infrastructure for the FM-LLM Solver

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-fm-llm-solver}"
REGION="${REGION:-us-central1}"
DB_INSTANCE_NAME="fm-llm-postgres"
DB_NAME="fm_llm_solver"
DB_USER="fm_llm_user"
SECRET_PREFIX="fm-llm"

echo "🚀 Setting up FM-LLM Solver on GCP"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"

# Enable required APIs
echo "📡 Enabling required APIs..."
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  sql-component.googleapis.com \
  sqladmin.googleapis.com \
  secretmanager.googleapis.com \
  storage-component.googleapis.com \
  storage.googleapis.com \
  cloudidentity.googleapis.com \
  --project=$PROJECT_ID

# Create secrets
echo "🔐 Creating secrets..."

# Anthropic API Key (must be provided)
if [[ -z "$ANTHROPIC_API_KEY" ]]; then
    echo "❌ ANTHROPIC_API_KEY environment variable must be set"
    exit 1
fi

echo -n "$ANTHROPIC_API_KEY" | gcloud secrets create ${SECRET_PREFIX}-anthropic-api-key \
    --data-file=- \
    --project=$PROJECT_ID \
    --replication-policy="automatic" || echo "Secret already exists, updating..."

echo -n "$ANTHROPIC_API_KEY" | gcloud secrets versions add ${SECRET_PREFIX}-anthropic-api-key \
    --data-file=- \
    --project=$PROJECT_ID || true

# JWT Secret (generate if not provided)
JWT_SECRET="${JWT_SECRET:-$(openssl rand -base64 32)}"
echo -n "$JWT_SECRET" | gcloud secrets create ${SECRET_PREFIX}-jwt-secret \
    --data-file=- \
    --project=$PROJECT_ID \
    --replication-policy="automatic" || echo "Secret already exists, updating..."

echo -n "$JWT_SECRET" | gcloud secrets versions add ${SECRET_PREFIX}-jwt-secret \
    --data-file=- \
    --project=$PROJECT_ID || true

# Database password (generate if not provided)
DB_PASSWORD="${DB_PASSWORD:-$(openssl rand -base64 20 | tr -d '=+/' | cut -c1-16)}"
echo -n "$DB_PASSWORD" | gcloud secrets create ${SECRET_PREFIX}-db-password \
    --data-file=- \
    --project=$PROJECT_ID \
    --replication-policy="automatic" || echo "Secret already exists, updating..."

echo -n "$DB_PASSWORD" | gcloud secrets versions add ${SECRET_PREFIX}-db-password \
    --data-file=- \
    --project=$PROJECT_ID || true

# Create Cloud SQL instance
echo "💾 Creating Cloud SQL PostgreSQL instance..."
gcloud sql instances create $DB_INSTANCE_NAME \
    --database-version=POSTGRES_14 \
    --tier=db-f1-micro \
    --region=$REGION \
    --storage-type=SSD \
    --storage-size=10GB \
    --storage-auto-increase \
    --backup-start-time=03:00 \
    --maintenance-window-day=SUN \
    --maintenance-window-hour=04 \
    --maintenance-release-channel=production \
    --deletion-protection \
    --project=$PROJECT_ID || echo "SQL instance already exists"

# Set database password
echo "🔑 Setting database password..."
gcloud sql users set-password postgres \
    --instance=$DB_INSTANCE_NAME \
    --password="$DB_PASSWORD" \
    --project=$PROJECT_ID

# Create database
echo "📊 Creating database..."
gcloud sql databases create $DB_NAME \
    --instance=$DB_INSTANCE_NAME \
    --project=$PROJECT_ID || echo "Database already exists"

# Create database user
echo "👤 Creating database user..."
gcloud sql users create $DB_USER \
    --instance=$DB_INSTANCE_NAME \
    --password="$DB_PASSWORD" \
    --project=$PROJECT_ID || echo "User already exists"

# Create storage bucket for artifacts
echo "🪣 Creating storage bucket..."
BUCKET_NAME="${PROJECT_ID}-artifacts"
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$BUCKET_NAME/ || echo "Bucket already exists"

# Enable uniform bucket-level access
gsutil uniformbucketlevelaccess set on gs://$BUCKET_NAME/

# Build and deploy backend
echo "🏗️ Building and deploying backend..."
cd ../backend

# Build container image
gcloud builds submit \
    --tag gcr.io/$PROJECT_ID/fm-llm-backend:latest \
    --project=$PROJECT_ID

# Deploy to Cloud Run
gcloud run deploy fm-llm-backend \
    --image gcr.io/$PROJECT_ID/fm-llm-backend:latest \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --port 3000 \
    --memory 2Gi \
    --cpu 2 \
    --min-instances 0 \
    --max-instances 10 \
    --timeout 900 \
    --concurrency 80 \
    --set-env-vars NODE_ENV=production \
    --set-env-vars PORT=3000 \
    --set-env-vars DATABASE_URL="postgresql://$DB_USER:$DB_PASSWORD@//$DB_NAME?host=/cloudsql/$PROJECT_ID:$REGION:$DB_INSTANCE_NAME" \
    --set-env-vars BUCKET_NAME=$BUCKET_NAME \
    --set-secrets ANTHROPIC_API_KEY=${SECRET_PREFIX}-anthropic-api-key:latest \
    --set-secrets JWT_SECRET=${SECRET_PREFIX}-jwt-secret:latest \
    --add-cloudsql-instances $PROJECT_ID:$REGION:$DB_INSTANCE_NAME \
    --project=$PROJECT_ID

# Get backend URL
BACKEND_URL=$(gcloud run services describe fm-llm-backend \
    --platform managed \
    --region $REGION \
    --format 'value(status.url)' \
    --project=$PROJECT_ID)

echo "✅ Backend deployed to: $BACKEND_URL"

# Build and deploy frontend
echo "🎨 Building and deploying frontend..."
cd ../frontend

# Set backend URL in build
export VITE_API_URL="$BACKEND_URL"

# Build container image
gcloud builds submit \
    --tag gcr.io/$PROJECT_ID/fm-llm-frontend:latest \
    --build-arg VITE_API_URL=$BACKEND_URL \
    --project=$PROJECT_ID

# Deploy to Cloud Run
gcloud run deploy fm-llm-frontend \
    --image gcr.io/$PROJECT_ID/fm-llm-frontend:latest \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --port 8080 \
    --memory 512Mi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 5 \
    --timeout 60 \
    --concurrency 100 \
    --set-env-vars BACKEND_URL=$BACKEND_URL \
    --project=$PROJECT_ID

# Get frontend URL
FRONTEND_URL=$(gcloud run services describe fm-llm-frontend \
    --platform managed \
    --region $REGION \
    --format 'value(status.url)' \
    --project=$PROJECT_ID)

echo "✅ Frontend deployed to: $FRONTEND_URL"

# Run database migrations
echo "🔄 Running database migrations..."
cd ../backend

# Create a temporary Cloud Run job for migrations
gcloud run jobs create fm-llm-migrate \
    --image gcr.io/$PROJECT_ID/fm-llm-backend:latest \
    --region $REGION \
    --task-timeout 900 \
    --memory 1Gi \
    --cpu 1 \
    --set-env-vars NODE_ENV=production \
    --set-env-vars DATABASE_URL="postgresql://$DB_USER:$DB_PASSWORD@//$DB_NAME?host=/cloudsql/$PROJECT_ID:$REGION:$DB_INSTANCE_NAME" \
    --add-cloudsql-instances $PROJECT_ID:$REGION:$DB_INSTANCE_NAME \
    --command npm \
    --args run,db:migrate \
    --project=$PROJECT_ID || echo "Migration job already exists"

# Execute migration
gcloud run jobs execute fm-llm-migrate \
    --region $REGION \
    --wait \
    --project=$PROJECT_ID

echo "🎉 Deployment complete!"
echo ""
echo "📋 Deployment Summary:"
echo "Frontend URL: $FRONTEND_URL"
echo "Backend URL: $BACKEND_URL"
echo "Database: $DB_INSTANCE_NAME"
echo "Storage Bucket: gs://$BUCKET_NAME"
echo ""
echo "🔧 Next steps:"
echo "1. Update your DNS to point to $FRONTEND_URL"
echo "2. Set up custom domain in Cloud Run console"
echo "3. Configure SSL certificate"
echo "4. Test the application functionality"
echo ""
echo "📚 Useful commands:"
echo "- View logs: gcloud run services logs tail fm-llm-backend --region=$REGION"
echo "- Update backend: gcloud run services update fm-llm-backend --region=$REGION"
echo "- Scale down: gcloud run services update fm-llm-backend --max-instances=0 --region=$REGION"
