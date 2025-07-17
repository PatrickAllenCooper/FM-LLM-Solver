#!/bin/bash
set -e

# FM-LLM Solver Model Storage Setup
# This script sets up Google Cloud Storage for models and knowledge base files

echo "🚀 Setting up FM-LLM Solver Model Storage on Google Cloud..."

# Configuration
PROJECT_ID=${PROJECT_ID:-"fmgen-net-production"}
REGION=${REGION:-"us-central1"}
BUCKET_NAME="fm-llm-models-${PROJECT_ID}"
KB_BUCKET_NAME="fm-llm-knowledge-base-${PROJECT_ID}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    print_error "Please authenticate with Google Cloud: gcloud auth login"
    exit 1
fi

# Set project
gcloud config set project $PROJECT_ID
print_status "Using project: $PROJECT_ID"

echo ""
echo "📦 Creating Google Cloud Storage buckets..."

# Create models bucket
if gsutil ls -b gs://$BUCKET_NAME >/dev/null 2>&1; then
    print_warning "Models bucket already exists: gs://$BUCKET_NAME"
else
    gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$BUCKET_NAME
    print_status "Created models bucket: gs://$BUCKET_NAME"
fi

# Create knowledge base bucket
if gsutil ls -b gs://$KB_BUCKET_NAME >/dev/null 2>&1; then
    print_warning "Knowledge base bucket already exists: gs://$KB_BUCKET_NAME"
else
    gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$KB_BUCKET_NAME
    print_status "Created knowledge base bucket: gs://$KB_BUCKET_NAME"
fi

echo ""
echo "🔐 Setting up bucket permissions..."

# Set bucket permissions for the compute service account
SERVICE_ACCOUNT="${PROJECT_ID}-compute@developer.gserviceaccount.com"

# Grant storage object viewer access to models bucket
gsutil iam ch serviceAccount:$SERVICE_ACCOUNT:objectViewer gs://$BUCKET_NAME
print_status "Granted access to models bucket for service account"

# Grant storage object viewer access to knowledge base bucket
gsutil iam ch serviceAccount:$SERVICE_ACCOUNT:objectViewer gs://$KB_BUCKET_NAME
print_status "Granted access to knowledge base bucket for service account"

echo ""
echo "📁 Checking for local models and knowledge base..."

# Function to upload directory if it exists
upload_if_exists() {
    local source_dir=$1
    local bucket=$2
    local description=$3
    
    if [ -d "$source_dir" ]; then
        echo "📤 Uploading $description..."
        gsutil -m rsync -r -d "$source_dir" "gs://$bucket/"
        print_status "Uploaded $description to gs://$bucket/"
    else
        print_warning "$description not found at $source_dir - you can upload it later"
        echo "   To upload when ready: gsutil -m rsync -r -d \"$source_dir\" \"gs://$bucket/\""
    fi
}

# Upload models if they exist
MODEL_DIRS=(
    "models"
    "fine_tuning/output"
    "output/models"
)

for model_dir in "${MODEL_DIRS[@]}"; do
    if [ -d "$model_dir" ]; then
        upload_if_exists "$model_dir" "$BUCKET_NAME" "Models from $model_dir"
        break
    fi
done

# Upload knowledge bases if they exist
KB_DIRS=(
    "knowledge_base/output"
    "output/knowledge_base"
    "data/knowledge_base"
)

for kb_dir in "${KB_DIRS[@]}"; do
    if [ -d "$kb_dir" ]; then
        upload_if_exists "$kb_dir" "$KB_BUCKET_NAME" "Knowledge base from $kb_dir"
        break
    fi
done

echo ""
echo "🔧 Creating download script for containers..."

# Create a download script that containers can use
cat > download-models.sh << 'EOF'
#!/bin/bash
# Model download script for containers
set -e

MODELS_BUCKET=${MODELS_BUCKET:-""}
KB_BUCKET=${KB_BUCKET:-""}

if [ -n "$MODELS_BUCKET" ]; then
    echo "📥 Downloading models from $MODELS_BUCKET..."
    mkdir -p /app/models
    gsutil -m rsync -r gs://$MODELS_BUCKET/ /app/models/
    echo "✅ Models downloaded"
fi

if [ -n "$KB_BUCKET" ]; then
    echo "📥 Downloading knowledge base from $KB_BUCKET..."
    mkdir -p /app/knowledge_base
    gsutil -m rsync -r gs://$KB_BUCKET/ /app/knowledge_base/
    echo "✅ Knowledge base downloaded"
fi
EOF

chmod +x download-models.sh
print_status "Created download-models.sh script"

echo ""
echo "📋 Storage Setup Summary:"
echo "========================"
echo "Models Bucket:      gs://$BUCKET_NAME"
echo "Knowledge Base:     gs://$KB_BUCKET_NAME" 
echo "Service Account:    $SERVICE_ACCOUNT"
echo "Download Script:    ./download-models.sh"

echo ""
echo "🎯 Next Steps:"
echo "1. Upload your trained models: gsutil -m rsync -r -d <model_directory> gs://$BUCKET_NAME/"
echo "2. Upload knowledge base: gsutil -m rsync -r -d <kb_directory> gs://$KB_BUCKET_NAME/"
echo "3. Update Kubernetes deployment with bucket environment variables"

echo ""
echo "🌟 Environment Variables for Deployment:"
echo "MODELS_BUCKET=$BUCKET_NAME"
echo "KB_BUCKET=$KB_BUCKET_NAME"

print_status "Model storage setup complete!" 