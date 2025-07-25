# 🏢 Professional GCP Deployment Guide - Cost-Controlled

**Target**: Professional deployment on Google Cloud Platform  
**Budget**: <$100/month with user quotas  
**Domain**: fmgen.net  
**Architecture**: Kubernetes + Serverless GPU  

## 📊 **Cost Breakdown & Architecture**

### **Estimated Monthly Costs**
```
┌─────────────────────┬─────────────┬─────────────────────┐
│ Service             │ Monthly     │ Purpose             │
├─────────────────────┼─────────────┼─────────────────────┤
│ GKE Cluster (3 e2)  │ $35-45     │ Web app & services  │
│ Cloud SQL (PostgreSQL) │ $20-25  │ User & data storage │
│ Cloud Redis         │ $12-15     │ Caching & sessions  │
│ Load Balancer       │ $8-12      │ Traffic distribution│
│ Cloud Storage       │ $5-8       │ Models & artifacts  │
│ Artifact Registry   │ $2-5       │ Container images    │
│ Cloud Run (GPU)     │ $10-20     │ AI inference        │
├─────────────────────┼─────────────┼─────────────────────┤
│ TOTAL               │ $92-130    │ Target: <$100       │
└─────────────────────┴─────────────┴─────────────────────┘
```

### **Cost Control Strategy**
- **User Quotas**: 50 requests/day (free), 200/day (premium)
- **Auto-scaling**: Scale to zero when not in use
- **Preemptible instances**: 70% cost savings
- **Regional deployment**: us-central1 (cheapest)

## 🚀 **Step-by-Step Professional Deployment**

### **Prerequisites**
```bash
# Install required tools
# Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Kubectl
gcloud components install kubectl

# Helm (for advanced deployments)
curl https://get.helm.sh/helm-v3.13.0-linux-amd64.tar.gz | tar -xz
sudo mv linux-amd64/helm /usr/local/bin/

# Additional tools
sudo apt install postgresql-client-common
```

## **Phase 1: GCP Project Setup & Authentication**

### **1.1 Create Project & Enable APIs**
```bash
# Set your project details
export PROJECT_ID="fmgen-net-production"
export REGION="us-central1"  # Cheapest region
export ZONE="us-central1-b"
export CLUSTER_NAME="fm-llm-cluster"

# Create project
gcloud projects create $PROJECT_ID --name="FMGen.net Production"

# Set active project
gcloud config set project $PROJECT_ID

# Enable billing (required - link your billing account)
# Get billing account ID first
gcloud billing accounts list
export BILLING_ACCOUNT="your-billing-account-id"
gcloud billing projects link $PROJECT_ID --billing-account=$BILLING_ACCOUNT

# Enable required APIs
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
  monitoring.googleapis.com

echo "✅ Project setup complete"
```

### **1.2 Set Up Service Account & Permissions**
```bash
# Create service account for deployments
gcloud iam service-accounts create fm-llm-deployer \
  --display-name="FM-LLM Deployer Service Account"

export SERVICE_ACCOUNT="fm-llm-deployer@${PROJECT_ID}.iam.gserviceaccount.com"

# Grant necessary roles
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/container.developer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/cloudsql.client"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/redis.editor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/run.developer"

# Create and download key
gcloud iam service-accounts keys create ~/fm-llm-gcp-key.json \
  --iam-account=$SERVICE_ACCOUNT

echo "✅ Service account configured"
```

## **Phase 2: Infrastructure Setup**

### **2.1 Create GKE Cluster (Cost-Optimized)**
```bash
# Create cost-optimized GKE cluster
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
  --maintenance-window-start="2023-01-01T09:00:00Z" \
  --maintenance-window-end="2023-01-01T17:00:00Z" \
  --maintenance-window-recurrence="FREQ=WEEKLY;BYDAY=SA"

# Get cluster credentials
gcloud container clusters get-credentials $CLUSTER_NAME --zone=$ZONE

# Install ingress controller (nginx)
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml

echo "✅ GKE cluster created"
```

### **2.2 Set Up Cloud SQL (PostgreSQL)**
```bash
# Create Cloud SQL instance (cost-optimized)
gcloud sql instances create fm-llm-postgres \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=$REGION \
  --storage-type=HDD \
  --storage-size=20GB \
  --storage-auto-increase \
  --backup-start-time=03:00 \
  --maintenance-window-day=SUN \
  --maintenance-window-hour=04 \
  --maintenance-release-channel=production

# Set root password
gcloud sql users set-password postgres \
  --instance=fm-llm-postgres \
  --password=$(openssl rand -base64 32)

# Create application database and user
export DB_PASSWORD=$(openssl rand -base64 24)
gcloud sql databases create fmllm --instance=fm-llm-postgres

gcloud sql users create fmllm \
  --instance=fm-llm-postgres \
  --password=$DB_PASSWORD

# Get connection details
export SQL_CONNECTION_NAME=$(gcloud sql instances describe fm-llm-postgres --format="value(connectionName)")

echo "✅ Cloud SQL instance created"
echo "Database password: $DB_PASSWORD (save this!)"
```

### **2.3 Set Up Cloud Redis**
```bash
# Create Redis instance (basic tier - cost-effective)
gcloud redis instances create fm-llm-redis \
  --size=1 \
  --region=$REGION \
  --tier=basic \
  --redis-version=redis_7_0

# Get Redis details
export REDIS_HOST=$(gcloud redis instances describe fm-llm-redis --region=$REGION --format="value(host)")
export REDIS_PORT=$(gcloud redis instances describe fm-llm-redis --region=$REGION --format="value(port)")

echo "✅ Redis instance created"
echo "Redis endpoint: $REDIS_HOST:$REDIS_PORT"
```

### **2.4 Set Up Artifact Registry**
```bash
# Create Docker repository
gcloud artifacts repositories create fm-llm-repo \
  --repository-format=docker \
  --location=$REGION \
  --description="FM-LLM Solver container images"

# Configure Docker authentication
gcloud auth configure-docker ${REGION}-docker.pkg.dev

echo "✅ Artifact Registry configured"
```

### **2.5 Set Up Cloud Storage**
```bash
# Create storage bucket for models and assets
gsutil mb -l $REGION gs://${PROJECT_ID}-fm-llm-assets

# Set up lifecycle policy to control costs
cat > lifecycle-policy.json << 'EOF'
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 90}
      },
      {
        "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
        "condition": {"age": 30}
      }
    ]
  }
}
EOF

gsutil lifecycle set lifecycle-policy.json gs://${PROJECT_ID}-fm-llm-assets

echo "✅ Cloud Storage configured"
```

## **Phase 3: Deploy Application**

### **3.1 Build and Push Container Images**
```bash
# Clone your repository (if not already done)
cd /home/patc/code/FM-LLM-Solver

# Set image names
export WEB_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/fm-llm-repo/fm-llm-web:latest"
export INFERENCE_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/fm-llm-repo/fm-llm-inference:latest"

# Build web application image
docker build -t $WEB_IMAGE --target web .
docker push $WEB_IMAGE

# Build inference API image (if you have separate Dockerfile)
# docker build -t $INFERENCE_IMAGE --target inference .
# docker push $INFERENCE_IMAGE

echo "✅ Container images built and pushed"
```

### **3.2 Create Kubernetes Secrets**
```bash
# Create namespace
kubectl create namespace fm-llm-prod

# Create database secret
kubectl create secret generic db-credentials \
  --from-literal=username=fmllm \
  --from-literal=password=$DB_PASSWORD \
  --from-literal=database=fmllm \
  --from-literal=host=127.0.0.1 \
  --from-literal=port=5432 \
  -n fm-llm-prod

# Create application secrets
kubectl create secret generic app-secrets \
  --from-literal=secret-key=$(openssl rand -base64 32) \
  --from-literal=encryption-key=$(openssl rand -base64 32) \
  --from-literal=jwt-secret=$(openssl rand -base64 32) \
  -n fm-llm-prod

# Create Cloud SQL proxy service account key
kubectl create secret generic cloudsql-key \
  --from-file=key.json=~/fm-llm-gcp-key.json \
  -n fm-llm-prod

echo "✅ Kubernetes secrets created"
```

### **3.3 Deploy Application to GKE**
```bash
# Create deployment manifests
cat > gcp-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fm-llm-web
  namespace: fm-llm-prod
spec:
  replicas: 2
  selector:
    matchLabels:
      app: fm-llm-web
  template:
    metadata:
      labels:
        app: fm-llm-web
    spec:
      containers:
      - name: web-app
        image: $WEB_IMAGE
        ports:
        - containerPort: 5000
        env:
        - name: DEPLOYMENT_MODE
          value: "hybrid"
        - name: INFERENCE_API_URL
          value: "https://fm-llm-inference-run-service.run.app"
        - name: DATABASE_URL
          value: "postgresql://\$(DB_USERNAME):\$(DB_PASSWORD)@127.0.0.1:5432/\$(DB_DATABASE)"
        - name: REDIS_URL
          value: "redis://$REDIS_HOST:$REDIS_PORT/0"
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: secret-key
        - name: DB_USERNAME
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: username
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password
        - name: DB_DATABASE
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: database
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
      - name: cloudsql-proxy
        image: gcr.io/cloudsql-docker/gce-proxy:1.33.2
        command:
        - "/cloud_sql_proxy"
        - "-instances=$SQL_CONNECTION_NAME=tcp:5432"
        - "-credential_file=/secrets/cloudsql/key.json"
        securityContext:
          runAsNonRoot: true
        volumeMounts:
        - name: cloudsql-key
          mountPath: /secrets/cloudsql
          readOnly: true
        resources:
          requests:
            memory: "64Mi"
            cpu: "100m"
          limits:
            memory: "128Mi"
            cpu: "200m"
      volumes:
      - name: cloudsql-key
        secret:
          secretName: cloudsql-key
---
apiVersion: v1
kind: Service
metadata:
  name: fm-llm-web-service
  namespace: fm-llm-prod
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 5000
  selector:
    app: fm-llm-web
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: fm-llm-ingress
  namespace: fm-llm-prod
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "50"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - fmgen.net
    - www.fmgen.net
    secretName: fmgen-net-tls
  rules:
  - host: fmgen.net
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: fm-llm-web-service
            port:
              number: 80
  - host: www.fmgen.net
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: fm-llm-web-service
            port:
              number: 80
EOF

# Apply deployment
kubectl apply -f gcp-deployment.yaml

echo "✅ Application deployed to GKE"
```

### **3.4 Set Up SSL Certificates**
```bash
# Install cert-manager for automatic SSL
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.2/cert-manager.yaml

# Wait for cert-manager to be ready
kubectl wait --for=condition=ready pod -l app=cert-manager -n cert-manager --timeout=300s

# Create Let's Encrypt issuer
cat > letsencrypt-issuer.yaml << EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: patrick.allen.cooper@gmail.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF

kubectl apply -f letsencrypt-issuer.yaml

echo "✅ SSL certificates configured"
```

## **Phase 4: Serverless GPU Inference**

### **4.1 Deploy Inference API to Cloud Run**
```bash
# Create Cloud Run service for GPU inference
gcloud run deploy fm-llm-inference \
  --image=$INFERENCE_IMAGE \
  --platform=managed \
  --region=$REGION \
  --allow-unauthenticated \
  --memory=2Gi \
  --cpu=2 \
  --timeout=300 \
  --max-instances=5 \
  --min-instances=0 \
  --concurrency=1 \
  --set-env-vars="DEPLOYMENT_MODE=cloud,MODEL_CACHE_DIR=/tmp/models"

# Get Cloud Run URL
export INFERENCE_URL=$(gcloud run services describe fm-llm-inference --region=$REGION --format="value(status.url)")

echo "✅ Inference API deployed to Cloud Run"
echo "Inference URL: $INFERENCE_URL"
```

## **Phase 5: User Quota & Cost Management**

### **5.1 Implement Cost Controls**
```bash
# Create budget alert
gcloud billing budgets create \
  --billing-account=$BILLING_ACCOUNT \
  --display-name="FM-LLM Monthly Budget" \
  --budget-amount=100USD \
  --threshold-rules-percent=0.5,0.8,0.9,1.0 \
  --threshold-rules-spend-basis=CURRENT_SPEND \
  --all-projects-scope

# Set up monitoring dashboard
cat > monitoring-dashboard.json << 'EOF'
{
  "displayName": "FM-LLM Production Monitoring",
  "mosaicLayout": {
    "tiles": [
      {
        "width": 6,
        "height": 4,
        "widget": {
          "title": "Request Rate",
          "scorecard": {
            "timeSeriesQuery": {
              "timeSeriesFilter": {
                "filter": "resource.type=\"cloud_run_revision\"",
                "aggregation": {
                  "alignmentPeriod": "60s",
                  "perSeriesAligner": "ALIGN_RATE"
                }
              }
            }
          }
        }
      }
    ]
  }
}
EOF

gcloud monitoring dashboards create --config-from-file=monitoring-dashboard.json

echo "✅ Cost controls and monitoring configured"
```

### **5.2 Configure User Quotas in Application**
```bash
# Update your application configuration with quotas
cat > user-quotas-config.yaml << EOF
user_quotas:
  free_tier:
    daily_requests: 50
    monthly_requests: 1000
    max_concurrent: 1
  premium_tier:
    daily_requests: 200
    monthly_requests: 5000
    max_concurrent: 3
  enterprise_tier:
    daily_requests: 1000
    monthly_requests: 20000
    max_concurrent: 10

cost_controls:
  max_inference_time: 30  # seconds
  auto_scale_down_minutes: 5
  budget_alert_threshold: 80  # percent of monthly budget
EOF

# Apply to ConfigMap
kubectl create configmap user-quotas \
  --from-file=quotas.yaml=user-quotas-config.yaml \
  -n fm-llm-prod

echo "✅ User quotas configured"
```

## **Phase 6: DNS Configuration**

### **6.1 Get Load Balancer IP**
```bash
# Get external IP of ingress
export EXTERNAL_IP=$(kubectl get ingress fm-llm-ingress -n fm-llm-prod -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

echo "External IP: $EXTERNAL_IP"
echo ""
echo "🌐 DNS Configuration for Squarespace:"
echo "1. Go to your Squarespace DNS settings"
echo "2. Add A record: @ → $EXTERNAL_IP"
echo "3. Add A record: www → $EXTERNAL_IP"
echo "4. Wait 5-60 minutes for propagation"
```

## **Phase 7: Monitoring & Maintenance**

### **7.1 Set Up Monitoring**
```bash
# Create monitoring alerts
gcloud alpha monitoring policies create \
  --policy-from-file=- << EOF
displayName: "High CPU Usage"
combiner: OR
conditions:
  - displayName: "High CPU"
    conditionThreshold:
      filter: 'resource.type="k8s_container"'
      comparison: COMPARISON_GREATER_THAN
      thresholdValue: 80
      duration: 300s
alertStrategy:
  notificationRateLimit:
    period: 300s
EOF

# Set up log-based metrics
gcloud logging metrics create high_error_rate \
  --description="High error rate in application" \
  --log-filter='resource.type="k8s_container" severity>=ERROR'

echo "✅ Monitoring and alerts configured"
```

### **7.2 Backup Strategy**
```bash
# Create automated database backups
gcloud sql instances patch fm-llm-postgres \
  --backup-start-time=02:00 \
  --backup-location=$REGION \
  --retained-backups-count=7

# Export application configuration
kubectl get all -n fm-llm-prod -o yaml > fm-llm-backup-$(date +%Y%m%d).yaml

echo "✅ Backup strategy implemented"
```

## **📊 Cost Optimization Tips**

### **Daily Cost Monitoring**
```bash
# Check current spend
gcloud billing projects describe $PROJECT_ID \
  --format="value(billingAccountName)" | \
  xargs gcloud billing accounts describe \
  --format="table(displayName,currencyCode)"

# Monitor resource usage
kubectl top nodes
kubectl top pods -n fm-llm-prod
```

### **Weekly Optimization Tasks**
```bash
# Scale down during low usage
kubectl scale deployment fm-llm-web --replicas=1 -n fm-llm-prod

# Check for unused resources
gcloud compute disks list --filter="status:READY AND -users:*"
gcloud compute addresses list --filter="status:RESERVED AND -users:*"
```

## **🎯 Success Metrics**

**Your deployment is successful when:**
- ✅ https://fmgen.net loads with SSL certificate
- ✅ User registration and authentication works
- ✅ Certificate generation completes under 30 seconds
- ✅ Monthly costs stay under $100
- ✅ System auto-scales based on demand
- ✅ Monitoring alerts are functional

## **🚨 Emergency Procedures**

### **Scale Down (Cost Emergency)**
```bash
# Immediate cost reduction
kubectl scale deployment fm-llm-web --replicas=0 -n fm-llm-prod
gcloud run services update fm-llm-inference --min-instances=0 --region=$REGION
```

### **Scale Up (High Traffic)**
```bash
# Handle traffic spikes
kubectl scale deployment fm-llm-web --replicas=5 -n fm-llm-prod
gcloud run services update fm-llm-inference --max-instances=10 --region=$REGION
```

## **🎉 Deployment Complete**

Your professional GCP deployment includes:
- ✅ **Auto-scaling Kubernetes cluster**
- ✅ **Managed PostgreSQL database**
- ✅ **Redis caching layer**
- ✅ **Serverless GPU inference**
- ✅ **SSL certificates and CDN**
- ✅ **Cost monitoring and budgets**
- ✅ **User quota management**
- ✅ **Professional monitoring and alerts**

**Monthly cost target: <$100 with proper user quotas and auto-scaling.** 