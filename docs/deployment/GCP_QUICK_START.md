# 🚀 GCP Professional Deployment - Quick Start

**Deploy FM-LLM-Solver to Google Cloud Platform in under 30 minutes**  
**Budget-controlled: <$100/month with user quotas**

## 📋 Prerequisites Checklist

- [ ] Google Cloud SDK installed (`gcloud --version`)
- [ ] Docker installed and running (`docker --version`)
- [ ] Active GCP billing account
- [ ] Domain: fmgen.net (configured)

## ⚡ One-Command Deployment

```bash
# Set your configuration
export PROJECT_ID="fmgen-net-production"
export BILLING_ACCOUNT="your-billing-account-id"  # Get from: gcloud billing accounts list
export SSL_EMAIL="patrick.allen.cooper@gmail.com"
export REGION="us-central1"

# Deploy everything
./scripts/deploy-gcp.sh
```

**That's it!** The script will:
- ✅ Create GCP project and enable APIs
- ✅ Set up GKE cluster with cost optimization
- ✅ Deploy PostgreSQL and Redis
- ✅ Build and deploy your application
- ✅ Configure SSL certificates
- ✅ Set up monitoring and budgets

## 🎯 Expected Results

After deployment (15-20 minutes):

```
🎉 FM-LLM-Solver Professional GCP Deployment Complete!

📊 Deployment Details:
   Project ID: fmgen-net-production
   Region: us-central1
   Cluster: fm-llm-cluster
   Domain: fmgen.net
   External IP: 34.123.456.789

🌐 DNS Configuration:
   1. Go to your Squarespace DNS settings
   2. Add A record: @ → 34.123.456.789
   3. Add A record: www → 34.123.456.789
   4. Wait 5-60 minutes for propagation

💰 Cost Management:
   • Monthly budget set to $100
   • Preemptible nodes for cost savings
   • Auto-scaling enabled
```

## 🌐 DNS Setup (Squarespace)

1. **Login to Squarespace** → Settings → Domains
2. **DNS Settings** for fmgen.net
3. **Add A Records:**
   ```
   Type: A
   Host: @
   Value: [External IP from deployment]
   
   Type: A  
   Host: www
   Value: [External IP from deployment]
   ```
4. **Wait 5-60 minutes** for DNS propagation

## 💰 Cost Monitoring

### Check Current Costs
```bash
./scripts/gcp-cost-monitor.sh
```

### Daily Cost Reports
```bash
./scripts/gcp-cost-monitor.sh report
```

### Emergency Scale Down (if costs spike)
```bash
./scripts/gcp-cost-monitor.sh emergency-scale-down
```

## 🔧 Management Commands

### View Deployment Status
```bash
kubectl get all -n fm-llm-prod
```

### Scale Application
```bash
# Scale up for high traffic
kubectl scale deployment fm-llm-web --replicas=3 -n fm-llm-prod

# Scale down to save costs
kubectl scale deployment fm-llm-web --replicas=1 -n fm-llm-prod
```

### View Logs
```bash
kubectl logs -f deployment/fm-llm-web -n fm-llm-prod
```

### Monitor Resource Usage
```bash
kubectl top pods -n fm-llm-prod
kubectl top nodes
```

## 📊 User Quotas (Built-in Cost Control)

| Tier | Daily Requests | Monthly Limit | Max Concurrent | Monthly Cost Impact |
|------|----------------|---------------|----------------|-------------------|
| **Free** | 50 | 1,000 | 1 | ~$0 |
| **Premium** | 200 | 5,000 | 3 | ~$5-10 |
| **Enterprise** | 1,000 | 20,000 | 10 | ~$20-40 |

## 🏗️ Architecture Overview

```
Internet → Load Balancer → GKE Cluster → Web App
                            ↓
                        Cloud SQL (PostgreSQL)
                            ↓
                        Redis Cache
                            ↓
                        Cloud Run (GPU Inference)
```

**Cost Breakdown:**
- **GKE Cluster**: ~$35-45/month (2x e2-small preemptible)
- **Cloud SQL**: ~$20-25/month (db-f1-micro)
- **Redis**: ~$12-15/month (basic tier)
- **Load Balancer**: ~$8-12/month
- **Storage**: ~$5-8/month
- **Total**: **~$80-105/month**

## 🎮 Testing Your Deployment

### 1. Web Interface Test
```bash
# Once DNS propagates
curl -I https://fmgen.net
# Should return: HTTP/2 200
```

### 2. Application Health Check
```bash
kubectl get pods -n fm-llm-prod
# All pods should be "Running"
```

### 3. SSL Certificate Check
```bash
echo | openssl s_client -connect fmgen.net:443 2>/dev/null | openssl x509 -noout -dates
# Should show valid certificate dates
```

## 🚨 Troubleshooting

### If Deployment Fails

```bash
# Check cluster status
gcloud container clusters describe fm-llm-cluster --zone=us-central1-b

# Check pod logs
kubectl describe pods -n fm-llm-prod

# Restart deployment
kubectl rollout restart deployment/fm-llm-web -n fm-llm-prod
```

### If Costs Are Too High

```bash
# Emergency scale down
./scripts/gcp-cost-monitor.sh emergency-scale-down

# Check for unused resources
gcloud compute disks list --filter="status:READY AND -users:*"
gcloud compute addresses list --filter="status:RESERVED AND -users:*"
```

### If SSL Doesn't Work

```bash
# Check certificate status
kubectl get certificate -n fm-llm-prod

# Force certificate renewal
kubectl delete certificate fmgen-net-tls -n fm-llm-prod
kubectl apply -f deployment/kubernetes/gcp-production.yaml
```

## 🔄 Updates & Maintenance

### Deploy New Code
```bash
# Build new image
docker build -t us-central1-docker.pkg.dev/fmgen-net-production/fm-llm-repo/fm-llm-web:latest .
docker push us-central1-docker.pkg.dev/fmgen-net-production/fm-llm-repo/fm-llm-web:latest

# Rolling update
kubectl rollout restart deployment/fm-llm-web -n fm-llm-prod
```

### Weekly Maintenance
```bash
# Check costs
./scripts/gcp-cost-monitor.sh report

# Clean up unused resources
./scripts/gcp-cost-monitor.sh optimize

# Update dependencies
gcloud components update
```

## 🏆 Success Metrics

Your deployment is successful when:

- ✅ **https://fmgen.net** loads with green SSL lock
- ✅ User registration and login works
- ✅ Certificate generation completes in <30 seconds
- ✅ Monthly costs stay under $100
- ✅ Auto-scaling responds to traffic
- ✅ Cost monitoring sends alerts at 80% budget

## 🎉 What's Next?

1. **Test User Flows**: Register, login, generate certificates
2. **Monitor Costs**: Run daily cost checks
3. **Optimize Performance**: Monitor response times and scale as needed
4. **Add Features**: Implement premium tiers, advanced analytics
5. **Scale Globally**: Add additional regions if needed

## 📞 Support Commands

```bash
# Get all deployment info
./scripts/deploy-gcp.sh status

# View application logs
./scripts/deploy-gcp.sh logs

# Scale application
./scripts/deploy-gcp.sh scale 3

# Clean up everything (careful!)
./scripts/deploy-gcp.sh cleanup
```

---

**🎯 Result**: Professional-grade FM-LLM-Solver deployment on GCP with automatic cost controls, user quotas, SSL certificates, and monitoring - all under $100/month!** 