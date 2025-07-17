# 🌐 fmgen.net Setup Guide - Complete Deployment

**Your Domain**: `fmgen.net` (Squarespace)  
**Renewal**: July 16, 2026  
**Privacy**: Enabled ✅  

## 🎯 **What We've Built**

### ✅ **Infrastructure Configured**
- **CI/CD Pipeline**: Complete automation with GitHub Actions
- **Container Registry**: GitHub Container Registry (GHCR)
- **Domain Integration**: fmgen.net configured in all deployment files
- **Multi-Environment**: Production, Staging, Development subdomains
- **SSL/TLS**: Automatic Let's Encrypt certificates
- **Security**: Comprehensive scanning and headers

### 🌍 **Domain Structure**
```
https://fmgen.net          → Production (main website)
https://www.fmgen.net      → Production (redirects to main)
https://staging.fmgen.net  → Staging environment
https://dev.fmgen.net      → Development environment
```

## 🚀 **Quick Deploy Options**

### **Option 1: Railway (Recommended - Easiest)**
**Cost**: $0-20/month • **Setup Time**: 5 minutes • **Difficulty**: Beginner

#### Step 1: Deploy to Railway
1. Go to [railway.app](https://railway.app)
2. "Deploy from GitHub" → Connect your GitHub account
3. Select `PatrickAllenCooper/FM-LLM-Solver` repository
4. Choose `main` branch
5. Railway auto-detects Dockerfile and deploys

#### Step 2: Configure Environment Variables in Railway
```bash
# Required
SECRET_KEY=your-32-character-secret-key
DB_PASSWORD=your-secure-password

# Optional APIs
MATHPIX_APP_ID=your-mathpix-id
MATHPIX_APP_KEY=your-mathpix-key
UNPAYWALL_EMAIL=your@email.com
```

#### Step 3: Get Railway URL
Railway provides: `https://your-app-name.railway.app`

### **Option 2: Render (Alternative)**
**Cost**: $0-25/month • **Setup Time**: 10 minutes • **Difficulty**: Beginner

1. Go to [render.com](https://render.com)
2. "New Web Service" → Connect GitHub
3. Select your repository
4. Configure build command: `docker build .`
5. Add environment variables

### **Option 3: Full Cloud (Production)**
**Cost**: $50-200/month • **Setup Time**: 2-4 hours • **Difficulty**: Advanced

#### Cloud Providers:
- **AWS EKS** (Most popular)
- **Google GKE** (AI-friendly)
- **Azure AKS** (Microsoft ecosystem)
- **DigitalOcean Kubernetes** (Simpler)

## 🌐 **DNS Configuration in Squarespace**

### **Step 1: Access DNS Settings**
1. Log into your Squarespace account
2. Go to **Domains** → **fmgen.net** 
3. Click **DNS Settings**

### **Step 2: Add DNS Records**

#### **For Railway/Render Deployment:**
```dns
# Main domain
Type: CNAME
Name: @
Value: your-app.railway.app (or your-app.onrender.com)

# WWW subdomain  
Type: CNAME
Name: www
Value: your-app.railway.app

# Staging subdomain
Type: CNAME
Name: staging
Value: your-staging-app.railway.app

# Development subdomain
Type: CNAME
Name: dev
Value: your-dev-app.railway.app
```

#### **For Cloud Deployment (Kubernetes):**
```dns
# Main domain - points to load balancer
Type: A
Name: @
Value: YOUR_LOAD_BALANCER_IP

# WWW subdomain
Type: CNAME
Name: www
Value: fmgen.net

# Staging
Type: A
Name: staging
Value: YOUR_STAGING_LOAD_BALANCER_IP

# Development
Type: A
Name: dev
Value: YOUR_DEV_LOAD_BALANCER_IP
```

### **Step 3: Verify DNS Propagation**
```bash
# Check if DNS is working (takes 5-60 minutes)
nslookup fmgen.net
dig fmgen.net
```

## 🔐 **Required Secrets & Configuration**

### **GitHub Repository Secrets**
Go to your GitHub repo → **Settings** → **Secrets and Variables** → **Actions**

```bash
# For Kubernetes deployment (optional)
KUBE_CONFIG_STAGING=your-staging-cluster-config
KUBE_CONFIG_PRODUCTION=your-production-cluster-config

# For notifications (optional)
SLACK_WEBHOOK_URL=your-slack-webhook
```

### **Application Environment Variables**
```bash
# Required for production
SECRET_KEY=your-32-character-secret-key
DB_PASSWORD=your-secure-database-password
ENCRYPTION_KEY=your-encryption-key

# Optional external APIs
MATHPIX_APP_ID=your-mathpix-app-id
MATHPIX_APP_KEY=your-mathpix-app-key
UNPAYWALL_EMAIL=your-email@domain.com
SEMANTIC_SCHOLAR_API_KEY=your-api-key

# Cloud storage (if using AWS)
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
```

## 📋 **Deployment Workflow**

### **Automatic Deployments**
```bash
# Your CI/CD pipeline automatically deploys:
git push origin development  # → Deploys to dev.fmgen.net
git push origin staging      # → Deploys to staging.fmgen.net  
git push origin main         # → Deploys to fmgen.net (PRODUCTION)
```

### **Manual Deployment**
```bash
# Force deploy to any environment
gh workflow run "Comprehensive CI/CD Pipeline" \
  --ref main \
  -f environment=production \
  -f force_deploy=true
```

## 🎯 **Complete Setup Checklist**

### ✅ **Already Complete**
- [x] Domain purchased and configured (fmgen.net)
- [x] CI/CD pipeline built and tested
- [x] Docker containers configured
- [x] Security scanning active
- [x] Multi-environment setup
- [x] SSL/TLS auto-configuration
- [x] Database schemas created

### 🔧 **Next Steps to Go Live**

#### **Phase 1: Quick Deploy (5 minutes)**
- [ ] Deploy to Railway/Render
- [ ] Configure environment variables
- [ ] Update DNS CNAME records in Squarespace
- [ ] Test deployment

#### **Phase 2: Custom Domain (30 minutes)**  
- [ ] Configure custom domain in Railway/Render
- [ ] Update DNS records to point to your hosting
- [ ] Enable SSL certificates
- [ ] Test all URLs (www, staging, dev)

#### **Phase 3: Production Setup (2-4 hours)**
- [ ] Choose cloud provider (AWS/GCP/Azure)
- [ ] Set up Kubernetes cluster
- [ ] Configure secrets management
- [ ] Deploy and configure monitoring

## 💰 **Cost Breakdown**

### **Current Costs**
- **Domain (fmgen.net)**: $20/year ✅ (Already paid until 2026)

### **Hosting Options**
```
┌─────────────────┬──────────────┬─────────────────┬──────────────────┐
│ Option          │ Monthly Cost │ Setup Time      │ Best For         │
├─────────────────┼──────────────┼─────────────────┼──────────────────┤
│ Railway Free    │ $0           │ 5 minutes       │ Testing          │
│ Railway Pro     │ $20          │ 5 minutes       │ Small production │
│ Render Free     │ $0           │ 10 minutes      │ Testing          │
│ Render Pro      │ $25          │ 10 minutes      │ Small production │
│ AWS/GCP Cloud   │ $50-200      │ 2-4 hours       │ Full production  │
└─────────────────┴──────────────┴─────────────────┴──────────────────┘
```

## 🎉 **Quick Start: Railway Deployment**

### **1. Deploy Now (5 minutes)**
```bash
# 1. Go to https://railway.app
# 2. Sign in with GitHub
# 3. "Deploy from GitHub" → Select your repo
# 4. Railway automatically builds and deploys!
```

### **2. Configure DNS (10 minutes)**
```bash
# In Squarespace DNS settings:
# Add CNAME: @ → your-app.railway.app
# Add CNAME: www → your-app.railway.app
```

### **3. Test Your Website**
```bash
# Your website will be live at:
# https://fmgen.net (when DNS propagates)
# https://your-app.railway.app (immediately)
```

## 🔧 **Troubleshooting**

### **Common Issues**

#### **DNS Not Working**
```bash
# Check DNS propagation
dig fmgen.net
nslookup fmgen.net

# Wait 5-60 minutes for propagation
# Clear browser cache: Ctrl+Shift+R
```

#### **SSL Certificate Issues**
```bash
# Most hosting providers auto-configure SSL
# If not working, check hosting provider SSL settings
# Enable "Force HTTPS" in your hosting dashboard
```

#### **Application Not Starting**
```bash
# Check environment variables are set
# Verify SECRET_KEY is at least 32 characters
# Check application logs in hosting dashboard
```

### **Support Resources**
- **Railway**: [railway.app/help](https://railway.app/help)
- **Render**: [render.com/docs](https://render.com/docs)
- **GitHub Actions**: Check workflow logs in Actions tab
- **DNS**: Use online DNS checker tools

## 🌟 **Your Website Architecture**

```
fmgen.net Domain (Squarespace)
│
├── DNS Records point to → Hosting Provider
│                          │
│                          ├── Load Balancer
│                          ├── SSL Termination
│                          └── Your Application
│
├── GitHub Repository
│   ├── Source Code
│   ├── CI/CD Pipeline (GitHub Actions)
│   └── Container Registry (GHCR)
│
└── Automatic Deployments
    ├── main branch → fmgen.net
    ├── staging branch → staging.fmgen.net
    └── development branch → dev.fmgen.net
```

## 🎯 **Success Criteria**

**Your deployment is successful when:**
- ✅ https://fmgen.net loads your application
- ✅ User account system works (registration/login)
- ✅ Certificate generation works
- ✅ SSL certificate is valid (green lock)
- ✅ CI/CD pipeline deploys automatically on push

**Ready to launch? Start with Railway for the quickest deployment!** 