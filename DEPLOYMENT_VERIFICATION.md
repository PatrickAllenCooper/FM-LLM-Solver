# 🚀 GCP Deployment Verification Guide

## External IP Address
**34.173.180.103**

## DNS Configuration Required
Add these A records in Squarespace DNS:
```
@ → 34.173.180.103
www → 34.173.180.103
```

## Verification Steps (After DNS propagation - 5-60 minutes)

### 1. Basic Connectivity
```bash
# Test HTTP redirect
curl -I http://fmgen.net
# Should return: 308 Permanent Redirect → https://fmgen.net

# Test HTTPS
curl -I https://fmgen.net
# Should return: 200 OK (once SSL cert is ready)
```

### 2. SSL Certificate Status
```bash
kubectl describe certificate fmgen-net-tls -n fm-llm-prod
# Look for: Type: Ready, Status: True
```

### 3. Application Health
```bash
kubectl get pods -n fm-llm-prod
# Should show: 2/2 Running for fm-llm-web-*
```

### 4. Browser Tests
- Visit https://fmgen.net
- Check for valid SSL certificate (green lock)
- Verify application loads properly
- Test user registration functionality

## Monitoring Commands
```bash
# Check ingress status
kubectl get ingress -n fm-llm-prod

# Monitor SSL certificate
kubectl get certificates -n fm-llm-prod

# View application logs
kubectl logs -n fm-llm-prod -l app=fm-llm-web
```

## Expected Results
- ✅ HTTPS redirect working
- ✅ Valid SSL certificate 
- ✅ Application responds on https://fmgen.net
- ✅ User registration functions
- ✅ Professional production setup complete

## Cost Monitoring
```bash
# Run daily cost monitoring
./scripts/gcp-cost-monitor.sh
```

Target: <$100/month with user quotas and auto-scaling active. 