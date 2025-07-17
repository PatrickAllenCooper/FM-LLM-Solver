# 🚀 FM-LLM Solver Enhanced User Account System Deployment Guide

## Overview

This guide walks you through deploying the enhanced FM-LLM Solver with comprehensive user account management, credential tracking, and certificate generation history.

## 🎯 **What's New in the Enhanced System**

### **User Account Features**
- ✅ **Enhanced User Profiles**: Full name, organization, bio, location, timezone
- ✅ **Subscription Management**: Free, premium, enterprise tiers with usage limits
- ✅ **API Key Management**: Generate, revoke, track usage
- ✅ **Two-Factor Authentication**: Enhanced security options
- ✅ **Session Tracking**: Device, browser, IP, login method tracking
- ✅ **Activity Logging**: Comprehensive user activity analytics

### **Certificate Generation Tracking**
- ✅ **Detailed History**: System type, complexity, model used, processing time
- ✅ **User Ratings & Feedback**: 5-star rating system with comments
- ✅ **Certificate Favorites**: Save, tag, and organize certificates
- ✅ **Performance Metrics**: Token usage, cost estimates, quality scores
- ✅ **Domain Bounds Tracking**: Certificate validity regions
- ✅ **Verification Integration**: Track verification attempts and success rates

### **Analytics & Monitoring**
- ✅ **Usage Statistics**: Daily/monthly limits, request tracking
- ✅ **Quality Metrics**: Mathematical soundness, confidence scores
- ✅ **Security Logs**: Login attempts, rate limiting, IP blocking
- ✅ **Performance Tracking**: Response times, error rates

## 📋 **Pre-Deployment Checklist**

### **1. Environment Setup**
```bash
# Clone the repository (if not already done)
git clone https://github.com/your-repo/FM-LLM-Solver.git
cd FM-LLM-Solver

# Verify Python version (3.8+)
python --version

# Install dependencies
pip install -r requirements.txt
```

### **2. Database Configuration**
```bash
# Create directories
mkdir -p instance uploads logs

# Initialize database with enhanced schema
sqlite3 instance/production.db < sql/init.sql

# Verify tables were created
sqlite3 instance/production.db ".tables"
```

### **3. Environment Variables**
Update your `.env` file with the enhanced configuration:

```env
# =================================
# PRODUCTION SECURITY CONFIGURATION
# =================================
SECRET_KEY=your-secret-key-here
DB_PASSWORD=your-db-password
REDIS_PASSWORD=your-redis-password
JWT_SECRET_KEY=your-jwt-secret
ENCRYPTION_KEY=your-encryption-key

# DEPLOYMENT CONFIGURATION
FM_LLM_ENV=production
DEPLOYMENT_MODE=hybrid

# DATABASE (Production)
DATABASE_URL=sqlite:///instance/production.db
REDIS_URL=redis://localhost:6379/0

# USER ACCOUNT SYSTEM SETTINGS
USER_REGISTRATION_ENABLED=true
EMAIL_VERIFICATION_REQUIRED=false
PASSWORD_RESET_ENABLED=true
TWO_FACTOR_AUTH_ENABLED=true
SESSION_TIMEOUT_HOURS=24
API_KEY_EXPIRY_DAYS=365

# EMAIL SETTINGS (for user notifications)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_USE_TLS=true
FROM_EMAIL=noreply@fm-llm-solver.com

# ANALYTICS AND MONITORING
ANALYTICS_ENABLED=true
ACTIVITY_LOGGING_ENABLED=true
SECURITY_LOGGING_ENABLED=true
USER_METRICS_ENABLED=true

# API KEYS (existing)
MATHPIX_APP_ID=your-mathpix-id
MATHPIX_APP_KEY=your-mathpix-key
UNPAYWALL_EMAIL=your-email@domain.com
SEMANTIC_SCHOLAR_API_KEY=your-semantic-scholar-key

# INFERENCE API (Modal or cloud endpoint)
INFERENCE_API_URL=your-modal-endpoint-url
INFERENCE_API_KEY=your-inference-api-key
```

## 🚀 **Deployment Options**

### **Option 1: Quick Local Development Setup**

```bash
# 1. Start Redis (if not using Docker)
redis-server

# 2. Run the enhanced web interface
python web_interface/app.py

# 3. Access the application
# Web Interface: http://localhost:5000
# Default Admin: admin / admin123!
```

### **Option 2: Hybrid Production Deployment (Recommended)**

```bash
# 1. Deploy inference API to Modal
modal token new  # Authenticate with Modal
modal deploy modal_inference_app.py

# 2. Update INFERENCE_API_URL in .env with the Modal endpoint

# 3. Start services with Docker Compose
docker-compose -f docker-compose.hybrid.yml up -d

# 4. Verify deployment
curl http://localhost:5000/health
```

### **Option 3: Full Production with PostgreSQL & Monitoring**

```bash
# Start all services including PostgreSQL and monitoring
docker-compose -f docker-compose.hybrid.yml --profile postgres --profile monitoring up -d

# Access services:
# - Web App: http://localhost:5000
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9090
```

## 🧪 **Testing the Enhanced System**

### **1. Run Unit Tests**
```bash
# Test the user account system
python -m pytest tests/test_user_account_system.py -v

# Test certificate generation with user tracking
python -m pytest tests/test_certificate_pipeline.py -v

# Run all tests
python -m pytest tests/ -v --tb=short
```

### **2. Manual Testing Checklist**

#### **User Registration & Authentication**
- [ ] Register new user account
- [ ] Login/logout functionality
- [ ] Password strength validation
- [ ] API key generation and revocation
- [ ] Profile updates and preferences

#### **Certificate Generation with Tracking**
- [ ] Generate certificate as logged-in user
- [ ] Verify history tracking in profile
- [ ] Rate limiting enforcement
- [ ] User activity logging
- [ ] Certificate favorites and ratings

#### **Admin Features**
- [ ] Admin panel access
- [ ] User management capabilities
- [ ] System statistics and analytics
- [ ] Security logs monitoring

## 📊 **User Dashboard Features**

### **Profile Overview**
- User statistics (certificates generated, verifications)
- Usage limits and current consumption
- Subscription status and renewal dates
- Recent activity summary

### **Certificate History**
- Complete generation history with metadata
- Search and filter capabilities
- Export functionality (JSON/CSV)
- Favorites management with tags

### **Analytics Dashboard**
- Usage patterns and trends
- Model performance comparisons
- Cost tracking and optimization
- Quality metrics over time

### **Account Management**
- Profile information and preferences
- Security settings and session management
- API key management and usage tracking
- Data export and account deletion

## 🔧 **Configuration Management**

### **User Subscription Tiers**

```python
# Free Tier (Default)
daily_request_limit = 50
monthly_request_limit = 1000
max_concurrent_requests = 3

# Premium Tier
daily_request_limit = 200
monthly_request_limit = 5000
max_concurrent_requests = 10

# Enterprise Tier
daily_request_limit = 1000
monthly_request_limit = 25000
max_concurrent_requests = 50
```

### **Model Access Controls**
- Free users: Access to smaller models (1.5B parameters)
- Premium users: Access to medium models (up to 7B parameters)
- Enterprise users: Access to all models including custom fine-tuned versions

### **Rate Limiting Configuration**
- API endpoints: 1000 requests/hour
- Web interface: 100 requests/day
- Certificate generation: Based on subscription tier
- Verification requests: 50% of generation limit

## 🔐 **Security Features**

### **Authentication Security**
- Password strength enforcement (8+ chars, mixed case, numbers, symbols)
- Account lockout after 5 failed attempts
- Session timeout and management
- Two-factor authentication support
- API key rotation and expiry

### **Data Protection**
- User data encryption at rest
- Secure password hashing (pbkdf2:sha256)
- IP address tracking and blocking
- Rate limiting and DDoS protection
- GDPR-compliant data export and deletion

### **Audit Logging**
- All user activities logged with timestamps
- Security events tracked (login failures, suspicious activity)
- Certificate generation history with full metadata
- API usage tracking and monitoring

## 📈 **Monitoring & Analytics**

### **Key Metrics Tracked**
- **User Engagement**: Login frequency, session duration, feature usage
- **System Performance**: Response times, error rates, throughput
- **Certificate Quality**: Success rates, verification accuracy, user ratings
- **Resource Usage**: Token consumption, cost per request, model efficiency

### **Grafana Dashboards**
- User activity and engagement metrics
- System performance and health monitoring
- Certificate generation analytics
- Cost and resource utilization tracking

## 🚨 **Troubleshooting Common Issues**

### **Database Issues**
```bash
# Reset database with new schema
rm instance/production.db
sqlite3 instance/production.db < sql/init.sql

# Check database integrity
sqlite3 instance/production.db "PRAGMA integrity_check;"
```

### **Authentication Issues**
```bash
# Create emergency admin user
python -c "
from web_interface.app import app
from web_interface.models import db, User
with app.app_context():
    admin = User(username='emergency', email='admin@example.com', role='admin')
    admin.set_password('TempPass123!')
    db.session.add(admin)
    db.session.commit()
"
```

### **Performance Issues**
```bash
# Monitor Redis usage
redis-cli info memory

# Check database performance
sqlite3 instance/production.db "ANALYZE;"

# Monitor API response times
curl -w "@curl-format.txt" http://localhost:5000/api/models
```

## 🔄 **Backup & Recovery**

### **Database Backup**
```bash
# Backup SQLite database
cp instance/production.db backups/production_$(date +%Y%m%d_%H%M%S).db

# Backup with compression
tar -czf backups/fm-llm-backup-$(date +%Y%m%d).tar.gz instance/ logs/ uploads/
```

### **User Data Export**
```bash
# Export all user data (GDPR compliance)
python scripts/export_user_data.py --user-id 123 --output-format json

# Bulk export for migration
python scripts/bulk_export.py --start-date 2024-01-01 --output-dir exports/
```

## 📱 **API Documentation**

### **Enhanced API Endpoints**

```bash
# User Management
GET    /api/user/profile          # Get user profile
PUT    /api/user/profile          # Update user profile
GET    /api/user/usage-stats      # Get usage statistics
GET    /api/user/certificates     # Get certificate history

# Certificate Management
POST   /api/certificates/generate # Generate with user tracking
GET    /api/certificates/{id}     # Get certificate details
POST   /api/certificates/{id}/favorite # Add to favorites
POST   /api/certificates/{id}/rate     # Rate certificate

# Admin Endpoints
GET    /api/admin/users           # List all users
GET    /api/admin/statistics      # System statistics
GET    /api/admin/security-logs   # Security event logs
```

## 🎉 **Success Verification**

After deployment, verify these features work:

### **✅ User Account Features**
- [ ] User registration and email verification
- [ ] Login/logout with session tracking
- [ ] Profile management and preferences
- [ ] API key generation and management
- [ ] Subscription tier enforcement

### **✅ Certificate Generation Tracking**
- [ ] Generate certificate with user context
- [ ] View generation history in profile
- [ ] Rate and favorite certificates
- [ ] Export certificate data
- [ ] Track verification results

### **✅ Analytics & Monitoring**
- [ ] User activity logging
- [ ] Performance metrics collection
- [ ] Security event monitoring
- [ ] Usage statistics and reporting

### **✅ Admin Features**
- [ ] User management interface
- [ ] System health monitoring
- [ ] Security log analysis
- [ ] Performance optimization tools

## 🎯 **Next Steps**

1. **Set up email notifications** for user account events
2. **Configure monitoring alerts** for system health
3. **Implement subscription billing** (Stripe integration ready)
4. **Add mobile-responsive design** improvements
5. **Deploy to production infrastructure** with load balancing

---

## 📞 **Support & Documentation**

- **API Reference**: `/api/docs` (when running)
- **User Guide**: `docs/USER_GUIDE.md`
- **Admin Guide**: `docs/ADMIN_GUIDE.md`
- **Security Guide**: `docs/SECURITY.md`

The enhanced FM-LLM Solver now provides a complete user account system with comprehensive tracking, analytics, and management capabilities while maintaining the high-quality mathematical verification that made it successful.

**🎉 Your users now have a professional, secure, and feature-rich experience for barrier certificate generation and verification!** 