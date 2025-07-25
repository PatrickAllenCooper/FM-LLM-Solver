# 🚀 Quick Deployment: FM-LLM-Solver Enhanced User System

**Transform your FM-LLM-Solver into a production-ready SaaS platform in under 15 minutes!**

## ⚡ One-Command Deployment

```bash
# Run this single command to deploy everything:
./scripts/quick-deploy.sh
```

**That's it!** The script will:
- ✅ Configure your production environment
- ✅ Set up SSL certificates (Let's Encrypt or self-signed)
- ✅ Initialize PostgreSQL database with enhanced user schema
- ✅ Deploy Docker containers
- ✅ Create management scripts
- ✅ Run health checks
- ✅ Set up monitoring (optional)
- ✅ Create your first admin user

## 📋 Prerequisites (Install First)

### Ubuntu/Debian
```bash
sudo apt update && sudo apt install -y \
    docker.io docker-compose \
    postgresql-client \
    nginx certbot \
    curl wget git python3

sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER
```

### CentOS/RHEL
```bash
sudo yum install -y \
    docker docker-compose \
    postgresql redis \
    nginx certbot \
    curl wget git python3

sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER
```

## 🎯 What You Get

### 🔐 Enhanced User Account System
- **Professional user profiles** with 50+ fields
- **Subscription management** (free/pro/enterprise tiers)
- **API key management** for integrations
- **Admin dashboard** for user oversight
- **Security features** (2FA ready, rate limiting)

### 📊 Certificate Generation Tracking
- **Every certificate linked to its creator** ✅ **(Your main requirement!)**
- **Complete generation history** with metadata
- **User interaction tracking** (ratings, favorites)
- **Quality metrics** and verification results
- **Data export** capabilities

### 🏗️ Production Infrastructure
- **Docker containerization** for easy deployment
- **PostgreSQL database** with comprehensive schema
- **SSL encryption** with automatic certificate renewal
- **Nginx reverse proxy** with security headers
- **Redis caching** for performance
- **Monitoring dashboards** (Prometheus + Grafana)

## 📝 During Deployment You'll Be Asked For:

1. **Domain name** (e.g., `yourdomain.com`)
2. **Admin email** for SSL certificates and notifications
3. **SMTP settings** for user email notifications
4. **Monitoring preference** (enable Grafana dashboards?)

The script generates all passwords and security keys automatically.

## 🌐 After Deployment

### Access Your Platform
- **Main Site:** `https://yourdomain.com`
- **User Registration:** `https://yourdomain.com/auth/register`
- **Admin Login:** `https://yourdomain.com/auth/login`
- **User Profiles:** `https://yourdomain.com/auth/profile`

### Management Commands
```bash
./scripts/manage.sh start      # Start all services
./scripts/manage.sh status     # Check system health
./scripts/manage.sh logs       # View application logs
./scripts/create-admin.sh      # Create additional admin users
```

## 🎉 Success Verification

After deployment, test that **certificate generation tracking** works:

1. **Register a new user** at your domain
2. **Generate a certificate** through the web interface
3. **Check the user's profile** - you should see the certificate in their history
4. **Admin panel** should show all user activity and certificate generations

**🎯 Mission Complete:** Every certificate is now tracked with full user attribution!

## 📚 Documentation

- **`DEPLOYMENT_GUIDE_CLI.md`** - Comprehensive step-by-step manual deployment
- **`DEPLOYMENT_PACKAGE_SUMMARY.md`** - Complete feature overview and architecture
- **`ENHANCED_USER_SYSTEM_TEST_REPORT.md`** - Testing results and verification
- **`USER_ACCOUNT_DEPLOYMENT_GUIDE.md`** - Original detailed deployment guide

## 🔧 Alternative: Manual Deployment

If you prefer to understand each step:

```bash
# Follow the detailed guide
cat DEPLOYMENT_GUIDE_CLI.md
```

## 🆘 Need Help?

### Common Issues
```bash
# Check service status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs web

# Restart services
./scripts/manage.sh restart

# Test database connection
docker exec fm-llm-solver-postgres-1 pg_isready
```

### Log Locations
- Application: `logs/` directory
- Nginx: `/var/log/nginx/`
- Database: `docker logs fm-llm-solver-postgres-1`

---

**Ready to deploy? Run: `./scripts/quick-deploy.sh`** 