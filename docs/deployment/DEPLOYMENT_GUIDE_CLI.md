# FM-LLM-Solver Enhanced User System - CLI Deployment Guide

**Target Environment:** Production Linux Server  
**Deployment Method:** Docker Compose + CLI Tools  
**Database:** PostgreSQL with Enhanced User Schema  
**Monitoring:** Prometheus + Grafana (Optional)  

## 📋 Prerequisites

### System Requirements
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y \
    docker.io docker-compose \
    postgresql-client \
    redis-tools \
    curl wget git \
    python3 python3-pip \
    nginx certbot

# CentOS/RHEL
sudo yum install -y \
    docker docker-compose \
    postgresql \
    redis \
    curl wget git \
    python3 python3-pip \
    nginx certbot

# Start Docker
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER
```

### Clone and Prepare Repository
```bash
# Clone the repository
git clone <your-repo-url> fm-llm-solver
cd fm-llm-solver

# Verify enhanced files exist
ls -la sql/init.sql
ls -la docker-compose.hybrid.yml
ls -la web_interface/templates/auth/profile.html
echo "✅ Enhanced user system files verified"
```

## 🔧 Step 1: Environment Configuration

### Create Production Environment File
```bash
# Create production environment configuration
cat > .env.production << 'EOF'
# Flask Configuration
FLASK_ENV=production
FLASK_SECRET_KEY=$(openssl rand -hex 32)
FLASK_DEBUG=False

# Database Configuration
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=fm_llm_solver
POSTGRES_USER=fm_user
POSTGRES_PASSWORD=$(openssl rand -base64 32)
DATABASE_URL=postgresql://fm_user:${POSTGRES_PASSWORD}@postgres:5432/fm_llm_solver

# Redis Configuration
REDIS_URL=redis://redis:6379/0
REDIS_PASSWORD=$(openssl rand -base64 24)

# User Account System Features
USER_ACCOUNT_FEATURES=true
ENABLE_SUBSCRIPTIONS=true
ENABLE_API_KEYS=true
ENABLE_USER_ANALYTICS=true
ENABLE_ADMIN_PANEL=true

# Email Configuration (Replace with your SMTP settings)
EMAIL_ENABLED=true
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USE_TLS=true
SMTP_USERNAME=your-email@domain.com
SMTP_PASSWORD=your-app-password
DEFAULT_FROM_EMAIL=noreply@yourdomain.com

# Security Settings
JWT_SECRET_KEY=$(openssl rand -hex 32)
ENCRYPTION_KEY=$(openssl rand -hex 32)
SESSION_TIMEOUT=3600
MAX_LOGIN_ATTEMPTS=5
RATE_LIMIT_ENABLED=true

# File Upload Settings
MAX_UPLOAD_SIZE=10485760
UPLOAD_FOLDER=/app/uploads
ALLOWED_EXTENSIONS=pdf,txt,json

# Monitoring (Optional)
ENABLE_PROMETHEUS=true
ENABLE_GRAFANA=true
GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 16)

# Application Settings
DOMAIN_NAME=yourdomain.com
HTTPS_ENABLED=true
EOF

# Load environment variables
export $(cat .env.production | grep -v '^#' | xargs)
echo "✅ Environment configuration created"
```

### Generate SSL Certificates (Production)
```bash
# Option 1: Let's Encrypt (Recommended for production)
sudo certbot certonly --standalone \
    -d yourdomain.com \
    -d www.yourdomain.com \
    --email your-email@domain.com \
    --agree-tos \
    --non-interactive

# Option 2: Self-signed (Development/Testing)
mkdir -p ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout ssl/privkey.pem \
    -out ssl/fullchain.pem \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=yourdomain.com"

echo "✅ SSL certificates configured"
```

## 🗄️ Step 2: Database Setup

### Initialize PostgreSQL with Enhanced Schema
```bash
# Start PostgreSQL temporarily for setup
docker run --name temp-postgres -d \
    -e POSTGRES_DB=$POSTGRES_DB \
    -e POSTGRES_USER=$POSTGRES_USER \
    -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
    -p 5432:5432 \
    postgres:13

# Wait for PostgreSQL to start
sleep 10

# Initialize database with enhanced schema
PGPASSWORD=$POSTGRES_PASSWORD psql \
    -h localhost \
    -U $POSTGRES_USER \
    -d $POSTGRES_DB \
    -f sql/init.sql

# Verify database setup
PGPASSWORD=$POSTGRES_PASSWORD psql \
    -h localhost \
    -U $POSTGRES_USER \
    -d $POSTGRES_DB \
    -c "\dt" \
    -c "SELECT username, email, role FROM users WHERE role = 'admin';"

# Stop temporary container
docker stop temp-postgres
docker rm temp-postgres

echo "✅ Database initialized with enhanced user schema"
```

### Create Database Backup Script
```bash
# Create automated backup script
cat > scripts/backup-db.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/var/backups/fm-llm-solver"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR

# Database backup
docker exec fm-llm-solver-postgres-1 pg_dump \
    -U $POSTGRES_USER \
    -d $POSTGRES_DB \
    --clean --if-exists \
    > $BACKUP_DIR/fm_llm_solver_$DATE.sql

# Compress and clean old backups
gzip $BACKUP_DIR/fm_llm_solver_$DATE.sql
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete

echo "Database backup completed: fm_llm_solver_$DATE.sql.gz"
EOF

chmod +x scripts/backup-db.sh
echo "✅ Database backup script created"
```

## 🐋 Step 3: Docker Deployment

### Prepare Docker Environment
```bash
# Create required directories
mkdir -p {uploads,logs,instance,ssl,prometheus,grafana}

# Set proper permissions
sudo chown -R 1000:1000 uploads logs instance
sudo chmod 755 uploads logs instance

# Copy SSL certificates to ssl directory
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ssl/
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ssl/
sudo chown 1000:1000 ssl/*

echo "✅ Docker environment prepared"
```

### Update Docker Compose Configuration
```bash
# Update docker-compose.hybrid.yml with production values
envsubst < docker-compose.hybrid.yml > docker-compose.production.yml

# Verify configuration
docker-compose -f docker-compose.production.yml config

echo "✅ Docker Compose configuration updated"
```

### Deploy Application Stack
```bash
# Pull latest images
docker-compose -f docker-compose.production.yml pull

# Start services in production mode
docker-compose -f docker-compose.production.yml up -d

# Wait for services to start
sleep 30

# Verify all services are running
docker-compose -f docker-compose.production.yml ps

echo "✅ Application stack deployed"
```

## 🔍 Step 4: Verification and Health Checks

### Application Health Check
```bash
# Create health check script
cat > scripts/health-check.sh << 'EOF'
#!/bin/bash
echo "🔍 FM-LLM-Solver Health Check"
echo "================================"

# Check web service
if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ Web service: HEALTHY"
else
    echo "❌ Web service: FAILED"
fi

# Check database connection
if docker exec fm-llm-solver-postgres-1 pg_isready -U $POSTGRES_USER > /dev/null 2>&1; then
    echo "✅ Database: HEALTHY"
else
    echo "❌ Database: FAILED"
fi

# Check Redis
if docker exec fm-llm-solver-redis-1 redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis: HEALTHY"
else
    echo "❌ Redis: FAILED"
fi

# Check user account system
RESPONSE=$(curl -s http://localhost:8080/auth/login)
if echo "$RESPONSE" | grep -q "login" > /dev/null 2>&1; then
    echo "✅ User account system: HEALTHY"
else
    echo "❌ User account system: FAILED"
fi

# Check admin user
USER_COUNT=$(docker exec fm-llm-solver-postgres-1 psql -U $POSTGRES_USER -d $POSTGRES_DB -t -c "SELECT COUNT(*) FROM users WHERE role = 'admin';" | tr -d ' ')
if [ "$USER_COUNT" -gt 0 ]; then
    echo "✅ Admin user: EXISTS ($USER_COUNT)"
else
    echo "❌ Admin user: MISSING"
fi

echo "================================"
EOF

chmod +x scripts/health-check.sh
./scripts/health-check.sh
```

### Enhanced User System Verification
```bash
# Test enhanced user features
cat > scripts/test-user-system.sh << 'EOF'
#!/bin/bash
echo "🧪 Testing Enhanced User Account System"
echo "======================================="

# Test database schema
echo "📊 Database Schema Check:"
docker exec fm-llm-solver-postgres-1 psql -U $POSTGRES_USER -d $POSTGRES_DB -c "
SELECT 
    tablename,
    schemaname
FROM pg_tables 
WHERE schemaname = 'public' 
    AND tablename IN ('users', 'user_activities', 'user_sessions', 'certificate_favorites', 'query_logs')
ORDER BY tablename;
"

# Test admin user capabilities
echo "👑 Admin User Check:"
docker exec fm-llm-solver-postgres-1 psql -U $POSTGRES_USER -d $POSTGRES_DB -c "
SELECT 
    username,
    email,
    role,
    subscription_type,
    is_active,
    created_at
FROM users 
WHERE role = 'admin';
"

# Test enhanced user fields
echo "🔍 Enhanced User Fields Check:"
docker exec fm-llm-solver-postgres-1 psql -U $POSTGRES_USER -d $POSTGRES_DB -c "
SELECT 
    COUNT(*) as total_users,
    COUNT(CASE WHEN subscription_type IS NOT NULL THEN 1 END) as users_with_subscription,
    COUNT(CASE WHEN api_key IS NOT NULL THEN 1 END) as users_with_api_key
FROM users;
"

# Test model configurations
echo "🤖 Model Configuration Check:"
docker exec fm-llm-solver-postgres-1 psql -U $POSTGRES_USER -d $POSTGRES_DB -c "
SELECT name, base_model_name, is_active 
FROM model_configuration 
WHERE is_active = true;
"

echo "======================================="
EOF

chmod +x scripts/test-user-system.sh
./scripts/test-user-system.sh
```

## 🌐 Step 5: Reverse Proxy Setup (Nginx)

### Configure Nginx
```bash
# Create Nginx configuration
sudo tee /etc/nginx/sites-available/fm-llm-solver << 'EOF'
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;

    client_max_body_size 10M;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_redirect off;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Static files
    location /static/ {
        proxy_pass http://localhost:8080;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Health check endpoint
    location /health {
        proxy_pass http://localhost:8080;
        access_log off;
    }
}
EOF

# Enable site and restart Nginx
sudo ln -sf /etc/nginx/sites-available/fm-llm-solver /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

echo "✅ Nginx reverse proxy configured"
```

## 📊 Step 6: Monitoring Setup (Optional)

### Configure Prometheus
```bash
# Create Prometheus configuration
mkdir -p prometheus
cat > prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'fm-llm-solver'
    static_configs:
      - targets: ['web:8080']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
EOF

# Create Grafana datasources
mkdir -p grafana/provisioning/{datasources,dashboards}
cat > grafana/provisioning/datasources/prometheus.yml << 'EOF'
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

echo "✅ Monitoring configuration created"
```

### Install Node Exporter (System Metrics)
```bash
# Download and install node_exporter
cd /tmp
wget https://github.com/prometheus/node_exporter/releases/latest/download/node_exporter-*linux-amd64.tar.gz
tar xzf node_exporter-*linux-amd64.tar.gz
sudo mv node_exporter-*linux-amd64/node_exporter /usr/local/bin/

# Create systemd service
sudo tee /etc/systemd/system/node_exporter.service << 'EOF'
[Unit]
Description=Node Exporter
After=network.target

[Service]
User=nobody
Group=nobody
Type=simple
ExecStart=/usr/local/bin/node_exporter

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable node_exporter
sudo systemctl start node_exporter

echo "✅ Node Exporter installed and started"
```

## 🔄 Step 7: Automation and Maintenance

### Create Management Scripts
```bash
# Create deployment management script
cat > scripts/manage.sh << 'EOF'
#!/bin/bash
set -e

COMPOSE_FILE="docker-compose.production.yml"

case "$1" in
    start)
        echo "🚀 Starting FM-LLM-Solver..."
        docker-compose -f $COMPOSE_FILE up -d
        sleep 10
        ./scripts/health-check.sh
        ;;
    stop)
        echo "🛑 Stopping FM-LLM-Solver..."
        docker-compose -f $COMPOSE_FILE down
        ;;
    restart)
        echo "🔄 Restarting FM-LLM-Solver..."
        docker-compose -f $COMPOSE_FILE restart
        sleep 10
        ./scripts/health-check.sh
        ;;
    update)
        echo "📦 Updating FM-LLM-Solver..."
        git pull
        docker-compose -f $COMPOSE_FILE pull
        docker-compose -f $COMPOSE_FILE up -d
        sleep 10
        ./scripts/health-check.sh
        ;;
    backup)
        echo "💾 Creating backup..."
        ./scripts/backup-db.sh
        ;;
    logs)
        docker-compose -f $COMPOSE_FILE logs -f
        ;;
    status)
        docker-compose -f $COMPOSE_FILE ps
        ./scripts/health-check.sh
        ;;
    test)
        ./scripts/test-user-system.sh
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|update|backup|logs|status|test}"
        exit 1
        ;;
esac
EOF

chmod +x scripts/manage.sh
echo "✅ Management scripts created"
```

### Setup Automated Backups
```bash
# Add cron job for daily backups
(crontab -l 2>/dev/null; echo "0 2 * * * $(pwd)/scripts/backup-db.sh") | crontab -

# Add weekly health checks
(crontab -l 2>/dev/null; echo "0 8 * * 1 $(pwd)/scripts/health-check.sh | mail -s 'FM-LLM-Solver Health Report' admin@yourdomain.com") | crontab -

echo "✅ Automated maintenance configured"
```

## 🎯 Step 8: Final Verification

### Complete System Test
```bash
# Run comprehensive system test
echo "🎯 Final System Verification"
echo "============================="

# 1. Check all services
./scripts/manage.sh status

# 2. Test enhanced user system
./scripts/test-user-system.sh

# 3. Test web interface
echo "🌐 Testing web interface..."
curl -I https://yourdomain.com/
curl -I https://yourdomain.com/auth/login

# 4. Test API endpoints
echo "🔌 Testing API endpoints..."
curl -H "Content-Type: application/json" \
     -X GET https://yourdomain.com/api/health

# 5. Check SSL certificate
echo "🔒 SSL Certificate Check:"
openssl s_client -connect yourdomain.com:443 -servername yourdomain.com < /dev/null 2>/dev/null | openssl x509 -noout -dates

echo "✅ System verification complete!"
```

### Create Admin User via CLI
```bash
# Create first admin user via database
cat > scripts/create-admin.sh << 'EOF'
#!/bin/bash
read -p "Admin Username: " ADMIN_USER
read -p "Admin Email: " ADMIN_EMAIL
read -s -p "Admin Password: " ADMIN_PASS
echo

# Generate password hash using Python
HASH=$(python3 -c "
from werkzeug.security import generate_password_hash
print(generate_password_hash('$ADMIN_PASS'))
")

# Insert admin user
docker exec fm-llm-solver-postgres-1 psql -U $POSTGRES_USER -d $POSTGRES_DB -c "
INSERT INTO users (
    username, email, password_hash, role, 
    subscription_type, is_active, is_verified,
    daily_request_limit, monthly_request_limit,
    created_at
) VALUES (
    '$ADMIN_USER', '$ADMIN_EMAIL', '$HASH', 'admin',
    'enterprise', true, true,
    10000, 100000,
    NOW()
) ON CONFLICT (email) DO NOTHING;
"

echo "✅ Admin user created: $ADMIN_USER"
EOF

chmod +x scripts/create-admin.sh
./scripts/create-admin.sh
```

## 📚 Post-Deployment Checklist

```bash
# Final checklist
cat > DEPLOYMENT_CHECKLIST.md << 'EOF'
# FM-LLM-Solver Deployment Checklist

## ✅ Pre-Deployment
- [ ] Server meets system requirements
- [ ] Docker and Docker Compose installed
- [ ] SSL certificates configured
- [ ] Environment variables set
- [ ] Database initialized

## ✅ Deployment
- [ ] All services started successfully
- [ ] Health checks passing
- [ ] Enhanced user system verified
- [ ] Nginx reverse proxy configured
- [ ] Admin user created

## ✅ Post-Deployment
- [ ] SSL certificate working (https://)
- [ ] User registration working
- [ ] Certificate generation working
- [ ] User profiles accessible
- [ ] Admin panel accessible
- [ ] Monitoring dashboards accessible (if enabled)
- [ ] Automated backups scheduled
- [ ] Log rotation configured

## ✅ Security
- [ ] Firewall configured (ports 80, 443, 22 only)
- [ ] Database not accessible externally
- [ ] Strong passwords set for all services
- [ ] SSL/TLS properly configured
- [ ] Security headers enabled

## 📞 Support
- Health Check: `./scripts/manage.sh status`
- View Logs: `./scripts/manage.sh logs`
- Backup Database: `./scripts/manage.sh backup`
- Restart Services: `./scripts/manage.sh restart`
EOF

echo "📋 Deployment checklist created"
```

## 🎉 Deployment Complete!

Your FM-LLM-Solver with enhanced user account system is now deployed and ready for production use!

### Quick Commands Reference:
```bash
# Start system
./scripts/manage.sh start

# Check status
./scripts/manage.sh status

# Test user system
./scripts/test-user-system.sh

# View logs
./scripts/manage.sh logs

# Update system
./scripts/manage.sh update

# Create backup
./scripts/manage.sh backup
```

### Access Points:
- **Web Interface**: https://yourdomain.com
- **Admin Login**: https://yourdomain.com/auth/login
- **User Profiles**: https://yourdomain.com/auth/profile
- **API Health**: https://yourdomain.com/api/health
- **Grafana** (if enabled): https://yourdomain.com:3000
- **Prometheus** (if enabled): https://yourdomain.com:9090

The enhanced user account system is now live with complete certificate generation tracking linked to users! 