#!/bin/bash
set -e

echo "🚀 FM-LLM-Solver Enhanced User System - Quick Deploy"
echo "=================================================="

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "❌ This script should not be run as root for security reasons"
   echo "   Please run as a regular user with sudo privileges"
   exit 1
fi

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check PostgreSQL client
if ! command -v psql &> /dev/null; then
    echo "❌ PostgreSQL client is not installed. Please install postgresql-client first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Get deployment configuration
echo ""
echo "📝 Deployment Configuration"
echo "==========================="

read -p "Domain name (e.g., yourdomain.com): " DOMAIN_NAME
read -p "Admin email: " ADMIN_EMAIL
read -p "SMTP host (e.g., smtp.gmail.com): " SMTP_HOST
read -p "SMTP username: " SMTP_USERNAME
read -s -p "SMTP password: " SMTP_PASSWORD
echo ""
read -p "Enable monitoring (Prometheus/Grafana)? (y/n): " ENABLE_MONITORING

# Generate secure passwords and keys
echo "🔐 Generating secure configuration..."

FLASK_SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET_KEY=$(openssl rand -hex 32)
ENCRYPTION_KEY=$(openssl rand -hex 32)
POSTGRES_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 24)
GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 16)

# Create environment file
echo "📄 Creating production environment..."

cat > .env.production << EOF
# Flask Configuration
FLASK_ENV=production
FLASK_SECRET_KEY=$FLASK_SECRET_KEY
FLASK_DEBUG=False

# Database Configuration
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=fm_llm_solver
POSTGRES_USER=fm_user
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
DATABASE_URL=postgresql://fm_user:$POSTGRES_PASSWORD@postgres:5432/fm_llm_solver

# Redis Configuration
REDIS_URL=redis://redis:6379/0
REDIS_PASSWORD=$REDIS_PASSWORD

# User Account System Features
USER_ACCOUNT_FEATURES=true
ENABLE_SUBSCRIPTIONS=true
ENABLE_API_KEYS=true
ENABLE_USER_ANALYTICS=true
ENABLE_ADMIN_PANEL=true

# Email Configuration
EMAIL_ENABLED=true
SMTP_HOST=$SMTP_HOST
SMTP_PORT=587
SMTP_USE_TLS=true
SMTP_USERNAME=$SMTP_USERNAME
SMTP_PASSWORD=$SMTP_PASSWORD
DEFAULT_FROM_EMAIL=noreply@$DOMAIN_NAME

# Security Settings
JWT_SECRET_KEY=$JWT_SECRET_KEY
ENCRYPTION_KEY=$ENCRYPTION_KEY
SESSION_TIMEOUT=3600
MAX_LOGIN_ATTEMPTS=5
RATE_LIMIT_ENABLED=true

# File Upload Settings
MAX_UPLOAD_SIZE=10485760
UPLOAD_FOLDER=/app/uploads
ALLOWED_EXTENSIONS=pdf,txt,json

# Monitoring
ENABLE_PROMETHEUS=$([[ "$ENABLE_MONITORING" == "y" ]] && echo "true" || echo "false")
ENABLE_GRAFANA=$([[ "$ENABLE_MONITORING" == "y" ]] && echo "true" || echo "false")
GRAFANA_ADMIN_PASSWORD=$GRAFANA_ADMIN_PASSWORD

# Application Settings
DOMAIN_NAME=$DOMAIN_NAME
HTTPS_ENABLED=true
EOF

# Load environment variables
export $(cat .env.production | grep -v '^#' | xargs)
echo "✅ Environment configuration created"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p {uploads,logs,instance,ssl,prometheus,grafana,scripts}
mkdir -p grafana/provisioning/{datasources,dashboards}

# Set proper permissions
sudo chown -R 1000:1000 uploads logs instance 2>/dev/null || true
chmod 755 uploads logs instance

echo "✅ Directories created"

# Generate SSL certificates
echo "🔒 Setting up SSL certificates..."

if [[ "$DOMAIN_NAME" != "localhost" ]] && [[ "$DOMAIN_NAME" != "127.0.0.1" ]]; then
    echo "Would you like to use Let's Encrypt for SSL certificates? (y/n)"
    read -p "This requires the domain to point to this server: " USE_LETSENCRYPT
    
    if [[ "$USE_LETSENCRYPT" == "y" ]]; then
        sudo certbot certonly --standalone \
            -d $DOMAIN_NAME \
            --email $ADMIN_EMAIL \
            --agree-tos \
            --non-interactive || {
                echo "⚠️  Let's Encrypt failed, falling back to self-signed certificates"
                USE_LETSENCRYPT="n"
            }
    fi
else
    USE_LETSENCRYPT="n"
fi

if [[ "$USE_LETSENCRYPT" != "y" ]]; then
    echo "📝 Generating self-signed SSL certificates..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout ssl/privkey.pem \
        -out ssl/fullchain.pem \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=$DOMAIN_NAME"
fi

echo "✅ SSL certificates configured"

# Initialize database
echo "🗄️  Setting up database..."

# Start temporary PostgreSQL for initialization
docker run --name temp-postgres -d \
    -e POSTGRES_DB=$POSTGRES_DB \
    -e POSTGRES_USER=$POSTGRES_USER \
    -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
    -p 5432:5432 \
    postgres:13

echo "⏳ Waiting for PostgreSQL to start..."
sleep 15

# Check if database is ready
max_attempts=30
attempt=1
while ! PGPASSWORD=$POSTGRES_PASSWORD psql -h localhost -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT 1;" &>/dev/null; do
    if [ $attempt -eq $max_attempts ]; then
        echo "❌ PostgreSQL failed to start after $max_attempts attempts"
        docker stop temp-postgres
        docker rm temp-postgres
        exit 1
    fi
    echo "   Attempt $attempt/$max_attempts - waiting for PostgreSQL..."
    sleep 2
    ((attempt++))
done

# Initialize database with enhanced schema
echo "📊 Initializing database schema..."
PGPASSWORD=$POSTGRES_PASSWORD psql \
    -h localhost \
    -U $POSTGRES_USER \
    -d $POSTGRES_DB \
    -f sql/init.sql

# Verify database setup
echo "✅ Verifying database setup..."
PGPASSWORD=$POSTGRES_PASSWORD psql \
    -h localhost \
    -U $POSTGRES_USER \
    -d $POSTGRES_DB \
    -c "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename;" \
    -c "SELECT username, email, role FROM users WHERE role = 'admin';"

# Stop temporary container
docker stop temp-postgres
docker rm temp-postgres

echo "✅ Database initialized"

# Create monitoring configuration if enabled
if [[ "$ENABLE_MONITORING" == "y" ]]; then
    echo "📊 Setting up monitoring configuration..."
    
    cat > prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

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
EOF

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
fi

# Create Docker Compose production file
echo "🐋 Preparing Docker Compose configuration..."
envsubst < docker-compose.hybrid.yml > docker-compose.production.yml

# Validate Docker Compose configuration
docker-compose -f docker-compose.production.yml config > /dev/null
echo "✅ Docker Compose configuration validated"

# Create management scripts
echo "🛠️  Creating management scripts..."

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

echo "================================"
EOF

cat > scripts/manage.sh << 'EOF'
#!/bin/bash
set -e

COMPOSE_FILE="docker-compose.production.yml"

case "$1" in
    start)
        echo "🚀 Starting FM-LLM-Solver..."
        docker-compose -f $COMPOSE_FILE up -d
        sleep 15
        ./scripts/health-check.sh
        ;;
    stop)
        echo "🛑 Stopping FM-LLM-Solver..."
        docker-compose -f $COMPOSE_FILE down
        ;;
    restart)
        echo "🔄 Restarting FM-LLM-Solver..."
        docker-compose -f $COMPOSE_FILE restart
        sleep 15
        ./scripts/health-check.sh
        ;;
    status)
        docker-compose -f $COMPOSE_FILE ps
        ./scripts/health-check.sh
        ;;
    logs)
        docker-compose -f $COMPOSE_FILE logs -f
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
EOF

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

chmod +x scripts/*.sh
echo "✅ Management scripts created"

# Deploy the application
echo "🚀 Deploying application stack..."

# Pull latest images
docker-compose -f docker-compose.production.yml pull

# Start services
docker-compose -f docker-compose.production.yml up -d

echo "⏳ Waiting for services to start..."
sleep 30

# Verify deployment
echo "🔍 Verifying deployment..."
docker-compose -f docker-compose.production.yml ps

# Run health check
./scripts/health-check.sh

# Display summary
echo ""
echo "🎉 Deployment Complete!"
echo "======================="
echo ""
echo "🌐 Application URL: https://$DOMAIN_NAME"
echo "👤 Admin Login: https://$DOMAIN_NAME/auth/login"
echo "📊 User Profiles: https://$DOMAIN_NAME/auth/profile"
echo ""

if [[ "$ENABLE_MONITORING" == "y" ]]; then
    echo "📈 Monitoring:"
    echo "   Grafana: https://$DOMAIN_NAME:3000 (admin / $GRAFANA_ADMIN_PASSWORD)"
    echo "   Prometheus: https://$DOMAIN_NAME:9090"
    echo ""
fi

echo "🛠️  Management Commands:"
echo "   Start: ./scripts/manage.sh start"
echo "   Stop: ./scripts/manage.sh stop"
echo "   Status: ./scripts/manage.sh status"
echo "   Logs: ./scripts/manage.sh logs"
echo "   Create Admin: ./scripts/create-admin.sh"
echo ""

echo "📝 Important Files:"
echo "   Environment: .env.production"
echo "   SSL Certificates: ssl/"
echo "   Logs: logs/"
echo "   Uploads: uploads/"
echo ""

echo "🔐 Security Information:"
echo "   Database Password: $POSTGRES_PASSWORD"
echo "   Redis Password: $REDIS_PASSWORD"
if [[ "$ENABLE_MONITORING" == "y" ]]; then
    echo "   Grafana Admin Password: $GRAFANA_ADMIN_PASSWORD"
fi
echo ""
echo "⚠️  Save these passwords in a secure location!"
echo ""

# Prompt to create admin user
echo "Would you like to create an admin user now? (y/n)"
read -p "This is recommended for initial setup: " CREATE_ADMIN

if [[ "$CREATE_ADMIN" == "y" ]]; then
    ./scripts/create-admin.sh
fi

echo ""
echo "✅ FM-LLM-Solver with Enhanced User Account System is now deployed!"
echo "🔗 Visit https://$DOMAIN_NAME to get started" 