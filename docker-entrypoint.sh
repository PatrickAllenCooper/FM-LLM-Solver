#!/bin/bash
set -e

echo "🚀 Starting FM-LLM Solver Production Web Interface"

# Wait for database
echo "⏳ Waiting for database..."
while ! nc -z ${DB_HOST:-postgres-service} ${DB_PORT:-5432}; do
  echo "PostgreSQL is not ready yet..."
  sleep 2
done
echo "✅ Database is ready"

# Wait for Redis
echo "⏳ Waiting for Redis..."
while ! nc -z ${REDIS_HOST:-redis-service} ${REDIS_PORT:-6379}; do
  echo "Redis is not ready yet..."
  sleep 2
done
echo "✅ Redis is ready"

# Initialize database
echo "🔧 Initializing database..."
python -c "
import sys
sys.path.insert(0, '/app')

from web_interface.app import create_app
from web_interface.models import db

print('Creating Flask app...')
app = create_app()

print('Initializing database...')
with app.app_context():
    db.create_all()
    print('✅ Database tables created')
    
    # Create admin user if none exists
    from web_interface.models import User
    if not User.query.filter_by(role='admin').first():
        admin = User(
            username='admin',
            email='admin@fmgen.net',
            role='admin',
            is_active=True,
            is_verified=True,
            subscription_type='enterprise'
        )
        admin.set_password('admin123')  # Change this in production!
        db.session.add(admin)
        db.session.commit()
        print('✅ Admin user created (username: admin, password: admin123)')
"

# Start application
echo "🎯 Starting web application with ${WORKERS:-4} workers..."
exec gunicorn \
    --bind 0.0.0.0:5000 \
    --workers ${WORKERS:-4} \
    --timeout ${TIMEOUT:-300} \
    --keep-alive 2 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --preload \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    "web_interface.app:create_app()" 