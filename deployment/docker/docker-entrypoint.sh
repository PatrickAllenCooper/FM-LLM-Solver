#!/bin/bash
set -e

# FM-LLM Solver Docker Entrypoint
# Handles different deployment modes: web, inference, both, dev

MODE=${1:-web}

# Function to initialize the database
init_database() {
    echo "🗄️  Initializing database..."
    python -c "
try:
    from web_interface.app import create_app
    from web_interface.models import db
    app = create_app()
    with app.app_context():
        db.create_all()
    print('✅ Database initialized')
except Exception as e:
    print(f'⚠️  Database init warning: {e}')
"
}

# Function to wait for services
wait_for_services() {
    echo "⏳ Waiting for required services..."

    # Wait for database if configured
    if [ -n "$DATABASE_URL" ]; then
        echo "Waiting for database..."
        while ! python -c "
import psycopg2
import os
try:
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    conn.close()
    print('✅ Database ready')
except:
    exit(1)
" 2>/dev/null; do
            echo "Database not ready, waiting..."
  sleep 2
done
    fi

    # Wait for Redis if configured
    if [ -n "$REDIS_URL" ]; then
        echo "Waiting for Redis..."
        while ! python -c "
import redis
import os
try:
    r = redis.from_url(os.environ['REDIS_URL'])
    r.ping()
    print('✅ Redis ready')
except:
    exit(1)
" 2>/dev/null; do
            echo "Redis not ready, waiting..."
  sleep 2
done
    fi
}

# Function to start web interface
start_web() {
    echo "🌐 Starting FM-LLM Solver Web Interface..."

# Initialize database
    init_database
    
    # Start with gunicorn for production
    exec gunicorn \
        --bind 0.0.0.0:5000 \
        --workers 4 \
        --worker-class gevent \
        --worker-connections 1000 \
        --timeout 120 \
        --keepalive 5 \
        --preload \
        --access-logfile - \
        --error-logfile - \
        --log-level info \
        --worker-tmp-dir /dev/shm \
        "web_interface.app:create_app()"
}

# Function to start inference API
start_inference() {
    echo "🤖 Starting FM-LLM Solver Inference API..."
    
    # Start with uvicorn
    exec uvicorn \
        inference_api.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 1 \
        --timeout-keep-alive 120 \
        --access-log \
        --log-level info
}

# Function to start both services
start_both() {
    echo "🚀 Starting FM-LLM Solver Full Stack..."
    
    # Initialize database
    init_database
    
    # Start inference API in background
    echo "Starting inference API..."
    uvicorn inference_api.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 1 \
        --timeout-keep-alive 120 \
        --access-log \
        --log-level info &
    
    # Wait a moment for inference to start
    sleep 5
    
    # Start web interface in foreground
    echo "Starting web interface..."
exec gunicorn \
    --bind 0.0.0.0:5000 \
        --workers 4 \
        --worker-class gevent \
        --worker-connections 1000 \
        --timeout 120 \
        --keepalive 5 \
    --preload \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
        --worker-tmp-dir /dev/shm \
    "web_interface.app:create_app()" 
}

# Function to start development mode
start_dev() {
    echo "🛠️  Starting FM-LLM Solver Development Mode..."
    
    # Initialize database
    init_database
    
    if [ "$#" -gt 1 ]; then
        case "$2" in
            "web")
                echo "Starting web interface in debug mode..."
                exec python run_application.py web --debug --host 0.0.0.0
                ;;
            "inference")
                echo "Starting inference API in debug mode..."
                exec uvicorn inference_api.main:app --host 0.0.0.0 --port 8000 --reload
                ;;
            "jupyter")
                echo "Starting Jupyter notebook..."
                exec jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
                ;;
            "test")
                echo "Running tests..."
                exec pytest "${@:3}"
                ;;
            "shell")
                echo "Starting interactive shell..."
                exec python
                ;;
            *)
                echo "Running custom command: ${@:2}"
                exec "${@:2}"
                ;;
        esac
    else
        echo "Starting development shell..."
        exec /bin/bash
    fi
}

# Wait for external services
wait_for_services

# Route to appropriate mode
case "$MODE" in
    "web")
        start_web
        ;;
    "inference")
        start_inference
        ;;
    "both")
        start_both
        ;;
    "dev")
        start_dev "$@"
        ;;
    *)
        echo "❌ Unknown mode: $MODE"
        echo "Available modes: web, inference, both, dev"
        echo "Usage: docker run fm-llm-solver [web|inference|both|dev] [dev-options]"
        exit 1
        ;;
esac 