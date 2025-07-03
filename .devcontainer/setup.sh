#!/bin/bash

# Development environment setup script for FM-LLM-Solver
set -e

echo "🚀 Setting up FM-LLM-Solver development environment..."

# Install additional development tools
echo "📦 Installing additional development tools..."
pip install --upgrade pip
pip install -e .

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install

# Set up git configuration for the container
echo "🔧 Configuring git..."
git config --global --add safe.directory /workspace
git config --global init.defaultBranch main

# Create development directories
echo "📁 Creating development directories..."
mkdir -p logs
mkdir -p data/temp
mkdir -p output/dev
mkdir -p kb_data
mkdir -p instance
mkdir -p .pytest_cache

# Install VS Code extensions helper
echo "🔌 Setting up VS Code extensions..."
code --install-extension ms-python.python --force || true
code --install-extension ms-python.black-formatter --force || true
code --install-extension ms-python.flake8 --force || true

# Create development configuration
echo "⚙️ Creating development configuration..."
cat > config/dev.yaml << 'DEVCONFIG'
# Development-specific configuration
environment: development

# Development database
database:
  primary:
    host: "postgres"
    port: 5432
    database: "fm_llm_dev"
    username: "dev_user"
    password: "dev_password"
    pool_size: 5
    max_overflow: 5
    echo: true  # Enable SQL logging in development

# Development cache (using Redis)
cache:
  backend: "redis"
  redis_url: "redis://redis:6379/1"
  max_size: 1000
  default_ttl: 300

# Development logging
logging:
  log_directory: "/workspace/logs"
  root_level: "DEBUG"
  loggers:
    api:
      level: "DEBUG"
      handlers: ["console"]
      json_format: false
    model_operations:
      level: "DEBUG"
      handlers: ["console"]
      json_format: false

# Development monitoring
monitoring:
  enabled: true
  metrics:
    prometheus_enabled: true
    custom_metrics_retention_hours: 24

# Development security (relaxed for dev)
security:
  rate_limit:
    default: "10000/hour"  # Very high for development
    api_endpoints: "100000/hour"
  cors:
    enabled: true
    origins: ["*"]  # Allow all origins in development
  headers:
    force_https: false
    content_security_policy: false

# Development web interface
web_interface:
  host: "0.0.0.0"
  port: 5000
  debug: true
  cors_origins: ["http://localhost:3000", "http://127.0.0.1:3000"]

# Performance settings optimized for development
performance:
  async:
    max_thread_workers: 4
    max_process_workers: 2
    default_timeout: 30.0
  memory:
    gc_threshold_mb: 50
    monitoring_interval: 30
    warning_threshold_mb: 500
    critical_threshold_mb: 1000

DEVCONFIG

# Create development environment file
echo "🔐 Creating development environment file..."
cat > .env.development << 'ENVFILE'
# Development environment variables
FM_LLM_ENV=development
SECRET_KEY=dev-secret-key-not-for-production
DB_PASSWORD=dev_password
REDIS_URL=redis://redis:6379/1
FLASK_ENV=development
FLASK_DEBUG=1
PYTHONPATH=/workspace

# External service placeholders (optional for development)
MATHPIX_APP_ID=
MATHPIX_APP_KEY=
UNPAYWALL_EMAIL=
SEMANTIC_SCHOLAR_API_KEY=

ENVFILE

# Set up Jupyter configuration
echo "📓 Setting up Jupyter configuration..."
mkdir -p ~/.jupyter
cat > ~/.jupyter/jupyter_notebook_config.py << 'JUPYTERCONFIG'
# Jupyter configuration for development
c.NotebookApp.allow_root = True
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
c.NotebookApp.token = ''
c.NotebookApp.password = ''
c.NotebookApp.notebook_dir = '/workspace'
c.NotebookApp.allow_origin = '*'
c.NotebookApp.disable_check_xsrf = True

JUPYTERCONFIG

# Create development scripts
echo "📜 Creating development scripts..."
mkdir -p scripts/dev

cat > scripts/dev/setup-local.sh << 'SETUPSCRIPT'
#!/bin/bash
# Local development setup (outside container)

echo "Setting up local development environment..."

# Install Python dependencies
pip install -r requirements.txt
pip install -r requirements/requirements.txt  
pip install -r requirements/web_requirements.txt

# Install development dependencies
pip install pre-commit pytest pytest-cov black flake8 mypy jupyter

# Install pre-commit hooks
pre-commit install

# Create local directories
mkdir -p logs data output kb_data instance

echo "Local setup complete! Use 'docker-compose up' to start services."

SETUPSCRIPT

cat > scripts/dev/reset-dev-env.sh << 'RESETSCRIPT'
#!/bin/bash
# Reset development environment

echo "Resetting development environment..."

# Clear cache
rm -rf .pytest_cache __pycache__ */__pycache__ */*/__pycache__
rm -rf .mypy_cache
rm -rf *.egg-info

# Clear logs
rm -f logs/*.log

# Reset database (if needed)
docker-compose exec postgres psql -U dev_user -d fm_llm_dev -c "DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;"

# Clear Redis cache
docker-compose exec redis redis-cli FLUSHALL

echo "Development environment reset complete!"

RESETSCRIPT

cat > scripts/dev/run-tests.sh << 'TESTSCRIPT'
#!/bin/bash
# Run development tests

echo "Running development tests..."

# Unit tests
echo "Running unit tests..."
python -m pytest tests/unit/ -v --cov=fm_llm_solver --cov-report=html:test_results/coverage

# Integration tests
echo "Running integration tests..."
python -m pytest tests/integration/ -v

# Performance tests (quick)
echo "Running quick performance tests..."
python -m pytest tests/performance/ -v -m "not slow"

echo "Tests complete! Check test_results/ for coverage report."

TESTSCRIPT

chmod +x scripts/dev/*.sh

# Create VS Code workspace settings
echo "💻 Creating VS Code workspace settings..."
mkdir -p .vscode
cat > .vscode/settings.json << 'VSCODECONFIG'
{
    "python.defaultInterpreterPath": "/usr/local/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": [
        "tests/"
    ],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        ".mypy_cache": true,
        "*.egg-info": true
    },
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.associations": {
        "*.yaml": "yaml",
        "*.yml": "yaml"
    },
    "yaml.validate": true,
    "yaml.format.enable": true
}

VSCODECONFIG

cat > .vscode/launch.json << 'LAUNCHCONFIG'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Flask Development Server",
            "type": "python",
            "request": "launch",
            "program": "-m",
            "args": ["flask", "--app", "fm_llm_solver.web.app", "run", "--debug"],
            "env": {
                "FLASK_ENV": "development",
                "FLASK_DEBUG": "1",
                "FM_LLM_ENV": "development",
                "PYTHONPATH": "/workspace"
            },
            "console": "integratedTerminal",
            "justMyCode": false
        },
        {
            "name": "Pytest Current File",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["${file}", "-v"],
            "env": {
                "FM_LLM_ENV": "testing",
                "PYTHONPATH": "/workspace"
            },
            "console": "integratedTerminal",
            "justMyCode": false
        },
        {
            "name": "CLI Tool",
            "type": "python",
            "request": "launch",
            "program": "scripts/fm-llm",
            "args": ["--help"],
            "env": {
                "FM_LLM_ENV": "development",
                "PYTHONPATH": "/workspace"
            },
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}

LAUNCHCONFIG

# Create development README
echo "📖 Creating development README..."
cat > docs/DEVELOPMENT.md << 'DEVREADME'
# FM-LLM-Solver Development Guide

## Quick Start

### Using Dev Container (Recommended)
1. Open in VS Code
2. Install "Dev Containers" extension
3. Press `Ctrl+Shift+P` and select "Dev Containers: Reopen in Container"
4. Wait for setup to complete

### Local Development
1. Run `bash scripts/dev/setup-local.sh`
2. Start services: `docker-compose up -d postgres redis`
3. Run application: `flask --app fm_llm_solver.web.app run --debug`

## Development Workflow

### Testing
```bash
# Run all tests
bash scripts/dev/run-tests.sh

# Run specific test types
python -m pytest tests/unit/ -v          # Unit tests
python -m pytest tests/integration/ -v   # Integration tests
python -m pytest tests/performance/ -v   # Performance tests
```

### Code Quality
```bash
# Format code
black fm_llm_solver/ tests/

# Lint code
flake8 fm_llm_solver/ tests/

# Type checking
mypy fm_llm_solver/

# Pre-commit checks
pre-commit run --all-files
```

### CLI Tools
```bash
# Initialize database
flask --app fm_llm_solver.web.app init-db

# Build knowledge base
scripts/fm-llm kb build

# Run performance benchmark
flask --app fm_llm_solver.web.app benchmark

# Generate performance report
flask --app fm_llm_solver.web.app performance-report
```

### Debugging
- Use VS Code debugger with provided launch configurations
- Check logs in `logs/` directory
- Monitor with Prometheus at http://localhost:9090
- Use Jupyter at http://localhost:8888

### Environment Reset
```bash
# Reset development environment
bash scripts/dev/reset-dev-env.sh
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system architecture.

## Contributing

1. Create feature branch
2. Make changes with tests
3. Run quality checks
4. Submit pull request

DEVREADME

echo "✅ Development environment setup complete!"
echo ""
echo "🎉 Welcome to FM-LLM-Solver development!"
echo ""
echo "Available services:"
echo "  - Web Interface: http://localhost:5000"
echo "  - Jupyter Notebook: http://localhost:8888"  
echo "  - Prometheus: http://localhost:9090"
echo "  - PostgreSQL: localhost:5432"
echo "  - Redis: localhost:6379"
echo ""
echo "Quick commands:"
echo "  - Run tests: bash scripts/dev/run-tests.sh"
echo "  - Reset environment: bash scripts/dev/reset-dev-env.sh"
echo "  - Performance report: flask --app fm_llm_solver.web.app performance-report"
echo ""
echo "Happy coding! 🚀"
