#!/bin/bash

# FM-LLM-Solver Automated Setup Script
# This script automates the installation and setup of FM-LLM-Solver

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
ENVIRONMENT="development"
SKIP_DEPS=false
VERBOSE=false

usage() {
    cat << EOF
FM-LLM-Solver Automated Setup

Usage: $0 [OPTIONS]

OPTIONS:
    -e, --environment ENV    Setup environment (development, staging, production)
    -d, --skip-deps         Skip dependency installation
    -v, --verbose           Enable verbose output
    -h, --help              Show this help message

EXAMPLES:
    $0                                    # Development setup
    $0 -e production                      # Production setup

REQUIREMENTS:
    - Python 3.10+
    - Git
    - Docker and Docker Compose (optional)

EOF
}

check_requirements() {
    log "Checking system requirements..."
    
    # Check Python
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        log "Python version: $PYTHON_VERSION"
    else
        error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check Git
    if command -v git >/dev/null 2>&1; then
        GIT_VERSION=$(git --version | cut -d' ' -f3)
        log "Git version: $GIT_VERSION"
    else
        error "Git is required but not installed"
        exit 1
    fi
}

setup_python_environment() {
    log "Setting up Python environment..."
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]]; then
        log "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    log "Upgrading pip..."
    pip install --upgrade pip
    
    # Install requirements
    if [[ "$SKIP_DEPS" != true ]]; then
        log "Installing Python dependencies..."
        
        if [[ -f "requirements.txt" ]]; then
            pip install -r requirements.txt
        fi
        
        if [[ -f "requirements/requirements.txt" ]]; then
            pip install -r requirements/requirements.txt
        fi
        
        if [[ -f "requirements/web_requirements.txt" ]]; then
            pip install -r requirements/web_requirements.txt
        fi
        
        # Install development dependencies
        if [[ "$ENVIRONMENT" == "development" ]]; then
            log "Installing development dependencies..."
            pip install pytest pytest-cov pytest-asyncio black flake8 mypy pre-commit jupyter ipython
        fi
        
        # Install package in development mode
        pip install -e .
    fi
}

setup_configuration() {
    log "Setting up configuration for $ENVIRONMENT environment..."
    
    # Create directories
    mkdir -p logs data output kb_data instance config
    
    # Create environment-specific configuration
    if [[ "$ENVIRONMENT" == "development" ]]; then
        log "Setting up development configuration..."
        cat > .env << 'DEVENV'
FM_LLM_ENV=development
SECRET_KEY=dev-secret-key-not-for-production
DB_PASSWORD=dev_password
REDIS_URL=redis://localhost:6379/1
FLASK_ENV=development
FLASK_DEBUG=1
DEVENV
    fi
}

setup_development_tools() {
    if [[ "$ENVIRONMENT" != "development" ]]; then
        return
    fi
    
    log "Setting up development tools..."
    
    # Setup pre-commit hooks
    if command -v pre-commit >/dev/null 2>&1; then
        log "Installing pre-commit hooks..."
        pre-commit install
    fi
    
    # Create development scripts
    mkdir -p scripts/dev
    
    # Test runner script
    cat > scripts/dev/test.sh << 'TESTSCRIPT'
#!/bin/bash
echo "Running FM-LLM-Solver tests..."
python -m pytest tests/ -v --cov=fm_llm_solver --cov-report=html:test_results/coverage
echo "Coverage report available at: test_results/coverage/index.html"
TESTSCRIPT
    
    chmod +x scripts/dev/test.sh
    
    # Development server script
    cat > scripts/dev/dev-server.sh << 'DEVSCRIPT'
#!/bin/bash
echo "Starting FM-LLM-Solver development server..."
export FLASK_ENV=development
export FLASK_DEBUG=1
export FM_LLM_ENV=development
flask --app fm_llm_solver.web.app run --debug --host=0.0.0.0 --port=5000
DEVSCRIPT
    
    chmod +x scripts/dev/dev-server.sh
}

show_completion_message() {
    log "Setup completed successfully! 🎉"
    echo ""
    echo "Next steps:"
    echo ""
    
    if [[ "$ENVIRONMENT" == "development" ]]; then
        echo "For development:"
        echo "  1. Activate virtual environment: source venv/bin/activate"
        echo "  2. Start development server: bash scripts/dev/dev-server.sh"
        echo "  3. Run tests: bash scripts/dev/test.sh"
        echo "  4. Access web interface: http://localhost:5000"
        echo ""
        echo "Development tools:"
        echo "  - VS Code Dev Container: Open in VS Code and use Dev Containers extension"
        echo "  - Jupyter: jupyter notebook --ip=0.0.0.0 --port=8888"
    fi
    
    echo ""
    echo "Documentation:"
    echo "  - Development: docs/DEVELOPMENT.md"
    echo "  - API Reference: docs/API_REFERENCE.md"
    echo "  - User Guide: docs/USER_GUIDE.md"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -d|--skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
log "Starting FM-LLM-Solver setup for $ENVIRONMENT environment..."

check_requirements
setup_python_environment
setup_configuration
setup_development_tools
show_completion_message

log "Setup complete! 🚀" 