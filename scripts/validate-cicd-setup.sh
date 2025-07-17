#!/bin/bash

# FM-LLM-Solver CI/CD Pipeline Setup Validation Script
# This script validates that all CI/CD components are properly configured

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    local status=$1
    local message=$2
    if [[ $status == "ok" ]]; then
        echo -e "${GREEN}✅ $message${NC}"
    elif [[ $status == "warn" ]]; then
        echo -e "${YELLOW}⚠️  $message${NC}"
    elif [[ $status == "error" ]]; then
        echo -e "${RED}❌ $message${NC}"
    else
        echo -e "${BLUE}ℹ️  $message${NC}"
    fi
}

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

# Validation functions
check_files() {
    print_header "Checking CI/CD Files"
    
    local files=(
        ".github/workflows/cicd-pipeline.yml"
        ".github/workflows/rollback.yml"
        ".github/environments/development.yml"
        ".github/environments/staging.yml"
        ".github/environments/production.yml"
        ".github/branch-protection.yml"
        "tests/performance/playwright.config.js"
        "tests/performance/load-tests.spec.js"
        "tests/performance/global-setup.js"
        "tests/performance/global-teardown.js"
        "docs/CICD_DEPLOYMENT_GUIDE.md"
    )
    
    for file in "${files[@]}"; do
        if [[ -f "$file" ]]; then
            print_status "ok" "Found $file"
        else
            print_status "error" "Missing $file"
        fi
    done
}

check_dependencies() {
    print_header "Checking Dependencies"
    
    # Check for required tools
    local tools=("docker" "kubectl" "git" "curl" "jq")
    
    for tool in "${tools[@]}"; do
        if command -v "$tool" >/dev/null 2>&1; then
            print_status "ok" "$tool is available"
        else
            print_status "warn" "$tool is not installed"
        fi
    done
    
    # Check for Node.js (for Playwright)
    if command -v node >/dev/null 2>&1; then
        local node_version=$(node --version)
        print_status "ok" "Node.js is available ($node_version)"
    else
        print_status "warn" "Node.js is not installed (required for performance tests)"
    fi
    
    # Check for Python
    if command -v python3 >/dev/null 2>&1; then
        local python_version=$(python3 --version)
        print_status "ok" "Python is available ($python_version)"
    else
        print_status "error" "Python 3 is not installed"
    fi
}

check_environment_files() {
    print_header "Checking Environment Configuration"
    
    # Check for Docker configuration
    if [[ -f "docker-compose.yml" ]]; then
        print_status "ok" "Found docker-compose.yml"
    else
        print_status "warn" "Missing docker-compose.yml"
    fi
    
    if [[ -f "docker-compose.hybrid.yml" ]]; then
        print_status "ok" "Found docker-compose.hybrid.yml"
    else
        print_status "warn" "Missing docker-compose.hybrid.yml"
    fi
    
    # Check for Dockerfile
    if [[ -f "Dockerfile" ]]; then
        print_status "ok" "Found Dockerfile"
    else
        print_status "error" "Missing Dockerfile"
    fi
    
    # Check for Kubernetes manifests
    if [[ -d "deployment/kubernetes" ]]; then
        print_status "ok" "Found Kubernetes deployment directory"
        
        local k8s_files=("web-app.yaml" "namespace.yaml" "configmap.yaml" "secrets.yaml")
        for file in "${k8s_files[@]}"; do
            if [[ -f "deployment/kubernetes/$file" ]]; then
                print_status "ok" "Found deployment/kubernetes/$file"
            else
                print_status "warn" "Missing deployment/kubernetes/$file"
            fi
        done
    else
        print_status "warn" "Missing deployment/kubernetes directory"
    fi
}

check_testing_setup() {
    print_header "Checking Testing Configuration"
    
    # Check for test directories
    local test_dirs=("tests/unit" "tests/integration" "tests/performance")
    for dir in "${test_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            print_status "ok" "Found $dir directory"
        else
            print_status "warn" "Missing $dir directory"
        fi
    done
    
    # Check for specific test files
    if [[ -f "tests/test_user_account_system.py" ]]; then
        print_status "ok" "Found user account system tests"
    else
        print_status "warn" "Missing user account system tests"
    fi
    
    # Check for pytest configuration
    if [[ -f "pytest.ini" ]] || [[ -f "pyproject.toml" ]]; then
        print_status "ok" "Found pytest configuration"
    else
        print_status "warn" "Missing pytest configuration"
    fi
    
    # Check for requirements files
    if [[ -f "requirements.txt" ]]; then
        print_status "ok" "Found requirements.txt"
    else
        print_status "error" "Missing requirements.txt"
    fi
    
    if [[ -f "requirements/web_requirements.txt" ]]; then
        print_status "ok" "Found web requirements"
    else
        print_status "warn" "Missing requirements/web_requirements.txt"
    fi
}

check_security_setup() {
    print_header "Checking Security Configuration"
    
    # Check for security tools configuration
    if [[ -f ".bandit" ]] || grep -q "bandit" "pyproject.toml" 2>/dev/null; then
        print_status "ok" "Found Bandit configuration"
    else
        print_status "warn" "Missing Bandit configuration"
    fi
    
    # Check for pre-commit
    if [[ -f ".pre-commit-config.yaml" ]]; then
        print_status "ok" "Found pre-commit configuration"
    else
        print_status "warn" "Missing pre-commit configuration"
    fi
    
    # Check for GitHub security workflows
    if [[ -f ".github/workflows/security.yml" ]]; then
        print_status "ok" "Found security workflow"
    else
        print_status "warn" "Missing security workflow"
    fi
    
    # Check for dependabot
    if [[ -f ".github/dependabot.yml" ]]; then
        print_status "ok" "Found Dependabot configuration"
    else
        print_status "warn" "Missing Dependabot configuration"
    fi
}

validate_workflow_syntax() {
    print_header "Validating Workflow Syntax"
    
    # Basic YAML syntax check for workflows
    local workflow_files=(".github/workflows/"*.yml)
    
    for file in "${workflow_files[@]}"; do
        if [[ -f "$file" ]]; then
            if python3 -c "import yaml; yaml.safe_load(open('$file'))" 2>/dev/null; then
                print_status "ok" "Valid YAML syntax: $(basename "$file")"
            else
                print_status "error" "Invalid YAML syntax: $(basename "$file")"
            fi
        fi
    done
}

check_performance_tests() {
    print_header "Checking Performance Test Setup"
    
    # Check if Playwright is installed (if package.json exists)
    if [[ -f "package.json" ]]; then
        if grep -q "@playwright/test" "package.json"; then
            print_status "ok" "Playwright is configured in package.json"
        else
            print_status "warn" "Playwright not found in package.json"
        fi
    else
        print_status "info" "No package.json found - Playwright will be installed in CI"
    fi
    
    # Check performance test configuration
    if [[ -f "tests/performance/playwright.config.js" ]]; then
        if node -c "tests/performance/playwright.config.js" 2>/dev/null; then
            print_status "ok" "Valid Playwright configuration"
        else
            print_status "warn" "Playwright configuration may have syntax issues"
        fi
    fi
}

generate_setup_checklist() {
    print_header "Repository Setup Checklist"
    
    cat << EOF
To complete the CI/CD setup, you need to configure the following in your GitHub repository:

📝 Repository Settings → Secrets and Variables:

Required Secrets:
  □ KUBE_CONFIG_DEV (base64 encoded kubeconfig for dev cluster)
  □ KUBE_CONFIG_STAGING (base64 encoded kubeconfig for staging cluster)  
  □ KUBE_CONFIG_PROD (base64 encoded kubeconfig for production cluster)
  □ DEV_DATABASE_URL (development database connection string)
  □ STAGING_DATABASE_URL (staging database connection string)
  □ PROD_DATABASE_URL (production database connection string)
  □ DEV_REDIS_URL (development Redis connection string)
  □ STAGING_REDIS_URL (staging Redis connection string)
  □ PROD_REDIS_URL (production Redis connection string)
  □ SLACK_WEBHOOK (for notifications)
  □ DATADOG_API_KEY (optional, for monitoring)
  □ SENTRY_DSN (optional, for error tracking)

🔒 Repository Settings → Branches:

Development Branch:
  □ Require pull request reviews: 1
  □ Require status checks: code-quality, test-matrix (ubuntu-latest, 3.10), security-scan
  □ Allow force pushes: true

Staging Branch:
  □ Require pull request reviews: 2
  □ Require status checks: all development checks + build-and-push
  □ Allow force pushes: false
  □ Require linear history: true

Main Branch:
  □ Require pull request reviews: 3
  □ Require status checks: all staging checks + deploy-staging
  □ Restrict pushes to teams: senior-developers, devops-team, security-team
  □ Require signed commits: true
  □ Allow force pushes: false

🌍 Repository Settings → Environments:

Development Environment:
  □ Required reviewers: None
  □ Wait timer: 0 minutes
  □ Deployment branches: All branches

Staging Environment:
  □ Required reviewers: QA team members
  □ Wait timer: 5 minutes
  □ Deployment branches: Protected branches only

Production Environment:
  □ Required reviewers: Senior developers + DevOps team
  □ Wait timer: 10 minutes
  □ Deployment branches: Protected branches only
  □ Prevent self-review: true

🏗️ Infrastructure Setup:

  □ Create Kubernetes clusters for dev, staging, and production
  □ Set up monitoring stack (Prometheus, Grafana)
  □ Configure SSL certificates for domains
  □ Set up database backups
  □ Configure alerting and notification channels

EOF
}

run_basic_tests() {
    print_header "Running Basic Tests"
    
    # Test if Python tests can be discovered
    if command -v python3 >/dev/null 2>&1; then
        if python3 -m pytest --collect-only tests/ >/dev/null 2>&1; then
            print_status "ok" "Test discovery successful"
        else
            print_status "warn" "Test discovery failed - check test configuration"
        fi
    fi
    
    # Test Docker build (if Dockerfile exists)
    if [[ -f "Dockerfile" ]] && command -v docker >/dev/null 2>&1; then
        print_status "info" "Testing Docker build syntax..."
        if docker build --no-cache -t fm-llm-solver-test . >/dev/null 2>&1; then
            print_status "ok" "Docker build successful"
            docker rmi fm-llm-solver-test >/dev/null 2>&1 || true
        else
            print_status "warn" "Docker build failed - check Dockerfile"
        fi
    fi
}

# Main execution
main() {
    echo -e "${BLUE}"
    cat << "EOF"
  _____ __  __       _      _      __  __       _____       _                 
 |  ___||  \/  |     | |    | |    |  \/  |     / ____|     | |                
 | |_   | \  / |_____| |    | |    | \  / |____| (___   ___ | |_   _____ _ __ 
 |  _|  | |\/| |_____| |    | |    | |\/| |_____\___ \ / _ \| \ \ / / _ \ '__|
 | |    | |  | |     | |____| |____| |  | |     ____) | (_) | |\ V /  __/ |   
 |_|    |_|  |_|     |______|______|_|  |_|    |_____/ \___/|_| \_/ \___|_|   

                    CI/CD Pipeline Validation Script
EOF
    echo -e "${NC}\n"
    
    # Run all checks
    check_files
    check_dependencies
    check_environment_files
    check_testing_setup
    check_security_setup
    validate_workflow_syntax
    check_performance_tests
    run_basic_tests
    
    # Generate setup checklist
    generate_setup_checklist
    
    print_header "Validation Complete"
    print_status "info" "Review the checklist above to complete your CI/CD setup"
    print_status "info" "For detailed instructions, see docs/CICD_DEPLOYMENT_GUIDE.md"
}

# Run the main function
main "$@" 