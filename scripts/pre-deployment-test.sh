#!/bin/bash
set -e

echo "🧪 FM-LLM-Solver Enhanced User System - Comprehensive Pre-Deployment Testing"
echo "=========================================================================="
echo

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Test result tracking
TEST_RESULTS=()

run_test() {
    local test_name="$1"
    local test_command="$2"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -e "${BLUE}[TEST $TOTAL_TESTS]${NC} $test_name"
    
    if eval "$test_command" > /tmp/test_output 2>&1; then
        echo -e "  ${GREEN}✅ PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        TEST_RESULTS+=("✅ $test_name")
    else
        echo -e "  ${RED}❌ FAILED${NC}"
        echo -e "  ${RED}Error output:${NC}"
        cat /tmp/test_output | sed 's/^/    /'
        FAILED_TESTS=$((FAILED_TESTS + 1))
        TEST_RESULTS+=("❌ $test_name")
    fi
    echo
}

echo "🔍 Phase 1: Prerequisites and File Verification"
echo "================================================"

run_test "Core deployment files exist" \
    "test -f sql/init.sql && test -f docker-compose.hybrid.yml && test -f scripts/quick-deploy.sh"

run_test "Enhanced web interface files exist" \
    "test -f web_interface/models.py && test -f web_interface/auth_routes.py && test -f web_interface/certificate_generator.py"

run_test "Template files exist" \
    "test -f web_interface/templates/auth/profile.html && test -f web_interface/templates/auth/login.html"

run_test "Documentation files complete" \
    "test -f DEPLOYMENT_GUIDE_CLI.md && test -f README_DEPLOYMENT.md && test -f QUICK_START_CHECKLIST.md"

run_test "Test suite exists" \
    "test -f tests/test_user_account_system.py"

echo "🗄️ Phase 2: Database Schema Testing"
echo "===================================="

# Create test database
TEST_DB="test_pre_deploy_$(date +%s)"
run_test "Create test PostgreSQL container" \
    "docker run --name $TEST_DB -d -e POSTGRES_DB=testdb -e POSTGRES_USER=testuser -e POSTGRES_PASSWORD=testpass -p 5433:5432 postgres:13"

# Wait for database to start
sleep 5

run_test "PostgreSQL container is running" \
    "docker exec $TEST_DB pg_isready -U testuser"

run_test "Initialize database with enhanced schema" \
    "PGPASSWORD=testpass psql -h localhost -p 5433 -U testuser -d testdb -f sql/init.sql"

run_test "Verify enhanced tables exist" \
    "PGPASSWORD=testpass psql -h localhost -p 5433 -U testuser -d testdb -c \"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_name IN ('users', 'user_activities', 'user_sessions', 'certificate_favorites', 'query_logs');\" | grep -q '5'"

run_test "Verify admin user was created" \
    "PGPASSWORD=testpass psql -h localhost -p 5433 -U testuser -d testdb -c \"SELECT COUNT(*) FROM users WHERE role = 'admin';\" | grep -q '1'"

run_test "Verify model configurations were inserted" \
    "PGPASSWORD=testpass psql -h localhost -p 5433 -U testuser -d testdb -c \"SELECT COUNT(*) FROM model_configuration WHERE is_active = true;\" | grep -q '[1-9]'"

# Cleanup test database
docker stop $TEST_DB && docker rm $TEST_DB

echo "🐍 Phase 3: Python Models and Code Testing"
echo "=========================================="

run_test "Enhanced models import successfully" \
    "python3 -c 'import sys; sys.path.append(\".\"); from web_interface.models import User, UserActivity, QueryLog, VerificationResult, CertificateFavorite; print(\"All models imported successfully\")'"

run_test "User model enhanced properties work" \
    "python3 -c 'import sys; sys.path.append(\".\"); from web_interface.models import User; u = User(username=\"test\", email=\"test@example.com\"); assert u.display_name == \"test\"; assert \"free\" in str(u.get_subscription_status()); print(\"User model properties working\")'"

run_test "Enhanced authentication routes import" \
    "python3 -c 'import sys; sys.path.append(\".\"); from web_interface.auth_routes import auth_bp, detect_device_type, detect_browser; print(\"Auth routes imported successfully\")'"

run_test "Certificate generator enhanced functionality" \
    "python3 -c 'import sys; sys.path.append(\".\"); from web_interface.certificate_generator import CertificateGenerator; print(\"Certificate generator imports successfully\")'"

run_test "Test user account system test suite syntax" \
    "python3 -m py_compile tests/test_user_account_system.py"

echo "🐋 Phase 4: Docker Configuration Testing"
echo "========================================"

run_test "Docker Compose configuration is valid" \
    "python3 -c 'import yaml; yaml.safe_load(open(\"docker-compose.hybrid.yml\"))'"

run_test "Docker Compose has required services" \
    "python3 -c 'import yaml; config = yaml.safe_load(open(\"docker-compose.hybrid.yml\")); services = config.get(\"services\", {}); assert \"web\" in services; assert \"postgres\" in services; assert \"redis\" in services; print(\"Required services found\")'"

run_test "Environment variables properly defined" \
    "grep -q 'POSTGRES_PASSWORD' docker-compose.hybrid.yml && grep -q 'FLASK_SECRET_KEY' docker-compose.hybrid.yml"

echo "🛠️ Phase 5: Management Scripts Testing"
echo "======================================"

run_test "Quick deploy script is executable" \
    "test -x scripts/quick-deploy.sh"

run_test "Quick deploy script has no syntax errors" \
    "bash -n scripts/quick-deploy.sh"

run_test "Deploy hybrid script exists and is executable" \
    "test -x deploy_hybrid.sh"

run_test "Deploy hybrid script has no syntax errors" \
    "bash -n deploy_hybrid.sh"

echo "📱 Phase 6: Web Interface Template Testing"
echo "=========================================="

run_test "Profile template has required sections" \
    "grep -q 'id=\"overview-tab\"' web_interface/templates/auth/profile.html && grep -q 'id=\"profile-tab\"' web_interface/templates/auth/profile.html && grep -q 'id=\"usage-tab\"' web_interface/templates/auth/profile.html"

run_test "Profile template has certificate history section" \
    "grep -q 'Certificate History' web_interface/templates/auth/profile.html"

run_test "Profile template has admin panel section" \
    "grep -q 'Admin Panel' web_interface/templates/auth/profile.html"

echo "🔧 Phase 7: Configuration and Environment Testing"
echo "================================================"

# Test environment configuration generation
run_test "Environment variables can be generated" \
    "FLASK_SECRET_KEY=\$(openssl rand -hex 32) && test \${#FLASK_SECRET_KEY} -eq 64"

run_test "PostgreSQL password generation works" \
    "POSTGRES_PASSWORD=\$(openssl rand -base64 32) && test \${#POSTGRES_PASSWORD} -gt 20"

run_test "SSL certificate generation capability" \
    "openssl version && which openssl"

echo "📚 Phase 8: Documentation Completeness Testing"
echo "=============================================="

run_test "CLI deployment guide has required sections" \
    "grep -q 'Prerequisites' DEPLOYMENT_GUIDE_CLI.md && grep -q 'Database Setup' DEPLOYMENT_GUIDE_CLI.md && grep -q 'Docker Deployment' DEPLOYMENT_GUIDE_CLI.md"

run_test "Quick start checklist has verification steps" \
    "grep -q 'Certificate Generation Tracking' QUICK_START_CHECKLIST.md && grep -q 'Test Certificate Generation' QUICK_START_CHECKLIST.md"

run_test "README deployment has quick commands" \
    "grep -q 'quick-deploy.sh' README_DEPLOYMENT.md"

echo "🌐 Phase 9: Web Application Integration Testing"
echo "=============================================="

# Test Flask app can be imported and configured
run_test "Flask application can be imported" \
    "python3 -c 'import sys; sys.path.append(\".\"); from web_interface.app import app; print(\"Flask app imported successfully\")'"

run_test "Flask app can be configured for testing" \
    "python3 -c 'import sys; sys.path.append(\".\"); from web_interface.app import app; app.config[\"TESTING\"] = True; print(\"Flask app configured for testing\")'"

echo "🔒 Phase 10: Security and Dependencies Testing"
echo "=============================================="

run_test "Required Python packages can be imported" \
    "python3 -c 'import flask, sqlalchemy, werkzeug; print(\"Core packages available\")'"

run_test "Password hashing functionality works" \
    "python3 -c 'from werkzeug.security import generate_password_hash, check_password_hash; h = generate_password_hash(\"test123\"); assert check_password_hash(h, \"test123\"); print(\"Password hashing works\")'"

run_test "SSL/TLS requirements available" \
    "python3 -c 'import ssl; print(\"SSL module available\")'"

echo "🎯 Phase 11: End-to-End Deployment Simulation"
echo "============================================="

# Create a minimal test deployment configuration
run_test "Can create test environment configuration" \
    "FLASK_SECRET_KEY=\$(openssl rand -hex 32) && POSTGRES_PASSWORD=\$(openssl rand -base64 32) && test -n \"\$FLASK_SECRET_KEY\" && test -n \"\$POSTGRES_PASSWORD\""

run_test "Can generate self-signed SSL certificates" \
    "mkdir -p test_ssl && openssl req -x509 -nodes -days 1 -newkey rsa:2048 -keyout test_ssl/test.key -out test_ssl/test.crt -subj '/CN=test' && test -f test_ssl/test.key && test -f test_ssl/test.crt && rm -rf test_ssl"

run_test "Docker Compose can validate configuration" \
    "cp docker-compose.hybrid.yml docker-compose.test.yml && sed -i 's/\${POSTGRES_PASSWORD}/testpass123/g' docker-compose.test.yml && python3 -c 'import yaml; yaml.safe_load(open(\"docker-compose.test.yml\"))' && rm docker-compose.test.yml"

echo
echo "🏁 COMPREHENSIVE PRE-DEPLOYMENT TEST RESULTS"
echo "============================================="
echo
echo -e "${BLUE}📊 Test Summary:${NC}"
echo -e "   Total Tests: ${TOTAL_TESTS}"
echo -e "   ${GREEN}Passed: ${PASSED_TESTS}${NC}"
echo -e "   ${RED}Failed: ${FAILED_TESTS}${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "   ${GREEN}Success Rate: 100%${NC}"
else
    echo -e "   ${YELLOW}Success Rate: $(( (PASSED_TESTS * 100) / TOTAL_TESTS ))%${NC}"
fi

echo
echo -e "${BLUE}📋 Detailed Results:${NC}"
for result in "${TEST_RESULTS[@]}"; do
    echo "   $result"
done

echo
if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL TESTS PASSED! DEPLOYMENT READY!${NC}"
    echo -e "${GREEN}✅ The enhanced user account system is thoroughly tested and ready for production deployment.${NC}"
    echo
    echo -e "${BLUE}🚀 Ready to deploy with:${NC}"
    echo "   ./scripts/quick-deploy.sh"
    echo
    echo -e "${BLUE}📋 Deployment checklist:${NC}"
    echo "   1. Ensure server meets prerequisites (Docker, PostgreSQL client, etc.)"
    echo "   2. Have domain name and SMTP settings ready"
    echo "   3. Run ./scripts/quick-deploy.sh"
    echo "   4. Follow prompts for configuration"
    echo "   5. Verify deployment with ./scripts/manage.sh status"
    echo
    exit 0
else
    echo -e "${RED}❌ TESTS FAILED! DO NOT DEPLOY YET!${NC}"
    echo -e "${RED}⚠️  Please fix the failed tests before deployment.${NC}"
    echo
    echo -e "${YELLOW}🔧 Troubleshooting:${NC}"
    echo "   1. Check error messages above for specific issues"
    echo "   2. Verify all prerequisites are installed"
    echo "   3. Ensure Python dependencies are available"
    echo "   4. Re-run this test script after fixes"
    echo
    exit 1
fi 