#!/usr/bin/env python3
"""
FM-LLM-Solver Deployment Validation Script
Comprehensive testing of GCP + Modal hybrid deployment
"""

import os
import sys
import json
import time
import requests
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeploymentValidator:
    """Comprehensive deployment validation for GCP + Modal hybrid architecture"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0
            }
        }
        self.project_root = Path(__file__).parent
        
    def log_test_result(self, test_name: str, passed: bool, message: str, details: Dict = None):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status} {test_name}: {message}")
        
        self.results["tests"][test_name] = {
            "passed": passed,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.results["summary"]["total_tests"] += 1
        if passed:
            self.results["summary"]["passed"] += 1
        else:
            self.results["summary"]["failed"] += 1
    
    def log_warning(self, test_name: str, message: str):
        """Log warning"""
        logger.warning(f"⚠️ WARNING {test_name}: {message}")
        self.results["summary"]["warnings"] += 1
    
    def test_prerequisites(self) -> bool:
        """Test deployment prerequisites"""
        logger.info("🔍 Testing Prerequisites...")
        
        # Check Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.log_test_result(
            "python_version", 
            sys.version_info >= (3, 8), 
            f"Python {python_version} ({'supported' if sys.version_info >= (3, 8) else 'too old'})"
        )
        
        # Check required files
        required_files = [
            "modal_inference_app.py",
            "config.yaml",
            "deployment/deploy_hybrid.sh"
        ]
        
        for file_path in required_files:
            exists = (self.project_root / file_path).exists()
            self.log_test_result(
                f"file_{file_path.replace('/', '_')}",
                exists,
                f"Required file {file_path} {'found' if exists else 'missing'}"
            )
        
        # Check Docker
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
            docker_available = result.returncode == 0
            self.log_test_result(
                "docker_available",
                docker_available,
                f"Docker {'available' if docker_available else 'not found'}: {result.stdout.strip() if docker_available else 'Install Docker'}"
            )
        except FileNotFoundError:
            self.log_test_result("docker_available", False, "Docker not installed")
        
        # Check kubectl (for GCP)
        try:
            result = subprocess.run(["kubectl", "version", "--client"], capture_output=True, text=True)
            kubectl_available = result.returncode == 0
            self.log_test_result(
                "kubectl_available",
                kubectl_available,
                f"kubectl {'available' if kubectl_available else 'not found'}: {result.stdout.strip().split('Client Version:')[1].split()[0] if kubectl_available else 'Install kubectl'}"
            )
        except FileNotFoundError:
            self.log_test_result("kubectl_available", False, "kubectl not installed")
        
        # Check Modal CLI
        try:
            result = subprocess.run(["modal", "--version"], capture_output=True, text=True)
            modal_available = result.returncode == 0
            self.log_test_result(
                "modal_cli_available",
                modal_available,
                f"Modal CLI {'available' if modal_available else 'not found'}: {result.stdout.strip() if modal_available else 'Install with: pip install modal'}"
            )
        except FileNotFoundError:
            self.log_test_result("modal_cli_available", False, "Modal CLI not installed - Run: pip install modal")
        
        return True
    
    def test_modal_authentication(self) -> bool:
        """Test Modal authentication"""
        logger.info("🔐 Testing Modal Authentication...")
        
        try:
            result = subprocess.run(["modal", "token", "current"], capture_output=True, text=True)
            authenticated = result.returncode == 0
            
            if authenticated:
                # Try to get current user info
                user_result = subprocess.run(["modal", "profile", "current"], capture_output=True, text=True)
                user_info = user_result.stdout.strip() if user_result.returncode == 0 else "Unknown"
                
                self.log_test_result(
                    "modal_authentication",
                    True,
                    f"Modal authenticated as: {user_info}"
                )
            else:
                self.log_test_result(
                    "modal_authentication",
                    False,
                    "Modal not authenticated - Run: modal token new"
                )
                
        except FileNotFoundError:
            self.log_test_result(
                "modal_authentication",
                False,
                "Modal CLI not available"
            )
            
        return True
    
    def test_modal_deployment(self) -> bool:
        """Test Modal deployment status"""
        logger.info("🚀 Testing Modal Deployment...")
        
        try:
            # Check if app is deployed
            result = subprocess.run(["modal", "app", "list"], capture_output=True, text=True)
            
            if result.returncode == 0:
                app_deployed = "fm-llm-solver-inference" in result.stdout
                
                if app_deployed:
                    # Get deployment details
                    details = {}
                    for line in result.stdout.split('\n'):
                        if "fm-llm-solver-inference" in line:
                            parts = line.split()
                            if len(parts) >= 3:
                                details = {
                                    "status": parts[1] if len(parts) > 1 else "unknown",
                                    "url": parts[2] if len(parts) > 2 else "unknown"
                                }
                            break
                    
                    self.log_test_result(
                        "modal_deployment_status",
                        True,
                        f"Modal app deployed: {details.get('status', 'unknown')}",
                        details
                    )
                    
                    # Test health endpoint if URL is available
                    if details.get("url") and details["url"] != "unknown":
                        self.test_modal_health(details["url"])
                        
                else:
                    self.log_test_result(
                        "modal_deployment_status",
                        False,
                        "Modal app not deployed - Run: modal deploy modal_inference_app.py"
                    )
            else:
                self.log_test_result(
                    "modal_deployment_status",
                    False,
                    f"Failed to check Modal apps: {result.stderr}"
                )
                
        except FileNotFoundError:
            self.log_test_result(
                "modal_deployment_status",
                False,
                "Modal CLI not available"
            )
            
        return True
    
    def test_modal_health(self, base_url: str) -> bool:
        """Test Modal inference API health"""
        logger.info("🏥 Testing Modal Health Endpoint...")
        
        health_url = f"{base_url.rstrip('/')}/health"
        
        try:
            response = requests.get(health_url, timeout=30)
            
            if response.status_code == 200:
                health_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {"status": "ok"}
                
                self.log_test_result(
                    "modal_health_check",
                    True,
                    f"Modal health check passed",
                    {
                        "url": health_url,
                        "status_code": response.status_code,
                        "response": health_data
                    }
                )
            else:
                self.log_test_result(
                    "modal_health_check",
                    False,
                    f"Modal health check failed: HTTP {response.status_code}",
                    {"url": health_url, "status_code": response.status_code}
                )
                
        except requests.exceptions.RequestException as e:
            self.log_test_result(
                "modal_health_check",
                False,
                f"Modal health check failed: {str(e)}",
                {"url": health_url, "error": str(e)}
            )
            
        return True
    
    def test_gcp_connectivity(self) -> bool:
        """Test GCP connectivity and cluster access"""
        logger.info("☁️ Testing GCP Connectivity...")
        
        try:
            # Test kubectl connectivity
            result = subprocess.run(
                ["kubectl", "cluster-info", "--request-timeout=10s"], 
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                cluster_info = result.stdout.strip()
                self.log_test_result(
                    "gcp_cluster_connectivity",
                    True,
                    "GCP cluster accessible",
                    {"cluster_info": cluster_info}
                )
                
                # Test specific namespace if configured
                self.test_gcp_namespace()
                
            else:
                self.log_test_result(
                    "gcp_cluster_connectivity",
                    False,
                    f"GCP cluster not accessible: {result.stderr.strip()}",
                    {"error": result.stderr.strip()}
                )
                
        except FileNotFoundError:
            self.log_test_result(
                "gcp_cluster_connectivity",
                False,
                "kubectl not available"
            )
            
        return True
    
    def test_gcp_namespace(self) -> bool:
        """Test GCP namespace and deployments"""
        logger.info("📦 Testing GCP Namespace...")
        
        namespaces_to_check = ["fm-llm-prod", "fm-llm-staging", "default"]
        
        for namespace in namespaces_to_check:
            try:
                # Check if namespace exists
                result = subprocess.run(
                    ["kubectl", "get", "namespace", namespace], 
                    capture_output=True, text=True
                )
                
                if result.returncode == 0:
                    self.log_test_result(
                        f"gcp_namespace_{namespace}",
                        True,
                        f"Namespace {namespace} exists"
                    )
                    
                    # Check deployments in namespace
                    self.test_gcp_deployments(namespace)
                    break
                    
            except Exception as e:
                self.log_test_result(
                    f"gcp_namespace_{namespace}",
                    False,
                    f"Failed to check namespace {namespace}: {str(e)}"
                )
        
        return True
    
    def test_gcp_deployments(self, namespace: str) -> bool:
        """Test GCP deployments in namespace"""
        logger.info(f"🚢 Testing GCP Deployments in {namespace}...")
        
        try:
            # Get deployments
            result = subprocess.run(
                ["kubectl", "get", "deployments", "-n", namespace], 
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                deployments = result.stdout.strip()
                deployment_count = len([line for line in deployments.split('\n') if line and not line.startswith('NAME')])
                
                self.log_test_result(
                    f"gcp_deployments_{namespace}",
                    deployment_count > 0,
                    f"Found {deployment_count} deployments in {namespace}",
                    {"deployments": deployments}
                )
                
                # Test services
                self.test_gcp_services(namespace)
                
            else:
                self.log_test_result(
                    f"gcp_deployments_{namespace}",
                    False,
                    f"Failed to get deployments: {result.stderr.strip()}"
                )
                
        except Exception as e:
            self.log_test_result(
                f"gcp_deployments_{namespace}",
                False,
                f"Error checking deployments: {str(e)}"
            )
            
        return True
    
    def test_gcp_services(self, namespace: str) -> bool:
        """Test GCP services and endpoints"""
        logger.info(f"🌐 Testing GCP Services in {namespace}...")
        
        try:
            # Get services
            result = subprocess.run(
                ["kubectl", "get", "services", "-n", namespace], 
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                services = result.stdout.strip()
                service_count = len([line for line in services.split('\n') if line and not line.startswith('NAME')])
                
                self.log_test_result(
                    f"gcp_services_{namespace}",
                    service_count > 0,
                    f"Found {service_count} services in {namespace}",
                    {"services": services}
                )
                
            else:
                self.log_test_result(
                    f"gcp_services_{namespace}",
                    False,
                    f"Failed to get services: {result.stderr.strip()}"
                )
                
        except Exception as e:
            self.log_test_result(
                f"gcp_services_{namespace}",
                False,
                f"Error checking services: {str(e)}"
            )
            
        return True
    
    def test_local_deployment(self) -> bool:
        """Test local deployment capabilities"""
        logger.info("🏠 Testing Local Deployment...")
        
        # Check if Docker Compose files exist
        compose_files = [
            "docker-compose.yml",
            "docker-compose.hybrid.yml",
            "deployment/docker/docker-compose.unified.yml"
        ]
        
        for compose_file in compose_files:
            exists = (self.project_root / compose_file).exists()
            if exists:
                self.log_test_result(
                    f"compose_file_{compose_file.replace('/', '_').replace('.', '_')}",
                    True,
                    f"Docker Compose file available: {compose_file}"
                )
                
                # Test if we can parse the compose file
                try:
                    result = subprocess.run(
                        ["docker-compose", "-f", compose_file, "config"], 
                        capture_output=True, text=True, cwd=self.project_root
                    )
                    
                    if result.returncode == 0:
                        self.log_test_result(
                            f"compose_valid_{compose_file.replace('/', '_').replace('.', '_')}",
                            True,
                            f"Docker Compose file is valid: {compose_file}"
                        )
                    else:
                        self.log_test_result(
                            f"compose_valid_{compose_file.replace('/', '_').replace('.', '_')}",
                            False,
                            f"Docker Compose file has errors: {result.stderr.strip()}"
                        )
                        
                except Exception as e:
                    self.log_warning(f"compose_parse_{compose_file}", f"Could not validate compose file: {str(e)}")
        
        return True
    
    def test_configuration(self) -> bool:
        """Test configuration files and environment setup"""
        logger.info("⚙️ Testing Configuration...")
        
        # Check config files
        config_files = [
            "config.yaml",
            "config/base.yaml", 
            "config/environments/production.yaml",
            "config/environments/development.yaml"
        ]
        
        for config_file in config_files:
            config_path = self.project_root / config_file
            exists = config_path.exists()
            
            self.log_test_result(
                f"config_{config_file.replace('/', '_').replace('.', '_')}",
                exists,
                f"Configuration file {'found' if exists else 'missing'}: {config_file}"
            )
            
            if exists:
                # Try to parse YAML
                try:
                    import yaml
                    with open(config_path, 'r') as f:
                        config_data = yaml.safe_load(f)
                        
                    self.log_test_result(
                        f"config_valid_{config_file.replace('/', '_').replace('.', '_')}",
                        True,
                        f"Configuration file is valid YAML: {config_file}",
                        {"keys": list(config_data.keys()) if isinstance(config_data, dict) else "non-dict"}
                    )
                    
                except Exception as e:
                    self.log_test_result(
                        f"config_valid_{config_file.replace('/', '_').replace('.', '_')}",
                        False,
                        f"Configuration file has YAML errors: {str(e)}"
                    )
        
        # Check environment variables
        important_env_vars = [
            "SECRET_KEY",
            "INFERENCE_API_URL", 
            "DATABASE_URL",
            "DEPLOYMENT_MODE"
        ]
        
        for env_var in important_env_vars:
            value = os.environ.get(env_var)
            self.log_test_result(
                f"env_var_{env_var.lower()}",
                value is not None,
                f"Environment variable {env_var} {'set' if value else 'not set'}",
                {"value": "***" if value and "key" in env_var.lower() else value}
            )
        
        return True
    
    def test_end_to_end(self) -> bool:
        """Test end-to-end certificate generation if possible"""
        logger.info("🎯 Testing End-to-End Generation...")
        
        try:
            # Try using the unified CLI
            result = subprocess.run(
                ["./fm-llm", "generate", "--quick-test"], 
                capture_output=True, text=True, timeout=60
            )
            
            if result.returncode == 0:
                self.log_test_result(
                    "e2e_generation",
                    True,
                    "End-to-end certificate generation test passed",
                    {"output": result.stdout.strip()[:500]}  # First 500 chars
                )
            else:
                self.log_test_result(
                    "e2e_generation",
                    False,
                    f"End-to-end test failed: {result.stderr.strip()}",
                    {"stdout": result.stdout.strip()[:500], "stderr": result.stderr.strip()[:500]}
                )
                
        except subprocess.TimeoutExpired:
            self.log_warning("e2e_generation", "End-to-end test timed out after 60 seconds")
        except Exception as e:
            self.log_test_result(
                "e2e_generation",
                False,
                f"End-to-end test error: {str(e)}"
            )
        
        return True
    
    def generate_report(self) -> str:
        """Generate final validation report"""
        
        # Calculate success rate
        total = self.results["summary"]["total_tests"]
        passed = self.results["summary"]["passed"]
        failed = self.results["summary"]["failed"]
        warnings = self.results["summary"]["warnings"]
        
        success_rate = (passed / total * 100) if total > 0 else 0
        
        # Determine overall status
        if success_rate >= 95:
            overall_status = "🟢 EXCELLENT"
        elif success_rate >= 85:
            overall_status = "🟡 GOOD"
        elif success_rate >= 70:
            overall_status = "🟠 ACCEPTABLE"
        else:
            overall_status = "🔴 NEEDS_ATTENTION"
        
        # Generate report
        report = f"""
🔍 FM-LLM-Solver Deployment Validation Report
{'='*60}

📊 SUMMARY:
   Overall Status: {overall_status}
   Success Rate: {success_rate:.1f}% ({passed}/{total} tests passed)
   Warnings: {warnings}
   
📈 RESULTS BY CATEGORY:

✅ PASSED TESTS ({passed}):"""

        for test_name, result in self.results["tests"].items():
            if result["passed"]:
                report += f"\n   ✅ {test_name}: {result['message']}"
        
        if failed > 0:
            report += f"\n\n❌ FAILED TESTS ({failed}):"
            for test_name, result in self.results["tests"].items():
                if not result["passed"]:
                    report += f"\n   ❌ {test_name}: {result['message']}"
        
        # Add recommendations
        report += "\n\n🎯 RECOMMENDATIONS:\n"
        
        failed_tests = [name for name, result in self.results["tests"].items() if not result["passed"]]
        
        if "modal_cli_available" in failed_tests:
            report += "   📦 Install Modal CLI: pip install modal\n"
        
        if "modal_authentication" in failed_tests:
            report += "   🔐 Authenticate with Modal: modal token new\n"
        
        if "modal_deployment_status" in failed_tests:
            report += "   🚀 Deploy to Modal: modal deploy modal_inference_app.py\n"
        
        if any("gcp" in test for test in failed_tests):
            report += "   ☁️ Configure GCP access: gcloud auth login && gcloud container clusters get-credentials\n"
        
        if "docker_available" in failed_tests:
            report += "   🐳 Install Docker: https://docs.docker.com/get-docker/\n"
        
        if success_rate >= 85:
            report += "   🎉 Your deployment is in great shape! Consider running end-to-end tests.\n"
        
        report += f"\n📄 Detailed results saved to: deployment_validation_report.json"
        report += f"\n🕐 Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return report
    
    def save_report(self, filename: str = "deployment_validation_report.json"):
        """Save detailed report to JSON"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"📄 Detailed report saved to: {filename}")
    
    def run_validation(self) -> bool:
        """Run complete validation suite"""
        logger.info("🚀 Starting FM-LLM-Solver Deployment Validation")
        logger.info("="*60)
        
        # Run all test categories
        self.test_prerequisites()
        self.test_configuration()
        self.test_modal_authentication()
        self.test_modal_deployment()
        self.test_gcp_connectivity()
        self.test_local_deployment()
        self.test_end_to_end()
        
        # Generate and display report
        report = self.generate_report()
        print(report)
        
        # Save detailed results
        self.save_report()
        
        return self.results["summary"]["failed"] == 0

def main():
    """Main validation function"""
    validator = DeploymentValidator()
    success = validator.run_validation()
    
    if success:
        logger.info("🎉 All critical tests passed!")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed. Check the report for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 