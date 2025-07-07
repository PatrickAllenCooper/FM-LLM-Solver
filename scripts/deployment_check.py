#!/usr/bin/env python3
"""
Deployment Readiness Check for FM-LLM Solver.

Validates all deployment configurations and infrastructure readiness:
- Docker configuration validation
- Kubernetes manifest validation
- Environment configuration checks
- CI/CD pipeline validation
- Production readiness assessment
"""

import os
import sys
import json
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DeploymentChecker:
    """Comprehensive deployment readiness checker."""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "deployment_status": "UNKNOWN",
            "readiness_score": 0,
            "checks": {},
            "critical_issues": [],
            "warnings": [],
            "recommendations": []
        }
        
    def run_deployment_checks(self) -> Dict:
        """Run all deployment readiness checks."""
        print("🚀 FM-LLM Solver Deployment Readiness Check")
        print("=" * 60)
        
        check_categories = [
            ("Docker Configuration", self._check_docker),
            ("Kubernetes Manifests", self._check_kubernetes),
            ("Environment Configuration", self._check_environment),
            ("CI/CD Pipeline", self._check_cicd),
            ("Production Configuration", self._check_production_config),
            ("Health Checks", self._check_health_endpoints),
            ("Resource Requirements", self._check_resource_requirements),
            ("Security Configuration", self._check_security_config)
        ]
        
        total_score = 0
        max_score = 0
        
        for category_name, check_func in check_categories:
            print(f"\n🔍 Checking {category_name}...")
            category_results = check_func()
            self.results["checks"][category_name] = category_results
            
            score = category_results.get("score", 0)
            max_category_score = category_results.get("max_score", 100)
            total_score += score
            max_score += max_category_score
            
            status_emoji = "✅" if score >= max_category_score * 0.9 else "⚠️" if score >= max_category_score * 0.7 else "❌"
            print(f"  {status_emoji} {category_name}: {score}/{max_category_score} points")
        
        # Calculate overall readiness score
        self.results["readiness_score"] = (total_score / max_score * 100) if max_score > 0 else 0
        
        # Determine deployment status
        if self.results["readiness_score"] >= 90:
            self.results["deployment_status"] = "READY_TO_DEPLOY"
        elif self.results["readiness_score"] >= 75:
            self.results["deployment_status"] = "NEEDS_MINOR_FIXES"
        elif self.results["readiness_score"] >= 60:
            self.results["deployment_status"] = "NEEDS_MAJOR_FIXES"
        else:
            self.results["deployment_status"] = "NOT_READY"
        
        self._generate_deployment_report()
        return self.results
    
    def _check_docker(self) -> Dict:
        """Check Docker configuration."""
        issues = []
        score = 100
        
        # Check Dockerfile exists and is valid
        dockerfile_path = PROJECT_ROOT / "Dockerfile"
        if not dockerfile_path.exists():
            issues.append("Dockerfile is missing")
            score -= 30
        else:
            try:
                with open(dockerfile_path, 'r') as f:
                    dockerfile_content = f.read()
                    
                    # Check for multi-stage build
                    if "FROM" not in dockerfile_content:
                        issues.append("Invalid Dockerfile - no FROM instruction")
                        score -= 20
                    
                    # Check for security best practices
                    if "USER root" in dockerfile_content and "USER " not in dockerfile_content.split("USER root")[1]:
                        issues.append("Dockerfile runs as root user")
                        score -= 15
                    
                    # Check for health check
                    if "HEALTHCHECK" not in dockerfile_content:
                        issues.append("No health check in Dockerfile")
                        score -= 10
                    
                    # Check for multi-stage optimization
                    if dockerfile_content.count("FROM") < 2:
                        issues.append("Single-stage Dockerfile - consider multi-stage build")
                        score -= 5
                        
            except Exception as e:
                issues.append(f"Error reading Dockerfile: {e}")
                score -= 25
        
        # Check docker-compose.yml
        compose_path = PROJECT_ROOT / "docker-compose.yml"
        if not compose_path.exists():
            issues.append("docker-compose.yml is missing")
            score -= 20
        else:
            try:
                with open(compose_path, 'r') as f:
                    compose_content = yaml.safe_load(f)
                    
                    services = compose_content.get('services', {})
                    if not services:
                        issues.append("No services defined in docker-compose.yml")
                        score -= 15
                    
                    # Check for health checks in services
                    for service_name, service_config in services.items():
                        if 'healthcheck' not in service_config:
                            issues.append(f"No health check for service {service_name}")
                            score -= 5
                    
                    # Check for resource limits
                    has_resource_limits = any(
                        'deploy' in service and 'resources' in service['deploy']
                        for service in services.values()
                    )
                    if not has_resource_limits:
                        issues.append("No resource limits defined")
                        score -= 10
                        
            except Exception as e:
                issues.append(f"Error reading docker-compose.yml: {e}")
                score -= 20
        
        # Check .dockerignore
        dockerignore_path = PROJECT_ROOT / ".dockerignore"
        if not dockerignore_path.exists():
            issues.append(".dockerignore is missing")
            score -= 5
        
        return {
            "score": max(0, score),
            "max_score": 100,
            "issues": issues,
            "dockerfile_exists": dockerfile_path.exists(),
            "compose_exists": compose_path.exists(),
            "dockerignore_exists": dockerignore_path.exists()
        }
    
    def _check_kubernetes(self) -> Dict:
        """Check Kubernetes manifests."""
        issues = []
        score = 100
        
        k8s_dir = PROJECT_ROOT / "deployment" / "kubernetes"
        if not k8s_dir.exists():
            issues.append("Kubernetes deployment directory missing")
            return {"score": 0, "max_score": 100, "issues": issues, "manifests_found": 0}
        
        # Required manifests
        required_manifests = [
            "namespace.yaml",
            "web-app.yaml", 
            "postgres.yaml",
            "redis.yaml",
            "configmap.yaml",
            "secrets.yaml",
            "ingress.yaml"
        ]
        
        found_manifests = 0
        manifest_files = list(k8s_dir.glob("*.yaml"))
        
        for required in required_manifests:
            manifest_path = k8s_dir / required
            if manifest_path.exists():
                found_manifests += 1
                
                # Validate YAML syntax
                try:
                    with open(manifest_path, 'r') as f:
                        yaml.safe_load(f)
                except yaml.YAMLError as e:
                    issues.append(f"Invalid YAML in {required}: {e}")
                    score -= 10
            else:
                issues.append(f"Missing Kubernetes manifest: {required}")
                score -= 15
        
        # Check for resource limits and requests
        for manifest_file in manifest_files:
            try:
                with open(manifest_file, 'r') as f:
                    manifest = yaml.safe_load(f)
                    
                    if isinstance(manifest, dict) and manifest.get('kind') == 'Deployment':
                        containers = (manifest.get('spec', {})
                                    .get('template', {})
                                    .get('spec', {})
                                    .get('containers', []))
                        
                        for container in containers:
                            if 'resources' not in container:
                                issues.append(f"No resource limits in {manifest_file.name}")
                                score -= 5
                                break
                            
            except Exception as e:
                issues.append(f"Error validating {manifest_file.name}: {e}")
                score -= 5
        
        return {
            "score": max(0, score),
            "max_score": 100,
            "issues": issues,
            "manifests_found": found_manifests,
            "required_manifests": len(required_manifests),
            "total_manifests": len(manifest_files)
        }
    
    def _check_environment(self) -> Dict:
        """Check environment configuration."""
        issues = []
        score = 100
        
        # Check for environment configuration files
        env_files = [
            "config/env.example",
            ".env.example",
            "config/config.yaml"
        ]
        
        env_config_found = False
        for env_file in env_files:
            if (PROJECT_ROOT / env_file).exists():
                env_config_found = True
                break
        
        if not env_config_found:
            issues.append("No environment configuration template found")
            score -= 25
        
        # Check configuration structure
        config_path = PROJECT_ROOT / "config" / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    
                    # Check for required configuration sections
                    required_sections = ['model', 'web_interface', 'database', 'logging']
                    for section in required_sections:
                        if section not in config:
                            issues.append(f"Missing configuration section: {section}")
                            score -= 10
                    
                    # Check for environment variable usage
                    config_str = str(config)
                    if not any(env_marker in config_str for env_marker in ['${', '{{', 'ENV']):
                        issues.append("Configuration doesn't use environment variables")
                        score -= 15
                        
            except Exception as e:
                issues.append(f"Error reading configuration: {e}")
                score -= 20
        
        # Check for secrets management
        secrets_files = [
            "deployment/kubernetes/secrets.yaml",
            "config/secrets.yaml"
        ]
        
        secrets_config_found = any((PROJECT_ROOT / f).exists() for f in secrets_files)
        if not secrets_config_found:
            issues.append("No secrets configuration found")
            score -= 15
        
        return {
            "score": max(0, score),
            "max_score": 100,
            "issues": issues,
            "env_config_found": env_config_found,
            "secrets_config_found": secrets_config_found
        }
    
    def _check_cicd(self) -> Dict:
        """Check CI/CD pipeline configuration."""
        issues = []
        score = 100
        
        # Check GitHub Actions workflows
        workflows_dir = PROJECT_ROOT / ".github" / "workflows"
        if not workflows_dir.exists():
            issues.append("GitHub Actions workflows directory missing")
            score -= 30
        else:
            workflow_files = list(workflows_dir.glob("*.yml"))
            if len(workflow_files) < 2:
                issues.append("Insufficient GitHub Actions workflows")
                score -= 20
            
            # Check for essential workflows
            essential_workflows = ['ci.yml', 'ci-cd.yml']
            for workflow in essential_workflows:
                workflow_path = workflows_dir / workflow
                if not workflow_path.exists():
                    issues.append(f"Missing essential workflow: {workflow}")
                    score -= 15
                else:
                    # Validate workflow syntax
                    try:
                        with open(workflow_path, 'r') as f:
                            yaml.safe_load(f)
                    except yaml.YAMLError as e:
                        issues.append(f"Invalid YAML in {workflow}: {e}")
                        score -= 10
        
        # Check for deployment automation
        deploy_scripts = [
            "deploy.sh",
            "deployment/deploy.py",
            "scripts/deploy.py"
        ]
        
        deploy_automation_found = any((PROJECT_ROOT / script).exists() for script in deploy_scripts)
        if not deploy_automation_found:
            issues.append("No deployment automation scripts found")
            score -= 15
        
        return {
            "score": max(0, score),
            "max_score": 100,
            "issues": issues,
            "workflows_found": len(workflow_files) if 'workflow_files' in locals() else 0,
            "deploy_automation": deploy_automation_found
        }
    
    def _check_production_config(self) -> Dict:
        """Check production-specific configuration."""
        issues = []
        score = 100
        
        # Check for production configuration
        config_path = PROJECT_ROOT / "config" / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    
                    # Check debug mode
                    if config.get('debug', True):
                        issues.append("Debug mode enabled in configuration")
                        score -= 15
                    
                    # Check logging configuration
                    logging_config = config.get('logging', {})
                    if logging_config.get('level', 'DEBUG') == 'DEBUG':
                        issues.append("Debug logging level in production")
                        score -= 10
                    
                    # Check database configuration
                    db_config = config.get('database', {})
                    if 'sqlite' in str(db_config).lower():
                        issues.append("SQLite not recommended for production")
                        score -= 20
                        
            except Exception as e:
                issues.append(f"Error reading production config: {e}")
                score -= 15
        
        # Check for monitoring configuration
        monitoring_configs = [
            "deployment/prometheus.yml",
            "config/monitoring.yaml"
        ]
        
        monitoring_found = any((PROJECT_ROOT / config).exists() for config in monitoring_configs)
        if not monitoring_found:
            issues.append("No monitoring configuration found")
            score -= 15
        
        return {
            "score": max(0, score),
            "max_score": 100,
            "issues": issues,
            "monitoring_configured": monitoring_found
        }
    
    def _check_health_endpoints(self) -> Dict:
        """Check health check endpoints."""
        issues = []
        score = 100
        
        # Check for health check implementations
        health_check_found = False
        
        # Check web interface health checks
        web_files = []
        if (PROJECT_ROOT / "fm_llm_solver" / "web").exists():
            web_files.extend((PROJECT_ROOT / "fm_llm_solver" / "web").rglob("*.py"))
        if (PROJECT_ROOT / "web_interface").exists():
            web_files.extend((PROJECT_ROOT / "web_interface").rglob("*.py"))
        
        for web_file in web_files:
            try:
                with open(web_file, 'r') as f:
                    content = f.read()
                    if '/health' in content or 'health_check' in content:
                        health_check_found = True
                        break
            except Exception:
                continue
        
        if not health_check_found:
            issues.append("No health check endpoints found")
            score -= 20
        
        # Check for readiness probes
        k8s_dir = PROJECT_ROOT / "deployment" / "kubernetes"
        readiness_probe_found = False
        
        if k8s_dir.exists():
            for manifest_file in k8s_dir.glob("*.yaml"):
                try:
                    with open(manifest_file, 'r') as f:
                        content = f.read()
                        if 'readinessProbe' in content or 'livenessProbe' in content:
                            readiness_probe_found = True
                            break
                except Exception:
                    continue
        
        if not readiness_probe_found:
            issues.append("No readiness/liveness probes in Kubernetes manifests")
            score -= 15
        
        return {
            "score": max(0, score),
            "max_score": 100,
            "issues": issues,
            "health_endpoints": health_check_found,
            "k8s_probes": readiness_probe_found
        }
    
    def _check_resource_requirements(self) -> Dict:
        """Check resource requirements and limits."""
        issues = []
        score = 100
        
        # Check for resource documentation
        resource_docs = [
            "docs/DEPLOYMENT.md",
            "docs/INSTALLATION.md",
            "README.md"
        ]
        
        resource_info_found = False
        for doc_file in resource_docs:
            doc_path = PROJECT_ROOT / doc_file
            if doc_path.exists():
                try:
                    with open(doc_path, 'r') as f:
                        content = f.read().lower()
                        if any(keyword in content for keyword in ['memory', 'cpu', 'requirements', 'resources']):
                            resource_info_found = True
                            break
                except Exception:
                    continue
        
        if not resource_info_found:
            issues.append("No resource requirements documentation found")
            score -= 15
        
        # Check Docker resource limits
        compose_path = PROJECT_ROOT / "docker-compose.yml"
        docker_limits_found = False
        
        if compose_path.exists():
            try:
                with open(compose_path, 'r') as f:
                    compose_content = yaml.safe_load(f)
                    
                    services = compose_content.get('services', {})
                    for service in services.values():
                        if 'deploy' in service and 'resources' in service['deploy']:
                            docker_limits_found = True
                            break
                            
            except Exception:
                pass
        
        if not docker_limits_found:
            issues.append("No resource limits in Docker Compose")
            score -= 10
        
        return {
            "score": max(0, score),
            "max_score": 100,
            "issues": issues,
            "resource_docs": resource_info_found,
            "docker_limits": docker_limits_found
        }
    
    def _check_security_config(self) -> Dict:
        """Check security configuration for deployment."""
        issues = []
        score = 100
        
        # Check for HTTPS configuration
        ingress_files = list((PROJECT_ROOT / "deployment" / "kubernetes").glob("ingress*.yaml")) if (PROJECT_ROOT / "deployment" / "kubernetes").exists() else []
        
        https_configured = False
        for ingress_file in ingress_files:
            try:
                with open(ingress_file, 'r') as f:
                    content = f.read()
                    if 'tls:' in content or 'https' in content:
                        https_configured = True
                        break
            except Exception:
                continue
        
        if not https_configured:
            issues.append("HTTPS not configured in ingress")
            score -= 20
        
        # Check for security context in Kubernetes
        security_context_found = False
        k8s_dir = PROJECT_ROOT / "deployment" / "kubernetes"
        
        if k8s_dir.exists():
            for manifest_file in k8s_dir.glob("*.yaml"):
                try:
                    with open(manifest_file, 'r') as f:
                        content = f.read()
                        if 'securityContext' in content:
                            security_context_found = True
                            break
                except Exception:
                    continue
        
        if not security_context_found:
            issues.append("No security context in Kubernetes manifests")
            score -= 15
        
        # Check for network policies
        network_policy_found = any(
            'NetworkPolicy' in f.read_text() 
            for f in k8s_dir.glob("*.yaml") 
            if k8s_dir.exists()
        )
        
        if not network_policy_found:
            issues.append("No network policies defined")
            score -= 10
        
        return {
            "score": max(0, score),
            "max_score": 100,
            "issues": issues,
            "https_configured": https_configured,
            "security_context": security_context_found,
            "network_policies": network_policy_found
        }
    
    def _generate_deployment_report(self):
        """Generate comprehensive deployment readiness report."""
        print("\n" + "=" * 60)
        print("🚀 DEPLOYMENT READINESS ASSESSMENT")
        print("=" * 60)
        
        # Overall status
        status_emoji = {
            "READY_TO_DEPLOY": "🟢",
            "NEEDS_MINOR_FIXES": "🟡",
            "NEEDS_MAJOR_FIXES": "🟠",
            "NOT_READY": "🔴"
        }
        
        emoji = status_emoji.get(self.results["deployment_status"], "❓")
        print(f"\n{emoji} Deployment Status: {self.results['deployment_status']}")
        print(f"📊 Readiness Score: {self.results['readiness_score']:.1f}/100")
        
        # Category breakdown
        print("\n📋 Deployment Check Results:")
        for category, results in self.results["checks"].items():
            score = results["score"]
            max_score = results["max_score"]
            rate = (score / max_score * 100) if max_score > 0 else 0
            status = "✅" if rate >= 90 else "⚠️" if rate >= 70 else "❌"
            print(f"  {status} {category}: {score}/{max_score} ({rate:.0f}%)")
        
        # Critical issues
        all_issues = []
        for category, results in self.results["checks"].items():
            for issue in results.get("issues", []):
                all_issues.append(f"{category}: {issue}")
        
        if all_issues:
            print(f"\n🚨 Issues to Address ({len(all_issues)}):")
            for issue in all_issues[:10]:  # Show first 10
                print(f"  • {issue}")
            if len(all_issues) > 10:
                print(f"  ... and {len(all_issues) - 10} more issues")
        
        # Deployment recommendations
        print("\n🔧 Deployment Recommendations:")
        
        if self.results["deployment_status"] == "READY_TO_DEPLOY":
            print("  ✅ System is ready for production deployment!")
            print("  • Run final security audit before deployment")
            print("  • Ensure monitoring is configured")
            print("  • Prepare rollback procedures")
        else:
            print("  📝 Complete the following before deployment:")
            
            # Prioritize recommendations based on scores
            for category, results in self.results["checks"].items():
                if results["score"] < 70:
                    print(f"  • Fix {category} issues (Score: {results['score']}/100)")
        
        # Save detailed report
        report_path = PROJECT_ROOT / "deployment_readiness_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📄 Detailed deployment report saved to: {report_path}")
        
        return self.results


def main():
    """Run deployment readiness check."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FM-LLM Solver Deployment Readiness Check")
    parser.add_argument("--category", help="Check specific category only")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    checker = DeploymentChecker()
    results = checker.run_deployment_checks()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Deployment readiness results saved to {args.output}")
    
    # Return appropriate exit code
    if results["deployment_status"] == "READY_TO_DEPLOY":
        sys.exit(0)
    elif results["deployment_status"] == "NEEDS_MINOR_FIXES":
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main() 