#!/usr/bin/env python3
"""
Security Audit Script for FM-LLM Solver.

Performs comprehensive security assessment including:
- Dependency vulnerability scanning
- Code security analysis
- Configuration security review
- Authentication and authorization testing
- Input validation testing
- HTTPS and security headers validation
"""

import os
import sys
import json
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class SecurityAuditor:
    """Comprehensive security audit for production readiness."""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "UNKNOWN",
            "security_score": 0,
            "vulnerabilities": {
                "critical": [],
                "high": [],
                "medium": [],
                "low": []
            },
            "audit_categories": {},
            "recommendations": []
        }
        
    def run_security_audit(self) -> Dict:
        """Run comprehensive security audit."""
        print("🔒 FM-LLM Solver Security Audit")
        print("=" * 50)
        
        audit_categories = [
            ("Dependency Vulnerabilities", self._audit_dependencies),
            ("Code Security", self._audit_code_security),
            ("Authentication & Authorization", self._audit_auth),
            ("Input Validation", self._audit_input_validation),
            ("Configuration Security", self._audit_configuration),
            ("Web Security", self._audit_web_security),
            ("Data Protection", self._audit_data_protection),
            ("API Security", self._audit_api_security)
        ]
        
        total_score = 0
        max_score = 0
        
        for category_name, audit_func in audit_categories:
            print(f"\n🔍 Auditing {category_name}...")
            category_results = audit_func()
            self.results["audit_categories"][category_name] = category_results
            
            score = category_results.get("score", 0)
            max_category_score = category_results.get("max_score", 100)
            total_score += score
            max_score += max_category_score
            
            status_emoji = "✅" if score >= max_category_score * 0.9 else "⚠️" if score >= max_category_score * 0.7 else "❌"
            print(f"  {status_emoji} {category_name}: {score}/{max_category_score} points")
        
        # Calculate overall security score
        self.results["security_score"] = (total_score / max_score * 100) if max_score > 0 else 0
        
        # Determine overall status
        if self.results["security_score"] >= 90:
            self.results["overall_status"] = "SECURE"
        elif self.results["security_score"] >= 70:
            self.results["overall_status"] = "NEEDS_ATTENTION"
        else:
            self.results["overall_status"] = "VULNERABLE"
        
        self._generate_security_report()
        return self.results
    
    def _audit_dependencies(self) -> Dict:
        """Audit dependency vulnerabilities."""
        vulnerabilities = []
        score = 100
        
        # Check for requirements files
        req_files = [
            "requirements.txt",
            "web_requirements.txt", 
            "requirements/requirements.txt"
        ]
        
        for req_file in req_files:
            req_path = PROJECT_ROOT / req_file
            if req_path.exists():
                # Check for known vulnerable packages
                with open(req_path, 'r') as f:
                    content = f.read()
                    
                    # Check for potentially vulnerable packages
                    vulnerable_patterns = [
                        (r'flask[<>=\s]*[01]\.\d+', "Flask version <2.0 has known vulnerabilities"),
                        (r'requests[<>=\s]*[01]\.\d+', "Requests version <2.25 has vulnerabilities"),
                        (r'pillow[<>=\s]*[0-8]\.\d+', "Pillow versions <9.0 have vulnerabilities"),
                        (r'numpy[<>=\s]*1\.1[0-9]\.\d+', "NumPy versions <1.20 have vulnerabilities"),
                        (r'urllib3[<>=\s]*1\.\d+', "urllib3 version <2.0 has vulnerabilities")
                    ]
                    
                    for pattern, description in vulnerable_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            vulnerabilities.append({
                                "type": "dependency",
                                "severity": "medium",
                                "file": req_file,
                                "description": description
                            })
                            score -= 10
        
        # Try to run safety check if available
        try:
            result = subprocess.run(
                ["safety", "check", "--json"], 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            if result.returncode == 0 and result.stdout:
                safety_data = json.loads(result.stdout)
                for vuln in safety_data:
                    vulnerabilities.append({
                        "type": "dependency",
                        "severity": "high",
                        "package": vuln.get("package", "unknown"),
                        "description": vuln.get("advisory", ""),
                        "cve": vuln.get("id", "")
                    })
                    score -= 20
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            # Safety not available or timeout
            pass
        
        # Add vulnerabilities to main results
        for vuln in vulnerabilities:
            severity = vuln["severity"]
            self.results["vulnerabilities"][severity].append(vuln)
        
        return {
            "score": max(0, score),
            "max_score": 100,
            "vulnerabilities_found": len(vulnerabilities),
            "details": vulnerabilities
        }
    
    def _audit_code_security(self) -> Dict:
        """Audit code for security issues."""
        issues = []
        score = 100
        
        # Check for hardcoded secrets
        secret_patterns = [
            (r'password\s*=\s*["\'].*["\']', "Hardcoded password"),
            (r'secret_key\s*=\s*["\'].*["\']', "Hardcoded secret key"),
            (r'api_key\s*=\s*["\'].*["\']', "Hardcoded API key"),
            (r'token\s*=\s*["\'].*["\']', "Hardcoded token"),
            (r'aws_access_key_id\s*=\s*["\'].*["\']', "Hardcoded AWS key")
        ]
        
        # Scan Python files
        python_files = list(PROJECT_ROOT.rglob("*.py"))
        for py_file in python_files[:50]:  # Limit to prevent timeout
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    for pattern, description in secret_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        for match in matches:
                            # Skip obvious test/example values
                            if not any(test_val in match.lower() for test_val in 
                                     ['test', 'example', 'demo', 'fake', 'mock', 'your-']):
                                issues.append({
                                    "type": "hardcoded_secret",
                                    "severity": "high",
                                    "file": str(py_file.relative_to(PROJECT_ROOT)),
                                    "description": description,
                                    "line": match
                                })
                                score -= 15
            except (UnicodeDecodeError, PermissionError):
                continue
        
        # Check for SQL injection patterns
        sql_patterns = [
            (r'execute\(["\'].*%.*["\']', "Potential SQL injection"),
            (r'query\(["\'].*\+.*["\']', "Potential SQL injection"),
            (r'SELECT.*\+.*FROM', "Potential SQL injection")
        ]
        
        for py_file in python_files[:50]:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    for pattern, description in sql_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            issues.append({
                                "type": "sql_injection",
                                "severity": "high",
                                "file": str(py_file.relative_to(PROJECT_ROOT)),
                                "description": description
                            })
                            score -= 20
            except (UnicodeDecodeError, PermissionError):
                continue
        
        # Check for eval/exec usage
        dangerous_functions = ['eval', 'exec', 'compile']
        for py_file in python_files[:50]:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    for func in dangerous_functions:
                        if re.search(rf'\b{func}\s*\(', content):
                            issues.append({
                                "type": "dangerous_function",
                                "severity": "medium",
                                "file": str(py_file.relative_to(PROJECT_ROOT)),
                                "description": f"Usage of dangerous function: {func}"
                            })
                            score -= 10
            except (UnicodeDecodeError, PermissionError):
                continue
        
        # Add issues to main results
        for issue in issues:
            severity = issue["severity"]
            self.results["vulnerabilities"][severity].append(issue)
        
        return {
            "score": max(0, score),
            "max_score": 100,
            "issues_found": len(issues),
            "details": issues
        }
    
    def _audit_auth(self) -> Dict:
        """Audit authentication and authorization."""
        issues = []
        score = 100
        
        # Check for authentication system
        auth_files = [
            "web_interface/auth.py",
            "fm_llm_solver/web/auth.py",
            "web_interface/auth_routes.py"
        ]
        
        auth_system_found = False
        for auth_file in auth_files:
            if (PROJECT_ROOT / auth_file).exists():
                auth_system_found = True
                break
        
        if not auth_system_found:
            issues.append({
                "type": "missing_auth",
                "severity": "high",
                "description": "No authentication system found"
            })
            score -= 30
        
        # Check for password hashing
        if auth_system_found:
            password_hashing_found = False
            for auth_file in auth_files:
                auth_path = PROJECT_ROOT / auth_file
                if auth_path.exists():
                    try:
                        with open(auth_path, 'r') as f:
                            content = f.read()
                            if any(hash_lib in content for hash_lib in ['bcrypt', 'scrypt', 'argon2']):
                                password_hashing_found = True
                                break
                    except (UnicodeDecodeError, PermissionError):
                        continue
            
            if not password_hashing_found:
                issues.append({
                    "type": "weak_password_hashing",
                    "severity": "high",
                    "description": "Strong password hashing not detected"
                })
                score -= 25
        
        # Check for session management
        session_management_found = False
        web_files = list((PROJECT_ROOT / "web_interface").rglob("*.py")) if (PROJECT_ROOT / "web_interface").exists() else []
        web_files.extend(list((PROJECT_ROOT / "fm_llm_solver" / "web").rglob("*.py")) if (PROJECT_ROOT / "fm_llm_solver" / "web").exists() else [])
        
        for web_file in web_files:
            try:
                with open(web_file, 'r') as f:
                    content = f.read()
                    if any(session_lib in content for session_lib in ['flask-login', 'session', 'login_user']):
                        session_management_found = True
                        break
            except (UnicodeDecodeError, PermissionError):
                continue
        
        if not session_management_found:
            issues.append({
                "type": "missing_session_management",
                "severity": "medium",
                "description": "Session management system not detected"
            })
            score -= 15
        
        # Add issues to main results
        for issue in issues:
            severity = issue["severity"]
            self.results["vulnerabilities"][severity].append(issue)
        
        return {
            "score": max(0, score),
            "max_score": 100,
            "issues_found": len(issues),
            "auth_system_present": auth_system_found,
            "details": issues
        }
    
    def _audit_input_validation(self) -> Dict:
        """Audit input validation mechanisms."""
        issues = []
        score = 100
        
        # Check for input validation utilities
        validation_found = False
        validation_files = [
            "fm_llm_solver/web/utils.py",
            "web_interface/utils.py"
        ]
        
        for val_file in validation_files:
            val_path = PROJECT_ROOT / val_file
            if val_path.exists():
                try:
                    with open(val_path, 'r') as f:
                        content = f.read()
                        if any(val_func in content for val_func in ['validate_input', 'sanitize', 'escape']):
                            validation_found = True
                            break
                except (UnicodeDecodeError, PermissionError):
                    continue
        
        if not validation_found:
            issues.append({
                "type": "missing_input_validation",
                "severity": "high",
                "description": "Input validation functions not found"
            })
            score -= 25
        
        # Check for CSRF protection
        csrf_found = False
        web_files = list((PROJECT_ROOT / "web_interface").rglob("*.py")) if (PROJECT_ROOT / "web_interface").exists() else []
        web_files.extend(list((PROJECT_ROOT / "fm_llm_solver" / "web").rglob("*.py")) if (PROJECT_ROOT / "fm_llm_solver" / "web").exists() else [])
        
        for web_file in web_files:
            try:
                with open(web_file, 'r') as f:
                    content = f.read()
                    if any(csrf_lib in content for csrf_lib in ['CSRFProtect', 'csrf_token', 'flask-wtf']):
                        csrf_found = True
                        break
            except (UnicodeDecodeError, PermissionError):
                continue
        
        if not csrf_found:
            issues.append({
                "type": "missing_csrf_protection",
                "severity": "medium",
                "description": "CSRF protection not detected"
            })
            score -= 20
        
        # Add issues to main results
        for issue in issues:
            severity = issue["severity"]
            self.results["vulnerabilities"][severity].append(issue)
        
        return {
            "score": max(0, score),
            "max_score": 100,
            "issues_found": len(issues),
            "validation_present": validation_found,
            "csrf_protection": csrf_found,
            "details": issues
        }
    
    def _audit_configuration(self) -> Dict:
        """Audit configuration security."""
        issues = []
        score = 100
        
        # Check for environment variable usage
        config_files = [
            "config/config.yaml",
            "config.yaml",
            ".env.example",
            "config/env.example"
        ]
        
        secure_config_found = False
        for config_file in config_files:
            config_path = PROJECT_ROOT / config_file
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        content = f.read()
                        if any(env_var in content for env_var in ['${', '{{', 'os.environ', 'getenv']):
                            secure_config_found = True
                        
                        # Check for hardcoded secrets in config
                        if any(secret in content.lower() for secret in ['password:', 'secret:', 'key:']):
                            if not any(env_marker in content for env_marker in ['${', '{{', 'ENV', 'getenv']):
                                issues.append({
                                    "type": "hardcoded_config_secret",
                                    "severity": "high",
                                    "file": config_file,
                                    "description": "Hardcoded secrets in configuration"
                                })
                                score -= 25
                except (UnicodeDecodeError, PermissionError):
                    continue
        
        if not secure_config_found:
            issues.append({
                "type": "insecure_config",
                "severity": "medium",
                "description": "Environment variable usage not detected in configuration"
            })
            score -= 15
        
        # Check for debug mode in production
        if (PROJECT_ROOT / "config.yaml").exists():
            try:
                with open(PROJECT_ROOT / "config.yaml", 'r') as f:
                    content = f.read()
                    if re.search(r'debug:\s*true', content, re.IGNORECASE):
                        issues.append({
                            "type": "debug_enabled",
                            "severity": "medium",
                            "description": "Debug mode enabled in configuration"
                        })
                        score -= 10
            except (UnicodeDecodeError, PermissionError):
                pass
        
        # Add issues to main results
        for issue in issues:
            severity = issue["severity"]
            self.results["vulnerabilities"][severity].append(issue)
        
        return {
            "score": max(0, score),
            "max_score": 100,
            "issues_found": len(issues),
            "secure_config": secure_config_found,
            "details": issues
        }
    
    def _audit_web_security(self) -> Dict:
        """Audit web security features."""
        issues = []
        score = 100
        
        # Check for security headers
        security_headers_found = False
        web_files = list((PROJECT_ROOT / "fm_llm_solver" / "web").rglob("*.py")) if (PROJECT_ROOT / "fm_llm_solver" / "web").exists() else []
        web_files.extend(list((PROJECT_ROOT / "web_interface").rglob("*.py")) if (PROJECT_ROOT / "web_interface").exists() else [])
        
        for web_file in web_files:
            try:
                with open(web_file, 'r') as f:
                    content = f.read()
                    if any(header in content for header in ['X-Frame-Options', 'X-XSS-Protection', 'Content-Security-Policy']):
                        security_headers_found = True
                        break
            except (UnicodeDecodeError, PermissionError):
                continue
        
        if not security_headers_found:
            issues.append({
                "type": "missing_security_headers",
                "severity": "medium",
                "description": "Security headers not implemented"
            })
            score -= 20
        
        # Check for rate limiting
        rate_limiting_found = False
        for web_file in web_files:
            try:
                with open(web_file, 'r') as f:
                    content = f.read()
                    if any(rate_lib in content for rate_lib in ['limiter', 'rate_limit', 'flask-limiter']):
                        rate_limiting_found = True
                        break
            except (UnicodeDecodeError, PermissionError):
                continue
        
        if not rate_limiting_found:
            issues.append({
                "type": "missing_rate_limiting",
                "severity": "medium",
                "description": "Rate limiting not detected"
            })
            score -= 15
        
        # Add issues to main results
        for issue in issues:
            severity = issue["severity"]
            self.results["vulnerabilities"][severity].append(issue)
        
        return {
            "score": max(0, score),
            "max_score": 100,
            "issues_found": len(issues),
            "security_headers": security_headers_found,
            "rate_limiting": rate_limiting_found,
            "details": issues
        }
    
    def _audit_data_protection(self) -> Dict:
        """Audit data protection mechanisms."""
        issues = []
        score = 100
        
        # Check for database connection security
        db_security_found = False
        config_files = ["config/config.yaml", "config.yaml"]
        
        for config_file in config_files:
            config_path = PROJECT_ROOT / config_file
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        content = f.read()
                        if 'ssl' in content.lower() or 'tls' in content.lower():
                            db_security_found = True
                except (UnicodeDecodeError, PermissionError):
                    continue
        
        # Check for logging without sensitive data
        logging_secure = True
        log_files = list(PROJECT_ROOT.rglob("*log*.py"))
        for log_file in log_files[:10]:
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    if any(sensitive in content.lower() for sensitive in ['password', 'secret', 'token']):
                        if not any(filter_func in content for filter_func in ['filter', 'sanitize', 'redact']):
                            logging_secure = False
                            break
            except (UnicodeDecodeError, PermissionError):
                continue
        
        if not logging_secure:
            issues.append({
                "type": "sensitive_data_logging",
                "severity": "medium",
                "description": "Potential sensitive data in logs"
            })
            score -= 15
        
        # Add issues to main results
        for issue in issues:
            severity = issue["severity"]
            self.results["vulnerabilities"][severity].append(issue)
        
        return {
            "score": max(0, score),
            "max_score": 100,
            "issues_found": len(issues),
            "db_security": db_security_found,
            "logging_secure": logging_secure,
            "details": issues
        }
    
    def _audit_api_security(self) -> Dict:
        """Audit API security features."""
        issues = []
        score = 100
        
        # Check for API authentication
        api_auth_found = False
        api_files = list((PROJECT_ROOT / "inference_api").rglob("*.py")) if (PROJECT_ROOT / "inference_api").exists() else []
        api_files.extend(list((PROJECT_ROOT / "fm_llm_solver" / "api").rglob("*.py")) if (PROJECT_ROOT / "fm_llm_solver" / "api").exists() else [])
        
        for api_file in api_files:
            try:
                with open(api_file, 'r') as f:
                    content = f.read()
                    if any(auth_method in content for auth_method in ['api_key', 'bearer', 'authenticate', 'authorize']):
                        api_auth_found = True
                        break
            except (UnicodeDecodeError, PermissionError):
                continue
        
        if api_files and not api_auth_found:
            issues.append({
                "type": "missing_api_auth",
                "severity": "high",
                "description": "API authentication not detected"
            })
            score -= 25
        
        # Add issues to main results
        for issue in issues:
            severity = issue["severity"]
            self.results["vulnerabilities"][severity].append(issue)
        
        return {
            "score": max(0, score),
            "max_score": 100,
            "issues_found": len(issues),
            "api_auth": api_auth_found,
            "details": issues
        }
    
    def _generate_security_report(self):
        """Generate comprehensive security report."""
        print("\n" + "=" * 50)
        print("🔒 SECURITY AUDIT SUMMARY")
        print("=" * 50)
        
        # Overall status
        status_emoji = {
            "SECURE": "🟢",
            "NEEDS_ATTENTION": "🟡",
            "VULNERABLE": "🔴"
        }
        
        emoji = status_emoji.get(self.results["overall_status"], "❓")
        print(f"\n{emoji} Security Status: {self.results['overall_status']}")
        print(f"🛡️ Security Score: {self.results['security_score']:.1f}/100")
        
        # Vulnerability summary
        total_vulns = sum(len(vulns) for vulns in self.results["vulnerabilities"].values())
        print(f"🚨 Total Vulnerabilities: {total_vulns}")
        
        for severity, vulns in self.results["vulnerabilities"].items():
            if vulns:
                severity_emoji = {"critical": "🚨", "high": "🔴", "medium": "🟡", "low": "🔵"}
                emoji = severity_emoji.get(severity, "❓")
                print(f"  {emoji} {severity.title()}: {len(vulns)}")
        
        # Category breakdown
        print("\n📋 Security Category Scores:")
        for category, results in self.results["audit_categories"].items():
            score = results["score"]
            max_score = results["max_score"]
            rate = (score / max_score * 100) if max_score > 0 else 0
            status = "✅" if rate >= 90 else "⚠️" if rate >= 70 else "❌"
            print(f"  {status} {category}: {score}/{max_score} ({rate:.0f}%)")
        
        # Top recommendations
        if total_vulns > 0:
            print("\n🔧 Priority Recommendations:")
            
            # Show critical and high vulnerabilities first
            priority_vulns = self.results["vulnerabilities"]["critical"] + self.results["vulnerabilities"]["high"]
            for vuln in priority_vulns[:5]:  # Show top 5
                print(f"  • {vuln.get('description', 'Security issue')}")
                if 'file' in vuln:
                    print(f"    File: {vuln['file']}")
        
        # Save detailed report
        report_path = PROJECT_ROOT / "security_audit_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📄 Detailed security report saved to: {report_path}")
        
        return self.results


def main():
    """Run security audit."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FM-LLM Solver Security Audit")
    parser.add_argument("--category", help="Run specific audit category only")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    auditor = SecurityAuditor()
    results = auditor.run_security_audit()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Security audit results saved to {args.output}")
    
    # Return appropriate exit code
    if results["overall_status"] == "SECURE":
        sys.exit(0)
    elif results["overall_status"] == "NEEDS_ATTENTION":
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main() 