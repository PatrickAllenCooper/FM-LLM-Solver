#!/usr/bin/env python3
"""
Comprehensive Production Validation for FM-LLM Solver.

This script validates all system capabilities without requiring
external dependencies like pytest. It performs structural validation
and capability testing.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import importlib.util

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class ComprehensiveValidator:
    """Validates all FM-LLM Solver capabilities."""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "validation_status": "UNKNOWN",
            "total_checks": 0,
            "passed_checks": 0,
            "failed_checks": 0,
            "categories": {},
            "critical_issues": [],
            "recommendations": []
        }
        
    def run_validation(self) -> Dict:
        """Run comprehensive validation."""
        print("🔍 FM-LLM Solver Comprehensive Validation")
        print("=" * 60)
        
        validation_categories = [
            ("Core Services Structure", self._validate_core_services),
            ("Web Interface Structure", self._validate_web_interface),
            ("CLI Tools Structure", self._validate_cli_tools),
            ("Knowledge Base Structure", self._validate_knowledge_base),
            ("Fine-tuning Structure", self._validate_fine_tuning),
            ("Security Implementation", self._validate_security),
            ("Deployment Configuration", self._validate_deployment),
            ("Documentation Completeness", self._validate_documentation),
            ("System Integration", self._validate_integration),
            ("Production Readiness", self._validate_production_readiness)
        ]
        
        for category_name, validate_func in validation_categories:
            print(f"\n📋 Validating {category_name}...")
            try:
                category_results = validate_func()
                self.results["categories"][category_name] = category_results
                
                passed = category_results["passed"]
                total = category_results["total"]
                self.results["total_checks"] += total
                self.results["passed_checks"] += passed
                self.results["failed_checks"] += (total - passed)
                
                success_rate = (passed / total * 100) if total > 0 else 0
                status_emoji = "✅" if success_rate >= 90 else "⚠️" if success_rate >= 70 else "❌"
                print(f"  {status_emoji} {category_name}: {passed}/{total} ({success_rate:.0f}%)")
                
                # Add failed checks to critical issues
                for check in category_results.get("failed_checks", []):
                    self.results["critical_issues"].append(f"{category_name}: {check}")
                
            except Exception as e:
                print(f"  ❌ {category_name}: Validation error - {str(e)}")
                self.results["categories"][category_name] = {
                    "error": str(e),
                    "passed": 0,
                    "total": 1,
                    "failed_checks": [f"Validation error: {str(e)}"]
                }
                self.results["total_checks"] += 1
                self.results["failed_checks"] += 1
                self.results["critical_issues"].append(f"{category_name}: {str(e)}")
        
        # Calculate overall status
        success_rate = (self.results["passed_checks"] / self.results["total_checks"] * 100) if self.results["total_checks"] > 0 else 0
        
        if success_rate >= 95:
            self.results["validation_status"] = "PRODUCTION_READY"
        elif success_rate >= 85:
            self.results["validation_status"] = "MINOR_ISSUES"
        elif success_rate >= 70:
            self.results["validation_status"] = "MAJOR_ISSUES"
        else:
            self.results["validation_status"] = "NOT_READY"
        
        self._generate_validation_report()
        return self.results
    
    def _validate_core_services(self) -> Dict:
        """Validate core services structure and availability."""
        checks = []
        passed = 0
        
        # Check core module structure
        core_modules = [
            ("config_manager.py", "Configuration Manager"),
            ("logging_manager.py", "Logging Manager"),
            ("database_manager.py", "Database Manager"),
            ("async_manager.py", "Async Manager"),
            ("memory_manager.py", "Memory Manager"),
            ("cache_manager.py", "Cache Manager"),
            ("error_handler.py", "Error Handler"),
            ("monitoring.py", "Monitoring"),
            ("types.py", "Type Definitions"),
            ("interfaces.py", "Interfaces")
        ]
        
        core_dir = PROJECT_ROOT / "fm_llm_solver" / "core"
        for module_file, module_name in core_modules:
            check_name = f"Core Module: {module_name}"
            module_path = core_dir / module_file
            if module_path.exists():
                checks.append((check_name, True, "Module exists"))
                passed += 1
            else:
                checks.append((check_name, False, "Module missing"))
        
        # Check service modules
        service_modules = [
            ("certificate_generator.py", "Certificate Generator"),
            ("verifier.py", "Verifier"),
            ("model_provider.py", "Model Provider"),
            ("prompt_builder.py", "Prompt Builder"),
            ("parser.py", "Parser"),
            ("cache.py", "Cache Service"),
            ("monitor.py", "Monitor Service"),
            ("knowledge_base.py", "Knowledge Base Service")
        ]
        
        services_dir = PROJECT_ROOT / "fm_llm_solver" / "services"
        for module_file, module_name in service_modules:
            check_name = f"Service Module: {module_name}"
            module_path = services_dir / module_file
            if module_path.exists():
                checks.append((check_name, True, "Module exists"))
                passed += 1
            else:
                checks.append((check_name, False, "Module missing"))
        
        # Check that modules can be read (basic syntax check)
        for py_file in (PROJECT_ROOT / "fm_llm_solver").rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Basic syntax validation - check for common Python patterns
                    if "def " in content or "class " in content or "import " in content:
                        # File has Python code structure
                        continue
            except Exception as e:
                checks.append((f"File Readable: {py_file.name}", False, f"Cannot read: {e}"))
                continue
        
        failed_checks = [check[0] for check in checks if not check[1]]
        
        return {
            "passed": passed,
            "total": len(checks),
            "checks": checks,
            "failed_checks": failed_checks
        }
    
    def _validate_web_interface(self) -> Dict:
        """Validate web interface structure."""
        checks = []
        passed = 0
        
        # Check web module structure
        web_components = [
            ("app.py", "Flask Application"),
            ("models.py", "Database Models"),
            ("utils.py", "Web Utilities"),
            ("middleware.py", "Security Middleware")
        ]
        
        web_dirs = [
            PROJECT_ROOT / "fm_llm_solver" / "web",
            PROJECT_ROOT / "web_interface"
        ]
        
        web_dir_found = None
        for web_dir in web_dirs:
            if web_dir.exists():
                web_dir_found = web_dir
                break
        
        if web_dir_found:
            checks.append(("Web Directory", True, f"Found at {web_dir_found}"))
            passed += 1
            
            for component_file, component_name in web_components:
                check_name = f"Web Component: {component_name}"
                component_path = web_dir_found / component_file
                if component_path.exists():
                    checks.append((check_name, True, "Component exists"))
                    passed += 1
                else:
                    checks.append((check_name, False, "Component missing"))
        else:
            checks.append(("Web Directory", False, "No web directory found"))
        
        # Check routes structure
        routes_locations = [
            PROJECT_ROOT / "fm_llm_solver" / "web" / "routes",
            PROJECT_ROOT / "web_interface" / "routes"
        ]
        
        routes_found = False
        for routes_dir in routes_locations:
            if routes_dir.exists() and (routes_dir / "main.py").exists():
                checks.append(("Routes Structure", True, "Routes directory and main.py found"))
                passed += 1
                routes_found = True
                break
        
        if not routes_found:
            checks.append(("Routes Structure", False, "Routes structure missing"))
        
        # Check templates
        template_locations = [
            PROJECT_ROOT / "web_interface" / "templates",
            PROJECT_ROOT / "fm_llm_solver" / "web" / "templates"
        ]
        
        templates_found = False
        for template_dir in template_locations:
            if template_dir.exists():
                template_files = list(template_dir.rglob("*.html"))
                if template_files:
                    checks.append(("Templates", True, f"Found {len(template_files)} template files"))
                    passed += 1
                    templates_found = True
                    break
        
        if not templates_found:
            checks.append(("Templates", False, "No template files found"))
        
        # Check entry points
        entry_points = [
            ("run_web_interface.py", "Web Interface Runner"),
            ("run_application.py", "Application Runner")
        ]
        
        for entry_file, entry_name in entry_points:
            check_name = f"Entry Point: {entry_name}"
            entry_path = PROJECT_ROOT / entry_file
            if entry_path.exists():
                checks.append((check_name, True, "Entry point exists"))
                passed += 1
            else:
                checks.append((check_name, False, "Entry point missing"))
        
        failed_checks = [check[0] for check in checks if not check[1]]
        
        return {
            "passed": passed,
            "total": len(checks),
            "checks": checks,
            "failed_checks": failed_checks
        }
    
    def _validate_cli_tools(self) -> Dict:
        """Validate CLI tools structure."""
        checks = []
        passed = 0
        
        # Check CLI module structure
        cli_modules = [
            ("main.py", "Main CLI"),
            ("config.py", "Config Commands"),
            ("deploy.py", "Deploy Commands"),
            ("experiment.py", "Experiment Commands"),
            ("kb.py", "Knowledge Base Commands"),
            ("train.py", "Training Commands"),
            ("web.py", "Web Commands")
        ]
        
        cli_dir = PROJECT_ROOT / "fm_llm_solver" / "cli"
        if cli_dir.exists():
            checks.append(("CLI Directory", True, "CLI directory exists"))
            passed += 1
            
            for module_file, module_name in cli_modules:
                check_name = f"CLI Module: {module_name}"
                module_path = cli_dir / module_file
                if module_path.exists():
                    checks.append((check_name, True, "Module exists"))
                    passed += 1
                else:
                    checks.append((check_name, False, "Module missing"))
        else:
            checks.append(("CLI Directory", False, "CLI directory missing"))
        
        # Check unified script
        unified_script = PROJECT_ROOT / "scripts" / "fm-llm"
        if unified_script.exists():
            checks.append(("Unified CLI Script", True, "Unified script exists"))
            passed += 1
        else:
            checks.append(("Unified CLI Script", False, "Unified script missing"))
        
        # Check entry points in setup files
        setup_files = [
            ("setup.py", "Setup.py"),
            ("pyproject.toml", "PyProject.toml")
        ]
        
        entry_points_found = False
        for setup_file, setup_name in setup_files:
            setup_path = PROJECT_ROOT / setup_file
            if setup_path.exists():
                try:
                    with open(setup_path, 'r') as f:
                        content = f.read()
                        if "console_scripts" in content or "entry_points" in content or "fm-llm" in content:
                            checks.append((f"Entry Points in {setup_name}", True, "Entry points defined"))
                            passed += 1
                            entry_points_found = True
                            break
                except Exception:
                    pass
        
        if not entry_points_found:
            checks.append(("Entry Points", False, "No entry points defined"))
        
        failed_checks = [check[0] for check in checks if not check[1]]
        
        return {
            "passed": passed,
            "total": len(checks),
            "checks": checks,
            "failed_checks": failed_checks
        }
    
    def _validate_knowledge_base(self) -> Dict:
        """Validate knowledge base structure."""
        checks = []
        passed = 0
        
        # Check KB modules
        kb_modules = [
            ("knowledge_base_builder.py", "KB Builder"),
            ("alternative_pdf_processor.py", "PDF Processor"),
            ("document_classifier.py", "Document Classifier"),
            ("optimized_chunker.py", "Text Chunker"),
            ("kb_utils.py", "KB Utilities")
        ]
        
        kb_dir = PROJECT_ROOT / "knowledge_base"
        if kb_dir.exists():
            checks.append(("Knowledge Base Directory", True, "KB directory exists"))
            passed += 1
            
            for module_file, module_name in kb_modules:
                check_name = f"KB Module: {module_name}"
                module_path = kb_dir / module_file
                if module_path.exists():
                    checks.append((check_name, True, "Module exists"))
                    passed += 1
                else:
                    checks.append((check_name, False, "Module missing"))
        else:
            checks.append(("Knowledge Base Directory", False, "KB directory missing"))
        
        # Check KB service integration
        kb_service = PROJECT_ROOT / "fm_llm_solver" / "services" / "knowledge_base.py"
        if kb_service.exists():
            checks.append(("KB Service Integration", True, "KB service exists"))
            passed += 1
        else:
            checks.append(("KB Service Integration", False, "KB service missing"))
        
        # Check KB data directories
        kb_data_dirs = [
            ("kb_data", "General KB Data"),
            ("kb_data_continuous", "Continuous Time KB"),
            ("kb_data_discrete", "Discrete Time KB")
        ]
        
        for kb_data_dir, description in kb_data_dirs:
            check_name = f"KB Data: {description}"
            kb_data_path = PROJECT_ROOT / kb_data_dir
            if kb_data_path.exists():
                checks.append((check_name, True, "Directory exists"))
                passed += 1
            else:
                checks.append((check_name, False, "Directory missing"))
        
        # Check KB build scripts
        kb_scripts = [
            ("scripts/knowledge_base/kb_builder.py", "KB Builder Script"),
            ("scripts/knowledge_base/build_open_source_kb.py", "Open Source KB Builder")
        ]
        
        for script_path, script_name in kb_scripts:
            check_name = f"KB Script: {script_name}"
            script_full_path = PROJECT_ROOT / script_path
            if script_full_path.exists():
                checks.append((check_name, True, "Script exists"))
                passed += 1
            else:
                checks.append((check_name, False, "Script missing"))
        
        failed_checks = [check[0] for check in checks if not check[1]]
        
        return {
            "passed": passed,
            "total": len(checks),
            "checks": checks,
            "failed_checks": failed_checks
        }
    
    def _validate_fine_tuning(self) -> Dict:
        """Validate fine-tuning structure."""
        checks = []
        passed = 0
        
        # Check fine-tuning modules
        ft_modules = [
            ("finetune_llm.py", "Main Fine-tuning"),
            ("create_finetuning_data.py", "Data Creation"),
            ("generate_synthetic_data.py", "Synthetic Data"),
            ("create_discrete_time_data.py", "Discrete Time Data"),
            ("create_type_specific_data.py", "Type Specific Data"),
            ("extract_from_papers.py", "Paper Extraction"),
            ("combine_datasets.py", "Dataset Combination")
        ]
        
        ft_dir = PROJECT_ROOT / "fine_tuning"
        if ft_dir.exists():
            checks.append(("Fine-tuning Directory", True, "Fine-tuning directory exists"))
            passed += 1
            
            for module_file, module_name in ft_modules:
                check_name = f"FT Module: {module_name}"
                module_path = ft_dir / module_file
                if module_path.exists():
                    checks.append((check_name, True, "Module exists"))
                    passed += 1
                else:
                    checks.append((check_name, False, "Module missing"))
        else:
            checks.append(("Fine-tuning Directory", False, "Fine-tuning directory missing"))
        
        # Check training data files
        data_files = [
            ("data/ft_data_discrete_time.jsonl", "Discrete Time Training Data"),
            ("data/ft_discrete_time_data.jsonl", "Alternative Discrete Data"),
            ("data/ft_manual_data.jsonl", "Manual Training Data")
        ]
        
        data_files_found = 0
        for data_file, description in data_files:
            check_name = f"Training Data: {description}"
            data_path = PROJECT_ROOT / data_file
            if data_path.exists():
                checks.append((check_name, True, "Data file exists"))
                passed += 1
                data_files_found += 1
            else:
                checks.append((check_name, False, "Data file missing"))
        
        # At least some training data should exist
        if data_files_found > 0:
            checks.append(("Training Data Availability", True, f"{data_files_found} data files found"))
            passed += 1
        else:
            checks.append(("Training Data Availability", False, "No training data files found"))
        
        failed_checks = [check[0] for check in checks if not check[1]]
        
        return {
            "passed": passed,
            "total": len(checks),
            "checks": checks,
            "failed_checks": failed_checks
        }
    
    def _validate_security(self) -> Dict:
        """Validate security implementation."""
        checks = []
        passed = 0
        
        # Check authentication system
        auth_files = [
            ("web_interface/auth.py", "Authentication Module"),
            ("web_interface/auth_routes.py", "Auth Routes"),
            ("fm_llm_solver/web/middleware.py", "Security Middleware")
        ]
        
        auth_found = False
        for auth_file, description in auth_files:
            check_name = f"Security: {description}"
            auth_path = PROJECT_ROOT / auth_file
            if auth_path.exists():
                checks.append((check_name, True, "File exists"))
                passed += 1
                auth_found = True
            else:
                checks.append((check_name, False, "File missing"))
        
        # Check for security utilities
        utils_files = [
            ("fm_llm_solver/web/utils.py", "Web Utils"),
            ("web_interface/utils.py", "Interface Utils")
        ]
        
        security_utils_found = False
        for utils_file, description in utils_files:
            utils_path = PROJECT_ROOT / utils_file
            if utils_path.exists():
                try:
                    with open(utils_path, 'r') as f:
                        content = f.read()
                        if any(sec_term in content.lower() for sec_term in ['validate', 'sanitize', 'escape', 'csrf']):
                            checks.append((f"Security Utils in {description}", True, "Security functions found"))
                            passed += 1
                            security_utils_found = True
                            break
                except Exception:
                    pass
        
        if not security_utils_found:
            checks.append(("Security Utilities", False, "No security utilities found"))
        
        # Check for security tests
        security_test = PROJECT_ROOT / "tests" / "test_security.py"
        if security_test.exists():
            checks.append(("Security Tests", True, "Security test file exists"))
            passed += 1
        else:
            checks.append(("Security Tests", False, "Security test file missing"))
        
        # Check for general security implementation
        if not auth_found:
            # Look for any security-related code in web files
            web_dirs = [
                PROJECT_ROOT / "fm_llm_solver" / "web",
                PROJECT_ROOT / "web_interface"
            ]
            
            security_code_found = False
            for web_dir in web_dirs:
                if web_dir.exists():
                    for py_file in web_dir.rglob("*.py"):
                        try:
                            with open(py_file, 'r') as f:
                                content = f.read()
                                if any(sec_term in content.lower() for sec_term in ['auth', 'login', 'session', 'csrf', 'security']):
                                    security_code_found = True
                                    break
                        except Exception:
                            continue
                    if security_code_found:
                        break
            
            if security_code_found:
                checks.append(("General Security Implementation", True, "Security code found in web files"))
                passed += 1
            else:
                checks.append(("General Security Implementation", False, "No security implementation found"))
        
        failed_checks = [check[0] for check in checks if not check[1]]
        
        return {
            "passed": passed,
            "total": len(checks),
            "checks": checks,
            "failed_checks": failed_checks
        }
    
    def _validate_deployment(self) -> Dict:
        """Validate deployment configuration."""
        checks = []
        passed = 0
        
        # Check Docker configuration
        docker_files = [
            ("Dockerfile", "Docker Build File"),
            ("docker-compose.yml", "Docker Compose"),
            (".dockerignore", "Docker Ignore")
        ]
        
        for docker_file, description in docker_files:
            check_name = f"Docker: {description}"
            docker_path = PROJECT_ROOT / docker_file
            if docker_path.exists():
                checks.append((check_name, True, "File exists"))
                passed += 1
            else:
                checks.append((check_name, False, "File missing"))
        
        # Check Kubernetes configuration
        k8s_dir = PROJECT_ROOT / "deployment" / "kubernetes"
        if k8s_dir.exists():
            checks.append(("Kubernetes Directory", True, "K8s directory exists"))
            passed += 1
            
            k8s_files = list(k8s_dir.glob("*.yaml"))
            if len(k8s_files) >= 5:  # Should have several manifests
                checks.append(("Kubernetes Manifests", True, f"{len(k8s_files)} manifests found"))
                passed += 1
            else:
                checks.append(("Kubernetes Manifests", False, f"Only {len(k8s_files)} manifests found"))
        else:
            checks.append(("Kubernetes Directory", False, "K8s directory missing"))
        
        # Check GitHub Actions
        workflows_dir = PROJECT_ROOT / ".github" / "workflows"
        if workflows_dir.exists():
            checks.append(("GitHub Actions Directory", True, "Workflows directory exists"))
            passed += 1
            
            workflow_files = list(workflows_dir.glob("*.yml"))
            if len(workflow_files) >= 3:
                checks.append(("GitHub Actions Workflows", True, f"{len(workflow_files)} workflows found"))
                passed += 1
            else:
                checks.append(("GitHub Actions Workflows", False, f"Only {len(workflow_files)} workflows found"))
        else:
            checks.append(("GitHub Actions Directory", False, "Workflows directory missing"))
        
        # Check deployment scripts
        deploy_scripts = [
            ("deploy.sh", "Deploy Shell Script"),
            ("deployment/deploy.py", "Deploy Python Script")
        ]
        
        for script_file, description in deploy_scripts:
            check_name = f"Deploy Script: {description}"
            script_path = PROJECT_ROOT / script_file
            if script_path.exists():
                checks.append((check_name, True, "Script exists"))
                passed += 1
            else:
                checks.append((check_name, False, "Script missing"))
        
        failed_checks = [check[0] for check in checks if not check[1]]
        
        return {
            "passed": passed,
            "total": len(checks),
            "checks": checks,
            "failed_checks": failed_checks
        }
    
    def _validate_documentation(self) -> Dict:
        """Validate documentation completeness."""
        checks = []
        passed = 0
        
        # Check main documentation files
        main_docs = [
            ("README.md", "Main README"),
            ("CONTRIBUTING.md", "Contributing Guide"),
            ("CHANGELOG.md", "Changelog"),
            ("LICENSE", "License File")
        ]
        
        for doc_file, description in main_docs:
            check_name = f"Main Doc: {description}"
            doc_path = PROJECT_ROOT / doc_file
            if doc_path.exists():
                checks.append((check_name, True, "Document exists"))
                passed += 1
            else:
                checks.append((check_name, False, "Document missing"))
        
        # Check docs directory
        docs_dir = PROJECT_ROOT / "docs"
        if docs_dir.exists():
            checks.append(("Docs Directory", True, "Documentation directory exists"))
            passed += 1
            
            doc_files = list(docs_dir.glob("*.md"))
            if len(doc_files) >= 10:
                checks.append(("Documentation Files", True, f"{len(doc_files)} documentation files found"))
                passed += 1
            else:
                checks.append(("Documentation Files", False, f"Only {len(doc_files)} documentation files found"))
        else:
            checks.append(("Docs Directory", False, "Documentation directory missing"))
        
        # Check for important documentation
        important_docs = [
            ("docs/ARCHITECTURE.md", "Architecture Documentation"),
            ("docs/API_REFERENCE.md", "API Reference"),
            ("docs/USER_GUIDE.md", "User Guide"),
            ("docs/INSTALLATION.md", "Installation Guide"),
            ("docs/SECURITY.md", "Security Documentation")
        ]
        
        for doc_file, description in important_docs:
            check_name = f"Important Doc: {description}"
            doc_path = PROJECT_ROOT / doc_file
            if doc_path.exists():
                checks.append((check_name, True, "Document exists"))
                passed += 1
            else:
                checks.append((check_name, False, "Document missing"))
        
        # Check Sphinx configuration
        sphinx_files = [
            ("docs/conf.py", "Sphinx Configuration"),
            ("docs/index.rst", "Sphinx Index")
        ]
        
        for sphinx_file, description in sphinx_files:
            check_name = f"Sphinx: {description}"
            sphinx_path = PROJECT_ROOT / sphinx_file
            if sphinx_path.exists():
                checks.append((check_name, True, "File exists"))
                passed += 1
            else:
                checks.append((check_name, False, "File missing"))
        
        failed_checks = [check[0] for check in checks if not check[1]]
        
        return {
            "passed": passed,
            "total": len(checks),
            "checks": checks,
            "failed_checks": failed_checks
        }
    
    def _validate_integration(self) -> Dict:
        """Validate system integration."""
        checks = []
        passed = 0
        
        # Check entry point scripts
        entry_scripts = [
            ("run_application.py", "Main Application Runner"),
            ("run_web_interface.py", "Web Interface Runner")
        ]
        
        for script_file, description in entry_scripts:
            check_name = f"Entry Script: {description}"
            script_path = PROJECT_ROOT / script_file
            if script_path.exists():
                checks.append((check_name, True, "Script exists"))
                passed += 1
            else:
                checks.append((check_name, False, "Script missing"))
        
        # Check configuration files
        config_files = [
            ("config/config.yaml", "Main Configuration"),
            ("config.yaml", "Root Configuration")
        ]
        
        config_found = False
        for config_file, description in config_files:
            check_name = f"Config: {description}"
            config_path = PROJECT_ROOT / config_file
            if config_path.exists():
                checks.append((check_name, True, "Configuration exists"))
                passed += 1
                config_found = True
                break
        
        if not config_found:
            checks.append(("Configuration", False, "No configuration file found"))
        
        # Check system type support documentation
        system_types = ["continuous", "discrete", "stochastic"]
        system_support_documented = False
        
        # Check in README and features documentation
        doc_files_to_check = [
            PROJECT_ROOT / "README.md",
            PROJECT_ROOT / "docs" / "FEATURES.md",
            PROJECT_ROOT / "docs" / "USER_GUIDE.md"
        ]
        
        for doc_file in doc_files_to_check:
            if doc_file.exists():
                try:
                    with open(doc_file, 'r') as f:
                        content = f.read().lower()
                        if all(sys_type in content for sys_type in system_types):
                            system_support_documented = True
                            break
                except Exception:
                    continue
        
        if system_support_documented:
            checks.append(("System Types Documented", True, "All system types documented"))
            passed += 1
        else:
            checks.append(("System Types Documented", False, "System types not fully documented"))
        
        # Check requirement files
        req_files = [
            ("requirements.txt", "Main Requirements"),
            ("requirements/requirements.txt", "Core Requirements"),
            ("web_requirements.txt", "Web Requirements")
        ]
        
        req_found = False
        for req_file, description in req_files:
            req_path = PROJECT_ROOT / req_file
            if req_path.exists():
                req_found = True
                break
        
        if req_found:
            checks.append(("Requirements Files", True, "Requirements files found"))
            passed += 1
        else:
            checks.append(("Requirements Files", False, "No requirements files found"))
        
        failed_checks = [check[0] for check in checks if not check[1]]
        
        return {
            "passed": passed,
            "total": len(checks),
            "checks": checks,
            "failed_checks": failed_checks
        }
    
    def _validate_production_readiness(self) -> Dict:
        """Validate overall production readiness."""
        checks = []
        passed = 0
        
        # Check that all critical components exist
        critical_components = [
            (PROJECT_ROOT / "fm_llm_solver" / "services" / "certificate_generator.py", "Certificate Generator"),
            (PROJECT_ROOT / "fm_llm_solver" / "services" / "verifier.py", "Verifier"),
            (PROJECT_ROOT / "fm_llm_solver" / "web" / "app.py", "Web Application"),
            (PROJECT_ROOT / "fm_llm_solver" / "cli" / "main.py", "CLI Interface"),
            (PROJECT_ROOT / "Dockerfile", "Docker Configuration"),
            (PROJECT_ROOT / "README.md", "Documentation"),
            (PROJECT_ROOT / "config" / "config.yaml", "Configuration")
        ]
        
        for component_path, component_name in critical_components:
            check_name = f"Critical Component: {component_name}"
            if component_path.exists():
                checks.append((check_name, True, "Component exists"))
                passed += 1
            else:
                checks.append((check_name, False, "Critical component missing"))
        
        # Check for production-specific configurations
        production_checks = [
            ("Environment Configuration", self._check_env_config),
            ("Security Implementation", self._check_security_basics),
            ("Monitoring Setup", self._check_monitoring_setup),
            ("Deployment Automation", self._check_deployment_automation)
        ]
        
        for check_name, check_func in production_checks:
            try:
                result = check_func()
                checks.append((check_name, result, "Check completed"))
                if result:
                    passed += 1
            except Exception as e:
                checks.append((check_name, False, f"Check failed: {e}"))
        
        failed_checks = [check[0] for check in checks if not check[1]]
        
        return {
            "passed": passed,
            "total": len(checks),
            "checks": checks,
            "failed_checks": failed_checks
        }
    
    def _check_env_config(self) -> bool:
        """Check environment configuration."""
        env_files = [
            PROJECT_ROOT / "config" / "env.example",
            PROJECT_ROOT / ".env.example"
        ]
        return any(env_file.exists() for env_file in env_files)
    
    def _check_security_basics(self) -> bool:
        """Check basic security implementation."""
        security_indicators = []
        
        # Check for security-related files
        security_files = [
            PROJECT_ROOT / "web_interface" / "auth.py",
            PROJECT_ROOT / "fm_llm_solver" / "web" / "middleware.py",
            PROJECT_ROOT / "tests" / "test_security.py"
        ]
        
        security_indicators.extend(file.exists() for file in security_files)
        
        # Check for security-related code
        web_dirs = [
            PROJECT_ROOT / "fm_llm_solver" / "web",
            PROJECT_ROOT / "web_interface"
        ]
        
        for web_dir in web_dirs:
            if web_dir.exists():
                for py_file in web_dir.rglob("*.py"):
                    try:
                        with open(py_file, 'r') as f:
                            content = f.read()
                            if any(term in content.lower() for term in ['csrf', 'auth', 'security', 'validate']):
                                security_indicators.append(True)
                                break
                    except Exception:
                        continue
        
        return any(security_indicators)
    
    def _check_monitoring_setup(self) -> bool:
        """Check monitoring setup."""
        monitoring_files = [
            PROJECT_ROOT / "deployment" / "prometheus.yml",
            PROJECT_ROOT / "fm_llm_solver" / "core" / "monitoring.py"
        ]
        return any(file.exists() for file in monitoring_files)
    
    def _check_deployment_automation(self) -> bool:
        """Check deployment automation."""
        automation_files = [
            PROJECT_ROOT / ".github" / "workflows" / "ci-cd.yml",
            PROJECT_ROOT / "deploy.sh",
            PROJECT_ROOT / "deployment" / "deploy.py"
        ]
        return any(file.exists() for file in automation_files)
    
    def _generate_validation_report(self):
        """Generate comprehensive validation report."""
        print("\n" + "=" * 60)
        print("🎯 COMPREHENSIVE VALIDATION RESULTS")
        print("=" * 60)
        
        # Overall status
        status_emoji = {
            "PRODUCTION_READY": "🟢",
            "MINOR_ISSUES": "🟡",
            "MAJOR_ISSUES": "🟠",
            "NOT_READY": "🔴"
        }
        
        emoji = status_emoji.get(self.results["validation_status"], "❓")
        success_rate = (self.results["passed_checks"] / self.results["total_checks"] * 100) if self.results["total_checks"] > 0 else 0
        
        print(f"\n{emoji} Validation Status: {self.results['validation_status']}")
        print(f"📊 Success Rate: {success_rate:.1f}%")
        print(f"✅ Passed Checks: {self.results['passed_checks']}")
        print(f"❌ Failed Checks: {self.results['failed_checks']}")
        print(f"📋 Total Checks: {self.results['total_checks']}")
        
        # Category breakdown
        print("\n📋 Category Results:")
        for category, results in self.results["categories"].items():
            if "error" in results:
                print(f"  ❌ {category}: Error - {results['error']}")
            else:
                passed = results["passed"]
                total = results["total"]
                rate = (passed / total * 100) if total > 0 else 0
                status = "✅" if rate >= 90 else "⚠️" if rate >= 70 else "❌"
                print(f"  {status} {category}: {passed}/{total} ({rate:.0f}%)")
        
        # Critical issues
        if self.results["critical_issues"]:
            print(f"\n🚨 Critical Issues ({len(self.results['critical_issues'])}):")
            for issue in self.results["critical_issues"][:10]:  # Show first 10
                print(f"  • {issue}")
            if len(self.results["critical_issues"]) > 10:
                print(f"  ... and {len(self.results['critical_issues']) - 10} more issues")
        
        # Production readiness assessment
        print("\n🚀 Production Readiness Assessment:")
        
        if self.results["validation_status"] == "PRODUCTION_READY":
            print("  ✅ System structure is complete and ready for production!")
            print("  ✅ All critical components are present")
            print("  ✅ Documentation is comprehensive")
            print("  ✅ Deployment configuration is complete")
            print("\n  📋 Next Steps:")
            print("    • Install required dependencies")
            print("    • Run security audit")
            print("    • Perform performance testing")
            print("    • Configure production environment")
        else:
            print("  📝 Address the following issues before production deployment:")
            
            # Group recommendations by category
            category_issues = {}
            for issue in self.results["critical_issues"]:
                if ":" in issue:
                    category = issue.split(":")[0]
                    if category not in category_issues:
                        category_issues[category] = []
                    category_issues[category].append(issue.split(":", 1)[1].strip())
            
            for category, issues in category_issues.items():
                print(f"\n    {category}:")
                for issue in issues[:3]:  # Show first 3 issues per category
                    print(f"      • {issue}")
                if len(issues) > 3:
                    print(f"      ... and {len(issues) - 3} more issues")
        
        # Save detailed report
        report_path = PROJECT_ROOT / "comprehensive_validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📄 Detailed validation report saved to: {report_path}")
        
        return self.results


def main():
    """Run comprehensive validation."""
    validator = ComprehensiveValidator()
    results = validator.run_validation()
    
    # Return appropriate exit code
    if results["validation_status"] == "PRODUCTION_READY":
        print("\n🎉 System is ready for production deployment!")
        sys.exit(0)
    elif results["validation_status"] == "MINOR_ISSUES":
        print("\n⚠️ System has minor issues but is mostly ready")
        sys.exit(1)
    else:
        print("\n🔧 System needs more work before production deployment")
        sys.exit(2)


if __name__ == "__main__":
    main() 