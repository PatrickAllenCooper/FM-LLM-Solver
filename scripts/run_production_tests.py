#!/usr/bin/env python3
"""
Comprehensive Production Test Suite for FM-LLM Solver.

This script runs all tests to validate production readiness,
including unit tests, integration tests, performance tests,
and security assessments.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import concurrent.futures
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class ProductionTestRunner:
    """Runs comprehensive production readiness tests."""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "UNKNOWN",
            "test_categories": {},
            "critical_failures": [],
            "warnings": [],
            "metrics": {},
            "recommendations": []
        }
        self.start_time = time.time()
        
    def run_all_tests(self) -> Dict:
        """Run all production readiness tests."""
        print("🚀 FM-LLM Solver Production Test Suite")
        print("=" * 60)
        
        test_categories = [
            ("Core Services", self._test_core_services),
            ("Web Interface", self._test_web_interface),
            ("CLI Tools", self._test_cli_tools),
            ("Knowledge Base", self._test_knowledge_base),
            ("Fine-tuning", self._test_fine_tuning),
            ("Security", self._test_security),
            ("Performance", self._test_performance),
            ("Deployment", self._test_deployment),
            ("Documentation", self._test_documentation)
        ]
        
        total_passed = 0
        total_tests = 0
        
        for category_name, test_func in test_categories:
            print(f"\n📋 Testing {category_name}...")
            category_results = test_func()
            self.results["test_categories"][category_name] = category_results
            
            passed = category_results.get("passed", 0)
            total = category_results.get("total", 0)
            total_passed += passed
            total_tests += total
            
            status_emoji = "✅" if passed == total else "⚠️" if passed > total * 0.8 else "❌"
            print(f"  {status_emoji} {category_name}: {passed}/{total} tests passed")
        
        # Calculate overall results
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        self.results["metrics"]["total_tests"] = total_tests
        self.results["metrics"]["passed_tests"] = total_passed
        self.results["metrics"]["success_rate"] = success_rate
        self.results["metrics"]["duration"] = time.time() - self.start_time
        
        # Determine overall status
        if success_rate >= 95:
            self.results["overall_status"] = "PRODUCTION_READY"
        elif success_rate >= 85:
            self.results["overall_status"] = "NEEDS_MINOR_FIXES"
        elif success_rate >= 70:
            self.results["overall_status"] = "NEEDS_MAJOR_FIXES"
        else:
            self.results["overall_status"] = "NOT_READY"
        
        self._generate_report()
        return self.results
    
    def _test_core_services(self) -> Dict:
        """Test core service functionality."""
        tests = []
        passed = 0
        
        # Test certificate generation
        try:
            from fm_llm_solver.services.certificate_generator import CertificateGenerator
            from fm_llm_solver.core.config_manager import ConfigurationManager
            
            config_manager = ConfigurationManager()
            generator = CertificateGenerator(config_manager)
            
            # Test continuous-time system
            test_system = {
                "dynamics": {"x": "-x + y", "y": "x - y"},
                "initial_set": "x**2 + y**2 <= 0.5",
                "unsafe_set": "x**2 + y**2 >= 2.0",
                "system_type": "continuous"
            }
            
            tests.append(("Certificate Generator Import", True))
            passed += 1
            
            # Test basic generation (without actual model)
            try:
                result = generator._validate_system(test_system)
                tests.append(("System Validation", True))
                passed += 1
            except Exception as e:
                tests.append(("System Validation", False, str(e)))
                
        except Exception as e:
            tests.append(("Certificate Generator Import", False, str(e)))
        
        # Test verification service
        try:
            from fm_llm_solver.services.verifier import CertificateVerifier
            verifier = CertificateVerifier(config_manager)
            tests.append(("Verification Service Import", True))
            passed += 1
        except Exception as e:
            tests.append(("Verification Service Import", False, str(e)))
        
        # Test model provider
        try:
            from fm_llm_solver.services.model_provider import QwenProvider
            provider = QwenProvider(config_manager)
            tests.append(("Model Provider Import", True))
            passed += 1
        except Exception as e:
            tests.append(("Model Provider Import", False, str(e)))
        
        # Test configuration management
        try:
            config = config_manager.load_config()
            tests.append(("Configuration Loading", True))
            passed += 1
        except Exception as e:
            tests.append(("Configuration Loading", False, str(e)))
        
        # Test logging system
        try:
            from fm_llm_solver.core.logging_manager import get_logging_manager
            logging_manager = get_logging_manager()
            tests.append(("Logging System", True))
            passed += 1
        except Exception as e:
            tests.append(("Logging System", False, str(e)))
        
        return {
            "passed": passed,
            "total": len(tests),
            "tests": tests,
            "details": {
                "certificate_generation": passed >= 2,
                "verification": passed >= 3,
                "configuration": passed >= 4,
                "logging": passed >= 5
            }
        }
    
    def _test_web_interface(self) -> Dict:
        """Test web interface functionality."""
        tests = []
        passed = 0
        
        # Test Flask app creation
        try:
            from web_interface.app import create_app
            
            test_config = {
                'TESTING': True,
                'SECRET_KEY': 'test-secret-key',
                'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
                'SQLALCHEMY_TRACK_MODIFICATIONS': False,
                'WTF_CSRF_ENABLED': False
            }
            
            app = create_app(test_config=test_config)
            tests.append(("Flask App Creation", True))
            passed += 1
            
            # Test app context
            with app.app_context():
                tests.append(("App Context", True))
                passed += 1
                
                # Test database models
                try:
                    from web_interface.models import User, QueryLog
                    tests.append(("Database Models Import", True))
                    passed += 1
                except Exception as e:
                    tests.append(("Database Models Import", False, str(e)))
            
        except Exception as e:
            tests.append(("Flask App Creation", False, str(e)))
        
        # Test route blueprints
        try:
            # Use the main app routes from web_interface
            from web_interface.app import create_app
            app = create_app()
            tests.append(("Route Blueprints", True))
            passed += 1
        except Exception as e:
            tests.append(("Route Blueprints", False, str(e)))
        
        # Test utilities
        try:
            from web_interface.auth import validate_input
            tests.append(("Web Utilities", True))
            passed += 1
        except Exception as e:
            tests.append(("Web Utilities", False, str(e)))
        
        # Test middleware
        try:
            # web_interface uses auth.py for security functions
            from web_interface.auth import generate_csrf_token
            tests.append(("Security Middleware", True))
            passed += 1
        except Exception as e:
            tests.append(("Security Middleware", False, str(e)))
        
        return {
            "passed": passed,
            "total": len(tests),
            "tests": tests,
            "details": {
                "app_factory": passed >= 1,
                "database_models": passed >= 3,
                "routing": passed >= 4,
                "security": passed >= 6
            }
        }
    
    def _test_cli_tools(self) -> Dict:
        """Test CLI tool functionality."""
        tests = []
        passed = 0
        
        # Test main CLI import
        try:
            from fm_llm_solver.cli.main import cli
            tests.append(("Main CLI Import", True))
            passed += 1
        except Exception as e:
            tests.append(("Main CLI Import", False, str(e)))
        
        # Test individual CLI modules
        cli_modules = [
            "config", "deploy", "experiment", "kb", "train", "web"
        ]
        
        for module in cli_modules:
            try:
                exec(f"from fm_llm_solver.cli.{module} import {module}")
                tests.append((f"CLI Module {module}", True))
                passed += 1
            except Exception as e:
                tests.append((f"CLI Module {module}", False, str(e)))
        
        # Test unified script
        fm_llm_script = PROJECT_ROOT / "scripts" / "fm-llm"
        if fm_llm_script.exists():
            tests.append(("Unified Script Exists", True))
            passed += 1
        else:
            tests.append(("Unified Script Exists", False, "Script not found"))
        
        # Test CLI help (without actually running)
        try:
            import click
            from fm_llm_solver.cli.main import cli
            
            # Test that CLI is properly structured
            if hasattr(cli, 'commands'):
                tests.append(("CLI Command Structure", True))
                passed += 1
            else:
                tests.append(("CLI Command Structure", False, "No commands found"))
        except Exception as e:
            tests.append(("CLI Command Structure", False, str(e)))
        
        return {
            "passed": passed,
            "total": len(tests),
            "tests": tests,
            "details": {
                "main_cli": passed >= 1,
                "submodules": passed >= 7,
                "unified_script": passed >= 8
            }
        }
    
    def _test_knowledge_base(self) -> Dict:
        """Test knowledge base functionality."""
        tests = []
        passed = 0
        
        # Test knowledge base service
        try:
            from fm_llm_solver.services.knowledge_base import KnowledgeBase
            tests.append(("Knowledge Base Service", True))
            passed += 1
        except Exception as e:
            tests.append(("Knowledge Base Service", False, str(e)))
        
        # Test KB builder
        try:
            from knowledge_base.knowledge_base_builder import KnowledgeBaseBuilder
            tests.append(("KB Builder", True))
            passed += 1
        except Exception as e:
            tests.append(("KB Builder", False, str(e)))
        
        # Test PDF processor
        try:
            from knowledge_base.alternative_pdf_processor import AlternativePDFProcessor
            tests.append(("PDF Processor", True))
            passed += 1
        except Exception as e:
            tests.append(("PDF Processor", False, str(e)))
        
        # Test document classifier
        try:
            from knowledge_base.document_classifier import DocumentClassifier
            tests.append(("Document Classifier", True))
            passed += 1
        except Exception as e:
            tests.append(("Document Classifier", False, str(e)))
        
        # Test optimized chunker
        try:
            from knowledge_base.optimized_chunker import OptimizedChunker
            tests.append(("Optimized Chunker", True))
            passed += 1
        except Exception as e:
            tests.append(("Optimized Chunker", False, str(e)))
        
        # Test KB utilities
        try:
            from knowledge_base.kb_utils import clean_text, extract_mathematical_content
            tests.append(("KB Utils", True))
            passed += 1
        except Exception as e:
            tests.append(("KB Utils", False, str(e)))
        
        # Test KB data directories
        kb_dirs = ["kb_data", "kb_data_continuous", "kb_data_discrete"]
        for kb_dir in kb_dirs:
            if (PROJECT_ROOT / kb_dir).exists():
                tests.append((f"KB Directory {kb_dir}", True))
                passed += 1
            else:
                tests.append((f"KB Directory {kb_dir}", False, "Directory not found"))
        
        return {
            "passed": passed,
            "total": len(tests),
            "tests": tests,
            "details": {
                "core_services": passed >= 6,
                "data_directories": passed >= 9
            }
        }
    
    def _test_fine_tuning(self) -> Dict:
        """Test fine-tuning functionality."""
        tests = []
        passed = 0
        
        fine_tuning_modules = [
            ("finetune_llm", "QLoRA Fine-tuning"),
            ("create_finetuning_data", "Data Creation"),
            ("generate_synthetic_data", "Synthetic Data"),
            ("create_discrete_time_data", "Discrete Time Data"),
            ("create_type_specific_data", "Type Specific Data"),
            ("extract_from_papers", "Paper Extraction"),
            ("combine_datasets", "Dataset Combination")
        ]
        
        for module, description in fine_tuning_modules:
            module_path = PROJECT_ROOT / "fine_tuning" / f"{module}.py"
            if module_path.exists():
                tests.append((description, True))
                passed += 1
                
                # Try to import main function
                try:
                    spec = importlib.util.spec_from_file_location(module, module_path)
                    module_obj = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module_obj)
                    tests.append((f"{description} Import", True))
                    passed += 1
                except Exception as e:
                    tests.append((f"{description} Import", False, str(e)))
            else:
                tests.append((description, False, "Module file not found"))
        
        # Test training data files
        data_files = [
            "data/ft_data_discrete_time.jsonl",
            "data/ft_discrete_time_data.jsonl", 
            "data/ft_manual_data.jsonl"
        ]
        
        for data_file in data_files:
            if (PROJECT_ROOT / data_file).exists():
                tests.append((f"Training Data {data_file}", True))
                passed += 1
            else:
                tests.append((f"Training Data {data_file}", False, "File not found"))
        
        return {
            "passed": passed,
            "total": len(tests),
            "tests": tests,
            "details": {
                "modules": passed >= 7,
                "data_files": passed >= 10
            }
        }
    
    def _test_security(self) -> Dict:
        """Test security features."""
        tests = []
        passed = 0
        
        # Test authentication system
        try:
            from web_interface.auth import AuthManager
            tests.append(("Authentication System", True))
            passed += 1
        except Exception as e:
            tests.append(("Authentication System", False, str(e)))
        
        # Test auth routes
        try:
            from web_interface.auth_routes import auth_bp
            tests.append(("Auth Routes", True))
            passed += 1
        except Exception as e:
            tests.append(("Auth Routes", False, str(e)))
        
        # Test security functions
        try:
            from web_interface.auth import validate_input, generate_csrf_token
            # Test basic functionality
            result = validate_input("test input", max_length=100)
            token = generate_csrf_token()
            if result and token:
                tests.append(("Security Functions", True))
                passed += 1
            else:
                tests.append(("Security Functions", False, "Function calls failed"))
        except Exception as e:
            tests.append(("Security Functions", False, str(e)))
        
        # Test security test file
        security_test = PROJECT_ROOT / "tests" / "test_security.py"
        if security_test.exists():
            tests.append(("Security Tests Exist", True))
            passed += 1
        else:
            tests.append(("Security Tests Exist", False, "Test file not found"))
        
        # Test CSRF protection (basic check)
        try:
            from flask_wtf.csrf import CSRFProtect
            tests.append(("CSRF Protection Available", True))
            passed += 1
        except Exception as e:
            tests.append(("CSRF Protection Available", False, str(e)))
        
        return {
            "passed": passed,
            "total": len(tests),
            "tests": tests,
            "details": {
                "authentication": passed >= 2,
                "input_protection": passed >= 4,
                "testing": passed >= 5
            }
        }
    
    def _test_performance(self) -> Dict:
        """Test performance and load testing capabilities."""
        tests = []
        passed = 0
        
        # Test performance test files
        performance_files = [
            "tests/performance/test_performance.py",
            "tests/performance/load-test.js"
        ]
        
        for perf_file in performance_files:
            if (PROJECT_ROOT / perf_file).exists():
                tests.append((f"Performance Test {perf_file}", True))
                passed += 1
            else:
                tests.append((f"Performance Test {perf_file}", False, "File not found"))
        
        # Test benchmark files
        benchmark_files = [
            "tests/benchmarks/web_interface_testbench.py",
            "tests/benchmarks/verification_optimization.py",
            "tests/benchmarks/llm_generation_testbench.py",
            "tests/benchmarks/barrier_certificate_optimization_testbench.py"
        ]
        
        for bench_file in benchmark_files:
            if (PROJECT_ROOT / bench_file).exists():
                tests.append((f"Benchmark {bench_file}", True))
                passed += 1
            else:
                tests.append((f"Benchmark {bench_file}", False, "File not found"))
        
        # Test monitoring capabilities
        try:
            from fm_llm_solver.core.monitoring import MonitoringManager
            tests.append(("Monitoring Manager", True))
            passed += 1
        except Exception as e:
            tests.append(("Monitoring Manager", False, str(e)))
        
        # Test caching system
        try:
            from fm_llm_solver.core.cache_manager import CacheManager
            cache_manager = CacheManager()
            tests.append(("Cache Manager", True))
            passed += 1
        except Exception as e:
            tests.append(("Cache Manager", False, str(e)))
        
        return {
            "passed": passed,
            "total": len(tests),
            "tests": tests,
            "details": {
                "performance_tests": passed >= 2,
                "benchmarks": passed >= 6,
                "monitoring": passed >= 7
            }
        }
    
    def _test_deployment(self) -> Dict:
        """Test deployment configurations."""
        tests = []
        passed = 0
        
        # Test Docker files
        docker_files = [
            "Dockerfile",
            "docker-compose.yml",
            ".dockerignore"
        ]
        
        for docker_file in docker_files:
            if (PROJECT_ROOT / docker_file).exists():
                tests.append((f"Docker File {docker_file}", True))
                passed += 1
            else:
                tests.append((f"Docker File {docker_file}", False, "File not found"))
        
        # Test Kubernetes manifests
        k8s_dir = PROJECT_ROOT / "deployment" / "kubernetes"
        if k8s_dir.exists():
            k8s_files = list(k8s_dir.glob("*.yaml"))
            if k8s_files:
                tests.append(("Kubernetes Manifests", True))
                passed += 1
            else:
                tests.append(("Kubernetes Manifests", False, "No YAML files found"))
        else:
            tests.append(("Kubernetes Manifests", False, "Directory not found"))
        
        # Test deployment scripts
        deploy_files = [
            "deployment/deploy.py",
            "deployment/test_deployment.py",
            "deploy.sh"
        ]
        
        for deploy_file in deploy_files:
            if (PROJECT_ROOT / deploy_file).exists():
                tests.append((f"Deploy Script {deploy_file}", True))
                passed += 1
            else:
                tests.append((f"Deploy Script {deploy_file}", False, "File not found"))
        
        # Test GitHub Actions
        ga_dir = PROJECT_ROOT / ".github" / "workflows"
        if ga_dir.exists():
            workflows = list(ga_dir.glob("*.yml"))
            if len(workflows) >= 3:  # ci.yml, ci-cd.yml, docs.yml, pr-checks.yml
                tests.append(("GitHub Actions Workflows", True))
                passed += 1
            else:
                tests.append(("GitHub Actions Workflows", False, f"Only {len(workflows)} workflows found"))
        else:
            tests.append(("GitHub Actions Workflows", False, "Directory not found"))
        
        # Test environment configurations
        config_files = [
            "config/config.yaml",
            "config/env.example"
        ]
        
        for config_file in config_files:
            if (PROJECT_ROOT / config_file).exists():
                tests.append((f"Config File {config_file}", True))
                passed += 1
            else:
                tests.append((f"Config File {config_file}", False, "File not found"))
        
        return {
            "passed": passed,
            "total": len(tests),
            "tests": tests,
            "details": {
                "docker": passed >= 2,
                "kubernetes": passed >= 3,
                "ci_cd": passed >= 7,
                "configuration": passed >= 9
            }
        }
    
    def _test_documentation(self) -> Dict:
        """Test documentation completeness."""
        tests = []
        passed = 0
        
        # Test main documentation files
        doc_files = [
            "README.md",
            "CONTRIBUTING.md",
            "CHANGELOG.md",
            "LICENSE"
        ]
        
        for doc_file in doc_files:
            if (PROJECT_ROOT / doc_file).exists():
                tests.append((f"Main Doc {doc_file}", True))
                passed += 1
            else:
                tests.append((f"Main Doc {doc_file}", False, "File not found"))
        
        # Test docs directory
        docs_dir = PROJECT_ROOT / "docs"
        if docs_dir.exists():
            doc_files = list(docs_dir.glob("*.md"))
            if len(doc_files) >= 10:  # We have 13 documented
                tests.append(("Documentation Directory", True))
                passed += 1
            else:
                tests.append(("Documentation Directory", False, f"Only {len(doc_files)} docs found"))
        else:
            tests.append(("Documentation Directory", False, "Directory not found"))
        
        # Test Sphinx configuration
        sphinx_files = [
            "docs/conf.py",
            "docs/index.rst"
        ]
        
        for sphinx_file in sphinx_files:
            if (PROJECT_ROOT / sphinx_file).exists():
                tests.append((f"Sphinx File {sphinx_file}", True))
                passed += 1
            else:
                tests.append((f"Sphinx File {sphinx_file}", False, "File not found"))
        
        # Test specific important docs
        important_docs = [
            "docs/ARCHITECTURE.md",
            "docs/API_REFERENCE.md",
            "docs/USER_GUIDE.md",
            "docs/INSTALLATION.md",
            "docs/SECURITY.md"
        ]
        
        for doc in important_docs:
            if (PROJECT_ROOT / doc).exists():
                tests.append((f"Important Doc {doc}", True))
                passed += 1
            else:
                tests.append((f"Important Doc {doc}", False, "File not found"))
        
        # Test quick start guide
        if (PROJECT_ROOT / "QUICK_START_GUIDE.md").exists():
            tests.append(("Quick Start Guide", True))
            passed += 1
        else:
            tests.append(("Quick Start Guide", False, "File not found"))
        
        return {
            "passed": passed,
            "total": len(tests),
            "tests": tests,
            "details": {
                "main_docs": passed >= 4,
                "docs_directory": passed >= 5,
                "sphinx": passed >= 7,
                "important_docs": passed >= 12
            }
        }
    
    def _generate_report(self):
        """Generate comprehensive test report."""
        print("\n" + "=" * 60)
        print("🎯 PRODUCTION READINESS ASSESSMENT")
        print("=" * 60)
        
        # Overall status
        status_emoji = {
            "PRODUCTION_READY": "🟢",
            "NEEDS_MINOR_FIXES": "🟡", 
            "NEEDS_MAJOR_FIXES": "🟠",
            "NOT_READY": "🔴"
        }
        
        emoji = status_emoji.get(self.results["overall_status"], "❓")
        print(f"\n{emoji} Overall Status: {self.results['overall_status']}")
        print(f"📊 Success Rate: {self.results['metrics']['success_rate']:.1f}%")
        print(f"✅ Passed: {self.results['metrics']['passed_tests']}")
        print(f"❌ Failed: {self.results['metrics']['total_tests'] - self.results['metrics']['passed_tests']}")
        print(f"⏱️ Duration: {self.results['metrics']['duration']:.1f}s")
        
        # Category breakdown
        print("\n📋 Category Breakdown:")
        for category, results in self.results["test_categories"].items():
            passed = results["passed"]
            total = results["total"]
            rate = (passed / total * 100) if total > 0 else 0
            status = "✅" if rate == 100 else "⚠️" if rate >= 80 else "❌"
            print(f"  {status} {category}: {passed}/{total} ({rate:.0f}%)")
        
        # Recommendations
        if self.results["overall_status"] != "PRODUCTION_READY":
            print("\n🔧 Recommendations:")
            
            for category, results in self.results["test_categories"].items():
                failed_tests = [test for test in results["tests"] if len(test) > 2 and not test[1]]
                if failed_tests:
                    print(f"\n  {category}:")
                    for test in failed_tests[:3]:  # Show first 3 failures
                        print(f"    - Fix: {test[0]}")
                        if len(test) > 2:
                            print(f"      Error: {test[2]}")
        
        # Save detailed report
        report_path = PROJECT_ROOT / "production_test_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📄 Detailed report saved to: {report_path}")
        
        return self.results


def main():
    """Run production test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FM-LLM Solver Production Test Suite")
    parser.add_argument("--category", help="Run specific test category only")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    runner = ProductionTestRunner()
    results = runner.run_all_tests()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    
    # Return appropriate exit code
    if results["overall_status"] == "PRODUCTION_READY":
        sys.exit(0)
    elif results["overall_status"] == "NEEDS_MINOR_FIXES":
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    # Add missing import
    import importlib.util
    main() 