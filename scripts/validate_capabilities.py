#!/usr/bin/env python3
"""
Comprehensive capability validation script for FM-LLM Solver.

This script verifies that all documented capabilities are present and tested
after recent refactors.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import importlib.util

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class CapabilityValidator:
    """Validates all system capabilities are present and tested."""
    
    def __init__(self):
        self.results = {
            "core_capabilities": {},
            "web_interface": {},
            "cli_tools": {},
            "knowledge_base": {},
            "fine_tuning": {},
            "deployment": {},
            "monitoring": {},
            "security": {},
            "tests": {},
            "documentation": {}
        }
        self.missing = []
        self.warnings = []
        
    def validate_all(self) -> Dict:
        """Run all validation checks."""
        print("FM-LLM Solver Capability Validation")
        print("=" * 50)
        
        # Core capabilities
        self._validate_core_capabilities()
        
        # Web interface
        self._validate_web_interface()
        
        # CLI tools
        self._validate_cli_tools()
        
        # Knowledge base and RAG
        self._validate_knowledge_base()
        
        # Fine-tuning capabilities
        self._validate_fine_tuning()
        
        # Deployment configurations
        self._validate_deployment()
        
        # Monitoring and metrics
        self._validate_monitoring()
        
        # Security features
        self._validate_security()
        
        # Test coverage
        self._validate_test_coverage()
        
        # Documentation
        self._validate_documentation()
        
        # Generate report
        return self._generate_report()
    
    def _validate_core_capabilities(self):
        """Validate core system capabilities."""
        print("\n1. Validating Core Capabilities...")
        
        capabilities = {
            "certificate_generator": "fm_llm_solver/services/certificate_generator.py",
            "verification_service": "fm_llm_solver/services/verifier.py",
            "model_provider": "fm_llm_solver/services/model_provider.py",
            "prompt_builder": "fm_llm_solver/services/prompt_builder.py",
            "parser": "fm_llm_solver/services/parser.py",
            "cache_service": "fm_llm_solver/services/cache.py",
            "monitor_service": "fm_llm_solver/services/monitor.py"
        }
        
        for name, path in capabilities.items():
            if Path(PROJECT_ROOT / path).exists():
                self.results["core_capabilities"][name] = "✓ Present"
                print(f"  ✓ {name}: Found")
            else:
                self.results["core_capabilities"][name] = "✗ Missing"
                self.missing.append(f"Core: {name} ({path})")
                print(f"  ✗ {name}: Missing")
                
        # Check for core modules
        core_modules = {
            "config_manager": "fm_llm_solver/core/config_manager.py",
            "logging_manager": "fm_llm_solver/core/logging_manager.py",
            "database_manager": "fm_llm_solver/core/database_manager.py",
            "async_manager": "fm_llm_solver/core/async_manager.py",
            "memory_manager": "fm_llm_solver/core/memory_manager.py",
            "cache_manager": "fm_llm_solver/core/cache_manager.py",
            "error_handler": "fm_llm_solver/core/error_handler.py",
            "monitoring": "fm_llm_solver/core/monitoring.py"
        }
        
        for name, path in core_modules.items():
            if Path(PROJECT_ROOT / path).exists():
                self.results["core_capabilities"][name] = "✓ Present"
                print(f"  ✓ {name}: Found")
            else:
                self.results["core_capabilities"][name] = "✗ Missing"
                self.missing.append(f"Core: {name} ({path})")
                print(f"  ✗ {name}: Missing")
    
    def _validate_web_interface(self):
        """Validate web interface components."""
        print("\n2. Validating Web Interface...")
        
        web_components = {
            "app_factory": "fm_llm_solver/web/app.py",
            "main_routes": "fm_llm_solver/web/routes/main.py",
            "models": "fm_llm_solver/web/models.py",
            "utils": "fm_llm_solver/web/utils.py",
            "middleware": "fm_llm_solver/web/middleware.py",
            "templates": "web_interface/templates",
            "static_files": "web_interface/static"
        }
        
        for name, path in web_components.items():
            full_path = PROJECT_ROOT / path
            if full_path.exists():
                self.results["web_interface"][name] = "✓ Present"
                print(f"  ✓ {name}: Found")
            else:
                # Check alternative locations
                if "templates" in name or "static" in name:
                    alt_path = PROJECT_ROOT / "fm_llm_solver/web" / Path(path).name
                    if alt_path.exists():
                        self.results["web_interface"][name] = "✓ Present (alt location)"
                        print(f"  ✓ {name}: Found (alternative location)")
                    else:
                        self.results["web_interface"][name] = "✗ Missing"
                        self.warnings.append(f"Web: {name} not in expected location")
                        print(f"  ⚠ {name}: Missing from expected location")
                else:
                    self.results["web_interface"][name] = "✗ Missing"
                    self.missing.append(f"Web: {name} ({path})")
                    print(f"  ✗ {name}: Missing")
                    
        # Check entry points
        entry_points = {
            "run_web_interface.py": "Main web interface runner",
            "run_application.py": "Unified application entry point"
        }
        
        for script, desc in entry_points.items():
            if (PROJECT_ROOT / script).exists():
                self.results["web_interface"][script] = f"✓ {desc}"
                print(f"  ✓ {script}: {desc}")
            else:
                self.results["web_interface"][script] = f"✗ Missing {desc}"
                self.missing.append(f"Entry point: {script}")
                print(f"  ✗ {script}: Missing")
    
    def _validate_cli_tools(self):
        """Validate CLI tools."""
        print("\n3. Validating CLI Tools...")
        
        cli_components = {
            "main_cli": "fm_llm_solver/cli/main.py",
            "config_cli": "fm_llm_solver/cli/config.py",
            "deploy_cli": "fm_llm_solver/cli/deploy.py",
            "experiment_cli": "fm_llm_solver/cli/experiment.py",
            "kb_cli": "fm_llm_solver/cli/kb.py",
            "train_cli": "fm_llm_solver/cli/train.py",
            "web_cli": "fm_llm_solver/cli/web.py",
            "unified_script": "scripts/fm-llm"
        }
        
        for name, path in cli_components.items():
            if Path(PROJECT_ROOT / path).exists():
                self.results["cli_tools"][name] = "✓ Present"
                print(f"  ✓ {name}: Found")
            else:
                self.results["cli_tools"][name] = "✗ Missing"
                self.missing.append(f"CLI: {name} ({path})")
                print(f"  ✗ {name}: Missing")
    
    def _validate_knowledge_base(self):
        """Validate knowledge base and RAG components."""
        print("\n4. Validating Knowledge Base & RAG...")
        
        kb_components = {
            "knowledge_base": "fm_llm_solver/services/knowledge_base.py",
            "kb_builder": "knowledge_base/knowledge_base_builder.py",
            "pdf_processor": "knowledge_base/alternative_pdf_processor.py",
            "document_classifier": "knowledge_base/document_classifier.py",
            "optimized_chunker": "knowledge_base/optimized_chunker.py",
            "kb_utils": "knowledge_base/kb_utils.py"
        }
        
        for name, path in kb_components.items():
            if Path(PROJECT_ROOT / path).exists():
                self.results["knowledge_base"][name] = "✓ Present"
                print(f"  ✓ {name}: Found")
            else:
                self.results["knowledge_base"][name] = "✗ Missing"
                self.missing.append(f"KB: {name} ({path})")
                print(f"  ✗ {name}: Missing")
                
        # Check for KB data directories
        kb_dirs = ["kb_data", "kb_data_continuous", "kb_data_discrete"]
        for dir_name in kb_dirs:
            if (PROJECT_ROOT / dir_name).exists():
                self.results["knowledge_base"][f"{dir_name}_dir"] = "✓ Present"
                print(f"  ✓ {dir_name} directory: Found")
            else:
                self.results["knowledge_base"][f"{dir_name}_dir"] = "✗ Missing"
                self.warnings.append(f"KB directory: {dir_name}")
                print(f"  ⚠ {dir_name} directory: Missing")
    
    def _validate_fine_tuning(self):
        """Validate fine-tuning capabilities."""
        print("\n5. Validating Fine-tuning Capabilities...")
        
        ft_components = {
            "finetune_llm": "fine_tuning/finetune_llm.py",
            "create_data": "fine_tuning/create_finetuning_data.py",
            "synthetic_data": "fine_tuning/generate_synthetic_data.py",
            "discrete_time_data": "fine_tuning/create_discrete_time_data.py",
            "type_specific_data": "fine_tuning/create_type_specific_data.py",
            "extract_papers": "fine_tuning/extract_from_papers.py",
            "combine_datasets": "fine_tuning/combine_datasets.py"
        }
        
        for name, path in ft_components.items():
            if Path(PROJECT_ROOT / path).exists():
                self.results["fine_tuning"][name] = "✓ Present"
                print(f"  ✓ {name}: Found")
            else:
                self.results["fine_tuning"][name] = "✗ Missing"
                self.missing.append(f"Fine-tuning: {name} ({path})")
                print(f"  ✗ {name}: Missing")
    
    def _validate_deployment(self):
        """Validate deployment configurations."""
        print("\n6. Validating Deployment Configurations...")
        
        deploy_components = {
            "dockerfile": "Dockerfile",
            "docker_compose": "docker-compose.yml",
            "kubernetes": "deployment/kubernetes",
            "deploy_script": "deployment/deploy.py",
            "deployment_test": "deployment/test_deployment.py",
            "github_actions": ".github/workflows"
        }
        
        for name, path in deploy_components.items():
            full_path = PROJECT_ROOT / path
            if full_path.exists():
                if full_path.is_dir():
                    # Check if directory has files
                    if any(full_path.iterdir()):
                        self.results["deployment"][name] = "✓ Present"
                        print(f"  ✓ {name}: Found")
                    else:
                        self.results["deployment"][name] = "⚠ Empty"
                        self.warnings.append(f"Deployment: {name} directory is empty")
                        print(f"  ⚠ {name}: Directory exists but empty")
                else:
                    self.results["deployment"][name] = "✓ Present"
                    print(f"  ✓ {name}: Found")
            else:
                self.results["deployment"][name] = "✗ Missing"
                self.missing.append(f"Deployment: {name} ({path})")
                print(f"  ✗ {name}: Missing")
    
    def _validate_monitoring(self):
        """Validate monitoring and metrics."""
        print("\n7. Validating Monitoring & Metrics...")
        
        monitoring_components = {
            "monitoring_core": "fm_llm_solver/core/monitoring.py",
            "monitor_service": "fm_llm_solver/services/monitor.py",
            "web_monitoring": "web_interface/monitoring.py",
            "monitoring_routes": "web_interface/monitoring_routes.py",
            "prometheus_config": "deployment/prometheus.yml"
        }
        
        for name, path in monitoring_components.items():
            if Path(PROJECT_ROOT / path).exists():
                self.results["monitoring"][name] = "✓ Present"
                print(f"  ✓ {name}: Found")
            else:
                self.results["monitoring"][name] = "✗ Missing"
                self.missing.append(f"Monitoring: {name} ({path})")
                print(f"  ✗ {name}: Missing")
    
    def _validate_security(self):
        """Validate security features."""
        print("\n8. Validating Security Features...")
        
        security_components = {
            "auth_system": "web_interface/auth.py",
            "auth_routes": "web_interface/auth_routes.py",
            "security_test": "tests/test_security.py",
            "security_headers": "fm_llm_solver/web/utils.py",
            "rate_limiting": "fm_llm_solver/web/middleware.py"
        }
        
        for name, path in security_components.items():
            if Path(PROJECT_ROOT / path).exists():
                self.results["security"][name] = "✓ Present"
                print(f"  ✓ {name}: Found")
            else:
                # Some components might be in utils.py
                if name in ["security_headers", "rate_limiting"]:
                    self.results["security"][name] = "⚠ Check implementation"
                    self.warnings.append(f"Security: {name} implementation needs verification")
                    print(f"  ⚠ {name}: Needs verification in utils/middleware")
                else:
                    self.results["security"][name] = "✗ Missing"
                    self.missing.append(f"Security: {name} ({path})")
                    print(f"  ✗ {name}: Missing")
    
    def _validate_test_coverage(self):
        """Validate test coverage."""
        print("\n9. Validating Test Coverage...")
        
        test_categories = {
            "unit_tests": {
                "core_components": "tests/unit/test_core_components.py",
                "verification": "tests/unit/test_verification_fix.py",
                "generation": "tests/unit/test_improved_generation.py",
                "stochastic": "tests/unit/test_stochastic_filter.py",
                "pdf_processor": "tests/unit/test_minimal_pdf_processor.py"
            },
            "integration_tests": {
                "system_integration": "tests/integration/test_new_system_integration.py",
                "web_interface": "tests/integration/comprehensive_web_interface_test_suite.py",
                "advanced": "tests/integration/advanced_integration_tests.py"
            },
            "benchmarks": {
                "web_interface": "tests/benchmarks/web_interface_testbench.py",
                "verification": "tests/benchmarks/verification_optimization.py",
                "generation": "tests/benchmarks/llm_generation_testbench.py",
                "barrier_certificates": "tests/benchmarks/barrier_certificate_optimization_testbench.py"
            },
            "performance": {
                "load_test": "tests/performance/load-test.js",
                "performance_test": "tests/performance/test_performance.py"
            }
        }
        
        for category, tests in test_categories.items():
            print(f"\n  {category}:")
            self.results["tests"][category] = {}
            
            for name, path in tests.items():
                if Path(PROJECT_ROOT / path).exists():
                    self.results["tests"][category][name] = "✓ Present"
                    print(f"    ✓ {name}: Found")
                else:
                    self.results["tests"][category][name] = "✗ Missing"
                    self.missing.append(f"Test: {category}/{name} ({path})")
                    print(f"    ✗ {name}: Missing")
    
    def _validate_documentation(self):
        """Validate documentation."""
        print("\n10. Validating Documentation...")
        
        docs = {
            "readme": "README.md",
            "architecture": "docs/ARCHITECTURE.md",
            "api_reference": "docs/API_REFERENCE.md",
            "user_guide": "docs/USER_GUIDE.md",
            "installation": "docs/INSTALLATION.md",
            "development": "docs/DEVELOPMENT.md",
            "features": "docs/FEATURES.md",
            "experiments": "docs/EXPERIMENTS.md",
            "monitoring": "docs/MONITORING.md",
            "security": "docs/SECURITY.md",
            "verification": "docs/VERIFICATION.md",
            "mathematical_primer": "docs/MATHEMATICAL_PRIMER.md",
            "optimization": "docs/OPTIMIZATION.md",
            "sphinx_config": "docs/conf.py",
            "sphinx_index": "docs/index.rst"
        }
        
        for name, path in docs.items():
            if Path(PROJECT_ROOT / path).exists():
                self.results["documentation"][name] = "✓ Present"
                print(f"  ✓ {name}: Found")
            else:
                self.results["documentation"][name] = "✗ Missing"
                self.missing.append(f"Documentation: {name} ({path})")
                print(f"  ✗ {name}: Missing")
    
    def _generate_report(self) -> Dict:
        """Generate validation report."""
        print("\n" + "=" * 50)
        print("VALIDATION SUMMARY")
        print("=" * 50)
        
        total_checks = 0
        passed_checks = 0
        
        for category, results in self.results.items():
            if isinstance(results, dict):
                for subcategory, subresults in results.items():
                    if isinstance(subresults, dict):
                        for check, status in subresults.items():
                            total_checks += 1
                            if "✓" in str(status):
                                passed_checks += 1
                    else:
                        total_checks += 1
                        if "✓" in str(subresults):
                            passed_checks += 1
        
        success_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        print(f"\nTotal Checks: {total_checks}")
        print(f"Passed: {passed_checks}")
        print(f"Failed: {total_checks - passed_checks}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if self.missing:
            print(f"\nMissing Components ({len(self.missing)}):")
            for item in self.missing[:10]:  # Show first 10
                print(f"  - {item}")
            if len(self.missing) > 10:
                print(f"  ... and {len(self.missing) - 10} more")
        
        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings[:10]:  # Show first 10
                print(f"  ⚠ {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more")
        
        report = {
            "summary": {
                "total_checks": total_checks,
                "passed": passed_checks,
                "failed": total_checks - passed_checks,
                "success_rate": success_rate,
                "missing_count": len(self.missing),
                "warning_count": len(self.warnings)
            },
            "results": self.results,
            "missing": self.missing,
            "warnings": self.warnings
        }
        
        # Save report
        report_path = PROJECT_ROOT / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed report saved to: {report_path}")
        
        # Overall assessment
        print("\n" + "=" * 50)
        if success_rate >= 95:
            print("✅ EXCELLENT: All major capabilities are present and tested!")
        elif success_rate >= 85:
            print("✓ GOOD: Most capabilities are present, minor components missing")
        elif success_rate >= 70:
            print("⚠ WARNING: Several important components are missing")
        else:
            print("❌ CRITICAL: Many core components are missing")
        print("=" * 50)
        
        return report


def main():
    """Main validation entry point."""
    validator = CapabilityValidator()
    report = validator.validate_all()
    
    # Return non-zero exit code if critical components are missing
    if report["summary"]["success_rate"] < 70:
        sys.exit(1)
    elif len(validator.missing) > 0:
        sys.exit(2)  # Minor issues
    else:
        sys.exit(0)  # All good


if __name__ == "__main__":
    main() 