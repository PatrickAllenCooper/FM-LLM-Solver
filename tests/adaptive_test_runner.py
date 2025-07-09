#!/usr/bin/env python3
"""
Adaptive Test Runner for FM-LLM Solver.

Automatically detects the environment and runs the most appropriate
test suite based on system capabilities and constraints.

Usage:
    python tests/adaptive_test_runner.py
    python tests/adaptive_test_runner.py --environment macbook
    python tests/adaptive_test_runner.py --scope essential
    python tests/adaptive_test_runner.py --verbose
"""

import os
import sys
import time
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime
import concurrent.futures
import resource
import psutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fm_llm_solver.core.environment_detector import get_environment_detector


class AdaptiveTestRunner:
    """Adaptive test runner that adjusts strategy based on environment."""
    
    def __init__(self, force_environment: Optional[str] = None):
        """Initialize the adaptive test runner."""
        self.detector = get_environment_detector()
        self.environment_type = force_environment or self.detector.get_environment_type()
        self.capabilities = self.detector.get_testing_capabilities()
        
        # Override capabilities if environment is forced
        if force_environment:
            self._adjust_capabilities_for_forced_environment(force_environment)
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "environment_type": self.environment_type,
            "environment_summary": self.detector.get_summary(),
            "test_strategy": {},
            "test_results": {},
            "performance_metrics": {},
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        self.start_time = time.time()
        
    def _adjust_capabilities_for_forced_environment(self, env_type: str):
        """Adjust capabilities when environment is manually specified."""
        if env_type == "macbook":
            self.capabilities.update({
                "recommended_test_scope": "essential",
                "max_parallel_jobs": min(self.capabilities["max_parallel_jobs"], 4),
                "can_run_load_tests": False,
                "timeout_multiplier": max(self.capabilities["timeout_multiplier"], 2.0)
            })
        elif env_type == "desktop":
            self.capabilities.update({
                "recommended_test_scope": "comprehensive",
                "can_run_load_tests": True,
                "timeout_multiplier": min(self.capabilities["timeout_multiplier"], 1.5)
            })
        elif env_type == "deployed":
            self.capabilities.update({
                "recommended_test_scope": "production",
                "max_parallel_jobs": min(self.capabilities["max_parallel_jobs"], 2),
                "timeout_multiplier": max(self.capabilities["timeout_multiplier"], 2.0)
            })
    
    def run_adaptive_tests(self, scope_override: Optional[str] = None, 
                          include_categories: Optional[Set[str]] = None,
                          exclude_categories: Optional[Set[str]] = None) -> Dict:
        """Run tests adaptively based on environment."""
        
        print("🔍 FM-LLM Solver Adaptive Test Runner")
        print("=" * 60)
        print(f"📍 {self.detector.get_summary()}")
        print("=" * 60)
        
        # Determine test strategy
        test_scope = scope_override or self.capabilities["recommended_test_scope"]
        test_strategy = self._build_test_strategy(test_scope, include_categories, exclude_categories)
        self.results["test_strategy"] = test_strategy
        
        print(f"\n🎯 Test Strategy: {test_scope}")
        print(f"📋 Categories: {', '.join(test_strategy['categories'])}")
        print(f"⚡ Parallel Jobs: {test_strategy['parallel_jobs']}")
        print(f"⏱️  Timeout Multiplier: {test_strategy['timeout_multiplier']:.1f}x")
        
        # Run tests by category
        total_passed = 0
        total_tests = 0
        
        for category in test_strategy["categories"]:
            print(f"\n🧪 Running {category} tests...")
            
            category_results = self._run_test_category(
                category, 
                test_strategy["parallel_jobs"],
                test_strategy["timeout_multiplier"]
            )
            
            self.results["test_results"][category] = category_results
            
            passed = category_results.get("passed", 0)
            total = category_results.get("total", 0)
            total_passed += passed
            total_tests += total
            
            # Display results
            if total > 0:
                success_rate = (passed / total) * 100
                status = "✅" if success_rate >= 90 else "⚠️" if success_rate >= 70 else "❌"
                print(f"  {status} {category}: {passed}/{total} ({success_rate:.1f}%)")
                
                if category_results.get("duration"):
                    print(f"    Duration: {category_results['duration']:.2f}s")
                if category_results.get("warnings"):
                    for warning in category_results["warnings"][:3]:  # Show first 3 warnings
                        print(f"    ⚠️ {warning}")
            else:
                print(f"  ❌ {category}: No tests run")
        
        # Calculate final results
        self._finalize_results(total_passed, total_tests)
        
        # Generate report
        self._generate_adaptive_report()
        
        return self.results
    
    def _build_test_strategy(self, scope: str, include: Optional[Set[str]], exclude: Optional[Set[str]]) -> Dict:
        """Build test strategy based on scope and environment."""
        
        # Define test categories by scope
        scope_categories = {
            "essential": [
                "unit_tests",
                "core_integration", 
                "basic_security"
            ],
            "comprehensive": [
                "unit_tests",
                "integration_tests",
                "security_tests",
                "performance_tests"
            ],
            "production": [
                "unit_tests",
                "integration_tests", 
                "security_tests",
                "deployment_tests",
                "monitoring_tests"
            ]
        }
        
        # Add GPU tests if available
        if self.capabilities["can_run_gpu_tests"]:
            scope_categories["comprehensive"].append("gpu_tests")
            scope_categories["production"].append("gpu_tests")
        
        # Add load tests if supported
        if self.capabilities["can_run_load_tests"]:
            scope_categories["comprehensive"].append("load_tests")
            scope_categories["production"].append("load_tests")
        
        # Add end-to-end tests if supported
        if self.capabilities["can_run_end_to_end_tests"]:
            scope_categories["comprehensive"].append("end_to_end_tests")
            scope_categories["production"].append("end_to_end_tests")
        
        categories = scope_categories.get(scope, scope_categories["essential"])
        
        # Apply include/exclude filters
        if include:
            categories = [cat for cat in categories if cat in include]
        if exclude:
            categories = [cat for cat in categories if cat not in exclude]
        
        return {
            "scope": scope,
            "categories": categories,
            "parallel_jobs": self.capabilities["max_parallel_jobs"],
            "timeout_multiplier": self.capabilities["timeout_multiplier"],
            "memory_constraints": self.capabilities["memory_constraints"]
        }
    
    def _run_test_category(self, category: str, parallel_jobs: int, timeout_multiplier: float) -> Dict:
        """Run tests for a specific category."""
        
        category_start = time.time()
        
        try:
            if category == "unit_tests":
                return self._run_unit_tests(parallel_jobs, timeout_multiplier)
            elif category == "core_integration":
                return self._run_core_integration_tests(parallel_jobs, timeout_multiplier)
            elif category == "integration_tests":
                return self._run_integration_tests(parallel_jobs, timeout_multiplier)
            elif category == "security_tests":
                return self._run_security_tests(timeout_multiplier)
            elif category == "basic_security":
                return self._run_basic_security_tests(timeout_multiplier)
            elif category == "performance_tests":
                return self._run_performance_tests(timeout_multiplier)
            elif category == "gpu_tests":
                return self._run_gpu_tests(timeout_multiplier)
            elif category == "load_tests":
                return self._run_load_tests(timeout_multiplier)
            elif category == "deployment_tests":
                return self._run_deployment_tests(timeout_multiplier)
            elif category == "monitoring_tests":
                return self._run_monitoring_tests(timeout_multiplier)
            elif category == "end_to_end_tests":
                return self._run_end_to_end_tests(timeout_multiplier)
            else:
                return {
                    "passed": 0,
                    "total": 0,
                    "duration": time.time() - category_start,
                    "error": f"Unknown test category: {category}"
                }
                
        except Exception as e:
            return {
                "passed": 0,
                "total": 0,
                "duration": time.time() - category_start,
                "error": f"Failed to run {category}: {str(e)}"
            }
    
    def _run_unit_tests(self, parallel_jobs: int, timeout_multiplier: float) -> Dict:
        """Run unit tests."""
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/unit/",
            "-v",
            "--tb=short",
            f"-n={parallel_jobs}",
            f"--timeout={int(300 * timeout_multiplier)}"
        ]
        
        # Add memory constraints for MacBook
        if self.environment_type == "macbook":
            cmd.extend(["--maxfail=10", "-x"])  # Stop on first 10 failures
        
        return self._execute_pytest_command(cmd, "unit_tests")
    
    def _run_core_integration_tests(self, parallel_jobs: int, timeout_multiplier: float) -> Dict:
        """Run core integration tests (subset of full integration)."""
        # Focus on essential integrations only
        test_files = [
            "tests/integration/test_new_system_integration.py::TestSystemIntegration::test_config_manager_integration",
            "tests/integration/test_new_system_integration.py::TestSystemIntegration::test_certificate_generator_integration",
            "tests/integration/test_new_system_integration.py::TestSystemIntegration::test_parser_integration"
        ]
        
        cmd = [
            sys.executable, "-m", "pytest",
            "-v",
            "--tb=short",
            f"--timeout={int(180 * timeout_multiplier)}"
        ] + test_files
        
        return self._execute_pytest_command(cmd, "core_integration")
    
    def _run_integration_tests(self, parallel_jobs: int, timeout_multiplier: float) -> Dict:
        """Run full integration tests."""
        cmd = [
            sys.executable, "-m", "pytest", 
            "tests/integration/",
            "-v",
            "--tb=short",
            f"-n={min(parallel_jobs, 2)}",  # Limit parallel jobs for integration tests
            f"--timeout={int(600 * timeout_multiplier)}"
        ]
        
        return self._execute_pytest_command(cmd, "integration_tests")
    
    def _run_security_tests(self, timeout_multiplier: float) -> Dict:
        """Run security tests."""
        # Run both pytest security tests and custom security audit
        pytest_result = self._execute_pytest_command([
            sys.executable, "-m", "pytest",
            "tests/test_security.py",
            "-v",
            f"--timeout={int(300 * timeout_multiplier)}"
        ], "security_pytest")
        
        # Run security audit script
        audit_result = self._run_security_audit(timeout_multiplier)
        
        # Combine results
        return {
            "passed": pytest_result.get("passed", 0) + (1 if audit_result.get("passed", 0) > 0 else 0),
            "total": pytest_result.get("total", 0) + 1,
            "duration": pytest_result.get("duration", 0) + audit_result.get("duration", 0),
            "details": {
                "pytest": pytest_result,
                "audit": audit_result
            }
        }
    
    def _run_basic_security_tests(self, timeout_multiplier: float) -> Dict:
        """Run basic security tests (subset for MacBook)."""
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/test_security.py::test_authentication",
            "-v",
            f"--timeout={int(120 * timeout_multiplier)}"
        ]
        
        return self._execute_pytest_command(cmd, "basic_security")
    
    def _run_performance_tests(self, timeout_multiplier: float) -> Dict:
        """Run performance tests."""
        if Path("scripts/performance_benchmark.py").exists():
            return self._execute_script("scripts/performance_benchmark.py", timeout_multiplier * 600)
        else:
            # Fallback to pytest performance tests
            cmd = [
                sys.executable, "-m", "pytest",
                "tests/performance/",
                "-v",
                f"--timeout={int(600 * timeout_multiplier)}"
            ]
            return self._execute_pytest_command(cmd, "performance_tests")
    
    def _run_gpu_tests(self, timeout_multiplier: float) -> Dict:
        """Run GPU-specific tests."""
        cmd = [
            sys.executable, "-m", "pytest",
            "-m", "gpu",
            "-v",
            f"--timeout={int(300 * timeout_multiplier)}"
        ]
        
        return self._execute_pytest_command(cmd, "gpu_tests")
    
    def _run_load_tests(self, timeout_multiplier: float) -> Dict:
        """Run load tests using K6."""
        if not Path("tests/performance/load-test.js").exists():
            return {"passed": 0, "total": 1, "error": "Load test script not found"}
        
        # Check if K6 is available
        try:
            subprocess.run(["k6", "version"], capture_output=True, check=True, timeout=10)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {"passed": 0, "total": 1, "error": "K6 not installed"}
        
        # Run load test
        try:
            start_time = time.time()
            result = subprocess.run([
                "k6", "run",
                "--vus", "10",
                "--duration", f"{int(30 * timeout_multiplier)}s",
                "tests/performance/load-test.js"
            ], capture_output=True, text=True, timeout=120 * timeout_multiplier)
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                return {"passed": 1, "total": 1, "duration": duration}
            else:
                return {"passed": 0, "total": 1, "duration": duration, "error": result.stderr}
                
        except subprocess.TimeoutExpired:
            return {"passed": 0, "total": 1, "error": "Load test timed out"}
    
    def _run_deployment_tests(self, timeout_multiplier: float) -> Dict:
        """Run deployment tests."""
        if Path("deployment/test_deployment.py").exists():
            return self._execute_script("deployment/test_deployment.py", timeout_multiplier * 300)
        else:
            return {"passed": 0, "total": 1, "error": "Deployment test script not found"}
    
    def _run_monitoring_tests(self, timeout_multiplier: float) -> Dict:
        """Run monitoring tests."""
        # Test monitoring endpoints and metrics
        cmd = [
            sys.executable, "-m", "pytest",
            "-k", "monitoring or metrics or health",
            "-v",
            f"--timeout={int(180 * timeout_multiplier)}"
        ]
        
        return self._execute_pytest_command(cmd, "monitoring_tests")
    
    def _run_end_to_end_tests(self, timeout_multiplier: float) -> Dict:
        """Run end-to-end tests."""
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/test_all_capabilities_comprehensive.py::TestSystemIntegration::test_all_capabilities_integrated",
            "-v",
            f"--timeout={int(900 * timeout_multiplier)}"
        ]
        
        return self._execute_pytest_command(cmd, "end_to_end_tests")
    
    def _run_security_audit(self, timeout_multiplier: float) -> Dict:
        """Run security audit script."""
        if Path("scripts/security_audit.py").exists():
            return self._execute_script("scripts/security_audit.py", timeout_multiplier * 300)
        else:
            return {"passed": 0, "total": 1, "error": "Security audit script not found"}
    
    def _execute_pytest_command(self, cmd: List[str], test_name: str) -> Dict:
        """Execute a pytest command and parse results."""
        start_time = time.time()
        
        try:
            # Add output options
            cmd.extend(["--tb=short", "--no-header", "-q"])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=cmd[-1].split("=")[1] if "--timeout=" in " ".join(cmd) else 600,
                cwd=PROJECT_ROOT
            )
            
            duration = time.time() - start_time
            
            # Parse pytest output
            return self._parse_pytest_output(result, duration, test_name)
            
        except subprocess.TimeoutExpired:
            return {
                "passed": 0,
                "total": 0,
                "duration": time.time() - start_time,
                "error": f"{test_name} timed out"
            }
        except Exception as e:
            return {
                "passed": 0,
                "total": 0,
                "duration": time.time() - start_time,
                "error": f"Failed to run {test_name}: {str(e)}"
            }
    
    def _execute_script(self, script_path: str, timeout: float) -> Dict:
        """Execute a Python script and parse results."""
        start_time = time.time()
        
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=PROJECT_ROOT
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                return {"passed": 1, "total": 1, "duration": duration}
            else:
                return {
                    "passed": 0,
                    "total": 1,
                    "duration": duration,
                    "error": result.stderr or "Script failed"
                }
                
        except subprocess.TimeoutExpired:
            return {
                "passed": 0,
                "total": 1,
                "duration": time.time() - start_time,
                "error": "Script timed out"
            }
        except Exception as e:
            return {
                "passed": 0,
                "total": 1,
                "duration": time.time() - start_time,
                "error": str(e)
            }
    
    def _parse_pytest_output(self, result: subprocess.CompletedProcess, duration: float, test_name: str) -> Dict:
        """Parse pytest output to extract test results."""
        output = result.stdout + result.stderr
        
        # Look for pytest summary line
        passed = 0
        failed = 0
        total = 0
        warnings = []
        
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            
            # Parse summary line
            if "passed" in line and ("failed" in line or "error" in line):
                # Format: "X failed, Y passed in Z.ZZs"
                import re
                match = re.search(r'(\d+)\s+failed.*?(\d+)\s+passed', line)
                if match:
                    failed = int(match.group(1))
                    passed = int(match.group(2))
                    total = passed + failed
            elif "passed" in line and "failed" not in line:
                # Format: "X passed in Z.ZZs"
                import re
                match = re.search(r'(\d+)\s+passed', line)
                if match:
                    passed = int(match.group(1))
                    total = passed
            elif "FAILED" in line:
                failed += 1
            elif "PASSED" in line:
                passed += 1
            elif "WARNING" in line or "warning" in line:
                warnings.append(line)
        
        # If we couldn't parse, use return code
        if total == 0:
            if result.returncode == 0:
                passed = 1
                total = 1
            else:
                failed = 1
                total = 1
        
        return {
            "passed": passed,
            "total": total,
            "duration": duration,
            "warnings": warnings[:5],  # Limit warnings
            "raw_output": output if result.returncode != 0 else None
        }
    
    def _finalize_results(self, total_passed: int, total_tests: int):
        """Finalize test results with summary metrics."""
        
        duration = time.time() - self.start_time
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Determine overall status
        if success_rate >= 95:
            status = "EXCELLENT"
        elif success_rate >= 85:
            status = "GOOD"
        elif success_rate >= 70:
            status = "ACCEPTABLE"
        else:
            status = "NEEDS_WORK"
        
        self.results["performance_metrics"] = {
            "total_duration": duration,
            "total_tests": total_tests,
            "total_passed": total_passed,
            "success_rate": success_rate,
            "overall_status": status,
            "tests_per_second": total_tests / duration if duration > 0 else 0
        }
        
        # Add recommendations based on results
        self._generate_recommendations(success_rate, duration)
    
    def _generate_recommendations(self, success_rate: float, duration: float):
        """Generate recommendations based on test results."""
        recommendations = []
        
        if success_rate < 85:
            recommendations.append("Consider running tests individually to isolate failures")
            recommendations.append("Check system dependencies and environment setup")
        
        if self.environment_type == "macbook" and duration > 300:
            recommendations.append("Tests are running slowly on MacBook - consider running on desktop for development")
        
        if self.environment_type == "desktop" and not self.capabilities["can_run_gpu_tests"]:
            recommendations.append("GPU not detected on desktop - install CUDA drivers for better performance")
        
        if self.environment_type == "deployed" and success_rate >= 95:
            recommendations.append("All tests passing in deployment environment - system is production ready")
        
        # Environment-specific recommendations
        if self.environment_type == "macbook":
            recommendations.append("Use 'essential' scope for faster development cycles")
            recommendations.append("Run comprehensive tests on desktop before committing")
        elif self.environment_type == "desktop":
            recommendations.append("Take advantage of GPU acceleration for ML model tests")
            recommendations.append("Run full test suite before deployment")
        else:  # deployed
            recommendations.append("Focus on monitoring and security tests in production")
            recommendations.append("Use minimal test scope for quick health checks")
        
        self.results["recommendations"] = recommendations
    
    def _generate_adaptive_report(self):
        """Generate comprehensive test report."""
        
        print("\n" + "=" * 60)
        print("🎯 ADAPTIVE TEST RESULTS")
        print("=" * 60)
        
        metrics = self.results["performance_metrics"]
        print(f"\n📊 Overall Results:")
        print(f"  Status: {metrics['overall_status']}")
        print(f"  Success Rate: {metrics['success_rate']:.1f}%")
        print(f"  Tests: {metrics['total_passed']}/{metrics['total_tests']}")
        print(f"  Duration: {metrics['total_duration']:.1f}s")
        print(f"  Speed: {metrics['tests_per_second']:.1f} tests/sec")
        
        print(f"\n📋 Test Strategy Used:")
        strategy = self.results["test_strategy"]
        print(f"  Scope: {strategy['scope']}")
        print(f"  Categories: {len(strategy['categories'])}")
        print(f"  Parallel Jobs: {strategy['parallel_jobs']}")
        print(f"  Timeout Multiplier: {strategy['timeout_multiplier']:.1f}x")
        
        print(f"\n🔧 Recommendations:")
        for rec in self.results["recommendations"][:5]:
            print(f"  • {rec}")
        
        # Save detailed results
        report_file = PROJECT_ROOT / "test_results" / f"adaptive_test_report_{int(time.time())}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")


def main():
    """Main entry point for adaptive test runner."""
    parser = argparse.ArgumentParser(description="FM-LLM Solver Adaptive Test Runner")
    parser.add_argument("--environment", choices=["macbook", "desktop", "deployed"], 
                       help="Force specific environment type")
    parser.add_argument("--scope", choices=["essential", "comprehensive", "production"],
                       help="Override test scope")
    parser.add_argument("--include", nargs="+", help="Include specific test categories")
    parser.add_argument("--exclude", nargs="+", help="Exclude specific test categories")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Show strategy without running tests")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    # Create runner
    runner = AdaptiveTestRunner(force_environment=args.environment)
    
    if args.dry_run:
        print("🔍 Test Strategy (Dry Run)")
        print("=" * 40)
        print(f"Environment: {runner.environment_type}")
        print(f"Summary: {runner.detector.get_summary()}")
        
        strategy = runner._build_test_strategy(
            args.scope or runner.capabilities["recommended_test_scope"],
            set(args.include) if args.include else None,
            set(args.exclude) if args.exclude else None
        )
        
        print(f"Scope: {strategy['scope']}")
        print(f"Categories: {', '.join(strategy['categories'])}")
        print(f"Parallel Jobs: {strategy['parallel_jobs']}")
        print(f"Timeout Multiplier: {strategy['timeout_multiplier']:.1f}x")
        return
    
    # Run tests
    try:
        results = runner.run_adaptive_tests(
            scope_override=args.scope,
            include_categories=set(args.include) if args.include else None,
            exclude_categories=set(args.exclude) if args.exclude else None
        )
        
        # Exit with appropriate code
        success_rate = results["performance_metrics"]["success_rate"]
        if success_rate >= 95:
            sys.exit(0)
        elif success_rate >= 85:
            sys.exit(1)  # Minor issues
        else:
            sys.exit(2)  # Major issues
            
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 