#!/usr/bin/env python3
"""
Comprehensive test runner for FM-LLM-Solver.

This script runs various test suites and generates coverage reports.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path


def run_command(cmd, description, ignore_errors=False):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        duration = time.time() - start_time
        print(f"✅ {description} completed successfully in {duration:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"❌ {description} failed after {duration:.2f}s with exit code {e.returncode}")
        if not ignore_errors:
            return False
        print("⚠️ Continuing despite errors...")
        return True


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive tests for FM-LLM-Solver")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests")
    parser.add_argument("--all", action="store_true", help="Run all test suites")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--output-dir", default="test_results", help="Output directory for test results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # If no specific test type is selected, run basic tests
    if not any([args.unit, args.integration, args.performance, args.smoke, args.all]):
        args.unit = True
        args.integration = True
    
    if args.all:
        args.unit = args.integration = args.performance = args.smoke = True
    
    # Ensure we're in the project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Common pytest arguments
    pytest_args = [
        "python", "-m", "pytest",
        "--tb=short",
        f"--junitxml={output_dir}/junit.xml"
    ]
    
    if args.verbose:
        pytest_args.append("-v")
    else:
        pytest_args.append("-q")
    
    if args.parallel:
        pytest_args.extend(["-n", "auto"])
    
    if args.fast:
        pytest_args.extend(["-m", "not slow"])
    
    if args.coverage:
        pytest_args.extend([
            "--cov=fm_llm_solver",
            "--cov-report=html:" + str(output_dir / "coverage_html"),
            "--cov-report=xml:" + str(output_dir / "coverage.xml"),
            "--cov-report=term-missing"
        ])
    
    success = True
    
    # Run unit tests
    if args.unit:
        unit_cmd = pytest_args + [
            "tests/unit/",
            "-m", "unit or not integration and not performance and not smoke"
        ]
        if not run_command(unit_cmd, "Unit Tests"):
            success = False
    
    # Run integration tests
    if args.integration:
        integration_cmd = pytest_args + [
            "tests/integration/",
            "-m", "integration or not unit and not performance and not smoke"
        ]
        if not run_command(integration_cmd, "Integration Tests", ignore_errors=True):
            print("⚠️ Some integration tests failed, but continuing...")
    
    # Run performance tests
    if args.performance:
        performance_cmd = pytest_args + [
            "tests/performance/",
            "-m", "performance or not unit and not integration and not smoke"
        ]
        if not run_command(performance_cmd, "Performance Tests", ignore_errors=True):
            print("⚠️ Some performance tests failed, but continuing...")
    
    # Run smoke tests
    if args.smoke:
        smoke_cmd = pytest_args + [
            "-m", "smoke",
            "tests/"
        ]
        if not run_command(smoke_cmd, "Smoke Tests"):
            success = False
    
    # Run security tests
    security_cmd = [
        "python", "-m", "pytest",
        "tests/test_security.py",
        "-v"
    ]
    if not run_command(security_cmd, "Security Tests", ignore_errors=True):
        print("⚠️ Some security tests failed, but continuing...")
    
    # Run linting
    lint_cmd = ["python", "-m", "flake8", "fm_llm_solver/", "tests/", "--max-line-length=88"]
    if not run_command(lint_cmd, "Code Linting", ignore_errors=True):
        print("⚠️ Linting found issues, but continuing...")
    
    # Run type checking
    type_cmd = ["python", "-m", "mypy", "fm_llm_solver/", "--ignore-missing-imports"]
    if not run_command(type_cmd, "Type Checking", ignore_errors=True):
        print("⚠️ Type checking found issues, but continuing...")
    
    # Generate final report
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    if success:
        print("✅ All critical tests passed!")
    else:
        print("❌ Some critical tests failed!")
    
    print(f"\nTest results saved to: {output_dir.absolute()}")
    
    if args.coverage and (output_dir / "coverage_html" / "index.html").exists():
        print(f"Coverage report: {(output_dir / 'coverage_html' / 'index.html').absolute()}")
    
    # Generate test metrics
    try:
        import xml.etree.ElementTree as ET
        junit_file = output_dir / "junit.xml"
        if junit_file.exists():
            tree = ET.parse(junit_file)
            root = tree.getroot()
            
            total_tests = int(root.get('tests', 0))
            failures = int(root.get('failures', 0))
            errors = int(root.get('errors', 0))
            skipped = int(root.get('skipped', 0))
            time_taken = float(root.get('time', 0))
            
            passed = total_tests - failures - errors - skipped
            success_rate = (passed / total_tests * 100) if total_tests > 0 else 0
            
            print(f"\nTest Metrics:")
            print(f"  Total Tests: {total_tests}")
            print(f"  Passed: {passed}")
            print(f"  Failed: {failures}")
            print(f"  Errors: {errors}")
            print(f"  Skipped: {skipped}")
            print(f"  Success Rate: {success_rate:.1f}%")
            print(f"  Total Time: {time_taken:.2f}s")
    except Exception as e:
        print(f"Could not parse test metrics: {e}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
