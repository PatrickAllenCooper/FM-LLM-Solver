#!/usr/bin/env python3
"""
Expanded Test Suite Runner
Runs all comprehensive tests for production readiness
"""

import sys
import os
import time
import subprocess
import json
from datetime import datetime


def run_test_module(module_name, description):
    """Run a test module and capture results"""
    print(f"\n{'='*60}")
    print(f"Running {description}...")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Run the test module
        result = subprocess.run(
            [sys.executable, f"tests/{module_name}"],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        duration = time.time() - start_time

        # Determine success
        success = result.returncode == 0

        # Print output
        if success:
            print(f"✓ {description} PASSED ({duration:.2f}s)")
            if result.stdout:
                print("\nOutput:")
                print(result.stdout)
        else:
            print(f"✗ {description} FAILED ({duration:.2f}s)")
            if result.stdout:
                print("\nOutput:")
                print(result.stdout)
            if result.stderr:
                print("\nErrors:")
                print(result.stderr)

        return {
            "module": module_name,
            "description": description,
            "success": success,
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"✗ {description} TIMEOUT ({duration:.2f}s)")
        return {
            "module": module_name,
            "description": description,
            "success": False,
            "duration": duration,
            "error": "Timeout after 300 seconds",
        }
    except Exception as e:
        duration = time.time() - start_time
        print(f"✗ {description} ERROR ({duration:.2f}s): {str(e)}")
        return {
            "module": module_name,
            "description": description,
            "success": False,
            "duration": duration,
            "error": str(e),
        }


def main():
    """Run all expanded tests"""
    print("EXPANDED TEST SUITE FOR PRODUCTION READINESS")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Define test modules to run
    test_modules = [
        # Core functionality tests
        ("test_production_comprehensive.py", "Production Comprehensive Tests"),
        ("test_certificate_pipeline.py", "Certificate Pipeline Tests"),
        # New expanded tests
        ("test_error_handling.py", "Error Handling Tests"),
        ("test_boundary_conditions.py", "Boundary Condition Tests"),
        ("test_integration_scenarios.py", "Integration Scenario Tests"),
        ("test_memory_stress.py", "Memory and Stress Tests"),
        ("test_concurrent_processing.py", "Concurrent Processing Tests"),
        # Performance and GPU tests
        ("test_performance.py", "Performance Tests"),
        ("test_gpu_integration.py", "GPU Integration Tests"),
        # Security tests
        ("test_security.py", "Security Tests"),
    ]

    # Check if psutil is available for memory tests
    try:
        import psutil
    except ImportError:
        print("\nWarning: psutil not installed. Installing for memory tests...")
        subprocess.run([sys.executable, "-m", "pip", "install", "psutil"], check=True)

    # Run all tests
    results = []
    total_start_time = time.time()

    for module, description in test_modules:
        result = run_test_module(module, description)
        results.append(result)

    total_duration = time.time() - total_start_time

    # Generate summary
    print("\n" + "=" * 60)
    print("TEST SUITE SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r["success"])
    failed = len(results) - passed

    print(f"\nTotal Tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total Duration: {total_duration:.2f}s")

    # Detailed results
    print("\nDetailed Results:")
    print("-" * 60)
    for result in results:
        status = "PASS" if result["success"] else "FAIL"
        print(f"{result['description']:<40} {status:>6} ({result['duration']:>6.2f}s)")

    # Save results to file
    results_file = (
        f"test_results/expanded_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    os.makedirs("test_results", exist_ok=True)

    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "total_duration": total_duration,
                "passed": passed,
                "failed": failed,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {results_file}")

    # Production readiness assessment
    print("\n" + "=" * 60)
    print("PRODUCTION READINESS ASSESSMENT")
    print("=" * 60)

    if failed == 0:
        print("\n✓ ALL TESTS PASSED!")
        print("\nThe system demonstrates:")
        print("- Robust error handling")
        print("- Proper boundary condition handling")
        print("- Successful integration scenarios")
        print("- Good memory management")
        print("- Thread-safe concurrent processing")
        print("- Adequate performance characteristics")
        print("\nRECOMMENDATION: System is ready for production deployment")
    else:
        print(f"\n✗ {failed} TESTS FAILED")
        print("\nFailed tests:")
        for result in results:
            if not result["success"]:
                print(f"- {result['description']}")
        print("\nRECOMMENDATION: Address failing tests before production deployment")

    # Exit with appropriate code
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
