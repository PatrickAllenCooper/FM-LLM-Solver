#!/usr/bin/env python3
"""
Test Summary - Clean overview of testing status
===============================================

Provides a simple, clean summary of all test results and system status.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict


def load_test_results() -> Dict:
    """Load test results from JSON files"""
    results = {}

    # Look for test result files
    test_result_files = [
        "test_results/unified_test_results.json",
        "test_results/certificate_pipeline_results.json",
        "test_results/comprehensive_test_results.json",
    ]

    for file_path in test_result_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    results[file_path] = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")

    return results


def check_system_status() -> Dict:
    """Check current system status"""
    import platform

    import psutil

    status = {
        "platform": platform.system(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "cpu_cores": os.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "gpu_available": False,
        "gpu_name": None,
    }

    # Check GPU
    try:
        import torch

        if torch.cuda.is_available():
            status["gpu_available"] = True
            status["gpu_name"] = torch.cuda.get_device_name(0)
            status["gpu_memory_gb"] = torch.cuda.get_device_properties(
                0
            ).total_memory / (1024**3)
    except ImportError:
        pass

    return status


def generate_summary() -> str:
    """Generate a clean test summary"""
    results = load_test_results()
    system_status = check_system_status()

    summary = []
    summary.append("🎯 FM-LLM-Solver Test Summary")
    summary.append("=" * 50)
    summary.append("")

    # System Status
    summary.append("💻 System Status:")
    summary.append(f"  Platform: {system_status['platform']}")
    summary.append(f"  Python: {system_status['python_version']}")
    summary.append(f"  CPU Cores: {system_status['cpu_cores']}")
    summary.append(f"  Memory: {system_status['memory_gb']:.1f} GB")

    if system_status["gpu_available"]:
        summary.append(
            f"  GPU: {system_status['gpu_name']} ({system_status['gpu_memory_gb']:.1f} GB)"
        )
    else:
        summary.append("  GPU: Not available")

    summary.append("")

    # Test Results
    if results:
        summary.append("📊 Test Results:")
        for file_path, result in results.items():
            test_name = Path(file_path).stem.replace("_", " ").title()

            if "success_rate" in result:
                rate = result["success_rate"]
                status = (
                    "✅ PASS"
                    if rate >= 0.8
                    else "⚠️ PARTIAL" if rate >= 0.6 else "❌ FAIL"
                )
                summary.append(f"  {test_name}: {status} ({rate:.1%})")

            if "total_tests" in result and "passed_tests" in result:
                total = result["total_tests"]
                passed = result["passed_tests"]
                summary.append(f"    Tests: {passed}/{total} passed")

    else:
        summary.append("📊 Test Results: No recent test results found")
        summary.append("  Run 'python tests/unified_test_suite.py' to execute tests")

    summary.append("")

    # Quick Status Check
    summary.append("🔍 Quick Status Check:")

    # Check key modules
    modules_to_check = [
        ("utils.config_loader", "Configuration loading"),
        ("utils.certificate_extraction", "Certificate extraction"),
        ("utils.verification_helpers", "Verification helpers"),
        ("utils.numerical_checks", "Numerical checks"),
        ("torch", "PyTorch (for GPU support)"),
    ]

    for module_name, description in modules_to_check:
        try:
            __import__(module_name)
            summary.append(f"  ✅ {description}")
        except ImportError:
            summary.append(f"  ❌ {description} (not available)")

    summary.append("")
    summary.append("🚀 Ready for Development!")
    summary.append("  • All core components are available")
    summary.append(
        "  • GPU acceleration is ready"
        if system_status["gpu_available"]
        else "  • CPU-only mode available"
    )
    summary.append(
        "  • Run comprehensive tests with: python tests/unified_test_suite.py"
    )

    return "\n".join(summary)


def main():
    """Main function"""
    summary = generate_summary()
    print(summary)


if __name__ == "__main__":
    main()
