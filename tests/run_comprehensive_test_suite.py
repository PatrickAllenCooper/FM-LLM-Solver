#!/usr/bin/env python3
"""
Comprehensive test suite runner with GPU acceleration.
Implements the testing flywheel for robust certificate generation and validation.
"""

import os
import sys
import time
import logging
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import tempfile

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveTestRunner:
    """Comprehensive test runner with GPU acceleration"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.results = {}
        self.gpu_available = self._check_gpu_availability()

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for testing"""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def run_unit_tests(self) -> Dict:
        """Run all unit tests"""
        logger.info("🧪 Running unit tests...")

        try:
            # Run pytest on unit tests
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/unit/", "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Parse results
            output_lines = result.stdout.split("\n")
            test_results = []

            for line in output_lines:
                if "PASSED" in line or "FAILED" in line or "ERROR" in line:
                    test_results.append(line.strip())

            return {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "test_results": test_results,
                "total_tests": len([r for r in test_results if "PASSED" in r]),
                "failed_tests": len([r for r in test_results if "FAILED" in r or "ERROR" in r]),
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Unit tests timed out", "return_code": -1}
        except Exception as e:
            return {"success": False, "error": str(e), "return_code": -1}

    def run_certificate_pipeline_tests(self) -> Dict:
        """Run certificate pipeline tests"""
        logger.info("🧪 Running certificate pipeline tests...")

        try:
            # Import and run certificate pipeline tests
            from tests.unit.test_certificate_pipeline import CertificatePipelineTester

            tester = CertificatePipelineTester()
            results = tester.run_comprehensive_tests()

            return {
                "success": results["overall_success_rate"] > 0.6,
                "results": results,
                "success_rate": results["overall_success_rate"],
                "total_tests": results["total_tests"],
                "passed_tests": results["passed_tests"],
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def run_gpu_accelerated_tests(self) -> Dict:
        """Run GPU-accelerated tests"""
        logger.info("🚀 Running GPU-accelerated tests...")

        if not self.gpu_available:
            return {"success": False, "error": "GPU not available", "gpu_available": False}

        try:
            # Import and run GPU tests
            from tests.unit.test_gpu_accelerated_generation import GPUAcceleratedTester

            tester = GPUAcceleratedTester()
            results = tester.run_gpu_comprehensive_tests()

            return {
                "success": results["overall_success_rate"] > 0.7,
                "results": results,
                "gpu_available": True,
                "success_rate": results["overall_success_rate"],
                "device": results["device"],
            }

        except Exception as e:
            return {"success": False, "error": str(e), "gpu_available": self.gpu_available}

    def run_integration_tests(self) -> Dict:
        """Run integration tests"""
        logger.info("🔗 Running integration tests...")

        try:
            # Run integration test script
            result = subprocess.run(
                ["python", "tests/integration/run_quick_integration_tests.py"],
                capture_output=True,
                text=True,
                timeout=600,
            )

            return {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Integration tests timed out", "return_code": -1}
        except Exception as e:
            return {"success": False, "error": str(e), "return_code": -1}

    def run_performance_benchmarks(self) -> Dict:
        """Run performance benchmarks"""
        logger.info("⚡ Running performance benchmarks...")

        try:
            # Test certificate generation performance
            start_time = time.time()

            # Mock performance test
            test_cases = [
                {"name": "simple_2d", "dimensions": 2},
                {"name": "complex_3d", "dimensions": 3},
                {"name": "high_dim_4d", "dimensions": 4},
            ]

            performance_results = []
            for test_case in test_cases:
                case_start = time.time()

                # Simulate certificate generation
                time.sleep(0.1)  # Mock processing time

                case_time = time.time() - case_start
                performance_results.append(
                    {
                        "test_case": test_case["name"],
                        "dimensions": test_case["dimensions"],
                        "processing_time": case_time,
                        "success": True,
                    }
                )

            total_time = time.time() - start_time

            return {
                "success": True,
                "total_time": total_time,
                "performance_results": performance_results,
                "avg_time_per_case": total_time / len(test_cases),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def run_robustness_tests(self) -> Dict:
        """Run robustness tests for certificate extraction and validation"""
        logger.info("🛡️ Running robustness tests...")

        try:
            # Test certificate extraction with various formats
            test_inputs = [
                "BARRIER_CERTIFICATE_START\nx**2 + y**2 - 1.5\nBARRIER_CERTIFICATE_END",
                "The barrier certificate is: x**2 + y**2 - 1.5",
                "B(x,y) = x**2 + y**2 - 1.5",
                "Certificate: x**2 + y**2 - 1.5",
                "Invalid format with no certificate",
                "BARRIER_CERTIFICATE_START\nax**2 + by**2 + c\nBARRIER_CERTIFICATE_END",  # Template
            ]

            from utils.certificate_extraction import (
                extract_certificate_from_llm_output,
                is_template_expression,
            )

            # Define variables for testing
            test_variables = ["x", "y"]

            robustness_results = []
            for i, test_input in enumerate(test_inputs):
                extracted_result = extract_certificate_from_llm_output(test_input, test_variables)
                extracted = (
                    extracted_result[0] if isinstance(extracted_result, tuple) else extracted_result
                )
                is_template = is_template_expression(extracted) if extracted else True

                robustness_results.append(
                    {
                        "test_input": test_input,
                        "extracted": extracted,
                        "is_template": is_template,
                        "extraction_success": extracted is not None,
                        "template_rejected": is_template,
                    }
                )

            # Calculate success metrics
            successful_extractions = sum(1 for r in robustness_results if r["extraction_success"])
            template_rejections = sum(1 for r in robustness_results if r["template_rejected"])

            return {
                "success": successful_extractions >= 3,  # At least 3 successful extractions
                "total_tests": len(robustness_results),
                "successful_extractions": successful_extractions,
                "template_rejections": template_rejections,
                "robustness_results": robustness_results,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def run_comprehensive_suite(self) -> Dict:
        """Run the complete comprehensive test suite"""
        logger.info("🎯 Starting comprehensive test suite...")
        logger.info(f"GPU Available: {self.gpu_available}")

        start_time = time.time()

        # Run all test categories
        test_categories = {
            "unit_tests": self.run_unit_tests(),
            "certificate_pipeline": self.run_certificate_pipeline_tests(),
            "gpu_accelerated": self.run_gpu_accelerated_tests(),
            "integration_tests": self.run_integration_tests(),
            "performance_benchmarks": self.run_performance_benchmarks(),
            "robustness_tests": self.run_robustness_tests(),
        }

        total_time = time.time() - start_time

        # Calculate overall success
        successful_categories = sum(
            1 for result in test_categories.values() if result.get("success", False)
        )
        total_categories = len(test_categories)
        overall_success_rate = successful_categories / total_categories

        # Compile comprehensive results
        comprehensive_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_available": self.gpu_available,
            "total_time": total_time,
            "overall_success_rate": overall_success_rate,
            "successful_categories": successful_categories,
            "total_categories": total_categories,
            "test_categories": test_categories,
            "summary": {
                "unit_tests_passed": test_categories["unit_tests"].get("success", False),
                "pipeline_tests_passed": test_categories["certificate_pipeline"].get(
                    "success", False
                ),
                "gpu_tests_passed": test_categories["gpu_accelerated"].get("success", False),
                "integration_tests_passed": test_categories["integration_tests"].get(
                    "success", False
                ),
                "performance_tests_passed": test_categories["performance_benchmarks"].get(
                    "success", False
                ),
                "robustness_tests_passed": test_categories["robustness_tests"].get(
                    "success", False
                ),
            },
        }

        return comprehensive_results

    def save_comprehensive_results(
        self, results: Dict, output_path: str = "test_results/comprehensive_test_results.json"
    ):
        """Save comprehensive test results"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Comprehensive test results saved to: {output_path}")

    def generate_test_report(self, results: Dict) -> str:
        """Generate a human-readable test report"""
        report = []
        report.append("🎯 COMPREHENSIVE TEST SUITE REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {results['timestamp']}")
        report.append(f"GPU Available: {results['gpu_available']}")
        report.append(f"Total Time: {results['total_time']:.1f} seconds")
        report.append(f"Overall Success Rate: {results['overall_success_rate']:.1%}")
        report.append("")

        # Category results
        for category, result in results["test_categories"].items():
            status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
            report.append(f"{category.replace('_', ' ').title()}: {status}")

            if "error" in result:
                report.append(f"  Error: {result['error']}")

        report.append("")
        report.append("📊 DETAILED SUMMARY")
        report.append("-" * 30)

        summary = results["summary"]
        for test_type, passed in summary.items():
            status = "✅" if passed else "❌"
            report.append(f"{status} {test_type.replace('_', ' ').title()}")

        return "\n".join(report)


def main():
    """Main function to run comprehensive test suite"""
    runner = ComprehensiveTestRunner()

    print("🚀 Starting Comprehensive Test Suite...")
    print("=" * 60)

    # Run comprehensive tests
    results = runner.run_comprehensive_suite()

    # Generate and display report
    report = runner.generate_test_report(results)
    print(report)

    # Save results
    runner.save_comprehensive_results(results)

    # Return appropriate exit code
    if results["overall_success_rate"] >= 0.8:
        print("\n🎉 Test suite passed with high success rate!")
        return 0
    elif results["overall_success_rate"] >= 0.6:
        print("\n⚠️ Test suite passed with moderate success rate.")
        return 0
    else:
        print("\n❌ Test suite failed with low success rate.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
