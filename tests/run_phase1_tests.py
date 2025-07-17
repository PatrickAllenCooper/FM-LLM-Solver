#!/usr/bin/env python3
"""
Automated Test Runner for Phase 1 Barrier Certificate Validation
Runs all Phase 1 tests and generates comprehensive reports
"""

import sys
import os
import subprocess
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules
from test_harness import BarrierCertificateTestHarness
from tests.test_theory_compliance import run_theory_compliance_tests
from tests.integration.test_validation_pipeline import ValidationPipelineIntegrationTests
from tests.unit.test_extraction_edge_cases import (
    TestDecimalExtraction,
    TestTemplateDetection,
    TestFormatSupport,
    TestEdgeCases,
)


# Color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


class Phase1TestRunner:
    """Orchestrates all Phase 1 tests"""

    def __init__(self, verbose: bool = False, output_dir: str = "test_results"):
        self.verbose = verbose
        self.output_dir = output_dir
        self.results = {}
        self.start_time = None
        self.end_time = None

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def print_header(self, text: str, color: str = Colors.BLUE):
        """Print a formatted header"""
        print(f"\n{color}{Colors.BOLD}{'='*80}{Colors.RESET}")
        print(f"{color}{Colors.BOLD}{text.center(80)}{Colors.RESET}")
        print(f"{color}{Colors.BOLD}{'='*80}{Colors.RESET}\n")

    def print_status(self, test_name: str, status: str, details: str = ""):
        """Print test status"""
        if status == "PASS":
            color = Colors.GREEN
            symbol = "✓"
        elif status == "FAIL":
            color = Colors.RED
            symbol = "✗"
        elif status == "SKIP":
            color = Colors.YELLOW
            symbol = "○"
        else:
            color = Colors.CYAN
            symbol = "●"

        print(f"{color}{symbol} {test_name:<50} [{status}]{Colors.RESET}", end="")
        if details:
            print(f" - {details}")
        else:
            print()

    def run_unit_tests(self) -> Dict[str, any]:
        """Run all unit tests"""
        self.print_header("UNIT TESTS", Colors.CYAN)
        unit_results = {}

        # Theory compliance tests
        try:
            print("Running theory compliance tests...")
            theory_passed, theory_details = run_theory_compliance_tests()
            unit_results["theory_compliance"] = {"passed": theory_passed, "details": theory_details}
            self.print_status(
                "Theory Compliance",
                "PASS" if theory_passed else "FAIL",
                f"{theory_details['passed']}/{theory_details['total']} passed",
            )
        except Exception as e:
            unit_results["theory_compliance"] = {"passed": False, "error": str(e)}
            self.print_status("Theory Compliance", "FAIL", f"Error: {str(e)[:50]}...")

        # Extraction edge case tests
        extraction_tests = [
            ("Decimal Extraction", TestDecimalExtraction),
            ("Template Detection", TestTemplateDetection),
            ("Format Support", TestFormatSupport),
            ("Edge Cases", TestEdgeCases),
        ]

        for test_name, test_class in extraction_tests:
            try:
                print(f"Running {test_name} tests...")
                test_instance = test_class()
                passed = 0
                failed = 0

                # Run all test methods
                for method_name in dir(test_instance):
                    if method_name.startswith("test_"):
                        try:
                            method = getattr(test_instance, method_name)
                            method()
                            passed += 1
                        except Exception as e:
                            failed += 1
                            if self.verbose:
                                print(f"  {method_name} failed: {e}")

                unit_results[test_name.lower().replace(" ", "_")] = {
                    "passed": failed == 0,
                    "details": {"passed": passed, "failed": failed},
                }
                self.print_status(
                    test_name, "PASS" if failed == 0 else "FAIL", f"{passed}/{passed+failed} passed"
                )

            except Exception as e:
                unit_results[test_name.lower().replace(" ", "_")] = {
                    "passed": False,
                    "error": str(e),
                }
                self.print_status(test_name, "FAIL", f"Error: {str(e)[:50]}...")

        return unit_results

    def run_integration_tests(self) -> Dict[str, any]:
        """Run integration tests"""
        self.print_header("INTEGRATION TESTS", Colors.MAGENTA)
        integration_results = {}

        try:
            print("Running validation pipeline integration tests...")
            test_suite = ValidationPipelineIntegrationTests()
            test_suite.setUp()

            test_methods = [
                "test_basic_validation_flow",
                "test_level_set_computation",
                "test_lie_derivative_validation",
                "test_pipeline_with_set_membership",
                "test_adaptive_tolerance_integration",
            ]

            passed = 0
            failed = 0

            for method_name in test_methods:
                if hasattr(test_suite, method_name):
                    try:
                        method = getattr(test_suite, method_name)
                        method()
                        passed += 1
                        if self.verbose:
                            print(f"  ✓ {method_name}")
                    except Exception as e:
                        failed += 1
                        if self.verbose:
                            print(f"  ✗ {method_name}: {e}")

            integration_results["validation_pipeline"] = {
                "passed": failed == 0,
                "details": {"passed": passed, "failed": failed, "total": len(test_methods)},
            }

            self.print_status(
                "Validation Pipeline",
                "PASS" if failed == 0 else "FAIL",
                f"{passed}/{len(test_methods)} passed",
            )

        except Exception as e:
            integration_results["validation_pipeline"] = {"passed": False, "error": str(e)}
            self.print_status("Validation Pipeline", "FAIL", f"Error: {str(e)[:50]}...")

        return integration_results

    def run_ground_truth_tests(self) -> Dict[str, any]:
        """Run ground truth barrier certificate tests"""
        self.print_header("GROUND TRUTH TESTS", Colors.YELLOW)

        try:
            print("Running barrier certificate test harness...")
            harness = BarrierCertificateTestHarness()

            # Run tests
            harness.run_all_tests()

            # Generate reports
            report_file = os.path.join(self.output_dir, "harness_report.txt")
            json_file = os.path.join(self.output_dir, "harness_results.json")
            harness.generate_report(report_file)
            harness.export_results_json(json_file)

            # Calculate statistics
            passed = sum(1 for r in harness.results if r.correct == True)
            failed = sum(1 for r in harness.results if r.correct == False)
            errors = sum(1 for r in harness.results if r.correct is None)
            total = len(harness.results)

            ground_truth_results = {
                "passed": failed == 0 and errors == 0,
                "details": {
                    "passed": passed,
                    "failed": failed,
                    "errors": errors,
                    "total": total,
                    "report_file": report_file,
                    "json_file": json_file,
                },
            }

            self.print_status(
                "Ground Truth Tests",
                "PASS" if ground_truth_results["passed"] else "FAIL",
                f"{passed}/{total} passed, {failed} failed, {errors} errors",
            )

            # Print some specific results if verbose
            if self.verbose and (failed > 0 or errors > 0):
                print("\nFailed/Error tests:")
                for r in harness.results:
                    if r.correct == False:
                        print(
                            f"  ✗ {r.test_id}: Expected {r.expected_valid}, "
                            f"Got {r.new_validator_result}"
                        )
                    elif r.correct is None:
                        print(f"  ! {r.test_id}: Error in validation")

            return {"ground_truth": ground_truth_results}

        except Exception as e:
            traceback.print_exc()
            return {"ground_truth": {"passed": False, "error": str(e)}}

    def run_performance_tests(self) -> Dict[str, any]:
        """Run basic performance tests"""
        self.print_header("PERFORMANCE TESTS", Colors.GREEN)
        perf_results = {}

        try:
            # Test validation speed on a simple case
            from utils.level_set_tracker import BarrierCertificateValidator
            from omegaconf import DictConfig

            print("Testing validation performance...")

            # Simple test case
            certificate = "x**2 + y**2 - 1.0"
            system_info = {
                "variables": ["x", "y"],
                "dynamics": ["-x", "-y"],
                "initial_set_conditions": ["x**2 + y**2 <= 0.25"],
                "unsafe_set_conditions": ["x**2 + y**2 >= 4.0"],
                "safe_set_conditions": [],
                "sampling_bounds": {"x": (-3, 3), "y": (-3, 3)},
            }

            config = DictConfig(
                {
                    "numerical_tolerance": 1e-6,
                    "num_samples_boundary": 5000,
                    "num_samples_lie": 10000,
                    "optimization_maxiter": 100,
                    "optimization_popsize": 30,
                }
            )

            # Time multiple runs
            times = []
            for i in range(3):
                start = time.time()
                validator = BarrierCertificateValidator(certificate, system_info, config)
                result = validator.validate()
                elapsed = time.time() - start
                times.append(elapsed)

            avg_time = sum(times) / len(times)

            perf_results["validation_speed"] = {
                "passed": avg_time < 5.0,  # Should complete in < 5 seconds
                "avg_time": avg_time,
                "times": times,
            }

            self.print_status(
                "Validation Speed",
                "PASS" if perf_results["validation_speed"]["passed"] else "FAIL",
                f"Avg: {avg_time:.3f}s",
            )

        except Exception as e:
            perf_results["validation_speed"] = {"passed": False, "error": str(e)}
            self.print_status("Validation Speed", "FAIL", f"Error: {str(e)[:50]}...")

        return perf_results

    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        report_file = os.path.join(self.output_dir, "phase1_test_summary.txt")

        with open(report_file, "w") as f:
            f.write("PHASE 1 TEST SUMMARY REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Duration: {self.end_time - self.start_time:.2f} seconds\n\n")

            # Overall summary
            all_passed = all(
                result.get("passed", False)
                for category in self.results.values()
                for result in category.values()
            )

            f.write(f"Overall Result: {'PASS' if all_passed else 'FAIL'}\n\n")

            # Category summaries
            for category, results in self.results.items():
                f.write(f"\n{category.upper()}\n")
                f.write("-" * 40 + "\n")

                for test_name, result in results.items():
                    status = "PASS" if result.get("passed", False) else "FAIL"
                    f.write(f"{test_name:<30} {status}\n")

                    if "details" in result:
                        details = result["details"]
                        if isinstance(details, dict):
                            for key, value in details.items():
                                if key not in ["report_file", "json_file"]:
                                    f.write(f"  {key}: {value}\n")

                    if "error" in result:
                        f.write(f"  Error: {result['error']}\n")

        print(f"\nSummary report saved to: {report_file}")

        # Also save JSON results
        json_file = os.path.join(self.output_dir, "phase1_test_results.json")
        with open(json_file, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "duration": self.end_time - self.start_time,
                    "overall_passed": all_passed,
                    "results": self.results,
                },
                f,
                indent=2,
            )

        print(f"JSON results saved to: {json_file}")

    def run_all_tests(self):
        """Run all Phase 1 tests"""
        self.start_time = time.time()

        self.print_header("PHASE 1 TEST RUNNER", Colors.BOLD)
        print(f"Output directory: {self.output_dir}")
        print(f"Verbose mode: {self.verbose}")

        # Run test categories
        self.results["unit_tests"] = self.run_unit_tests()
        self.results["integration_tests"] = self.run_integration_tests()
        self.results["ground_truth_tests"] = self.run_ground_truth_tests()
        self.results["performance_tests"] = self.run_performance_tests()

        self.end_time = time.time()

        # Generate summary
        self.generate_summary_report()

        # Print final summary
        self.print_header("TEST SUMMARY", Colors.BOLD)

        total_tests = 0
        passed_tests = 0

        for category, results in self.results.items():
            for test_name, result in results.items():
                total_tests += 1
                if result.get("passed", False):
                    passed_tests += 1

        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Duration: {self.end_time - self.start_time:.2f} seconds")

        if success_rate == 100:
            print(f"\n{Colors.GREEN}{Colors.BOLD}ALL TESTS PASSED!{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}SOME TESTS FAILED{Colors.RESET}")

        return success_rate == 100


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run Phase 1 barrier certificate tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "-o", "--output", default="test_results", help="Output directory for results"
    )
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")

    args = parser.parse_args()

    # Disable colors if requested
    if args.no_color:
        for attr in dir(Colors):
            if not attr.startswith("_"):
                setattr(Colors, attr, "")

    # Run tests
    runner = Phase1TestRunner(verbose=args.verbose, output_dir=args.output)
    success = runner.run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
