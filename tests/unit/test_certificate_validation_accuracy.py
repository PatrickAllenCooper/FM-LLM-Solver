#!/usr/bin/env python3
"""
Test certificate validation accuracy with correct barrier certificate theory
"""

import json
import logging
import os
import sys
import time
from typing import Dict, List

import numpy as np
import sympy

# Add parent directory to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from utils.certificate_extraction import extract_certificate_from_llm_output
from utils.level_set_tracker import LevelSetTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CertificateValidationTester:
    """Tests for certificate validation accuracy with correct theory"""

    def __init__(self):
        self.level_tracker = LevelSetTracker()

    def generate_test_systems(self) -> List[Dict]:
        """Generate test systems with known valid/invalid certificates"""
        systems = []

        # 2D stable linear system
        systems.append(
            {
                "name": "2D Stable Linear",
                "dynamics": ["dx/dt = -x", "dy/dt = -y"],
                "initial_set": ["x**2 + y**2 <= 0.25"],  # r = 0.5
                "unsafe_set": ["x**2 + y**2 >= 4.0"],  # r = 2.0
                "valid_certificates": [
                    "x**2 + y**2 - 1.0",  # Valid: separates at r=1
                    "x**2 + y**2 - 1.5",  # Valid: separates at r=1.22
                    "x**2 + y**2 - 0.75",  # Valid: separates at r=0.87
                    "2*x**2 + 2*y**2 - 2.0",  # Valid: same as first
                    "x**2 + y**2 - 0.3",  # Invalid: too close to initial set
                    "x**2 + y**2 - 0.25",  # Invalid: no separation
                    "x**2 + y**2 - 0.2",  # Invalid: intersects initial set
                    "x**2 + y**2",  # Invalid: B > 0 everywhere
                    "x**2 + y**2 - 5.0",  # Invalid: B < 0 on unsafe set
                ],
                "expected_valid": [
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            }
        )

        # 2D nonlinear system
        systems.append(
            {
                "name": "2D Nonlinear",
                "dynamics": ["dx/dt = -x + x**3", "dy/dt = -y"],
                "initial_set": ["x**2 + y**2 <= 0.25"],
                "unsafe_set": ["x**2 + y**2 >= 4.0"],
                "valid_certificates": [
                    "x**2 + y**2 - 1.0",
                    "x**2 + y**2 - 0.5",
                    "x**2 + y**2 - 0.3",
                ],
                "expected_valid": [
                    True,
                    True,
                    False,
                ],  # Adjusted based on correct theory
            }
        )

        # 3D stable linear system
        systems.append(
            {
                "name": "3D Stable Linear",
                "dynamics": ["dx/dt = -x", "dy/dt = -y", "dz/dt = -z"],
                "initial_set": ["x**2 + y**2 + z**2 <= 0.25"],
                "unsafe_set": ["x**2 + y**2 + z**2 >= 4.0"],
                "valid_certificates": [
                    "x**2 + y**2 + z**2 - 1.0",
                    "x**2 + y**2 + z**2 - 0.5",
                    "x**2 + y**2 + z**2 - 0.3",
                ],
                "expected_valid": [True, True, False],
            }
        )

        return systems

    def validate_certificate_mathematically(
        self, certificate: str, system: Dict, n_samples: int = 10
    ) -> Dict:
        """
        Validate a certificate mathematically using correct barrier certificate theory.

        Correct conditions:
        1. B(x) ≤ c1 for all x in Initial Set
        2. B(x) ≥ c2 for all x in Unsafe Set
        3. c1 < c2 (separation condition)
        4. dB/dt ≤ 0 in the region where B(x) ∈ [c1, c2]
        """
        try:
            # Define symbolic variables
            x, y, z = sympy.symbols("x y z")

            # Determine dimension
            if len(system["dynamics"]) == 3:
                vars_sympy = [x, y, z]
            else:
                vars_sympy = [x, y]

            # Extract variable names
            variables = [str(v) for v in vars_sympy]

            # Step 1: Compute level sets using the level tracker
            level_info = self.level_tracker.compute_level_sets(
                certificate,
                system["initial_set"],
                system["unsafe_set"],
                variables,
                n_samples=max(n_samples * 10, 100),  # Use more samples for accuracy
            )

            # Check if we have valid level sets
            if level_info.initial_samples == 0:
                return {
                    "valid": False,
                    "error": "Could not sample initial set",
                    "certificate": certificate,
                }
            if level_info.unsafe_samples == 0:
                return {
                    "valid": False,
                    "error": "Could not sample unsafe set",
                    "certificate": certificate,
                }

            # Parse certificate expression
            B = sympy.parse_expr(certificate)

            # Parse dynamics
            dynamics = []
            for dyn in system["dynamics"]:
                if "=" in dyn:
                    rhs = dyn.split("=")[1].strip()
                    dynamics.append(sympy.parse_expr(rhs))
                else:
                    dynamics.append(sympy.parse_expr(dyn))

            # Calculate Lie derivative
            lie_derivative = sum(
                sympy.diff(B, var) * f for var, f in zip(vars_sympy, dynamics)
            )

            # Convert expressions to numpy functions
            B_func = sympy.lambdify(vars_sympy, B, "numpy")
            lie_func = sympy.lambdify(vars_sympy, lie_derivative, "numpy")

            # Check barrier conditions
            violations = []

            # Step 2: Check separation condition
            if not level_info.is_valid:
                violations.append(
                    f"Separation violation: c1={level_info.initial_max:.6f} >= c2={level_info.unsafe_min:.6f}"
                )

            # Step 3: Verify conditions on sample points
            # We already know max(B) on initial and min(B) on unsafe from level_info
            # but let's do spot checks for validation

            # Sample and check a few points
            np.random.default_rng(42)
            n_check = min(n_samples * 2, 20)

            # Check initial set condition: B(x) ≤ c1
            initial_samples = self.level_tracker._sample_constrained_set(
                system["initial_set"], variables, n_check
            )
            for point in initial_samples[:10]:  # Check first 10
                B_val = B_func(*point)
                if B_val > level_info.initial_max + 1e-6:
                    violations.append(
                        f"Initial set violation at {point}: B={B_val:.6f} > c1={level_info.initial_max:.6f}"
                    )

            # Check unsafe set condition: B(x) ≥ c2
            unsafe_samples = self.level_tracker._sample_constrained_set(
                system["unsafe_set"], variables, n_check
            )
            for point in unsafe_samples[:10]:  # Check first 10
                B_val = B_func(*point)
                if B_val < level_info.unsafe_min - 1e-6:
                    violations.append(
                        f"Unsafe set violation at {point}: B={B_val:.6f} < c2={level_info.unsafe_min:.6f}"
                    )

            # Step 4: Check Lie derivative in critical region
            # Sample points where B(x) ∈ [c1, c2]
            critical_violations = 0
            critical_samples = 0

            # Generate random points and check if they're in critical region
            bounds = self.level_tracker._estimate_bounds(
                system["initial_set"] + system["unsafe_set"], variables
            )

            for _ in range(n_samples * 10):
                point = tuple(
                    np.random.uniform(bounds[v][0], bounds[v][1]) for v in variables
                )
                B_val = B_func(*point)

                # Check if in critical region [c1, c2]
                if level_info.initial_max <= B_val <= level_info.unsafe_min:
                    critical_samples += 1
                    lie_val = lie_func(*point)

                    if lie_val > 1e-6:  # Should be ≤ 0
                        critical_violations += 1
                        if len(violations) < 10:  # Limit violation reports
                            violations.append(
                                f"Lie derivative violation at {point}: dB/dt={lie_val:.6f} > 0"
                            )

            # If we found critical region samples, check violation rate
            if critical_samples > 0:
                violation_rate = critical_violations / critical_samples
                if violation_rate > 0.05:  # Allow 5% numerical tolerance
                    violations.append(
                        f"High Lie derivative violation rate: {violation_rate:.1%} ({critical_violations}/{critical_samples})"
                    )

            is_valid = len(violations) == 0

            return {
                "valid": is_valid,
                "violations": violations[:10],  # Limit to first 10
                "certificate": certificate,
                "lie_derivative": str(lie_derivative),
                "num_violations": len(violations),
                "level_sets": {
                    "c1": level_info.initial_max,
                    "c2": level_info.unsafe_min,
                    "separation": level_info.separation,
                    "valid_separation": level_info.is_valid,
                },
            }

        except Exception as e:
            return {"valid": False, "error": str(e), "certificate": certificate}

    def test_certificate_extraction_accuracy(self) -> Dict:
        """Test certificate extraction accuracy"""
        logger.info("Testing certificate extraction accuracy...")

        # Test cases with known expected results
        test_cases = [
            {
                "input": "BARRIER_CERTIFICATE_START\nx**2 + y**2 - 1.5\nBARRIER_CERTIFICATE_END",
                "expected": "x**2 + y**2 - 1.5",
                "variables": ["x", "y"],
            },
            {
                "input": "B(x,y) = x**2 + y**2 - 1.5",
                "expected": "x**2 + y**2 - 1.5",
                "variables": ["x", "y"],
            },
            {
                "input": "Certificate: x**2 + y**2 - 1.5",
                "expected": "x**2 + y**2 - 1.5",
                "variables": ["x", "y"],
            },
            {
                "input": "Invalid format with no certificate",
                "expected": None,
                "variables": ["x", "y"],
            },
            {
                "input": "BARRIER_CERTIFICATE_START\nax**2 + by**2 + c\nBARRIER_CERTIFICATE_END",
                "expected": None,  # Template should be rejected
                "variables": ["x", "y"],
            },
        ]

        correct_extractions = 0
        results = []

        for i, test_case in enumerate(test_cases):
            try:
                extracted_result = extract_certificate_from_llm_output(
                    test_case["input"], test_case["variables"]
                )
                extracted = (
                    extracted_result[0]
                    if isinstance(extracted_result, tuple)
                    else extracted_result
                )

                # Check if extraction matches expected
                if extracted == test_case["expected"]:
                    correct_extractions += 1
                    status = "CORRECT"
                else:
                    status = "INCORRECT"

                results.append(
                    {
                        "test_case": i + 1,
                        "input": (
                            test_case["input"][:50] + "..."
                            if len(test_case["input"]) > 50
                            else test_case["input"]
                        ),
                        "expected": test_case["expected"],
                        "extracted": extracted,
                        "correct": extracted == test_case["expected"],
                        "status": status,
                    }
                )

            except Exception as e:
                results.append(
                    {
                        "test_case": i + 1,
                        "input": (
                            test_case["input"][:50] + "..."
                            if len(test_case["input"]) > 50
                            else test_case["input"]
                        ),
                        "expected": test_case["expected"],
                        "extracted": None,
                        "correct": False,
                        "status": f"ERROR: {str(e)}",
                    }
                )

        accuracy = correct_extractions / len(test_cases)

        return {
            "accuracy": accuracy,
            "correct_extractions": correct_extractions,
            "total_tests": len(test_cases),
            "results": results,
        }

    def test_certificate_validation_accuracy(self) -> Dict:
        """Test certificate validation accuracy"""
        logger.info("Testing certificate validation accuracy...")

        systems = self.generate_test_systems()
        all_results = []

        for system in systems:
            logger.info(f"Testing system: {system['name']}")
            system_results = []

            for i, certificate in enumerate(system["valid_certificates"]):
                expected_valid = system["expected_valid"][i]

                # Validate certificate mathematically
                validation_result = self.validate_certificate_mathematically(
                    certificate, system
                )

                # Check if validation matches expectation
                actual_valid = validation_result.get("valid", False)
                validation_correct = actual_valid == expected_valid

                system_results.append(
                    {
                        "certificate": certificate,
                        "expected_valid": expected_valid,
                        "actual_valid": actual_valid,
                        "validation_correct": validation_correct,
                        "violations": validation_result.get("violations", []),
                        "lie_derivative": validation_result.get("lie_derivative", ""),
                        "num_violations": validation_result.get("num_violations", 0),
                        "level_sets": validation_result.get("level_sets", {}),
                    }
                )

            # Calculate accuracy for this system
            correct_validations = sum(
                1 for r in system_results if r["validation_correct"]
            )
            system_accuracy = correct_validations / len(system_results)

            all_results.append(
                {
                    "system_name": system["name"],
                    "accuracy": system_accuracy,
                    "correct_validations": correct_validations,
                    "total_certificates": len(system_results),
                    "results": system_results,
                }
            )

        # Calculate overall accuracy
        total_correct = sum(r["correct_validations"] for r in all_results)
        total_tests = sum(r["total_certificates"] for r in all_results)
        overall_accuracy = total_correct / total_tests if total_tests > 0 else 0

        return {
            "overall_accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_tests": total_tests,
            "system_results": all_results,
        }

    def test_end_to_end_accuracy(self) -> Dict:
        """Test end-to-end accuracy from LLM output to validation"""
        logger.info("Testing end-to-end accuracy...")

        # Simulate LLM outputs with certificates
        llm_outputs = [
            "BARRIER_CERTIFICATE_START\nx**2 + y**2 - 1.0\nBARRIER_CERTIFICATE_END",
            "B(x,y) = x**2 + y**2 - 0.75",
            "Certificate: x**2 + y**2 - 0.3",
            "BARRIER_CERTIFICATE_START\nax**2 + by**2 + c\nBARRIER_CERTIFICATE_END",  # Template
        ]

        system = {
            "dynamics": ["dx/dt = -x", "dy/dt = -y"],
            "initial_set": ["x**2 + y**2 <= 0.25"],
            "unsafe_set": ["x**2 + y**2 >= 4.0"],
        }

        expected_results = [True, True, False, False]  # Based on correct theory

        correct_end_to_end = 0
        results = []

        for i, llm_output in enumerate(llm_outputs):
            try:
                # Extract certificate
                extracted_result = extract_certificate_from_llm_output(
                    llm_output, ["x", "y"]
                )
                extracted = (
                    extracted_result[0]
                    if isinstance(extracted_result, tuple)
                    else extracted_result
                )

                if extracted:
                    # Validate certificate
                    validation_result = self.validate_certificate_mathematically(
                        extracted, system
                    )
                    actual_valid = validation_result.get("valid", False)
                else:
                    actual_valid = False

                expected_valid = expected_results[i]
                end_to_end_correct = actual_valid == expected_valid

                if end_to_end_correct:
                    correct_end_to_end += 1

                results.append(
                    {
                        "llm_output": (
                            llm_output[:50] + "..."
                            if len(llm_output) > 50
                            else llm_output
                        ),
                        "extracted": extracted,
                        "expected_valid": expected_valid,
                        "actual_valid": actual_valid,
                        "end_to_end_correct": end_to_end_correct,
                        "validation_details": (
                            validation_result
                            if extracted
                            else {"valid": False, "error": "No extraction"}
                        ),
                    }
                )

            except Exception as e:
                results.append(
                    {
                        "llm_output": (
                            llm_output[:50] + "..."
                            if len(llm_output) > 50
                            else llm_output
                        ),
                        "extracted": None,
                        "expected_valid": expected_results[i],
                        "actual_valid": False,
                        "end_to_end_correct": False,
                        "validation_details": {"valid": False, "error": str(e)},
                    }
                )

        end_to_end_accuracy = correct_end_to_end / len(llm_outputs)

        return {
            "end_to_end_accuracy": end_to_end_accuracy,
            "correct_end_to_end": correct_end_to_end,
            "total_tests": len(llm_outputs),
            "results": results,
        }

    def run_comprehensive_accuracy_tests(self) -> Dict:
        """Run all accuracy tests"""
        logger.info("Starting comprehensive accuracy tests...")

        start_time = time.time()

        # Run all accuracy tests
        extraction_results = self.test_certificate_extraction_accuracy()
        validation_results = self.test_certificate_validation_accuracy()
        end_to_end_results = self.test_end_to_end_accuracy()

        total_time = time.time() - start_time

        # Compile comprehensive results
        comprehensive_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_time": total_time,
            "extraction_accuracy": extraction_results["accuracy"],
            "validation_accuracy": validation_results["overall_accuracy"],
            "end_to_end_accuracy": end_to_end_results["end_to_end_accuracy"],
            "overall_accuracy": (
                extraction_results["accuracy"]
                + validation_results["overall_accuracy"]
                + end_to_end_results["end_to_end_accuracy"]
            )
            / 3,
            "detailed_results": {
                "extraction": extraction_results,
                "validation": validation_results,
                "end_to_end": end_to_end_results,
            },
        }

        return comprehensive_results

    def save_accuracy_results(
        self,
        results: Dict,
        output_path: str = "test_results/certificate_accuracy_results.json",
    ):
        """Save accuracy test results"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Accuracy results saved to: {output_path}")

    def generate_accuracy_report(self, results: Dict) -> str:
        """Generate human-readable accuracy report"""
        report = []
        report.append("CERTIFICATE ACCURACY REPORT")
        report.append("=" * 50)
        report.append(f"Timestamp: {results['timestamp']}")
        report.append(f"Total Time: {results['total_time']:.1f} seconds")
        report.append("")

        # Overall accuracy
        overall_acc = results["overall_accuracy"]
        report.append(f"OVERALL ACCURACY: {overall_acc:.1%}")

        if overall_acc >= 0.95:
            report.append("EXCELLENT: Near-perfect accuracy achieved!")
        elif overall_acc >= 0.85:
            report.append("GOOD: High accuracy achieved")
        elif overall_acc >= 0.70:
            report.append("MODERATE: Some accuracy issues detected")
        else:
            report.append("POOR: Significant accuracy issues detected")

        report.append("")

        # Component accuracies
        report.append("COMPONENT ACCURACIES:")
        report.append(f"  Extraction: {results['extraction_accuracy']:.1%}")
        report.append(f"  Validation: {results['validation_accuracy']:.1%}")
        report.append(f"  End-to-End: {results['end_to_end_accuracy']:.1%}")

        report.append("")

        # Detailed results
        report.append("DETAILED RESULTS:")

        # Extraction results
        ext_results = results["detailed_results"]["extraction"]
        report.append(
            f"  Extraction Tests: {ext_results['correct_extractions']}/{ext_results['total_tests']} correct"
        )

        # Validation results
        val_results = results["detailed_results"]["validation"]
        report.append(
            f"  Validation Tests: {val_results['total_correct']}/{val_results['total_tests']} correct"
        )

        # Show system-level breakdown
        for system_result in val_results["system_results"]:
            report.append(
                f"    {system_result['system_name']}: {system_result['correct_validations']}/{system_result['total_certificates']} correct"
            )

        # End-to-end results
        e2e_results = results["detailed_results"]["end_to_end"]
        report.append(
            f"  End-to-End Tests: {e2e_results['correct_end_to_end']}/{e2e_results['total_tests']} correct"
        )

        return "\n".join(report)


def main():
    """Main function to run accuracy tests"""
    tester = CertificateValidationTester()

    print("Starting Certificate Accuracy Tests with Correct Theory...")
    print("=" * 50)

    # Run comprehensive accuracy tests
    results = tester.run_comprehensive_accuracy_tests()

    # Generate and display report
    report = tester.generate_accuracy_report(results)
    print(report)

    # Save results
    tester.save_accuracy_results(results)

    # Return appropriate exit code
    if results["overall_accuracy"] >= 0.95:
        print("\nNear-perfect accuracy achieved!")
        return 0
    elif results["overall_accuracy"] >= 0.85:
        print("\nHigh accuracy achieved")
        return 0
    elif results["overall_accuracy"] >= 0.70:
        print("\nModerate accuracy - improvements needed")
        return 1
    else:
        print("\nPoor accuracy - significant improvements needed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
