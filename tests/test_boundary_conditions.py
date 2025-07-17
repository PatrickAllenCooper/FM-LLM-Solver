#!/usr/bin/env python3
"""
Boundary Condition Tests
Tests for edge cases and boundary values in certificate validation
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.unit.test_certificate_validation_accuracy import CertificateValidationTester
from utils.certificate_extraction import extract_certificate_from_llm_output


class TestBoundaryConditions:
    """Test boundary conditions and edge cases"""

    def test_set_boundaries(self):
        """Test certificates at exact set boundaries"""
        tester = CertificateValidationTester()

        # System with well-defined boundaries
        system = {
            "dynamics": ["dx/dt = -x", "dy/dt = -y"],
            "initial_set": ["x**2 + y**2 <= 0.25"],  # radius 0.5
            "unsafe_set": ["x**2 + y**2 >= 4.0"],  # radius 2.0
        }

        boundary_cases = [
            # Certificates exactly at boundaries
            (
                "x**2 + y**2 - 0.25",
                "At initial set boundary",
                False,
            ),  # B=0 at initial boundary
            (
                "x**2 + y**2 - 4.0",
                "At unsafe set boundary",
                False,
            ),  # B=0 at unsafe boundary
            # Just inside/outside boundaries
            ("x**2 + y**2 - 0.24", "Just inside initial boundary", False),
            ("x**2 + y**2 - 0.26", "Just outside initial boundary", False),
            ("x**2 + y**2 - 3.99", "Just inside unsafe boundary", False),
            ("x**2 + y**2 - 4.01", "Just outside unsafe boundary", False),
            # Exactly between sets
            ("x**2 + y**2 - 2.125", "Exactly between sets", True),  # (0.25 + 4.0) / 2
            # Golden ratio between sets
            ("x**2 + y**2 - 1.618", "Golden ratio position", True),
        ]

        print("Testing boundary conditions...")
        for cert, description, expected_valid in boundary_cases:
            result = tester.validate_certificate_mathematically(
                cert, system, n_samples=20
            )
            actual_valid = result["valid"]
            print(
                f"{description}: {'PASS' if actual_valid == expected_valid else 'FAIL'}"
            )
            if actual_valid != expected_valid:
                print(f"  Expected: {expected_valid}, Got: {actual_valid}")
                if result.get("violations"):
                    print(f"  First violation: {result['violations'][0]}")

    def test_dimension_boundaries(self):
        """Test different dimensional systems"""
        tester = CertificateValidationTester()

        # 1D system (edge case - not typically supported)
        system_1d = {
            "dynamics": ["dx/dt = -x"],
            "initial_set": ["x**2 <= 0.25"],
            "unsafe_set": ["x**2 >= 4.0"],
        }

        # 2D system (standard)
        system_2d = {
            "dynamics": ["dx/dt = -x", "dy/dt = -y"],
            "initial_set": ["x**2 + y**2 <= 0.25"],
            "unsafe_set": ["x**2 + y**2 >= 4.0"],
        }

        # 3D system
        system_3d = {
            "dynamics": ["dx/dt = -x", "dy/dt = -y", "dz/dt = -z"],
            "initial_set": ["x**2 + y**2 + z**2 <= 0.25"],
            "unsafe_set": ["x**2 + y**2 + z**2 >= 4.0"],
        }

        # 4D system (edge case - might not be supported)
        system_4d = {
            "dynamics": ["dx/dt = -x", "dy/dt = -y", "dz/dt = -z", "dw/dt = -w"],
            "initial_set": ["x**2 + y**2 + z**2 + w**2 <= 0.25"],
            "unsafe_set": ["x**2 + y**2 + z**2 + w**2 >= 4.0"],
        }

        print("\nTesting dimensional boundaries...")

        # Test 2D certificate on 2D system (should work)
        result = tester.validate_certificate_mathematically(
            "x**2 + y**2 - 1.0", system_2d, n_samples=10
        )
        print(f"2D certificate on 2D system: {'PASS' if result['valid'] else 'FAIL'}")

        # Test 3D certificate on 3D system (should work)
        result = tester.validate_certificate_mathematically(
            "x**2 + y**2 + z**2 - 1.0", system_3d, n_samples=10
        )
        print(f"3D certificate on 3D system: {'PASS' if result['valid'] else 'FAIL'}")

        # Test dimension mismatches
        try:
            result = tester.validate_certificate_mathematically(
                "x**2 + y**2 + z**2 - 1.0", system_2d, n_samples=10
            )
            print(
                f"3D certificate on 2D system: {'Handled' if not result['valid'] else 'Unexpected success'}"
            )
        except Exception:
            print("3D certificate on 2D system: Exception (expected)")

    def test_numerical_precision(self):
        """Test numerical precision boundaries"""
        tester = CertificateValidationTester()
        system = {
            "dynamics": ["dx/dt = -x", "dy/dt = -y"],
            "initial_set": ["x**2 + y**2 <= 0.25"],
            "unsafe_set": ["x**2 + y**2 >= 4.0"],
        }

        precision_cases = [
            # Different decimal precisions
            ("x**2 + y**2 - 1", "Integer constant"),
            ("x**2 + y**2 - 1.0", "One decimal"),
            ("x**2 + y**2 - 1.00", "Two decimals"),
            ("x**2 + y**2 - 1.000000000000001", "Near machine epsilon"),
            # Scientific notation edge cases
            ("x**2 + y**2 - 1e0", "Scientific notation 10^0"),
            ("x**2 + y**2 - 1.0e0", "Scientific with decimal"),
            ("x**2 + y**2 - 10e-1", "Scientific 10^-1"),
            # Very small differences
            ("x**2 + y**2 - 0.9999999999", "Just below 1.0"),
            ("x**2 + y**2 - 1.0000000001", "Just above 1.0"),
        ]

        print("\nTesting numerical precision...")
        for cert, description in precision_cases:
            try:
                extracted = extract_certificate_from_llm_output(
                    f"B(x,y) = {cert}", ["x", "y"]
                )
                extracted_cert = (
                    extracted[0] if isinstance(extracted, tuple) else extracted
                )
                if extracted_cert:
                    result = tester.validate_certificate_mathematically(
                        extracted_cert, system, n_samples=5
                    )
                    print(f"{description}: {'Valid' if result['valid'] else 'Invalid'}")
                else:
                    print(f"{description}: Failed to extract")
            except Exception as e:
                print(f"{description}: Exception - {type(e).__name__}")

    def test_coefficient_boundaries(self):
        """Test boundary cases for coefficients"""
        test_cases = [
            # Symmetric coefficients
            ("B(x,y) = x**2 + y**2 - 1", "Standard symmetric"),
            ("B(x,y) = 2*x**2 + 2*y**2 - 2", "Scaled symmetric"),
            # Asymmetric coefficients
            ("B(x,y) = 2*x**2 + y**2 - 1", "Asymmetric x-heavy"),
            ("B(x,y) = x**2 + 3*y**2 - 1", "Asymmetric y-heavy"),
            # Extreme asymmetry
            ("B(x,y) = 100*x**2 + y**2 - 1", "Extreme x-heavy"),
            ("B(x,y) = x**2 + 100*y**2 - 1", "Extreme y-heavy"),
            # Mixed signs (should be rejected)
            ("B(x,y) = x**2 - y**2 - 1", "Mixed signs"),
            ("B(x,y) = -x**2 + y**2 - 1", "Mixed signs reversed"),
            # All negative (should be rejected)
            ("B(x,y) = -x**2 - y**2 - 1", "All negative"),
        ]

        print("\nTesting coefficient boundaries...")
        for expr, description in test_cases:
            result = extract_certificate_from_llm_output(expr, ["x", "y"])
            extracted = result[0] if isinstance(result, tuple) else result

            if "Mixed signs" in description or "All negative" in description:
                # These should ideally be rejected or marked invalid
                print(
                    f"{description}: {'Correctly rejected' if not extracted else 'Extracted (check validation)'}"
                )
            else:
                print(
                    f"{description}: {'Extracted' if extracted else 'Failed to extract'}"
                )

    def test_safety_margin_boundaries(self):
        """Test safety margin edge cases"""
        tester = CertificateValidationTester()
        system = {
            "dynamics": ["dx/dt = -x", "dy/dt = -y"],
            "initial_set": ["x**2 + y**2 <= 0.25"],  # radius 0.5
            "unsafe_set": ["x**2 + y**2 >= 4.0"],  # radius 2.0
        }

        # Test certificates with different safety margins
        # Initial set boundary is at r=0.5 (x²+y²=0.25)
        margin_cases = [
            ("x**2 + y**2 - 0.2601", 2.0),  # 2% margin: sqrt(0.2601) = 0.51
            ("x**2 + y**2 - 0.2704", 4.0),  # 4% margin: sqrt(0.2704) = 0.52
            ("x**2 + y**2 - 0.3025", 10.0),  # 10% margin: sqrt(0.3025) = 0.55
            ("x**2 + y**2 - 0.36", 20.0),  # 20% margin: sqrt(0.36) = 0.6
            ("x**2 + y**2 - 0.5625", 50.0),  # 50% margin: sqrt(0.5625) = 0.75
        ]

        print("\nTesting safety margin boundaries...")
        for cert, margin_percent in margin_cases:
            result = tester.validate_certificate_mathematically(
                cert, system, n_samples=10
            )
            print(
                f"{margin_percent}% margin: {'Valid' if result['valid'] else 'Invalid'}"
            )
            if not result["valid"] and result.get("violations"):
                # Check if it's a safety margin violation
                safety_violations = [
                    v for v in result["violations"] if "Safety margin" in v
                ]
                if safety_violations:
                    print(f"  {safety_violations[0]}")


def main():
    """Run boundary condition tests"""
    print("Boundary Condition Test Suite")
    print("=" * 60)

    test = TestBoundaryConditions()

    # Run all boundary tests
    test.test_set_boundaries()
    test.test_dimension_boundaries()
    test.test_numerical_precision()
    test.test_coefficient_boundaries()
    test.test_safety_margin_boundaries()

    print("\nBoundary condition tests completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
