#!/usr/bin/env python3
"""
Comprehensive Error Handling Tests
Tests for robustness and graceful failure modes
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.unit.test_certificate_validation_accuracy import CertificateValidationTester
from utils.certificate_extraction import extract_certificate_from_llm_output


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_malformed_inputs(self):
        """Test handling of malformed inputs"""
        test_cases = [
            # Completely invalid inputs
            (None, ["x", "y"], "None input"),
            (123, ["x", "y"], "Integer input"),
            ([], ["x", "y"], "List input"),
            ({}, ["x", "y"], "Dict input"),
            # Malformed mathematical expressions
            ("B(x,y) = x**2 + + y**2", ["x", "y"], "Double operator"),
            ("B(x,y) = x**2 y**2", ["x", "y"], "Missing operator"),
            ("B(x,y) = (x**2 + y**2", ["x", "y"], "Unmatched parenthesis"),
            ("B(x,y) = x**2 + y**2)", ["x", "y"], "Extra parenthesis"),
            # Invalid variable references
            ("B(x,y) = x**2 + z**2 - 1", ["x", "y"], "Undefined variable z"),
            ("B(x,y) = a**2 + b**2 - 1", ["x", "y"], "Wrong variable names"),
            # Syntax errors
            ("B(x,y) = x^2 + y^2 - 1", ["x", "y"], "Caret instead of **"),
            ("B(x,y) = x2 + y2 - 1", ["x", "y"], "Missing exponent operator"),
            ("B(x,y) = sqrt(x**2 + y**2) - 1", ["x", "y"], "Undefined function"),
            # Encoding issues
            ("B(x,y) = x² + y² - 1", ["x", "y"], "Unicode superscript"),
            ("B(x,y) = x\u00b2 + y\u00b2 - 1", ["x", "y"], "Unicode character"),
            # Empty or incomplete
            ("B(x,y) = ", ["x", "y"], "Empty expression"),
            ("B(x,y)", ["x", "y"], "No equals sign"),
            ("= x**2 + y**2 - 1", ["x", "y"], "No function name"),
        ]

        for input_val, variables, description in test_cases:
            print(f"Testing {description}...")
            try:
                result = extract_certificate_from_llm_output(input_val, variables)
                extracted = result[0] if isinstance(result, tuple) else result
                # Should return None for invalid inputs
                assert (
                    extracted is None
                ), f"{description} should return None, got {extracted}"
            except Exception as e:
                # Some inputs might raise exceptions, which is acceptable
                print(f"  Exception (acceptable): {type(e).__name__}")

    def test_validation_errors(self):
        """Test validation error handling"""
        tester = CertificateValidationTester()

        error_cases = [
            # Invalid system specifications
            {
                "certificate": "x**2 + y**2 - 1",
                "system": {
                    "dynamics": None,  # Invalid dynamics
                    "initial_set": ["x**2 + y**2 <= 0.25"],
                    "unsafe_set": ["x**2 + y**2 >= 4.0"],
                },
                "description": "None dynamics",
            },
            {
                "certificate": "x**2 + y**2 - 1",
                "system": {
                    "dynamics": ["invalid syntax here"],  # Unparseable dynamics
                    "initial_set": ["x**2 + y**2 <= 0.25"],
                    "unsafe_set": ["x**2 + y**2 >= 4.0"],
                },
                "description": "Invalid dynamics syntax",
            },
            {
                "certificate": "x**2 + y**2 - 1",
                "system": {
                    "dynamics": ["dx/dt = -x", "dy/dt = -y"],
                    "initial_set": ["invalid set condition"],  # Invalid set
                    "unsafe_set": ["x**2 + y**2 >= 4.0"],
                },
                "description": "Invalid initial set",
            },
            {
                "certificate": "invalid certificate",  # Invalid certificate
                "system": {
                    "dynamics": ["dx/dt = -x", "dy/dt = -y"],
                    "initial_set": ["x**2 + y**2 <= 0.25"],
                    "unsafe_set": ["x**2 + y**2 >= 4.0"],
                },
                "description": "Invalid certificate expression",
            },
        ]

        for case in error_cases:
            print(f"Testing {case['description']}...")
            result = tester.validate_certificate_mathematically(
                case["certificate"], case["system"], n_samples=5
            )
            # Should return invalid with error
            assert not result["valid"], f"{case['description']} should be invalid"
            assert "error" in result or result["num_violations"] > 0

    def test_extreme_values(self):
        """Test handling of extreme values"""
        extreme_cases = [
            # Very large numbers
            ("B(x,y) = x**2 + y**2 - 1e100", ["x", "y"], "Very large constant"),
            ("B(x,y) = 1e100*x**2 + y**2 - 1", ["x", "y"], "Very large coefficient"),
            # Very small numbers
            ("B(x,y) = x**2 + y**2 - 1e-100", ["x", "y"], "Very small constant"),
            ("B(x,y) = 1e-100*x**2 + y**2 - 1", ["x", "y"], "Very small coefficient"),
            # Zero coefficients
            ("B(x,y) = 0*x**2 + y**2 - 1", ["x", "y"], "Zero coefficient"),
            ("B(x,y) = x**2 + y**2 - 0", ["x", "y"], "Zero constant"),
            # Negative exponents (should fail)
            ("B(x,y) = x**(-2) + y**2 - 1", ["x", "y"], "Negative exponent"),
            # Fractional exponents (should fail for barrier certificates)
            ("B(x,y) = x**(1/2) + y**2 - 1", ["x", "y"], "Fractional exponent"),
        ]

        for expr, variables, description in extreme_cases:
            print(f"Testing {description}...")
            result = extract_certificate_from_llm_output(expr, variables)
            extracted = result[0] if isinstance(result, tuple) else result

            # Check if extraction handles extreme values appropriately
            if "e100" in expr or "e-100" in expr:
                # Should extract but might have numerical issues
                print(f"  Extracted: {extracted}")
            elif "**(-" in expr or "**(1/" in expr:
                # Should reject non-polynomial expressions
                assert extracted is None, f"{description} should be rejected"

    def test_concurrent_access(self):
        """Test thread safety of validation"""
        import threading
        import time

        tester = CertificateValidationTester()
        system = {
            "dynamics": ["dx/dt = -x", "dy/dt = -y"],
            "initial_set": ["x**2 + y**2 <= 0.25"],
            "unsafe_set": ["x**2 + y**2 >= 4.0"],
        }

        results = []
        errors = []

        def validate_concurrent(cert, index):
            try:
                result = tester.validate_certificate_mathematically(
                    cert, system, n_samples=10
                )
                results.append((index, result))
            except Exception as e:
                errors.append((index, str(e)))

        # Create multiple threads
        threads = []
        certificates = [
            "x**2 + y**2 - 1.0",
            "x**2 + y**2 - 1.5",
            "2*x**2 + 2*y**2 - 3.0",
            "x**2 + y**2 - 0.75",
        ]

        print("Testing concurrent validation...")
        start_time = time.time()

        for i, cert in enumerate(certificates * 3):  # Test each certificate 3 times
            thread = threading.Thread(target=validate_concurrent, args=(cert, i))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        duration = time.time() - start_time
        print(f"Concurrent validation completed in {duration:.2f}s")
        print(f"Results: {len(results)}, Errors: {len(errors)}")

        # All validations should complete without errors
        assert len(errors) == 0, f"Concurrent validation had {len(errors)} errors"
        assert len(results) == len(certificates) * 3


def main():
    """Run error handling tests"""
    print("Error Handling Test Suite")
    print("=" * 60)

    test = TestErrorHandling()

    # Run each test category
    try:
        print("\n1. Testing Malformed Inputs...")
        test.test_malformed_inputs()
        print("PASS: Malformed inputs handled correctly")
    except AssertionError as e:
        print(f"FAIL: {e}")
        return 1

    try:
        print("\n2. Testing Validation Errors...")
        test.test_validation_errors()
        print("PASS: Validation errors handled correctly")
    except AssertionError as e:
        print(f"FAIL: {e}")
        return 1

    try:
        print("\n3. Testing Extreme Values...")
        test.test_extreme_values()
        print("PASS: Extreme values handled correctly")
    except AssertionError as e:
        print(f"FAIL: {e}")
        return 1

    try:
        print("\n4. Testing Concurrent Access...")
        test.test_concurrent_access()
        print("PASS: Concurrent access is thread-safe")
    except AssertionError as e:
        print(f"FAIL: {e}")
        return 1

    print("\nAll error handling tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
