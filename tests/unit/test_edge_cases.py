#!/usr/bin/env python3
"""Edge case tests for certificate validation"""

import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import sympy

from utils.certificate_extraction import extract_certificate_from_llm_output


def test_edge_cases():
    """Test edge cases in certificate extraction and validation"""
    print("Testing edge cases...")

    test_cases = [
        # Empty inputs
        ("", ["x", "y"], None, "Empty input"),
        ("   \n\n   ", ["x", "y"], None, "Whitespace only"),
        # Malformed certificates
        ("B(x,y) = ", ["x", "y"], None, "Incomplete equation"),
        ("B(x,y) = x**2 + ", ["x", "y"], None, "Incomplete expression"),
        # Invalid syntax
        ("B(x,y) = x^2 + y^2", ["x", "y"], None, "Wrong exponent syntax"),
        ("B(x,y) = x2 + y2 - 1", ["x", "y"], None, "Missing operators"),
        # Mixed variables
        ("B(x,y) = x**2 + z**2 - 1", ["x", "y"], None, "Wrong variables"),
        # Complex expressions that should work
        (
            "B(x,y) = 2*x**2 + 3*y**2 - 1.5",
            ["x", "y"],
            "2*x**2 + 3*y**2 - 1.5",
            "Weighted quadratic",
        ),
        (
            "B(x,y,z) = x**2 + y**2 + z**2 - 2.0",
            ["x", "y", "z"],
            "x**2 + y**2 + z**2 - 2.0",
            "3D certificate",
        ),
        # Multiple certificates (should extract first valid one)
        (
            "B(x,y) = x**2 + y**2 - 1\nB(x,y) = x**2 + y**2 - 2",
            ["x", "y"],
            "x**2 + y**2 - 1",
            "Multiple certificates",
        ),
        # Nested expressions
        (
            "The certificate is B(x,y) = (x**2 + y**2) - 1.0",
            ["x", "y"],
            "x**2 + y**2 - 1.0",
            "Parentheses",
        ),
        # Scientific notation
        (
            "B(x,y) = x**2 + y**2 - 1e-3",
            ["x", "y"],
            "x**2 + y**2 - 0.001",
            "Scientific notation",
        ),
        # Unicode and special characters (should fail)
        ("B(x,y) = x² + y² - 1", ["x", "y"], None, "Unicode exponents"),
    ]

    passed = 0
    failed = 0

    for i, (input_text, variables, expected, description) in enumerate(test_cases):
        try:
            result = extract_certificate_from_llm_output(input_text, variables)
            extracted = result[0] if isinstance(result, tuple) else result

            # Normalize expressions for comparison
            if extracted and expected:
                try:
                    extracted_expr = sympy.parse_expr(extracted)
                    expected_expr = sympy.parse_expr(expected)
                    match = extracted_expr.equals(expected_expr)
                except Exception:
                    match = extracted == expected
            else:
                match = extracted == expected

            if match:
                print(f"PASS Test {i+1}: {description}")
                passed += 1
            else:
                print(f"FAIL Test {i+1}: {description}")
                print(f"   Expected: {expected}")
                print(f"   Got: {extracted}")
                failed += 1

        except Exception as e:
            print(f"FAIL Test {i+1}: {description} - ERROR: {str(e)}")
            failed += 1

    print(f"\nEdge case tests: {passed}/{passed+failed} passed")
    return failed == 0


if __name__ == "__main__":
    success = test_edge_cases()
    sys.exit(0 if success else 1)
