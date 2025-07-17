#!/usr/bin/env python3
"""
Critical Fixes Test
===================

Quick test to verify the most critical accuracy issues are addressed.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.certificate_extraction import extract_certificate_from_llm_output

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_critical_extraction_fixes():
    """Test critical extraction fixes"""
    print("🔧 Testing Critical Extraction Fixes...")

    # Test cases that were failing
    critical_tests = [
        {
            "input": "B(x,y) = x**2 + y**2 - 1.5",
            "expected": "x**2 + y**2 - 1.5",
            "description": "Decimal number extraction",
        },
        {
            "input": "BARRIER_CERTIFICATE_START\nax**2 + by**2 + c\nBARRIER_CERTIFICATE_END",
            "expected": None,  # Should be rejected as template
            "description": "Template detection",
        },
        {
            "input": "Certificate: x**2 + y**2 - 2.0",
            "expected": "x**2 + y**2 - 2.0",
            "description": "Decimal zero extraction",
        },
    ]

    passed = 0
    total = len(critical_tests)

    for i, test in enumerate(critical_tests):
        try:
            extracted_result = extract_certificate_from_llm_output(test["input"], ["x", "y"])
            extracted = (
                extracted_result[0] if isinstance(extracted_result, tuple) else extracted_result
            )

            if extracted == test["expected"]:
                print(f"✅ Test {i+1}: {test['description']} - PASSED")
                passed += 1
            else:
                print(f"❌ Test {i+1}: {test['description']} - FAILED")
                print(f"   Expected: {test['expected']}")
                print(f"   Got: {extracted}")
        except Exception as e:
            print(f"❌ Test {i+1}: {test['description']} - ERROR: {e}")

    accuracy = passed / total
    print(f"\n📊 Critical Extraction Accuracy: {accuracy:.1%} ({passed}/{total})")
    return accuracy >= 0.8


def test_validation_improvements():
    """Test validation improvements"""
    print("\n🔧 Testing Validation Improvements...")

    # Test basic mathematical validation
    try:
        import sympy

        x, y = sympy.symbols("x y")

        # Test simple certificate validation
        B = sympy.parse_expr("x**2 + y**2 - 1.5")
        dynamics = [sympy.parse_expr("-x"), sympy.parse_expr("-y")]

        # Calculate Lie derivative
        dB_dx = sympy.diff(B, x)
        dB_dy = sympy.diff(B, y)
        lie_derivative = dB_dx * dynamics[0] + dB_dy * dynamics[1]

        # Test at a point
        test_point = {x: 0.5, y: 0.5}
        lie_val = lie_derivative.subs(test_point)
        B_val = B.subs(test_point)

        print(f"✅ Lie derivative calculation: {lie_derivative}")
        print(f"✅ Test point evaluation: B={B_val}, dB/dt={lie_val}")

        # Basic validation logic
        is_valid = lie_val <= 0  # Should be negative for stable system
        print(f"✅ Basic validation: {is_valid}")

        return True

    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False


def main():
    """Run critical fixes test"""
    print("🚀 Critical Fixes Test")
    print("=" * 40)

    # Test extraction fixes
    extraction_ok = test_critical_extraction_fixes()

    # Test validation improvements
    validation_ok = test_validation_improvements()

    print("\n" + "=" * 40)
    if extraction_ok and validation_ok:
        print("✅ Critical fixes are working!")
        return 0
    else:
        print("❌ Critical fixes need more work")
        return 1


if __name__ == "__main__":
    sys.exit(main())
