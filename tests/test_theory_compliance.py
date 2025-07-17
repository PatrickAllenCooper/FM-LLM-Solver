#!/usr/bin/env python3
"""
Theory Compliance Tests for Barrier Certificates
Tests that the implementation follows correct mathematical theory
"""

import unittest
import sympy as sp
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.level_set_tracker import LevelSetTracker
from tests.unit.test_certificate_validation_accuracy import CertificateValidationTester


class TheoryComplianceTests(unittest.TestCase):
    """Test suite to ensure barrier certificate theory is correctly implemented"""

    def setUp(self):
        self.tracker = LevelSetTracker()
        self.validator = CertificateValidationTester()

    def test_level_set_separation_required(self):
        """Test that proper separation c1 < c2 is enforced"""
        # System with clear separation
        system = {
            "dynamics": ["dx/dt = -x", "dy/dt = -y"],
            "initial_set": ["x**2 + y**2 <= 0.25"],  # r = 0.5
            "unsafe_set": ["x**2 + y**2 >= 4.0"],  # r = 2.0
        }

        # Valid certificate with proper separation
        valid_cert = "x**2 + y**2 - 1.0"  # B = 0 at r = 1
        result = self.validator.validate_certificate_mathematically(valid_cert, system)
        self.assertTrue(
            result["valid"], f"Valid certificate rejected: {result.get('violations', [])}"
        )

        # Invalid certificate - no separation
        invalid_cert = "x**2 + y**2"  # B > 0 everywhere
        result = self.validator.validate_certificate_mathematically(invalid_cert, system)
        self.assertFalse(result["valid"], "Certificate with no separation should be invalid")

        # Invalid certificate - B = 0 inside initial set
        invalid_cert = "x**2 + y**2 - 0.1"  # B = 0 at r = 0.316 < 0.5
        result = self.validator.validate_certificate_mathematically(invalid_cert, system)
        self.assertFalse(
            result["valid"], "Certificate with B = 0 inside initial set should be invalid"
        )

    def test_initial_set_condition(self):
        """Test that B(x) ≤ c1 for all x in Initial Set"""
        system = {
            "dynamics": ["dx/dt = -x", "dy/dt = -y"],
            "initial_set": ["x**2 + y**2 <= 0.25"],
            "unsafe_set": ["x**2 + y**2 >= 4.0"],
        }

        # Compute level sets
        cert = "x**2 + y**2 - 1.0"
        level_info = self.tracker.compute_level_sets(
            cert, system["initial_set"], system["unsafe_set"], ["x", "y"]
        )

        # c1 should be max(B) on initial set
        # For B = x² + y² - 1 and initial set x² + y² ≤ 0.25
        # max occurs at boundary: B = 0.25 - 1 = -0.75
        self.assertAlmostEqual(
            level_info.initial_max,
            -0.75,
            places=2,
            msg=f"c1 should be -0.75, got {level_info.initial_max}",
        )

    def test_unsafe_set_condition(self):
        """Test that B(x) ≥ c2 for all x in Unsafe Set"""
        system = {
            "dynamics": ["dx/dt = -x", "dy/dt = -y"],
            "initial_set": ["x**2 + y**2 <= 0.25"],
            "unsafe_set": ["x**2 + y**2 >= 4.0"],
        }

        # Compute level sets
        cert = "x**2 + y**2 - 1.0"
        level_info = self.tracker.compute_level_sets(
            cert, system["initial_set"], system["unsafe_set"], ["x", "y"]
        )

        # c2 should be min(B) on unsafe set
        # For B = x² + y² - 1 and unsafe set x² + y² ≥ 4
        # min occurs at boundary: B = 4 - 1 = 3
        self.assertAlmostEqual(
            level_info.unsafe_min,
            3.0,
            places=2,
            msg=f"c2 should be 3.0, got {level_info.unsafe_min}",
        )

    def test_lie_derivative_condition(self):
        """Test that dB/dt ≤ 0 in the critical region [c1, c2]"""
        system = {
            "dynamics": ["dx/dt = -x", "dy/dt = -y"],
            "initial_set": ["x**2 + y**2 <= 0.25"],
            "unsafe_set": ["x**2 + y**2 >= 4.0"],
        }

        cert = "x**2 + y**2 - 1.0"

        # Compute Lie derivative manually
        x, y = sp.symbols("x y")
        B = sp.parse_expr(cert)
        f = [-x, -y]  # dynamics

        # Lie derivative = ∇B · f = 2x(-x) + 2y(-y) = -2x² - 2y²
        lie_derivative = sum(sp.diff(B, var) * fi for var, fi in zip([x, y], f))
        lie_simplified = sp.simplify(lie_derivative)

        # Should be -2*x**2 - 2*y**2
        expected = -2 * x**2 - 2 * y**2
        self.assertEqual(
            lie_simplified, expected, f"Lie derivative should be {expected}, got {lie_simplified}"
        )

        # Verify it's negative in the critical region
        lie_func = sp.lambdify([x, y], lie_derivative, "numpy")

        # Test points in critical region (where 0.5 < r < 2)
        test_points = [
            (0.6, 0),  # r = 0.6
            (1.0, 0),  # r = 1.0
            (0, 1.5),  # r = 1.5
            (1.4, 1.4),  # r ≈ 1.98
        ]

        for point in test_points:
            lie_val = lie_func(*point)
            self.assertLessEqual(
                lie_val, 0, f"Lie derivative should be ≤ 0 at {point}, got {lie_val}"
            )

    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        system = {
            "dynamics": ["dx/dt = -x", "dy/dt = -y"],
            "initial_set": ["x**2 + y**2 <= 0.25"],
            "unsafe_set": ["x**2 + y**2 >= 4.0"],
        }

        # Edge case 1: B = 0 exactly at initial set boundary
        edge_cert1 = "x**2 + y**2 - 0.25"
        result = self.validator.validate_certificate_mathematically(edge_cert1, system)
        self.assertFalse(
            result["valid"], "Certificate with B = 0 at initial boundary should be invalid"
        )

        # Edge case 2: B = 0 exactly at unsafe set boundary
        edge_cert2 = "x**2 + y**2 - 4.0"
        result = self.validator.validate_certificate_mathematically(edge_cert2, system)
        self.assertFalse(
            result["valid"], "Certificate with B = 0 at unsafe boundary should be invalid"
        )

        # Edge case 3: Very small separation
        edge_cert3 = "x**2 + y**2 - 0.26"  # Just barely outside initial set
        result = self.validator.validate_certificate_mathematically(edge_cert3, system)
        # Should be invalid due to insufficient separation margin
        self.assertFalse(result["valid"], "Certificate with tiny separation should be invalid")

    def test_3d_system(self):
        """Test that 3D systems work correctly"""
        system = {
            "dynamics": ["dx/dt = -x", "dy/dt = -y", "dz/dt = -z"],
            "initial_set": ["x**2 + y**2 + z**2 <= 0.25"],
            "unsafe_set": ["x**2 + y**2 + z**2 >= 4.0"],
        }

        # Valid 3D certificate
        cert3d = "x**2 + y**2 + z**2 - 1.0"
        result = self.validator.validate_certificate_mathematically(cert3d, system)
        self.assertTrue(
            result["valid"], f"Valid 3D certificate rejected: {result.get('violations', [])}"
        )

        # Check level sets
        level_info = self.tracker.compute_level_sets(
            cert3d, system["initial_set"], system["unsafe_set"], ["x", "y", "z"]
        )

        self.assertAlmostEqual(level_info.initial_max, -0.75, places=2)
        self.assertAlmostEqual(level_info.unsafe_min, 3.0, places=2)

    def test_nonlinear_dynamics(self):
        """Test with nonlinear dynamics"""
        system = {
            "dynamics": ["dx/dt = -x + x**3", "dy/dt = -y"],
            "initial_set": ["x**2 + y**2 <= 0.25"],
            "unsafe_set": ["x**2 + y**2 >= 4.0"],
        }

        # This certificate should still work for small enough regions
        cert = "x**2 + y**2 - 1.0"
        result = self.validator.validate_certificate_mathematically(cert, system)

        # The Lie derivative is 2x(-x + x³) + 2y(-y) = -2x² + 2x⁴ - 2y²
        # This is negative when x² + y² < 1, which includes our critical region
        self.assertTrue(result["valid"], "Certificate should be valid for nonlinear system")

    def test_different_certificate_forms(self):
        """Test various mathematical forms of certificates"""
        system = {
            "dynamics": ["dx/dt = -x", "dy/dt = -y"],
            "initial_set": ["x**2 + y**2 <= 0.25"],
            "unsafe_set": ["x**2 + y**2 >= 4.0"],
        }

        # Scaled version
        cert_scaled = "2*x**2 + 2*y**2 - 2.0"  # Same as x² + y² - 1
        result = self.validator.validate_certificate_mathematically(cert_scaled, system)
        self.assertTrue(result["valid"], "Scaled certificate should be valid")

        # With coefficients
        cert_coeff = "1.0*x**2 + 1.0*y**2 - 1.0"
        result = self.validator.validate_certificate_mathematically(cert_coeff, system)
        self.assertTrue(result["valid"], "Certificate with explicit coefficients should be valid")

    def test_known_good_certificates(self):
        """Test a comprehensive set of known-good certificates"""
        test_cases = [
            {
                "system": {
                    "dynamics": ["dx/dt = -x", "dy/dt = -y"],
                    "initial_set": ["x**2 + y**2 <= 0.25"],
                    "unsafe_set": ["x**2 + y**2 >= 4.0"],
                },
                "certificates": [
                    ("x**2 + y**2 - 1.0", True, "Standard quadratic"),
                    ("x**2 + y**2 - 1.5", True, "Larger separation"),
                    ("x**2 + y**2 - 0.75", True, "Smaller but valid separation"),
                    ("0.5*x**2 + 0.5*y**2 - 0.5", True, "Scaled down version"),
                ],
            },
            {
                "system": {
                    "dynamics": ["dx/dt = -2*x", "dy/dt = -2*y"],
                    "initial_set": ["x**2 + y**2 <= 0.25"],
                    "unsafe_set": ["x**2 + y**2 >= 4.0"],
                },
                "certificates": [
                    ("x**2 + y**2 - 1.0", True, "Works with faster dynamics"),
                ],
            },
        ]

        for test_case in test_cases:
            system = test_case["system"]
            for cert, expected, desc in test_case["certificates"]:
                result = self.validator.validate_certificate_mathematically(cert, system)
                self.assertEqual(
                    result["valid"],
                    expected,
                    f"{desc}: Expected {expected}, got {result['valid']}. "
                    f"Violations: {result.get('violations', [])}",
                )

    def test_known_bad_certificates(self):
        """Test a comprehensive set of known-bad certificates"""
        system = {
            "dynamics": ["dx/dt = -x", "dy/dt = -y"],
            "initial_set": ["x**2 + y**2 <= 0.25"],
            "unsafe_set": ["x**2 + y**2 >= 4.0"],
        }

        bad_certificates = [
            ("x**2 + y**2 - 0.1", "B = 0 inside initial set"),
            ("x**2 + y**2 - 0.25", "B = 0 at initial set boundary"),
            ("x**2 + y**2", "No level set separation (B > 0 everywhere)"),
            ("x**2 + y**2 - 5.0", "B < 0 on unsafe set"),
            ("x**2 + y**2 + 1.0", "B > 0 everywhere (wrong sign)"),
            ("-(x**2 + y**2) + 1.0", "Inverted barrier (increases outward)"),
        ]

        for cert, reason in bad_certificates:
            result = self.validator.validate_certificate_mathematically(cert, system)
            self.assertFalse(
                result["valid"], f"Bad certificate should be rejected ({reason}): {cert}"
            )


class TestLevelSetComputations(unittest.TestCase):
    """Specific tests for level set computations"""

    def setUp(self):
        self.tracker = LevelSetTracker()

    def test_circular_set_sampling(self):
        """Test sampling of circular sets"""
        # Test initial set sampling
        initial_set = ["x**2 + y**2 <= 1.0"]
        samples = self.tracker._sample_constrained_set(initial_set, ["x", "y"], 100)

        self.assertGreater(len(samples), 50, "Should get enough samples")

        # Check all samples are inside the set
        for x, y in samples:
            self.assertLessEqual(x**2 + y**2, 1.0 + 1e-6, f"Sample ({x}, {y}) outside initial set")

    def test_level_set_accuracy(self):
        """Test accuracy of level set computation"""
        # Simple test case with known values
        barrier = "x**2 + y**2"  # B(x,y) = x² + y²
        initial_set = ["x**2 + y**2 <= 1.0"]  # Circle of radius 1
        unsafe_set = ["x**2 + y**2 >= 4.0"]  # Outside circle of radius 2

        level_info = self.tracker.compute_level_sets(
            barrier, initial_set, unsafe_set, ["x", "y"], n_samples=1000
        )

        # For B = x² + y², max on x² + y² ≤ 1 is 1
        self.assertAlmostEqual(level_info.initial_max, 1.0, places=2)

        # For B = x² + y², min on x² + y² ≥ 4 is 4
        self.assertAlmostEqual(level_info.unsafe_min, 4.0, places=2)

        # Separation should be 4 - 1 = 3
        self.assertAlmostEqual(level_info.separation, 3.0, places=2)


def run_theory_compliance_tests():
    """Run all theory compliance tests"""
    # Create test suite
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TheoryComplianceTests))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLevelSetComputations))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return success
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_theory_compliance_tests()
    sys.exit(0 if success else 1)
