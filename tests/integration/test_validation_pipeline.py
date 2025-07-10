#!/usr/bin/env python3
"""
Integration Tests for Validation Pipeline
Tests the complete pipeline from certificate extraction to validation
"""

import unittest
import sys
import os
import time
import json
import numpy as np
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.certificate_extraction import extract_certificate_from_llm_output
from utils.level_set_tracker import LevelSetTracker
from utils.set_membership import SetMembershipTester
from utils.adaptive_tolerance import AdaptiveTolerance, ToleranceManager
from tests.unit.test_certificate_validation_accuracy import CertificateValidationTester


class ValidationPipelineIntegrationTests(unittest.TestCase):
    """Integration tests for the complete validation pipeline"""
    
    def setUp(self):
        self.extractor = extract_certificate_from_llm_output
        self.validator = CertificateValidationTester()
        self.level_tracker = LevelSetTracker()
        self.set_tester = SetMembershipTester()
        self.tolerance_manager = ToleranceManager()
        
    def test_complete_pipeline_linear_system(self):
        """Test complete pipeline with a linear system"""
        # Define system
        system = {
            'dynamics': ['dx/dt = -x', 'dy/dt = -y'],
            'initial_set': ['x**2 + y**2 <= 0.25'],
            'unsafe_set': ['x**2 + y**2 >= 4.0']
        }
        variables = ['x', 'y']
        
        # Simulate LLM output
        llm_output = """
        Based on the system dynamics, I'll construct a barrier certificate.
        
        BARRIER_CERTIFICATE_START
        x**2 + y**2 - 1.0
        BARRIER_CERTIFICATE_END
        
        This certificate creates a barrier at radius 1.0.
        """
        
        # Step 1: Extract certificate
        extracted = self.extractor(llm_output, variables)
        self.assertIsNotNone(extracted)
        self.assertEqual(extracted[0], "x**2 + y**2 - 1.0")
        
        # Step 2: Compute level sets
        level_info = self.level_tracker.compute_level_sets(
            extracted[0],
            system['initial_set'],
            system['unsafe_set'],
            variables
        )
        
        # Verify level sets
        self.assertAlmostEqual(level_info.initial_max, -0.75, places=2)
        self.assertAlmostEqual(level_info.unsafe_min, 3.0, places=2)
        self.assertTrue(level_info.is_valid)
        
        # Step 3: Validate certificate
        validation_result = self.validator.validate_certificate_mathematically(
            extracted[0], system
        )
        
        self.assertTrue(validation_result['valid'])
        self.assertEqual(len(validation_result.get('violations', [])), 0)
        
    def test_pipeline_with_invalid_certificate(self):
        """Test pipeline correctly rejects invalid certificates"""
        system = {
            'dynamics': ['dx/dt = -x', 'dy/dt = -y'],
            'initial_set': ['x**2 + y**2 <= 0.25'],
            'unsafe_set': ['x**2 + y**2 >= 4.0']
        }
        variables = ['x', 'y']
        
        # Invalid certificate - no separation
        llm_output = "B(x,y) = x**2 + y**2 - 0.1"
        
        # Extract
        extracted = self.extractor(llm_output, variables)
        self.assertEqual(extracted[0], "x**2 + y**2 - 0.1")
        
        # Compute level sets
        level_info = self.level_tracker.compute_level_sets(
            extracted[0],
            system['initial_set'],
            system['unsafe_set'],
            variables
        )
        
        # Should have invalid separation
        self.assertFalse(level_info.is_valid)
        
        # Validate - should fail
        validation_result = self.validator.validate_certificate_mathematically(
            extracted[0], system
        )
        self.assertFalse(validation_result['valid'])
        
    def test_pipeline_with_adaptive_tolerance(self):
        """Test pipeline with adaptive tolerance handling"""
        # Large-scale system
        system = {
            'dynamics': ['dx/dt = -x', 'dy/dt = -y'],
            'initial_set': ['x**2 + y**2 <= 100.0'],  # r = 10
            'unsafe_set': ['x**2 + y**2 >= 10000.0']  # r = 100
        }
        variables = ['x', 'y']
        
        # Setup tolerance manager
        self.tolerance_manager.setup_problem(
            system['initial_set'],
            system['unsafe_set'],
            variables,
            'linear'
        )
        
        # Certificate at intermediate scale
        certificate = "x**2 + y**2 - 1000.0"  # r = 31.6
        
        # Validate with adaptive tolerance
        validation_result = self.validator.validate_certificate_mathematically(
            certificate, system
        )
        
        self.assertTrue(validation_result['valid'])
        
        # Check that appropriate tolerances were used
        initial_tol = self.tolerance_manager.get_tolerance('initial_set')
        self.assertGreater(initial_tol, 1e-6)  # Should be scaled up
        
    def test_pipeline_with_set_membership(self):
        """Test set membership integration"""
        system = {
            'dynamics': ['dx/dt = -x - y', 'dy/dt = x - y'],
            'initial_set': ['x**2 + y**2 <= 0.25'],
            'unsafe_set': ['x >= 2.0', 'y >= 2.0']  # Box constraint
        }
        variables = ['x', 'y']
        
        # Test points
        test_points = [
            ((0, 0), True, False),      # Origin: in initial, not unsafe
            ((0.5, 0), True, False),    # Boundary of initial
            ((3, 3), False, True),      # In unsafe set
            ((1, 1), False, False),     # In neither
        ]
        
        for point, expected_initial, expected_unsafe in test_points:
            in_initial = self.set_tester.is_in_set(
                point, system['initial_set'], variables
            )
            in_unsafe = self.set_tester.is_in_set(
                point, system['unsafe_set'], variables
            )
            
            self.assertEqual(in_initial, expected_initial,
                           f"Point {point} initial set membership")
            self.assertEqual(in_unsafe, expected_unsafe,
                           f"Point {point} unsafe set membership")
            
    def test_pipeline_performance(self):
        """Test pipeline performance on various system sizes"""
        results = []
        
        for dim in [2, 3]:
            if dim == 2:
                system = {
                    'dynamics': ['dx/dt = -x', 'dy/dt = -y'],
                    'initial_set': ['x**2 + y**2 <= 0.25'],
                    'unsafe_set': ['x**2 + y**2 >= 4.0']
                }
                variables = ['x', 'y']
                certificate = "x**2 + y**2 - 1.0"
            else:
                system = {
                    'dynamics': ['dx/dt = -x', 'dy/dt = -y', 'dz/dt = -z'],
                    'initial_set': ['x**2 + y**2 + z**2 <= 0.25'],
                    'unsafe_set': ['x**2 + y**2 + z**2 >= 4.0']
                }
                variables = ['x', 'y', 'z']
                certificate = "x**2 + y**2 + z**2 - 1.0"
                
            start_time = time.time()
            
            # Run validation
            validation_result = self.validator.validate_certificate_mathematically(
                certificate, system, n_samples=50
            )
            
            elapsed = time.time() - start_time
            
            results.append({
                'dimension': dim,
                'time': elapsed,
                'valid': validation_result['valid']
            })
            
            # Performance requirement: < 1 second
            self.assertLess(elapsed, 1.0,
                          f"{dim}D validation took {elapsed:.3f}s")
            
        return results
    
    def test_pipeline_robustness(self):
        """Test pipeline robustness to various inputs"""
        system = {
            'dynamics': ['dx/dt = -x', 'dy/dt = -y'],
            'initial_set': ['x**2 + y**2 <= 0.25'],
            'unsafe_set': ['x**2 + y**2 >= 4.0']
        }
        variables = ['x', 'y']
        
        # Test various certificate formats
        test_cases = [
            # Format variations
            ("B(x,y) = x**2 + y**2 - 1.0", True),
            ("Certificate: x**2 + y**2 - 1.0", True),
            ("x^2 + y^2 - 1.0", False),  # Wrong power notation
            ("x**2 + y**2 - 1", True),   # Integer constant
            ("1.0*x**2 + 1.0*y**2 - 1.0", True),  # Explicit coefficients
            
            # Edge cases
            ("x**2 + y**2 - 1.000000001", True),  # High precision
            ("x**2+y**2-1.0", True),  # No spaces
            ("  x**2 + y**2 - 1.0  ", True),  # Extra spaces
        ]
        
        for llm_output, should_extract in test_cases:
            extracted = self.extractor(llm_output, variables)
            
            if should_extract:
                self.assertIsNotNone(extracted,
                                   f"Failed to extract from: {llm_output}")
                # Normalize and compare
                normalized = extracted[0].replace(' ', '')
                self.assertIn('x**2+y**2', normalized)
            else:
                self.assertIsNone(extracted,
                                f"Should not extract from: {llm_output}")
    
    def test_pipeline_with_nonlinear_system(self):
        """Test pipeline with nonlinear dynamics"""
        system = {
            'dynamics': ['dx/dt = -x + x**3', 'dy/dt = -y'],
            'initial_set': ['x**2 + y**2 <= 0.25'],
            'unsafe_set': ['x**2 + y**2 >= 4.0']
        }
        variables = ['x', 'y']
        
        # This certificate should work for the nonlinear system
        certificate = "x**2 + y**2 - 1.0"
        
        # Validate
        validation_result = self.validator.validate_certificate_mathematically(
            certificate, system
        )
        
        # Should be valid (Lie derivative is negative in critical region)
        self.assertTrue(validation_result['valid'])
        
    def test_boundary_cases(self):
        """Test pipeline behavior on boundary cases"""
        system = {
            'dynamics': ['dx/dt = -x', 'dy/dt = -y'],
            'initial_set': ['x**2 + y**2 <= 0.25'],
            'unsafe_set': ['x**2 + y**2 >= 4.0']
        }
        variables = ['x', 'y']
        
        # Test boundary sampling
        initial_boundary = self.set_tester.sample_boundary(
            system['initial_set'], variables, n_samples=8
        )
        
        # Check all points are on boundary
        for point in initial_boundary:
            dist = point[0]**2 + point[1]**2
            self.assertAlmostEqual(dist, 0.25, places=5)
            
        # Test boundary detection
        boundary_point = (0.5, 0)  # On initial set boundary
        is_boundary = self.set_tester.is_on_boundary(
            boundary_point, system['initial_set'], variables
        )
        self.assertTrue(is_boundary)
        
        interior_point = (0.1, 0.1)  # Inside initial set
        is_boundary = self.set_tester.is_on_boundary(
            interior_point, system['initial_set'], variables
        )
        self.assertFalse(is_boundary)


class TestPipelineEndToEnd(unittest.TestCase):
    """End-to-end tests for complete scenarios"""
    
    def test_successful_verification_scenario(self):
        """Test a complete successful verification scenario"""
        # 1. System definition
        problem = {
            'system': {
                'dynamics': ['dx/dt = -x - y', 'dy/dt = x - y'],
                'initial_set': ['x**2 + y**2 <= 0.25'],
                'unsafe_set': ['x**2 + y**2 >= 4.0']
            },
            'variables': ['x', 'y']
        }
        
        # 2. LLM generates certificate
        llm_response = """
        I'll analyze this system and generate a barrier certificate.
        
        The system has stable dynamics with eigenvalues having negative real parts.
        A suitable Lyapunov-like barrier certificate is:
        
        BARRIER_CERTIFICATE_START
        x**2 + y**2 - 1.0
        BARRIER_CERTIFICATE_END
        
        This separates the initial set (r ≤ 0.5) from unsafe set (r ≥ 2.0).
        """
        
        # 3. Extract certificate
        extractor = extract_certificate_from_llm_output
        extracted = extractor(llm_response, problem['variables'])
        
        self.assertIsNotNone(extracted)
        certificate = extracted[0]
        
        # 4. Full validation
        validator = CertificateValidationTester()
        result = validator.validate_certificate_mathematically(
            certificate, problem['system']
        )
        
        # 5. Check results
        self.assertTrue(result['valid'])
        self.assertIn('level_sets', result)
        self.assertTrue(result['level_sets']['valid_separation'])
        
        # 6. Generate report
        report = {
            'status': 'VERIFIED',
            'certificate': certificate,
            'level_sets': result['level_sets'],
            'validation_time': 'N/A',
            'confidence': 'HIGH'
        }
        
        self.assertEqual(report['status'], 'VERIFIED')
        
    def test_failed_verification_scenario(self):
        """Test a complete failed verification scenario"""
        problem = {
            'system': {
                'dynamics': ['dx/dt = x', 'dy/dt = y'],  # Unstable!
                'initial_set': ['x**2 + y**2 <= 0.25'],
                'unsafe_set': ['x**2 + y**2 >= 4.0']
            },
            'variables': ['x', 'y']
        }
        
        # LLM might still try standard certificate
        llm_response = "B(x,y) = x**2 + y**2 - 1.0"
        
        # Extract
        extracted = extract_certificate_from_llm_output(
            llm_response, problem['variables']
        )
        certificate = extracted[0]
        
        # Validate - should fail due to positive Lie derivative
        validator = CertificateValidationTester()
        result = validator.validate_certificate_mathematically(
            certificate, problem['system']
        )
        
        self.assertFalse(result['valid'])
        self.assertGreater(len(result.get('violations', [])), 0)
        
        # Check that Lie derivative violations are detected
        lie_violations = [v for v in result['violations'] 
                         if 'Lie derivative' in v]
        self.assertGreater(len(lie_violations), 0)


def run_integration_tests():
    """Run all integration tests"""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(
        ValidationPipelineIntegrationTests
    ))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(
        TestPipelineEndToEnd
    ))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1) 