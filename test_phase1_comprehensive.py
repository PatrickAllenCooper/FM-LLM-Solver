#!/usr/bin/env python3
"""
Comprehensive test suite for Phase 1 implementation
Tests all modules and integration systematically
"""

import sys
import os
import traceback
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'=' * 60}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'=' * 60}{RESET}")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_error(text):
    print(f"{RED}✗ {text}{RESET}")

def print_info(text):
    print(f"{YELLOW}ℹ {text}{RESET}")

def test_dependencies():
    """Test that all required dependencies are installed"""
    print_header("Testing Dependencies")
    
    dependencies = {
        'numpy': 'NumPy',
        'sympy': 'SymPy',
        'typing': 'Typing',
        'dataclasses': 'Dataclasses'
    }
    
    all_ok = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print_success(f"{name} is installed")
        except ImportError:
            print_error(f"{name} is NOT installed")
            all_ok = False
    
    return all_ok

def test_module_imports():
    """Test that all Phase 1 modules can be imported"""
    print_header("Testing Module Imports")
    
    modules = [
        ('utils.level_set_tracker', 'LevelSetTracker, LevelSetInfo'),
        ('utils.set_membership', 'SetMembershipTester'),
        ('utils.adaptive_tolerance', 'AdaptiveTolerance, ToleranceManager'),
        ('utils.certificate_extraction', 'extract_certificate_from_llm_output'),
        ('tests.unit.test_certificate_validation_accuracy', 'CertificateValidationTester')
    ]
    
    imported_modules = {}
    all_ok = True
    
    for module_path, items in modules:
        try:
            module = __import__(module_path, fromlist=items.split(', '))
            imported_modules[module_path] = module
            print_success(f"Imported {module_path}")
        except Exception as e:
            print_error(f"Failed to import {module_path}: {str(e)}")
            all_ok = False
    
    return all_ok, imported_modules

def test_level_set_tracker():
    """Test the level set tracker functionality"""
    print_header("Testing Level Set Tracker")
    
    try:
        from utils.level_set_tracker import LevelSetTracker
        
        tracker = LevelSetTracker()
        
        # Test case 1: Valid barrier certificate
        print_info("Test 1: Valid barrier certificate")
        barrier = "x**2 + y**2 - 1.0"
        initial_set = ["x**2 + y**2 <= 0.25"]  # r <= 0.5
        unsafe_set = ["x**2 + y**2 >= 4.0"]    # r >= 2.0
        variables = ["x", "y"]
        
        level_info = tracker.compute_level_sets(
            barrier, initial_set, unsafe_set, variables, n_samples=500
        )
        
        print(f"  c1 (initial max): {level_info.initial_max:.6f}")
        print(f"  c2 (unsafe min): {level_info.unsafe_min:.6f}")
        print(f"  Separation: {level_info.separation:.6f}")
        print(f"  Valid separation: {level_info.is_valid}")
        
        # Check expected values
        if abs(level_info.initial_max - (-0.75)) < 0.01:
            print_success("c1 value correct (≈ -0.75)")
        else:
            print_error(f"c1 value incorrect: expected -0.75, got {level_info.initial_max}")
        
        if abs(level_info.unsafe_min - 3.0) < 0.01:
            print_success("c2 value correct (≈ 3.0)")
        else:
            print_error(f"c2 value incorrect: expected 3.0, got {level_info.unsafe_min}")
        
        if level_info.is_valid:
            print_success("Separation condition satisfied")
        else:
            print_error("Separation condition not satisfied")
        
        # Test case 2: Invalid barrier certificate
        print_info("\nTest 2: Invalid barrier certificate")
        barrier_invalid = "x**2 + y**2 - 0.1"
        
        level_info_invalid = tracker.compute_level_sets(
            barrier_invalid, initial_set, unsafe_set, variables, n_samples=500
        )
        
        print(f"  c1 (initial max): {level_info_invalid.initial_max:.6f}")
        print(f"  c2 (unsafe min): {level_info_invalid.unsafe_min:.6f}")
        print(f"  Valid separation: {level_info_invalid.is_valid}")
        
        if not level_info_invalid.is_valid:
            print_success("Invalid certificate correctly identified")
        else:
            print_error("Invalid certificate not detected")
        
        return True
        
    except Exception as e:
        print_error(f"Level set tracker test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_set_membership():
    """Test set membership functionality"""
    print_header("Testing Set Membership")
    
    try:
        from utils.set_membership import SetMembershipTester
        
        tester = SetMembershipTester()
        
        # Test circular set
        print_info("Test 1: Circular set membership")
        constraints = ["x**2 + y**2 <= 1.0"]
        
        test_points = [
            ((0, 0), True, "Origin"),
            ((0.5, 0.5), True, "Interior point"),
            ((1.0, 0), True, "Boundary point"),
            ((0.707, 0.707), True, "Near boundary"),
            ((2, 0), False, "Outside point"),
        ]
        
        all_passed = True
        for point, expected, desc in test_points:
            result = tester.is_in_set(point, constraints, ['x', 'y'])
            if result == expected:
                print_success(f"{desc} {point}: {result}")
            else:
                print_error(f"{desc} {point}: got {result}, expected {expected}")
                all_passed = False
        
        # Test boundary detection
        print_info("\nTest 2: Boundary detection")
        boundary_tests = [
            ((1.0, 0), True, "Exact boundary"),
            ((0.9999, 0), True, "Near boundary"),
            ((0.5, 0), False, "Interior"),
        ]
        
        for point, expected, desc in boundary_tests:
            result = tester.is_on_boundary(point, constraints, ['x', 'y'], tolerance=0.01)
            if result == expected:
                print_success(f"{desc} {point}: on_boundary={result}")
            else:
                print_error(f"{desc} {point}: got {result}, expected {expected}")
                all_passed = False
        
        # Test distance computation
        print_info("\nTest 3: Distance to set")
        dist1 = tester.distance_to_set((0, 0), constraints, ['x', 'y'])
        dist2 = tester.distance_to_set((2, 0), constraints, ['x', 'y'])
        
        print(f"  Distance from (0,0): {dist1:.3f}")
        print(f"  Distance from (2,0): {dist2:.3f}")
        
        if dist1 < 0 and dist2 > 0:
            print_success("Distance signs correct (negative inside, positive outside)")
        else:
            print_error("Distance signs incorrect")
            all_passed = False
        
        return all_passed
        
    except Exception as e:
        print_error(f"Set membership test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_adaptive_tolerance():
    """Test adaptive tolerance functionality"""
    print_header("Testing Adaptive Tolerance")
    
    try:
        from utils.adaptive_tolerance import AdaptiveTolerance, ToleranceManager
        
        adaptive = AdaptiveTolerance()
        
        # Test tolerance scaling
        print_info("Test 1: Tolerance scaling with problem size")
        
        test_cases = [
            ({'x': (-1, 1), 'y': (-1, 1)}, "Small scale (2x2)"),
            ({'x': (-10, 10), 'y': (-10, 10)}, "Medium scale (20x20)"),
            ({'x': (-100, 100), 'y': (-100, 100)}, "Large scale (200x200)"),
        ]
        
        tolerances = []
        for bounds, desc in test_cases:
            tol = adaptive.compute_tolerance(bounds)
            tolerances.append(tol)
            print(f"  {desc}: {tol:.2e}")
        
        # Check that tolerance increases with scale
        if tolerances[0] < tolerances[1] < tolerances[2]:
            print_success("Tolerance correctly scales with problem size")
        else:
            print_error("Tolerance scaling incorrect")
        
        # Test set-specific tolerances
        print_info("\nTest 2: Set-specific tolerances")
        initial_set = ["x**2 + y**2 <= 0.25"]
        unsafe_set = ["x**2 + y**2 >= 100.0"]
        
        set_tols = adaptive.compute_set_tolerance(initial_set, unsafe_set, ['x', 'y'])
        
        for key, value in set_tols.items():
            print(f"  {key}: {value:.2e}")
        
        # Test tolerance manager
        print_info("\nTest 3: Tolerance manager")
        manager = ToleranceManager()
        manager.setup_problem(initial_set, unsafe_set, ['x', 'y'], 'nonlinear')
        
        # Test validation
        test_validations = [
            (1.0, 1.00001, 'initial_set', True, "Very close values"),
            (1.0, 1.1, 'initial_set', False, "Far values"),
            (0.0, 1e-8, 'separation', True, "Near zero"),
        ]
        
        all_passed = True
        for computed, expected, check_type, should_pass, desc in test_validations:
            result = manager.validate(computed, expected, check_type)
            if result == should_pass:
                print_success(f"{desc}: {result}")
            else:
                print_error(f"{desc}: got {result}, expected {should_pass}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print_error(f"Adaptive tolerance test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_validation_pipeline():
    """Test the complete validation pipeline"""
    print_header("Testing Complete Validation Pipeline")
    
    try:
        from tests.unit.test_certificate_validation_accuracy import CertificateValidationTester
        from utils.certificate_extraction import extract_certificate_from_llm_output
        
        validator = CertificateValidationTester()
        
        # Define test system
        system = {
            'dynamics': ['dx/dt = -x', 'dy/dt = -y'],
            'initial_set': ['x**2 + y**2 <= 0.25'],
            'unsafe_set': ['x**2 + y**2 >= 4.0']
        }
        
        # Test 1: Valid certificate
        print_info("Test 1: Valid certificate validation")
        valid_cert = "x**2 + y**2 - 1.0"
        
        start_time = time.time()
        result = validator.validate_certificate_mathematically(valid_cert, system, n_samples=20)
        validation_time = time.time() - start_time
        
        print(f"  Validation completed in {validation_time:.3f} seconds")
        print(f"  Valid: {result['valid']}")
        if 'level_sets' in result:
            print(f"  Level sets: c1={result['level_sets']['c1']:.3f}, c2={result['level_sets']['c2']:.3f}")
            print(f"  Separation: {result['level_sets']['separation']:.3f}")
        
        if result['valid']:
            print_success("Valid certificate correctly accepted")
        else:
            print_error(f"Valid certificate rejected: {result.get('violations', [])}")
        
        # Test 2: Invalid certificate
        print_info("\nTest 2: Invalid certificate validation")
        invalid_cert = "x**2 + y**2 - 0.1"
        
        result2 = validator.validate_certificate_mathematically(invalid_cert, system, n_samples=20)
        
        print(f"  Valid: {result2['valid']}")
        if not result2['valid'] and result2.get('violations'):
            print(f"  First violation: {result2['violations'][0]}")
        
        if not result2['valid']:
            print_success("Invalid certificate correctly rejected")
        else:
            print_error("Invalid certificate incorrectly accepted")
        
        # Test 3: Certificate extraction
        print_info("\nTest 3: Certificate extraction from LLM output")
        llm_output = """
        I'll create a barrier certificate for this system.
        
        BARRIER_CERTIFICATE_START
        x**2 + y**2 - 1.5
        BARRIER_CERTIFICATE_END
        
        This should work well.
        """
        
        extracted = extract_certificate_from_llm_output(llm_output, ['x', 'y'])
        if extracted and extracted[0] == "x**2 + y**2 - 1.5":
            print_success(f"Certificate extracted correctly: {extracted[0]}")
        else:
            print_error(f"Certificate extraction failed: {extracted}")
        
        return result['valid'] and not result2['valid']
        
    except Exception as e:
        print_error(f"Validation pipeline test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_integration():
    """Run integration tests"""
    print_header("Testing Integration")
    
    try:
        # Test that all modules work together
        from utils.level_set_tracker import LevelSetTracker
        from utils.set_membership import SetMembershipTester
        from utils.adaptive_tolerance import ToleranceManager
        from tests.unit.test_certificate_validation_accuracy import CertificateValidationTester
        
        print_info("Testing module integration")
        
        # Create instances
        tracker = LevelSetTracker()
        set_tester = SetMembershipTester()
        tol_manager = ToleranceManager()
        validator = CertificateValidationTester()
        
        # Test system
        system = {
            'dynamics': ['dx/dt = -x + 0.1*y', 'dy/dt = -0.1*x - y'],
            'initial_set': ['x**2 + y**2 <= 0.25'],
            'unsafe_set': ['x**2 + y**2 >= 4.0']
        }
        
        # Setup tolerance
        tol_manager.setup_problem(
            system['initial_set'],
            system['unsafe_set'],
            ['x', 'y'],
            'linear'
        )
        
        # Test certificate
        cert = "x**2 + y**2 - 1.0"
        
        # Validate
        result = validator.validate_certificate_mathematically(cert, system, n_samples=20)
        
        print(f"  Integration test result: {'PASSED' if result['valid'] else 'FAILED'}")
        
        if result['valid']:
            print_success("All modules working together correctly")
            return True
        else:
            print_error("Integration test failed")
            return False
        
    except Exception as e:
        print_error(f"Integration test failed: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print(f"{BLUE}{'#' * 60}{RESET}")
    print(f"{BLUE}#  Phase 1 Comprehensive Test Suite{RESET}")
    print(f"{BLUE}{'#' * 60}{RESET}")
    
    # Track results
    results = {}
    
    # Test dependencies
    print_info("Starting dependency check...")
    results['dependencies'] = test_dependencies()
    
    if not results['dependencies']:
        print_error("Missing dependencies. Please install required packages.")
        return
    
    # Test imports
    print_info("\nChecking module imports...")
    imports_ok, modules = test_module_imports()
    results['imports'] = imports_ok
    
    if not imports_ok:
        print_error("Some modules failed to import. Check for syntax errors.")
        return
    
    # Test individual modules
    print_info("\nTesting individual modules...")
    results['level_set_tracker'] = test_level_set_tracker()
    results['set_membership'] = test_set_membership()
    results['adaptive_tolerance'] = test_adaptive_tolerance()
    results['validation_pipeline'] = test_validation_pipeline()
    results['integration'] = test_integration()
    
    # Summary
    print_header("Test Summary")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    print(f"\nTotal tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    
    print("\nDetailed results:")
    for test_name, passed in results.items():
        if passed:
            print_success(f"{test_name}")
        else:
            print_error(f"{test_name}")
    
    if passed_tests == total_tests:
        print(f"\n{GREEN}{'=' * 60}{RESET}")
        print(f"{GREEN}All tests passed! Phase 1 implementation is working correctly.{RESET}")
        print(f"{GREEN}{'=' * 60}{RESET}")
    else:
        print(f"\n{RED}{'=' * 60}{RESET}")
        print(f"{RED}Some tests failed. Please check the errors above.{RESET}")
        print(f"{RED}{'=' * 60}{RESET}")

if __name__ == "__main__":
    main() 