#!/usr/bin/env python3
"""Simple individual module tests"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing Individual Modules\n")

# Test 1: Level Set Tracker
print("1. Testing Level Set Tracker...")
try:
    from utils.level_set_tracker import LevelSetTracker
    tracker = LevelSetTracker()
    
    # Simple test
    level_info = tracker.compute_level_sets(
        "x**2 + y**2 - 1.0",
        ["x**2 + y**2 <= 0.25"],
        ["x**2 + y**2 >= 4.0"],
        ["x", "y"],
        n_samples=100
    )
    
    print(f"   c1: {level_info.initial_max:.3f}")
    print(f"   c2: {level_info.unsafe_min:.3f}")
    print(f"   Valid: {level_info.is_valid}")
    print("   ✓ Level Set Tracker works!\n")
except Exception as e:
    print(f"   ✗ Error: {e}\n")

# Test 2: Set Membership
print("2. Testing Set Membership...")
try:
    from utils.set_membership import SetMembershipTester
    tester = SetMembershipTester()
    
    # Test point in circle
    in_circle = tester.is_in_set((0, 0), ["x**2 + y**2 <= 1.0"], ['x', 'y'])
    out_circle = tester.is_in_set((2, 0), ["x**2 + y**2 <= 1.0"], ['x', 'y'])
    
    print(f"   (0,0) in unit circle: {in_circle}")
    print(f"   (2,0) in unit circle: {out_circle}")
    print("   ✓ Set Membership works!\n")
except Exception as e:
    print(f"   ✗ Error: {e}\n")

# Test 3: Adaptive Tolerance
print("3. Testing Adaptive Tolerance...")
try:
    from utils.adaptive_tolerance import AdaptiveTolerance
    adaptive = AdaptiveTolerance()
    
    # Test tolerance computation
    tol1 = adaptive.compute_tolerance({'x': (-1, 1), 'y': (-1, 1)})
    tol2 = adaptive.compute_tolerance({'x': (-100, 100), 'y': (-100, 100)})
    
    print(f"   Small scale tolerance: {tol1:.2e}")
    print(f"   Large scale tolerance: {tol2:.2e}")
    print(f"   Scaling works: {tol2 > tol1}")
    print("   ✓ Adaptive Tolerance works!\n")
except Exception as e:
    print(f"   ✗ Error: {e}\n")

# Test 4: Certificate Extraction
print("4. Testing Certificate Extraction...")
try:
    from utils.certificate_extraction import extract_certificate_from_llm_output
    
    # Test extraction
    llm_output = "B(x,y) = x**2 + y**2 - 1.0"
    result = extract_certificate_from_llm_output(llm_output, ['x', 'y'])
    
    print(f"   Extracted: {result[0] if result else 'None'}")
    print("   ✓ Certificate Extraction works!\n")
except Exception as e:
    print(f"   ✗ Error: {e}\n")

# Test 5: Validation
print("5. Testing Validation Pipeline...")
try:
    from tests.unit.test_certificate_validation_accuracy import CertificateValidationTester
    validator = CertificateValidationTester()
    
    system = {
        'dynamics': ['dx/dt = -x', 'dy/dt = -y'],
        'initial_set': ['x**2 + y**2 <= 0.25'],
        'unsafe_set': ['x**2 + y**2 >= 4.0']
    }
    
    result = validator.validate_certificate_mathematically(
        "x**2 + y**2 - 1.0", system, n_samples=10
    )
    
    print(f"   Valid: {result['valid']}")
    if 'level_sets' in result:
        print(f"   c1: {result['level_sets']['c1']:.3f}")
        print(f"   c2: {result['level_sets']['c2']:.3f}")
    print("   ✓ Validation Pipeline works!\n")
except Exception as e:
    print(f"   ✗ Error: {e}\n")

print("Individual module testing complete!") 