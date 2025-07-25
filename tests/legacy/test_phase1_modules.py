#!/usr/bin/env python3
"""
Quick test script to verify Phase 1 modules are working correctly
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing Phase 1 Module Imports...")
print("=" * 50)

# Test imports
try:
    from utils.level_set_tracker import LevelSetTracker, LevelSetInfo
    print("✓ Level set tracker imported successfully")
except Exception as e:
    print(f"✗ Level set tracker import failed: {e}")

try:
    from utils.set_membership import SetMembershipTester
    print("✓ Set membership tester imported successfully")
except Exception as e:
    print(f"✗ Set membership import failed: {e}")

try:
    from utils.adaptive_tolerance import AdaptiveTolerance, ToleranceManager
    print("✓ Adaptive tolerance imported successfully")
except Exception as e:
    print(f"✗ Adaptive tolerance import failed: {e}")

try:
    from tests.unit.test_certificate_validation_accuracy import CertificateValidationTester
    print("✓ Certificate validation tester imported successfully")
except Exception as e:
    print(f"✗ Certificate validation import failed: {e}")

print("\n" + "=" * 50)
print("Testing Basic Functionality...")
print("=" * 50)

# Test 1: Level Set Computation
print("\n1. Testing Level Set Computation:")
try:
    tracker = LevelSetTracker()
    barrier = "x**2 + y**2 - 1.0"
    initial_set = ["x**2 + y**2 <= 0.25"]
    unsafe_set = ["x**2 + y**2 >= 4.0"]
    variables = ["x", "y"]
    
    level_info = tracker.compute_level_sets(barrier, initial_set, unsafe_set, variables, n_samples=100)
    print(f"   c1 (initial max): {level_info.initial_max:.3f}")
    print(f"   c2 (unsafe min): {level_info.unsafe_min:.3f}")
    print(f"   Separation: {level_info.separation:.3f}")
    print(f"   Valid: {level_info.is_valid}")
    print("   ✓ Level set computation working")
except Exception as e:
    print(f"   ✗ Level set computation failed: {e}")

# Test 2: Set Membership
print("\n2. Testing Set Membership:")
try:
    tester = SetMembershipTester()
    constraints = ["x**2 + y**2 <= 1.0"]
    
    # Test points
    test_cases = [
        ((0, 0), True, "Origin"),
        ((1, 0), True, "Boundary"),
        ((2, 0), False, "Outside"),
    ]
    
    all_passed = True
    for point, expected, desc in test_cases:
        result = tester.is_in_set(point, constraints, ['x', 'y'])
        if result == expected:
            print(f"   ✓ {desc}: {point} → {result}")
        else:
            print(f"   ✗ {desc}: {point} → {result} (expected {expected})")
            all_passed = False
    
    if all_passed:
        print("   ✓ Set membership testing working")
except Exception as e:
    print(f"   ✗ Set membership testing failed: {e}")

# Test 3: Adaptive Tolerance
print("\n3. Testing Adaptive Tolerance:")
try:
    adaptive = AdaptiveTolerance()
    
    # Small scale
    bounds1 = {'x': (-1, 1), 'y': (-1, 1)}
    tol1 = adaptive.compute_tolerance(bounds1)
    print(f"   Small scale tolerance: {tol1:.2e}")
    
    # Large scale
    bounds2 = {'x': (-100, 100), 'y': (-100, 100)}
    tol2 = adaptive.compute_tolerance(bounds2)
    print(f"   Large scale tolerance: {tol2:.2e}")
    
    if tol2 > tol1:
        print("   ✓ Adaptive tolerance scaling working")
    else:
        print("   ✗ Tolerance not scaling with problem size")
except Exception as e:
    print(f"   ✗ Adaptive tolerance failed: {e}")

# Test 4: Complete Validation Pipeline
print("\n4. Testing Complete Validation Pipeline:")
try:
    validator = CertificateValidationTester()
    
    system = {
        'dynamics': ['dx/dt = -x', 'dy/dt = -y'],
        'initial_set': ['x**2 + y**2 <= 0.25'],
        'unsafe_set': ['x**2 + y**2 >= 4.0']
    }
    
    # Valid certificate
    valid_cert = "x**2 + y**2 - 1.0"
    result = validator.validate_certificate_mathematically(valid_cert, system, n_samples=10)
    
    if result['valid']:
        print(f"   ✓ Valid certificate accepted")
        print(f"     Level sets: c1={result['level_sets']['c1']:.3f}, c2={result['level_sets']['c2']:.3f}")
    else:
        print(f"   ✗ Valid certificate rejected: {result.get('violations', [])}")
    
    # Invalid certificate
    invalid_cert = "x**2 + y**2 - 0.1"
    result = validator.validate_certificate_mathematically(invalid_cert, system, n_samples=10)
    
    if not result['valid']:
        print(f"   ✓ Invalid certificate rejected")
        print(f"     Reason: {result.get('violations', ['Unknown'])[0]}")
    else:
        print(f"   ✗ Invalid certificate accepted")
        
except Exception as e:
    print(f"   ✗ Validation pipeline failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("Phase 1 Module Testing Complete!")
print("=" * 50) 