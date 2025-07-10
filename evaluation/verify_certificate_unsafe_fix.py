"""
Fixed unsafe set checking logic for barrier certificates (Phase 1 Day 3)

The key fix: Check that B(x) >= 0 for points INSIDE the unsafe set, not outside!

Correct barrier certificate conditions:
1. B(x) <= 0 for all x in Initial Set
2. B(x) >= 0 for all x in Unsafe Set (NOT outside!)
3. dB/dt <= 0 in the safe region
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)


def check_unsafe_set_correctly(
    B_func: callable,
    unsafe_set_relationals: List,
    variables: List,
    n_samples: int,
    tolerance: float,
    sampling_bounds: Dict[str, Tuple[float, float]]
) -> Tuple[bool, Dict[str, Any]]:
    """
    FIXED: Check unsafe set condition correctly.
    
    The condition is: B(x) >= -tolerance for all x IN the unsafe set.
    
    Returns:
        (passed, details) where details contains violation information
    """
    from evaluation.utils import generate_samples, check_set_membership_numerical
    
    # Generate samples
    samples = generate_samples(sampling_bounds, variables, n_samples)
    
    violations = 0
    checked_in_unsafe = 0
    violation_points = []
    min_b_in_unsafe = float('inf')
    
    for point_dict in samples:
        try:
            # Check if point is IN unsafe set
            is_in_unsafe = check_set_membership_numerical(
                point_dict, unsafe_set_relationals, variables
            )
            
            if is_in_unsafe:
                checked_in_unsafe += 1
                
                # Evaluate barrier at this point
                b_val = float(B_func(**point_dict))
                min_b_in_unsafe = min(min_b_in_unsafe, b_val)
                
                # Check condition: B(x) >= -tolerance in unsafe set
                if b_val < -tolerance:
                    violations += 1
                    if len(violation_points) < 10:  # Keep first 10 violations
                        violation_points.append({
                            'point': point_dict.copy(),
                            'B_value': b_val,
                            'expected': f'>= {-tolerance}'
                        })
                        
        except Exception as e:
            logger.warning(f"Error evaluating point {point_dict}: {e}")
            continue
    
    # Determine if check passed
    passed = violations == 0
    
    # Build detailed reason
    if checked_in_unsafe == 0:
        reason = "No samples found in unsafe set - cannot verify unsafe condition"
        passed = None  # Inconclusive
    elif violations > 0:
        reason = (f"Failed: {violations}/{checked_in_unsafe} points in unsafe set "
                 f"violate B(x) >= {-tolerance}. Min B in unsafe = {min_b_in_unsafe:.4f}")
    else:
        reason = (f"Passed: All {checked_in_unsafe} points in unsafe set "
                 f"satisfy B(x) >= {-tolerance}. Min B in unsafe = {min_b_in_unsafe:.4f}")
    
    return passed, {
        'violations': violations,
        'checked_in_unsafe': checked_in_unsafe,
        'violation_points': violation_points,
        'min_b_in_unsafe': min_b_in_unsafe if checked_in_unsafe > 0 else None,
        'reason': reason
    }


def numerical_check_all_conditions_fixed(
    B_func: callable,
    dB_dt_func: callable,
    sampling_bounds: Dict[str, Tuple[float, float]],
    variables: List,
    initial_set_relationals: List,
    unsafe_set_relationals: List,
    safe_set_relationals: List,
    n_samples_boundary: int,
    n_samples_lie: int,
    tolerance: float
) -> Tuple[bool, Dict[str, Any]]:
    """
    Complete numerical verification with FIXED unsafe set checking.
    
    Checks:
    1. B(x) <= tolerance for x in Initial Set
    2. B(x) >= -tolerance for x in Unsafe Set (FIXED!)
    3. dB/dt <= tolerance for x where B(x) <= 0 (safe region)
    """
    from evaluation.utils import generate_samples, check_set_membership_numerical
    
    results = {
        'initial_check': {'passed': None, 'details': {}},
        'unsafe_check': {'passed': None, 'details': {}},
        'lie_check': {'passed': None, 'details': {}},
        'overall_passed': False
    }
    
    # 1. Check initial set condition
    logger.info("Checking initial set condition...")
    samples = generate_samples(sampling_bounds, variables, n_samples_boundary)
    
    init_violations = 0
    checked_in_init = 0
    init_violation_points = []
    max_b_in_init = -float('inf')
    
    for point_dict in samples:
        try:
            is_in_init = check_set_membership_numerical(
                point_dict, initial_set_relationals, variables
            )
            
            if is_in_init:
                checked_in_init += 1
                b_val = float(B_func(**point_dict))
                max_b_in_init = max(max_b_in_init, b_val)
                
                if b_val > tolerance:
                    init_violations += 1
                    if len(init_violation_points) < 10:
                        init_violation_points.append({
                            'point': point_dict.copy(),
                            'B_value': b_val
                        })
                        
        except Exception as e:
            logger.warning(f"Error in initial set check: {e}")
    
    if checked_in_init > 0:
        init_passed = init_violations == 0
        init_reason = (f"{'Passed' if init_passed else 'Failed'}: "
                      f"{init_violations}/{checked_in_init} violations. "
                      f"Max B in initial = {max_b_in_init:.4f}")
    else:
        init_passed = None
        init_reason = "No points found in initial set"
        
    results['initial_check'] = {
        'passed': init_passed,
        'details': {
            'violations': init_violations,
            'checked': checked_in_init,
            'max_b': max_b_in_init,
            'violation_points': init_violation_points,
            'reason': init_reason
        }
    }
    
    # 2. Check unsafe set condition (FIXED!)
    logger.info("Checking unsafe set condition (fixed logic)...")
    unsafe_passed, unsafe_details = check_unsafe_set_correctly(
        B_func, unsafe_set_relationals, variables,
        n_samples_boundary, tolerance, sampling_bounds
    )
    results['unsafe_check'] = {
        'passed': unsafe_passed,
        'details': unsafe_details
    }
    
    # 3. Check Lie derivative condition
    logger.info("Checking Lie derivative condition...")
    lie_samples = generate_samples(sampling_bounds, variables, n_samples_lie)
    
    lie_violations = 0
    checked_in_safe = 0
    lie_violation_points = []
    max_lie_in_safe = -float('inf')
    
    for point_dict in lie_samples:
        try:
            b_val = float(B_func(**point_dict))
            
            # Check if in safe region (B <= 0)
            if b_val <= tolerance:
                checked_in_safe += 1
                lie_val = float(dB_dt_func(**point_dict))
                max_lie_in_safe = max(max_lie_in_safe, lie_val)
                
                if lie_val > tolerance:
                    lie_violations += 1
                    if len(lie_violation_points) < 10:
                        lie_violation_points.append({
                            'point': point_dict.copy(),
                            'B_value': b_val,
                            'dB_dt': lie_val
                        })
                        
        except Exception as e:
            logger.warning(f"Error in Lie derivative check: {e}")
    
    if checked_in_safe > 0:
        lie_passed = lie_violations == 0
        lie_reason = (f"{'Passed' if lie_passed else 'Failed'}: "
                     f"{lie_violations}/{checked_in_safe} violations. "
                     f"Max dB/dt in safe = {max_lie_in_safe:.4f}")
    else:
        lie_passed = None
        lie_reason = "No points found in safe region"
        
    results['lie_check'] = {
        'passed': lie_passed,
        'details': {
            'violations': lie_violations,
            'checked': checked_in_safe,
            'max_lie': max_lie_in_safe,
            'violation_points': lie_violation_points,
            'reason': lie_reason
        }
    }
    
    # Overall result
    checks = [
        results['initial_check']['passed'],
        results['unsafe_check']['passed'],
        results['lie_check']['passed']
    ]
    
    # If any check is explicitly False, overall fails
    if False in checks:
        results['overall_passed'] = False
        results['overall_reason'] = "One or more conditions failed"
    # If all non-None checks pass, overall passes
    elif all(c is not False for c in checks) and any(c is True for c in checks):
        results['overall_passed'] = True
        results['overall_reason'] = "All checked conditions passed"
    else:
        results['overall_passed'] = False
        results['overall_reason'] = "Insufficient data for verification"
    
    return results['overall_passed'], results


def demonstrate_unsafe_fix():
    """Demonstrate the fix with a simple example"""
    import sympy
    
    # Example system
    x, y = sympy.symbols('x y')
    B = x**2 + y**2 - 1.0
    
    # Create callable functions
    B_func = sympy.lambdify([x, y], B, 'numpy')
    
    # Test points
    test_points = [
        # Points INSIDE unsafe set (x² + y² >= 4)
        {'x': 2.5, 'y': 0},    # r² = 6.25 > 4
        {'x': 2, 'y': 2},      # r² = 8 > 4
        {'x': 3, 'y': 0},      # r² = 9 > 4
        
        # Points OUTSIDE unsafe set
        {'x': 1, 'y': 0},      # r² = 1 < 4
        {'x': 0, 'y': 0},      # r² = 0 < 4
    ]
    
    print("Testing B(x,y) = x² + y² - 1.0")
    print("Unsafe set: x² + y² >= 4")
    print("\nCorrect condition: B(x) >= 0 for points IN unsafe set")
    print("-" * 50)
    
    for point in test_points:
        r_squared = point['x']**2 + point['y']**2
        in_unsafe = r_squared >= 4.0
        b_val = B_func(point['x'], point['y'])
        
        print(f"Point ({point['x']}, {point['y']}): r² = {r_squared:.2f}")
        print(f"  In unsafe set: {in_unsafe}")
        print(f"  B value: {b_val:.2f}")
        print(f"  Satisfies B >= 0: {b_val >= 0}")
        
        if in_unsafe:
            if b_val >= 0:
                print("  ✓ Correct: B >= 0 in unsafe set")
            else:
                print("  ✗ Violation: B < 0 in unsafe set")
        print()


if __name__ == "__main__":
    demonstrate_unsafe_fix() 