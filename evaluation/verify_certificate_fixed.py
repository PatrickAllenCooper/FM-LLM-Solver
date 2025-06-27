#!/usr/bin/env python3
"""
FIXED Barrier Certificate Verification System

This fixes the critical boundary detection bug where the verification system
was incorrectly checking B(x) >= 0 for points OUTSIDE the unsafe set instead
of INSIDE the unsafe set.

Fixed Logic:
- B(x) <= 0 on initial set ✓
- B(x) >= 0 on points INSIDE unsafe set ✓ (was checking OUTSIDE - BUG!)
- dB/dt <= 0 in safe region ✓
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from omegaconf import DictConfig

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# Import all the existing verification functions
from evaluation.verify_certificate import *

def numerical_check_boundary_FIXED(B_func, sampling_bounds, variables, initial_set_relationals, 
                                   unsafe_set_relationals, n_samples, tolerance):
    """
    FIXED: Numerical check of boundary conditions with correct barrier certificate logic.
    
    Correct Logic:
    - Check B(x) <= tolerance for points INSIDE initial set
    - Check B(x) >= -tolerance for points INSIDE unsafe set (NOT outside!)
    """
    
    # Generate samples
    x_samples_raw = generate_samples(sampling_bounds, variables, n_samples)
    
    # 1. Initial Set Check: B(x) <= tolerance for points IN initial set
    init_violations = 0
    checked_in_init = 0
    init_violation_points = []
    
    if initial_set_relationals:
        for i, x_sample in enumerate(x_samples_raw):
            point_dict = {var.name: val for var, val in zip(variables, x_sample)}
            
            # Check if point is IN initial set
            if check_set_membership_numerical(point_dict, initial_set_relationals, variables):
                checked_in_init += 1
                try:
                    b_val = B_func(**point_dict)
                    if b_val > tolerance:  # Violation: B should be <= tolerance in initial set
                        init_violations += 1
                        if len(init_violation_points) < 20:
                            init_violation_points.append((point_dict.copy(), b_val))
                except Exception as e:
                    logging.warning(f"Error evaluating B at initial set point {point_dict}: {e}")
    
    # 2. FIXED: Unsafe Set Check: B(x) >= -tolerance for points IN unsafe set
    unsafe_violations = 0 
    checked_in_unsafe = 0  # FIXED: Now checking points IN unsafe set
    unsafe_violation_points = []
    
    if unsafe_set_relationals:
        for i, x_sample in enumerate(x_samples_raw):
            point_dict = {var.name: val for var, val in zip(variables, x_sample)}
            
            # FIXED: Check if point is IN unsafe set (not outside!)
            if check_set_membership_numerical(point_dict, unsafe_set_relationals, variables):
                checked_in_unsafe += 1
                try:
                    b_val = B_func(**point_dict)
                    if b_val < -tolerance:  # FIXED: Violation: B should be >= -tolerance in unsafe set
                        unsafe_violations += 1
                        if len(unsafe_violation_points) < 20:
                            unsafe_violation_points.append((point_dict.copy(), b_val))
                except Exception as e:
                    logging.warning(f"Error evaluating B at unsafe set point {point_dict}: {e}")
    
    # Logging with corrected logic
    if init_violations > 0:
        logging.info(f"Found {init_violations}/{checked_in_init} initial set violations (B > {tolerance})")
    if unsafe_violations > 0:
        logging.info(f"Found {unsafe_violations}/{checked_in_unsafe} unsafe set violations (B < {-tolerance}) IN unsafe set")
    
    # Build reason with corrected logic
    reason = []
    boundary_ok = True
    
    if initial_set_relationals:
        if checked_in_init > 0:
            if init_violations > 0:
                boundary_ok = False
                reason.append(f"Failed Initial Set ({init_violations}/{checked_in_init} violates B <= {tolerance}).")
            else:
                reason.append(f"Passed Initial Set ({checked_in_init} samples).")
        else:
            reason.append("Initial Set check skipped (no samples in set).")
    else:
        reason.append("Initial Set check skipped (no conditions).")

    if unsafe_set_relationals:
        if checked_in_unsafe > 0:  # FIXED: Now reporting samples IN unsafe set
            if unsafe_violations > 0:
                boundary_ok = False
                reason.append(f"Failed Unsafe Set ({unsafe_violations}/{checked_in_unsafe} violates B >= {-tolerance} IN unsafe set).")
            else:
                reason.append(f"Passed Unsafe Set ({checked_in_unsafe} samples IN unsafe set).")
        else:
            reason.append("Unsafe Set check skipped (no samples in unsafe set).")
    else:
        reason.append("Unsafe Set check skipped (no conditions).")

    if not reason:
        reason.append("No boundary checks performed.")
    
    result = {
        'boundary_ok': boundary_ok,
        'reason': " | ".join(reason),
        'init_violations': init_violations,
        'unsafe_violations': unsafe_violations,
        'init_violation_points': init_violation_points[:10],
        'unsafe_violation_points': unsafe_violation_points[:10],
        'checked_in_init': checked_in_init,
        'checked_in_unsafe': checked_in_unsafe  # FIXED: Now tracks samples IN unsafe set
    }

    return boundary_ok, result


def verify_barrier_certificate_FIXED(candidate_B_str: str, system_info: dict, verification_cfg: DictConfig):
    """
    FIXED version of barrier certificate verification with correct boundary logic.
    """
    
    logging.info(f"--- Verifying Candidate: {candidate_B_str} ---")
    
    start_time = time.time()
    
    # Initialize result structure
    result = {
        "overall_success": False,
        "parsing": {"success": False, "time": 0},
        "sos_verification": {"success": False, "attempted": False, "time": 0},
        "numerical_verification": {"success": False, "time": 0},
        "symbolic_verification": {"success": False, "attempted": False, "time": 0},
        "reason": "Not started"
    }
    
    try:
        # Parse and extract system information
        variables_list = system_info.get('state_variables', system_info.get('variables', ['x', 'y']))
        dynamics_str_list = system_info.get('dynamics', [])
        initial_conditions = system_info.get('initial_set_conditions', [])
        unsafe_conditions = system_info.get('unsafe_set_conditions', [])
        safe_conditions = system_info.get('safe_set_conditions', [])
        sampling_bounds = system_info.get('sampling_bounds', {})
        
        # Set default sampling bounds if not provided
        for var in variables_list:
            if var not in sampling_bounds:
                sampling_bounds[var] = (-2.0, 2.0)
        
        # Convert to sympy variables
        variables_sympy = [sympy.Symbol(var) for var in variables_list]
        
        logging.info(f"System ID: {system_info.get('system_id', 'N/A')}")
        
        # Detect system type
        system_type = detect_system_type(dynamics_str_list)
        logging.info(f"Detected system type: {system_type}")
        
        # Parse expressions
        parsing_start = time.time()
        
        try:
            B = parse_expression(candidate_B_str, variables_sympy)
            logging.info(f"Successfully parsed candidate B(x) = {B} with state variables {variables_list}")
            
            # Substitute system parameters if present 
            system_params = system_info.get('system_parameters', {})
            if system_params:
                param_subs = {sympy.Symbol(k): v for k, v in system_params.items()}
                B = B.subs(param_subs)
                logging.info(f"B(x) after substituting system parameters = {B}")
            
        except Exception as e:
            error_msg = f"Failed to parse barrier certificate candidate '{candidate_B_str}': {e}"
            logging.error(error_msg)
            result["parsing"]["error"] = error_msg
            result["reason"] = error_msg
            return result
        
        # Parse set conditions
        try:
            initial_set_relationals = parse_set_conditions(initial_conditions, variables_sympy)
            unsafe_set_relationals = parse_set_conditions(unsafe_conditions, variables_sympy) 
            safe_set_relationals = parse_set_conditions(safe_conditions, variables_sympy)
            
            logging.info("Parsed set conditions successfully.")
            
        except Exception as e:
            error_msg = f"Failed to parse set conditions: {e}"
            logging.error(error_msg)
            result["parsing"]["error"] = error_msg
            result["reason"] = error_msg
            return result
        
        result["parsing"]["success"] = True
        result["parsing"]["time"] = time.time() - parsing_start
        
        # Calculate dynamics-related expressions
        if system_type == 'discrete':
            # Handle discrete-time systems
            try:
                next_state_functions = parse_discrete_dynamics(dynamics_str_list, variables_sympy)
                delta_B = calculate_discrete_difference(B, variables_sympy, next_state_functions)
                
                if delta_B is None:
                    raise ValueError("Failed to calculate discrete difference B(f(x)) - B(x)")
                
                logging.info(f"Symbolic B(f(x)) - B(x) = {delta_B}")
                
                # For discrete systems, we check if B(f(x)) - B(x) <= 0
                dB_dt = delta_B  # Use delta_B as the "derivative" for discrete systems
                
            except Exception as e:
                error_msg = f"Error processing discrete-time system: {e}"
                logging.error(error_msg)
                result["reason"] = error_msg
                return result
        else:
            # Handle continuous-time systems
            try:
                dB_dt = calculate_lie_derivative(B, variables_sympy, dynamics_str_list)
                if dB_dt is None:
                    raise ValueError("Failed to calculate Lie derivative")
                
                logging.info(f"Symbolic dB/dt = {dB_dt}")
                
            except Exception as e:
                error_msg = f"Error calculating Lie derivative: {e}"
                logging.error(error_msg)
                result["reason"] = error_msg
                return result
        
        # Check if system is suitable for SOS
        is_polynomial = check_polynomial(B, dB_dt, *initial_set_relationals, *unsafe_set_relationals, 
                                       *safe_set_relationals, variables=variables_sympy)
        logging.info(f"System is polynomial and suitable for SOS: {is_polynomial}")
        
        # SOS Verification (if applicable)
        sos_verification_attempted = False
        if is_polynomial and verification_cfg.get('enable_sos', True):
            sos_start = time.time()
            result["sos_verification"]["attempted"] = True
            sos_verification_attempted = True
            
            try:
                logging.info("Attempting SOS verification...")
                
                # Convert relationals to polynomials for SOS
                initial_polys = relationals_to_polynomials(initial_set_relationals, variables_sympy)
                unsafe_polys = relationals_to_polynomials(unsafe_set_relationals, variables_sympy)
                safe_polys = relationals_to_polynomials(safe_set_relationals, variables_sympy)
                
                sos_degree = verification_cfg.get('sos_degree', 2)
                sos_solver = verification_cfg.get('sos_solver', 'MOSEK')
                
                sos_passed, sos_reason, sos_details = verify_sos(
                    B, dB_dt, initial_polys, unsafe_polys, safe_polys, 
                    variables_sympy, sos_degree, sos_solver
                )
                
                result["sos_verification"]["success"] = sos_passed if sos_passed is not None else False
                result["sos_verification"]["reason"] = sos_reason
                result["sos_verification"]["details"] = sos_details
                result["sos_verification"]["time"] = time.time() - sos_start
                
                logging.info(f"SOS Verification Result: Passed={sos_passed}, Reason={sos_reason}")
                
            except Exception as e:
                logging.error(f"SOS verification failed with error: {e}")
                result["sos_verification"]["success"] = False
                result["sos_verification"]["error"] = str(e)
                result["sos_verification"]["time"] = time.time() - sos_start
        
        # Numerical Verification - USING FIXED BOUNDARY CHECK
        numerical_start = time.time()
        
        try:
            # Create lambda functions for numerical evaluation
            B_func = lambdify_expression(B, variables_sympy)
            dB_dt_func = lambdify_expression(dB_dt, variables_sympy) if dB_dt else None
            
            # Extract numerical parameters
            n_samples_lie = verification_cfg.get('num_samples_lie', 5000)
            n_samples_boundary = verification_cfg.get('num_samples_boundary', 2000)
            tolerance = verification_cfg.get('numerical_tolerance', 1e-6)
            
            # Check Lie derivative / discrete difference
            if system_type == 'discrete':
                lie_ok, lie_result = numerical_check_discrete_difference(
                    dB_dt_func, sampling_bounds, variables_sympy, safe_set_relationals, 
                    n_samples_lie, tolerance
                )
            else:
                lie_ok, lie_result = numerical_check_lie_derivative(
                    dB_dt_func, sampling_bounds, variables_sympy, safe_set_relationals,
                    n_samples_lie, tolerance
                )
            
            # FIXED: Use corrected boundary check
            boundary_ok, boundary_result = numerical_check_boundary_FIXED(
                B_func, sampling_bounds, variables_sympy, initial_set_relationals,
                unsafe_set_relationals, n_samples_boundary, tolerance
            )
            
            numerical_passed = lie_ok and boundary_ok
            
            result["numerical_verification"]["success"] = numerical_passed
            result["numerical_verification"]["lie_derivative"] = lie_result
            result["numerical_verification"]["boundary_conditions"] = boundary_result
            result["numerical_verification"]["time"] = time.time() - numerical_start
            
            # Construct detailed reason
            reasons = []
            if system_type == 'discrete':
                lie_status = "Passed" if lie_ok else "Failed"
                reasons.append(f"Discrete Difference: {lie_status}")
            else:
                lie_status = "Passed" if lie_ok else "Failed" 
                reasons.append(f"Lie Derivative: {lie_status}")
            
            boundary_status = "Passed" if boundary_ok else "Failed"
            reasons.append(f"Boundary: {boundary_status}")
            
            if lie_result.get('reason'):
                reasons.append(lie_result['reason'])
            if boundary_result.get('reason'):
                reasons.append(boundary_result['reason'])
            
            result["numerical_verification"]["reason"] = " | ".join(reasons)
            
        except Exception as e:
            logging.error(f"Numerical verification failed: {e}")
            result["numerical_verification"]["success"] = False
            result["numerical_verification"]["error"] = str(e)
            result["numerical_verification"]["time"] = time.time() - numerical_start
        
        # Optimization-based falsification
        if verification_cfg.get('enable_optimization', True):
            try:
                opt_max_iter = verification_cfg.get('optimization_max_iter', 50)
                opt_pop_size = verification_cfg.get('optimization_pop_size', 15)
                
                violation_found, opt_results = optimization_based_falsification(
                    B_func, dB_dt_func, sampling_bounds, variables_sympy,
                    initial_set_relationals, unsafe_set_relationals, safe_set_relationals,
                    opt_max_iter, opt_pop_size, tolerance
                )
                
                result["optimization_verification"] = {
                    "violation_found": violation_found,
                    "results": opt_results
                }
                
                # Update numerical verification with optimization results
                if violation_found:
                    result["numerical_verification"]["success"] = False
                    opt_reason = "Optimization found counterexample!"
                    if result["numerical_verification"].get("reason"):
                        result["numerical_verification"]["reason"] += f" | {opt_reason}"
                    else:
                        result["numerical_verification"]["reason"] = opt_reason
                
            except Exception as e:
                logging.warning(f"Optimization-based falsification failed: {e}")
        
        # Determine overall success
        numerical_ok = result["numerical_verification"]["success"]
        sos_ok = result["sos_verification"]["success"] if sos_verification_attempted else True
        
        overall_success = numerical_ok and sos_ok
        result["overall_success"] = overall_success
        
        # Final reason
        if overall_success:
            if sos_verification_attempted:
                result["reason"] = "Passed both SOS and numerical verification"
            else:
                result["reason"] = "Passed numerical verification"
        else:
            failure_reasons = []
            if not numerical_ok:
                failure_reasons.append("Failed numerical checks")
            if sos_verification_attempted and not sos_ok:
                failure_reasons.append("Failed SOS verification")
            result["reason"] = ". ".join(failure_reasons) + "."
            
            # Add specific failure details
            if result["numerical_verification"].get("reason"):
                result["reason"] += f" Reason: {result['numerical_verification']['reason']}"
        
        total_time = time.time() - start_time
        result["total_verification_time"] = total_time
        
        logging.info(f"Final Verdict: {'Passed' if overall_success else 'Failed'}. Reason: {result['reason']}")
        
        return result
        
    except Exception as e:
        logging.error(f"Verification failed with unexpected error: {e}", exc_info=True)
        result["overall_success"] = False
        result["reason"] = f"Verification error: {str(e)}"
        result["total_verification_time"] = time.time() - start_time
        return result


def test_fixed_verification():
    """Test the fixed verification system with theory-correct certificates."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    print("🔧 Testing FIXED Barrier Certificate Verification System")
    print("=" * 60)
    
    # Load configuration
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from utils.config_loader import load_config
    config = load_config("config.yaml")
    
    # Test cases with corrected certificates
    test_cases = [
        {
            "name": "radial_separation_FIXED",
            "certificate": "x**2 + y**2 - 2.0",
            "system_info": {
                'variables': ['x', 'y'],
                'dynamics': ['-0.5*x', '-0.5*y'],
                'initial_set_conditions': ['x**2 + y**2 - 0.25'],  # x² + y² ≤ 0.25
                'unsafe_set_conditions': ['4.0 - x**2 - y**2'],   # x² + y² ≥ 4.0 → 4-x²-y² ≤ 0
                'safe_set_conditions': ['x**2 + y**2 - 4.0'],      # Safe: x² + y² < 4.0 → x²+y²-4 < 0
                'sampling_bounds': {'x': (-3, 3), 'y': (-3, 3)}
            },
            "expected_success": True
        },
        
        {
            "name": "scaled_separation_FIXED", 
            "certificate": "x**2 + y**2 - 1.25",
            "system_info": {
                'variables': ['x', 'y'],
                'dynamics': ['-x', '-2*y'],
                'initial_set_conditions': ['x**2 + y**2 - 0.5'],   # x² + y² ≤ 0.5
                'unsafe_set_conditions': ['2.0 - x**2 - y**2'],    # x² + y² ≥ 2.0 → 2-x²-y² ≤ 0
                'safe_set_conditions': ['x**2 + y**2 - 2.0'],       # Safe: x² + y² < 2.0
                'sampling_bounds': {'x': (-2, 2), 'y': (-2, 2)}
            },
            "expected_success": True
        }
    ]
    
    # Verification configuration
    verification_cfg = DictConfig({
        'num_samples_lie': 200,
        'num_samples_boundary': 100,
        'numerical_tolerance': 1e-6,
        'enable_sos': True,
        'sos_degree': 2,
        'sos_solver': 'SCS',
        'enable_optimization': True,
        'optimization_max_iter': 30,
        'optimization_pop_size': 10
    })
    
    results = {}
    successful_tests = 0
    
    for test_case in test_cases:
        print(f"\n🧪 Testing: {test_case['name']}")
        print(f"   Certificate: {test_case['certificate']}")
        
        try:
            result = verify_barrier_certificate_FIXED(
                test_case['certificate'],
                test_case['system_info'], 
                verification_cfg
            )
            
            success = result['overall_success']
            expected = test_case['expected_success']
            correct_prediction = success == expected
            
            results[test_case['name']] = {
                'success': success,
                'expected': expected,
                'correct_prediction': correct_prediction,
                'reason': result.get('reason', 'N/A'),
                'sos_passed': result.get('sos_verification', {}).get('success', False),
                'numerical_passed': result.get('numerical_verification', {}).get('success', False)
            }
            
            if correct_prediction:
                successful_tests += 1
                status = "✅ CORRECT"
            else:
                status = "❌ WRONG"
            
            print(f"   Result: {status} ({'PASS' if success else 'FAIL'}, expected {expected})")
            print(f"   SOS: {'✅' if result.get('sos_verification', {}).get('success', False) else '❌'}")
            print(f"   Numerical: {'✅' if result.get('numerical_verification', {}).get('success', False) else '❌'}")
            print(f"   Reason: {result.get('reason', 'N/A')}")
            
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            results[test_case['name']] = {
                'success': False,
                'expected': test_case['expected_success'],
                'correct_prediction': False,
                'error': str(e)
            }
    
    # Summary
    success_rate = successful_tests / len(test_cases)
    print(f"\n🎯 FIXED VERIFICATION TEST RESULTS")
    print(f"   Success Rate: {success_rate:.1%}")
    print(f"   Tests Passed: {successful_tests}/{len(test_cases)}")
    
    if success_rate >= 0.8:
        print(f"   🎉 VERIFICATION SYSTEM FIXED!")
    else:
        print(f"   ⚠️  Still needs work")
    
    return results


if __name__ == "__main__":
    test_fixed_verification() 