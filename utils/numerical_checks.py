"""
Simplified numerical checking utilities for verification.

This module extracts complex numerical checking logic and reduces parameter counts
by using data structures and helper functions.
"""

import logging
import random
from typing import Dict, List, Tuple, Optional, Any
import sympy
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NumericalCheckConfig:
    """Configuration for numerical checks."""
    n_samples: int
    tolerance: float
    max_iter: int
    pop_size: int


@dataclass
class ViolationInfo:
    """Information about verification violations."""
    point: Dict[str, float]
    violation_type: str
    value: float
    expected: str


@dataclass
class NumericalCheckResult:
    """Result of numerical verification checks."""
    passed: bool
    reason: str
    violations: int
    violation_points: List[ViolationInfo]
    samples_checked: Dict[str, int]


def check_single_point_conditions(
    point_dict: Dict[str, float],
    B_func: callable,
    primary_condition_func: callable,
    context: 'VerificationContext',
    tolerance: float
) -> List[ViolationInfo]:
    """
    Check all conditions for a single point.
    
    Returns:
        List of violations found at this point
    """
    violations = []
    
    try:
        # Evaluate barrier function at this point
        b_val_sympy = B_func(**point_dict)
        b_val = float(b_val_sympy)
        
        # Evaluate primary condition (dB/dt or ΔB)
        condition_val_sympy = primary_condition_func(**point_dict)
        condition_val = float(condition_val_sympy)
        
        # Check initial set condition: B(x) ≤ 0 in initial set
        is_in_initial = check_set_membership_numerical(point_dict, context.initial_set_relationals, context.variables_sympy)
        if is_in_initial and b_val > tolerance:
            violations.append(ViolationInfo(
                point=point_dict,
                violation_type='initial_set',
                value=b_val,
                expected='≤ 0'
            ))
        
        # Check unsafe set condition: B(x) ≥ 0 outside unsafe set
        is_in_unsafe = check_set_membership_numerical(point_dict, context.unsafe_set_relationals, context.variables_sympy)
        if not is_in_unsafe and b_val < -tolerance:
            violations.append(ViolationInfo(
                point=point_dict,
                violation_type='unsafe_set',
                value=b_val,
                expected='≥ 0'
            ))
        
        # Check primary condition: dB/dt ≤ 0 or ΔB ≤ 0 in safe set
        is_in_safe = check_set_membership_numerical(point_dict, context.safe_set_relationals, context.variables_sympy)
        if is_in_safe and condition_val > tolerance:
            violations.append(ViolationInfo(
                point=point_dict,
                violation_type='primary_condition',
                value=condition_val,
                expected='≤ 0'
            ))
            
    except Exception as e:
        logger.warning(f"Error evaluating point {point_dict} in numerical check: {e}")
    
    return violations


def check_domain_bounds_simplified(
    context: 'VerificationContext',
    config: NumericalCheckConfig
) -> NumericalCheckResult:
    """
    Simplified domain bounds verification with reduced parameters.
    
    This function replaces the complex numerical_check_domain_bounds function
    with a cleaner interface using data structures.
    """
    logger.info(f"Performing domain bounds verification with {config.n_samples} samples...")
    
    if not context.system_info.certificate_domain_bounds:
        return NumericalCheckResult(
            passed=True,
            reason="No domain bounds specified",
            violations=0,
            violation_points=[],
            samples_checked={'total': 0}
        )
    
    try:
        # Generate samples within domain bounds
        domain_samples = generate_samples(
            context.system_info.certificate_domain_bounds,
            context.variables_sympy,
            config.n_samples
        )
        
        all_violations = []
        checked_counts = {
            'initial': 0,
            'outside_unsafe': 0,
            'safe': 0,
            'total': len(domain_samples)
        }
        
        for point_dict in domain_samples:
            violations = check_single_point_conditions(
                point_dict, context.B_func, context.primary_condition_func,
                context, config.tolerance
            )
            
            # Update counts
            is_in_initial = check_set_membership_numerical(point_dict, context.initial_set_relationals, context.variables_sympy)
            is_in_unsafe = check_set_membership_numerical(point_dict, context.unsafe_set_relationals, context.variables_sympy)
            is_in_safe = check_set_membership_numerical(point_dict, context.safe_set_relationals, context.variables_sympy)
            
            if is_in_initial:
                checked_counts['initial'] += 1
            if not is_in_unsafe:
                checked_counts['outside_unsafe'] += 1
            if is_in_safe:
                checked_counts['safe'] += 1
            
            all_violations.extend(violations)
        
        # Determine result
        passed = len(all_violations) == 0
        
        reason = (f"Domain bounds check: {len(all_violations)} violations found. "
                f"Checked {checked_counts['initial']} initial points, {checked_counts['outside_unsafe']} non-unsafe points, "
                f"{checked_counts['safe']} safe points within domain bounds.")
        
        return NumericalCheckResult(
            passed=passed,
            reason=reason,
            violations=len(all_violations),
            violation_points=all_violations,
            samples_checked=checked_counts
        )
        
    except Exception as e:
        logger.error(f"Error during domain bounds verification: {e}")
        return NumericalCheckResult(
            passed=False,
            reason=f"Domain bounds check error: {e}",
            violations=-1,
            violation_points=[],
            samples_checked={'total': 0}
        )


def check_lie_derivative_simplified(
    context: 'VerificationContext',
    config: NumericalCheckConfig
) -> NumericalCheckResult:
    """
    Simplified Lie derivative numerical check.
    """
    if not context.numerical_sampling_bounds:
        return NumericalCheckResult(
            passed=None,
            reason="No sampling bounds available",
            violations=0,
            violation_points=[],
            samples_checked={'total': 0}
        )
    
    try:
        samples = generate_samples(
            context.numerical_sampling_bounds,
            context.variables_sympy,
            config.n_samples
        )
        
        violations = []
        checked_in_safe = 0
        
        for point_dict in samples:
            try:
                # Check if point is in safe set
                is_in_safe = check_set_membership_numerical(
                    point_dict, context.safe_set_relationals, context.variables_sympy
                )
                
                if is_in_safe:
                    checked_in_safe += 1
                    
                    # Evaluate primary condition (dB/dt or ΔB)
                    condition_val_sympy = context.primary_condition_func(**point_dict)
                    condition_val = float(condition_val_sympy)
                    
                    if condition_val > config.tolerance:
                        violations.append(ViolationInfo(
                            point=point_dict,
                            violation_type='lie_derivative',
                            value=condition_val,
                            expected='≤ 0'
                        ))
                        
            except Exception as e:
                logger.warning(f"Error evaluating point {point_dict} in Lie derivative check: {e}")
                continue
        
        passed = len(violations) == 0
        reason = f"Lie derivative check: {len(violations)} violations found in {checked_in_safe} safe points"
        
        return NumericalCheckResult(
            passed=passed,
            reason=reason,
            violations=len(violations),
            violation_points=violations,
            samples_checked={'safe': checked_in_safe, 'total': len(samples)}
        )
        
    except Exception as e:
        logger.error(f"Error during Lie derivative numerical check: {e}")
        return NumericalCheckResult(
            passed=False,
            reason=f"Lie derivative check error: {e}",
            violations=-1,
            violation_points=[],
            samples_checked={'total': 0}
        )


def check_boundary_conditions_simplified(
    context: 'VerificationContext',
    config: NumericalCheckConfig
) -> NumericalCheckResult:
    """
    Simplified boundary conditions numerical check.
    """
    if not context.numerical_sampling_bounds:
        return NumericalCheckResult(
            passed=None,
            reason="No sampling bounds available",
            violations=0,
            violation_points=[],
            samples_checked={'total': 0}
        )
    
    try:
        samples = generate_samples(
            context.numerical_sampling_bounds,
            context.variables_sympy,
            config.n_samples
        )
        
        initial_violations = []
        unsafe_violations = []
        checked_in_initial = 0
        checked_outside_unsafe = 0
        
        for point_dict in samples:
            try:
                # Evaluate barrier function
                b_val_sympy = context.B_func(**point_dict)
                b_val = float(b_val_sympy)
                
                # Check initial set condition
                is_in_initial = check_set_membership_numerical(
                    point_dict, context.initial_set_relationals, context.variables_sympy
                )
                if is_in_initial:
                    checked_in_initial += 1
                    if b_val > config.tolerance:
                        initial_violations.append(ViolationInfo(
                            point=point_dict,
                            violation_type='initial_set',
                            value=b_val,
                            expected='≤ 0'
                        ))
                
                # Check unsafe set condition
                is_in_unsafe = check_set_membership_numerical(
                    point_dict, context.unsafe_set_relationals, context.variables_sympy
                )
                if not is_in_unsafe:
                    checked_outside_unsafe += 1
                    if b_val < -config.tolerance:
                        unsafe_violations.append(ViolationInfo(
                            point=point_dict,
                            violation_type='unsafe_set',
                            value=b_val,
                            expected='≥ 0'
                        ))
                        
            except Exception as e:
                logger.warning(f"Error evaluating point {point_dict} in boundary check: {e}")
                continue
        
        total_violations = len(initial_violations) + len(unsafe_violations)
        passed = total_violations == 0
        
        reason = (f"Boundary check: {total_violations} violations found. "
                f"Checked {checked_in_initial} initial points, {checked_outside_unsafe} non-unsafe points")
        
        return NumericalCheckResult(
            passed=passed,
            reason=reason,
            violations=total_violations,
            violation_points=initial_violations + unsafe_violations,
            samples_checked={
                'initial': checked_in_initial,
                'outside_unsafe': checked_outside_unsafe,
                'total': len(samples)
            }
        )
        
    except Exception as e:
        logger.error(f"Error during boundary conditions numerical check: {e}")
        return NumericalCheckResult(
            passed=False,
            reason=f"Boundary check error: {e}",
            violations=-1,
            violation_points=[],
            samples_checked={'total': 0}
        )


# Import required functions from evaluation module
def check_set_membership_numerical(point_dict, set_relationals, variables_sympy_list):
    """Check if a point belongs to a set defined by relationals."""
    from evaluation.verify_certificate import check_set_membership_numerical as _check_set
    return _check_set(point_dict, set_relationals, variables_sympy_list)


def generate_samples(sampling_bounds, variables, n_samples):
    """Generate random samples within bounds."""
    from evaluation.verify_certificate import generate_samples as _generate_samples
    return _generate_samples(sampling_bounds, variables, n_samples) 