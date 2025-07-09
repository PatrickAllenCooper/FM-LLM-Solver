"""
Simplified verification function using utility modules.

This module provides a cleaner, more maintainable version of the main verification
function by using the extracted utility modules.
"""

import logging
import time
from typing import Dict, Any, Optional
from omegaconf import DictConfig

from utils.verification_helpers import (
    VerificationConfig, SystemInfo, VerificationContext,
    create_verification_context, validate_candidate_expression,
    process_dynamics_for_system_type, create_numerical_functions,
    build_verification_summaries
)
from utils.numerical_checks import (
    NumericalCheckConfig, check_domain_bounds_simplified,
    check_lie_derivative_simplified, check_boundary_conditions_simplified
)
from utils.condition_parser import parse_set_conditions_simplified

logger = logging.getLogger(__name__)


def verify_barrier_certificate_simplified(
    candidate_B_str: str, 
    system_info_dict: dict, 
    verification_cfg: DictConfig
) -> Dict[str, Any]:
    """
    Simplified version of verify_barrier_certificate using utility modules.
    
    This function reduces complexity by:
    1. Using data classes to reduce parameter counts
    2. Extracting complex logic into utility functions
    3. Reducing nested conditionals
    4. Improving error handling and logging
    
    Parameters
    ----------
    candidate_B_str : str
        String representation of the candidate barrier function B(x)
    system_info_dict : dict
        Dictionary containing system information
    verification_cfg : DictConfig
        Configuration object with verification parameters
    
    Returns
    -------
    dict
        Verification results dictionary
    """
    start_time = time.time()
    logger.info(f"--- Verifying Candidate: {candidate_B_str} --- ")
    logger.info(f"System ID: {system_info_dict.get('id', 'N/A')}")

    # Initialize results with defaults
    results = _initialize_results(candidate_B_str, system_info_dict)
    
    try:
        # Convert dict to SystemInfo and create config
        system_info = _dict_to_system_info(system_info_dict)
        config = _dict_config_to_verification_config(verification_cfg)
        
        # Create verification context
        context = create_verification_context(system_info, config)
        
        # Validate candidate expression
        B, error_msg = validate_candidate_expression(candidate_B_str, context)
        if B is None:
            results.update(_create_error_result(error_msg, "Parsing Failed"))
            return _finalize_results(results, start_time)
        
        results['parsing_B_successful'] = True
        logger.info(f"Successfully parsed candidate B(x) = {B}")
        
        # Process dynamics and get primary condition
        system_type, primary_condition = process_dynamics_for_system_type(context)
        results['system_type'] = system_type
        results['lie_derivative_calculated'] = str(primary_condition)
        
        # Create numerical functions
        create_numerical_functions(B, primary_condition, context)
        
        # Check if system is suitable for SOS verification
        is_poly = _check_polynomial_system(B, primary_condition, context)
        results['is_polynomial_system'] = is_poly
        
        # Perform SOS verification if applicable
        if is_poly and config.attempt_sos:
            _perform_sos_verification(B, primary_condition, context, results)
        
        # Perform symbolic checks
        _perform_symbolic_checks(primary_condition, context, results)
        
        # Perform numerical checks
        numerical_config = NumericalCheckConfig(
            n_samples=config.num_samples_lie,
            tolerance=config.numerical_tolerance,
            max_iter=config.optimization_max_iter,
            pop_size=config.optimization_pop_size
        )
        
        _perform_numerical_checks(context, numerical_config, results)
        
        # Perform domain bounds verification
        if system_info.certificate_domain_bounds:
            _perform_domain_bounds_verification(context, numerical_config, results)
        
        # Determine final verdict
        _determine_final_verdict(results)
        
        # Build standardized summaries
        summaries = build_verification_summaries(results, context)
        results.update(summaries)
        
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        results.update(_create_error_result(f"Verification error: {e}", "Verification Error"))
    
    return _finalize_results(results, start_time)


def _initialize_results(candidate_B_str: str, system_info_dict: dict) -> Dict[str, Any]:
    """Initialize results dictionary with default values."""
    return {
        "candidate_B": candidate_B_str,
        "system_id": system_info_dict.get('id', 'N/A'),
        "parsing_B_successful": False,
        "parsing_sets_successful": False,
        "is_polynomial_system": None,
        "lie_derivative_calculated": None,
        "sos_attempted": False,
        "sos_passed": None,
        "sos_reason": "Not Attempted",
        "symbolic_lie_check_passed": None,
        "symbolic_boundary_check_passed": None,
        "numerical_sampling_lie_passed": None,
        "numerical_sampling_boundary_passed": None,
        "numerical_sampling_reason": "Not Attempted",
        "numerical_opt_attempted": False,
        "numerical_opt_reason": "Not Attempted",
        "domain_bounds_check_passed": True,
        "domain_bounds_violations": 0,
        "domain_bounds_reason": "No domain bounds specified",
        "final_verdict": "Verification Error",
        "reason": "Initialization",
        "verification_time_seconds": 0
    }


def _dict_to_system_info(system_info_dict: dict) -> SystemInfo:
    """Convert dictionary to SystemInfo object."""
    return SystemInfo(
        id=system_info_dict.get('id', 'unknown'),
        state_variables=system_info_dict.get('state_variables', []),
        dynamics=system_info_dict.get('dynamics', []),
        initial_set_conditions=system_info_dict.get('initial_set_conditions', []),
        unsafe_set_conditions=system_info_dict.get('unsafe_set_conditions', []),
        safe_set_conditions=system_info_dict.get('safe_set_conditions', []),
        sampling_bounds=system_info_dict.get('sampling_bounds'),
        parameters=system_info_dict.get('parameters', {}),
        certificate_domain_bounds=system_info_dict.get('certificate_domain_bounds')
    )


def _dict_config_to_verification_config(verification_cfg: DictConfig) -> VerificationConfig:
    """Convert DictConfig to VerificationConfig object."""
    return VerificationConfig(
        num_samples_lie=verification_cfg.num_samples_lie,
        num_samples_boundary=verification_cfg.num_samples_boundary,
        numerical_tolerance=verification_cfg.numerical_tolerance,
        sos_default_degree=verification_cfg.sos_default_degree,
        sos_solver=verification_cfg.get('sos_solver', 'MOSEK'),
        sos_epsilon=verification_cfg.sos_epsilon,
        optimization_max_iter=verification_cfg.optimization_max_iter,
        optimization_pop_size=verification_cfg.optimization_pop_size,
        attempt_sos=verification_cfg.attempt_sos,
        attempt_optimization=verification_cfg.attempt_optimization
    )


def _check_polynomial_system(B, primary_condition, context: VerificationContext) -> bool:
    """Check if system is polynomial and suitable for SOS verification."""
    from evaluation.verify_certificate import (
        check_polynomial, relationals_to_polynomials
    )
    
    init_polys = relationals_to_polynomials(context.initial_set_relationals, context.variables_sympy)
    unsafe_polys = relationals_to_polynomials(context.unsafe_set_relationals, context.variables_sympy)
    safe_polys = relationals_to_polynomials(context.safe_set_relationals, context.variables_sympy)
    
    return (check_polynomial(B, primary_condition, variables=context.variables_sympy) and
            init_polys is not None and unsafe_polys is not None and safe_polys is not None)


def _perform_sos_verification(B, primary_condition, context: VerificationContext, results: Dict[str, Any]):
    """Perform SOS verification if applicable."""
    try:
        from evaluation.verify_certificate import verify_sos
        import cvxpy as cp
        
        sos_solver_pref = getattr(cp, context.config.sos_solver, cp.MOSEK)
        
        logger.info("Attempting SOS verification...")
        sos_passed, sos_reason, sos_details = verify_sos(
            B.as_poly(*context.variables_sympy),
            primary_condition.as_poly(*context.variables_sympy),
            relationals_to_polynomials(context.initial_set_relationals, context.variables_sympy),
            relationals_to_polynomials(context.unsafe_set_relationals, context.variables_sympy),
            relationals_to_polynomials(context.safe_set_relationals, context.variables_sympy),
            context.variables_sympy,
            degree=context.config.sos_default_degree,
            sos_solver_pref=sos_solver_pref
        )
        
        results.update({
            'sos_attempted': True,
            'sos_passed': sos_passed,
            'sos_reason': sos_reason,
            'sos_lie_passed': sos_details.get('lie_passed'),
            'sos_init_passed': sos_details.get('init_passed'),
            'sos_unsafe_passed': sos_details.get('unsafe_passed')
        })
        
        if sos_passed:
            results['final_verdict'] = "Passed SOS Checks"
            results['reason'] = sos_reason
            
    except ImportError:
        logger.warning("CVXPY not installed. Skipping SOS verification.")
        results['sos_reason'] = "CVXPY not installed"
    except Exception as e:
        logger.error(f"Error during SOS verification: {e}")
        results.update({
            'sos_attempted': True,
            'sos_passed': False,
            'sos_reason': f"SOS Error: {e}"
        })


def _perform_symbolic_checks(primary_condition, context: VerificationContext, results: Dict[str, Any]):
    """Perform symbolic verification checks."""
    from evaluation.verify_certificate import (
        check_lie_derivative_symbolic, check_boundary_symbolic,
        check_discrete_difference_symbolic
    )
    
    if context.system_info.system_type == 'discrete':
        sym_condition_passed, sym_condition_reason = check_discrete_difference_symbolic(
            primary_condition, context.variables_sympy, context.safe_set_relationals
        )
        results['symbolic_discrete_check_passed'] = sym_condition_passed
        results['symbolic_lie_check_passed'] = sym_condition_passed
    else:
        sym_condition_passed, sym_condition_reason = check_lie_derivative_symbolic(
            primary_condition, context.variables_sympy, context.safe_set_relationals
        )
        results['symbolic_lie_check_passed'] = sym_condition_passed
    
    sym_bound_passed, sym_bound_reason = check_boundary_symbolic(
        context.B_func, context.variables_sympy, 
        context.initial_set_relationals, context.unsafe_set_relationals
    )
    results['symbolic_boundary_check_passed'] = sym_bound_passed
    
    # Store reasons for later use
    results['symbolic_lie_reason'] = sym_condition_reason
    results['symbolic_boundary_reason'] = sym_bound_reason


def _perform_numerical_checks(context: VerificationContext, config: NumericalCheckConfig, results: Dict[str, Any]):
    """Perform numerical verification checks."""
    if not context.B_func or not context.primary_condition_func or not context.numerical_sampling_bounds:
        results['numerical_sampling_reason'] = "Numerical checks skipped (functions unavailable or no bounds)"
        results['numerical_overall_passed'] = None
        return
    
    # Perform Lie derivative check
    lie_result = check_lie_derivative_simplified(context, config)
    results['numerical_sampling_lie_passed'] = lie_result.passed
    results['numerical_sampling_lie_reason'] = lie_result.reason
    
    # Perform boundary check
    boundary_result = check_boundary_conditions_simplified(context, config)
    results['numerical_sampling_boundary_passed'] = boundary_result.passed
    results['numerical_sampling_boundary_reason'] = boundary_result.reason
    
    # Combine results
    numerical_overall_passed = (lie_result.passed and boundary_result.passed)
    results['numerical_overall_passed'] = numerical_overall_passed
    
    combined_reason = f"Lie: {lie_result.reason} | Boundary: {boundary_result.reason}"
    results['numerical_sampling_reason'] = combined_reason


def _perform_domain_bounds_verification(context: VerificationContext, config: NumericalCheckConfig, results: Dict[str, Any]):
    """Perform domain bounds verification."""
    domain_result = check_domain_bounds_simplified(context, config)
    
    results.update({
        'domain_bounds_check_passed': domain_result.passed,
        'domain_bounds_violations': domain_result.violations,
        'domain_bounds_reason': domain_result.reason
    })
    
    if not domain_result.passed:
        results['numerical_overall_passed'] = False
        results['reason'] += f" | Domain bounds violated: {domain_result.violations} violations"


def _determine_final_verdict(results: Dict[str, Any]):
    """Determine the final verification verdict."""
    if results['final_verdict'] != "Passed SOS Checks":
        if results['numerical_overall_passed'] is True:
            results['final_verdict'] = "Passed Numerical Checks"
            results['reason'] = f"Numerical Checks Passed: {results['numerical_sampling_reason']}"
        elif results['numerical_overall_passed'] is False:
            results['final_verdict'] = "Failed Numerical Checks"
            results['reason'] = f"Numerical Checks Failed: {results['numerical_sampling_reason']}"
        elif (results.get('symbolic_lie_check_passed') and 
              results.get('symbolic_boundary_check_passed')):
            condition_name = "Discrete Diff" if results.get('system_type') == 'discrete' else "Lie"
            results['final_verdict'] = "Passed Symbolic Checks (Basic)"
            results['reason'] = f"Symbolic Checks Passed: {condition_name}: {results.get('symbolic_lie_reason')} | Boundary: {results.get('symbolic_boundary_reason')}"
        else:
            condition_name = "Discrete Diff" if results.get('system_type') == 'discrete' else "Lie"
            results['final_verdict'] = "Failed Symbolic Checks / Inconclusive / Error"
            results['reason'] = f"Symbolic Checks Failed/Inconclusive: {condition_name}: {results.get('symbolic_lie_reason')} | Boundary: {results.get('symbolic_boundary_reason')}"


def _create_error_result(error_msg: str, verdict: str) -> Dict[str, Any]:
    """Create error result dictionary."""
    return {
        'reason': error_msg,
        'final_verdict': verdict,
        'parsing_B_successful': False
    }


def _finalize_results(results: Dict[str, Any], start_time: float) -> Dict[str, Any]:
    """Finalize results with timing and logging."""
    results['verification_time_seconds'] = time.time() - start_time
    logger.info(f"Final Verdict: {results['final_verdict']}. Reason: {results['reason']}")
    return results 