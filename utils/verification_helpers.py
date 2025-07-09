"""
Shared utilities for verification logic to reduce complexity and parameter counts.

This module consolidates complex verification logic and provides data structures
to reduce function parameter counts and improve maintainability.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Callable
import sympy

logger = logging.getLogger(__name__)


@dataclass
class VerificationConfig:
    """Configuration for verification parameters."""
    num_samples_lie: int
    num_samples_boundary: int
    numerical_tolerance: float
    sos_default_degree: int
    sos_solver: str
    sos_epsilon: float
    optimization_max_iter: int
    optimization_pop_size: int
    attempt_sos: bool
    attempt_optimization: bool


@dataclass
class SystemInfo:
    """System information for verification."""
    id: str
    state_variables: List[str]
    dynamics: List[str]
    initial_set_conditions: List[str]
    unsafe_set_conditions: List[str]
    safe_set_conditions: List[str]
    sampling_bounds: Optional[Dict[str, List[float]]]
    parameters: Dict[str, float]
    certificate_domain_bounds: Optional[Dict[str, List[float]]]
    system_type: str = "continuous"


@dataclass
class VerificationContext:
    """Context object for verification to reduce parameter passing."""
    system_info: SystemInfo
    config: VerificationConfig
    variables_sympy: List[sympy.Symbol]
    system_params_sympy: Dict[sympy.Symbol, float]
    numerical_sampling_bounds: Dict[str, List[float]]
    initial_set_relationals: List
    unsafe_set_relationals: List
    safe_set_relationals: List
    B_func: Optional[Callable] = None
    primary_condition_func: Optional[Callable] = None


def create_verification_context(system_info: SystemInfo, config: VerificationConfig) -> VerificationContext:
    """Create verification context from system info and config."""
    variables_sympy = [sympy.symbols(var) for var in system_info.state_variables]
    system_params_sympy = {sympy.symbols(k): v for k, v in system_info.parameters.items()}
    
    # Process sampling bounds
    numerical_sampling_bounds = {}
    if system_info.sampling_bounds:
        for var_name_str, bounds_list_sym in system_info.sampling_bounds.items():
            try:
                nb_min = sympy.sympify(bounds_list_sym[0]).subs(system_params_sympy).evalf()
                nb_max = sympy.sympify(bounds_list_sym[1]).subs(system_params_sympy).evalf()
                numerical_sampling_bounds[var_name_str] = [float(nb_min), float(nb_max)]
            except (AttributeError, TypeError, sympy.SympifyError) as e:
                logger.error(f"Could not process sampling bound '{bounds_list_sym}' for var '{var_name_str}': {e}")
                try:
                    numerical_sampling_bounds[var_name_str] = [float(bounds_list_sym[0]), float(bounds_list_sym[1])]
                except Exception as e_fb:
                    logger.error(f"Fallback to float for sampling bounds for '{var_name_str}' also failed: {e_fb}")
                    raise ValueError(f"Invalid sampling bounds for {var_name_str}")
    
    # Parse set conditions
    from evaluation.verify_certificate import parse_set_conditions
    
    raw_initial_rels, _ = parse_set_conditions(system_info.initial_set_conditions, variables_sympy)
    raw_unsafe_rels, _ = parse_set_conditions(system_info.unsafe_set_conditions, variables_sympy)
    raw_safe_rels, _ = parse_set_conditions(system_info.safe_set_conditions, variables_sympy)
    
    if raw_initial_rels is None or raw_unsafe_rels is None or raw_safe_rels is None:
        raise ValueError("Failed to parse set conditions")
    
    # Substitute system parameters into parsed relationals
    initial_set_relationals = [r.subs(system_params_sympy) for r in raw_initial_rels] if raw_initial_rels else []
    unsafe_set_relationals = [r.subs(system_params_sympy) for r in raw_unsafe_rels] if raw_unsafe_rels else []
    safe_set_relationals = [r.subs(system_params_sympy) for r in raw_safe_rels] if raw_safe_rels else []
    
    return VerificationContext(
        system_info=system_info,
        config=config,
        variables_sympy=variables_sympy,
        system_params_sympy=system_params_sympy,
        numerical_sampling_bounds=numerical_sampling_bounds,
        initial_set_relationals=initial_set_relationals,
        unsafe_set_relationals=unsafe_set_relationals,
        safe_set_relationals=safe_set_relationals
    )


def validate_candidate_expression(candidate_B_str: str, context: VerificationContext) -> Tuple[Optional[sympy.Expr], str]:
    """
    Validate and parse candidate barrier certificate expression.
    
    Returns:
        Tuple of (parsed_expression, error_message)
    """
    if not candidate_B_str:
        return None, "No usable certificate string provided by LLM after retries."
    
    from evaluation.verify_certificate import parse_expression
    
    B = parse_expression(candidate_B_str, context.variables_sympy)
    if B is None:
        return None, f"Failed to parse candidate string '{candidate_B_str}' with SymPy using state variables {context.system_info.state_variables}."
    
    # Check for unexpected symbols
    allowed_symbols_before_param_sub = set(context.variables_sympy) | set(context.system_params_sympy.keys())
    actual_free_symbols = B.free_symbols
    unexpected_symbols = actual_free_symbols - allowed_symbols_before_param_sub
    
    if unexpected_symbols:
        return None, f"Candidate expression contains unexpected symbols: {unexpected_symbols}."
    
    # Substitute known system parameters into B
    B_after_param_sub = B.subs(context.system_params_sympy)
    
    # Check symbols after parameter substitution
    actual_free_symbols_after_param_sub = B_after_param_sub.free_symbols
    unexpected_symbols_after_param_sub = actual_free_symbols_after_param_sub - set(context.variables_sympy)
    
    if unexpected_symbols_after_param_sub:
        return None, f"Candidate expression contains unexpected symbols after parameter substitution: {unexpected_symbols_after_param_sub}."
    
    return B_after_param_sub, ""


def process_dynamics_for_system_type(context: VerificationContext) -> Tuple[str, sympy.Expr]:
    """
    Process dynamics based on system type (continuous vs discrete).
    
    Returns:
        Tuple of (system_type, primary_condition_expression)
    """
    from evaluation.verify_certificate import (
        detect_system_type, parse_discrete_dynamics, calculate_discrete_difference,
        calculate_lie_derivative, parse_expression
    )
    
    system_type = detect_system_type(context.system_info.dynamics)
    context.system_info.system_type = system_type
    
    if system_type == 'discrete':
        # Process discrete-time dynamics
        preprocessed_dynamics = []
        for dyn_eq_str in context.system_info.dynamics:
            try:
                combined_symbols_for_dyn = {**{v.name: v for v in context.variables_sympy}, **context.system_params_sympy}
                temp_dyn_expr = sympy.parse_expr(dyn_eq_str.split('=')[1].strip() if '=' in dyn_eq_str else dyn_eq_str, 
                                               local_dict=combined_symbols_for_dyn)
                if '=' in dyn_eq_str:
                    lhs = dyn_eq_str.split('=')[0].strip()
                    preprocessed_dynamics.append(f"{lhs} = {temp_dyn_expr.subs(context.system_params_sympy)}")
                else:
                    preprocessed_dynamics.append(str(temp_dyn_expr.subs(context.system_params_sympy)))
            except Exception:
                preprocessed_dynamics.append(dyn_eq_str)
        
        next_state_functions = parse_discrete_dynamics(preprocessed_dynamics, context.variables_sympy)
        delta_B = calculate_discrete_difference(context.B_func, context.variables_sympy, next_state_functions)
        return system_type, delta_B
    else:
        # Process continuous-time dynamics
        parsed_dynamics_str = []
        for dyn_eq_str in context.system_info.dynamics:
            try:
                combined_symbols_for_dyn = {**{v.name: v for v in context.variables_sympy}, **context.system_params_sympy}
                temp_dyn_expr = sympy.parse_expr(dyn_eq_str, local_dict=combined_symbols_for_dyn)
                parsed_dynamics_str.append(str(temp_dyn_expr.subs(context.system_params_sympy)))
            except Exception as e_dyn_parse:
                raise ValueError(f"Error processing dynamics: {dyn_eq_str} - {e_dyn_parse}")
        
        dB_dt = calculate_lie_derivative(context.B_func, context.variables_sympy, parsed_dynamics_str)
        return system_type, dB_dt


def create_numerical_functions(B: sympy.Expr, primary_condition: sympy.Expr, context: VerificationContext) -> None:
    """Create numerical functions for verification."""
    from evaluation.verify_certificate import lambdify_expression
    
    context.B_func = lambdify_expression(B, context.variables_sympy)
    context.primary_condition_func = lambdify_expression(primary_condition, context.variables_sympy)
    
    if not context.B_func or not context.primary_condition_func:
        raise ValueError("Failed to create numerical functions")


def build_verification_summaries(results: Dict[str, Any], context: VerificationContext) -> Dict[str, Any]:
    """Build standardized verification summaries for web interface."""
    # SOS summary
    sos_summary = {
        'attempted': results.get('sos_attempted', False),
        'success': results.get('sos_passed', False),
        'reason': results.get('sos_reason', 'Not Attempted'),
        'details': {
            'lie_passed': results.get('sos_lie_passed'),
            'init_passed': results.get('sos_init_passed'),
            'unsafe_passed': results.get('sos_unsafe_passed')
        }
    }
    
    # Symbolic summary
    symbolic_summary = {
        'success': bool(results.get('symbolic_lie_check_passed') and results.get('symbolic_boundary_check_passed')),
        'reason': f"Lie: {results.get('symbolic_lie_reason', '')} | Boundary: {results.get('symbolic_boundary_reason', '')}",
        'details': {
            'lie_passed': results.get('symbolic_lie_check_passed'),
            'boundary_passed': results.get('symbolic_boundary_check_passed')
        }
    }
    
    # Numerical summary
    numerical_success_flag = (results.get('numerical_overall_passed') is True)
    numerical_summary = {
        'success': numerical_success_flag,
        'reason': results.get('numerical_sampling_reason', 'Numerical checks not executed'),
        'details': results.get('numerical_sampling_details', {})
    }
    
    # Detect inconsistencies
    conflict_detected = sos_summary['success'] and (numerical_summary['success'] is False)
    if conflict_detected:
        numerical_summary['reason'] += ' | WARNING: Numerical sampling failed while SOS succeeded.'
    
    return {
        'sos_verification': sos_summary,
        'symbolic_verification': symbolic_summary,
        'numerical_verification': numerical_summary,
        'parsing': {
            'candidate_parsed': results.get('parsing_B_successful'),
            'sets_parsed': results.get('parsing_sets_successful'),
            'is_polynomial_system': results.get('is_polynomial_system'),
            'system_type': results.get('system_type')
        },
        'overall_success': True if results.get('final_verdict', '').startswith('Passed') else False,
        'conflict': 'sos_passed_numerical_failed' if conflict_detected else None
    } 