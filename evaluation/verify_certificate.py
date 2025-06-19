import sympy
import logging
import re
import numpy as np
import time
import json
import cvxpy as cp
from itertools import product, combinations_with_replacement
from collections import defaultdict
from scipy.optimize import differential_evolution # Import for optimization
from omegaconf import DictConfig # To type hint config object
# Fix the import error - try the correct import path
try:
    from sympy.polys.monomials import itermonomials
except ImportError:
    # Alternative import paths based on different sympy versions
    try:
        from sympy.polys import itermonomials
    except ImportError:
        # If both fail, create a fallback implementation
        def itermonomials(variables, degree):
            """Fallback implementation for generating monomials up to a given degree."""
            if not variables:
                return [sympy.S.One]
            
            result = [sympy.S.One]
            for d in range(1, degree + 1):
                for combo in combinations_with_replacement(variables, d):
                    term = sympy.prod(combo)
                    result.append(term)
            return result

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants are now loaded from config passed into verify_barrier_certificate
# --- Constants for Numerical Checks ---
NUM_SAMPLES_LIE = 10000  # Samples for checking dB/dt <= 0
NUM_SAMPLES_BOUNDARY = 5000 # Samples for checking init/unsafe set conditions
NUMERICAL_TOLERANCE = 1e-6 # Tolerance for checking <= 0 or >= 0
SOS_DEFAULT_DEGREE = 2 # Default degree for SOS multipliers (adjust as needed)
SOS_SOLVER = cp.MOSEK # Preferred solver (requires license)
# SOS_SOLVER = cp.SCS   # Open-source alternative
SOS_EPSILON = 1e-7 # Small value for strict inequalities -> non-strict for SOS
OPTIMIZATION_MAX_ITER = 100 # Iterations for differential_evolution
OPTIMIZATION_POP_SIZE = 15 # Population size multiplier for diff_evolution

# --- Parsing and Symbolic Functions ---

def parse_expression(expr_str, variables):
    """Parses a string expression into a SymPy expression."""
    try:
        # Ensure variables are SymPy symbols
        local_dict = {var.name: var for var in variables}
        # Use sympy.parse_expr for safer evaluation than eval
        expr = sympy.parse_expr(expr_str, local_dict=local_dict, transformations='all')
        return expr
    except (SyntaxError, TypeError, sympy.SympifyError) as e:
        logging.error(f"Failed to parse expression '{expr_str}': {e}")
        return None

def parse_set_conditions(condition_strings, variables):
    """Parses a list of inequality strings into SymPy Relational objects."""
    if condition_strings is None:
        return [], "No conditions provided"
    if not isinstance(condition_strings, list):
        logging.error("Set conditions must be a list of strings.")
        return None, "Invalid condition format (not a list)"

    relationals = []
    local_dict = {var.name: var for var in variables}
    for cond_str in condition_strings:
        try:
            # Special handling for logical OR conditions which can cause issues
            if " or " in cond_str:
                # Split the OR condition and parse each part separately
                parts = cond_str.split(" or ")
                sub_relationals = []
                valid_parts = True
                
                for part in parts:
                    part = part.strip()
                    try:
                        rel_part = sympy.sympify(part, locals=local_dict)
                        if not isinstance(rel_part, sympy.core.relational.Relational):
                            logging.warning(f"Part of OR condition '{part}' is not a relational.")
                            valid_parts = False
                            break
                        sub_relationals.append(rel_part)
                    except Exception as sub_e:
                        logging.warning(f"Failed to parse part of OR condition: '{part}'. Error: {str(sub_e)}")
                        valid_parts = False
                        break
                
                if valid_parts and sub_relationals:
                    # Combine the parts with logical OR
                    combined_rel = sub_relationals[0]
                    for i in range(1, len(sub_relationals)):
                        combined_rel = sympy.logic.boolalg.Or(combined_rel, sub_relationals[i])
                    relationals.append(combined_rel)
                    continue
                else:
                    # If any part failed to parse, try the original approach
                    pass
            
            # Standard parsing for non-OR conditions or if OR special handling failed
            rel = sympy.sympify(cond_str, locals=local_dict)
            if not (
                isinstance(rel, sympy.logic.boolalg.BooleanAtom) or
                isinstance(rel, sympy.logic.boolalg.BooleanFunction) or  # Ensure this is included
                isinstance(rel, sympy.core.relational.Relational)
            ):
                raise TypeError(f"Parsed condition '{cond_str}' (result: '{rel}', type: {type(rel)}) is not a recognized Relational or SymPy Boolean construct.")
            relationals.append(rel)
        except (SyntaxError, TypeError, sympy.SympifyError) as e:
            logging.error(f"Failed to parse set condition '{cond_str}'. Error: {str(e)}")
            return None, f"Parsing error in condition: {cond_str}"
    return relationals, "Conditions parsed successfully"

def calculate_lie_derivative(B, variables, dynamics):
    """Calculates the Lie derivative (time derivative) dB/dt = ∇B ⋅ f(x)."""
    if B is None or not variables or len(variables) != len(dynamics):
        logging.error("Invalid input for Lie derivative calculation.")
        return None
    try:
        grad_B = [sympy.diff(B, var) for var in variables]
        dB_dt = sympy.S.Zero
        for i in range(len(variables)):
            # Need to parse dynamics strings into expressions here
            f_i = parse_expression(dynamics[i], variables)
            if f_i is None:
                logging.error(f"Could not parse dynamic component: {dynamics[i]}")
                return None # Indicate failure
            dB_dt += grad_B[i] * f_i

        # Attempt to simplify the resulting expression
        return sympy.simplify(dB_dt)
    except Exception as e:
        logging.error(f"Error calculating Lie derivative: {e}")
        return None

def check_polynomial(*expressions, variables):
    """Checks if all provided SymPy expressions are polynomials in the given variables."""
    try:
        return all(expr is not None and expr.is_polynomial(*variables) for expr in expressions)
    except Exception as e:
        logging.error(f"Error checking polynomial status: {e}")
        return False

def relationals_to_polynomials(relationals, variables):
    """Converts SymPy relationals (e.g., x>=0) to polynomials (e.g., x) for SOS."""
    # Assumes relationals are in the form 'expr >= 0' or 'expr <= 0'
    # Returns list of polynomials p_i such that the set is {x | p_i(x) >= 0}
    polynomials = []
    for rel in relationals:
        if isinstance(rel, sympy.logic.boolalg.BooleanTrue):
             continue # Trivial condition
        if isinstance(rel, sympy.logic.boolalg.BooleanFunction):
            logging.warning(f"Cannot convert BooleanFunction {type(rel)}: '{rel}' to polynomial for SOS. SOS may be disabled for this set.")
            return None # Or handle as appropriate for SOS (e.g., empty list might also signify failure)
        if not hasattr(rel, 'lhs') or not hasattr(rel, 'rhs'):
             logging.warning(f"Cannot convert non-relational {rel} to polynomial for SOS.")
             return None
        # Convert expr >= C or expr >= 0 to expr - C >= 0
        if rel.rel_op == '>=' or rel.rel_op == '>': # Treat > as >= for SOS relaxation
            poly = rel.lhs - rel.rhs
        # Convert expr <= C or expr <= 0 to C - expr >= 0
        elif rel.rel_op == '<=' or rel.rel_op == '<': # Treat < as <= for SOS relaxation
            poly = rel.rhs - rel.lhs
        else:
            logging.warning(f"Unsupported relational operator '{rel.rel_op}' for SOS conversion.")
            return None
        if not check_polynomial(poly, variables=variables):
             logging.warning(f"Set condition '{rel}' is not polynomial.")
             return None
        polynomials.append(poly.as_poly(*variables))
    return polynomials

# --- Symbolic Checking --- #
def check_lie_derivative_symbolic(dB_dt, variables, safe_set_relationals):
    """Performs a basic symbolic check if dB/dt <= 0 within the safe set."""
    # This remains very basic
    if dB_dt is None: return False, "Lie derivative calculation failed."
    if dB_dt == 0: return True, "Symbolic check passed (dB/dt == 0)"
    # TODO: Add more sophisticated symbolic checks if desired (e.g., using solveset)
    return False, "Symbolic check inconclusive (basic implementation)"

def check_boundary_symbolic(B, variables, initial_set_relationals, unsafe_set_relationals):
    """Placeholder for symbolic boundary checks."""
    # Remains extremely difficult symbolically
    return True, "Symbolic boundary checks not implemented"


# --- Numerical Checking Helper ---
def _evaluate_single_condition_numerical(condition, point_dict, variables_sympy_list_for_debug):
    # variables_sympy_list_for_debug is currently unused but kept for potential future debug logging
    if condition is sympy.true: # Use 'is' for singletons
        return True
    if condition is sympy.false:
        return False

    if isinstance(condition, sympy.logic.boolalg.And):
        for arg in condition.args:
            if not _evaluate_single_condition_numerical(arg, point_dict, variables_sympy_list_for_debug):
                return False 
        return True 
    
    if isinstance(condition, sympy.logic.boolalg.Or):
        any_true = False
        for arg in condition.args:
            if _evaluate_single_condition_numerical(arg, point_dict, variables_sympy_list_for_debug):
                any_true = True
                break
        return any_true

    if isinstance(condition, sympy.core.relational.Relational):
        try:
            # Replace symbolic variables with their numeric values
            lhs_val = condition.lhs.subs(point_dict)
            rhs_val = condition.rhs.subs(point_dict)
            
            # Evaluate left and right sides to get numeric values if possible
            try:
                lhs_num = float(lhs_val.evalf(chop=True))
                rhs_num = float(rhs_val.evalf(chop=True))
                
                # Compare numeric values based on the relation operator
                if condition.rel_op == '>=':
                    return lhs_num >= rhs_num
                elif condition.rel_op == '>':
                    return lhs_num > rhs_num
                elif condition.rel_op == '<=':
                    return lhs_num <= rhs_num
                elif condition.rel_op == '<':
                    return lhs_num < rhs_num
                elif condition.rel_op == '==':
                    return abs(lhs_num - rhs_num) < 1e-10
                elif condition.rel_op == '!=':
                    return abs(lhs_num - rhs_num) >= 1e-10
                else:
                    logging.warning(f"Unsupported relational operator: {condition.rel_op}")
                    return False
            except (TypeError, ValueError):
                # If we can't convert to float, try using the standard substitution approach
                substituted_rel = condition.subs(point_dict)
                
                if substituted_rel is sympy.true:
                    return True
                elif substituted_rel is sympy.false:
                    return False
                elif hasattr(substituted_rel, 'evalf'):
                    eval_result = substituted_rel.evalf(chop=True)
                    if eval_result is sympy.true:
                        return True
                    elif eval_result is sympy.false:
                        return False
                    else:
                        logging.warning(
                            f"Relational condition '{condition}' for point {point_dict} (original sub: '{substituted_rel}') "
                            f"evaluated to non-boolean SymPy expression after evalf: '{eval_result}' (type: {type(eval_result)}). Treating as failure."
                        )
                        return False
                else:
                    logging.warning(
                        f"Relational condition '{condition}' for point {point_dict} substituted to non-evaluatable "
                        f"and non-boolean expression: '{substituted_rel}' (type: {type(substituted_rel)}). Treating as failure."
                    )
                    return False
        except TypeError as te:
            logging.error(
                f"TypeError during evaluation of relational condition '{condition}' "
                f"for point {point_dict}: {te}. Treating as failure."
            )
            return False
        except Exception as e_inner:
            logging.error(
                f"Unexpected exception during evaluation of relational condition '{condition}' "
                f"for point {point_dict}: {e_inner}. Treating as failure."
            )
            return False
    
    logging.warning(f"Unhandled condition type '{type(condition)}' for condition '{condition}' at point {point_dict}. Treating as failure.")
    return False

# --- Numerical Checking Functions ---

def lambdify_expression(expr, variables_sympy_list):
    """Converts a SymPy expression to a NumPy-callable function."""
    if expr is None:
        return None
    try:
        func = sympy.lambdify(variables_sympy_list, expr, modules=['numpy'])
        return func
    except Exception as e:
        logging.error(f"Failed to lambdify expression {expr}: {e}")
        return None

def check_set_membership_numerical(point_dict, set_relationals, variables_sympy_list):
    """Numerically checks if a point satisfies a list of parsed SymPy relationals."""
    if not set_relationals: # No conditions means the point is in the set (e.g. entire space)
        return True

    try:
        for rel_condition in set_relationals: 
            if not _evaluate_single_condition_numerical(rel_condition, point_dict, variables_sympy_list):
                return False 
        return True  
    except Exception as e: 
        logging.error(f"Outer error evaluating set membership for point {point_dict} with conditions {set_relationals}: {e}")
        return False

def generate_samples(sampling_bounds, variables, n_samples):
    """Generates random samples within the specified bounds."""
    samples = []
    var_names = [var.name for var in variables]
    mins = [sampling_bounds[name][0] for name in var_names]
    maxs = [sampling_bounds[name][1] for name in var_names]
    dim = len(variables)

    random_points = np.random.uniform(low=mins, high=maxs, size=(n_samples, dim))
    for point in random_points:
        samples.append({var_names[i]: point[i] for i in range(dim)})
    return samples

def numerical_check_lie_derivative(dB_dt_func, sampling_bounds, variables, safe_set_relationals, n_samples, tolerance):
    """Numerically checks if dB/dt <= tolerance within the safe set using sampling."""
    logging.info(f"Performing numerical check for Lie derivative (<= {tolerance}) with {n_samples} samples...")
    if dB_dt_func is None: return False, "Lie derivative function invalid (lambdify failed?)."
    if not sampling_bounds: return False, "Sampling bounds not provided."
    if safe_set_relationals is None: return False, "Safe set conditions failed to parse."

    samples = generate_samples(sampling_bounds, variables, n_samples)
    violations = 0
    checked_in_safe_set = 0
    violation_points = []

    for point_dict in samples:
        is_in_safe_set = check_set_membership_numerical(point_dict, safe_set_relationals, variables)

        if is_in_safe_set:
            checked_in_safe_set += 1
            try:
                lie_val_sympy = dB_dt_func(**point_dict)
                try:
                    lie_val = float(lie_val_sympy)
                except (TypeError, ValueError) as conversion_e:
                    logging.warning(f"Could not convert Lie derivative value '{lie_val_sympy}' to float at {point_dict}: {conversion_e}. Skipping sample.")
                    continue # Skip this sample if conversion fails

                if lie_val > tolerance: # Use passed tolerance
                    violations += 1
                    # Store violation data
                    violation_data = {'point': point_dict, 'dB_dt_value': lie_val}
                    violation_points.append(violation_data)
                    logging.debug(f"Violation dB/dt <= 0: value={lie_val:.4g} at {point_dict}")
            except Exception as e:
                logging.error(f"Error evaluating Lie derivative at {point_dict} (original sympy_val: {lie_val_sympy if 'lie_val_sympy' in locals() else 'N/A'}): {e}")
    
    # Log summary instead of individual violations
    if violations > 0:
        logging.info(f"Found {violations}/{checked_in_safe_set} Lie derivative violations (dB/dt > {tolerance})")

    if checked_in_safe_set == 0: 
        return False, {
            'success': False,
            'reason': "No samples generated within the defined safe set/bounds.",
            'violations': violations,
            'checked_in_safe_set': checked_in_safe_set,
            'violation_points': []
        }
    
    if violations > 0: 
        return False, {
            'success': False,
            'reason': f"Found {violations}/{checked_in_safe_set} violations (dB/dt <= {tolerance}) in safe set samples.",
            'violations': violations,
            'checked_in_safe_set': checked_in_safe_set,
            'violation_points': violation_points[:10]  # Limit to 10 examples
        }
    else:
        logging.info(f"No violations found in {checked_in_safe_set} safe set samples.")
        return True, {
            'success': True,
            'reason': f"Passed numerical check (dB/dt <= {tolerance}) in {checked_in_safe_set} safe set samples.",
            'violations': 0,
            'checked_in_safe_set': checked_in_safe_set,
            'violation_points': []
        }

def numerical_check_boundary(B_func, sampling_bounds, variables, initial_set_relationals, unsafe_set_relationals, n_samples, tolerance):
    """Numerically checks B(x) conditions on initial and unsafe set boundaries using sampling."""
    logging.info(f"Performing numerical check for boundary conditions with {n_samples} samples...")
    if B_func is None: return False, "Barrier function invalid (lambdify failed?)."
    if not sampling_bounds: return False, "Sampling bounds not provided."
    if initial_set_relationals is None: return False, "Initial set conditions failed to parse."
    if unsafe_set_relationals is None: return False, "Unsafe set conditions failed to parse."

    samples = generate_samples(sampling_bounds, variables, n_samples)
    init_violations = 0
    unsafe_violations = 0
    checked_in_init = 0
    checked_outside_unsafe = 0
    
    # Store violation examples for potential visualization
    init_violation_points = []
    unsafe_violation_points = []

    for point_dict in samples:
        try:
            b_val_sympy = B_func(**point_dict)
            try:
                b_val = float(b_val_sympy)
            except (TypeError, ValueError) as conversion_e:
                logging.warning(f"Could not convert B(x) value '{b_val_sympy}' to float at {point_dict}: {conversion_e}. Skipping sample.")
                continue # Skip this sample if conversion fails

            # 1. Initial Set Check: B(x) <= tolerance inside X0
            is_in_init_set = check_set_membership_numerical(point_dict, initial_set_relationals, variables)
            if is_in_init_set:
                checked_in_init += 1
                if b_val > tolerance: # Use passed tolerance
                    init_violations += 1
                    # Store violation data
                    violation_data = {'point': point_dict, 'B_value': b_val}
                    init_violation_points.append(violation_data)
                    logging.debug(f"Violation B <= 0 in Init Set: B={b_val:.4g} at {point_dict}")

            # 2. Unsafe Set Check: B(x) >= -tolerance outside Xu
            is_in_unsafe_set = check_set_membership_numerical(point_dict, unsafe_set_relationals, variables)
            if not is_in_unsafe_set:
                 checked_outside_unsafe += 1
                 if b_val < -tolerance: # Use passed tolerance
                     unsafe_violations += 1
                     # Store violation data
                     violation_data = {'point': point_dict, 'B_value': b_val}
                     unsafe_violation_points.append(violation_data)
                     logging.debug(f"Violation B >= 0 outside Unsafe Set: B={b_val:.4g} at {point_dict}")

        except Exception as e:
            logging.error(f"Error evaluating B(x) or conditions at {point_dict} (original sympy_B_val: {b_val_sympy if 'b_val_sympy' in locals() else 'N/A'}): {e}")

    # Log summary information instead of individual violations
    if init_violations > 0:
        logging.info(f"Found {init_violations}/{checked_in_init} initial set violations (B > {tolerance})")
    if unsafe_violations > 0:
        logging.info(f"Found {unsafe_violations}/{checked_outside_unsafe} unsafe set violations (B < {-tolerance})")
            
    reason = []
    boundary_ok = True
    if initial_set_relationals: # Only report if conditions were given
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

    if unsafe_set_relationals: # Only report if conditions were given
        if checked_outside_unsafe > 0:
            if unsafe_violations > 0:
                boundary_ok = False
                reason.append(f"Failed Unsafe Set ({unsafe_violations}/{checked_outside_unsafe} violates B >= {-tolerance}).")
            else:
                reason.append(f"Passed Unsafe Set ({checked_outside_unsafe} samples).")
        else:
             reason.append("Unsafe Set check skipped (no samples outside set).")
    else:
        reason.append("Unsafe Set check skipped (no conditions).")

    if not reason: reason.append("No boundary checks performed.")
    
    # Store violation data in results for potential visualization
    result = {
        'boundary_ok': boundary_ok,
        'reason': " | ".join(reason),
        'init_violations': init_violations,
        'unsafe_violations': unsafe_violations,
        'init_violation_points': init_violation_points[:10],  # Limit to 10 examples 
        'unsafe_violation_points': unsafe_violation_points[:10]  # Limit to 10 examples
    }

    return boundary_ok, result


# --- SOS Helper Functions ---

def get_monomials(variables, degree):
    """Generates a list of sympy monomials up to a given degree."""
    # itermonomials generates monomials with total degree up to specified degree.
    # It includes S.One if degree >= 0.
    raw_monoms = list(itermonomials(variables, degree))
    # Ensure unique_monoms and sorted as per original logic.
    unique_monoms = sorted(list(set(raw_monoms)), key=sympy.default_sort_key)
    return unique_monoms

def sympy_poly_to_coeffs(sympy_poly, basis, variables):
    """Extracts coefficients of a sympy polynomial w.r.t. a given monomial basis."""
    if sympy_poly is None: return np.zeros(len(basis))
    if not isinstance(sympy_poly, sympy.Poly):
        sympy_poly = sympy_poly.as_poly(*variables)
    basis_map = {m: i for i, m in enumerate(basis)}
    coeffs = np.zeros(len(basis))
    poly_dict = sympy_poly.as_dict()
    for monom_tuple, coeff in poly_dict.items():
        monom_sympy = sympy.S.One
        for i, exp in enumerate(monom_tuple):
            monom_sympy *= variables[i]**exp
        if monom_sympy in basis_map:
            coeffs[basis_map[monom_sympy]] = float(coeff)
        elif float(coeff) != 0.0:
            logging.error(f"Monomial {monom_sympy} (coeff {coeff}) from {sympy_poly.expr} not in basis. Degree mismatch?")
            return None
    return coeffs

def calculate_sos_poly_coeffs(s_basis, Q_matrix, eq_basis, sympy_vars, multiplier=None):
    """Computes coeffs of (Z^T Q Z) * multiplier w.r.t eq_basis as CVXPY expressions."""
    s_basis_len = len(s_basis)
    eq_basis_len = len(eq_basis)
    eq_basis_map = {m: i for i, m in enumerate(eq_basis)}
    cvxpy_coeffs = [cp.Constant(0) for _ in range(eq_basis_len)]
    for i in range(s_basis_len):
        for j in range(i, s_basis_len):
            term_poly_expr = sympy.expand(s_basis[i] * s_basis[j])
            if multiplier:
                term_poly_expr = sympy.expand(term_poly_expr * multiplier.as_expr())
            term_poly = term_poly_expr.as_poly(*sympy_vars)
            term_coeffs = term_poly.as_dict()
            factor = 1 if i == j else 2
            for monom_tuple, coeff_val in term_coeffs.items():
                monom_sympy = sympy.S.One
                for k_var, exp in enumerate(monom_tuple):
                    monom_sympy *= sympy_vars[k_var]**exp
                if monom_sympy in eq_basis_map:
                    eq_idx = eq_basis_map[monom_sympy]
                    try:
                        cvxpy_coeffs[eq_idx] += float(coeff_val) * factor * Q_matrix[i, j]
                    except TypeError as te:
                        logging.error(f"TypeError converting coeff {coeff_val} for monom {monom_sympy}: {te}"); return None
                elif float(coeff_val) != 0.0:
                    logging.error(f"Monomial {monom_sympy} from SOS product not in equality basis."); return None
    return cp.vstack(cvxpy_coeffs)

def add_sos_constraints_poly(target_coeffs_np, set_polys, sympy_vars, mult_degree, eq_basis):
    """Helper to add SOS constraints: target_coeffs == coeffs(s0 + sum(si*gi))"""
    constraints = []
    s_basis_degree = mult_degree // 2
    s_basis = get_monomials(sympy_vars, s_basis_degree)
    s_basis_len = len(s_basis)
    Q_vars = {}
    # s0 term
    Q0 = cp.Variable((s_basis_len, s_basis_len), name="Q0", PSD=True)
    Q_vars["Q0"] = Q0
    s0_coeffs_expr = calculate_sos_poly_coeffs(s_basis, Q0, eq_basis, sympy_vars)
    if s0_coeffs_expr is None: logging.error("Failed calculating SOS coeffs for s0"); return None, None
    rhs_total_coeffs_expr = s0_coeffs_expr
    # s_i * g_i terms
    for i, g_poly in enumerate(set_polys):
        Qi = cp.Variable((s_basis_len, s_basis_len), name=f"Q_{i+1}", PSD=True)
        Q_vars[f"Q_{i+1}"] = Qi
        si_gi_coeffs_expr = calculate_sos_poly_coeffs(s_basis, Qi, eq_basis, sympy_vars, multiplier=g_poly)
        if si_gi_coeffs_expr is None: logging.error(f"Failed calculating SOS coeffs for s_{i+1}*g_{i+1}"); return None, None
        rhs_total_coeffs_expr += si_gi_coeffs_expr
    # Equality constraint (vectorized)
    constraints.append(target_coeffs_np.reshape(-1, 1) == rhs_total_coeffs_expr)
    return constraints, Q_vars

# --- Sum-of-Squares (SOS) Verification --- #
def verify_sos(B_poly, dB_dt_poly, initial_polys, unsafe_polys, safe_polys, variables, degree, sos_solver_pref):
    """Attempts to verify barrier conditions using SOS via CVXPY."""
    logging.info(f"Attempting SOS verification (degree {degree})...")
    available_solvers = cp.installed_solvers()
    solver = None
    if sos_solver_pref in available_solvers:
        solver = sos_solver_pref
    elif cp.SCS in available_solvers:
        solver = cp.SCS
        logging.warning(f"Preferred solver {sos_solver_pref} not found or specified incorrectly. Using SCS.")
    else:
        return False, "No suitable SDP solver (MOSEK or SCS) found.", { "lie_reason": "Solver not found", "init_reason": "Solver not found", "unsafe_reason": "Solver not found" }
    results = { "lie_passed": None, "lie_reason": "Not attempted", "init_passed": None, "init_reason": "Not attempted", "unsafe_passed": None, "unsafe_reason": "Not attempted" }

    try:
        # Determine max degree needed for equality basis
        max_deg_lie = max(dB_dt_poly.total_degree(), max((p.total_degree() + degree for p in safe_polys), default=0))
        max_deg_init = max(B_poly.total_degree(), max((p.total_degree() + degree for p in initial_polys), default=0))
        max_deg_unsafe = max(B_poly.total_degree(), max((p.total_degree() + degree for p in unsafe_polys), default=0))
        max_expr_deg = max(max_deg_lie, max_deg_init, max_deg_unsafe)
        equality_basis = get_monomials(variables, max_expr_deg)

        # --- 1. Lie Derivative Check (-dB/dt is SOS on Safe Set {g_i >= 0}) --- #
        logging.info("Checking Lie derivative condition via SOS...")
        target_coeffs_lie = sympy_poly_to_coeffs(-dB_dt_poly, equality_basis, variables)
        if target_coeffs_lie is None: raise ValueError("Failed getting coeffs for -dB/dt")
        constraints_lie, _ = add_sos_constraints_poly(target_coeffs_lie, safe_polys, variables, degree, equality_basis)
        if constraints_lie is None: raise ValueError("Failed formulating Lie SOS constraints.")
        prob_lie = cp.Problem(cp.Minimize(0), constraints_lie)
        prob_lie.solve(solver=solver, verbose=False)
        results["lie_passed"] = prob_lie.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]
        results["lie_reason"] = f"SOS Solver Status: {prob_lie.status}"

        # --- 2. Initial Set Check (-B is SOS on Initial Set {h_j >= 0}) --- #
        logging.info("Checking Initial Set condition via SOS...")
        target_coeffs_init = sympy_poly_to_coeffs(-B_poly, equality_basis, variables)
        if target_coeffs_init is None: raise ValueError("Failed getting coeffs for -B")
        constraints_init, _ = add_sos_constraints_poly(target_coeffs_init, initial_polys, variables, degree, equality_basis)
        if constraints_init is None: raise ValueError("Failed formulating Init SOS constraints.")
        prob_init = cp.Problem(cp.Minimize(0), constraints_init)
        prob_init.solve(solver=solver, verbose=False)
        results["init_passed"] = prob_init.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]
        results["init_reason"] = f"SOS Solver Status: {prob_init.status}"

        # --- 3. Unsafe Set Check (B is SOS on Unsafe Set {k_l >= 0}) --- #
        logging.info("Checking Unsafe Set condition (B>=0 on Unsafe) via SOS...")
        target_coeffs_unsafe = sympy_poly_to_coeffs(B_poly, equality_basis, variables)
        if target_coeffs_unsafe is None: raise ValueError("Failed getting coeffs for B")
        constraints_unsafe, _ = add_sos_constraints_poly(target_coeffs_unsafe, unsafe_polys, variables, degree, equality_basis)
        if constraints_unsafe is None: raise ValueError("Failed formulating Unsafe SOS constraints.")
        prob_unsafe = cp.Problem(cp.Minimize(0), constraints_unsafe)
        prob_unsafe.solve(solver=solver, verbose=False)
        results["unsafe_passed"] = prob_unsafe.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]
        results["unsafe_reason"] = f"SOS Solver Status: {prob_unsafe.status}"

    except ValueError as ve: err_msg = f"SOS Value Error: {ve}"; logging.error(err_msg); results['lie_reason'] = results['lie_reason'] if results['lie_reason'] != "Not attempted" else err_msg; results['init_reason'] = results['init_reason'] if results['init_reason'] != "Not attempted" else err_msg; results['unsafe_reason'] = results['unsafe_reason'] if results['unsafe_reason'] != "Not attempted" else err_msg
    except cp.SolverError as se: err_msg = f"CVXPY Solver Error: {se}"; logging.error(err_msg); results['lie_reason'] = results['lie_reason'] if results['lie_reason'] != "Not attempted" else err_msg; results['init_reason'] = results['init_reason'] if results['init_reason'] != "Not attempted" else err_msg; results['unsafe_reason'] = results['unsafe_reason'] if results['unsafe_reason'] != "Not attempted" else err_msg
    except Exception as e:
        err_msg = f"Unexpected SOS Error: {e}"
        logging.error(err_msg, exc_info=True)
        results['lie_reason'] = results['lie_reason'] if results['lie_reason'] != "Not attempted" else err_msg
        results['init_reason'] = results['init_reason'] if results['init_reason'] != "Not attempted" else err_msg
        results['unsafe_reason'] = results['unsafe_reason'] if results['unsafe_reason'] != "Not attempted" else err_msg
        if results['sos_reason'] == "Not Attempted" or results['sos_reason'] is None:
            results['sos_reason'] = err_msg
        results['sos_passed'] = False
        results['sos_attempted'] = True

    # --- Final Result --- #
    sos_passed = None
    if None not in [results["lie_passed"], results["init_passed"], results["unsafe_passed"]]:
        sos_passed = results["lie_passed"] and results["init_passed"] and results["unsafe_passed"]
    sos_reason = f"Lie: {results['lie_reason']} | Init: {results['init_reason']} | Unsafe: {results['unsafe_reason']}"
    return sos_passed, sos_reason, results


# --- Optimization-Based Falsification --- #

def objective_maximize_lie(x_numpy_array, dB_dt_func, variables, safe_set_relationals):
    """Objective function to maximize Lie derivative (-minimize -dBdt)
       Returns large penalty if outside safe set."""
    point_dict = {var.name: val for var, val in zip(variables, x_numpy_array)}
    
    # Detailed logging for optimization objective
    # logging.debug(f"[Opt Objective Lie] Point dict: {point_dict}")
    # logging.debug(f"[Opt Objective Lie] Safe set relationals: {safe_set_relationals}")

    if not check_set_membership_numerical(point_dict, safe_set_relationals, variables):
        # check_set_membership_numerical would have logged the specific relational failure if verbose enough
        return 1e10 # Large penalty for being outside safe set
    try:
        lie_val = dB_dt_func(**point_dict)
        return -lie_val # Minimize negative Lie derivative
    except Exception:
        return 1e10 # Penalize errors

def objective_maximize_b_in_init(x_numpy_array, b_func, variables, initial_set_relationals):
    """Objective function to maximize B(x) in initial set (-minimize -B)
       Returns large penalty if outside initial set."""
    point_dict = {var.name: val for var, val in zip(variables, x_numpy_array)}
    # logging.debug(f"[Opt Objective Init] Point dict: {point_dict}")
    # logging.debug(f"[Opt Objective Init] Initial set relationals: {initial_set_relationals}")
    if not check_set_membership_numerical(point_dict, initial_set_relationals, variables):
        return 1e10 # Large penalty
    try:
        b_val = b_func(**point_dict)
        return -b_val # Minimize negative B
    except Exception:
        return 1e10

def objective_minimize_b_outside_unsafe(x_numpy_array, b_func, variables, unsafe_set_relationals):
    """Objective function to minimize B(x) *outside* unsafe set.
       Returns large penalty if *inside* unsafe set."""
    point_dict = {var.name: val for var, val in zip(variables, x_numpy_array)}
    # logging.debug(f"[Opt Objective Unsafe] Point dict: {point_dict}")
    # logging.debug(f"[Opt Objective Unsafe] Unsafe set relationals: {unsafe_set_relationals}")
    if check_set_membership_numerical(point_dict, unsafe_set_relationals, variables): # Note: check if INSIDE unsafe
        return 1e10 # Large penalty for being *inside* unsafe set
    try:
        b_val = b_func(**point_dict)
        return b_val # Minimize B
    except Exception:
        return 1e10

def optimization_based_falsification(B_func, dB_dt_func, sampling_bounds, variables,
                                     initial_set_relationals, unsafe_set_relationals, safe_set_relationals,
                                     max_iter, pop_size, tolerance):
    """Uses differential evolution to search for counterexamples."""
    logging.info("Performing optimization-based falsification...")
    bounds = [(sampling_bounds[v.name][0], sampling_bounds[v.name][1]) for v in variables]
    results = {
        "lie_violation_found": None, "lie_max_val": None, "lie_point": None,
        "init_violation_found": None, "init_max_val": None, "init_point": None,
        "unsafe_violation_found": None, "unsafe_min_val": None, "unsafe_point": None,
        "reason": ""
    }

    try:
        # 1. Check Lie Derivative
        if dB_dt_func and safe_set_relationals is not None:
            opt_result_lie = differential_evolution(
                objective_maximize_lie, bounds,
                args=(dB_dt_func, variables, safe_set_relationals),
                maxiter=max_iter, popsize=pop_size, # Use config values
                tol=0.01, mutation=(0.5, 1), recombination=0.7, updating='immediate'
            )
            if opt_result_lie.success:
                max_lie_val = -opt_result_lie.fun
                results["lie_max_val"] = max_lie_val
                results["lie_point"] = {v.name: val for v, val in zip(variables, opt_result_lie.x)}
                if max_lie_val > tolerance: # Use config tolerance
                    results["lie_violation_found"] = True
                    logging.warning(f"Optimization found Lie violation: max(dB/dt)={max_lie_val:.4g} > {tolerance} at {results['lie_point']}")
                else:
                    results["lie_violation_found"] = False
            else:
                 results["lie_reason"] = "Lie opt failed/inconclusive."

        # 2. Check Initial Set
        if B_func and initial_set_relationals is not None:
            opt_result_init = differential_evolution(
                objective_maximize_b_in_init, bounds,
                args=(B_func, variables, initial_set_relationals),
                maxiter=max_iter, popsize=pop_size, # Use config values
                tol=0.01, mutation=(0.5, 1), recombination=0.7, updating='immediate'
            )
            if opt_result_init.success:
                 max_b_init = -opt_result_init.fun
                 results["init_max_val"] = max_b_init
                 results["init_point"] = {v.name: val for v, val in zip(variables, opt_result_init.x)}
                 if max_b_init > tolerance: # Use config tolerance
                     results["init_violation_found"] = True
                     logging.warning(f"Optimization found Initial Set violation: max(B)={max_b_init:.4g} > {tolerance} at {results['init_point']}")
                 else:
                     results["init_violation_found"] = False
            else:
                 results["init_reason"] = "Init opt failed/inconclusive."

        # 3. Check Unsafe Set
        if B_func and unsafe_set_relationals is not None:
             opt_result_unsafe = differential_evolution(
                 objective_minimize_b_outside_unsafe, bounds,
                 args=(B_func, variables, unsafe_set_relationals),
                 maxiter=max_iter, popsize=pop_size, # Use config values
                 tol=0.01, mutation=(0.5, 1), recombination=0.7, updating='immediate'
             )
             if opt_result_unsafe.success:
                 min_b_outside_unsafe = opt_result_unsafe.fun
                 results["unsafe_min_val"] = min_b_outside_unsafe
                 results["unsafe_point"] = {v.name: val for v, val in zip(variables, opt_result_unsafe.x)}
                 if min_b_outside_unsafe < -tolerance: # Use config tolerance
                     results["unsafe_violation_found"] = True
                     logging.warning(f"Optimization found Unsafe Set violation: min(B)={min_b_outside_unsafe:.4g} < {-tolerance} at {results['unsafe_point']}")
                 else:
                     results["unsafe_violation_found"] = False
             else:
                 results["unsafe_reason"] = "Unsafe opt failed/inconclusive."

    except Exception as e:
        logging.error(f"Error during optimization-based falsification: {e}")
        results["reason"] = f"Optimization Error: {e}"

    overall_violation_found = results["lie_violation_found"] or results["init_violation_found"] or results["unsafe_violation_found"]
    return overall_violation_found, results


# --- Discrete-Time Verification Functions ---

def detect_system_type(dynamics_str_list):
    """
    Detect if the system is discrete-time or continuous-time based on dynamics format.
    
    Parameters
    ----------
    dynamics_str_list : list of str
        List of dynamics equation strings
        
    Returns
    -------
    str
        'discrete' if discrete-time system, 'continuous' if continuous-time system
    """
    discrete_patterns = [
        r'[a-zA-Z_]\w*\s*\{\s*k\s*\+\s*1\s*\}\s*=',  # x{k+1} = 
        r'[a-zA-Z_]\w*_\{\s*k\s*\+\s*1\s*\}\s*=',    # x_{k+1} = 
        r'[a-zA-Z_]\w*\[\s*k\s*\+\s*1\s*\]\s*=',     # x[k+1] = 
        r'[a-zA-Z_]\w*\(\s*k\s*\+\s*1\s*\)\s*=',     # x(k+1) = 
    ]
    
    continuous_patterns = [
        r'd[a-zA-Z_]\w*/dt\s*=',                       # dx/dt = 
        r'[a-zA-Z_]\w*\'\s*=',                        # x' = 
        r'[a-zA-Z_]\w*_dot\s*=',                      # x_dot = 
    ]
    
    discrete_count = 0
    continuous_count = 0
    
    for dyn_str in dynamics_str_list:
        # Check for discrete-time patterns
        for pattern in discrete_patterns:
            if re.search(pattern, dyn_str, re.IGNORECASE):
                discrete_count += 1
                break
        
        # Check for continuous-time patterns  
        for pattern in continuous_patterns:
            if re.search(pattern, dyn_str, re.IGNORECASE):
                continuous_count += 1
                break
    
    if discrete_count > continuous_count:
        return 'discrete'
    elif continuous_count > discrete_count:
        return 'continuous'
    else:
        # Default to continuous if ambiguous
        logging.warning(f"Could not clearly detect system type from dynamics: {dynamics_str_list}. Defaulting to continuous.")
        return 'continuous'

def parse_discrete_dynamics(dynamics_str_list, variables_sympy):
    """
    Parse discrete-time dynamics of the form x_{k+1} = f(x_k, y_k).
    
    Parameters
    ----------
    dynamics_str_list : list of str
        List of discrete-time dynamics strings
    variables_sympy : list of sympy.Symbol
        List of state variable symbols
        
    Returns
    -------
    list of sympy expressions
        Parsed next-state functions [f1(x), f2(x), ...]
    """
    parsed_dynamics = []
    var_names = [str(var) for var in variables_sympy]
    
    for dyn_str in dynamics_str_list:
        try:
            # Extract the right-hand side of the equation
            # Handle various discrete-time formats
            patterns = [
                r'[a-zA-Z_]\w*\s*\{\s*k\s*\+\s*1\s*\}\s*=\s*(.+)',   # x{k+1} = ...
                r'[a-zA-Z_]\w*_\{\s*k\s*\+\s*1\s*\}\s*=\s*(.+)',     # x_{k+1} = ...
                r'[a-zA-Z_]\w*\[\s*k\s*\+\s*1\s*\]\s*=\s*(.+)',      # x[k+1] = ...
                r'[a-zA-Z_]\w*\(\s*k\s*\+\s*1\s*\)\s*=\s*(.+)',      # x(k+1) = ...
            ]
            
            rhs = None
            for pattern in patterns:
                match = re.search(pattern, dyn_str, re.IGNORECASE)
                if match:
                    rhs = match.group(1).strip()
                    break
            
            if rhs is None:
                raise ValueError(f"Could not parse discrete-time dynamics: {dyn_str}")
            
            # Clean up the right-hand side
            # Replace k subscripts with current variable names
            for var_name in var_names:
                # Replace var_k, var{k}, var[k], var(k) with var
                patterns_to_replace = [
                    rf'{var_name}_k\b',
                    rf'{var_name}_\{{k\}}',
                    rf'{var_name}\[k\]',
                    rf'{var_name}\(k\)',
                    rf'{var_name}\{{k\}}',
                ]
                for pattern in patterns_to_replace:
                    rhs = re.sub(pattern, var_name, rhs, flags=re.IGNORECASE)
            
            # Parse the expression
            local_dict = {var.name: var for var in variables_sympy}
            parsed_expr = sympy.parse_expr(rhs, local_dict=local_dict)
            parsed_dynamics.append(parsed_expr)
            
        except Exception as e:
            logging.error(f"Error parsing discrete dynamics '{dyn_str}': {e}")
            raise ValueError(f"Failed to parse discrete dynamics: {dyn_str}")
    
    return parsed_dynamics

def calculate_discrete_difference(B, variables_sympy, next_state_functions):
    """
    Calculate B(f(x)) - B(x) for discrete-time systems.
    
    Parameters
    ----------
    B : sympy expression
        Barrier function B(x)
    variables_sympy : list of sympy.Symbol
        State variables [x, y, ...]
    next_state_functions : list of sympy expressions
        Next-state functions [f1(x), f2(x), ...]
        
    Returns
    -------
    sympy expression
        B(f(x)) - B(x)
    """
    try:
        if len(variables_sympy) != len(next_state_functions):
            raise ValueError(f"Mismatch: {len(variables_sympy)} variables but {len(next_state_functions)} next-state functions")
        
        # Create substitution dictionary: x -> f1(x), y -> f2(x), etc.
        substitution_dict = {}
        for var, next_func in zip(variables_sympy, next_state_functions):
            substitution_dict[var] = next_func
        
        # Calculate B(f(x))
        B_next = B.subs(substitution_dict)
        
        # Calculate B(f(x)) - B(x)
        discrete_difference = B_next - B
        
        return sympy.simplify(discrete_difference)
        
    except Exception as e:
        logging.error(f"Error calculating discrete difference: {e}")
        return None

def check_discrete_difference_symbolic(delta_B, variables_sympy, safe_set_relationals):
    """
    Perform basic symbolic check if B(f(x)) - B(x) <= 0 within the safe set.
    
    Parameters
    ----------
    delta_B : sympy expression
        B(f(x)) - B(x)
    variables_sympy : list of sympy.Symbol
        State variables
    safe_set_relationals : list of sympy relationals
        Safe set constraints
        
    Returns
    -------
    tuple (bool, str)
        (passed, reason)
    """
    if delta_B is None:
        return False, "Discrete difference calculation failed."
    
    if delta_B == 0:
        return True, "Symbolic check passed (B(f(x)) - B(x) == 0)"
    
    # Check if delta_B is always non-positive
    try:
        # Simple cases
        if delta_B.is_negative:
            return True, "Symbolic check passed (B(f(x)) - B(x) < 0)"
        
        if delta_B.is_positive:
            return False, "Symbolic check failed (B(f(x)) - B(x) > 0)"
        
        # Try to determine sign under safe set constraints
        # This is a simplified check - more sophisticated analysis could be added
        return None, "Symbolic check inconclusive (complex expression)"
        
    except Exception as e:
        logging.warning(f"Error in symbolic discrete difference check: {e}")
        return None, "Symbolic check inconclusive (error in analysis)"

def numerical_check_discrete_difference(delta_B_func, sampling_bounds, variables_sympy, 
                                      safe_set_relationals, n_samples, tolerance):
    """
    Numerically check if B(f(x)) - B(x) <= tolerance within the safe set using sampling.
    
    Parameters
    ----------
    delta_B_func : callable
        Function that computes B(f(x)) - B(x)
    sampling_bounds : dict
        Sampling bounds for each variable
    variables_sympy : list of sympy.Symbol
        State variables
    safe_set_relationals : list of sympy relationals
        Safe set constraints
    n_samples : int
        Number of samples to test
    tolerance : float
        Numerical tolerance for the check
        
    Returns
    -------
    tuple (bool, dict)
        (passed, details)
    """
    logging.info(f"Performing numerical check for discrete difference (<= {tolerance}) with {n_samples} samples...")
    
    if delta_B_func is None:
        return False, {"reason": "Discrete difference function invalid (lambdify failed?)."}
    
    try:
        # Generate random samples
        samples = generate_samples(sampling_bounds, [var.name for var in variables_sympy], n_samples)
        
        violations = 0
        checked_in_safe_set = 0
        violation_points = []
        
        for point_dict in samples:
            try:
                # Check if point is in safe set
                if not check_set_membership_numerical(point_dict, safe_set_relationals, variables_sympy):
                    continue  # Skip points outside safe set
                
                checked_in_safe_set += 1
                
                # Evaluate B(f(x)) - B(x)
                delta_val_sympy = delta_B_func(**point_dict)
                
                try:
                    delta_val = float(delta_val_sympy)
                except (TypeError, ValueError) as conversion_e:
                    logging.warning(f"Could not convert discrete difference value '{delta_val_sympy}' to float at {point_dict}: {conversion_e}. Skipping sample.")
                    continue
                
                # Check if B(f(x)) - B(x) > tolerance (violation)
                if delta_val > tolerance:
                    violations += 1
                    violation_data = {'point': point_dict, 'delta_B_value': delta_val}
                    violation_points.append(violation_data)
                    logging.debug(f"Violation B(f(x)) - B(x) <= 0: value={delta_val:.4g} at {point_dict}")
                    
            except Exception as e:
                logging.error(f"Error evaluating discrete difference at {point_dict}: {e}")
                continue
        
        logging.info(f"Found {violations}/{checked_in_safe_set} discrete difference violations (B(f(x)) - B(x) > {tolerance})")
        
        if checked_in_safe_set == 0:
            return False, {
                'reason': "No valid samples found in safe set for discrete difference check.",
                'samples_checked': checked_in_safe_set,
                'violations_found': violations,
                'violation_points': violation_points
            }
        
        if violations > 0:
            return False, {
                'reason': f"Found {violations}/{checked_in_safe_set} violations (B(f(x)) - B(x) <= {tolerance}) in safe set samples.",
                'samples_checked': checked_in_safe_set,
                'violations_found': violations,
                'violation_points': violation_points
            }
        else:
            return True, {
                'reason': f"Passed numerical check (B(f(x)) - B(x) <= {tolerance}) in {checked_in_safe_set} safe set samples.",
                'samples_checked': checked_in_safe_set,
                'violations_found': violations,
                'violation_points': violation_points
            }
        
    except Exception as e:
        logging.error(f"Error during numerical discrete difference sampling: {e}")
        return False, {"reason": f"Numerical check error: {e}"}

# --- Main Verification Function (Integrates SOS and Optimization) ---

def verify_barrier_certificate(candidate_B_str: str, system_info: dict, verification_cfg: DictConfig):
    """
    Main verification function for barrier certificate candidates.
    
    This function verifies if a candidate barrier function B(x) satisfies the required conditions:
    1. B(x) <= 0 for all x in the initial set
    2. B(x) >= 0 for all x outside the unsafe set
    3. dB/dt <= 0 for all x in the safe set
    
    Parameters
    ----------
    candidate_B_str : str
        String representation of the candidate barrier function B(x)
    system_info : dict
        Dictionary containing system information:
        - id: System identifier
        - state_variables: List of state variable names
        - dynamics: List of dynamics equations (one per state variable)
        - initial_set_conditions: List of expressions defining the initial set
        - unsafe_set_conditions: List of expressions defining the unsafe set
        - safe_set_conditions: List of expressions defining the safe set
        - sampling_bounds: Dictionary mapping variable names to [min, max] bounds
    verification_cfg : DictConfig
        Configuration object with verification parameters:
        - num_samples_lie: Number of samples for checking Lie derivative
        - num_samples_boundary: Number of samples for checking boundary conditions
        - numerical_tolerance: Tolerance for numerical checks
        - sos_default_degree: Default degree for SOS multipliers
        - sos_solver: SDP solver preference (e.g., MOSEK, SCS)
        - attempt_sos: Whether to attempt SOS verification
        - attempt_optimization: Whether to attempt optimization-based falsification
    
    Returns
    -------
    dict
        Verification results dictionary containing:
        - candidate_B: The barrier function candidate string
        - system_id: System identifier
        - parsing_B_successful: Whether B was parsed successfully
        - parsing_sets_successful: Whether sets were parsed successfully
        - is_polynomial_system: Whether the system is polynomial
        - lie_derivative_calculated: The calculated Lie derivative
        - sos_attempted: Whether SOS verification was attempted
        - sos_passed: Whether SOS verification passed
        - symbolic_lie_check_passed: Whether symbolic Lie derivative check passed
        - numerical_sampling_lie_passed: Whether numerical Lie derivative check passed
        - numerical_sampling_boundary_passed: Whether numerical boundary check passed
        - numerical_opt_attempted: Whether optimization-based falsification was attempted
        - numerical_opt_lie_violation_found: Whether optimization found Lie derivative violations
        - numerical_opt_init_violation_found: Whether optimization found initial set violations
        - numerical_opt_unsafe_violation_found: Whether optimization found unsafe set violations
        - final_verdict: Overall verification result
        - reason: Explanation for the verdict
        - verification_time_seconds: Time taken for verification
    """
    start_time = time.time()
    logging.info(f"--- Verifying Candidate: {candidate_B_str} --- ")
    logging.info(f"System ID: {system_info.get('id', 'N/A')}")

    # Load verification parameters from config
    num_samples_lie = verification_cfg.num_samples_lie
    num_samples_boundary = verification_cfg.num_samples_boundary
    numerical_tolerance = verification_cfg.numerical_tolerance
    sos_default_degree = verification_cfg.sos_default_degree
    # sos_solver_pref = cp.MOSEK # Cannot serialize easily, handle lookup here
    sos_solver_pref_name = verification_cfg.get('sos_solver', 'MOSEK') # Get preferred solver name
    sos_solver_pref = getattr(cp, sos_solver_pref_name, cp.MOSEK) # Default to MOSEK if invalid
    sos_epsilon = verification_cfg.sos_epsilon 
    opt_max_iter = verification_cfg.optimization_max_iter
    opt_pop_size = verification_cfg.optimization_pop_size
    attempt_sos = verification_cfg.attempt_sos
    attempt_optimization = verification_cfg.attempt_optimization

    # Initialize results dictionary with default values
    results = {
        "candidate_B": candidate_B_str,
        "system_id": system_info.get('id', 'N/A'),
        "parsing_B_successful": False,
        "parsing_sets_successful": False,
        "is_polynomial_system": None,
        "lie_derivative_calculated": None,
        "sos_attempted": False,
        "sos_passed": None,
        "sos_lie_passed": None,
        "sos_init_passed": None,
        "sos_unsafe_passed": None,
        "sos_reason": "Not Attempted",
        "symbolic_lie_check_passed": None,
        "symbolic_boundary_check_passed": None,
        "numerical_sampling_lie_passed": None,
        "numerical_sampling_boundary_passed": None,
        "numerical_sampling_reason": "Not Attempted",
        "numerical_opt_attempted": False,
        "numerical_opt_lie_violation_found": None,
        "numerical_opt_init_violation_found": None,
        "numerical_opt_unsafe_violation_found": None,
        "numerical_opt_reason": "Not Attempted",
        "final_verdict": "Verification Error",
        "reason": "Initialization",
        "verification_time_seconds": 0
    }

    # --- STEP 1: Setup & Parsing --- #
    state_vars_str = system_info.get('state_variables', [])
    dynamics_str = system_info.get('dynamics', [])
    initial_conditions_list = system_info.get('initial_set_conditions', [])
    unsafe_conditions_list = system_info.get('unsafe_set_conditions', [])
    safe_conditions_list = system_info.get('safe_set_conditions', [])
    sampling_bounds = system_info.get('sampling_bounds', None)
    system_parameters = system_info.get('parameters', {})

    if not state_vars_str or not dynamics_str: 
        results['reason'] = "Incomplete system info (state_variables or dynamics missing)"
        results['verification_time_seconds'] = time.time() - start_time
        return results
    
    variables_sympy = [sympy.symbols(var) for var in state_vars_str]
    system_params_sympy = {sympy.symbols(k): v for k, v in system_parameters.items()}
    
    # --- STEP 1.5: Detect System Type (Discrete vs Continuous) --- #
    system_type = detect_system_type(dynamics_str)
    results['system_type'] = system_type
    logging.info(f"Detected system type: {system_type}")

    if not candidate_B_str: # Check if a candidate was actually provided (e.g. after LLM retries)
        results['reason'] = "No usable certificate string provided by LLM after retries."
        results['parsing_B_successful'] = False # Ensure this is marked false
        results['final_verdict'] = "Parsing Failed (No Candidate From LLM)"
        results['verification_time_seconds'] = time.time() - start_time
        return results
    
    B = parse_expression(candidate_B_str, variables_sympy)
    if B is None:
        results['reason'] = f"Failed to parse candidate string '{candidate_B_str}' with SymPy using state variables {state_vars_str}."
        results['parsing_B_successful'] = False
        results['final_verdict'] = "Parsing Failed (SymPy Error)"
        results['verification_time_seconds'] = time.time() - start_time
        return results
    
    results['parsing_B_successful'] = True # Parsed by SymPy, but check symbols next
    logging.info(f"Successfully parsed candidate B(x) = {B} with state variables {variables_sympy}")

    # CRITICAL CHECK 1: Free symbols in B must be a subset of (state_variables + system_parameters)
    allowed_symbols_before_param_sub = set(variables_sympy) | set(system_params_sympy.keys())
    actual_free_symbols = B.free_symbols
    unexpected_symbols = actual_free_symbols - allowed_symbols_before_param_sub

    if unexpected_symbols:
        logging.error(f"Parsed B(x) = {B} contains unexpected free symbols: {unexpected_symbols}. Allowed were state vars {variables_sympy} and params {list(system_params_sympy.keys())}. Original: '{candidate_B_str}'")
        results['reason'] = f"Candidate expression contains unexpected symbols: {unexpected_symbols}."
        results['final_verdict'] = "Parsing Failed (Unexpected Symbols in B)"
        results['verification_time_seconds'] = time.time() - start_time
        return results

    # Substitute known system parameters into B
    B_after_param_sub = B.subs(system_params_sympy)
    logging.info(f"B(x) after substituting system parameters = {B_after_param_sub}")

    # CRITICAL CHECK 2: Free symbols in B_after_param_sub must be a subset of state_variables only (or empty for constants)
    actual_free_symbols_after_param_sub = B_after_param_sub.free_symbols
    unexpected_symbols_after_param_sub = actual_free_symbols_after_param_sub - set(variables_sympy)
    
    if unexpected_symbols_after_param_sub:
        logging.error(f"Candidate B(x) = {B_after_param_sub} still contains unexpected free symbols after param substitution: {unexpected_symbols_after_param_sub}. Expected only state variables: {variables_sympy}. Original B: {B}, Original string: '{candidate_B_str}'")
        results['reason'] = f"Candidate expression contains unexpected symbols after parameter substitution: {unexpected_symbols_after_param_sub}."
        results['final_verdict'] = "Parsing Failed (Unexpected Symbols Post-Param-Sub)"
        results['verification_time_seconds'] = time.time() - start_time
        return results
    
    B = B_after_param_sub # Use this version for all further processing

    # Parse set conditions (they are parsed with state variables, then parameters are subbed)
    raw_initial_rels, init_parse_msg = parse_set_conditions(initial_conditions_list, variables_sympy)
    raw_unsafe_rels, unsafe_parse_msg = parse_set_conditions(unsafe_conditions_list, variables_sympy)
    raw_safe_rels, safe_parse_msg = parse_set_conditions(safe_conditions_list, variables_sympy)

    # Substitute system parameters into parsed relationals
    initial_set_relationals = [r.subs(system_params_sympy) for r in raw_initial_rels] if raw_initial_rels else []
    unsafe_set_relationals = [r.subs(system_params_sympy) for r in raw_unsafe_rels] if raw_unsafe_rels else []
    safe_set_relationals = [r.subs(system_params_sympy) for r in raw_safe_rels] if raw_safe_rels else []
    
    if raw_initial_rels is None or raw_unsafe_rels is None or raw_safe_rels is None:
        # Update reason if parsing itself failed for any of the raw sets
        err_parts = []
        if raw_initial_rels is None: err_parts.append(f"Init: {init_parse_msg}")
        if raw_unsafe_rels is None: err_parts.append(f"Unsafe: {unsafe_parse_msg}")
        if raw_safe_rels is None: err_parts.append(f"Safe: {safe_parse_msg}")
        results['reason'] = f"Set Parsing Failed: {'; '.join(err_parts)}"
        results['verification_time_seconds'] = time.time() - start_time
        return results
    
    results['parsing_sets_successful'] = True
    logging.info("Parsed set conditions successfully.")

    # --- STEP 2.5: Dynamics Processing (System-Type Dependent) --- #
    if system_type == 'discrete':
        # Parse discrete-time dynamics
        try:
            # Prepare dynamics with parameters substituted (for discrete-time)
            preprocessed_dynamics = []
            for dyn_eq_str in dynamics_str:
                # First, substitute known system parameters in the dynamics string if any
                try:
                    combined_symbols_for_dyn = {**{v.name: v for v in variables_sympy}, **system_params_sympy}
                    temp_dyn_expr = sympy.parse_expr(dyn_eq_str.split('=')[1].strip() if '=' in dyn_eq_str else dyn_eq_str, 
                                                   local_dict=combined_symbols_for_dyn)
                    # Reconstruct the equation with parameter substitution
                    if '=' in dyn_eq_str:
                        lhs = dyn_eq_str.split('=')[0].strip()
                        preprocessed_dynamics.append(f"{lhs} = {temp_dyn_expr.subs(system_params_sympy)}")
                    else:
                        preprocessed_dynamics.append(str(temp_dyn_expr.subs(system_params_sympy)))
                except Exception:
                    # If parameter substitution fails, use original
                    preprocessed_dynamics.append(dyn_eq_str)
            
            next_state_functions = parse_discrete_dynamics(preprocessed_dynamics, variables_sympy)
            delta_B = calculate_discrete_difference(B, variables_sympy, next_state_functions)
            
            results['discrete_difference_calculated'] = str(delta_B)
            logging.info(f"Symbolic B(f(x)) - B(x) = {delta_B}")
            
            # Store parsed dynamics for further use
            parsed_dynamics_expressions = next_state_functions
            primary_condition = delta_B  # For discrete systems, this is B(f(x)) - B(x)
            
        except Exception as e:
            logging.error(f"Error processing discrete-time dynamics: {e}")
            results['reason'] = f"Error processing discrete dynamics: {e}"
            results['verification_time_seconds'] = time.time() - start_time
            return results
    else:
        # Parse continuous-time dynamics (existing logic)
        # Prepare dynamics with parameters substituted
        parsed_dynamics_str = []
        for dyn_eq_str in dynamics_str:
            try:
                # Create a combined dict for parsing dynamics: state vars + system params
                # This allows dynamics to be defined like "-a*x - y" and 'a' will be a symbol
                combined_symbols_for_dyn = {**{v.name: v for v in variables_sympy}, **system_params_sympy}
                temp_dyn_expr = sympy.parse_expr(dyn_eq_str, local_dict=combined_symbols_for_dyn)
                # Substitute numerical values of parameters into the parsed dynamic equation
                parsed_dynamics_str.append(str(temp_dyn_expr.subs(system_params_sympy))) # Convert back to string for calculate_lie_derivative
            except Exception as e_dyn_parse:
                logging.error(f"Failed to parse or substitute params in dynamic equation '{dyn_eq_str}': {e_dyn_parse}")
                results['reason'] = f"Error processing dynamics: {dyn_eq_str}"
                results['verification_time_seconds'] = time.time() - start_time
                return results

        dB_dt = calculate_lie_derivative(B, variables_sympy, parsed_dynamics_str) # calculate_lie_derivative needs strings for dynamics
        
        results['lie_derivative_calculated'] = str(dB_dt)
        logging.info(f"Symbolic dB/dt = {dB_dt}")
        
        # Store parsed dynamics for further use
        parsed_dynamics_expressions = [parse_expression(d, variables_sympy) for d in parsed_dynamics_str]
        primary_condition = dB_dt  # For continuous systems, this is dB/dt

    # Prepare numerical sampling bounds by substituting parameters
    numerical_sampling_bounds = {}
    if sampling_bounds:
        for var_name_str, bounds_list_sym in sampling_bounds.items():
            # Ensure var_name is a symbol if it needs to be looked up in system_params_sympy, though typically it's a string key
            # For bounds, sympify then substitute, then evalf to ensure float
            try:
                nb_min = sympy.sympify(bounds_list_sym[0]).subs(system_params_sympy).evalf()
                nb_max = sympy.sympify(bounds_list_sym[1]).subs(system_params_sympy).evalf()
                numerical_sampling_bounds[var_name_str] = [float(nb_min), float(nb_max)]
            except (AttributeError, TypeError, sympy.SympifyError) as e:
                logging.error(f"Could not process sampling bound '{bounds_list_sym}' for var '{var_name_str}': {e}. Check if params are defined in benchmark.")
                # Fallback or error, for now, try to use original if sympify fails (e.g. already float)
                try:
                    numerical_sampling_bounds[var_name_str] = [float(bounds_list_sym[0]), float(bounds_list_sym[1])]
                except Exception as e_fb:
                    logging.error(f"Fallback to float for sampling bounds for '{var_name_str}' also failed: {e_fb}")
                    results['reason'] = f"Invalid sampling bounds for {var_name_str}"
                    results['verification_time_seconds'] = time.time() - start_time
                    return results # Cannot proceed without valid numerical bounds
    else:
        logging.warning("No sampling_bounds provided in system_info. Numerical checks might be limited or fail.")
        # Potentially set default wide bounds if appropriate, or let downstream functions handle missing bounds

    # --- STEP 2: Check if system is suitable for SOS verification --- #
    init_polys = relationals_to_polynomials(initial_set_relationals, variables_sympy)
    unsafe_polys = relationals_to_polynomials(unsafe_set_relationals, variables_sympy)
    safe_polys = relationals_to_polynomials(safe_set_relationals, variables_sympy)

    is_poly = check_polynomial(B, primary_condition, *parsed_dynamics_expressions, variables=variables_sympy) and \
              init_polys is not None and unsafe_polys is not None and safe_polys is not None
    results['is_polynomial_system'] = is_poly
    logging.info(f"System is polynomial and suitable for SOS: {is_poly}")

    # --- STEP 3: SOS Verification Attempt (if applicable) --- #
    sos_passed = None
    if is_poly and attempt_sos:
        try:
            # Use the globally imported cp module, no need to re-import
            logging.info("Attempting SOS verification...")
            sos_passed, sos_reason, sos_details = verify_sos(
                 B.as_poly(*variables_sympy),
                 primary_condition.as_poly(*variables_sympy),  # Use primary_condition (dB/dt for continuous, delta_B for discrete)
                 init_polys,
                 unsafe_polys,
                 safe_polys,
                 variables_sympy,
                 degree=sos_default_degree,
                 sos_solver_pref=sos_solver_pref
             )
            results['sos_attempted'] = True
            results['sos_passed'] = sos_passed
            results['sos_reason'] = sos_reason
            results['sos_lie_passed'] = sos_details.get('lie_passed')
            results['sos_init_passed'] = sos_details.get('init_passed')
            results['sos_unsafe_passed'] = sos_details.get('unsafe_passed')
            logging.info(f"SOS Verification Result: Passed={sos_passed}, Reason={sos_reason}")
            
            # If SOS verification passes, set final verdict and return early
            if sos_passed:
                results['final_verdict'] = "Passed SOS Checks"
                results['reason'] = sos_reason
                # Do not return early; continue to build full summary so
                # that downstream components (e.g., web interface) always
                # receive standardized verification sections.
            else:
                # SOS failed or inconclusive, will proceed to numerical checks
                results['reason'] = f"SOS Failed/Inconclusive: {sos_reason}"
        except ImportError:
             logging.warning("CVXPY not installed. Skipping SOS verification.")
             results['sos_reason'] = "CVXPY not installed"
        except Exception as e:
             logging.error(f"Error during SOS verification setup/call: {e}")
             results['sos_attempted'] = True
             results['sos_passed'] = False # Mark as failed if error occurs
             results['sos_reason'] = f"SOS Error: {e}"

    else:
        logging.info("System not polynomial or set conversion failed. Skipping SOS.")
        results['sos_reason'] = "Not applicable (non-polynomial)"

    # --- STEP 4: Symbolic Checks (basic fallback) --- #
    if system_type == 'discrete':
        # Use discrete-time symbolic checks
        sym_condition_passed, sym_condition_reason = check_discrete_difference_symbolic(delta_B, variables_sympy, safe_set_relationals)
        results['symbolic_discrete_check_passed'] = sym_condition_passed
        results['symbolic_lie_check_passed'] = sym_condition_passed  # For compatibility
    else:
        # Use continuous-time symbolic checks (existing logic)
        sym_condition_passed, sym_condition_reason = check_lie_derivative_symbolic(dB_dt, variables_sympy, safe_set_relationals)
        results['symbolic_lie_check_passed'] = sym_condition_passed
    
    sym_bound_passed, sym_bound_reason = check_boundary_symbolic(B, variables_sympy, initial_set_relationals, unsafe_set_relationals)
    results['symbolic_boundary_check_passed'] = sym_bound_passed
    
    # Update reason if SOS wasn't conclusive/skipped
    if results['final_verdict'] == "Verification Error":
        condition_type = "Discrete Diff" if system_type == 'discrete' else "Lie"
        symbolic_reason = f"Symbolic {condition_type}: {sym_condition_reason} | Symbolic Boundary: {sym_bound_reason}"
        results['reason'] = f"SOS: {results['sos_reason']} | {symbolic_reason}"

    # --- STEP 5: Numerical Checks (Sampling & Optimization) --- #
    # Create numerical functions from symbolic expressions
    B_func = lambdify_expression(B, variables_sympy)
    primary_condition_func = lambdify_expression(primary_condition, variables_sympy)

    if B_func and primary_condition_func and numerical_sampling_bounds:
        # Perform numerical sampling checks (system-type dependent)
        if system_type == 'discrete':
            # Use discrete-time numerical checks
            num_samp_condition_passed, num_samp_condition_details = numerical_check_discrete_difference(
                primary_condition_func, numerical_sampling_bounds, variables_sympy, safe_set_relationals,
                n_samples=num_samples_lie, tolerance=numerical_tolerance
            )
            results['numerical_sampling_discrete_passed'] = num_samp_condition_passed
            results['numerical_sampling_lie_passed'] = num_samp_condition_passed  # For compatibility
        else:
            # Use continuous-time numerical checks (existing logic)
            num_samp_condition_passed, num_samp_condition_details = numerical_check_lie_derivative(
                primary_condition_func, numerical_sampling_bounds, variables_sympy, safe_set_relationals,
                n_samples=num_samples_lie, tolerance=numerical_tolerance
            )
            results['numerical_sampling_lie_passed'] = num_samp_condition_passed
        num_samp_bound_result = numerical_check_boundary(
            B_func, numerical_sampling_bounds, variables_sympy, initial_set_relationals, unsafe_set_relationals,
            n_samples=num_samples_boundary, tolerance=numerical_tolerance
        )
        
        # Unpack boundary check results
        num_samp_bound_passed, num_samp_bound_details = num_samp_bound_result
        
        results['numerical_sampling_boundary_passed'] = num_samp_bound_passed
        
        # Store detailed numerical sampling data for visualization
        condition_type = "discrete" if system_type == 'discrete' else "lie"
        results['numerical_sampling_details'] = {
            f'{condition_type}_result': num_samp_condition_details if isinstance(num_samp_condition_details, dict) else {'reason': num_samp_condition_details},
            'boundary_result': num_samp_bound_details if isinstance(num_samp_bound_details, dict) else {'reason': num_samp_bound_details},
            f'{condition_type}_violation_points': num_samp_condition_details.get('violation_points', []) if isinstance(num_samp_condition_details, dict) else [],
            'init_violation_points': num_samp_bound_details.get('init_violation_points', []) if isinstance(num_samp_bound_details, dict) else [],
            'unsafe_violation_points': num_samp_bound_details.get('unsafe_violation_points', []) if isinstance(num_samp_bound_details, dict) else []
        }
        
        # For backward compatibility, also store as 'lie_result' and 'lie_violation_points'
        if system_type == 'discrete':
            results['numerical_sampling_details']['lie_result'] = results['numerical_sampling_details']['discrete_result']
            results['numerical_sampling_details']['lie_violation_points'] = results['numerical_sampling_details']['discrete_violation_points']
        
        if isinstance(num_samp_condition_details, dict) and isinstance(num_samp_bound_details, dict):
            condition_name = "Discrete Diff" if system_type == 'discrete' else "Lie"
            combined_reason = f"{condition_name}: {num_samp_condition_details.get('reason', '')} | Boundary: {num_samp_bound_details.get('reason', '')}"
        else:
            condition_name = "Discrete Diff" if system_type == 'discrete' else "Lie"
            combined_reason = f"{condition_name}: {num_samp_condition_details if isinstance(num_samp_condition_details, str) else ''} | Boundary: {num_samp_bound_details.get('reason', '') if isinstance(num_samp_bound_details, dict) else num_samp_bound_details}"
        
        results['numerical_sampling_reason'] = combined_reason

        # Perform optimization-based falsification (if enabled)
        if attempt_optimization:
            opt_violation_found, opt_details = optimization_based_falsification(
                B_func, primary_condition_func, numerical_sampling_bounds, variables_sympy,
                initial_set_relationals, unsafe_set_relationals, safe_set_relationals,
                max_iter=opt_max_iter, pop_size=opt_pop_size, tolerance=numerical_tolerance
            )
            results['numerical_opt_attempted'] = True
            results['numerical_opt_lie_violation_found'] = opt_details.get('lie_violation_found')
            results['numerical_opt_init_violation_found'] = opt_details.get('init_violation_found')
            results['numerical_opt_unsafe_violation_found'] = opt_details.get('unsafe_violation_found')
            results['numerical_opt_reason'] = opt_details.get('reason', "Optimization check ran.")
            
            # If optimization finds violations, update numerical check results
            if opt_violation_found:
                 results['reason'] += " | Optimization found counterexample!"
                 # Update numerical pass status to reflect optimization failure
                 if results['numerical_opt_lie_violation_found']: 
                     num_samp_condition_passed = False
                 if results['numerical_opt_init_violation_found'] or results['numerical_opt_unsafe_violation_found']: 
                     num_samp_bound_passed = False
            else:
                 results['reason'] += " | Optimization found no counterexample."
        else:
             results['numerical_opt_reason'] = "Not Attempted"

        # Update overall numerical pass status based on combined checks
        numerical_overall_passed = num_samp_condition_passed and num_samp_bound_passed
    else:
         results['reason'] += " | Numerical checks skipped (lambdify failed or no numerical_sampling_bounds)."
         numerical_overall_passed = None # Indicate checks not performed

    # --- STEP 6: Determine Final Verdict --- # 
    if results['final_verdict'] != "Passed SOS Checks": # If SOS didn't already pass
        if numerical_overall_passed is True:
            results['final_verdict'] = "Passed Numerical Checks"
            results['reason'] = f"Numerical Checks Passed: {results['numerical_sampling_reason']}"
        elif numerical_overall_passed is False:
             results['final_verdict'] = "Failed Numerical Checks"
             results['reason'] = f"Numerical Checks Failed: {results['numerical_sampling_reason']}"
        elif sym_condition_passed and sym_bound_passed: # Fallback to symbolic only if numerical checks couldn't run
             condition_name = "Discrete Diff" if system_type == 'discrete' else "Lie"
             results['final_verdict'] = "Passed Symbolic Checks (Basic)"
             results['reason'] = f"Symbolic Checks Passed: {condition_name}: {sym_condition_reason} | Boundary: {sym_bound_reason}"
        else:
             condition_name = "Discrete Diff" if system_type == 'discrete' else "Lie"
             results['final_verdict'] = "Failed Symbolic Checks / Inconclusive / Error"
             results['reason'] = f"Symbolic Checks Failed/Inconclusive: {condition_name}: {sym_condition_reason} | Boundary: {sym_bound_reason}"

    logging.info(f"Final Verdict: {results['final_verdict']}. Reason: {results['reason']}")
    # (time will be stored after summary block)

    # --- STEP 7: Build Standardized Summary Blocks For Web Interface --- #
    # 1. SOS summary
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

    # 2. Symbolic summary
    symbolic_summary = {
        'success': bool(results.get('symbolic_lie_check_passed') and results.get('symbolic_boundary_check_passed')),
        'reason': f"Lie: {sym_condition_reason} | Boundary: {sym_bound_reason}",
        'details': {
            'lie_passed': results.get('symbolic_lie_check_passed'),
            'boundary_passed': results.get('symbolic_boundary_check_passed')
        }
    }

    # 3. Numerical summary (include sampling and optimization)
    numerical_success_flag = (numerical_overall_passed is True)
    numerical_summary = {
        'success': numerical_success_flag,
        'reason': results.get('numerical_sampling_reason', 'Numerical checks not executed'),
        'details': results.get('numerical_sampling_details', {})
    }

    # 4. Detect inconsistencies between SOS and numerical checks
    conflict_detected = sos_summary['success'] and (numerical_summary['success'] is False)
    if conflict_detected:
        numerical_summary['reason'] += ' | WARNING: Numerical sampling failed while SOS succeeded.'

    # 5. Store in results for external consumption
    results['sos_verification'] = sos_summary
    results['symbolic_verification'] = symbolic_summary
    results['numerical_verification'] = numerical_summary
    # Parsing summary aggregation
    results['parsing'] = {
        'candidate_parsed': results.get('parsing_B_successful'),
        'sets_parsed': results.get('parsing_sets_successful'),
        'is_polynomial_system': results.get('is_polynomial_system'),
        'system_type': results.get('system_type')
    }

    results['overall_success'] = True if results['final_verdict'].startswith('Passed') else False

    # If conflict, still treat overall_success according to SOS preference but flag it
    if conflict_detected:
        results['overall_success'] = False  # or keep True? choose conservative
        results['conflict'] = 'sos_passed_numerical_failed'

    results['verification_time_seconds'] = time.time() - start_time
    return results


# --- Example Usage (Updated - Now needs a config object) ---
if __name__ == '__main__':
    print("Testing verification logic...")
    import sys # Import sys for exit
    # This example usage won't work directly without loading a config.
    # For standalone testing, manually create a DictConfig or load the main config.
    from utils.config_loader import load_config
    cfg = load_config() # Load the default config
    if not cfg:
        sys.exit("Could not load config for testing.")

    verification_cfg = cfg.evaluation.verification # Extract verification sub-config

    test_system_1 = {
        "id": "example_1",
        "description": "Simple 2D nonlinear system, known B=x^2+y^2",
        "state_variables": ["x", "y"],
        "dynamics": [
          "-x**3 - y",
          "x - y**3"
        ],
        "initial_set_conditions": ["0.1 - x**2 - y**2 >= 0"], # SOS format G>=0
        "unsafe_set_conditions": ["x - 1.5 >= 0"],         # SOS format H>=0 means unsafe
        "safe_set_conditions": ["1.5 - x >= 0"],           # SOS format S>=0 means safe (note strict -> non-strict)
        "sampling_bounds": { "x": [-2.0, 2.0], "y": [-2.0, 2.0] }
    }

    print("\nTest Case 1: Known Valid B(x)")
    candidate_1 = "x**2 + y**2"
    result_1 = verify_barrier_certificate(candidate_1, test_system_1, verification_cfg)
    print(f"Result for '{candidate_1}': {json.dumps(result_1, indent=2)}")

    print("\nTest Case 2: Invalid Lie Derivative B(x)")
    candidate_2 = "x + y"
    result_2 = verify_barrier_certificate(candidate_2, test_system_1, verification_cfg)
    print(f"Result for '{candidate_2}': {json.dumps(result_2, indent=2)}")

    print("\nTest Case 3: Invalid Boundary B(x)")
    candidate_3 = "-x**2 - y**2"
    result_3 = verify_barrier_certificate(candidate_3, test_system_1, verification_cfg)
    print(f"Result for '{candidate_3}': {json.dumps(result_3, indent=2)}")

    print("\nTest Case 4: Invalid Syntax B(x)")
    candidate_4 = "x^^2 + y"
    result_4 = verify_barrier_certificate(candidate_4, test_system_1, verification_cfg)
    print(f"Result for '{candidate_4}': {json.dumps(result_4, indent=2)}")

    print("\nTest Case 5: Non-polynomial Dynamics (Should Skip SOS)")
    test_system_np = {
        "id": "example_np",
        "state_variables": ["x", "y"],
        "dynamics": [
            "-sin(x) + y",
            "-y + cos(x)"
        ],
         "safe_set_conditions": ["4 - x**2 - y**2 > 0"],
         "initial_set_conditions": ["0.1 - x**2 - y**2 >= 0"],
         "unsafe_set_conditions": ["x - 1.8 > 0"],
         "sampling_bounds": { "x": [-2.0, 2.0], "y": [-2.0, 2.0] }
    }
    candidate_5 = "x**2 + y**2"
    result_5 = verify_barrier_certificate(candidate_5, test_system_np, verification_cfg)
    print(f"Result for '{candidate_5}' (Non-Poly System): {json.dumps(result_5, indent=2)}") 