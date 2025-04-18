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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
            # Use sympify which can handle inequalities directly
            rel = sympy.sympify(cond_str, locals=local_dict)
            if not isinstance(rel, sympy.logic.boolalg.BooleanFunction) and not isinstance(rel, sympy.BooleanTrue) and not isinstance(rel, sympy.BooleanFalse):
                 # Check if it's a relational type (Le, Gt, etc.)
                 if not hasattr(rel, 'rel_op'):
                     raise TypeError(f"Parsed condition '{cond_str}' is not a recognized relational or boolean.")
            relationals.append(rel)
        except (SyntaxError, TypeError, sympy.SympifyError) as e:
            logging.error(f"Failed to parse set condition '{cond_str}': {e}")
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


# --- Numerical Checking Functions ---

def lambdify_expression(expr, variables):
    """Converts a SymPy expression to a NumPy-callable function."""
    if expr is None:
        return None
    try:
        # Use 'numpy' module for numerical evaluation
        func = sympy.lambdify(variables, expr, modules=['numpy'])
        return func
    except Exception as e:
        logging.error(f"Failed to lambdify expression {expr}: {e}")
        return None

def check_set_membership_numerical(point_dict, set_relationals, variables):
    """Numerically checks if a point satisfies a list of parsed SymPy relationals."""
    if not set_relationals:
        # If no conditions define the set, is a point considered inside or outside?
        # Context matters: For safe set (e.g., x < 1.5), no condition might mean the whole space.
        # For initial set (e.g., x**2 <= 0.1), no condition means empty set?
        # Let's default to True (point is considered inside if no conditions specified)
        # This needs careful consideration based on how sets are defined.
        return True

    try:
        # Substitute numerical values into each relational and evaluate
        for rel in set_relationals:
            # .evalf() evaluates the expression after substitution
            # Need to handle potential numerical precision issues with strict inequalities
            eval_result = rel.subs(point_dict).evalf()

            # Check the type of the evaluated result
            if isinstance(eval_result, sympy.logic.boolalg.BooleanAtom):
                 if not eval_result:
                     # logging.debug(f"Point {point_dict} failed condition {rel}")
                     return False # One condition failed
            else:
                 # If evalf() doesn't return a Boolean, likely symbolic evaluation failed
                 # or resulted in a non-boolean expression (e.g., due to parameters)
                 logging.warning(f"Could not evaluate condition {rel} to Boolean at point {point_dict}. Got: {eval_result}")
                 return False # Treat as failure

        return True # All conditions passed
    except Exception as e:
        logging.error(f"Error evaluating set membership for point {point_dict} with conditions {set_relationals}: {e}")
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

def numerical_check_lie_derivative(dB_dt_func, sampling_bounds, variables, safe_set_relationals, n_samples):
    """Numerically checks if dB/dt <= tolerance within the safe set using sampling."""
    logging.info(f"Performing numerical check for Lie derivative (<= {NUMERICAL_TOLERANCE}) with {n_samples} samples...")
    if dB_dt_func is None: return False, "Lie derivative function invalid (lambdify failed?)."
    if not sampling_bounds: return False, "Sampling bounds not provided."
    if safe_set_relationals is None: return False, "Safe set conditions failed to parse."

    samples = generate_samples(sampling_bounds, variables, n_samples)
    violations = 0
    checked_in_safe_set = 0

    for point_dict in samples:
        # Use the robust check with parsed relationals
        is_in_safe_set = check_set_membership_numerical(point_dict, safe_set_relationals, variables)

        if is_in_safe_set:
            checked_in_safe_set += 1
            try:
                lie_val = dB_dt_func(**point_dict)
                if lie_val > NUMERICAL_TOLERANCE:
                    violations += 1
                    logging.warning(f"Violation dB/dt <= 0: value={lie_val:.4g} at {point_dict}")
                    # Optional early exit
            except Exception as e:
                logging.error(f"Error evaluating Lie derivative at {point_dict}: {e}")

    if checked_in_safe_set == 0: return False, "No samples generated within the defined safe set/bounds."
    if violations > 0: return False, f"Found {violations}/{checked_in_safe_set} violations (dB/dt <= {NUMERICAL_TOLERANCE}) in safe set samples."
    else:
        logging.info(f"No violations found in {checked_in_safe_set} safe set samples.")
        return True, f"Passed numerical check (dB/dt <= {NUMERICAL_TOLERANCE}) in {checked_in_safe_set} safe set samples."

def numerical_check_boundary(B_func, sampling_bounds, variables, initial_set_relationals, unsafe_set_relationals, n_samples):
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

    for point_dict in samples:
        try:
            b_val = B_func(**point_dict)

            # 1. Initial Set Check: B(x) <= tolerance inside X0
            is_in_init_set = check_set_membership_numerical(point_dict, initial_set_relationals, variables)
            if is_in_init_set:
                checked_in_init += 1
                if b_val > NUMERICAL_TOLERANCE:
                    init_violations += 1
                    logging.warning(f"Violation B <= 0 in Init Set: B={b_val:.4g} at {point_dict}")

            # 2. Unsafe Set Check: B(x) >= -tolerance outside Xu
            is_in_unsafe_set = check_set_membership_numerical(point_dict, unsafe_set_relationals, variables)
            if not is_in_unsafe_set:
                 checked_outside_unsafe += 1
                 if b_val < -NUMERICAL_TOLERANCE:
                     unsafe_violations += 1
                     logging.warning(f"Violation B >= 0 outside Unsafe Set: B={b_val:.4g} at {point_dict}")

        except Exception as e:
            logging.error(f"Error evaluating B(x) or conditions at {point_dict}: {e}")

    reason = []
    boundary_ok = True
    if initial_set_relationals: # Only report if conditions were given
        if checked_in_init > 0:
            if init_violations > 0:
                boundary_ok = False
                reason.append(f"Failed Initial Set ({init_violations}/{checked_in_init} violates B <= {NUMERICAL_TOLERANCE}).")
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
                reason.append(f"Failed Unsafe Set ({unsafe_violations}/{checked_outside_unsafe} violates B >= {-NUMERICAL_TOLERANCE}).")
            else:
                reason.append(f"Passed Unsafe Set ({checked_outside_unsafe} samples).")
        else:
             reason.append("Unsafe Set check skipped (no samples outside set).")
    else:
        reason.append("Unsafe Set check skipped (no conditions).")

    if not reason: reason.append("No boundary checks performed.")

    return boundary_ok, " | ".join(reason)

# --- SOS Helper Functions ---

def get_monomials(variables, degree):
    """Generates a list of sympy monomials up to a given degree."""
    monoms = [sympy.S.One]
    for i in range(1, degree + 1):
        monoms.extend(list(sympy.ordered(sympy.monomials(variables, i))))
    unique_monoms = sorted(list(set(monoms)), key=sympy.default_sort_key)
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

    # Efficiently compute coefficients using nested loops over basis monomials
    # Coeff( (sum_ij Qij Zi Zj) * g ) [m_k] = sum_ij Qij * Coeff( Zi Zj g )[m_k]
    for i in range(s_basis_len):
        for j in range(i, s_basis_len): # Exploit symmetry Q_ij = Q_ji
            # Calculate the sympy poly for Zi * Zj * g (or just Zi * Zj if no multiplier)
            term_poly_expr = sympy.expand(s_basis[i] * s_basis[j])
            if multiplier:
                term_poly_expr = sympy.expand(term_poly_expr * multiplier.as_expr())
            
            term_poly = term_poly_expr.as_poly(*sympy_vars)
            term_coeffs = term_poly.as_dict()
            
            factor = 1 if i == j else 2 # Account for symmetry Q_ij = Q_ji

            # Add contribution to the CVXPY coefficient vector
            for monom_tuple, coeff_val in term_coeffs.items():
                monom_sympy = sympy.S.One
                for k_var, exp in enumerate(monom_tuple):
                    monom_sympy *= sympy_vars[k_var]**exp
                    
                if monom_sympy in eq_basis_map:
                    eq_idx = eq_basis_map[monom_sympy]
                    # Add coeff_val * factor * Q[i, j] to the cvxpy expression
                    try:
                         cvxpy_coeffs[eq_idx] += float(coeff_val) * factor * Q_matrix[i, j]
                    except TypeError as te:
                         # Handle cases where coeff_val might not be purely numeric (though it should be)
                         logging.error(f"TypeError converting coeff {coeff_val} for monom {monom_sympy}: {te}")
                         return None
                elif float(coeff_val) != 0.0:
                    # If a non-zero coefficient exists for a monomial outside the equality basis, error
                    logging.error(f"Monomial {monom_sympy} (coeff {coeff_val}) from product Z_{i}*Z_{j}*{multiplier.expr if multiplier else 1} not in equality basis.")
                    return None
                    
    # Use vstack to create a column vector expression
    return cp.vstack(cvxpy_coeffs)

def add_sos_constraints_poly(target_coeffs_np, set_polys, cp_vars, sympy_vars, mult_degree, eq_basis):
    """Helper to add SOS constraints: target_coeffs == coeffs(s0 + sum(si*gi))"""
    constraints = []
    s_basis_degree = mult_degree // 2
    s_basis = get_monomials(sympy_vars, s_basis_degree)
    s_basis_len = len(s_basis)
    Q_vars = {}

    # s0 term
    Q0 = cp.Variable((s_basis_len, s_basis_len), name="Q0", PSD=True)
    # constraints.append(Q0 >> 0) # PSD=True implies this
    Q_vars["Q0"] = Q0
    s0_coeffs_expr = calculate_sos_poly_coeffs(s_basis, Q0, eq_basis, sympy_vars)
    if s0_coeffs_expr is None: logging.error("Failed calculating SOS coeffs for s0"); return None, None
    rhs_total_coeffs_expr = s0_coeffs_expr

    # s_i * g_i terms
    for i, g_poly in enumerate(set_polys):
        Qi = cp.Variable((s_basis_len, s_basis_len), name=f"Q_{i+1}", PSD=True)
        # constraints.append(Qi >> 0)
        Q_vars[f"Q_{i+1}"] = Qi
        si_gi_coeffs_expr = calculate_sos_poly_coeffs(s_basis, Qi, eq_basis, sympy_vars, multiplier=g_poly)
        if si_gi_coeffs_expr is None: logging.error(f"Failed calculating SOS coeffs for s_{i+1}*g_{i+1}"); return None, None
        rhs_total_coeffs_expr += si_gi_coeffs_expr

    # Equality constraint (vectorized)
    # Ensure target_coeffs_np is treated as a column vector if needed by CVXPY
    constraints.append(target_coeffs_np.reshape(-1, 1) == rhs_total_coeffs_expr)
    
    return constraints, Q_vars

# --- Sum-of-Squares (SOS) Verification --- #
def verify_sos(B_poly, dB_dt_poly, initial_polys, unsafe_polys, safe_polys, variables, degree=SOS_DEFAULT_DEGREE):
    """Attempts to verify barrier conditions using SOS via CVXPY."""
    logging.info(f"Attempting SOS verification (degree {degree})...")
    available_solvers = cp.installed_solvers()
    solver = None
    if SOS_SOLVER in available_solvers: solver = SOS_SOLVER
    elif cp.SCS in available_solvers: solver = cp.SCS; logging.warning(f"Preferred solver {SOS_SOLVER} not found. Using SCS.")
    else: return None, None, None, "No suitable SDP solver (MOSEK or SCS) found by CVXPY."

    results = { "lie_passed": None, "lie_reason": "Not attempted", "init_passed": None, "init_reason": "Not attempted", "unsafe_passed": None, "unsafe_reason": "Not attempted" }

    try:
        # --- SOS Problem Setup --- #
        # CVXPY variables are symbolic placeholders, not directly used in coeff calculation
        cp_vars_dummy = [cp.Variable(name=v.name) for v in variables]
        
        # Determine max degree needed for equality basis
        max_deg_lie = max(dB_dt_poly.total_degree(), max((p.total_degree() + degree for p in safe_polys), default=0))
        max_deg_init = max(B_poly.total_degree(), max((p.total_degree() + degree for p in initial_polys), default=0))
        # For unsafe check B>=0 on {k>=0}, max degree is max(deg(B), deg(k)+deg(s_k))
        max_deg_unsafe = max(B_poly.total_degree(), max((p.total_degree() + degree for p in unsafe_polys), default=0))
        max_expr_deg = max(max_deg_lie, max_deg_init, max_deg_unsafe)
        equality_basis = get_monomials(variables, max_expr_deg)

        # --- 1. Lie Derivative Check (-dB/dt is SOS on Safe Set {g_i >= 0}) --- #
        logging.info("Checking Lie derivative condition via SOS...")
        target_coeffs_lie = sympy_poly_to_coeffs(-dB_dt_poly, equality_basis, variables)
        if target_coeffs_lie is None: raise ValueError("Failed getting coeffs for -dB/dt")
        constraints_lie, _ = add_sos_constraints_poly(target_coeffs_lie, safe_polys, cp_vars_dummy, variables, degree, equality_basis)
        if constraints_lie is None: raise ValueError("Failed formulating Lie SOS constraints.")
        prob_lie = cp.Problem(cp.Minimize(0), constraints_lie)
        prob_lie.solve(solver=solver, verbose=False)
        results["lie_passed"] = prob_lie.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]
        results["lie_reason"] = f"SOS Solver Status: {prob_lie.status}"

        # --- 2. Initial Set Check (-B is SOS on Initial Set {h_j >= 0}) --- #
        logging.info("Checking Initial Set condition via SOS...")
        target_coeffs_init = sympy_poly_to_coeffs(-B_poly, equality_basis, variables)
        if target_coeffs_init is None: raise ValueError("Failed getting coeffs for -B")
        constraints_init, _ = add_sos_constraints_poly(target_coeffs_init, initial_polys, cp_vars_dummy, variables, degree, equality_basis)
        if constraints_init is None: raise ValueError("Failed formulating Init SOS constraints.")
        prob_init = cp.Problem(cp.Minimize(0), constraints_init)
        prob_init.solve(solver=solver, verbose=False)
        results["init_passed"] = prob_init.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]
        results["init_reason"] = f"SOS Solver Status: {prob_init.status}"

        # --- 3. Unsafe Set Check (B is SOS on Unsafe Set {k_l >= 0}) --- #
        # Checks if B(x) >= 0 when k_l(x) >= 0 (where {k_l>=0} defines unsafe set)
        # Adapt this if your unsafe condition requires B <= 0 inside unsafe
        logging.info("Checking Unsafe Set condition (B>=0 on Unsafe) via SOS...")
        target_coeffs_unsafe = sympy_poly_to_coeffs(B_poly, equality_basis, variables)
        if target_coeffs_unsafe is None: raise ValueError("Failed getting coeffs for B")
        constraints_unsafe, _ = add_sos_constraints_poly(target_coeffs_unsafe, unsafe_polys, cp_vars_dummy, variables, degree, equality_basis)
        if constraints_unsafe is None: raise ValueError("Failed formulating Unsafe SOS constraints.")
        prob_unsafe = cp.Problem(cp.Minimize(0), constraints_unsafe)
        prob_unsafe.solve(solver=solver, verbose=False)
        results["unsafe_passed"] = prob_unsafe.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]
        results["unsafe_reason"] = f"SOS Solver Status: {prob_unsafe.status}"

    # --- Handle Errors --- #
    except ValueError as ve:
        err_msg = f"SOS Value Error: {ve}"
        logging.error(err_msg)
        results['lie_reason'] = results['lie_reason'] if results['lie_reason'] != "Not attempted" else err_msg
        results['init_reason'] = results['init_reason'] if results['init_reason'] != "Not attempted" else err_msg
        results['unsafe_reason'] = results['unsafe_reason'] if results['unsafe_reason'] != "Not attempted" else err_msg
    except cp.SolverError as se:
        err_msg = f"CVXPY Solver Error: {se}"
        logging.error(err_msg)
        results['lie_reason'] = results['lie_reason'] if results['lie_reason'] != "Not attempted" else err_msg
        results['init_reason'] = results['init_reason'] if results['init_reason'] != "Not attempted" else err_msg
        results['unsafe_reason'] = results['unsafe_reason'] if results['unsafe_reason'] != "Not attempted" else err_msg
    except Exception as e:
        err_msg = f"Unexpected SOS Error: {e}"
        logging.error(err_msg, exc_info=True)
        results['lie_reason'] = results['lie_reason'] if results['lie_reason'] != "Not attempted" else err_msg
        results['init_reason'] = results['init_reason'] if results['init_reason'] != "Not attempted" else err_msg
        results['unsafe_reason'] = results['unsafe_reason'] if results['unsafe_reason'] != "Not attempted" else err_msg

    # --- Final Result --- #
    sos_passed = None
    if None not in [results["lie_passed"], results["init_passed"], results["unsafe_passed"]]:
        sos_passed = results["lie_passed"] and results["init_passed"] and results["unsafe_passed"]
    sos_reason = f"Lie: {results['lie_reason']} | Init: {results['init_reason']} | Unsafe: {results['unsafe_reason']}"
    return sos_passed, sos_reason, results

# --- Optimization-Based Falsification --- #

def objective_maximize_lie(x, dB_dt_func, variables, safe_set_relationals):
    """Objective function to maximize Lie derivative (-minimize -dBdt)
       Returns large penalty if outside safe set."""
    point_dict = {var.name: val for var, val in zip(variables, x)}
    if not check_set_membership_numerical(point_dict, safe_set_relationals, variables):
        return 1e10 # Large penalty for being outside safe set
    try:
        lie_val = dB_dt_func(**point_dict)
        return -lie_val # Minimize negative Lie derivative
    except Exception:
        return 1e10 # Penalize errors

def objective_maximize_b_in_init(x, b_func, variables, initial_set_relationals):
    """Objective function to maximize B(x) in initial set (-minimize -B)
       Returns large penalty if outside initial set."""
    point_dict = {var.name: val for var, val in zip(variables, x)}
    if not check_set_membership_numerical(point_dict, initial_set_relationals, variables):
        return 1e10 # Large penalty
    try:
        b_val = b_func(**point_dict)
        return -b_val # Minimize negative B
    except Exception:
        return 1e10

def objective_minimize_b_outside_unsafe(x, b_func, variables, unsafe_set_relationals):
    """Objective function to minimize B(x) *outside* unsafe set.
       Returns large penalty if *inside* unsafe set."""
    point_dict = {var.name: val for var, val in zip(variables, x)}
    if check_set_membership_numerical(point_dict, unsafe_set_relationals, variables):
        return 1e10 # Large penalty for being *inside* unsafe set
    try:
        b_val = b_func(**point_dict)
        return b_val # Minimize B
    except Exception:
        return 1e10

def optimization_based_falsification(B_func, dB_dt_func, sampling_bounds, variables,
                                     initial_set_relationals, unsafe_set_relationals, safe_set_relationals):
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
                maxiter=OPTIMIZATION_MAX_ITER, popsize=OPTIMIZATION_POP_SIZE,
                tol=0.01, mutation=(0.5, 1), recombination=0.7, updating='immediate'
            )
            if opt_result_lie.success:
                max_lie_val = -opt_result_lie.fun
                results["lie_max_val"] = max_lie_val
                results["lie_point"] = {v.name: val for v, val in zip(variables, opt_result_lie.x)}
                if max_lie_val > NUMERICAL_TOLERANCE:
                    results["lie_violation_found"] = True
                    logging.warning(f"Optimization found Lie violation: max(dB/dt)={max_lie_val:.4g} > {NUMERICAL_TOLERANCE} at {results['lie_point']}")
                else:
                    results["lie_violation_found"] = False
            else:
                 results["lie_reason"] = "Lie opt failed/inconclusive."

        # 2. Check Initial Set
        if B_func and initial_set_relationals is not None:
            opt_result_init = differential_evolution(
                objective_maximize_b_in_init, bounds,
                args=(B_func, variables, initial_set_relationals),
                maxiter=OPTIMIZATION_MAX_ITER, popsize=OPTIMIZATION_POP_SIZE,
                tol=0.01, mutation=(0.5, 1), recombination=0.7, updating='immediate'
            )
            if opt_result_init.success:
                 max_b_init = -opt_result_init.fun
                 results["init_max_val"] = max_b_init
                 results["init_point"] = {v.name: val for v, val in zip(variables, opt_result_init.x)}
                 if max_b_init > NUMERICAL_TOLERANCE:
                     results["init_violation_found"] = True
                     logging.warning(f"Optimization found Initial Set violation: max(B)={max_b_init:.4g} > {NUMERICAL_TOLERANCE} at {results['init_point']}")
                 else:
                     results["init_violation_found"] = False
            else:
                 results["init_reason"] = "Init opt failed/inconclusive."

        # 3. Check Unsafe Set
        if B_func and unsafe_set_relationals is not None:
             opt_result_unsafe = differential_evolution(
                 objective_minimize_b_outside_unsafe, bounds,
                 args=(B_func, variables, unsafe_set_relationals),
                 maxiter=OPTIMIZATION_MAX_ITER, popsize=OPTIMIZATION_POP_SIZE,
                 tol=0.01, mutation=(0.5, 1), recombination=0.7, updating='immediate'
             )
             if opt_result_unsafe.success:
                 min_b_outside_unsafe = opt_result_unsafe.fun
                 results["unsafe_min_val"] = min_b_outside_unsafe
                 results["unsafe_point"] = {v.name: val for v, val in zip(variables, opt_result_unsafe.x)}
                 # Check if B is significantly negative OUTSIDE unsafe set
                 if min_b_outside_unsafe < -NUMERICAL_TOLERANCE:
                     results["unsafe_violation_found"] = True
                     logging.warning(f"Optimization found Unsafe Set violation: min(B)={min_b_outside_unsafe:.4g} < {-NUMERICAL_TOLERANCE} at {results['unsafe_point']}")
                 else:
                     results["unsafe_violation_found"] = False
             else:
                 results["unsafe_reason"] = "Unsafe opt failed/inconclusive."

    except Exception as e:
        logging.error(f"Error during optimization-based falsification: {e}")
        results["reason"] = f"Optimization Error: {e}"

    overall_violation_found = results["lie_violation_found"] or results["init_violation_found"] or results["unsafe_violation_found"]
    return overall_violation_found, results


# --- Main Verification Function (Integrates SOS and Optimization) ---

def verify_barrier_certificate(candidate_B_str, system_info, attempt_sos=True, attempt_optimization=True):
    """Main verification function. Parses conditions, checks SOS, symbolic, numerical sampling, and optimization."""
    start_time = time.time()
    logging.info(f"--- Verifying Candidate: {candidate_B_str} --- ")
    logging.info(f"System ID: {system_info.get('id', 'N/A')}")

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

    # --- Basic Setup & Parsing --- #
    state_vars_str = system_info.get('state_variables', [])
    dynamics_str = system_info.get('dynamics', [])
    initial_conditions_list = system_info.get('initial_set_conditions', [])
    unsafe_conditions_list = system_info.get('unsafe_set_conditions', [])
    safe_conditions_list = system_info.get('safe_set_conditions', [])
    sampling_bounds = system_info.get('sampling_bounds', None)

    if not state_vars_str or not dynamics_str: results['reason'] = "Incomplete system info"; results['verification_time_seconds'] = time.time() - start_time; return results
    variables = [sympy.symbols(var) for var in state_vars_str]
    B = parse_expression(candidate_B_str, variables)
    if B is None: results['reason'] = "Failed to parse candidate B(x)."; results['verification_time_seconds'] = time.time() - start_time; return results
    results['parsing_B_successful'] = True
    logging.info(f"Parsed B(x) = {B}")

    initial_set_relationals, init_parse_msg = parse_set_conditions(initial_conditions_list, variables)
    unsafe_set_relationals, unsafe_parse_msg = parse_set_conditions(unsafe_conditions_list, variables)
    safe_set_relationals, safe_parse_msg = parse_set_conditions(safe_conditions_list, variables)
    if initial_set_relationals is None or unsafe_set_relationals is None or safe_set_relationals is None:
        results['reason'] = f"Set Parsing Failed: Init: {init_parse_msg}, Unsafe: {unsafe_parse_msg}, Safe: {safe_parse_msg}"; results['verification_time_seconds'] = time.time() - start_time; return results
    results['parsing_sets_successful'] = True
    logging.info("Parsed set conditions successfully.")

    dB_dt = calculate_lie_derivative(B, variables, dynamics_str)
    if dB_dt is None: results['reason'] = "Failed to calculate Lie derivative."; results['verification_time_seconds'] = time.time() - start_time; return results
    results['lie_derivative_calculated'] = str(dB_dt)
    logging.info(f"Symbolic dB/dt = {dB_dt}")

    # --- Check if Polynomial for SOS --- #
    dynamics_exprs = [parse_expression(d, variables) for d in dynamics_str]
    init_polys = relationals_to_polynomials(initial_set_relationals, variables)
    unsafe_polys = relationals_to_polynomials(unsafe_set_relationals, variables) # NOTE: Needs complement logic depending on B sign check
    safe_polys = relationals_to_polynomials(safe_set_relationals, variables)

    is_poly = check_polynomial(B, dB_dt, *dynamics_exprs, variables=variables) and \
              init_polys is not None and unsafe_polys is not None and safe_polys is not None
    results['is_polynomial_system'] = is_poly
    logging.info(f"System is polynomial and suitable for SOS: {is_poly}")

    # --- SOS Verification Attempt --- #
    sos_passed = None # Use None to indicate not attempted or failed setup
    if is_poly and attempt_sos:
        try:
            # Ensure cvxpy is available before calling SOS function
            import cvxpy as cp
            logging.info("Attempting SOS verification...")
            sos_passed, sos_reason, sos_details = verify_sos(
                 B.as_poly(*variables),
                 dB_dt.as_poly(*variables),
                 init_polys, # Requires >= 0 format
                 unsafe_polys, # Requires >= 0 format (Check sign usage in verify_sos)
                 safe_polys, # Requires >= 0 format
                 variables
             )
            results['sos_attempted'] = True
            results['sos_passed'] = sos_passed
            results['sos_reason'] = sos_reason
            results['sos_lie_passed'] = sos_details.get('lie_passed')
            results['sos_init_passed'] = sos_details.get('init_passed')
            results['sos_unsafe_passed'] = sos_details.get('unsafe_passed')
            logging.info(f"SOS Verification Result: Passed={sos_passed}, Reason={sos_reason}")
            # If SOS passes definitively, set final verdict and return
            if sos_passed:
                results['final_verdict'] = "Passed SOS Checks"
                results['reason'] = sos_reason
                results['verification_time_seconds'] = time.time() - start_time
                return results
            else:
                # SOS failed or inconclusive, will proceed to numerical
                results['reason'] = f"SOS Failed/Inconclusive: {sos_reason}"
                current_reason = results['reason']

        except ImportError:
             logging.warning("CVXPY not installed. Skipping SOS verification.")
             results['sos_reason'] = "CVXPY not installed"
             current_reason = "SOS Skipped (CVXPY not installed)"
        except Exception as e:
             logging.error(f"Error during SOS verification setup/call: {e}")
             results['sos_attempted'] = True
             results['sos_passed'] = False # Mark as failed if error occurs
             results['sos_reason'] = f"SOS Error: {e}"
             current_reason = results['sos_reason']
    else:
        logging.info("System not polynomial or set conversion failed. Skipping SOS.")
        results['sos_reason'] = "Not applicable (non-polynomial)"
        current_reason = "SOS Skipped (Non-polynomial)"

    # --- Symbolic Checks (Basic Fallback) --- #
    sym_lie_passed, sym_lie_reason = check_lie_derivative_symbolic(dB_dt, variables, safe_set_relationals)
    results['symbolic_lie_check_passed'] = sym_lie_passed
    sym_bound_passed, sym_bound_reason = check_boundary_symbolic(B, variables, initial_set_relationals, unsafe_set_relationals)
    results['symbolic_boundary_check_passed'] = sym_bound_passed
    # Append symbolic reason if SOS wasn't conclusive/skipped
    if results['final_verdict'] == "Verification Error":
        current_reason += f" | Symbolic Lie: {sym_lie_reason} | Symbolic Boundary: {sym_bound_reason}"

    # --- Numerical Checks (Sampling & Optimization) --- #
    B_func = lambdify_expression(B, variables)
    dB_dt_func = lambdify_expression(dB_dt, variables)

    if B_func and dB_dt_func and sampling_bounds:
        # Sampling
        num_samp_lie_passed, num_samp_lie_reason = numerical_check_lie_derivative(
            dB_dt_func, sampling_bounds, variables, safe_set_relationals, NUM_SAMPLES_LIE
        )
        num_samp_bound_passed, num_samp_bound_reason = numerical_check_boundary(
            B_func, sampling_bounds, variables, initial_set_relationals, unsafe_set_relationals, NUM_SAMPLES_BOUNDARY
        )
        results['numerical_sampling_lie_passed'] = num_samp_lie_passed
        results['numerical_sampling_boundary_passed'] = num_samp_bound_passed
        current_reason = f"Sampling Lie: {num_samp_lie_reason} | Sampling Boundary: {num_samp_bound_reason}"
        results['numerical_sampling_reason'] = current_reason

        # Optimization Falsification
        if attempt_optimization:
            opt_violation_found, opt_details = optimization_based_falsification(B_func, dB_dt_func, sampling_bounds, variables, initial_set_relationals, unsafe_set_relationals, safe_set_relationals)
            results['numerical_opt_attempted'] = True
            results['numerical_opt_lie_violation_found'] = opt_details.get('lie_violation_found')
            results['numerical_opt_init_violation_found'] = opt_details.get('init_violation_found')
            results['numerical_opt_unsafe_violation_found'] = opt_details.get('unsafe_violation_found')
            results['numerical_opt_reason'] = opt_details.get('reason', "Optimization check ran.")
            # If optimization finds a violation, it overrides sampling results for failure
            if opt_violation_found:
                 current_reason += " | Optimization found counterexample!"
                 # Ensure numerical pass status reflects opt failure
                 if results['numerical_opt_lie_violation_found']: num_samp_lie_passed = False
                 if results['numerical_opt_init_violation_found'] or results['numerical_opt_unsafe_violation_found']: num_samp_bound_passed = False
            else:
                 current_reason += " | Optimization found no counterexample."
        else:
             results['numerical_opt_reason'] = "Not Attempted"

        # Update overall numerical pass status based on combined sampling and optimization
        numerical_overall_passed = num_samp_lie_passed and num_samp_bound_passed
        # We already set num_*_passed to False if opt found violation

    else:
         current_reason += " | Numerical checks skipped (lambdify failed or no bounds)."
         numerical_overall_passed = None # Indicate checks not performed

    results['reason'] = current_reason

    # --- Final Verdict --- # 
    if results['final_verdict'] != "Passed SOS Checks": # If SOS didn't already pass
        if numerical_overall_passed is True:
            results['final_verdict'] = "Passed Numerical Checks"
        elif numerical_overall_passed is False:
             results['final_verdict'] = "Failed Numerical Checks"
        elif sym_lie_passed and sym_bound_passed: # Fallback to symbolic only if numerical checks couldn't run
             results['final_verdict'] = "Passed Symbolic Checks (Basic)"
        else:
             results['final_verdict'] = "Failed Symbolic Checks / Inconclusive / Error"

    logging.info(f"Final Verdict: {results['final_verdict']}. Reason: {results['reason']}")
    results['verification_time_seconds'] = time.time() - start_time
    return results


# --- Example Usage (Updated) ---
if __name__ == '__main__':
    print("Testing verification logic...")

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
    result_1 = verify_barrier_certificate(candidate_1, test_system_1)
    print(f"Result for '{candidate_1}': {json.dumps(result_1, indent=2)}")

    print("\nTest Case 2: Invalid Lie Derivative B(x)")
    candidate_2 = "x + y"
    result_2 = verify_barrier_certificate(candidate_2, test_system_1)
    print(f"Result for '{candidate_2}': {json.dumps(result_2, indent=2)}")

    print("\nTest Case 3: Invalid Boundary B(x)")
    candidate_3 = "-x**2 - y**2"
    result_3 = verify_barrier_certificate(candidate_3, test_system_1)
    print(f"Result for '{candidate_3}': {json.dumps(result_3, indent=2)}")

    print("\nTest Case 4: Invalid Syntax B(x)")
    candidate_4 = "x^^2 + y"
    result_4 = verify_barrier_certificate(candidate_4, test_system_1)
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
    result_5 = verify_barrier_certificate(candidate_5, test_system_np)
    print(f"Result for '{candidate_5}' (Non-Poly System): {json.dumps(result_5, indent=2)}") 