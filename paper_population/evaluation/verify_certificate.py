import sympy
import logging
import re
import numpy as np
import time
import json
import cvxpy as cp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants for Numerical Checks ---
NUM_SAMPLES_LIE = 10000  # Samples for checking dB/dt <= 0
NUM_SAMPLES_BOUNDARY = 5000 # Samples for checking init/unsafe set conditions
NUMERICAL_TOLERANCE = 1e-6 # Tolerance for checking <= 0 or >= 0
SOS_DEFAULT_DEGREE = 2 # Default degree for SOS multipliers (adjust as needed)
SOS_SOLVER = cp.MOSEK # Preferred solver (requires license)
# SOS_SOLVER = cp.SCS   # Open-source alternative

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

# --- Main Verification Function (Refactored) ---

def verify_barrier_certificate(candidate_B_str, system_info):
    """Main verification function. Parses conditions, checks SOS, symbolic, and numerical."""
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
        "numerical_lie_check_passed": None,
        "numerical_boundary_check_passed": None,
        "final_verdict": "Verification Error",
        "reason": "Initialization",
        "verification_time_seconds": 0
    }

    state_vars_str = system_info.get('state_variables', [])
    dynamics_str = system_info.get('dynamics', [])
    # Expect lists of strings now
    initial_conditions_list = system_info.get('initial_set_conditions', [])
    unsafe_conditions_list = system_info.get('unsafe_set_conditions', [])
    safe_conditions_list = system_info.get('safe_set_conditions', [])
    sampling_bounds = system_info.get('sampling_bounds', None)

    if not state_vars_str or not dynamics_str:
        results['reason'] = "Incomplete system info (state_variables or dynamics)."
        results['verification_time_seconds'] = time.time() - start_time
        return results

    variables = [sympy.symbols(var) for var in state_vars_str]

    # 1. Parse Candidate B(x)
    B = parse_expression(candidate_B_str, variables)
    if B is None: results['reason'] = "Failed to parse candidate B(x)."; results['verification_time_seconds'] = time.time() - start_time; return results
    results['parsing_B_successful'] = True
    logging.info(f"Parsed B(x) = {B}")

    # 2. Parse Set Conditions
    initial_set_relationals, init_parse_msg = parse_set_conditions(initial_conditions_list, variables)
    unsafe_set_relationals, unsafe_parse_msg = parse_set_conditions(unsafe_conditions_list, variables)
    safe_set_relationals, safe_parse_msg = parse_set_conditions(safe_conditions_list, variables)

    if initial_set_relationals is None or unsafe_set_relationals is None or safe_set_relationals is None:
        results['reason'] = f"Set Parsing Failed: Init: {init_parse_msg}, Unsafe: {unsafe_parse_msg}, Safe: {safe_parse_msg}"
        results['verification_time_seconds'] = time.time() - start_time
        return results
    results['parsing_sets_successful'] = True
    logging.info("Parsed set conditions successfully.")

    # 3. Calculate Lie Derivative Symbolically
    dB_dt = calculate_lie_derivative(B, variables, dynamics_str)
    if dB_dt is None: results['reason'] = "Failed to calculate Lie derivative."; results['verification_time_seconds'] = time.time() - start_time; return results
    results['lie_derivative_calculated'] = str(dB_dt)
    logging.info(f"Symbolic dB/dt = {dB_dt}")

    # 4. Perform Symbolic Checks (Basic)
    sym_lie_passed, sym_lie_reason = check_lie_derivative_symbolic(dB_dt, variables, safe_set_relationals)
    results['symbolic_lie_check_passed'] = sym_lie_passed
    sym_bound_passed, sym_bound_reason = check_boundary_symbolic(B, variables, initial_set_relationals, unsafe_set_relationals)
    results['symbolic_boundary_check_passed'] = sym_bound_passed
    current_reason = f"Symbolic Lie: {sym_lie_reason} | Symbolic Boundary: {sym_bound_reason}"

    # 5. Perform Numerical Checks
    num_lie_passed, num_lie_reason = None, "Numerical check skipped"
    num_bound_passed, num_bound_reason = None, "Numerical check skipped"

    B_func = lambdify_expression(B, variables)
    dB_dt_func = lambdify_expression(dB_dt, variables)

    if B_func and dB_dt_func and sampling_bounds:
        logging.info("Proceeding with numerical checks...")
        num_lie_passed, num_lie_reason = numerical_check_lie_derivative(
            dB_dt_func, sampling_bounds, variables, safe_set_relationals, NUM_SAMPLES_LIE
        )
        num_bound_passed, num_bound_reason = numerical_check_boundary(
            B_func, sampling_bounds, variables, initial_set_relationals, unsafe_set_relationals, NUM_SAMPLES_BOUNDARY
        )
        current_reason = f"Numerical Lie: {num_lie_reason} | Numerical Boundary: {num_bound_reason}"
    else:
         current_reason += " | Numerical checks skipped (lambdify failed or no bounds)."
         logging.warning("Skipping numerical checks: lambdify failed or sampling_bounds missing.")

    results['numerical_lie_check_passed'] = num_lie_passed
    results['numerical_boundary_check_passed'] = num_bound_passed
    results['reason'] = current_reason

    # 6. Determine Final Verdict
    # Prioritize numerical results if performed
    if num_lie_passed is not None and num_bound_passed is not None:
        if num_lie_passed and num_bound_passed:
            results['final_verdict'] = "Passed Numerical Checks"
        else:
            results['final_verdict'] = "Failed Numerical Checks"
    # Fallback to basic symbolic if numerical checks were skipped but symbolic passed
    elif sym_lie_passed and sym_bound_passed:
         results['final_verdict'] = "Passed Symbolic Checks (Basic)"
    # Otherwise, inconclusive or failed symbolic
    else:
         results['final_verdict'] = "Failed Symbolic Checks / Inconclusive"

    logging.info(f"Final Verdict: {results['final_verdict']}. Reason: {results['reason']}")
    results['verification_time_seconds'] = time.time() - start_time
    return results


# --- Example Usage (Updated for list-based conditions) ---
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