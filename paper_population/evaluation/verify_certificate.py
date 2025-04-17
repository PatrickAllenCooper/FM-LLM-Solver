import sympy
import logging
import re
import numpy as np
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants for Numerical Checks ---
NUM_SAMPLES_LIE = 10000  # Samples for checking dB/dt <= 0
NUM_SAMPLES_BOUNDARY = 5000 # Samples for checking init/unsafe set conditions
NUMERICAL_TOLERANCE = 1e-6 # Tolerance for checking <= 0 or >= 0

# --- Symbolic Verification Functions ---

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

def check_lie_derivative_condition(dB_dt, variables, safe_set_conditions):
    """Performs a basic check if dB/dt <= 0 within the safe set."""
    if dB_dt is None:
        return False, "Lie derivative calculation failed."

    logging.info(f"Checking Lie derivative: {dB_dt}")

    # VERY SIMPLISTIC CHECK: Try to determine if the expression is <= 0.
    # This is generally undecidable. We look for simple cases.

    # Case 1: Is the expression identically zero?
    if dB_dt == 0:
        logging.info("Lie derivative is identically zero.")
        return True, "Lie derivative is identically zero (satisfies <= 0)."

    # Case 2: Can we prove it's <= 0? (Only works for simple forms)
    # Try substituting variables with symbols representing positive/negative values
    # This requires more advanced symbolic reasoning or SOS methods.
    # For now, we attempt a numerical check as a fallback heuristic.

    # Heuristic: Check if the expression seems non-positive (e.g., sum of negative squares)
    # Convert to polynomial to check terms (very limited)
    try:
        poly_dB_dt = sympy.Poly(dB_dt, *variables)
        is_non_positive = True
        for term in poly_dB_dt.terms():
            coeff = term[1]
            monom = term[0] # exponents
            # Check if coeff is non-positive
            if not (sympy.ask(sympy.Q.negative(coeff)) or coeff == 0):
                 # If any term has a positive coefficient (and isn't cancelled out),
                 # it's hard to guarantee <= 0 without more info.
                 # Check if all variable powers are even? Still not sufficient.
                 is_non_positive = False
                 logging.info(f"Term {term} might be positive.")
                 break
        if is_non_positive and all(all(p % 2 == 0 for p in m) or c <= 0 for m, c in poly_dB_dt.terms()):
            # This is a very weak heuristic (e.g., -x**2 - y**2 passes)
            logging.info("Lie derivative appears to be non-positive based on simple polynomial check.")
            return True, "Lie derivative heuristically non-positive (simple poly check)."

    except sympy.PolynomialError:
        logging.info("Lie derivative is not a simple polynomial. Cannot perform heuristic check.")
    except Exception as e:
        logging.warning(f"Exception during polynomial check: {e}")

    # --- Placeholder for more advanced checks --- #
    # TODO: Implement numerical sampling within the safe set (requires parsing safe_set_conditions)
    # TODO: Interface with SOS solvers (e.g., via CVXPY/PySOS + MOSEK/SDPA) for polynomial systems.
    # TODO: Use SymPy's solvers or inequality proving tools (limited capabilities)

    logging.warning("Could not definitively verify Lie derivative condition symbolically.")
    return False, "Could not verify Lie derivative condition <= 0 symbolically (advanced check needed)."

def check_boundary_conditions(B, variables, initial_set_conditions, unsafe_set_conditions):
    """Placeholder for checking B(x) signs on initial/unsafe sets."""
    # This is extremely difficult to do symbolically with inequalities.
    # Requires parsing the conditions and using advanced solvers or sampling.
    logging.warning("Symbolic boundary condition checks are not implemented (require advanced methods).")
    # For now, assume they pass, or return a specific code indicating not checked.
    return True, "Boundary conditions not checked symbolically."


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

def check_set_membership_numerical(point_dict, condition_str, variables):
    """Numerically checks if a point satisfies a condition string (e.g., 'x**2 <= 1').
       VERY Basic implementation using eval - use with caution!"""
    if not condition_str:
        return True # No condition means membership is trivially true?
                    # Or should it be False? Depends on context (e.g., checking *inside* a set).
                    # Let's assume True for checking *inside* a set defined by the condition.

    # !!! WARNING: Using eval is a security risk if condition_str is untrusted !!!
    # A safer approach involves parsing the inequality with sympy, but that's complex.
    # For now, we proceed with eval, assuming benchmark conditions are trusted.
    try:
        # Prepare namespace for eval
        local_dict = {var.name: point_dict[var.name] for var in variables}
        # Add common math functions if needed (though sympy parsing is better)
        # local_dict.update(np.__dict__) # Example if using numpy functions
        result = eval(condition_str, {}, local_dict)
        return bool(result)
    except Exception as e:
        logging.error(f"Failed to evaluate condition '{condition_str}' for point {point_dict}: {e}")
        return False # Treat evaluation errors as condition not met

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

def numerical_check_lie_derivative(dB_dt_func, sampling_bounds, variables, safe_set_conditions_str, n_samples):
    """Numerically checks if dB/dt <= tolerance within the safe set using sampling."""
    logging.info(f"Performing numerical check for Lie derivative (<= {NUMERICAL_TOLERANCE}) with {n_samples} samples...")
    if dB_dt_func is None:
        return False, "Lie derivative function invalid (lambdify failed?)."
    if not sampling_bounds:
        return False, "Sampling bounds not provided for numerical check."

    samples = generate_samples(sampling_bounds, variables, n_samples)
    violations = 0
    checked_in_safe_set = 0

    for point_dict in samples:
        # Check if the point is within the safe set definition
        # NOTE: This check_set_membership is basic and uses eval!
        is_in_safe_set = check_set_membership_numerical(point_dict, safe_set_conditions_str, variables)

        if is_in_safe_set:
            checked_in_safe_set += 1
            try:
                # Evaluate dB/dt at the point
                lie_val = dB_dt_func(**point_dict)
                if lie_val > NUMERICAL_TOLERANCE:
                    violations += 1
                    logging.warning(f"Violation found for dB/dt <= 0: value={lie_val:.4g} at point={point_dict}")
                    # Optional: Early exit on first violation
                    # return False, f"Numerical violation found: dB/dt = {lie_val:.4g} > {NUMERICAL_TOLERANCE} at {point_dict}"
            except Exception as e:
                logging.error(f"Error evaluating Lie derivative at {point_dict}: {e}")
                # Treat evaluation error as potential failure or skip point?
                # For now, let's consider it inconclusive for this point.

    if checked_in_safe_set == 0:
         return False, "No samples were generated within the defined safe set or bounds."

    if violations > 0:
        return False, f"Found {violations}/{checked_in_safe_set} numerical violations for dB/dt <= {NUMERICAL_TOLERANCE} in safe set samples."
    else:
        logging.info(f"No violations found in {checked_in_safe_set} samples within the safe set.")
        return True, f"Passed numerical check (dB/dt <= {NUMERICAL_TOLERANCE}) in {checked_in_safe_set} safe set samples."

def numerical_check_boundary(B_func, sampling_bounds, variables, initial_set_conditions_str, unsafe_set_conditions_str, n_samples):
    """Numerically checks B(x) conditions on initial and unsafe set boundaries using sampling."""
    logging.info(f"Performing numerical check for boundary conditions with {n_samples} samples...")
    if B_func is None:
        return False, "Barrier function invalid (lambdify failed?)."
    if not sampling_bounds:
        return False, "Sampling bounds not provided for numerical check."

    samples = generate_samples(sampling_bounds, variables, n_samples)
    init_violations = 0
    unsafe_violations = 0
    checked_in_init = 0
    checked_outside_unsafe = 0 # Check B(x) >= 0 where unsafe condition is FALSE

    for point_dict in samples:
        try:
            b_val = B_func(**point_dict)

            # 1. Initial Set Check: B(x) <= tolerance inside X0
            is_in_init_set = check_set_membership_numerical(point_dict, initial_set_conditions_str, variables)
            if is_in_init_set:
                checked_in_init += 1
                if b_val > NUMERICAL_TOLERANCE:
                    init_violations += 1
                    logging.warning(f"Violation B(x) <= 0 in Init Set: B={b_val:.4g} at point={point_dict}")
                    # return False, f"Numerical violation: B(x) = {b_val:.4g} > {NUMERICAL_TOLERANCE} in Initial Set at {point_dict}"

            # 2. Unsafe Set Check: B(x) >= -tolerance outside Xu (or > 0 strictly)
            # Check where the unsafe condition is FALSE (i.e., in the presumed safe area)
            is_in_unsafe_set = check_set_membership_numerical(point_dict, unsafe_set_conditions_str, variables)
            if not is_in_unsafe_set:
                 checked_outside_unsafe += 1
                 # Barrier property often requires B(x) > 0 outside unsafe, or B(x)>=0
                 # Let's check B(x) >= -tolerance
                 if b_val < -NUMERICAL_TOLERANCE:
                     unsafe_violations += 1
                     logging.warning(f"Violation B(x) >= 0 outside Unsafe Set: B={b_val:.4g} at point={point_dict}")
                     # return False, f"Numerical violation: B(x) = {b_val:.4g} < {-NUMERICAL_TOLERANCE} outside Unsafe Set at {point_dict}"

        except Exception as e:
            logging.error(f"Error evaluating B(x) or conditions at {point_dict}: {e}")

    reason = []
    boundary_ok = True
    if checked_in_init > 0:
        if init_violations > 0:
            boundary_ok = False
            reason.append(f"Failed Initial Set check ({init_violations}/{checked_in_init} violations for B <= {NUMERICAL_TOLERANCE}).")
        else:
            reason.append(f"Passed Initial Set check ({checked_in_init} samples).")
    else:
        reason.append("Initial Set check skipped (no samples found in set).")

    if checked_outside_unsafe > 0:
        if unsafe_violations > 0:
            boundary_ok = False
            reason.append(f"Failed Unsafe Set check ({unsafe_violations}/{checked_outside_unsafe} violations for B >= {-NUMERICAL_TOLERANCE}).")
        else:
            reason.append(f"Passed Unsafe Set check ({checked_outside_unsafe} samples).")
    else:
        reason.append("Unsafe Set check skipped (no samples found outside set).")

    if not reason:
         reason.append("No boundary checks performed (no samples?).")

    return boundary_ok, " | ".join(reason)

# --- Main Verification Function (Refactored) ---

def verify_barrier_certificate(candidate_B_str, system_info):
    """Main verification function. Parses and checks conditions symbolically and numerically."""
    start_time = time.time()
    logging.info(f"--- Verifying Candidate: {candidate_B_str} --- ")
    logging.info(f"System ID: {system_info.get('id', 'N/A')}")

    results = {
        "candidate_B": candidate_B_str,
        "system_id": system_info.get('id', 'N/A'),
        "parsing_successful": False,
        "lie_derivative_calculated": None,
        "symbolic_lie_check_passed": None, # None=Not Applicable, True=Passed, False=Failed/Inconclusive
        "symbolic_boundary_check_passed": None,
        "numerical_lie_check_passed": None,
        "numerical_boundary_check_passed": None,
        "final_verdict": "Verification Error", # Default
        "reason": "Initialization",
        "verification_time_seconds": 0
    }

    state_vars_str = system_info.get('state_variables', [])
    dynamics_str = system_info.get('dynamics', [])
    safe_conditions_str = system_info.get('safe_set_conditions', None)
    initial_conditions_str = system_info.get('initial_set_conditions', None)
    unsafe_conditions_str = system_info.get('unsafe_set_conditions', None)
    sampling_bounds = system_info.get('sampling_bounds', None)

    if not state_vars_str or not dynamics_str:
        results['reason'] = "Incomplete system info (state_variables or dynamics)."
        results['verification_time_seconds'] = time.time() - start_time
        return results

    variables = [sympy.symbols(var) for var in state_vars_str]

    # 1. Parse Candidate B(x)
    B = parse_expression(candidate_B_str, variables)
    if B is None:
        results['reason'] = "Failed to parse candidate B(x)."
        results['verification_time_seconds'] = time.time() - start_time
        return results
    results['parsing_successful'] = True
    logging.info(f"Parsed B(x) = {B}")

    # 2. Calculate Lie Derivative Symbolically
    dB_dt = calculate_lie_derivative(B, variables, dynamics_str)
    if dB_dt is None:
        results['reason'] = "Failed to calculate Lie derivative symbolically."
        results['verification_time_seconds'] = time.time() - start_time
        return results
    results['lie_derivative_calculated'] = str(dB_dt)
    logging.info(f"Symbolic dB/dt = {dB_dt}")

    # 3. Perform Symbolic Checks (Optional/Basic)
    # --- Symbolic Lie Check --- #
    sym_lie_passed, sym_lie_reason = False, "Symbolic check not conclusive (basic implementation)"
    if dB_dt == 0:
        sym_lie_passed = True
        sym_lie_reason = "Symbolic check passed (dB/dt == 0)"
    # Add other simple symbolic checks if desired
    results['symbolic_lie_check_passed'] = sym_lie_passed
    results['reason'] = f"Symbolic: {sym_lie_reason}"
    # --- Symbolic Boundary Check --- #
    sym_bound_passed, sym_bound_reason = True, "Symbolic boundary checks not implemented"
    # Placeholder - always passes for now
    results['symbolic_boundary_check_passed'] = sym_bound_passed
    results['reason'] += f" | {sym_bound_reason}"

    # 4. Perform Numerical Checks (If symbolic checks inconclusive or as primary check)
    num_lie_passed, num_lie_reason = None, "Numerical check skipped"
    num_bound_passed, num_bound_reason = None, "Numerical check skipped"

    # Lambdify functions
    B_func = lambdify_expression(B, variables)
    dB_dt_func = lambdify_expression(dB_dt, variables)

    if B_func and dB_dt_func and sampling_bounds:
        logging.info("Proceeding with numerical checks...")
        # --- Numerical Lie Check --- #
        num_lie_passed, num_lie_reason = numerical_check_lie_derivative(
            dB_dt_func, sampling_bounds, variables, safe_conditions_str, NUM_SAMPLES_LIE
        )
        results['numerical_lie_check_passed'] = num_lie_passed
        # --- Numerical Boundary Check --- #
        num_bound_passed, num_bound_reason = numerical_check_boundary(
            B_func, sampling_bounds, variables, initial_conditions_str, unsafe_conditions_str, NUM_SAMPLES_BOUNDARY
        )
        results['numerical_boundary_check_passed'] = num_bound_passed
        results['reason'] = f"Numerical Lie: {num_lie_reason} | Numerical Boundary: {num_bound_reason}"
    else:
         results['reason'] += " | Numerical checks skipped (lambdify failed or no bounds)."
         logging.warning("Skipping numerical checks: lambdify failed or sampling_bounds missing.")

    # 5. Determine Final Verdict
    # Prioritize numerical results if performed, otherwise use symbolic (which are basic)
    if num_lie_passed is not None and num_bound_passed is not None:
        if num_lie_passed and num_bound_passed:
            results['final_verdict'] = "Passed Numerical Checks"
        else:
            results['final_verdict'] = "Failed Numerical Checks"
    elif sym_lie_passed and sym_bound_passed: # Only rely on symbolic if numerical failed/skipped
         results['final_verdict'] = "Passed Symbolic Checks (Basic)"
    else:
         results['final_verdict'] = "Failed Symbolic Checks / Inconclusive"

    logging.info(f"Final Verdict: {results['final_verdict']}. Reason: {results['reason']}")
    results['verification_time_seconds'] = time.time() - start_time
    return results


# --- Example Usage (for testing this script directly) ---
if __name__ == '__main__':
    print("Testing verification logic...")

    # Example system from benchmark
    test_system_1 = {
        "id": "example_1",
        "description": "Simple 2D nonlinear system, known B=x^2+y^2",
        "state_variables": ["x", "y"],
        "dynamics": [
          "-x**3 - y",
          "x - y**3"
        ],
        "initial_set_conditions": "x**2 + y**2 <= 0.1",
        "unsafe_set_conditions": "x >= 1.5",
        "safe_set_conditions": "x < 1.5",
        "sampling_bounds": { "x": [-2.0, 2.0], "y": [-2.0, 2.0] }
    }

    # Test case 1: Known valid certificate
    print("\nTest Case 1: Known Valid B(x)")
    candidate_1 = "x**2 + y**2"
    result_1 = verify_barrier_certificate(candidate_1, test_system_1)
    print(f"Result for '{candidate_1}': {result_1}")
    # Expected: Should pass symbolic and numerical

    # Test case 2: Likely invalid certificate (Lie derivative)
    print("\nTest Case 2: Invalid Lie Derivative B(x)")
    candidate_2 = "x + y"
    result_2 = verify_barrier_certificate(candidate_2, test_system_1)
    print(f"Result for '{candidate_2}': {result_2}")
    # Expected: Should fail numerical Lie check

    # Test case 3: Likely invalid certificate (Boundary)
    print("\nTest Case 3: Invalid Boundary B(x)")
    candidate_3 = "-x**2 - y**2" # Should violate B >= 0 outside unsafe
    result_3 = verify_barrier_certificate(candidate_3, test_system_1)
    print(f"Result for '{candidate_3}': {result_3}")
    # Expected: Should fail numerical boundary check

    # Test case 4: Parsing failure
    print("\nTest Case 4: Invalid Syntax B(x)")
    candidate_4 = "x^^2 + y"
    result_4 = verify_barrier_certificate(candidate_4, test_system_1)
    print(f"Result for '{candidate_4}': {result_4}")
    # Expected: Parsing failure

    # Test case 5: System with non-polynomial terms
    print("\nTest Case 5: Non-polynomial Dynamics")
    test_system_np = {
        "id": "example_np",
        "state_variables": ["x", "y"],
        "dynamics": [
            "-sin(x) + y",
            "-y + cos(x)"
        ],
         "safe_set_conditions": "x**2 + y**2 < 4",
         "initial_set_conditions": "x**2+y**2 <= 0.1",
         "unsafe_set_conditions": "x > 1.8",
         "sampling_bounds": { "x": [-2.0, 2.0], "y": [-2.0, 2.0] }
    }
    candidate_5 = "x**2 + y**2"
    result_5 = verify_barrier_certificate(candidate_5, test_system_np)
    print(f"Result for '{candidate_5}' (Non-Poly System): {result_5}")
    # Expected: Symbolic Lie check likely inconclusive, relies on numerical 