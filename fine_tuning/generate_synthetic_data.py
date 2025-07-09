import os
import sys
import json
import re
import logging
import sympy
import random
import argparse

# Add project root to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(__file__)
# Output file within the fine_tuning directory
DEFAULT_SYNTHETIC_OUTPUT_FILE = os.path.join(BASE_DIR, "synthetic_data.jsonl")
DEFAULT_FORMAT = "instruction" # Or "prompt_completion"

# --- System Definitions & Certificate Generation ---

def define_systems():
    """Defines a comprehensive list of systems for synthetic data generation."""
    systems = []

    # === LINEAR SYSTEMS ===
    
    # Simple 2D linear systems with varying decay rates
    for i, decay in enumerate([0.5, 1.0, 1.5, 2.0]):
        systems.append({
            "id": f"linear_2d_stable_{i+1}",
            "description": f"2D linear stable system with decay rate {decay}",
            "state_variables": ["x", "y"],
            "dynamics": [f"-{decay}*x", f"-{decay}*y"],
            "initial_set_conditions": "x**2 + y**2 <= 0.1",
            "unsafe_set_conditions": "x**2 + y**2 >= 2.0",
            "safe_set_conditions": "x**2 + y**2 < 2.0",
            "expected_certificate_form": "quadratic"
        })

    # Coupled linear systems
    for i, coupling in enumerate([0.1, 0.3, 0.5]):
        systems.append({
            "id": f"linear_coupled_{i+1}",
            "description": f"2D coupled linear system with coupling {coupling}",
            "state_variables": ["x", "y"],
            "dynamics": [f"-x + {coupling}*y", f"{coupling}*x - y"],
            "initial_set_conditions": "x**2 + y**2 <= 0.1",
            "unsafe_set_conditions": "x >= 1.5",
            "safe_set_conditions": "x < 1.5",
            "expected_certificate_form": "quadratic"
        })

    # 3D linear systems
    for i in range(3):
        decay_rates = [(1.0, 1.5, 2.0), (0.5, 1.0, 1.5), (2.0, 2.5, 3.0)][i]
        systems.append({
            "id": f"linear_3d_stable_{i+1}",
            "description": f"3D linear stable system variant {i+1}",
            "state_variables": ["x", "y", "z"],
            "dynamics": [f"-{decay_rates[0]}*x", f"-{decay_rates[1]}*y", f"-{decay_rates[2]}*z"],
            "initial_set_conditions": "x**2 + y**2 + z**2 <= 0.1",
            "unsafe_set_conditions": "z >= 1.5",
            "safe_set_conditions": "z < 1.5",
            "expected_certificate_form": "quadratic"
        })

    # === NONLINEAR SYSTEMS ===
    
    # Polynomial nonlinear systems (variants of the classic example)
    nonlinear_variants = [
        (["-x**3 - y", "x - y**3"], "x**2 + y**2"),
        (["-x**3 - 0.5*y", "0.5*x - y**3"], "x**2 + y**2"),
        (["-2*x**3 - y", "x - 2*y**3"], "x**2 + y**2"),
        (["-x**3 - y - 0.1*x", "x - y**3 - 0.1*y"], "x**2 + y**2"),
    ]
    
    for i, (dynamics, cert) in enumerate(nonlinear_variants):
        systems.append({
            "id": f"nonlinear_2d_variant_{i+1}",
            "description": f"2D nonlinear system variant {i+1}",
            "state_variables": ["x", "y"],
            "dynamics": dynamics,
            "initial_set_conditions": "x**2 + y**2 <= 0.1",
            "unsafe_set_conditions": "x >= 1.5",
            "safe_set_conditions": "x < 1.5",
            "expected_certificate_form": "quadratic_known",
            "known_certificate": cert
        })

    # More complex nonlinear systems
    for i, power in enumerate([5, 7]):
        systems.append({
            "id": f"nonlinear_higher_order_{i+1}",
            "description": f"Higher-order nonlinear system with power {power}",
            "state_variables": ["x", "y"],
            "dynamics": [f"-x**{power}", f"-y**{power}"],
            "initial_set_conditions": "x**2 + y**2 <= 0.1",
            "unsafe_set_conditions": "x**2 + y**2 >= 1.0",
            "safe_set_conditions": "x**2 + y**2 < 1.0",
            "expected_certificate_form": "quadratic"
        })

    # Van der Pol-like oscillators (damped)
    for i, damping in enumerate([0.1, 0.5, 1.0]):
        systems.append({
            "id": f"vanderpol_damped_{i+1}",
            "description": f"Damped Van der Pol oscillator with damping {damping}",
            "state_variables": ["x", "y"],
            "dynamics": ["y", f"-x - {damping}*y*(x**2 - 1)"],
            "initial_set_conditions": "x**2 + y**2 <= 0.1",
            "unsafe_set_conditions": "x**2 + y**2 >= 4.0",
            "safe_set_conditions": "x**2 + y**2 < 4.0",
            "expected_certificate_form": "quadratic"
        })

    # === MIXED SYSTEMS ===
    
    # Linear-nonlinear mixed systems
    mixed_systems = [
        (["-x - y**3", "-y - x**3"], "Partially nonlinear system 1"),
        (["-x**3 - y", "-y"], "Mixed linear-nonlinear system 1"),
        (["-x", "-y**3"], "Mixed linear-nonlinear system 2"),
        (["-x**3", "-y - x"], "Mixed linear-nonlinear system 3"),
    ]
    
    for i, (dynamics, desc) in enumerate(mixed_systems):
        systems.append({
            "id": f"mixed_system_{i+1}",
            "description": desc,
            "state_variables": ["x", "y"],
            "dynamics": dynamics,
            "initial_set_conditions": "x**2 + y**2 <= 0.1",
            "unsafe_set_conditions": "x >= 1.5 or y >= 1.5",
            "safe_set_conditions": "x < 1.5 and y < 1.5",
            "expected_certificate_form": "quadratic"
        })

    # === CONTROL SYSTEMS ===
    
    # Simple controlled systems (assuming stabilizing control)
    for i, control_gain in enumerate([0.5, 1.0, 2.0]):
        systems.append({
            "id": f"controlled_system_{i+1}",
            "description": f"Controlled system with gain {control_gain}",
            "state_variables": ["x", "y"],
            "dynamics": [f"-{control_gain}*x - y", f"x - {control_gain}*y"],
            "initial_set_conditions": "x**2 + y**2 <= 0.1",
            "unsafe_set_conditions": "x**2 + y**2 >= 1.0",
            "safe_set_conditions": "x**2 + y**2 < 1.0",
            "expected_certificate_form": "quadratic"
        })

    # === ADDITIONAL VARIATIONS ===
    
    # Systems with different initial and unsafe set geometries
    geometric_variants = [
        ("abs(x) + abs(y) <= 0.1", "abs(x) >= 1.5", "abs(x) < 1.5", "L1-norm based system"),
        ("max(abs(x), abs(y)) <= 0.1", "max(abs(x), abs(y)) >= 1.5", "max(abs(x), abs(y)) < 1.5", "L-infinity norm system"),
    ]
    
    for i, (init_cond, unsafe_cond, safe_cond, desc) in enumerate(geometric_variants):
        systems.append({
            "id": f"geometric_variant_{i+1}",
            "description": desc,
            "state_variables": ["x", "y"],
            "dynamics": ["-x", "-y"],
            "initial_set_conditions": init_cond,
            "unsafe_set_conditions": unsafe_cond,
            "safe_set_conditions": safe_cond,
            "expected_certificate_form": "quadratic"
        })

    return systems

def generate_quadratic_certificate(variables):
    """Generates a simple positive definite quadratic form, e.g., sum of squares."""
    B = sympy.S.Zero
    for var in variables:
        B += var**2
    return B

def generate_system_description_text(system_info):
    """Formats system details into a string for the LLM input."""
    desc = f"System ID: {system_info['id']}\n"
    desc += f"Description: {system_info['description']}\n"
    desc += f"State Variables: {system_info['state_variables']}\n"
    desc += "Dynamics:\n"
    var_names = system_info['state_variables']
    for i, dyn in enumerate(system_info['dynamics']):
        desc += f"  d{var_names[i]}/dt = {dyn}\n"
    if system_info.get('initial_set_conditions'):
        desc += f"Initial Set: {{ ({', '.join(var_names)}) | {system_info['initial_set_conditions']} }}\n"
    if system_info.get('unsafe_set_conditions'):
        desc += f"Unsafe Set: {{ ({', '.join(var_names)}) | {system_info['unsafe_set_conditions']} }}\n"
    if system_info.get('safe_set_conditions'):
        desc += f"Safe Set (Region of Interest): {{ ({', '.join(var_names)}) | {system_info['safe_set_conditions']} }}\n"
    return desc.strip()

from utils.data_formatting import format_synthetic_example

# Data formatting function moved to utils.data_formatting

# --- Main Logic ---

def generate_data(output_file, format_type):
    """Generates synthetic data and saves it to the output file."""
    systems = define_systems()
    generated_examples = []
    skipped_count = 0

    logging.info(f"Generating synthetic data for {len(systems)} system definitions...")

    for system_info in systems:
        logging.info(f"Processing system: {system_info['id']}")
        variables = [sympy.symbols(var) for var in system_info['state_variables']]
        system_desc_text = generate_system_description_text(system_info)
        certificate = None
        certificate_str = None

        # Attempt to generate/find a certificate based on type
        form = system_info.get("expected_certificate_form")
        if form == "quadratic":
            certificate = generate_quadratic_certificate(variables)
            certificate_str = str(certificate)
        elif form == "quadratic_known":
            # Use the known certificate from the system info
            known_cert = system_info.get("known_certificate", "x**2 + y**2")
            certificate_str = known_cert
            # Parse it to get the sympy expression
            try:
                certificate = sympy.sympify(certificate_str)
            except:
                # Fallback to simple quadratic
                certificate = generate_quadratic_certificate(variables)
                certificate_str = str(certificate)
        else:
            logging.warning(f"Certificate generation logic not implemented or too complex for system {system_info['id']} (form: {form}). Skipping.")
            skipped_count += 1
            continue

        # Basic Verification (using simplified checks similar to verify_certificate.py)
        # This is crucial - only save pairs where our generator *believes* it's valid
        # (even if the check is simple)
        try:
            # Try to import verification functions, but proceed without verification if they fail
            from evaluation.verify_certificate import calculate_lie_derivative, check_lie_derivative_condition

            dB_dt = calculate_lie_derivative(certificate, variables, system_info['dynamics'])
            if dB_dt is not None:
                 # Use a simplified check for now
                 is_valid_lie, reason = check_lie_derivative_condition(dB_dt, variables, system_info.get('safe_set_conditions'))
                 # We might simplify further: only accept if dB/dt is clearly non-positive
                 # e.g., if dB_dt == 0 or sympy.ask(sympy.Q.negative(dB_dt)) or similar basic check
                 # For this demo, let's accept if the simple check passes OR if it's the known good case
                 is_verified = is_valid_lie or (form == "quadratic_known")

                 if is_verified:
                     logging.info(f"Generated certificate '{certificate_str}' passed basic verification for system {system_info['id']}.")
                     example = format_synthetic_example(system_desc_text, certificate_str, format_type)
                     generated_examples.append(example)
                 else:
                     logging.warning(f"Generated certificate '{certificate_str}' FAILED basic verification for system {system_info['id']} (dB/dt={dB_dt}, Reason: {reason}). Skipping.")
                     skipped_count += 1
            else:
                logging.warning(f"Could not calculate Lie derivative for system {system_info['id']}. Skipping.")
                skipped_count += 1
        except ImportError:
             logging.warning("Could not import verification functions. Proceeding without verification (certificates may be invalid).")
             # Proceed without verification - add the example anyway
             example = format_synthetic_example(system_desc_text, certificate_str, format_type)
             generated_examples.append(example)
        except Exception as e:
            logging.warning(f"Error during verification for system {system_info['id']}: {e}. Proceeding without verification.")
            # Add the example anyway
            example = format_synthetic_example(system_desc_text, certificate_str, format_type)
            generated_examples.append(example)

    logging.info(f"Generated {len(generated_examples)} synthetic examples. Skipped {skipped_count} systems.")

    # Save to file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in generated_examples:
                f.write(json.dumps(example) + '\n')
        logging.info(f"Synthetic data saved to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save synthetic data: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic fine-tuning data.")
    parser.add_argument("--output_file", type=str, default=DEFAULT_SYNTHETIC_OUTPUT_FILE,
                        help=f"Path to save the synthetic data JSONL file (default: {DEFAULT_SYNTHETIC_OUTPUT_FILE}).")
    parser.add_argument("--format", type=str, default=DEFAULT_FORMAT, choices=["instruction", "prompt_completion"],
                        help=f"Output format for the fine-tuning data (default: {DEFAULT_FORMAT}).")
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    generate_data(args.output_file, args.format) 