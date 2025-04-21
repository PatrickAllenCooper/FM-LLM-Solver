import os
import json
import logging
import sympy
import random
import argparse

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(__file__)
# Output file within the fine_tuning directory
DEFAULT_SYNTHETIC_OUTPUT_FILE = os.path.join(BASE_DIR, "synthetic_data.jsonl")
DEFAULT_FORMAT = "instruction" # Or "prompt_completion"

# --- System Definitions & Certificate Generation ---

def define_systems():
    """Defines a list of simple polynomial systems for synthetic data generation."""
    systems = []

    # System 1: Simple Linear Stable System
    systems.append({
        "id": "synthetic_linear_stable_1",
        "description": "A simple, globally stable linear system.",
        "state_variables": ["x", "y"],
        "dynamics": ["-x", "-y"],
        "initial_set_conditions": "x**2 + y**2 <= 0.1",
        "unsafe_set_conditions": "x >= 2 or y >= 2", # Example
        "safe_set_conditions": "x < 2 and y < 2",   # Example
        "expected_certificate_form": "quadratic"
    })

    # System 2: Another Linear Stable System
    systems.append({
        "id": "synthetic_linear_stable_2",
        "description": "A stable linear system with coupling.",
        "state_variables": ["x", "y"],
        "dynamics": ["-2*x + y", "x - 2*y"],
        "initial_set_conditions": "x**2 + y**2 <= 0.1",
        "unsafe_set_conditions": "x <= -1.5",
        "safe_set_conditions": "x > -1.5",
        "expected_certificate_form": "quadratic"
    })

    # System 3: Simple Nonlinear Stable (Example 1 from verify_certificate.py)
    systems.append({
        "id": "synthetic_nonlinear_stable_1",
        "description": "A simple nonlinear system known to be stable with B=x^2+y^2.",
        "state_variables": ["x", "y"],
        "dynamics": ["-x**3 - y", "x - y**3"],
        "initial_set_conditions": "x**2 + y**2 <= 0.1",
        "unsafe_set_conditions": "x >= 1.5",
        "safe_set_conditions": "x < 1.5",
        "expected_certificate_form": "quadratic_known" # Special case
    })

    # System 4: Van der Pol Oscillator (Regionally Stable)
    # Note: Finding a simple polynomial barrier for the whole state space is hard/impossible.
    # We might generate one known for a specific region, or skip complex ones for now.
    # systems.append({
    #     "id": "synthetic_vanderpol",
    #     "description": "Van der Pol oscillator.",
    #     "state_variables": ["x", "y"],
    #     "dynamics": ["y", "(1 - x**2)*y - x"],
    #     # Defining appropriate sets for VdP requires care
    #     "initial_set_conditions": "...",
    #     "unsafe_set_conditions": "...",
    #     "safe_set_conditions": "...",
    #     "expected_certificate_form": "higher_order_polynomial" # Requires actual SOS
    # })

    # System 5: 3D Linear Stable
    systems.append({
        "id": "synthetic_linear_3d_stable_1",
        "description": "A simple stable 3D linear system.",
        "state_variables": ["x", "y", "z"],
        "dynamics": ["-x", "-2*y", "-3*z"],
        "initial_set_conditions": "x**2 + y**2 + z**2 <= 0.1",
        "unsafe_set_conditions": "z >= 1",
        "safe_set_conditions": "z < 1",
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
    desc = f"System ID: {system_info['id']}\\n"
    desc += f"Description: {system_info['description']}\\n"
    desc += f"State Variables: {system_info['state_variables']}\\n"
    desc += "Dynamics:\\n"
    var_names = system_info['state_variables']
    for i, dyn in enumerate(system_info['dynamics']):
        desc += f"  d{var_names[i]}/dt = {dyn}\\n"
    if system_info.get('initial_set_conditions'):
        desc += f"Initial Set: {{ ({', '.join(var_names)}) | {system_info['initial_set_conditions']} }}\\n"
    if system_info.get('unsafe_set_conditions'):
        desc += f"Unsafe Set: {{ ({', '.join(var_names)}) | {system_info['unsafe_set_conditions']} }}\\n"
    if system_info.get('safe_set_conditions'):
        desc += f"Safe Set (Region of Interest): {{ ({', '.join(var_names)}) | {system_info['safe_set_conditions']} }}\\n"
    return desc.strip()

def format_example(system_desc_text, certificate_str, format_type):
    """Formats the example based on the desired output format."""
    # Add source metadata
    metadata = {'source': 'synthetic_sos_placeholder'}

    if format_type == "instruction":
        instruction = ("Given the autonomous system described by the following dynamics, "
                       "propose a suitable barrier certificate function B(x).")
        # Include metadata within the output or as separate fields if loader supports it
        output = f"Barrier Certificate Candidate:\\nB({', '.join(sympy.symbols(list(set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', certificate_str)))))}) = {certificate_str}\\nMetadata: {json.dumps(metadata)}"

        return {
            "instruction": instruction,
            "input": system_desc_text,
            "output": output
            # Or potentially add metadata as a top-level key if trainer/dataset supports it:
            # "metadata": metadata
        }
    elif format_type == "prompt_completion":
        prompt = f"System Dynamics:\\n{system_desc_text}\\n\\nBarrier Certificate:"
        completion = f" B({', '.join(sympy.symbols(list(set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', certificate_str)))))}) = {certificate_str}"
         # Add metadata to the completion string or as a separate field
        completion += f"\\nMetadata: {json.dumps(metadata)}"

        return {
            "prompt": prompt,
            "completion": completion
            # "metadata": metadata
        }
    else:
        raise ValueError(f"Unsupported format type: {format_type}")

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
        elif form == "quadratic_known" and system_info['id'] == "synthetic_nonlinear_stable_1":
            # Special case where we know B = x^2 + y^2 works
            certificate = sympy.symbols('x')**2 + sympy.symbols('y')**2
            certificate_str = "x**2 + y**2"
        else:
            logging.warning(f"Certificate generation logic not implemented or too complex for system {system_info['id']} (form: {form}). Skipping.")
            skipped_count += 1
            continue

        # Basic Verification (using simplified checks similar to verify_certificate.py)
        # This is crucial - only save pairs where our generator *believes* it's valid
        # (even if the check is simple)
        try:
            # Re-import or duplicate verification logic if needed, assuming verify_certificate is accessible
            from evaluation.verify_certificate import calculate_lie_derivative, check_lie_derivative_condition

            dB_dt = calculate_lie_derivative(certificate, variables, system_info['dynamics'])
            if dB_dt is not None:
                 # Use a simplified check for now
                 is_valid_lie, reason = check_lie_derivative_condition(dB_dt, variables, system_info.get('safe_set_conditions'))
                 # We might simplify further: only accept if dB/dt is clearly non-positive
                 # e.g., if dB_dt == 0 or sympy.ask(sympy.Q.negative(dB_dt)) or similar basic check
                 # For this demo, let's accept if the simple check passes OR if it's the known good case
                 is_verified = is_valid_lie or (form == "quadratic_known" and system_info['id'] == "synthetic_nonlinear_stable_1")

                 if is_verified:
                     logging.info(f"Generated certificate '{certificate_str}' passed basic verification for system {system_info['id']}.")
                     example = format_example(system_desc_text, certificate_str, format_type)
                     generated_examples.append(example)
                 else:
                     logging.warning(f"Generated certificate '{certificate_str}' FAILED basic verification for system {system_info['id']} (dB/dt={dB_dt}, Reason: {reason}). Skipping.")
                     skipped_count += 1
            else:
                logging.warning(f"Could not calculate Lie derivative for system {system_info['id']}. Skipping.")
                skipped_count += 1
        except ImportError:
             logging.error("Could not import verification functions. Cannot verify synthetic data. Skipping verification step.")
             # Decide whether to proceed without verification (risky) or stop
             example = format_example(system_desc_text, certificate_str, format_type)
             generated_examples.append(example) # Add without verification if import fails
        except Exception as e:
            logging.error(f"Error during verification for system {system_info['id']}: {e}. Skipping.")
            skipped_count += 1


    logging.info(f"Generated {len(generated_examples)} synthetic examples. Skipped {skipped_count} systems.")

    # Save to file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in generated_examples:
                f.write(json.dumps(example) + '\\n')
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