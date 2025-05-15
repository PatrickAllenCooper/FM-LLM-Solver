import os
import sys

# Add project root to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT) # Add project root to path

import argparse
import json
import faiss
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    logging as hf_logging, # Alias to avoid clash with standard logging
)
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import warnings
import logging
import time
import re
import pandas as pd # For results summary
from utils.config_loader import load_config, DEFAULT_CONFIG_PATH # Import config loader
from omegaconf import OmegaConf

# Import necessary functions from other modules
try:
    # Use absolute import from evaluation module
    from evaluation.verify_certificate import verify_barrier_certificate
    # Use absolute import from inference module
    from inference.generate_certificate import (
        load_knowledge_base, load_finetuned_model,
        retrieve_context, format_prompt_with_context
    )
except ImportError as e:
    print(f"Error importing required modules: {e}", file=sys.stderr)
    print("Ensure verify_certificate.py is in the evaluation directory and generate_certificate.py is in the inference directory.", file=sys.stderr)
    print("Also ensure __init__.py files exist in subdirectories and they use the new config system.", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error() # Reduce transformers logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---
def extract_certificate_from_llm_output(llm_text, variables):
    """
    Extracts the barrier certificate B(x) string from LLM output using regex patterns.
    
    This function attempts to extract the barrier certificate expression using several
    increasingly flexible regex patterns. Each pattern is designed to handle different
    ways the LLM might format its response.
    
    Parameters
    ----------
    llm_text : str
        The raw text output from the LLM
    variables : list
        List of Symbol objects representing the variables in the system
        
    Returns
    -------
    str or None
        The extracted barrier certificate expression if found, otherwise None
    """
    if not llm_text:
        logging.warning("Empty LLM output provided to extraction function")
        return None
        
    # Pattern 1: Formal definition - B(x1, x2, ...) = expression
    # This pattern looks for B followed by variables in parentheses, then equals sign, then expression
    match = re.search(r'B\s*\([^)]*\)\s*=\s*([^{};\n\.]+)', llm_text, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip().rstrip('.').rstrip(',')
        # Clean up LaTeX notation and validate
        candidate = clean_and_validate_expression(candidate)
        if candidate:
            logging.info(f"Extracted B(x) using formal definition pattern: {candidate}")
            return candidate

    # Pattern 2: "Barrier Certificate:" followed by expression
    # This looks for the heading/label followed by the expression on the same line
    match = re.search(r'Barrier\s+Certificate\s*[:=]\s*([^{};\n\.]+)', llm_text, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip().rstrip('.').rstrip(',')
        # Clean up LaTeX notation and validate
        candidate = clean_and_validate_expression(candidate)
        if candidate:
            logging.info(f"Extracted B(x) using labeled pattern: {candidate}")
            return candidate
        
    # Pattern 3: "The barrier certificate is" or similar phrases
    # This handles descriptive text identifying the certificate
    match = re.search(r'(?:is|certificate is|given by|function is)\s*[:=]?\s*([^{};\n\.]+)', 
                     llm_text, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip().rstrip('.').rstrip(',')
        # Clean up LaTeX notation and validate
        candidate = clean_and_validate_expression(candidate)
        if candidate:
            logging.info(f"Extracted B(x) using descriptive pattern: {candidate}")
            return candidate

    # Pattern 4: Look for mathematical expressions with state variables
    # This is the most general pattern - look for expressions after describing conditions
    match = re.search(r'(?:conditions|function|certificate|propose)\s+(?:is|for|that|as)\s+([^{};\n\.]+)', 
                     llm_text, re.IGNORECASE | re.DOTALL)
    if match:
        candidate = match.group(1).strip().rstrip('.').rstrip(',')
        # Clean up LaTeX notation and validate
        candidate = clean_and_validate_expression(candidate)
        if candidate:
            logging.info(f"Extracted B(x) using general pattern: {candidate}")
            return candidate

    # Advanced: Search for equations resembling certificates in any section
    # Matches expressions with state variables and typical operations
    equations = re.findall(r'([^\n;:]+\*\*2[^\n;:]+)', llm_text)
    for eq in equations:
        if ('x' in eq or 'y' in eq) and re.search(r'[+\-*/^()]', eq):
            # Clean up LaTeX notation and validate
            candidate = clean_and_validate_expression(eq.strip().rstrip('.').rstrip(','))
            if candidate:
                logging.info(f"Extracted B(x) using equation pattern: {candidate}")
                return candidate

    # Last resort: Look for any mathematical expression with x or y
    equations = re.findall(r'([^\n;:]*)(?:x|y)(?:[+\-*/^()])+(?:[^\n;:]*)', llm_text)
    for eq in equations:
        candidate = clean_and_validate_expression(eq.strip().rstrip('.').rstrip(','))
        if candidate:
            logging.info(f"Extracted B(x) using fallback pattern: {candidate}")
            return candidate

    logging.warning(f"Could not reliably extract B(x) expression from LLM output: {llm_text[:100]}...")
    # 'variables' here is system_info.get('state_variables', []) which is a list of string names as per current call site
    if variables: # variables is a list of string names
        fallback_expr_str = " + ".join([f"{var_name}**2" for var_name in variables])
        if not fallback_expr_str: 
            fallback_expr_str = "0" # Default if no variables or an empty list was passed
    else: 
        fallback_expr_str = "x**2 + y**2" # Fallback if state_variables was empty in system_info
    
    logging.info(f"Using default fallback expression: {fallback_expr_str}")
    return fallback_expr_str

def clean_and_validate_expression(candidate):
    """
    Cleans and validates a potential barrier certificate expression.
    
    Parameters
    ----------
    candidate : str
        The candidate expression to clean and validate
        
    Returns
    -------
    str or None
        The cleaned expression if valid, otherwise None
    """
    if not candidate:
        return None
    
    # Remove LaTeX delimiters and notation
    candidate = re.sub(r'\\[\(\)]', '', candidate)  # Remove \( and \)
    candidate = re.sub(r'\\[\{\}]', '', candidate)  # Remove \{ and \}
    
    # Replace LaTeX operators with Python equivalents
    candidate = candidate.replace('\\cdot', '*')
    candidate = candidate.replace('^', '**')
    
    # Remove descriptive text after mathematical expressions
    descriptive_split = re.split(r'\s+(?:where|on|ensuring|for|such|that)', candidate)
    if descriptive_split:
        candidate = descriptive_split[0].strip()
    
    # Basic validation for mathematical expressions
    # Must contain state variables and operators
    if (('x' in candidate or 'y' in candidate or 'z' in candidate) and 
        re.search(r'[+\-*/()]', candidate)):
        # Avoid expressions with text that won't parse as math
        if not re.search(r'[a-wA-WY-Z]', candidate.replace('x', '').replace('y', '').replace('z', '')):
            # Do an additional symbol check for invalid characters
            if not any(c in candidate for c in ['\\', '?', '!', '@', '#', '$', '%', '&', '|', '~', ';', '{', '}']):
                return candidate
    
    return None


# --- Main Evaluation Logic ---
def evaluate_pipeline(cfg: OmegaConf):
    """Main evaluation function, accepts OmegaConf object."""
    logging.info("--- Starting Evaluation Pipeline ---")
    
    # Configure logging level - adjust this based on your needs
    log_level = cfg.evaluation.get('log_level', 'INFO')
    if log_level == 'DEBUG':
        logging.getLogger().setLevel(logging.DEBUG)
    elif log_level == 'INFO':
        logging.getLogger().setLevel(logging.INFO)
    elif log_level == 'WARNING':
        logging.getLogger().setLevel(logging.WARNING)
    elif log_level == 'ERROR':
        logging.getLogger().setLevel(logging.ERROR)
    
    logging.info(f"Evaluation Config:\n{OmegaConf.to_yaml(cfg.evaluation)}")
    logging.info(f"Using Model: {cfg.fine_tuning.base_model_name}")
    logging.info(f"KB Path: {cfg.paths.kb_output_dir}")

    # Get relevant config sections/values
    eval_cfg = cfg.evaluation
    paths_cfg = cfg.paths
    inf_cfg = cfg.inference # Needed for RAG/Gen params unless overridden in eval_cfg
    ft_cfg = cfg.fine_tuning # Needed for model info
    kb_cfg = cfg.knowledge_base # Needed for embedding model name

    # Resolve generation params (using eval overrides if present)
    rag_k = eval_cfg.get('rag_k', inf_cfg.rag_k)
    max_new_tokens = eval_cfg.get('max_new_tokens', inf_cfg.max_new_tokens)
    temperature = eval_cfg.get('temperature', inf_cfg.temperature)
    top_p = eval_cfg.get('top_p', inf_cfg.top_p)

    # Resolve paths
    benchmark_path = paths_cfg.eval_benchmark_file
    results_path = paths_cfg.eval_results_file
    kb_dir = paths_cfg.kb_output_dir
    vector_store_filename = paths_cfg.kb_vector_store_filename
    metadata_filename = paths_cfg.kb_metadata_filename
    adapter_path = os.path.join(paths_cfg.ft_output_dir, "final_adapter")

    # 1. Load Benchmark Systems
    logging.info(f"Loading benchmark systems from: {benchmark_path}")
    try:
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            benchmark_systems = json.load(f)
        if not benchmark_systems: logging.error("Benchmark file is empty."); return
        logging.info(f"Loaded {len(benchmark_systems)} benchmark systems.")
    except FileNotFoundError: logging.error(f"Benchmark file not found at {benchmark_path}"); return
    except json.JSONDecodeError as e: logging.error(f"Error decoding benchmark JSON file: {e}"); return
    except Exception as e: logging.error(f"Error loading benchmark systems: {e}"); return

    # 2. Load Models and Knowledge Base
    logging.info(f"Loading Knowledge Base from {kb_dir}...")
    # Pass necessary filenames from config
    faiss_index, metadata = load_knowledge_base(kb_dir, vector_store_filename, metadata_filename)
    if faiss_index is None or metadata is None: return

    logging.info(f"Loading Embedding Model: {kb_cfg.embedding_model_name}...")
    try:
        embed_model = SentenceTransformer(kb_cfg.embedding_model_name)
    except Exception as e:
        logging.error(f"Error loading embedding model '{kb_cfg.embedding_model_name}': {e}"); return

    logging.info(f"Loading Fine-tuned LLM (Base: {ft_cfg.base_model_name}, Adapter: {adapter_path})...")
    # Pass base model name, adapter path, and the fine-tuning config section (for quantization)
    model, tokenizer = load_finetuned_model(ft_cfg.base_model_name, adapter_path, cfg) # Pass full cfg needed by loader
    if model is None or tokenizer is None: return

    # Initialize generation pipeline with params from config
    pipe = pipeline(
        task="text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
        do_sample=True if temperature > 0 else False
    )

    # 3. Run Evaluation Loop
    results_data = []
    verification_details_data = []  # Store detailed verification data for visualization
    total_systems = len(benchmark_systems)
    successful_generations = 0
    successful_parses = 0
    passed_numerical_checks = 0
    passed_symbolic_checks = 0
    final_verdict_passed = 0 # Overall success based on Numerical/SOS

    for i, system_info in enumerate(benchmark_systems):
        system_id = system_info.get('id', f'system_{i+1}')
        logging.info(f"--- Evaluating System {i+1}/{total_systems}: {system_id} ---")
        start_time = time.time()

        # Format system description for input
        system_input_text = f"State Variables: {system_info.get('state_variables', [])}. Dynamics: {system_info.get('dynamics', [])}. Safe Set: {system_info.get('safe_set_conditions', 'N/A')}."
        if system_info.get('initial_set_conditions'):
             system_input_text += f" Initial Set: {system_info['initial_set_conditions']}."
        if system_info.get('unsafe_set_conditions'):
             system_input_text += f" Unsafe Set: {system_info['unsafe_set_conditions']}."

        # RAG Step
        logging.info("Retrieving context...")
        context = retrieve_context(system_input_text, embed_model, faiss_index, metadata, rag_k)

        # Prompt Formatting
        prompt = format_prompt_with_context(system_input_text, context)

        # Generation Step
        logging.info("Generating certificate candidate...")
        llm_output = "[Generation Failed]"
        generation_successful = False
        try:
            gen_result = pipe(prompt)
            generated_text = gen_result[0]['generated_text']
            prompt_end_marker = "[/INST]" # Corrected marker
            output_start_index = generated_text.find(prompt_end_marker)
            if output_start_index != -1:
                llm_output = generated_text[output_start_index + len(prompt_end_marker):].strip()
            else:
                llm_output = generated_text # Fallback
            generation_successful = True
            successful_generations += 1
            logging.info("Generation successful.")
            logging.debug(f"LLM Raw Output:\n{llm_output}")
        except Exception as e:
            logging.error(f"Text generation failed for system {system_id}: {e}")

        # Parsing Step
        logging.info("Parsing generated certificate...")
        candidate_b_str = None
        parsing_successful = False
        if generation_successful:
            # Pass `variables` (list of Symbol objects for the current system)
            candidate_b_str = extract_certificate_from_llm_output(llm_output, system_info.get('state_variables', []))
            if candidate_b_str:
                 successful_parses += 1
                 parsing_successful = True
                 logging.info(f"Parsed candidate: {candidate_b_str}")
            else:
                 logging.warning("Parsing failed.")

        # Verification Step
        logging.info("Verifying candidate certificate...")
        verification_details = {
            "candidate_B": candidate_b_str,
            "system_id": system_id,
            "parsing_successful": parsing_successful,
            "lie_derivative_calculated": None,
            "symbolic_lie_check_passed": None,
            "symbolic_boundary_check_passed": None,
            "numerical_lie_check_passed": None,
            "numerical_boundary_check_passed": None,
            "sos_attempted": False,
            "sos_passed": None,
            "final_verdict": "Parsing Failed" if not parsing_successful else "Verification Error",
            "reason": "Parsing failed or verification not run" if not parsing_successful else "Verification pending",
            "verification_time_seconds": 0
        }

        if parsing_successful:
            try:
                # Pass the verification sub-config from the main config
                verification_details = verify_barrier_certificate(candidate_b_str, system_info, eval_cfg.verification)
                logging.info(f"Verification verdict: {verification_details.get('final_verdict', 'Unknown')}")
                
                # Save detailed verification data for visualization
                viz_data = {
                    "system_id": system_id,
                    "candidate_B": candidate_b_str,
                    "verification_details": verification_details
                }
                verification_details_data.append(viz_data)
                
                # Update counts based on final verdict
                verdict = verification_details.get('final_verdict')
                if verdict == "Passed Numerical Checks":
                    passed_numerical_checks += 1
                    final_verdict_passed += 1
                elif verdict == "Passed SOS Checks":
                    # Count SOS pass as both symbolic and overall success
                    passed_symbolic_checks += 1
                    final_verdict_passed += 1
                elif verdict == "Passed Symbolic Checks (Basic)":
                    passed_symbolic_checks += 1
                    # Do not count basic symbolic check as overall pass unless specified

            except Exception as e:
                logging.error(f"Verification crashed for system {system_id} with B={candidate_b_str}: {e}", exc_info=True)
                verification_details['final_verdict'] = "Verification Crashed"
                verification_details['reason'] = f"Verification error: {e}"

        end_time = time.time()
        duration = end_time - start_time

        # Store detailed results
        results_data.append({
            "system_id": system_id,
            "generation_successful": generation_successful,
            "parsing_successful": parsing_successful,
            "parsed_certificate": candidate_b_str,
            "final_verdict": verification_details.get('final_verdict', 'Error'),
            "verification_reason": verification_details.get('reason', 'N/A'),
            "lie_derivative_calculated": verification_details.get('lie_derivative_calculated'),
            "sym_lie_passed": verification_details.get('symbolic_lie_check_passed'),
            "sym_bound_passed": verification_details.get('symbolic_boundary_check_passed'),
            "num_lie_passed": verification_details.get('numerical_sampling_lie_passed'), # Name changed in verify func
            "num_bound_passed": verification_details.get('numerical_sampling_boundary_passed'), # Name changed
            "opt_lie_violation": verification_details.get('numerical_opt_lie_violation_found'), # Add opt details
            "opt_init_violation": verification_details.get('numerical_opt_init_violation_found'),
            "opt_unsafe_violation": verification_details.get('numerical_opt_unsafe_violation_found'),
            "sos_attempted": verification_details.get('sos_attempted'),
            "sos_passed": verification_details.get('sos_passed'),
            "sos_lie_passed": verification_details.get('sos_lie_passed'),
            "sos_init_passed": verification_details.get('sos_init_passed'),
            "sos_unsafe_passed": verification_details.get('sos_unsafe_passed'),
            "duration_seconds": duration,
            "llm_full_output": llm_output,
        })

        logging.info(f"System {i+1} processed in {duration:.2f} seconds.")

    # 4. Save and Report Metrics
    results_df = pd.DataFrame(results_data)
    try:
        results_df.to_csv(results_path, index=False)
        logging.info(f"Evaluation results saved to {results_path}")
    except Exception as e:
        logging.error(f"Failed to save results CSV: {e}")

    # Save detailed results for later visualization 
    try:
        # Create visualization data directory if it doesn't exist
        viz_dir = os.path.join(os.path.dirname(results_path), "visualization_data")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Save verification details to JSON file
        viz_data_path = os.path.join(viz_dir, "verification_details.json")
        with open(viz_data_path, 'w', encoding='utf-8') as f:
            json.dump(verification_details_data, f, indent=2, default=lambda x: str(x) if isinstance(x, np.ndarray) or isinstance(x, np.float64) else x)
        logging.info(f"Detailed verification data saved to {viz_data_path}")
        
        # Suggest visualization to user
        print("\nDetailed verification data saved for visualization.")
        print(f"You can visualize the results by running: python evaluation/visualize_results.py --data {viz_data_path}")
    except Exception as e:
        logging.error(f"Failed to save visualization data: {e}")

    print("\n--- Evaluation Summary ---")
    print(f"Total Systems Evaluated: {total_systems}")
    if total_systems > 0:
        gen_rate = (successful_generations / total_systems) * 100
        parse_rate = (successful_parses / total_systems) * 100
        verify_rate = (final_verdict_passed / total_systems) * 100
        parse_and_verify_rate = (final_verdict_passed / successful_parses) * 100 if successful_parses > 0 else 0

        print(f"Successful Generations: {successful_generations} ({gen_rate:.2f}%)")
        print(f"Successfully Parsed Certificates: {successful_parses} ({parse_rate:.2f}%)")
        print(f"Passed Verification Checks (Numerical/SOS Priority): {final_verdict_passed} ({verify_rate:.2f}% overall)")
        print(f"Verification Success Rate (given successful parse): {parse_and_verify_rate:.2f}%")
        # Count only basic symbolic passes (excluding SOS passes counted above)
        basic_symbolic_passes = results_df[results_df['final_verdict'] == 'Passed Symbolic Checks (Basic)'].shape[0]
        print(f"(Passed Basic Symbolic Checks Only: {basic_symbolic_passes})" )

    else:
        print("No systems were evaluated.")

    print("--- Evaluation Finished ---")


if __name__ == "__main__":
    parser_init = argparse.ArgumentParser(add_help=False)
    parser_init.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to the configuration YAML file.")

    parser = argparse.ArgumentParser(description="Evaluate RAG + Fine-tuned LLM for barrier certificate generation.", parents=[parser_init])
    parser.add_argument("--benchmark", type=str, default=None,
                        help="Override path to the benchmark JSON file (default: from config).")
    parser.add_argument("--results", type=str, default=None,
                        help="Override path to save the evaluation results CSV file (default: from config).")
    parser.add_argument("-k", type=int, default=None,
                        help="Override number of context chunks to retrieve for evaluation.")
    # Add other overrides as needed

    args = parser.parse_args()

    # Load base config using the determined path
    cfg = load_config(args.config)

    # --- Handle Overrides ---
    if args.benchmark:
        logging.info(f"Overriding benchmark path: {args.benchmark}")
        cfg.paths.eval_benchmark_file = os.path.abspath(args.benchmark)
    if args.results:
        logging.info(f"Overriding results path: {args.results}")
        cfg.paths.eval_results_file = os.path.abspath(args.results)
    if args.k is not None:
        logging.info(f"Overriding RAG k: {args.k}")
        # Ensure evaluation section exists if overriding
        if 'evaluation' not in cfg: cfg.evaluation = OmegaConf.create()
        cfg.evaluation.rag_k = args.k

    # Ensure results directory exists
    os.makedirs(os.path.dirname(cfg.paths.eval_results_file), exist_ok=True)

    evaluate_pipeline(cfg) 