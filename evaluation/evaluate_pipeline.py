import os
import sys
import json
import logging
import time
import warnings

# Add project root to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

# Third-party imports
import faiss
import numpy as np
import torch
from transformers import pipeline, logging as hf_logging
from sentence_transformers import SentenceTransformer
from omegaconf import OmegaConf

# Local imports
from utils.config_loader import load_config, DEFAULT_CONFIG_PATH
from knowledge_base.kb_utils import get_active_kb_paths, determine_kb_type_from_config, validate_kb_config

# Import shared certificate extraction utilities
from utils.certificate_extraction import (
    extract_certificate_from_llm_output,
    clean_and_validate_expression
)

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
# Certificate extraction functions moved to utils.certificate_extraction


# --- Main Evaluation Logic ---
def evaluate_pipeline(cfg: OmegaConf):
    """Main evaluation function, accepts OmegaConf object."""
    logging.info("--- Starting Evaluation Pipeline ---")
    
    # Validate configuration
    if not validate_kb_config(cfg):
        logging.error("Invalid knowledge base configuration. Please check your config file.")
        return
    
    # Determine barrier certificate type and paths
    kb_type = determine_kb_type_from_config(cfg)
    kb_output_dir, kb_vector_path, kb_metadata_path = get_active_kb_paths(cfg)
    
    logging.info(f"Evaluating {kb_type} barrier certificate pipeline")
    
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
    logging.info(f"KB Path: {kb_output_dir}")

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
    # Use the determined KB paths
    kb_dir = kb_output_dir
    vector_store_filename = os.path.basename(kb_vector_path)
    metadata_filename = os.path.basename(kb_metadata_path)
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
        prompt = format_prompt_with_context(system_input_text, context, kb_type)

        # Generation and Parsing with Retries
        max_retries = eval_cfg.get('generation_max_retries', 10) # Increase default to 10 instead of 3
        generation_retry_delay_sec = eval_cfg.get('generation_retry_delay_sec', 1)

        candidate_b_str = None # Final candidate string to be verified
        llm_output_for_csv = "[No successful generation or parsing]" 
        
        generation_attempt_successful = False 
        parsing_of_llm_output_successful = False # True if LLM provided a usable, non-fallback certificate

        var_names_for_extraction = system_info.get('state_variables', []) 

        # Keep trying until we get a successful parse or reach max_retries
        retry_count = 0
        while not parsing_of_llm_output_successful and retry_count < max_retries:
            retry_count += 1
            logging.info(f"System {system_id}: Generating certificate candidate (Attempt {retry_count}/{max_retries})...")
            
            current_llm_raw_output = "[Generation Failed this Attempt]"
            current_generation_succeeded = False
            try:
                gen_result = pipe(prompt)
                generated_text = gen_result[0]['generated_text']
                prompt_end_marker = "[/INST]" 
                output_start_index = generated_text.find(prompt_end_marker)
                if output_start_index != -1:
                    current_llm_raw_output = generated_text[output_start_index + len(prompt_end_marker):].strip()
                else:
                    current_llm_raw_output = generated_text
                
                llm_output_for_csv = current_llm_raw_output 
                current_generation_succeeded = True
                if not generation_attempt_successful: 
                    generation_attempt_successful = True 
                logging.info(f"System {system_id}: Generation attempt {retry_count} successful.")
                logging.debug(f"LLM Raw Output (System {system_id}, Attempt {retry_count}):\\n{current_llm_raw_output}")

            except Exception as e:
                logging.error(f"System {system_id}: Text generation failed (Attempt {retry_count}): {e}")
                llm_output_for_csv = f"[Generation Failed on attempt {retry_count}]"
                if retry_count == max_retries: 
                    logging.warning(f"System {system_id}: All generation attempts failed.")
                time.sleep(generation_retry_delay_sec) # Wait before next attempt or finishing
                continue 

            # If generation was successful for this attempt, try to parse
            logging.info(f"System {system_id}: Parsing generated certificate (Attempt {retry_count})...")
            # extract_certificate_from_llm_output now returns (parsed_string_or_None, True_if_failed)
            parsed_expr_str, parsing_failed = extract_certificate_from_llm_output(current_llm_raw_output, var_names_for_extraction)
            
            if not parsing_failed and parsed_expr_str is not None:
                candidate_b_str = parsed_expr_str
                parsing_of_llm_output_successful = True 
                logging.info(f"System {system_id}: Successfully parsed LLM-specific certificate (Attempt {retry_count}): {candidate_b_str}")
                break # Exit retry loop, we have a good LLM-specific candidate
            else:
                candidate_b_str = None # Ensure it's None if parsing failed
                logging.warning(f"System {system_id}: Parsing LLM output failed to yield usable certificate (Attempt {retry_count}). LLM Output: {current_llm_raw_output[:100]}...")
                if retry_count < max_retries:
                    logging.info(f"System {system_id}: Retrying generation and parsing...")
                    time.sleep(generation_retry_delay_sec) 
                else:
                    logging.warning(f"System {system_id}: Max retries reached. No usable certificate parsed from LLM.")

        # Update overall successful_generations counter if any attempt within the loop was successful
        if generation_attempt_successful:
            successful_generations +=1 
        
        # successful_parses counter is incremented only if parsing_of_llm_output_successful becomes True
        if parsing_of_llm_output_successful: 
            successful_parses += 1

        # Verification Step
        # candidate_b_str will be None if all parsing attempts failed, or the successfully parsed string.
        logging.info(f"System {system_id}: Verifying candidate certificate: {candidate_b_str if candidate_b_str else '[No usable certificate from LLM]'}...")
        verification_details = {
            "candidate_B": candidate_b_str,
            "system_id": system_id,
            "parsing_successful": parsing_of_llm_output_successful, # This now means LLM output was successfully parsed (not a fallback)
            "lie_derivative_calculated": None,
            "symbolic_lie_check_passed": None,
            "symbolic_boundary_check_passed": None,
            "numerical_lie_check_passed": None,
            "numerical_boundary_check_passed": None,
            "sos_attempted": False,
            "sos_passed": None,
            "sos_lie_passed": None,
            "sos_init_passed": None,
            "sos_unsafe_passed": None,
            "final_verdict": "Parsing Failed" if not parsing_of_llm_output_successful else "Verification Pending",
            "reason": "LLM output parsing failed or yielded no usable certificate" if not parsing_of_llm_output_successful else "Verification pending",
            "verification_time_seconds": 0
        }

        if candidate_b_str: # Only proceed to verification if we have a candidate string
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
            "generation_successful": generation_attempt_successful,
            "parsing_successful": parsing_of_llm_output_successful,
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
            "llm_full_output": llm_output_for_csv,
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