import os
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
import sys

# Ensure other scripts in the project are importable
# Assuming this script is in paper_population/evaluation/
BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT) # Add project root to path

# Import necessary functions from other modules
try:
    # Import from sibling directory evaluation
    from .verify_certificate import verify_barrier_certificate
    # Import from sibling directory inference
    from inference.generate_certificate import (
        load_knowledge_base,
        load_finetuned_model,
        retrieve_context,
        format_prompt_with_context,
        KB_DIR as DEFAULT_KB_DIR, # Use defaults from generator
        VECTOR_STORE_FILENAME,
        METADATA_FILENAME,
        EMBEDDING_MODEL_NAME,
        BASE_MODEL_NAME as DEFAULT_BASE_MODEL,
        ADAPTER_PATH as DEFAULT_ADAPTER_PATH,
        NUM_CONTEXT_CHUNKS as DEFAULT_K,
        MAX_NEW_TOKENS, TEMPERATURE, TOP_P
    )
except ImportError as e:
    print(f"Error importing required modules: {e}", file=sys.stderr)
    print("Ensure verify_certificate.py is in the same directory (evaluation) and generate_certificate.py is in the inference directory.", file=sys.stderr)
    print("Also ensure __init__.py files exist in subdirectories.", file=sys.stderr)
    sys.exit(1)


# --- Configuration ---
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error() # Reduce transformers logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default paths relative to this script's directory (evaluation)
DEFAULT_BENCHMARK_PATH = os.path.join(BASE_DIR, "benchmark_systems.json")
DEFAULT_RESULTS_FILE = os.path.join(BASE_DIR, "evaluation_results.csv")

# --- Helper Functions ---
def extract_certificate_from_llm_output(llm_text):
    """Attempts to extract the barrier certificate B(x) string using regex."""
    # This is fragile and likely needs improvement based on actual LLM output formats.
    # Common patterns:
    # - B(x, y) = ...
    # - Barrier Certificate: ...
    # - B = ... (less specific)

    # Pattern 1: B(...) = expression_ending_at_newline_or_period_or_semicolon
    # Make it less greedy, look for common function patterns or standard variables
    match = re.search(r'B\s*\([^)]*\)\s*=\s*([^{};\n]+)', llm_text, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip().rstrip('.') # Remove trailing period
        logging.info(f"Extracted B(x) using pattern 1: {candidate}")
        return candidate

    # Pattern 2: Barrier Certificate: expression_ending_at_newline
    match = re.search(r'Barrier Certificate\s*[:=]\s*([^{};\n]+)', llm_text, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip().rstrip('.')
        logging.info(f"Extracted B(x) using pattern 2: {candidate}")
        return candidate

    # Pattern 3: Simple expression after keywords like "is given by", "is:", etc.
    match = re.search(r'(?:is|certificate is|given by)\s*[:=]?\s*([^{};\n]+)', llm_text, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip().rstrip('.')
        # Basic check if it looks like an equation/expression involving common vars
        if ('x' in candidate or 'y' in candidate or 'z' in candidate) and \
           any(op in candidate for op in ['+', '-', '*', '**']):
            logging.info(f"Extracted B(x) using pattern 3: {candidate}")
            return candidate


    # Pattern 4: Look for simple expression after mentioning conditions (less reliable)
    match = re.search(r'conditions it must satisfy\.?\s*([^{};\n]+)', llm_text, re.IGNORECASE | re.DOTALL)
    if match:
        candidate = match.group(1).strip().rstrip('.')
        # Basic check if it looks like an equation/expression
        if ('x' in candidate or 'y' in candidate or 'z' in candidate) and \
           any(op in candidate for op in ['+', '-', '*', '**']):
            logging.info(f"Extracted B(x) using pattern 4: {candidate}")
            return candidate

    logging.warning(f"Could not reliably extract B(x) expression from LLM output:\\n{llm_text}")
    return None


# --- Main Evaluation Logic ---
def evaluate_pipeline(benchmark_path, results_path, base_model_name, adapter_path, k, max_tokens, temp, kb_dir):
    logging.info("--- Starting Evaluation Pipeline ---")

    # 1. Load Benchmark Systems
    logging.info(f"Loading benchmark systems from: {benchmark_path}")
    try:
        # Benchmark path is now relative to script location by default
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            benchmark_systems = json.load(f)
        if not benchmark_systems:
            logging.error("Benchmark file is empty.")
            return # Exit function if empty
        logging.info(f"Loaded {len(benchmark_systems)} benchmark systems.")
    except FileNotFoundError:
        logging.error(f"Benchmark file not found at {benchmark_path}")
        return # Exit function
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding benchmark JSON file: {e}")
        return # Exit function
    except Exception as e:
        logging.error(f"Error loading benchmark systems: {e}")
        return # Exit function

    # 2. Load Models (Embedding and Fine-tuned LLM)
    logging.info(f"Loading Knowledge Base from {kb_dir} and Embedding Model...")
    faiss_index, metadata = load_knowledge_base(kb_dir, VECTOR_STORE_FILENAME, METADATA_FILENAME)
    if faiss_index is None or metadata is None:
        return # Exit function
    try:
        embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        logging.error(f"Error loading embedding model '{EMBEDDING_MODEL_NAME}': {e}")
        return # Exit function

    logging.info(f"Loading Fine-tuned LLM (Base: {base_model_name}, Adapter: {adapter_path})...")
    model, tokenizer = load_finetuned_model(base_model_name, adapter_path)
    if model is None or tokenizer is None:
        return # Exit function

    # Initialize generation pipeline
    pipe = pipeline(
        task="text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=max_tokens, temperature=temp, top_p=TOP_P, do_sample=True if temp > 0 else False
    )

    # 3. Run Evaluation Loop
    results_data = []
    total_systems = len(benchmark_systems)
    successful_generations = 0
    successful_parses = 0
    # Track detailed verification outcomes
    passed_numerical_checks = 0
    passed_symbolic_checks = 0
    final_verdict_passed = 0

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
        context = retrieve_context(system_input_text, embed_model, faiss_index, metadata, k)

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
            candidate_b_str = extract_certificate_from_llm_output(llm_output)
            if candidate_b_str:
                 successful_parses += 1
                 parsing_successful = True
                 logging.info(f"Parsed candidate: {candidate_b_str}")
            else:
                 logging.warning("Parsing failed.")

        # Verification Step
        logging.info("Verifying candidate certificate...")
        # Initialize detailed verification results
        verification_details = {
            "candidate_B": candidate_b_str,
            "system_id": system_id,
            "parsing_successful": parsing_successful,
            "lie_derivative_calculated": None,
            "symbolic_lie_check_passed": None,
            "symbolic_boundary_check_passed": None,
            "numerical_lie_check_passed": None,
            "numerical_boundary_check_passed": None,
            "final_verdict": "Parsing Failed" if not parsing_successful else "Verification Error",
            "reason": "Parsing failed or verification not run" if not parsing_successful else "Verification pending",
            "verification_time_seconds": 0
        }

        if parsing_successful:
            try:
                # Call the refactored verifier
                verification_details = verify_barrier_certificate(candidate_b_str, system_info)
                # Log the final verdict from the verifier
                logging.info(f"Verification verdict: {verification_details.get('final_verdict', 'Unknown')}")
                # Update counts based on final verdict
                if verification_details.get('final_verdict') == "Passed Numerical Checks":
                    passed_numerical_checks += 1
                    final_verdict_passed += 1 # Count numerical pass as overall pass for now
                elif verification_details.get('final_verdict') == "Passed Symbolic Checks (Basic)":
                    passed_symbolic_checks +=1
                    # Decide if symbolic pass counts as overall pass if numerical skipped/failed
                    # Let's be conservative: only numerical pass counts for the main metric
            except Exception as e:
                logging.error(f"Verification crashed for system {system_id} with B={candidate_b_str}: {e}")
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
            "lie_derivative_calculated": verification_details.get('lie_derivative_calculated', 'N/A'),
            "sym_lie_passed": verification_details.get('symbolic_lie_check_passed'),
            "sym_bound_passed": verification_details.get('symbolic_boundary_check_passed'),
            "num_lie_passed": verification_details.get('numerical_lie_check_passed'),
            "num_bound_passed": verification_details.get('numerical_boundary_check_passed'),
            "duration_seconds": duration,
            "llm_full_output": llm_output,
        })

        logging.info(f"System {i+1} processed in {duration:.2f} seconds.")

    # 4. Save and Report Metrics
    results_df = pd.DataFrame(results_data)
    try:
        # Save results relative to script location
        results_df.to_csv(results_path, index=False)
        logging.info(f"Evaluation results saved to {results_path}")
    except Exception as e:
        logging.error(f"Failed to save results CSV: {e}")

    print("\n--- Evaluation Summary ---")
    print(f"Total Systems Evaluated: {total_systems}")
    if total_systems > 0:
        gen_rate = (successful_generations / total_systems) * 100
        parse_rate = (successful_parses / total_systems) * 100
        # Report based on final verdict (prioritizing numerical checks)
        verify_rate = (final_verdict_passed / total_systems) * 100
        parse_and_verify_rate = (final_verdict_passed / successful_parses) * 100 if successful_parses > 0 else 0

        print(f"Successful Generations: {successful_generations} ({gen_rate:.2f}%)")
        print(f"Successfully Parsed Certificates: {successful_parses} ({parse_rate:.2f}%)")
        print(f"Passed Verification Checks (Numerical Priority): {final_verdict_passed} ({verify_rate:.2f}% overall)")
        print(f"Verification Success Rate (given successful parse): {parse_and_verify_rate:.2f}%")
        # Optionally report symbolic passes separately
        print(f"(Passed Basic Symbolic Checks: {passed_symbolic_checks})" )

    else:
        print("No systems were evaluated.")

    print("--- Evaluation Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG + Fine-tuned LLM for barrier certificate generation.")
    parser.add_argument("--benchmark", type=str, default=DEFAULT_BENCHMARK_PATH,
                        help=f"Path to the benchmark JSON file (default: {DEFAULT_BENCHMARK_PATH}).")
    parser.add_argument("--results", type=str, default=DEFAULT_RESULTS_FILE,
                        help=f"Path to save the evaluation results CSV file (default: {DEFAULT_RESULTS_FILE}).")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL,
                        help=f"Base model name used for fine-tuning (default: {DEFAULT_BASE_MODEL}).")
    parser.add_argument("--adapter", type=str, default=DEFAULT_ADAPTER_PATH,
                        help=f"Path to the fine-tuned LoRA adapter (default: {DEFAULT_ADAPTER_PATH}).")
    parser.add_argument("--kb_dir", type=str, default=DEFAULT_KB_DIR,
                         help=f"Path to the knowledge base directory (default: {DEFAULT_KB_DIR})")
    parser.add_argument("-k", type=int, default=DEFAULT_K,
                        help=f"Number of context chunks to retrieve (default: {DEFAULT_K}).")
    parser.add_argument("--max_tokens", type=int, default=MAX_NEW_TOKENS,
                        help=f"Maximum new tokens for generation (default: {MAX_NEW_TOKENS}).")
    parser.add_argument("--temp", type=float, default=TEMPERATURE,
                        help=f"Generation temperature (default: {TEMPERATURE}).")

    args = parser.parse_args()

    # Resolve potential relative paths from arguments based on PROJECT_ROOT if needed
    # Example: adapter_path = os.path.join(PROJECT_ROOT, args.adapter) if not os.path.isabs(args.adapter) else args.adapter
    # For simplicity, we assume paths provided via args are correct for now.

    evaluate_pipeline(
        benchmark_path=args.benchmark,
        results_path=args.results,
        base_model_name=args.base_model,
        adapter_path=args.adapter,
        k=args.k,
        max_tokens=args.max_tokens,
        temp=args.temp,
        kb_dir=args.kb_dir # Pass KB directory
    ) 