import os
import json
import logging
import argparse
from paper_population.utils.config_loader import load_config, DEFAULT_CONFIG_PATH # Import config loader

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Input from Phase 1 (Optional, for reference)
BASE_DIR = os.path.dirname(__file__) # Now fine_tuning directory
# Path to metadata relative to project root (paper_population)
# Assumes knowledge_base is a sibling directory to fine_tuning
KB_METADATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "knowledge_base", "knowledge_base_enhanced", "paper_metadata_enhanced.json"))

# Output file for fine-tuning data (within this directory)
FINETUNE_DATA_OUTPUT_FILE = os.path.join(BASE_DIR, "finetuning_data.jsonl")

# --- Helper Functions ---

def load_kb_metadata(metadata_path):
    """Loads the knowledge base metadata if available."""
    if not os.path.exists(metadata_path):
        logging.warning(f"Knowledge base metadata not found at {metadata_path}. Proceeding without it.")
        return None
    try:
        # Assuming JSONL format for metadata based on builder output
        metadata_map = {}
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    chunk_id = data.get('chunk_id')
                    if chunk_id is not None:
                        metadata_map[chunk_id] = data
                    else:
                         logging.warning(f"Skipping metadata line without 'chunk_id': {line.strip()}")

        logging.info(f"Loaded metadata for {len(metadata_map)} chunks from {metadata_path}.")
        return metadata_map
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode metadata JSONL line in {metadata_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Failed to load metadata from {metadata_path}: {e}")
        return None

def format_instruction_example(system_description, barrier_certificate):
    """Formats a single example into the instruction-following JSON structure."""
    instruction = ("Given the autonomous system described by the following dynamics, "
                   "propose a suitable barrier certificate function B(x) and, if possible, "
                   "briefly outline the conditions it must satisfy.")
    return {
        "instruction": instruction,
        "input": system_description, # The system dynamics, constraints, sets etc.
        "output": barrier_certificate, # The barrier function B(x), possibly with verification notes
        "metadata": { "source": "manual" } # Add source metadata
    }

def format_prompt_completion_example(system_description, barrier_certificate):
    """Formats a single example into a simpler prompt/completion structure."""
    prompt = f"System Dynamics:\n{system_description}\n\nBarrier Certificate:"
    completion = f" {barrier_certificate}"
    return {
        "prompt": prompt,
        "completion": completion,
        "metadata": { "source": "manual" } # Add source metadata
    }

# --- Main Logic ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manually create fine-tuning data for barrier certificate generation.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH,
                        help="Path to the configuration YAML file.")
    parser.add_argument("--format", type=str, default=None, choices=["instruction", "prompt_completion"],
                        help="Override the output format for the fine-tuning data (default: from config fine_tuning.data_format).")
    parser.add_argument("--append", action="store_true",
                        help="Append to the existing output file instead of overwriting.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Override path to save the fine-tuning data JSONL file (default: from config paths.ft_manual_data_file).")
    parser.add_argument("--kb_meta", type=str, default=None,
                         help="Override path to knowledge base metadata JSONL file (default: from config paths.kb_metadata_filename)")

    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Determine paths and format using config or overrides
    output_file_path = args.output_file if args.output_file else cfg.paths.ft_manual_data_file
    kb_metadata_path_to_use = args.kb_meta if args.kb_meta else cfg.paths.kb_metadata_filename
    data_format = args.format if args.format else cfg.fine_tuning.data_format

    # Load existing KB metadata for reference (optional)
    kb_metadata = load_kb_metadata(kb_metadata_path_to_use)
    if kb_metadata:
        print("Knowledge base metadata loaded. You can refer to chunk texts if needed.")
        # Example: Print a chunk to show how to access it
        # first_key = next(iter(kb_metadata), None)
        # if first_key is not None:
        #    print("\nExample Chunk (Index {first_key}):\n", kb_metadata.get(first_key, {}).get('text', 'Not Found'))

    # Determine file mode
    file_mode = 'a' if args.append else 'w'
    if file_mode == 'w' and os.path.exists(output_file_path):
        print(f"Output file {output_file_path} exists and will be overwritten.")
    elif file_mode == 'a' and not os.path.exists(output_file_path):
        print(f"Output file {output_file_path} does not exist. Creating new file.")
        file_mode = 'w' # Start fresh if appending to non-existent file

    print(f"\n--- Starting Data Entry (Format: {data_format}) ---")
    print(f"Saving to: {output_file_path}")
    print(f"Enter system description and barrier certificate pairs.")
    print("System description can be multi-line. Type 'END_DESC' on a new line to finish description.")
    print("Barrier certificate can be multi-line. Type 'END_CERT' on a new line to finish certificate.")
    print("Type 'SAVE_EXIT' for either input to save current entry and exit.")
    print("Type 'EXIT_NOW' for either input to exit immediately without saving current entry.")

    entry_count = 0
    try:
        # Ensure the directory for the output file exists
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, file_mode, encoding='utf-8') as f:
            while True:
                print(f"\n--- Entry {entry_count + 1} ---")
                print("Enter System Description (end with 'END_DESC'):")
                system_desc_lines = []
                while True:
                    line = input()
                    if line.strip().upper() == 'END_DESC':
                        break
                    if line.strip().upper() == 'SAVE_EXIT':
                        system_desc_lines = ["SAVE_EXIT"]
                        break
                    if line.strip().upper() == 'EXIT_NOW':
                        system_desc_lines = ["EXIT_NOW"]
                        break
                    system_desc_lines.append(line)
                system_desc = "\n".join(system_desc_lines).strip()

                if system_desc == 'SAVE_EXIT' or system_desc == 'EXIT_NOW':
                    print("Exiting data entry...")
                    break

                if not system_desc:
                    print("System description cannot be empty. Please try again or exit.")
                    continue

                print("\nEnter Barrier Certificate (end with 'END_CERT'):")
                barrier_cert_lines = []
                while True:
                    line = input()
                    if line.strip().upper() == 'END_CERT':
                        break
                    if line.strip().upper() == 'SAVE_EXIT':
                        barrier_cert_lines = ["SAVE_EXIT"]
                        break
                    if line.strip().upper() == 'EXIT_NOW':
                        barrier_cert_lines = ["EXIT_NOW"]
                        break
                    barrier_cert_lines.append(line)
                barrier_cert = "\n".join(barrier_cert_lines).strip()

                if barrier_cert == 'SAVE_EXIT' or barrier_cert == 'EXIT_NOW':
                     # Decide if we should save the entry even if cert is SAVE_EXIT
                     # For now, we just exit.
                     print("Exiting data entry...")
                     break

                if not barrier_cert:
                    print("Barrier certificate cannot be empty. Please try again or exit.")
                    continue

                # Format the example based on selected format
                if data_format == "instruction":
                    example = format_instruction_example(system_desc, barrier_cert)
                else: # prompt_completion
                    example = format_prompt_completion_example(system_desc, barrier_cert)

                # Write as JSON Line
                f.write(json.dumps(example) + '\n')
                entry_count += 1
                print(f"Entry {entry_count} saved.")

    except Exception as e:
        logging.error(f"An error occurred during data entry or saving: {e}")

    logging.info(f"Finished data entry. Saved {entry_count} new entries to {output_file_path}.")

    # Suggestion for next steps
    if entry_count > 0:
        print("\nNext steps:")
        print(f"1. Review the generated file: {output_file_path}")
        print("2. Add more high-quality examples.")
        # Reference the combined data file path from config for the next step
        print(f"3. Combine datasets (e.g., using combine_datasets.py to create {cfg.paths.ft_combined_data_file}).")
        print(f"4. Use the combined file ({cfg.paths.ft_combined_data_file}) as input for the fine-tuning script ('finetune_llm.py').")
    elif not args.append:
        print("\nNo entries were saved. The output file might be empty or unchanged.") 