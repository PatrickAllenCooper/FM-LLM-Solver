import os
import json
import logging
import argparse

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Input from Phase 1 (Optional, for reference)
BASE_DIR = os.path.dirname(__file__)
KB_METADATA_PATH = os.path.join(BASE_DIR, "knowledge_base_enhanced", "paper_metadata_enhanced.json")

# Output file for fine-tuning data
FINETUNE_DATA_OUTPUT_FILE = os.path.join(BASE_DIR, "finetuning_data.jsonl")

# --- Helper Functions ---

def load_kb_metadata(metadata_path):
    """Loads the knowledge base metadata if available."""
    if not os.path.exists(metadata_path):
        logging.warning(f"Knowledge base metadata not found at {metadata_path}. Proceeding without it.")
        return None
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_map = {int(k): v for k, v in json.load(f).items()}
        logging.info(f"Loaded metadata for {len(metadata_map)} chunks from {metadata_path}.")
        return metadata_map
    except Exception as e:
        logging.error(f"Failed to load metadata from {metadata_path}: {e}")
        return None

def format_instruction_example(system_description, barrier_certificate):
    """Formats a single example into the instruction-following JSON structure."""
    # You might refine this prompt structure based on the base model you choose
    instruction = ("Given the autonomous system described by the following dynamics, "
                   "propose a suitable barrier certificate function B(x) and, if possible, "
                   "briefly outline the conditions it must satisfy.")
    return {
        "instruction": instruction,
        "input": system_description, # The system dynamics, constraints, sets etc.
        "output": barrier_certificate # The barrier function B(x), possibly with verification notes
    }

def format_prompt_completion_example(system_description, barrier_certificate):
    """Formats a single example into a simpler prompt/completion structure."""
    # This format might be simpler for some models or tasks
    prompt = f"System Dynamics:\n{system_description}\n\nBarrier Certificate:" # Note: No space after colon
    completion = f" {barrier_certificate}" # Note: Leading space is important for some models
    return {
        "prompt": prompt,
        "completion": completion
    }

# --- Main Logic ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manually create fine-tuning data for barrier certificate generation.")
    parser.add_argument("--format", type=str, default="instruction", choices=["instruction", "prompt_completion"],
                        help="Output format for the fine-tuning data (default: instruction).")
    parser.add_argument("--append", action="store_true",
                        help="Append to the existing output file instead of overwriting.")
    args = parser.parse_args()

    # Load existing KB metadata for reference (optional)
    kb_metadata = load_kb_metadata(KB_METADATA_PATH)
    if kb_metadata:
        print("Knowledge base metadata loaded. You can refer to chunk texts if needed.")
        # Example: Print a chunk to show how to access it
        # print("\nExample Chunk (Index 0):\n", kb_metadata.get(0, {}).get('text', 'Not Found'))

    # Determine file mode
    file_mode = 'a' if args.append else 'w'
    if file_mode == 'w' and os.path.exists(FINETUNE_DATA_OUTPUT_FILE):
        print(f"Output file {FINETUNE_DATA_OUTPUT_FILE} exists and will be overwritten.")
    elif file_mode == 'a' and not os.path.exists(FINETUNE_DATA_OUTPUT_FILE):
        print(f"Output file {FINETUNE_DATA_OUTPUT_FILE} does not exist. Creating new file.")
        file_mode = 'w' # Start fresh if appending to non-existent file

    print(f"\n--- Starting Data Entry (Format: {args.format}) ---")
    print(f"Enter system description and barrier certificate pairs.")
    print("System description can be multi-line. Type 'END_DESC' on a new line to finish description.")
    print("Barrier certificate can be multi-line. Type 'END_CERT' on a new line to finish certificate.")
    print("Type 'SAVE_EXIT' for either input to save current entry and exit.")
    print("Type 'EXIT_NOW' for either input to exit immediately without saving current entry.")

    entry_count = 0
    try:
        with open(FINETUNE_DATA_OUTPUT_FILE, file_mode, encoding='utf-8') as f:
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
                if args.format == "instruction":
                    example = format_instruction_example(system_desc, barrier_cert)
                else: # prompt_completion
                    example = format_prompt_completion_example(system_desc, barrier_cert)

                # Write as JSON Line
                f.write(json.dumps(example) + '\n')
                entry_count += 1
                print(f"Entry {entry_count} saved.")

    except Exception as e:
        logging.error(f"An error occurred during data entry or saving: {e}")

    logging.info(f"Finished data entry. Saved {entry_count} new entries to {FINETUNE_DATA_OUTPUT_FILE}.")

    # Suggestion for next steps
    if entry_count > 0:
        print("\nNext steps:")
        print(f"1. Review the generated file: {FINETUNE_DATA_OUTPUT_FILE}")
        print("2. Add more high-quality examples.")
        print("3. Use this file as input for the fine-tuning script ('finetune_llm.py').")
    elif not args.append:
        print("\nNo entries were saved. The output file might be empty or unchanged.") 