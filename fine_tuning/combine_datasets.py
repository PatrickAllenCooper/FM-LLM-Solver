import os
import json
import logging
import argparse
import glob
import re # Import re
import sys

# Add project root to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

# Now we can import the utils module
from utils.config_loader import load_config, DEFAULT_CONFIG_PATH # Import config loader

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Logic ---

def combine_datasets(input_patterns, output_file, required_source=None):
    """Combines data from multiple JSONL files, optionally filtering by source."""
    combined_data = []
    total_loaded = 0
    total_added = 0
    processed_files = set()

    # Expand glob patterns to get list of files
    input_files = []
    # Note: input_patterns are now expected to be potentially relative to project root
    # or already resolved by the caller.
    for pattern in input_patterns:
        # Check if pattern is absolute
        if not os.path.isabs(pattern):
            # Assume relative to PROJECT_ROOT if not absolute
            # This requires PROJECT_ROOT to be available or passed
            # Let's assume caller passes absolute paths or patterns resolved relative to root
             pattern_to_glob = pattern # Rely on caller providing correct pattern
             # Alternative (if PROJECT_ROOT is known here):
             # PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
             # pattern_to_glob = os.path.join(PROJECT_ROOT, pattern)
        else:
             pattern_to_glob = pattern

        matched_files = glob.glob(pattern_to_glob, recursive=True)
        if not matched_files:
             logging.warning(f"Input pattern '{pattern}' (resolved to '{pattern_to_glob}') did not match any files.")
        input_files.extend(matched_files)

    if not input_files:
        logging.error("No input files found based on the provided patterns.")
        return

    logging.info(f"Found {len(input_files)} potential input files: {input_files}")

    for filepath in input_files:
        # Avoid processing the same file twice if patterns overlap
        abs_filepath = os.path.abspath(filepath)
        if abs_filepath in processed_files:
            continue
        processed_files.add(abs_filepath)

        logging.info(f"Processing file: {filepath}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    total_loaded += 1
                    try:
                        data = json.loads(line)
                        data_source = None

                        # --- Extract Source Metadata --- #
                        # Attempt 1: Check for top-level 'metadata.source'
                        if isinstance(data.get('metadata'), dict):
                            data_source = data['metadata'].get('source')

                        # Attempt 2: Check if source is embedded in 'output' or 'completion' (less ideal)
                        if not data_source:
                            output_text = data.get('output', '') + data.get('completion', '')
                            meta_match = re.search(r'Metadata:\s*({.*?})', output_text, re.IGNORECASE)
                            if meta_match:
                                try:
                                    embedded_meta = json.loads(meta_match.group(1))
                                    data_source = embedded_meta.get('source')
                                except json.JSONDecodeError:
                                    pass # Ignore malformed embedded metadata

                        # Attempt 3: Infer source from filename if not found in data
                        if not data_source:
                            if 'synthetic' in os.path.basename(filepath).lower():
                                data_source = 'synthetic_inferred'
                            elif 'extract' in os.path.basename(filepath).lower():
                                 data_source = 'llm_extracted_inferred'
                            elif 'manual' in os.path.basename(filepath).lower() or 'finetuning_data.jsonl' == os.path.basename(filepath):
                                 data_source = 'manual' # Assume default is manual
                            else:
                                 data_source = 'unknown'
                            logging.debug(f"Inferred source '{data_source}' for line {line_num+1} in {filepath}")
                            # Optionally add inferred source back to data if structure allows
                            if isinstance(data.get('metadata'), dict):
                                 data['metadata']['source'] = data_source
                            elif 'output' in data and 'metadata': # Simple heuristic add
                                data['output'] += f"\\nMetadata: {json.dumps({'source': data_source})}"
                            elif 'completion' in data and 'metadata':
                                data['completion'] += f"\\nMetadata: {json.dumps({'source': data_source})}"
                        # --- End Extract Source Metadata --- #

                        # Filter based on source if a required source is specified
                        if required_source and data_source != required_source:
                            continue # Skip this entry

                        # Add source info if missing and possible (redundant if Attempt 3 worked)
                        if not data.get('metadata', {}).get('source') and data_source != 'unknown':
                            if 'metadata' not in data: data['metadata'] = {}
                            if isinstance(data['metadata'], dict):
                                data['metadata']['source'] = data_source

                        combined_data.append(data)
                        total_added += 1

                    except json.JSONDecodeError as e:
                        logging.warning(f"Skipping invalid JSON line {line_num + 1} in {filepath}: {e}")
                    except Exception as e:
                        logging.warning(f"Skipping line {line_num + 1} in {filepath} due to error: {e}")
        except FileNotFoundError:
            logging.error(f"Input file not found: {filepath}")
        except Exception as e:
            logging.error(f"Error processing file {filepath}: {e}")

    logging.info(f"Total lines loaded: {total_loaded}. Total examples added to combined dataset: {total_added}.")

    # Save the combined data
    if not combined_data:
        logging.warning("No data collected. Output file will not be created.")
        return

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in combined_data:
                f.write(json.dumps(example) + '\n')
        logging.info(f"Combined dataset saved to {output_file} with {len(combined_data)} examples.")
    except Exception as e:
        logging.error(f"Failed to save combined dataset: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine and filter fine-tuning datasets from JSONL files.")
    # Keep config path override
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH,
                        help="Path to the configuration YAML file.")
    # Change input_patterns to be optional - will use defaults from config if not provided
    parser.add_argument("--input_patterns", nargs='*', default=None, # Default to None
                        help="Override glob pattern(s) for input JSONL files (default: use paths from config like ft_manual_data_file, ft_extracted_data_file). Use quotes for patterns.")
    parser.add_argument("--output_file", type=str, default=None, # Default to None
                        help="Override the path to save the combined data JSONL file (default: from config paths.ft_combined_data_file).")
    parser.add_argument("--required_source", type=str, default=None,
                        help="Optional: Only include data points with this specific 'source' value in their metadata.")

    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Determine input patterns
    if args.input_patterns:
        # Use patterns provided via command line
        patterns_to_use = args.input_patterns
        logging.info(f"Using input patterns from command line: {patterns_to_use}")
    else:
        # Default to using specific files defined in config
        patterns_to_use = [
            cfg.paths.ft_manual_data_file,
            cfg.paths.ft_extracted_data_file
            # Add more default sources here if needed
        ]
        logging.info(f"Using default input files from config: {patterns_to_use}")

    # Determine output file path
    output_file_path = args.output_file if args.output_file else cfg.paths.ft_combined_data_file

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Call the function with resolved paths/patterns
    combine_datasets(patterns_to_use, output_file_path, args.required_source) 