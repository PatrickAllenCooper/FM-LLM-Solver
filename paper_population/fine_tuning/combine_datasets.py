import os
import json
import logging
import argparse
import glob

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(__file__)
# Default output file within the fine_tuning directory
DEFAULT_COMBINED_OUTPUT_FILE = os.path.join(BASE_DIR, "finetuning_data_combined.jsonl")

# --- Main Logic ---

def combine_datasets(input_patterns, output_file, required_source=None):
    """Combines data from multiple JSONL files, optionally filtering by source."""
    combined_data = []
    total_loaded = 0
    total_added = 0
    processed_files = set()

    # Expand glob patterns to get list of files
    input_files = []
    for pattern in input_patterns:
        # Construct full path if pattern is relative
        if not os.path.isabs(pattern):
            pattern = os.path.join(BASE_DIR, pattern) # Assume relative to script dir
        matched_files = glob.glob(pattern, recursive=True)
        if not matched_files:
             logging.warning(f"Input pattern '{pattern}' did not match any files.")
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
    parser.add_argument("input_patterns", nargs='+', # Accept one or more patterns
                        help="Glob pattern(s) for input JSONL files (e.g., '*.jsonl', 'manual_data.jsonl synthetic_data.jsonl'). Assumed relative to script directory if not absolute.")
    parser.add_argument("--output_file", type=str, default=DEFAULT_COMBINED_OUTPUT_FILE,
                        help=f"Path to save the combined data JSONL file (default: {DEFAULT_COMBINED_OUTPUT_FILE}).")
    parser.add_argument("--required_source", type=str, default=None,
                        help="Optional: Only include data points with this specific 'source' value in their metadata.")

    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    combine_datasets(args.input_patterns, args.output_file, args.required_source) 