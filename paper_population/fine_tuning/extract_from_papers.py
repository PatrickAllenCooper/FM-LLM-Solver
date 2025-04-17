import os
import json
import logging
import argparse
import random

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Input: Knowledge Base metadata
DEFAULT_KB_METADATA_PATH = os.path.join(PROJECT_ROOT, "knowledge_base", "knowledge_base_enhanced", "paper_metadata_enhanced.json")

# Output: File where reviewed/verified LLM extractions should be saved
DEFAULT_LLM_EXTRACTION_OUTPUT_FILE = os.path.join(BASE_DIR, "extracted_data_verified.jsonl")

# --- Constants ---
# Define the JSON structure we want the LLM to output for each found pair
# Using the 'instruction' format as an example
LLM_OUTPUT_JSON_STRUCTURE = '''
{{
  "instruction": "Given the autonomous system described by the following dynamics, propose a suitable barrier certificate function B(x).",
  "input": "<Extracted System Description (Dynamics, Sets, etc. Use LaTeX if possible)>",
  "output": "<Extracted Barrier Certificate B(x) (Use LaTeX if possible)>",
  "metadata": {{
    "source": "llm_extracted",
    "original_paper_source": "<Source PDF Filename>",
    "original_paper_pages": [<Page Numbers>],
    "original_chunk_indices": [<Chunk Indices>],
    "llm_confidence": "<High/Medium/Low>" // LLM's estimated confidence
  }}
}}
'''

# --- Helper Functions ---

def load_kb_metadata(metadata_path):
    """Loads the knowledge base metadata."""
    if not os.path.exists(metadata_path):
        logging.error(f"Knowledge base metadata not found at {metadata_path}. Run knowledge base builder first.")
        return None
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            # Keys should already be integers from builder
            metadata_map = json.load(f)
            # Convert keys back to int just in case they were loaded as str
            metadata_map = {int(k): v for k, v in metadata_map.items()}
        logging.info(f"Loaded metadata for {len(metadata_map)} chunks from {metadata_path}.")
        return metadata_map
    except Exception as e:
        logging.error(f"Failed to load metadata from {metadata_path}: {e}")
        return None

def group_chunks_by_paper(metadata_map):
    """Groups chunks by their source paper for context."""
    papers = {}
    for chunk_idx, meta in metadata_map.items():
        source_file = meta.get('source')
        if not source_file:
            continue
        if source_file not in papers:
            papers[source_file] = []
        # Store index and page numbers along with text
        papers[source_file].append({
            'chunk_idx': chunk_idx,
            'pages': meta.get('pages', []),
            'text': meta.get('text', '')
        })

    # Sort chunks within each paper by page and then index (approximates reading order)
    for source_file in papers:
        papers[source_file].sort(key=lambda x: (min(x['pages']) if x['pages'] else 0, x['chunk_idx']))

    logging.info(f"Grouped chunks into {len(papers)} papers.")
    return papers

def generate_extraction_prompt(paper_chunks, source_filename):
    """Generates a prompt for an LLM to extract pairs from a paper's chunks."""
    context = ""
    chunk_indices = []
    all_pages = set()
    for chunk_info in paper_chunks:
        context += f"--- Chunk Index: {chunk_info['chunk_idx']} (Pages: {chunk_info['pages']}) ---\n"
        context += chunk_info['text'] + "\n\n"
        chunk_indices.append(chunk_info['chunk_idx'])
        all_pages.update(chunk_info['pages'])

    prompt = f"""
Read the following text chunks extracted from the research paper '{source_filename}'.
Identify all instances where a specific autonomous system (defined by its state variables and dynamics, potentially including initial/unsafe sets) is presented along with a corresponding barrier certificate function B(x) proposed or analyzed for it.

For EACH valid pair you identify, format it EXACTLY as a single JSON object matching the structure below. Output NOTHING ELSE except these JSON objects, one per line.

Required JSON Output Structure:
{LLM_OUTPUT_JSON_STRUCTURE}

- Replace placeholders like <Extracted System Description...> with the actual extracted text.
- Use LaTeX for mathematical notation if present in the source text or if it clarifies the expression.
- If multiple pairs are found, output each as a separate JSON object on a new line.
- If NO pairs are found, output the single word: NONE

Source Text Chunks:
{context.strip()}

Extracted JSON objects (or NONE):
"""
    return prompt, chunk_indices, sorted(list(all_pages))

# --- Main Logic ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prompts for LLM-based extraction of System-Certificate pairs from the knowledge base.")
    parser.add_argument("--kb_meta", type=str, default=DEFAULT_KB_METADATA_PATH,
                         help=f"Path to knowledge base metadata JSON (default: {DEFAULT_KB_METADATA_PATH})")
    parser.add_argument("--output_instructions_file", type=str, default="llm_extraction_prompts.txt",
                        help="File to save the generated prompts and instructions.")
    parser.add_argument("--num_papers", type=int, default=5,
                        help="Number of papers (or chunk groups) to generate prompts for (set to -1 for all). Default: 5.")
    parser.add_argument("--reviewed_output", type=str, default=DEFAULT_LLM_EXTRACTION_OUTPUT_FILE,
                         help=f"Path where the MANUALLY REVIEWED JSONL data should be saved (for informational purposes). Default: {DEFAULT_LLM_EXTRACTION_OUTPUT_FILE}")

    args = parser.parse_args()

    kb_metadata = load_kb_metadata(args.kb_meta)
    if not kb_metadata:
        sys.exit(1)

    papers = group_chunks_by_paper(kb_metadata)
    paper_keys = list(papers.keys())
    random.shuffle(paper_keys) # Process in random order

    num_to_process = args.num_papers
    if num_to_process == -1 or num_to_process > len(paper_keys):
        num_to_process = len(paper_keys)

    logging.info(f"Generating prompts for {num_to_process} papers...")

    all_prompts_content = """
# LLM Extraction Prompts and Instructions

This file contains prompts designed to be used with a powerful Large Language Model
(e.g., GPT-4, Claude 3 Opus) to extract (System Description, Barrier Certificate) pairs
from research paper text chunks.

**CRITICAL INSTRUCTIONS:**

1.  **Use a Capable LLM:** These prompts require strong reading comprehension and JSON formatting abilities.
2.  **Process Each Prompt:** Copy each prompt below (between the === START PROMPT === and === END PROMPT === markers) and paste it into the LLM interface.
3.  **Collect LLM Output:** Save the exact output generated by the LLM for each prompt.
4.  **MANUAL REVIEW (ESSENTIAL):**
    *   Carefully review the JSON objects output by the LLM.
    *   **Verify the correctness** of the extracted system dynamics and barrier certificate against the source paper context (or the original PDF if necessary).
    *   Check if the JSON format is perfectly valid.
    *   Discard any incorrect, nonsensical, or badly formatted JSON objects.
    *   If the LLM outputted "NONE", ensure no pairs were missed.
5.  **Save Verified Data:** Append ONLY the **verified and correct** JSON objects to your final reviewed data file (e.g., `{args.reviewed_output}`). Each valid JSON object should be on its own line.

Failure to perform careful manual review will likely introduce errors into your fine-tuning dataset.
"""

    processed_count = 0
    for i in range(num_to_process):
        source_filename = paper_keys[i]
        paper_chunks = papers[source_filename]

        if not paper_chunks:
            continue

        prompt, chunk_ids, page_nums = generate_extraction_prompt(paper_chunks, source_filename)

        all_prompts_content += f"\n\n{'='*20} START PROMPT {processed_count + 1} {'='*20}\n"
        all_prompts_content += f"# Source Paper: {source_filename}\n"
        all_prompts_content += f"# Relevant Pages: {page_nums}\n"
        all_prompts_content += f"# Relevant Chunk Indices: {chunk_ids}\n\n"
        all_prompts_content += prompt
        all_prompts_content += f"\n{'='*20} END PROMPT {processed_count + 1} {'='*20}\n"
        processed_count += 1

    # Save the prompts and instructions to a file
    try:
        with open(args.output_instructions_file, 'w', encoding='utf-8') as f:
            f.write(all_prompts_content)
        logging.info(f"Generated {processed_count} prompts. Instructions and prompts saved to: {args.output_instructions_file}")
        logging.info(f"Please manually process this file with a powerful LLM and save the reviewed, verified results to a file like: {args.reviewed_output}")
    except Exception as e:
        logging.error(f"Failed to write prompts file: {e}")

    print(f"\nGenerated prompts for {processed_count} papers.")
    print(f"See '{args.output_instructions_file}' for prompts and instructions.")
    print(f"Remember to MANUALLY REVIEW the LLM output before saving to '{args.reviewed_output}'.") 