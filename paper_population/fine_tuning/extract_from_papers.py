import os
import json
import logging
import argparse
import random
import sys

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Input: Knowledge Base metadata (Point to Mathpix output)
DEFAULT_KB_METADATA_PATH = os.path.join(PROJECT_ROOT, "knowledge_base", "knowledge_base_mathpix", "paper_metadata_mathpix.jsonl")

# Output: File where reviewed/verified LLM extractions should be saved
DEFAULT_LLM_EXTRACTION_OUTPUT_FILE = os.path.join(BASE_DIR, "extracted_data_verified.jsonl")

# --- Constants ---
# Define the JSON structure we want the LLM to output for each found pair
# Using the 'instruction' format as an example
LLM_OUTPUT_JSON_STRUCTURE = '''
{{
  "instruction": "Given the autonomous system described by the following dynamics, propose a suitable barrier certificate function B(x).",
  "input": "<Extracted System Description (Dynamics, Sets, etc. PRESERVE LaTeX format exactly)>",
  "output": "<Extracted Barrier Certificate B(x) (PRESERVE LaTeX format exactly)>",
  "metadata": {{
    "source": "llm_extracted",
    "original_paper_source": "<Source PDF Filename>",
    "original_paper_pages": [<Page Numbers (may be unknown)>],
    "original_chunk_indices": [<Chunk Indices>],
    "llm_confidence": "<High/Medium/Low>" // LLM's estimated confidence
  }}
}}
'''

# --- Helper Functions ---

def load_kb_metadata_jsonl(metadata_path):
    """Loads the knowledge base metadata from JSONL file."""
    if not os.path.exists(metadata_path):
        logging.error(f"Knowledge base metadata JSONL not found at {metadata_path}. Run knowledge base builder first.")
        return None
    metadata_map = {}
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip():
                    try:
                        data = json.loads(line)
                        # Use the line number or a dedicated chunk_id if present
                        chunk_id = data.get('chunk_id', i) 
                        metadata_map[chunk_id] = data
                    except json.JSONDecodeError as json_err:
                        logging.warning(f"Skipping invalid JSON line {i+1} in {metadata_path}: {json_err}")
        logging.info(f"Loaded metadata for {len(metadata_map)} chunks from {metadata_path}.")
        return metadata_map
    except Exception as e:
        logging.error(f"Failed to load metadata from {metadata_path}: {e}")
        return None

def group_chunks_by_paper(metadata_map):
    """Groups chunks (from JSONL structure) by their source paper for context."""
    papers = {}
    for chunk_id, chunk_data in metadata_map.items():
        # Access metadata nested within the loaded data object
        meta = chunk_data.get('metadata', {})
        source_file = meta.get('source')
        if not source_file:
            continue
        if source_file not in papers:
            papers[source_file] = []
        # Store index, pages, and the MMD text
        papers[source_file].append({
            'chunk_id': chunk_id,
            'pages': meta.get('pages', ['unknown']), # Pages might be unknown from Mathpix
            'text': chunk_data.get('text', '') # Get text from top level
        })

    # Sort chunks within each paper by chunk_id (approximates order)
    for source_file in papers:
        papers[source_file].sort(key=lambda x: x['chunk_id'])

    logging.info(f"Grouped chunks into {len(papers)} papers.")
    return papers

def generate_extraction_prompt(paper_chunks, source_filename):
    """Generates a prompt for an LLM to extract pairs from MMD paper content."""
    context = ""
    chunk_indices = []
    all_pages = set()
    for chunk_info in paper_chunks:
        context += f"--- Chunk Index: {chunk_info['chunk_id']} (Pages: {chunk_info['pages']}) ---\n"
        context += chunk_info['text'] + "\n\n"
        chunk_indices.append(chunk_info['chunk_id'])
        if chunk_info['pages'] != ['unknown']:
             all_pages.update(chunk_info['pages'])

    prompt = f"""
Read the following text chunks extracted from the research paper '{source_filename}'.
The text is in Markdown format (MMD) and includes mathematical notation using LaTeX delimiters (e.g., $...$, $$...$$).

TASK: Identify all instances where BOTH a specific autonomous system AND a corresponding barrier certificate function B(x) are clearly presented and associated with each other.

- The system description should include dynamics (differential equations), state variables, and potentially initial/unsafe sets.
- The barrier certificate B(x) should be an explicit mathematical function.

Pay close attention to section headings (like `## System Model` or `## Barrier Certificate Construction`) and LaTeX math content.

For EACH valid (System, Certificate) pair you identify, format it EXACTLY as a single JSON object matching the structure below. Output NOTHING ELSE except these JSON objects, one per line. **CRITICALLY, preserve the original LaTeX formatting for all mathematical expressions exactly as they appear in the source text.**

Required JSON Output Structure:
{LLM_OUTPUT_JSON_STRUCTURE}

- Replace placeholders like <Extracted System Description...> with the actual extracted text, keeping LaTeX intact.
- Replace <Extracted Barrier Certificate B(x)...> with the function definition, keeping LaTeX intact.
- Replace <Source PDF Filename> with '{source_filename}'.
- Replace [<Page Numbers>] with {sorted(list(all_pages)) if all_pages else 'unknown'}.
- Replace [<Chunk Indices>] with {chunk_indices}.
- Estimate your confidence (High/Medium/Low) in the correctness and association of the extracted pair.
- If multiple pairs are found, output each as a separate JSON object on a new line.
- If NO pairs are found, output the single word: NONE

Source MMD Text Chunks:
{context.strip()}

Extracted JSON objects (or NONE):
"""
    # We pass back chunk_indices and page numbers separately for metadata filling now
    return prompt, chunk_indices, sorted(list(all_pages)) if all_pages else ['unknown']

# --- Main Logic ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prompts for LLM-based extraction of System-Certificate pairs from Mathpix MMD knowledge base.")
    parser.add_argument("--kb_meta", type=str, default=DEFAULT_KB_METADATA_PATH,
                         help=f"Path to knowledge base metadata JSONL file (default: {DEFAULT_KB_METADATA_PATH})")
    parser.add_argument("--output_instructions_file", type=str, default="llm_extraction_prompts_mmd.txt",
                        help="File to save the generated prompts and instructions.")
    parser.add_argument("--num_papers", type=int, default=10,
                        help="Number of papers to generate prompts for (set to -1 for all). Default: 10.")
    parser.add_argument("--reviewed_output", type=str, default=DEFAULT_LLM_EXTRACTION_OUTPUT_FILE,
                         help=f"Path where the MANUALLY REVIEWED JSONL data should be saved (for informational purposes). Default: {DEFAULT_LLM_EXTRACTION_OUTPUT_FILE}")

    args = parser.parse_args()

    # Load from JSONL
    kb_metadata_map = load_kb_metadata_jsonl(args.kb_meta)
    if not kb_metadata_map:
        sys.exit(1)

    papers = group_chunks_by_paper(kb_metadata_map)
    paper_keys = list(papers.keys())
    random.shuffle(paper_keys)

    num_to_process = args.num_papers
    if num_to_process == -1 or num_to_process > len(paper_keys):
        num_to_process = len(paper_keys)

    logging.info(f"Generating MMD-based prompts for {num_to_process} papers...")

    all_prompts_content = """
# LLM Extraction Prompts and Instructions (from Mathpix MMD)

This file contains prompts designed to be used with a powerful Large Language Model
(e.g., GPT-4, Claude 3 Opus) to extract (System Description, Barrier Certificate) pairs
from Mathpix-generated Markdown (MMD) text containing LaTeX.

**CRITICAL INSTRUCTIONS:**

1.  **Use a Capable LLM:** These prompts require strong reading comprehension, understanding of structured markdown, LaTeX parsing, and JSON formatting abilities.
2.  **Process Each Prompt:** Copy each prompt below (between the === START PROMPT === and === END PROMPT === markers) and paste it into the LLM interface.
3.  **Collect LLM Output:** Save the exact output generated by the LLM for each prompt.
4.  **MANUAL REVIEW (ESSENTIAL):**
    *   Carefully review the JSON objects output by the LLM.
    *   **Verify the correctness** of the extracted system dynamics and barrier certificate, **paying close attention to the accuracy of the LaTeX math**, against the source paper context (or the original PDF if necessary).
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

        # Generate the prompt, get back associated chunk/page info for context
        prompt, chunk_ids, page_nums = generate_extraction_prompt(paper_chunks, source_filename)

        all_prompts_content += f"\n\n{'='*20} START PROMPT {processed_count + 1} {'='*20}\n"
        all_prompts_content += f"# Source Paper: {source_filename}\n"
        all_prompts_content += f"# Relevant Pages (if known): {page_nums}\n"
        all_prompts_content += f"# Relevant Chunk Indices: {chunk_ids}\n\n"
        all_prompts_content += prompt
        all_prompts_content += f"\n{'='*20} END PROMPT {processed_count + 1} {'='*20}\n"
        processed_count += 1

    # Save the prompts and instructions to a file
    try:
        output_instructions_path = os.path.join(BASE_DIR, args.output_instructions_file)
        with open(output_instructions_path, 'w', encoding='utf-8') as f:
            f.write(all_prompts_content)
        logging.info(f"Generated {processed_count} prompts. Instructions and prompts saved to: {output_instructions_path}")
        logging.info(f"Please manually process this file with a powerful LLM and save the reviewed, verified results to a file like: {args.reviewed_output}")
    except Exception as e:
        logging.error(f"Failed to write prompts file: {e}")

    print(f"\nGenerated prompts for {processed_count} papers.")
    print(f"See '{args.output_instructions_file}' (in '{BASE_DIR}') for prompts and instructions.")
    print(f"Remember to MANUALLY REVIEW the LLM output before saving to '{args.reviewed_output}'.") 