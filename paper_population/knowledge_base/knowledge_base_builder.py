import os
import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import spacy # Keep for potential MMD chunking or fallback
import logging
import sys
import requests # For MathPix API calls
import time
import argparse # For command-line arguments

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mathpix API Credentials (READ FROM ENVIRONMENT VARIABLES)
# IMPORTANT: Set these environment variables before running the script
# export MATHPIX_APP_ID='your_app_id'
# export MATHPIX_APP_KEY='your_app_key'
MATHPIX_APP_ID = os.environ.get("MATHPIX_APP_ID")
MATHPIX_APP_KEY = os.environ.get("MATHPIX_APP_KEY")
MATHPIX_API_URL = "https://api.mathpix.com/v3/pdf"

# Directories
BASE_DIR = os.path.dirname(__file__) # Directory of the script (knowledge_base)
# Assume project root is one level up
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
# Default PDF input relative to project root
DEFAULT_PDF_INPUT_DIR = os.path.join(PROJECT_ROOT, "recent_papers_all_sources_v2")
# Default KB output relative to this script's directory
DEFAULT_KB_OUTPUT_DIR = os.path.join(BASE_DIR, "knowledge_base_mathpix")

# Embedding Model
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2' # Good general-purpose starting model

# Vector Store
VECTOR_STORE_FILENAME = "paper_index_mathpix.faiss"
METADATA_FILENAME = "paper_metadata_mathpix.jsonl" # Use JSONL for potentially large MMD strings

# Chunking Parameters (adjust for MMD content)
# Chunking by paragraphs seems reasonable for MMD
CHUNK_TARGET_SIZE_MMD = 1000 # Target characters per chunk
CHUNK_OVERLAP_MMD = 150    # Character overlap

# --- SpaCy Model Loading (Optional - can be used for paragraph splitting) ---
# Keep SpaCy loading, but make it optional if we primarily split by markdown rules
SPACY_MODEL_NAME = "en_core_web_sm"
nlp = None
try:
    nlp = spacy.load(SPACY_MODEL_NAME)
    logging.info(f"SpaCy model '{SPACY_MODEL_NAME}' loaded (available for text processing).")
except OSError:
    logging.warning(f"SpaCy model '{SPACY_MODEL_NAME}' not found. Will fallback to basic newline splitting.")
except Exception as e:
    logging.error(f"An error occurred loading the SpaCy model: {e}")
    # Don't exit, just disable SpaCy features

# --- Helper Functions ---

def check_mathpix_credentials():
    """Checks if Mathpix credentials are set."""
    if not MATHPIX_APP_ID or not MATHPIX_APP_KEY:
        logging.error("Mathpix API credentials (MATHPIX_APP_ID, MATHPIX_APP_KEY) not found in environment variables.")
        logging.error("Please set them before running. Exiting.")
        sys.exit(1)
    logging.info("Mathpix credentials found.")

def process_pdf_with_mathpix(pdf_path):
    """Sends PDF to Mathpix API and retrieves MMD content."""
    logging.info(f"Processing '{os.path.basename(pdf_path)}' with Mathpix API...")
    headers = {
        'app_id': MATHPIX_APP_ID,
        'app_key': MATHPIX_APP_KEY
    }
    # Options for the conversion, requesting MMD format
    options = {
        "conversion_formats": {"mmd": True}, # Request Mathpix Markdown
        "math_inline_delimiters": ["$", "$"],
        "math_display_delimiters": ["$$", "$$"],
        # Add other options if needed, e.g., "include_line_data": True
    }
    payload = {'options_json': json.dumps(options)}

    try:
        with open(pdf_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(MATHPIX_API_URL, headers=headers, files=files, data=payload, timeout=300) # Long timeout for large PDFs

        if response.status_code == 200:
            response_data = response.json()
            if "request_id" in response_data: # Async processing started
                 pdf_id = response_data["request_id"]
                 logging.info(f"  Mathpix request submitted. PDF ID: {pdf_id}. Waiting for conversion...")
                 return get_mathpix_result(pdf_id)
            elif "mmd" in response_data: # Direct result (less common for PDF endpoint)
                 logging.info("  Mathpix returned result directly.")
                 return response_data["mmd"]
            else:
                 logging.error(f"  Mathpix API Error: Unexpected response format. {response_data.get('error', '')}")
                 return None
        else:
            logging.error(f"  Mathpix API request failed. Status: {response.status_code}, Response: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"  Error communicating with Mathpix API: {e}")
        return None
    except Exception as e:
        logging.error(f"  Unexpected error during Mathpix processing for {pdf_path}: {e}")
        return None

def get_mathpix_result(pdf_id, max_wait_sec=600, poll_interval=10):
    """Polls Mathpix API to get the result for a given PDF ID."""
    headers = {
        'app_id': MATHPIX_APP_ID,
        'app_key': MATHPIX_APP_KEY
    }
    url = f"{MATHPIX_API_URL}/{pdf_id}.mmd"
    start_time = time.time()

    while time.time() - start_time < max_wait_sec:
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                 logging.info(f"    Mathpix conversion complete for ID {pdf_id}.")
                 return response.text # Return MMD content
            elif response.status_code == 404: # Not ready yet
                 logging.info(f"    Result for {pdf_id} not ready yet, waiting {poll_interval}s...")
            else:
                 # Check for error message in response
                 try: error_info = response.json() 
                 except: error_info = response.text
                 logging.error(f"    Error fetching Mathpix result for {pdf_id}. Status: {response.status_code}, Info: {error_info}")
                 return None # Permanent error
        except requests.exceptions.RequestException as e:
            logging.error(f"    Error polling Mathpix result: {e}")
            # Could implement retries here
        except Exception as e:
             logging.error(f"    Unexpected error polling Mathpix: {e}")
             return None
        time.sleep(poll_interval)

    logging.error(f"Mathpix conversion timed out after {max_wait_sec} seconds for ID {pdf_id}.")
    return None

def chunk_mmd_content(mmd_text, target_size, overlap, source_pdf):
    """Chunks MMD content, trying to respect paragraph boundaries."""
    chunks = []
    if not mmd_text:
        return chunks

    # Split by double newlines (common paragraph separator in Markdown)
    paragraphs = re.split(r'\n\s*\n', mmd_text) # Split on one or more blank lines
    paragraphs = [p.strip() for p in paragraphs if p.strip()] # Remove empty paragraphs

    logging.info(f"Splitting MMD into {len(paragraphs)} paragraphs for chunking.")

    current_chunk_text = ""
    current_chunk_len = 0
    start_para_index = 0

    for i, para in enumerate(paragraphs):
        para_len = len(para)
        # If adding the next paragraph exceeds target size significantly,
        # and current chunk is not empty, finalize the current chunk.
        if current_chunk_len > 0 and current_chunk_len + para_len > target_size + overlap:
            # Check if current chunk alone is large enough
            if current_chunk_len >= target_size // 2:
                 metadata = {
                     'source': os.path.basename(source_pdf),
                     # Page numbers are lost with full PDF processing via Mathpix
                     # We could try to parse page markers if Mathpix includes them
                     'pages': ['unknown'],
                     'start_para_index': start_para_index,
                     'end_para_index': i - 1
                 }
                 chunks.append({'text': current_chunk_text, 'metadata': metadata})

                 # Start new chunk with overlap (based on characters for simplicity)
                 overlap_start_char = max(0, current_chunk_len - overlap)
                 current_chunk_text = current_chunk_text[overlap_start_char:] + "\n\n" + para
                 current_chunk_len = len(current_chunk_text)
                 start_para_index = i # This new chunk starts from current paragraph
            else:
                 # Current chunk is too small, just append the new paragraph
                 current_chunk_text += "\n\n" + para
                 current_chunk_len += len("\n\n") + para_len
        else:
            # Append paragraph to current chunk
            if current_chunk_len > 0:
                 current_chunk_text += "\n\n" + para
                 current_chunk_len += len("\n\n") + para_len
            else:
                 current_chunk_text = para
                 current_chunk_len = para_len
                 start_para_index = i

    # Add the last remaining chunk
    if current_chunk_text:
        metadata = {
           'source': os.path.basename(source_pdf),
           'pages': ['unknown'],
           'start_para_index': start_para_index,
           'end_para_index': len(paragraphs) - 1
        }
        chunks.append({'text': current_chunk_text, 'metadata': metadata})

    logging.info(f"Created {len(chunks)} MMD-based chunks for {os.path.basename(source_pdf)}")
    return chunks

# --- Main Logic (Refactored for Mathpix) ---
def build_knowledge_base(pdf_input_dir, output_dir):
    """Main function to build the vector store and metadata using Mathpix."""
    check_mathpix_credentials()

    os.makedirs(output_dir, exist_ok=True)

    all_chunks_with_metadata = []
    processed_files = 0
    failed_files = 0

    logging.info(f"Starting PDF processing from: {pdf_input_dir}")
    all_pdf_paths = []
    for root, _, files in os.walk(pdf_input_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                all_pdf_paths.append(os.path.join(root, file))

    logging.info(f"Found {len(all_pdf_paths)} PDF files.")

    for pdf_path in all_pdf_paths:
        logging.info(f"--- Processing: {os.path.basename(pdf_path)} ---")
        mmd_content = process_pdf_with_mathpix(pdf_path)

        if not mmd_content:
            logging.warning(f"Failed to get MMD content from Mathpix for {os.path.basename(pdf_path)}")
            failed_files += 1
            continue

        # Extract title from MMD (heuristic: first line if it looks like a title)
        lines = mmd_content.strip().split('\n')
        potential_title = lines[0].strip("# ") if lines else "Unknown Title"
        # You might add more heuristics here

        doc_chunks = chunk_mmd_content(
            mmd_content,
            CHUNK_TARGET_SIZE_MMD,
            CHUNK_OVERLAP_MMD,
            pdf_path
        )
        # Add potential title to metadata of each chunk
        for chunk in doc_chunks:
             chunk['metadata']['potential_title'] = potential_title

        all_chunks_with_metadata.extend(doc_chunks)
        processed_files += 1
        time.sleep(0.5) # Small delay between processing files

    if not all_chunks_with_metadata:
        logging.error("No text chunks were generated from any PDF using Mathpix. Exiting.")
        return

    logging.info(f"Successfully processed {processed_files} PDFs. Failed: {failed_files}.")
    logging.info(f"Total chunks created: {len(all_chunks_with_metadata)}.")

    # 2. Embed Chunks
    try:
        logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        logging.error(f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}")
        return

    logging.info("Embedding chunks...")
    chunk_texts = [chunk['text'] for chunk in all_chunks_with_metadata]
    try:
        embeddings = model.encode(chunk_texts, show_progress_bar=True, batch_size=32)
        embeddings = np.array(embeddings).astype('float32')
        logging.info(f"Embeddings generated with shape: {embeddings.shape}")
    except Exception as e:
        logging.error(f"Failed during embedding generation: {e}")
        return

    # 3. Build and Save Vector Store
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    try:
        index.add(embeddings)
        logging.info(f"Added {index.ntotal} vectors to FAISS index.")
        index_path = os.path.join(output_dir, VECTOR_STORE_FILENAME)
        faiss.write_index(index, index_path)
        logging.info(f"Saved FAISS index to {index_path}")
    except Exception as e:
        logging.error(f"Failed to build or save FAISS index: {e}")
        return

    # 4. Save Metadata (as JSON Lines)
    metadata_path = os.path.join(output_dir, METADATA_FILENAME)
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(all_chunks_with_metadata):
                # Ensure metadata keys are consistent if needed downstream
                data_to_save = {
                     "chunk_id": i,
                     "text": chunk['text'],
                     "metadata": chunk['metadata']
                }
                f.write(json.dumps(data_to_save) + '\n')
        logging.info(f"Saved metadata mapping to {metadata_path}")
    except Exception as e:
        logging.error(f"Failed to save metadata JSONL: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build knowledge base from PDFs using Mathpix API.")
    parser.add_argument("--pdf_dir", type=str, default=DEFAULT_PDF_INPUT_DIR,
                        help=f"Directory containing PDF files (default: {DEFAULT_PDF_INPUT_DIR})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_KB_OUTPUT_DIR,
                        help=f"Directory to save the knowledge base index and metadata (default: {DEFAULT_KB_OUTPUT_DIR})")
    # Add optional args for chunk size, overlap?
    args = parser.parse_args()

    # Basic dependency check
    try:
        import sentence_transformers
        import faiss
        import numpy
        import requests
        # SpaCy is optional now
    except ImportError as e:
        print(f"Error: Missing dependency - {e}", file=sys.stderr)
        print("Please install required packages:", file=sys.stderr)
        print("pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    build_knowledge_base(args.pdf_dir, args.output_dir)
    logging.info("Mathpix-based knowledge base build process finished.") 