import argparse # Keep argparse ONLY for --config override

# Add project root to Python path
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

# Now we can import the utils module
from utils.config_loader import load_config, DEFAULT_CONFIG_PATH # Import config loader

import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import spacy # Keep for potential MMD chunking or fallback
import logging
import requests # For MathPix API calls
import time

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
# We need argparse BEFORE loading config to allow overriding the config path
parser_init = argparse.ArgumentParser(add_help=False) # Initial parser for config path only
parser_init.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to the configuration YAML file.")
args_init, _ = parser_init.parse_known_args()
cfg = load_config(args_init.config)

# Mathpix API Credentials (READ FROM ENVIRONMENT VARIABLES)
MATHPIX_APP_ID = os.environ.get("MATHPIX_APP_ID")
MATHPIX_APP_KEY = os.environ.get("MATHPIX_APP_KEY")
if not MATHPIX_APP_ID or not MATHPIX_APP_KEY:
    logging.error("Mathpix API credentials (MATHPIX_APP_ID, MATHPIX_APP_KEY) not found in environment variables.")
    logging.error("Please set them before running. Exiting.")
    sys.exit(1)
logging.info("Mathpix credentials found.")

MATHPIX_API_URL = "https://api.mathpix.com/v3/pdf"

# Get parameters from config
EMBEDDING_MODEL_NAME = cfg.knowledge_base.embedding_model_name
VECTOR_STORE_FILENAME = cfg.paths.kb_vector_store_filename
METADATA_FILENAME = cfg.paths.kb_metadata_filename
CHUNK_TARGET_SIZE_MMD = cfg.knowledge_base.chunk_target_size_mmd
CHUNK_OVERLAP_MMD = cfg.knowledge_base.chunk_overlap_mmd
POLL_MAX_WAIT = cfg.knowledge_base.mathpix_poll_max_wait_sec
POLL_INTERVAL = cfg.knowledge_base.mathpix_poll_interval

# --- SpaCy Model Loading --- (Keep as is, not configuration-dependent)
SPACY_MODEL_NAME = "en_core_web_sm"
nlp = None
try:
    nlp = spacy.load(SPACY_MODEL_NAME)
    logging.info(f"SpaCy model '{SPACY_MODEL_NAME}' loaded (available for text processing).")
except OSError:
    logging.warning(f"SpaCy model '{SPACY_MODEL_NAME}' not found. Will fallback to basic newline splitting.")
except Exception as e:
    logging.error(f"An error occurred loading the SpaCy model: {e}")

# --- Helper Functions --- (Update functions needing config)

def check_mathpix_credentials():
    """Checks if Mathpix credentials are set (Already checked above)."""
    pass # Already checked when loading config

def process_pdf_with_mathpix(pdf_path, mathpix_app_id, mathpix_app_key):
    """Sends PDF to Mathpix API and retrieves MMD content."""
    logging.info(f"Processing '{os.path.basename(pdf_path)}' with Mathpix API...")
    headers = {
        'app_id': mathpix_app_id,
        'app_key': mathpix_app_key
    }
    # Options for the conversion, requesting MMD format
    options = {
        # "conversion_formats": {"mmd": True}, # REMOVED: MMD is fetched via GET .mmd
        "math_inline_delimiters": ["$", "$"],
        "math_display_delimiters": ["$$", "$$"],
        # Add other options if needed, e.g., "include_line_data": True
    }
    payload = {'options_json': json.dumps(options)}

    try:
        with open(pdf_path, 'rb') as f:
            files = {'file': f}
            # Use constants/variables loaded earlier
            response = requests.post(MATHPIX_API_URL, headers=headers, files=files, data=payload, timeout=300) # Long timeout for large PDFs

        if response.status_code == 200:
            response_data = response.json()
            # Check for pdf_id to handle async response
            if "pdf_id" in response_data: # CHANGED from request_id
                 pdf_id = response_data["pdf_id"] # CHANGED from request_id
                 logging.info(f"  Mathpix request submitted. PDF ID: {pdf_id}. Waiting for conversion...")
                 # Pass keys and config polling values to get_mathpix_result
                 return get_mathpix_result(pdf_id, mathpix_app_id, mathpix_app_key, POLL_MAX_WAIT, POLL_INTERVAL)
            elif "mmd" in response_data: # Direct result (less common for PDF endpoint)
                 logging.info("  Mathpix returned result directly.")
                 return response_data["mmd"]
            else:
                 # Log the raw response for debugging
                 logging.error(f"  Mathpix API Error: Unexpected response format. Response Text: {response.text} Raw Response Object: {response_data}")
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

def get_mathpix_result(pdf_id, mathpix_app_id, mathpix_app_key, max_wait_sec, poll_interval):
    """Polls Mathpix API to get the result for a given PDF ID."""
    headers = {
        'app_id': mathpix_app_id,
        'app_key': mathpix_app_key
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
def build_knowledge_base(cfg):
    """
    Main function to build the vector store and metadata using Mathpix.
    
    This function performs these key steps:
    1. Processes PDF files using Mathpix API to convert to MMD format
    2. Chunks the MMD content into manageable segments
    3. Embeds the chunks using a sentence transformer model
    4. Builds a FAISS vector index from the embeddings
    5. Saves the index and metadata to disk
    
    Parameters
    ----------
    cfg : OmegaConf
        Configuration object containing all necessary parameters
    
    Returns
    -------
    bool
        True if the knowledge base was built successfully, False otherwise.
    """
    # Extract configuration parameters
    output_dir = cfg.paths.kb_output_dir
    pdf_input_dir = cfg.paths.pdf_input_dir
    embedding_model_name = cfg.knowledge_base.embedding_model_name
    vector_store_filename = cfg.paths.kb_vector_store_filename
    metadata_filename = cfg.paths.kb_metadata_filename
    chunk_target_size = cfg.knowledge_base.chunk_target_size_mmd
    chunk_overlap = cfg.knowledge_base.chunk_overlap_mmd

    # Get Mathpix credentials loaded at top level
    mathpix_app_id = MATHPIX_APP_ID
    mathpix_app_key = MATHPIX_APP_KEY

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Storage for processed chunks
    all_chunks_with_metadata = []
    processed_files = 0
    failed_files = 0

    # --- STEP 1: Process PDF Files with Mathpix ---
    logging.info(f"Starting PDF processing from: {pdf_input_dir}")
    
    # Find all PDF files in input directory (including subdirectories)
    all_pdf_paths = []
    for root, _, files in os.walk(pdf_input_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                all_pdf_paths.append(os.path.join(root, file))

    logging.info(f"Found {len(all_pdf_paths)} PDF files.")
    
    if not all_pdf_paths:
        logging.error(f"No PDF files found in {pdf_input_dir}. Exiting.")
        return False # <-- RETURN False on failure

    # Process each PDF file
    for pdf_path in all_pdf_paths:
        logging.info(f"--- Processing: {os.path.basename(pdf_path)} ---")
        
        # Convert PDF to MMD using Mathpix API
        mmd_content = process_pdf_with_mathpix(pdf_path, mathpix_app_id, mathpix_app_key)

        if not mmd_content:
            logging.warning(f"Failed to get MMD content from Mathpix for {os.path.basename(pdf_path)}")
            failed_files += 1
            continue

        # Extract title heuristically (first line if it looks like a title)
        lines = mmd_content.strip().split('\n')
        potential_title = lines[0].strip("# ") if lines else "Unknown Title"

        # --- STEP 2: Chunk the MMD Content ---
        doc_chunks = chunk_mmd_content(
            mmd_content,
            chunk_target_size,
            chunk_overlap,
            pdf_path
        )
        
        # Add potential title to metadata of each chunk
        for chunk in doc_chunks:
             chunk['metadata']['potential_title'] = potential_title

        # Add chunks to collection
        all_chunks_with_metadata.extend(doc_chunks)
        processed_files += 1
        time.sleep(0.5)  # Small delay between processing files

    # Validate that we have chunks to process
    if not all_chunks_with_metadata:
        logging.error("No text chunks were generated from any PDF using Mathpix. Exiting.")
        return False # <-- RETURN False on failure

    logging.info(f"Successfully processed {processed_files} PDFs. Failed: {failed_files}.")
    logging.info(f"Total chunks created: {len(all_chunks_with_metadata)}.")

    # --- STEP 3: Embed Chunks ---
    try:
        logging.info(f"Loading embedding model: {embedding_model_name}")
        model = SentenceTransformer(embedding_model_name)
    except Exception as e:
        logging.error(f"Failed to load embedding model '{embedding_model_name}': {e}")
        return False # <-- RETURN False on failure

    logging.info("Embedding chunks...")
    chunk_texts = [chunk['text'] for chunk in all_chunks_with_metadata]
    try:
        # Generate embeddings with progress bar
        embeddings = model.encode(chunk_texts, show_progress_bar=True, batch_size=32)
        embeddings = np.array(embeddings).astype('float32')
        logging.info(f"Embeddings generated with shape: {embeddings.shape}")
    except Exception as e:
        logging.error(f"Failed during embedding generation: {e}")
        return False # <-- RETURN False on failure

    # --- STEP 4: Build and Save Vector Store ---
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    try:
        # Add vectors to FAISS index
        index.add(embeddings)
        logging.info(f"Added {index.ntotal} vectors to FAISS index.")
        
        # Save FAISS index to disk
        index_path = os.path.join(output_dir, vector_store_filename)
        faiss.write_index(index, index_path)
        logging.info(f"Saved FAISS index to {index_path}")
    except Exception as e:
        logging.error(f"Failed to build or save FAISS index: {e}")
        return False # <-- RETURN False on failure

    # --- STEP 5: Save Metadata ---
    metadata_path = os.path.join(output_dir, metadata_filename)
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(all_chunks_with_metadata):
                # Format data for storage
                data_to_save = {
                     "chunk_id": i,
                     "text": chunk['text'],
                     "metadata": chunk['metadata']
                }
                f.write(json.dumps(data_to_save) + '\n')
        logging.info(f"Saved metadata mapping to {metadata_path}")
    except Exception as e:
        logging.error(f"Failed to save metadata JSONL: {e}")
        return False # <-- Changed return to False
        
    logging.info("Knowledge base construction complete.")
    logging.info(f"FAISS index contains {index.ntotal} vectors.")
    logging.info(f"Metadata contains {len(all_chunks_with_metadata)} entries.")
    logging.info(f"Output saved to {output_dir}")
    return True # <-- RETURN True on full success

if __name__ == "__main__":
    # Basic dependency check
    try:
        import sentence_transformers
        import faiss
        import numpy
        import requests
        from omegaconf import OmegaConf # Add check for omegaconf
        # SpaCy is optional now
    except ImportError as e:
        print(f"Error: Missing dependency - {e}", file=sys.stderr)
        print("Please install required packages:", file=sys.stderr)
        print("pip install -r paper_population/requirements.txt", file=sys.stderr)
        sys.exit(1)

    # Config is already loaded at the top
    success = build_knowledge_base(cfg)
    if success:
        logging.info("Mathpix-based knowledge base build process finished successfully.")
        sys.exit(0)
    else:
        logging.error("Mathpix-based knowledge base build process failed.")
        sys.exit(1) # <-- Exit with non-zero code on failure 