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
import platform
import hashlib
from pathlib import Path
from omegaconf import OmegaConf
from typing import Dict, List, Optional

# Add tqdm for progress bars (with fallback to avoid dependency issues)
try:
    from tqdm import tqdm
    tqdm_available = True
except ImportError:
    tqdm_available = False
    # Define a simple no-op tqdm fallback
    class DummyTqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
            
        def __iter__(self):
            for item in self.iterable:
                yield item
                
        def update(self, n=1):
            pass
        
        def set_description(self, desc=""):
            pass
        
        def close(self):
            pass
    
    tqdm = DummyTqdm

# Alternative PDF processing pipeline
from knowledge_base.alternative_pdf_processor import process_pdf as process_pdf_open_source, detect_hardware, split_into_chunks

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
# We need argparse BEFORE loading config to allow overriding the config path
parser_init = argparse.ArgumentParser(add_help=False) # Initial parser for config path only
parser_init.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to the configuration YAML file.")
args_init, _ = parser_init.parse_known_args()
cfg = load_config(args_init.config)

# Determine which PDF processing pipeline to use: 'mathpix' or 'open_source'
PDF_PIPELINE = cfg.knowledge_base.pipeline

# Mathpix API Credentials (only required if pipeline is mathpix)
MATHPIX_APP_ID = None
MATHPIX_APP_KEY = None
if PDF_PIPELINE == "mathpix":
    MATHPIX_APP_ID = os.environ.get("MATHPIX_APP_ID")
    MATHPIX_APP_KEY = os.environ.get("MATHPIX_APP_KEY")
    if not MATHPIX_APP_ID or not MATHPIX_APP_KEY:
        logging.error("Mathpix API credentials (MATHPIX_APP_ID, MATHPIX_APP_KEY) not found in environment variables.")
        logging.error("Please set them before running with 'mathpix' pipeline. Exiting.")
        sys.exit(1)
    logging.info("Mathpix credentials found.")
else:
    logging.info("Using open-source PDF processing pipeline (no Mathpix credentials needed).")

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
    Builds a knowledge base from PDFs in the specified directory.
    
    Parameters
    ----------
    cfg : omegaconf.dictconfig.DictConfig
        Configuration object
        
    Returns
    -------
    bool
        True if the knowledge base was built successfully, False otherwise.
    """
    # Hardware detection for logging and optimization
    hardware_info = detect_hardware()
    
    # Determine the pipeline
    pipeline = cfg.knowledge_base.pipeline.lower()
    logging.info(f"Using '{pipeline}' PDF processing pipeline on {platform.machine()} architecture")
    
    # Generate a hash of the pipeline configuration to detect changes
    config_hash = generate_pipeline_config_hash(cfg)
    
    # Get paths
    paper_dir = cfg.paths.pdf_input_dir
    output_dir = cfg.paths.kb_output_dir
    vector_index_path = os.path.join(output_dir, cfg.paths.kb_vector_store_filename)
    metadata_path = os.path.join(output_dir, cfg.paths.kb_metadata_filename)
    
    # Check if config has changed to force rebuild
    config_changed = check_if_pipeline_config_changed(config_hash, output_dir)
    force_rebuild = config_changed

    # Add additional information if running on Apple Silicon
    if hardware_info["is_apple_silicon"]:
        logging.info("Running on Apple Silicon - optimizing for M-series chip")
    if hardware_info["has_gpu"]:
        logging.info("GPU detected - CUDA optimizations available")
    
    # Check for existing knowledge base files
    if os.path.exists(vector_index_path) and os.path.exists(metadata_path) and not force_rebuild:
        logging.info(f"Knowledge base files found ({os.path.basename(vector_index_path)}, {os.path.basename(metadata_path)}). Skipping build.")
        return True
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save pipeline config hash for future reference
    save_pipeline_config_hash(config_hash, output_dir)
    
    # Check if any papers are available
    if not os.path.exists(paper_dir) or not os.listdir(paper_dir):
        logging.warning(f"No paper PDFs found in {paper_dir}. Knowledge base build aborted.")
        return False

    # Get list of PDFs
    logging.info(f"Scanning for PDFs in {paper_dir}")
    pdf_files = []
    for root, dirs, files in os.walk(paper_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
                
    if not pdf_files:
        logging.warning("No PDF files found. Knowledge base build aborted.")
        return False
        
    logging.info(f"Found {len(pdf_files)} PDF files")
    
    # Process PDFs and build knowledge base
    metadata_list = []
    all_embeddings = []
    
    # Use SentenceTransformer for embeddings
    embedding_model = cfg.embeddings.model_name
    logging.info(f"Loading embedding model: {embedding_model}")
    
    # Get memory optimization settings
    low_memory_mode = cfg.knowledge_base.get('low_memory_mode', False)
    gpu_memory_limit = cfg.knowledge_base.get('gpu_memory_limit', 0)  # 0 means no limit
    batch_size = cfg.knowledge_base.get('embedding_batch_size', 32)
    
    # Also check in embeddings section
    if not batch_size:
        batch_size = cfg.embeddings.get('batch_size', 32)
    
    logging.info(f"Memory optimization settings: low_memory_mode={low_memory_mode}, gpu_memory_limit={gpu_memory_limit}MB, batch_size={batch_size}")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(embedding_model)
        
        # Memory management based on configuration
        use_gpu = False
        
        # Try to optimize for hardware if possible
        if hardware_info["is_apple_silicon"]:
            # Enable MPS if available for transformers
            logging.info("Attempting to enable Metal Performance Shaders for embeddings")
            try:
                import torch
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    if not low_memory_mode:
                        logging.info("MPS is available, using Apple Metal for embeddings")
                        model = model.to('mps')
                        use_gpu = True
                    else:
                        logging.info("Low memory mode enabled, using CPU for embeddings despite MPS availability")
            except (ImportError, AttributeError) as e:
                logging.warning(f"Could not enable MPS: {str(e)}")
        elif hardware_info["has_gpu"]:
            # Enable CUDA if available
            try:
                import torch
                if torch.cuda.is_available():
                    if not low_memory_mode:
                        logging.info("Using CUDA for embeddings")
                        
                        # Limit GPU memory usage if specified
                        if gpu_memory_limit > 0:
                            logging.info(f"Limiting GPU memory usage to {gpu_memory_limit}MB")
                            total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
                            fraction = (gpu_memory_limit * 1024 * 1024) / total_memory_bytes
                            if fraction < 0.01: # Safety check for extremely small fractions
                                logging.warning(f"Calculated GPU memory fraction {fraction} is very small. Clamping to 0.01 (1%).")
                                fraction = 0.01
                            elif fraction > 1.0: # Safety check if limit exceeds total
                                logging.warning(f"Calculated GPU memory fraction {fraction} > 1.0. Clamping to 1.0.")
                                fraction = 1.0
                            logging.info(f"Setting GPU memory fraction to: {fraction:.4f}")
                            torch.cuda.set_per_process_memory_fraction(fraction, 0) # Explicitly for device 0
                            
                        model = model.to('cuda')
                        use_gpu = True
                    else:
                        logging.info("Low memory mode enabled, using CPU for embeddings despite CUDA availability")
            except (ImportError, AttributeError) as e:
                logging.warning(f"Could not enable CUDA: {str(e)}")
    except ImportError:
        logging.error("Failed to load sentence-transformers. Please install with 'pip install sentence-transformers'")
        return False
    
    # Process each PDF with progress bar
    total_chunks_processed = 0
    total_chunks = 0  # Will be updated as we process
    
    # Create a progress bar for PDF processing
    if tqdm_available:
        logging.info("Progress bar enabled for processing")
    else:
        logging.warning("tqdm not installed - progress bars disabled. Install with 'pip install tqdm'")
    
    # Process each PDF with progress bar
    for i, pdf_path in enumerate(tqdm(pdf_files, desc="Processing PDFs", disable=not tqdm_available)):
        try:
            if tqdm_available:
                # When using tqdm, we don't need redundant logging for each file
                tqdm.write(f"Processing PDF {i+1}/{len(pdf_files)}: {os.path.basename(pdf_path)}")
            else:
                logging.info(f"Processing PDF {i+1}/{len(pdf_files)}: {os.path.basename(pdf_path)}")
            
            # Use the configured pipeline to extract text
            if pipeline == "mathpix":
                # Use existing Mathpix code
                mmd_content = process_pdf_with_mathpix(pdf_path, MATHPIX_APP_ID, MATHPIX_APP_KEY)
            elif pipeline == "open_source":
                # Use our alternative processor, passing config for memory optimization
                mmd_content = process_pdf_open_source(pdf_path, cfg)
            else:
                logging.error(f"Unknown pipeline: {pipeline}")
                continue
                
            if not mmd_content:
                logging.warning(f"No content extracted from {os.path.basename(pdf_path)}")
                continue
                
            # Split content into chunks
            chunk_size = cfg.embeddings.chunk_size
            overlap = cfg.embeddings.chunk_overlap
            chunks = split_into_chunks(mmd_content, chunk_size, overlap)
            
            if tqdm_available:
                tqdm.write(f"Generated {len(chunks)} chunks from PDF")
            else:
                logging.info(f"Generated {len(chunks)} chunks from PDF")
            
            # Update total chunks counter for overall progress
            total_chunks += len(chunks)
            
            # Create metadata for this PDF
            base_metadata = {
                "source": os.path.basename(pdf_path),
                "path": pdf_path
            }
            
            # ===== EMBEDDING GENERATION - CPU INTENSIVE PHASE =====
            # Tell user explicitly that we're now in the embedding phase
            if tqdm_available:
                tqdm.write(f"Starting embedding generation for {len(chunks)} chunks (slow on CPU - please wait)...")
                if low_memory_mode:
                    tqdm.write("Low memory mode is enabled, using CPU for embeddings (much slower than GPU)")
            else:
                logging.info(f"Starting embedding generation for {len(chunks)} chunks (slow on CPU - please wait)...")
                if low_memory_mode:
                    logging.info("Low memory mode is enabled, using CPU for embeddings (much slower than GPU)")
                
            # Process chunks in batches to manage memory
            current_chunks = []
            current_metadata = []
            
            # Create a progress bar for chunks in this PDF
            # Add a clear label to indicate this is the slow embedding phase
            chunk_progress = tqdm(
                total=len(chunks), 
                desc=f"Generating embeddings for {os.path.basename(pdf_path)} (CPU slow phase)", 
                disable=not tqdm_available,
                leave=False  # Don't leave the bar after completion to avoid cluttering the output
            )
            
            # Get timestamp for monitoring embedding time
            embedding_start_time = time.time()
            
            for chunk_id, chunk in enumerate(chunks):
                chunk_metadata = base_metadata.copy()
                chunk_metadata["chunk_id"] = chunk_id
                chunk_metadata["text"] = chunk
                current_metadata.append(chunk_metadata)
                current_chunks.append(chunk)
                
                # Process in batches
                if len(current_chunks) >= batch_size:
                    # Update description to show embedding progress clearly
                    batch_start_time = time.time()
                    chunk_progress.set_description(
                        f"Embedding batch {total_chunks_processed+1}-{total_chunks_processed+len(current_chunks)}/{total_chunks}"
                    )
                    
                    # Show a more detailed log when starting a batch (especially important on CPU)
                    if tqdm_available and (total_chunks_processed == 0 or total_chunks_processed % 100 == 0):
                        tqdm.write(f"Processing embedding batch {total_chunks_processed+1}-{total_chunks_processed+len(current_chunks)} of {total_chunks}...")
                    
                    # Create embeddings for the batch (with show_progress_bar=False to avoid nested bars)
                    batch_embeddings = model.encode(
                        current_chunks, 
                        batch_size=batch_size, 
                        show_progress_bar=False
                    )
                    
                    # Report how long the batch took (helpful for CPU timing)
                    batch_end_time = time.time()
                    batch_duration = batch_end_time - batch_start_time
                    if tqdm_available and (total_chunks_processed == 0 or total_chunks_processed % 100 == 0):
                        tqdm.write(f"Batch embedding took {batch_duration:.2f} seconds ({batch_duration/len(current_chunks):.2f} sec/chunk)")
                    
                    # Add embeddings and metadata to lists
                    all_embeddings.extend(batch_embeddings)
                    metadata_list.extend(current_metadata)
                    
                    # Update chunk progress
                    chunk_progress.update(len(current_chunks))
                    total_chunks_processed += len(current_chunks)
                    
                    # Clear batch
                    current_chunks = []
                    current_metadata = []
                    
                    # Clear GPU cache if using GPU
                    if use_gpu:
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except:
                            pass
            
            # Process remaining chunks
            if current_chunks:
                # Update description for final chunks
                batch_start_time = time.time()
                chunk_progress.set_description(
                    f"Embedding final chunks {total_chunks_processed+1}-{total_chunks_processed+len(current_chunks)}/{total_chunks}"
                )
                
                batch_embeddings = model.encode(current_chunks, batch_size=batch_size, show_progress_bar=False)
                
                # Report how long the batch took
                batch_end_time = time.time()
                batch_duration = batch_end_time - batch_start_time
                if tqdm_available:
                    tqdm.write(f"Final batch embedding took {batch_duration:.2f} seconds ({batch_duration/len(current_chunks):.2f} sec/chunk)")
                
                all_embeddings.extend(batch_embeddings)
                metadata_list.extend(current_metadata)
                
                # Update progress
                chunk_progress.update(len(current_chunks))
                total_chunks_processed += len(current_chunks)
                
                # Clear GPU cache if using GPU
                if use_gpu:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except:
                        pass
            
            # Close the chunk progress bar
            chunk_progress.close()
            
            # Report total embedding time for this PDF
            embedding_end_time = time.time()
            embedding_duration = embedding_end_time - embedding_start_time
            if tqdm_available:
                tqdm.write(f"Finished embedding all chunks for PDF {i+1}/{len(pdf_files)} in {embedding_duration:.2f} seconds")
                # Update the main progress bar now that the PDF is fully processed
                if hasattr(tqdm, 'refresh'):
                    tqdm.refresh()  # Force refresh of the main progress bar
            else:
                logging.info(f"Finished embedding all chunks for PDF {i+1}/{len(pdf_files)} in {embedding_duration:.2f} seconds")
                
        except Exception as e:
            logging.error(f"Error processing {os.path.basename(pdf_path)}: {str(e)}")
            continue
            
    # Check if we have any successful embeddings
    if not all_embeddings:
        logging.error("No embeddings were generated. Knowledge base build failed.")
        return False
        
    # Build FAISS index
    logging.info("Building FAISS index")
    try:
        import numpy as np
        import faiss
        
        # Convert embeddings to numpy array
        logging.info(f"Converting {len(all_embeddings)} embeddings to numpy array")
        embeddings_array = np.array(all_embeddings).astype(np.float32)
        
        # Create and train the index with progress indication
        dimension = embeddings_array.shape[1]
        logging.info(f"Creating FAISS index with dimension {dimension}")
        index = faiss.IndexFlatL2(dimension)
        
        logging.info("Adding vectors to FAISS index")
        index_progress = tqdm(total=len(embeddings_array), desc="Adding vectors to index", disable=not tqdm_available)
        
        # Add vectors in batches to show progress
        faiss_batch_size = 10000  # A reasonable batch size for FAISS
        for i in range(0, len(embeddings_array), faiss_batch_size):
            batch = embeddings_array[i:i+faiss_batch_size]
            index.add(batch)
            index_progress.update(len(batch))
        
        index_progress.close()
        
        # Save the index
        logging.info(f"Saving FAISS index to {vector_index_path}")
        faiss.write_index(index, vector_index_path)
        logging.info(f"Saved FAISS index with {index.ntotal} vectors")
        
        # Save metadata
        logging.info(f"Saving metadata to {metadata_path}")
        with open(metadata_path, 'w') as f:
            for item in metadata_list:
                f.write(json.dumps(item) + '\n')
        logging.info(f"Saved metadata for {len(metadata_list)} chunks")
        
    except Exception as e:
        logging.error(f"Error building FAISS index: {str(e)}")
        return False
        
    logging.info(f"{pipeline.capitalize()} knowledge base build process finished successfully.")
    return True

def generate_pipeline_config_hash(cfg):
    """Generate a hash of the pipeline configuration to detect changes"""
    # Extract the relevant configuration parameters
    config_str = f"{cfg.knowledge_base.pipeline}:{cfg.embeddings.model_name}:{cfg.embeddings.chunk_size}:{cfg.embeddings.chunk_overlap}"
    
    # Generate a hash
    return hashlib.md5(config_str.encode('utf-8')).hexdigest()

def check_if_pipeline_config_changed(new_hash, output_dir):
    """Check if the pipeline configuration has changed"""
    hash_file = os.path.join(output_dir, "pipeline_config.hash")
    
    if not os.path.exists(hash_file):
        return True
        
    try:
        with open(hash_file, 'r') as f:
            old_hash = f.read().strip()
            return old_hash != new_hash
    except:
        return True

def save_pipeline_config_hash(config_hash, output_dir):
    """Save the pipeline configuration hash"""
    os.makedirs(output_dir, exist_ok=True)
    hash_file = os.path.join(output_dir, "pipeline_config.hash")
    
    try:
        with open(hash_file, 'w') as f:
            f.write(config_hash)
    except Exception as e:
        logging.warning(f"Could not save pipeline config hash: {str(e)}")

if __name__ == "__main__":
    # Basic dependency check
    try:
        import sentence_transformers
        import faiss
        import numpy
        import requests
        from omegaconf import OmegaConf
        # SpaCy is optional now
    except ImportError as e:
        print(f"Error: Missing dependency - {e}", file=sys.stderr)
        print("Please install required packages:", file=sys.stderr)
        print("pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    # Config is already loaded at the top
    success = build_knowledge_base(cfg)
    if success:
        logging.info(f"{cfg.knowledge_base.pipeline.capitalize()}-based knowledge base build process finished successfully.")
    else:
        logging.error("Knowledge base building process failed.")
        sys.exit(1) 