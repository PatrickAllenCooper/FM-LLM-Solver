import argparse
import os
import sys
import re
import json
import logging
import time
import platform
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

# Third-party imports
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import spacy
import requests
from omegaconf import OmegaConf

# Local imports
from utils.config_loader import load_config, DEFAULT_CONFIG_PATH

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
# Document classifier for barrier certificate types
from knowledge_base.document_classifier import BarrierCertificateClassifier

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
    Builds knowledge base(s) from PDFs in the specified directory.
    Supports unified, discrete, or continuous barrier certificate knowledge bases.
    
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
    
    # Determine barrier certificate type and setup classifier
    barrier_cert_type = cfg.knowledge_base.get('barrier_certificate_type', 'unified')
    logging.info(f"Building knowledge base for barrier certificate type: {barrier_cert_type}")
    
    # Initialize classifier if needed
    classifier = None
    if barrier_cert_type in ['discrete', 'continuous'] or cfg.knowledge_base.classification.get('enable_auto_classification', False):
        try:
            classifier = BarrierCertificateClassifier(cfg)
            logging.info("Document classifier initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize document classifier: {e}")
            if barrier_cert_type in ['discrete', 'continuous']:
                logging.error("Classification is required for discrete/continuous modes. Aborting.")
                return False
            else:
                logging.warning("Classification disabled, using unified knowledge base")
    
    # Generate a hash of the pipeline configuration to detect changes
    config_hash = generate_pipeline_config_hash(cfg)
    
    # Get paper directory and check for papers
    paper_dir = cfg.paths.pdf_input_dir
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

    # Add additional information if running on Apple Silicon
    if hardware_info["is_apple_silicon"]:
        logging.info("Running on Apple Silicon - optimizing for M-series chip")
    if hardware_info["has_gpu"]:
        logging.info("GPU detected - CUDA optimizations available")

    # Determine which knowledge bases to build
    success = True
    
    if barrier_cert_type == "unified":
        # Build single unified knowledge base
        logging.info("Building unified knowledge base containing all documents")
        success = build_single_knowledge_base(cfg, "unified", pdf_files, classifier)
        
    elif barrier_cert_type == "discrete":
        # Build only discrete knowledge base
        logging.info("Building discrete barrier certificate knowledge base")
        success = build_single_knowledge_base(cfg, "discrete", pdf_files, classifier)
        
    elif barrier_cert_type == "continuous":
        # Build only continuous knowledge base
        logging.info("Building continuous barrier certificate knowledge base")
        success = build_single_knowledge_base(cfg, "continuous", pdf_files, classifier)
        
    else:
        logging.error(f"Unknown barrier certificate type: {barrier_cert_type}")
        return False
    
    if success:
        logging.info(f"{pipeline.capitalize()} knowledge base build process finished successfully.")
    else:
        logging.error("Knowledge base build process failed.")
        
    return success

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

def get_kb_paths_for_type(cfg, kb_type):
    """
    Get the appropriate knowledge base paths for a given type.
    
    Parameters
    ----------
    cfg : omegaconf.dictconfig.DictConfig
        Configuration object
    kb_type : str
        Knowledge base type ('unified', 'discrete', 'continuous')
        
    Returns
    -------
    Tuple[str, str, str]
        (output_dir, vector_store_filename, metadata_filename)
    """
    if kb_type == 'discrete':
        return (
            cfg.paths.kb_discrete_output_dir,
            cfg.paths.kb_discrete_vector_store_filename,
            cfg.paths.kb_discrete_metadata_filename
        )
    elif kb_type == 'continuous':
        return (
            cfg.paths.kb_continuous_output_dir,
            cfg.paths.kb_continuous_vector_store_filename,
            cfg.paths.kb_continuous_metadata_filename
        )
    else:  # unified or default
        return (
            cfg.paths.kb_output_dir,
            cfg.paths.kb_vector_store_filename,
            cfg.paths.kb_metadata_filename
        )

def build_single_knowledge_base(cfg, kb_type, pdf_files, classifier=None):
    """
    Build a single knowledge base for a specific type.
    
    Parameters
    ----------
    cfg : omegaconf.dictconfig.DictConfig
        Configuration object
    kb_type : str
        Knowledge base type ('unified', 'discrete', 'continuous')
    pdf_files : List[str]
        List of PDF file paths to process
    classifier : BarrierCertificateClassifier, optional
        Document classifier instance
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    logging.info(f"Building {kb_type} knowledge base with {len(pdf_files)} PDFs")
    
    # Get paths for this KB type
    output_dir, vector_store_filename, metadata_filename = get_kb_paths_for_type(cfg, kb_type)
    vector_index_path = os.path.join(output_dir, vector_store_filename)
    metadata_path = os.path.join(output_dir, metadata_filename)
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for existing knowledge base files
    config_hash = generate_pipeline_config_hash(cfg)
    config_changed = check_if_pipeline_config_changed(config_hash, output_dir)
    force_rebuild = config_changed
    
    if os.path.exists(vector_index_path) and os.path.exists(metadata_path) and not force_rebuild:
        logging.info(f"{kb_type.capitalize()} knowledge base files found. Skipping build.")
        return True
    
    # Save pipeline config hash
    save_pipeline_config_hash(config_hash, output_dir)
    
    # Filter PDFs based on classification if needed
    relevant_pdfs = []
    classification_results = []
    
    pipeline = cfg.knowledge_base.pipeline.lower()
    
    for pdf_path in pdf_files:
        include_pdf = True
        classification = "both"  # Default for unified
        confidence = 1.0
        details = {}
        
        # Classify document if classifier is available and we're not in unified mode
        if classifier and kb_type != "unified":
            try:
                # Extract text for classification
                if pipeline == "mathpix":
                    mmd_content = process_pdf_with_mathpix(pdf_path, MATHPIX_APP_ID, MATHPIX_APP_KEY)
                else:
                    mmd_content = process_pdf_open_source(pdf_path, cfg)
                
                if mmd_content:
                    # Split into chunks for classification
                    chunk_size = cfg.embeddings.chunk_size
                    overlap = cfg.embeddings.chunk_overlap
                    chunks = split_into_chunks(mmd_content, chunk_size, overlap)
                    
                    # Classify the document
                    classification, confidence, details = classifier.classify_chunks(chunks, pdf_path)
                    
                    # Apply stochastic filtering if enabled
                    stochastic_include = True
                    stochastic_reason = ""
                    stochastic_details = {}
                    
                    if cfg.knowledge_base.classification.stochastic_filter.get('enable', False):
                        stochastic_include, stochastic_reason, stochastic_details = classifier.should_include_document(
                            ' '.join(chunks), pdf_path
                        )
                        details['stochastic_filter'] = {
                            'include': stochastic_include,
                            'reason': stochastic_reason,
                            'details': stochastic_details
                        }
                    
                    # Determine if this PDF should be included in this KB
                    type_include = True
                    if kb_type == "discrete":
                        type_include = classification in ["discrete", "both"]
                    elif kb_type == "continuous":
                        type_include = classification in ["continuous", "both"]
                    
                    # Final inclusion decision: must pass both type and stochastic filters
                    include_pdf = type_include and stochastic_include
                    
                    # Store classification result
                    classification_results.append({
                        "pdf_path": pdf_path,
                        "classification": classification,
                        "confidence": confidence,
                        "details": details,
                        "included_in_kb": include_pdf,
                        "kb_type": kb_type
                    })
                else:
                    logging.warning(f"No content extracted from {os.path.basename(pdf_path)} for classification")
                    include_pdf = (kb_type == "unified")  # Only include in unified if no content
                    
            except Exception as e:
                logging.error(f"Error classifying {os.path.basename(pdf_path)}: {e}")
                include_pdf = (kb_type == "unified")  # Only include in unified if classification fails
        
        if include_pdf:
            relevant_pdfs.append(pdf_path)
            filter_info = f" | stochastic: {stochastic_reason}" if cfg.knowledge_base.classification.stochastic_filter.get('enable', False) else ""
            logging.info(f"Including {os.path.basename(pdf_path)} in {kb_type} KB (classification: {classification}, confidence: {confidence:.3f}{filter_info})")
        else:
            exclusion_reason = []
            if not type_include:
                exclusion_reason.append(f"type: {classification}")
            if not stochastic_include:
                exclusion_reason.append(f"stochastic: {stochastic_reason}")
            reason_str = " | ".join(exclusion_reason) if exclusion_reason else f"classification: {classification}"
            logging.info(f"Excluding {os.path.basename(pdf_path)} from {kb_type} KB ({reason_str})")
    
    # Save classification report
    if classification_results:
        report_path = os.path.join(output_dir, f"classification_report_{kb_type}.json")
        try:
            classifier.save_classification_report(classification_results, report_path)
        except Exception as e:
            logging.warning(f"Failed to save classification report: {e}")
    
    if not relevant_pdfs:
        logging.warning(f"No PDFs selected for {kb_type} knowledge base")
        return True  # Not an error, just no relevant documents
    
    logging.info(f"Processing {len(relevant_pdfs)} PDFs for {kb_type} knowledge base")
    
    # Now build the knowledge base with the filtered PDFs using the same logic as the original function
    # This is a simplified version - the full implementation would mirror the original processing logic
    metadata_list = []
    all_embeddings = []
    global_chunk_idx_counter = 0
    
    # Load embedding model
    embedding_model = cfg.embeddings.model_name
    logging.info(f"Loading embedding model: {embedding_model}")
    
    # Get memory optimization settings
    low_memory_mode = cfg.knowledge_base.get('low_memory_mode', False)
    gpu_memory_limit = cfg.knowledge_base.get('gpu_memory_limit', 0)
    batch_size = cfg.knowledge_base.get('embedding_batch_size', 32)
    
    if not batch_size:
        batch_size = cfg.embeddings.get('batch_size', 32)
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(embedding_model)
        # (Memory optimization code would go here - simplified for brevity)
    except ImportError:
        logging.error("Failed to load sentence-transformers")
        return False
    
    # Process PDFs (simplified processing loop)
    pdf_progress_bar = tqdm(relevant_pdfs, desc=f"Processing {kb_type} PDFs", disable=not tqdm_available)
    
    for i, pdf_path in enumerate(pdf_progress_bar):
        try:
            logging.info(f"Processing {kb_type} PDF {i+1}/{len(relevant_pdfs)}: {os.path.basename(pdf_path)}")
            
            # Extract content
            if pipeline == "mathpix":
                mmd_content = process_pdf_with_mathpix(pdf_path, MATHPIX_APP_ID, MATHPIX_APP_KEY)
            else:
                mmd_content = process_pdf_open_source(pdf_path, cfg)
            
            if not mmd_content:
                logging.warning(f"No content extracted from {os.path.basename(pdf_path)}")
                continue
            
            # Split into chunks
            chunk_size = cfg.embeddings.chunk_size
            overlap = cfg.embeddings.chunk_overlap
            chunks = split_into_chunks(mmd_content, chunk_size, overlap)
            
            # Create embeddings and metadata
            base_metadata = {
                "source": os.path.basename(pdf_path),
                "path": pdf_path,
                "kb_type": kb_type
            }
            
            current_chunks = []
            current_metadata = []
            
            for chunk_text in chunks:
                chunk_metadata = base_metadata.copy()
                chunk_metadata["chunk_id"] = global_chunk_idx_counter
                chunk_metadata["text"] = chunk_text
                current_metadata.append(chunk_metadata)
                current_chunks.append(chunk_text)
                global_chunk_idx_counter += 1
                
                # Process in batches
                if len(current_chunks) >= batch_size:
                    batch_embeddings = model.encode(current_chunks, batch_size=batch_size, show_progress_bar=False)
                    all_embeddings.extend(batch_embeddings)
                    metadata_list.extend(current_metadata)
                    current_chunks = []
                    current_metadata = []
            
            # Process remaining chunks
            if current_chunks:
                batch_embeddings = model.encode(current_chunks, batch_size=batch_size, show_progress_bar=False)
                all_embeddings.extend(batch_embeddings)
                metadata_list.extend(current_metadata)
                
        except Exception as e:
            logging.error(f"Error processing {os.path.basename(pdf_path)}: {e}")
            continue
    
    if not all_embeddings:
        logging.error(f"No embeddings generated for {kb_type} knowledge base")
        return False
    
    # Build FAISS index
    logging.info(f"Building FAISS index for {kb_type} knowledge base")
    try:
        import numpy as np
        import faiss
        
        embeddings_array = np.array(all_embeddings).astype(np.float32)
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        
        # Save index and metadata
        faiss.write_index(index, vector_index_path)
        
        with open(metadata_path, 'w') as f:
            for item in metadata_list:
                f.write(json.dumps(item) + '\n')
        
        logging.info(f"Successfully built {kb_type} knowledge base with {index.ntotal} vectors")
        return True
        
    except Exception as e:
        logging.error(f"Error building {kb_type} FAISS index: {e}")
        return False

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