import os
import argparse
import json
import faiss
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    logging,
)
from peft import PeftModel # To load LoRA adapter
from sentence_transformers import SentenceTransformer
import warnings
import sys # Import sys
import re

# Add project root to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

# Now we can import the utils module
from utils.config_loader import load_config, DEFAULT_CONFIG_PATH # Import config loader
from knowledge_base.kb_utils import get_active_kb_paths, determine_kb_type_from_config, validate_kb_config
from omegaconf import OmegaConf

# --- Configuration ---
warnings.filterwarnings("ignore")
# Reduce transformers logging verbosity
import logging as py_logging
py_logging.basicConfig(level=py_logging.INFO)
logging.set_verbosity_error()

# Load configuration early to allow overrides via CLI if needed
parser_init = argparse.ArgumentParser(add_help=False)
parser_init.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to the configuration YAML file.")
args_init, _ = parser_init.parse_known_args()
cfg = load_config(args_init.config)

# Validate configuration
if not validate_kb_config(cfg):
    py_logging.error("Invalid knowledge base configuration. Please check your config file.")
    sys.exit(1)

# Determine knowledge base type and get appropriate paths
kb_type = determine_kb_type_from_config(cfg)
kb_output_dir, kb_vector_path, kb_metadata_path = get_active_kb_paths(cfg)

# Extract filenames from full paths for compatibility
KB_DIR = kb_output_dir
VECTOR_STORE_FILENAME = os.path.basename(kb_vector_path)
METADATA_FILENAME = os.path.basename(kb_metadata_path)

py_logging.info(f"Using {kb_type} barrier certificate knowledge base")
py_logging.info(f"Knowledge base directory: {KB_DIR}")

# Get config values (Models will be derived from loaded cfg)
EMBEDDING_MODEL_NAME = cfg.knowledge_base.embedding_model_name
BASE_MODEL_NAME = cfg.fine_tuning.base_model_name
ADAPTER_PATH = os.path.join(cfg.paths.ft_output_dir, "final_adapter") # Construct adapter path

# RAG/Generation Parameters from config
NUM_CONTEXT_CHUNKS = cfg.inference.rag_k
MAX_NEW_TOKENS = cfg.inference.max_new_tokens
TEMPERATURE = cfg.inference.temperature
TOP_P = cfg.inference.top_p

# --- Helper Functions ---

def load_knowledge_base(kb_dir, index_filename, metadata_filename):
    """Loads the FAISS index and metadata map."""
    index_path = os.path.join(kb_dir, index_filename)
    metadata_path = os.path.join(kb_dir, metadata_filename)

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        py_logging.error(f"Knowledge base files not found in {kb_dir}. Run knowledge_base_builder.py first.")
        return None, None
    try:
        py_logging.info(f"Loading FAISS index from {index_path}...")
        index = faiss.read_index(index_path)
        py_logging.info(f"Loading metadata from {metadata_path} (JSONL)...")
        metadata_map = {}
        lines_read = 0
        lines_processed_into_map = 0
        lines_missing_chunk_id = 0
        lines_json_decode_error = 0

        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                lines_read += 1
                if line.strip():
                    try:
                        data = json.loads(line)
                        chunk_id = data.get('chunk_id')
                        if chunk_id is not None: # chunk_id = 0 is valid
                            metadata_map[chunk_id] = data 
                            lines_processed_into_map += 1
                        else:
                            lines_missing_chunk_id += 1
                            if lines_missing_chunk_id <= 10: # Log first few occurrences
                                py_logging.warning(f"Metadata line {line_num + 1} missing 'chunk_id': {line.strip()}")
                    except json.JSONDecodeError as e_json:
                        lines_json_decode_error += 1
                        if lines_json_decode_error <= 10: # Log first few occurrences
                            py_logging.warning(f"Error decoding metadata JSONL line {line_num + 1} in {metadata_path}: {e_json}. Line content: '{line.strip()}'")
                 
        py_logging.info(f"Metadata loading: Total lines read: {lines_read}, Lines processed into map: {lines_processed_into_map}, Lines missing chunk_id: {lines_missing_chunk_id}, Lines with JSON decode error: {lines_json_decode_error}.")
        py_logging.info(f"Knowledge base loaded: {index.ntotal} vectors, {len(metadata_map)} metadata entries.")
        if index.ntotal != len(metadata_map):
             py_logging.warning(f"Mismatch between index size ({index.ntotal}) and metadata size ({len(metadata_map)}).")
        return index, metadata_map
    except json.JSONDecodeError as e:
        py_logging.error(f"Error decoding metadata JSONL line in {metadata_path}: {e}")
        return None, None
    except Exception as e:
        py_logging.error(f"Failed to load knowledge base: {e}")
        return None, None

def load_finetuned_model(base_model_name, adapter_path, cfg):
    """Loads the base model with quantization and merges the LoRA adapter, uses config."""
    # Configure quantization from config
    quant_cfg = cfg.fine_tuning.quantization
    compute_dtype = getattr(torch, quant_cfg.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.use_4bit,
        bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_cfg.use_nested_quant,
    )
    # device_map_setting = cfg.fine_tuning.training.get("device_map", {"": 0})
    device_map_setting = {"": 0} # Simple default for inference

    try:
        py_logging.info(f"Loading base model: {base_model_name}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map=device_map_setting, # Use setting
            trust_remote_code=True
        )
        py_logging.info("Base model loaded.")

        # Check if we should skip the adapter (for base model only comparison)
        use_adapter = cfg.fine_tuning.get('use_adapter', True)
        if not use_adapter:
            py_logging.info("Skipping adapter loading - using base model only as configured.")
            model = base_model
        else:
            py_logging.info(f"Loading adapter from: {adapter_path}...")
            # Check if adapter exists before loading
            if not os.path.isdir(adapter_path):
                py_logging.error(f"Adapter directory not found at {adapter_path}. Ensure fine-tuning completed and path is correct.")
                return None, None
            model = PeftModel.from_pretrained(base_model, adapter_path)
            py_logging.info("Adapter loaded.")

            # Merge adapter
            py_logging.info("Merging adapter into base model...")
            model = model.merge_and_unload()
            py_logging.info("Adapter merged.")

        py_logging.info(f"Loading tokenizer for {base_model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        py_logging.info("Tokenizer loaded.")

        return model, tokenizer

    except FileNotFoundError: # Should be caught by the isdir check now
        py_logging.error(f"Adapter not found at {adapter_path}. Did fine-tuning complete successfully?")
        return None, None
    except Exception as e:
        py_logging.error(f"Failed to load fine-tuned model or tokenizer: {e}", exc_info=True)
        return None, None

def retrieve_context(query, embedding_model, index, metadata_map, k):
    """
    Retrieves top-k relevant text chunks from the knowledge base.
    
    This function performs semantic search using the following steps:
    1. Encodes the query into an embedding vector using the provided model
    2. Searches the FAISS index for the k nearest neighbors to the query embedding
    3. Retrieves and formats the corresponding text chunks and their metadata
    
    Parameters
    ----------
    query : str
        The user's query or system description
    embedding_model : SentenceTransformer
        Model to convert the query into an embedding vector
    index : faiss.Index
        FAISS index containing embeddings of the knowledge base chunks
    metadata_map : dict
        Dictionary mapping chunk indices to their content and metadata
    k : int
        Number of relevant chunks to retrieve
        
    Returns
    -------
    str
        Formatted context string containing k relevant chunks with metadata
        or empty string if retrieval fails
    """
    # Skip retrieval if k=0 (RAG disabled)
    if k <= 0:
        py_logging.info("Context retrieval skipped (k=0, RAG disabled)")
        return ""
        
    if index is None or metadata_map is None:
        py_logging.error("Cannot retrieve context: Knowledge base index or metadata is missing")
        return ""
    
    try:
        # Encode the query using the embedding model
        query_embedding = embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search for nearest neighbors
        distances, indices = index.search(query_embedding, k)
        
        # Format the results
        context = ""
        py_logging.info(f"Retrieved {len([i for i in indices[0] if i != -1])} relevant chunks")
        py_logging.debug(f"Chunk indices: {indices[0]}, distances: {distances[0]}")
        
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # FAISS returns -1 if fewer than k results are found
                continue
                
            chunk_data = metadata_map.get(idx)
            if not chunk_data:
                py_logging.warning(f"Metadata not found for retrieved FAISS index {idx} (type: {type(idx)}).") 
                continue
                
            # Extract metadata
            meta = chunk_data.get('metadata', {})
            source = meta.get('source', 'N/A')
            pages = meta.get('pages', 'N/A')
            
            # Format chunk with metadata
            context += f"--- Context Chunk {i+1} (Source: {source}, Pages: {pages}) ---\n"
            context += chunk_data.get('text', '[Error retrieving text]') + "\n\n"
            
        return context.strip()
        
    except Exception as e:
        py_logging.error(f"Error during context retrieval: {str(e)}")
        py_logging.debug("Exception details:", exc_info=True)
        return ""

def format_prompt_with_context(system_description, context, kb_type="unified", domain_bounds=None):
    """
    OPTIMIZED FORMAT: Concise prompt that generates correct barrier certificates.
    
    Args:
        system_description: String describing the dynamical system
        context: RAG-retrieved context chunks (optional)
        kb_type: Type of knowledge base ("discrete", "continuous", "unified")
        domain_bounds: Dictionary of domain bounds like {"x": [-2, 2], "y": [-1, 1]} (optional)
    """
    # Infer state variables (simplified)
    if 'x' in system_description.lower() and 'y' in system_description.lower() and 'z' in system_description.lower():
        state_vars = ['x', 'y', 'z']
    elif 'x' in system_description.lower() and 'y' in system_description.lower():
        state_vars = ['x', 'y']
    elif 'x' in system_description.lower():
        state_vars = ['x']
    else:
        state_vars = ['x', 'y']  # Default
    
    var_string = ", ".join(state_vars)
    
    # Domain bounds text
    domain_text = ""
    if domain_bounds:
        domain_desc = ", ".join([f"{var} ∈ [{bounds[0]}, {bounds[1]}]" for var, bounds in domain_bounds.items()])
        domain_text = f"Domain: {domain_desc}\n"
    
    # System type detection
    is_discrete = bool(re.search(r'\w+\[k\+1\]|\w+\[k \+ 1\]|\w+\(k\+1\)|\w+\(k \+ 1\)', system_description))
    
    # Extract bounds from system description for better constant selection
    initial_bound = None
    unsafe_bound = None
    
    # Try to extract numerical bounds
    initial_match = re.search(r'Initial Set:.*?<=\s*([\d\.]+)', system_description)
    unsafe_match = re.search(r'Unsafe Set:.*?>=\s*([\d\.]+)', system_description)
    
    if initial_match:
        initial_bound = float(initial_match.group(1))
    if unsafe_match:
        unsafe_bound = float(unsafe_match.group(1))
    
    # Generate guidance based on bounds
    bounds_guidance = ""
    if initial_bound is not None and unsafe_bound is not None:
        # For quadratic barriers B = x² + y² - c, choose c between initial and unsafe bounds
        suggested_c = (initial_bound + unsafe_bound) / 2
        bounds_guidance = f"""
MATHEMATICAL GUIDANCE for this system:
- Initial set bound: {initial_bound}
- Unsafe set bound: {unsafe_bound}  
- For quadratic barriers B(x,y) = x² + y² - c, choose c between {initial_bound} and {unsafe_bound}
- Suggested: c = {suggested_c:.1f} giving B(x,y) = x**2 + y**2 - {suggested_c:.1f}
"""

    # Core instruction (concise)
    instruction = f"""Generate a barrier certificate for:

{system_description}
{domain_text}
CRITICAL REQUIREMENTS:
- Use ONLY concrete numbers (1.0, -2.5, 3.14, etc.)
- NO placeholder variables (no α, β, γ, C, a, b, c, etc.)
- Complete the mathematical expression fully
- Format: B({var_string}) = <expression>
{bounds_guidance}
Examples of CORRECT outputs:
- B({var_string}) = x**2 + y**2 - 1.0
- B({var_string}) = 2*x**2 + 0.5*y**2 - 3.0
- B({var_string}) = x - 1.5

BARRIER_CERTIFICATE_START
B({var_string}) ="""

    # Include context if available
    if context:
        prompt = f"<s>[INST] {instruction}\n\nRelevant research context:\n{context[:500]}...\n\n[/INST]"
    else:
        prompt = f"<s>[INST] {instruction} [/INST]"
    
    return prompt

# --- Main Execution --- #

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate barrier certificate using RAG + Fine-tuned LLM.", parents=[parser_init]) # Inherit --config flag
    parser.add_argument("system_description", type=str,
                        help="Text description of the autonomous system (dynamics, constraints, sets). Put in quotes.")
    # Make other args optional overrides
    parser.add_argument("-k", type=int, default=None,
                        help=f"Override number of context chunks to retrieve (default: {NUM_CONTEXT_CHUNKS} from config).")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help=f"Override maximum new tokens for generation (default: {MAX_NEW_TOKENS} from config).")
    parser.add_argument("--temp", type=float, default=None,
                        help=f"Override generation temperature (default: {TEMPERATURE} from config).")

    args = parser.parse_args()

    # --- Determine final parameters (Config + CLI Overrides) ---
    k_to_use = args.k if args.k is not None else NUM_CONTEXT_CHUNKS
    max_tokens_to_use = args.max_tokens if args.max_tokens is not None else MAX_NEW_TOKENS
    temp_to_use = args.temp if args.temp is not None else TEMPERATURE
    base_model_to_use = BASE_MODEL_NAME
    adapter_to_use = ADAPTER_PATH
    kb_dir_to_use = KB_DIR
    embedding_model_to_use = EMBEDDING_MODEL_NAME
    vector_store_to_use = VECTOR_STORE_FILENAME
    metadata_to_use = METADATA_FILENAME

    print("=" * 60)
    print("BARRIER CERTIFICATE GENERATION PIPELINE")
    print("=" * 60)
    print(f"Config file: {args.config}")
    print(f"Parameters: k={k_to_use}, max_tokens={max_tokens_to_use}, temp={temp_to_use}")
    print(f"Base model: {base_model_to_use}")
    print(f"Knowledge base: {kb_dir_to_use}")
    print("-" * 60)

    # --- STEP 1: Load Knowledge Base ---
    print("\n[1/5] Loading Knowledge Base...")
    faiss_index, metadata = load_knowledge_base(kb_dir_to_use, vector_store_to_use, metadata_to_use)
    if faiss_index is None or metadata is None:
        print("ERROR: Failed to load knowledge base. Exiting.")
        sys.exit(1)
    print(f"Knowledge base loaded with {faiss_index.ntotal} documents.")

    # --- STEP 2: Load Embedding Model ---
    print("\n[2/5] Loading Embedding Model...")
    try:
        embed_model = SentenceTransformer(embedding_model_to_use)
        print(f"Embedding model '{embedding_model_to_use}' loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load embedding model: {e}")
        sys.exit(1)

    # --- STEP 3: Load Fine-tuned LLM ---
    print("\n[3/5] Loading Fine-tuned LLM...")
    model, tokenizer = load_finetuned_model(base_model_to_use, adapter_to_use, cfg)
    if model is None or tokenizer is None:
        print("ERROR: Failed to load model or tokenizer. Exiting.")
        sys.exit(1)
    print(f"Model and tokenizer loaded successfully.")
    
    # Initialize optimized generation pipeline
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens_to_use,
        temperature=temp_to_use,
        top_p=TOP_P,
        do_sample=True if temp_to_use > 0 else False,
        repetition_penalty=1.1,  # Prevent repetitive output
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # --- STEP 4: Retrieve Context (RAG) ---
    print(f"\n[4/5] Retrieving Context (top-{k_to_use} chunks)...")
    context = retrieve_context(args.system_description, embed_model, faiss_index, metadata, k_to_use)
    
    if context:
        print(f"Retrieved {context.count('--- Context Chunk')} relevant context chunks.")
        print("\nContext Preview:")
        # Print a preview (first 300 chars)
        context_preview = context[:300] + "..." if len(context) > 300 else context
        print(context_preview)
    else:
        print("No relevant context retrieved or knowledge base is empty.")

    # --- STEP 5: Generate Certificate ---
    print(f"\n[5/5] Generating {kb_type.capitalize()} Barrier Certificate...")
    prompt = format_prompt_with_context(args.system_description, context, kb_type)
    
    try:
        # Generate text
        result = pipe(prompt)
        generated_text = result[0]['generated_text']

        # Extract only the generated part after the prompt
        prompt_end_marker = "[/INST]"
        output_start_index = generated_text.find(prompt_end_marker)
        if output_start_index != -1:
             llm_output = generated_text[output_start_index + len(prompt_end_marker):].strip()
        else:
             # Fallback if marker not found
             llm_output = generated_text

        print("\n" + "=" * 60)
        print("GENERATED BARRIER CERTIFICATE")
        print("=" * 60)
        print(llm_output)
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: Text generation failed: {e}")
        sys.exit(1)

    print("\nGeneration complete! You may want to verify this certificate using: ")
    print("python evaluation/verify_certificate.py") 