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
    Format the LLM prompt with system description, RAG context, and domain bounds.
    
    Args:
        system_description: String describing the dynamical system
        context: RAG-retrieved context chunks (optional)
        kb_type: Type of knowledge base ("discrete", "continuous", "unified")
        domain_bounds: Dictionary of domain bounds like {"x": [-2, 2], "y": [-1, 1]} (optional)
    """
    state_vars_match = re.search(r"State Variables:\s*\[?([\w\s,]+)\]?", system_description, re.IGNORECASE)
    actual_state_vars_list = []
    if state_vars_match:
        extracted_vars = [v.strip() for v in state_vars_match.group(1).split(',') if v.strip()]
        if extracted_vars:
            actual_state_vars_list = extracted_vars
    
    if not actual_state_vars_list: # Fallback if regex fails or no vars found
        # Try to infer from common names if not explicitly listed - less reliable
        if 'x' in system_description.lower() and 'y' in system_description.lower() and 'z' in system_description.lower():
            actual_state_vars_list = ['x', 'y', 'z']
        elif 'x' in system_description.lower() and 'y' in system_description.lower():
            actual_state_vars_list = ['x', 'y']
        elif 'x' in system_description.lower():
            actual_state_vars_list = ['x']
        else:
            actual_state_vars_list = ['x'] # Default to x if nothing else

    state_vars_str_for_prompt = ", ".join(actual_state_vars_list)
    example_b_func_vars = state_vars_str_for_prompt
    example_b_expr_var = actual_state_vars_list[0] if actual_state_vars_list else "x"

    # Detect system type from dynamics notation
    is_discrete_time = bool(re.search(r'\w+\[k\+1\]|\w+\[k \+ 1\]|\w+\(k\+1\)|\w+\(k \+ 1\)', system_description))
    
    # Add barrier certificate type-specific guidance
    type_guidance = ""
    discrete_conditions = ""
    continuous_conditions = ""
    
    if is_discrete_time or kb_type == "discrete":
        type_guidance = (
            f"IMPORTANT: This is a DISCRETE-TIME system (x[k+1] notation). For discrete-time barrier certificates:\n"
            f"⚠️  CRITICAL MATHEMATICAL WARNING: State variables can be NEGATIVE! When checking ΔB ≤ 0:\n"
            f"   - For expressions like ΔB = -0.1x[k] - 0.1y[k]², do NOT assume x[k] ≥ 0!\n"
            f"   - If x[k] can be negative (from initial set), then -0.1x[k] can be POSITIVE\n"
            f"   - ALWAYS check the FULL RANGE of state variables from the initial set\n"
            f"   - Example: If initial set is x² + y² ≤ 0.25, then x ∈ [-0.5, 0.5] (includes negatives!)\n\n"
            f"Mathematical Requirements for Discrete Barriers:\n"
            f"- Use the discrete barrier condition: B(x[k+1], y[k+1]) - B(x[k], y[k]) ≤ 0 (non-increasing)\n"
            f"- Analyze state variable ranges from initial/safe sets before concluding ΔB ≤ 0\n"
            f"- Consider compensation terms for negative contributions\n"
            f"- For linear barriers B(x,y) = x - c, verify ΔB across the ENTIRE reachable set\n\n"
            f"Recommended Barrier Forms (with proper analysis):\n"
            f"  • Quadratic level sets: B(x,y) = x² + y² - C (always ΔB analysis needed)\n"
            f"  • Compensated linear: B(x,y) = x + αy² - C (α > 0 to handle negative x)\n"
            f"  • Mixed barrier: B(x,y) = x² + βx + γy² - C (quadratic dominance)\n"
            f"- Provide RIGOROUS mathematical verification of ΔB ≤ 0 over state ranges\n"
            f"- Consider worst-case scenarios for negative state variables\n\n"
        )
        discrete_conditions = (
            f"RIGOROUS VERIFICATION STEPS for discrete-time barrier B({example_b_func_vars}):\n\n"
            f"Step 1: ANALYZE STATE VARIABLE RANGES from initial/safe sets\n"
            f"   - Extract exact bounds: if x² + y² ≤ r², then x ∈ [-√r, √r], y ∈ [-√r, √r]\n"
            f"   - Identify if variables can be negative (critical for ΔB analysis)\n\n"
            f"Step 2: VERIFY INITIAL SET CONDITION\n"
            f"   - Check: B(x[0], y[0]) ≤ 0 for ALL points in initial set\n"
            f"   - Test extreme points and corners of initial set\n\n"
            f"Step 3: COMPUTE AND VERIFY DISCRETE DIFFERENCE\n"
            f"   - Calculate: ΔB = B(x[k+1], y[k+1]) - B(x[k], y[k]) using system dynamics\n"
            f"   - Substitute dynamics into barrier function carefully\n"
            f"   - Simplify algebraically to find ΔB expression\n\n"
            f"Step 4: CRITICAL - VERIFY ΔB ≤ 0 OVER FULL STATE RANGE\n"
            f"   - DO NOT assume variables are non-negative!\n"
            f"   - Test ΔB at extreme values: minimum and maximum of each variable\n"
            f"   - If ΔB has terms like -ax where x can be negative, check if -ax ≤ 0 always holds\n"
            f"   - Counterexample search: find any point where ΔB > 0\n\n"
            f"Step 5: VERIFY UNSAFE REGION SEPARATION\n"
            f"   - Check: B(x, y) ≥ 0 for points in unsafe region\n"
            f"   - Ensure barrier properly separates safe from unsafe\n\n"
            f"Mathematical Example of Common Error:\n"
            f"❌ WRONG: 'ΔB = -0.1x - 0.1y², since x ≥ 0 and y² ≥ 0, we have ΔB ≤ 0'\n"
            f"✅ CORRECT: 'ΔB = -0.1x - 0.1y². From initial set x² + y² ≤ 0.25, x ∈ [-0.5,0.5].\n"
            f"            When x = -0.3, ΔB = -0.1(-0.3) - 0.1y² = 0.03 - 0.1y² > 0 for small |y|. VIOLATION!'\n"
        )
    elif kb_type == "continuous" or not is_discrete_time:
        type_guidance = (
            f"IMPORTANT: This is a CONTINUOUS-TIME system (dx/dt notation). For continuous-time barrier certificates:\n"
            f"- Use the Lie derivative condition: dB/dt ≤ 0 in the safe set\n"
            f"- Consider energy-like functions and their time derivatives\n"
            f"- Focus on polynomial forms for SOS (Sum-of-Squares) verification\n"
            f"- Think about gradient and vector field alignment\n\n"
        )
        continuous_conditions = (
            f"For continuous-time systems, the barrier certificate B({example_b_func_vars}) must satisfy:\n"
            f"1. B(x, y) ≤ 0 for all initial states (safe in initial set)\n"
            f"2. dB/dt = ∇B · f(x) ≤ 0 in the safe set (non-increasing along trajectories)\n"
            f"3. B(x, y) ≥ 0 in the unsafe region (positive outside safe region)\n"
        )

    # Domain bounds guidance
    domain_guidance = ""
    if domain_bounds:
        domain_desc = ", ".join([f"{var} ∈ [{bounds[0]}, {bounds[1]}]" for var, bounds in domain_bounds.items()])
        domain_guidance = (
            f"🎯 DOMAIN BOUNDS CONSTRAINT: This barrier certificate must be valid within the specified domain:\n"
            f"   Domain: {domain_desc}\n"
            f"   - The certificate MUST satisfy all barrier conditions within this domain\n"
            f"   - Outside this domain, the certificate validity is not guaranteed\n"
            f"   - Choose coefficients and functional form appropriate for this bounded region\n"
            f"   - Consider domain boundaries when verifying barrier conditions\n\n"
        )

    instruction = (
        f"You are an expert in control theory and dynamical systems. Your task is to propose a barrier certificate for the given autonomous system.\n"
        f"The state variables for this system are: {state_vars_str_for_prompt}.\n\n"
        f"{domain_guidance}"
        f"{type_guidance}"
        f"Please follow these steps carefully:\n"
        f"1. Analyze the system dynamics, initial set, unsafe set, and safe set (if provided) from the 'System Description' below.\n"
        f"2. Consider any relevant context from the 'Relevant Context from Papers' (if provided) that might inspire a similar form or approach for the barrier certificate.\n"
        f"3. {'Consider the domain bounds constraint when selecting the barrier function form and coefficients.' if domain_bounds else ''}\n"
        f"4. Briefly explain your reasoning or the strategy you will use to propose a candidate barrier certificate function B({example_b_func_vars}). This reasoning should not contain the final certificate itself.\n"
        f"   ⚠️  For discrete-time systems: ANALYZE state variable ranges from initial set BEFORE assuming any non-negativity!\n"
        f"   {'🎯  Consider how the domain bounds affect your barrier function choice and verification.' if domain_bounds else ''}\n"
        f"5. After your reasoning, state the proposed barrier certificate function clearly and unambiguously using ONLY the specified state variables ({state_vars_str_for_prompt}). The function must be presented in the following exact format, on its own lines, without any surrounding text or explanations other than what is inside the B(...) notation:\n"
        f"BARRIER_CERTIFICATE_START\n"
        f"B({example_b_func_vars}) = <your_mathematical_expression_using_only_{state_vars_str_for_prompt}_and_constants>\n"
        f"BARRIER_CERTIFICATE_END\n\n"
        f"⚠️  CRITICAL REQUIREMENT: You MUST provide a CONCRETE mathematical expression with actual numerical coefficients. \n"
        f"❌ DO NOT use placeholder variables like 'a', 'b', 'c', 'd', 'e', 'f' or generic templates.\n"
        f"❌ DO NOT write: B(x, y) = ax**2 + bxy + cy**2 + dx + ey + f\n"
        f"❌ DO NOT write: B(x, y) = c1*x**2 + c2*y**2 + c3\n\n"
        f"✅ CORRECT examples with actual numbers:\n"
        f"BARRIER_CERTIFICATE_START\n"
        f"B(x, y) = x**2 + y**2 - 2.0\n"
        f"BARRIER_CERTIFICATE_END\n\n"
        f"BARRIER_CERTIFICATE_START\n"
        f"B(x, y) = x - 1.5\n"
        f"BARRIER_CERTIFICATE_END\n\n"
        f"BARRIER_CERTIFICATE_START\n"
        f"B(x, y) = 2*x**2 - 3*y + 1.5\n"
        f"BARRIER_CERTIFICATE_END\n\n"
        f"For discrete-time systems, properly analyzed examples:\n"
        f"• Quadratic barriers: B(x, y) = x**2 + y**2 - 2.0 (handles negative x/y gracefully)\n"  
        f"• Compensated barriers: B(x, y) = x + 0.5*y**2 - 1.5 (y² term compensates for negative x)\n"
        f"• Pure quadratic: B(x, y) = 2*x**2 + 3*y**2 - 1.0 (always non-negative quadratic terms)\n"
        f"⚠️  WARNING: Simple linear barriers like B(x,y) = x - 1.5 often FAIL for systems with negative state variables!\n\n"
        f"6. After stating the certificate in the specified block, briefly outline the conditions it must satisfy:\n"
        f"{discrete_conditions if (is_discrete_time or kb_type == 'discrete') else continuous_conditions}"
        f"{'   🎯  Note: All conditions must be verified within the specified domain bounds.' if domain_bounds else ''}"
    )

    if context:
        prompt = f"<s>[INST] {instruction}\n\nRelevant Context from Papers:\n{context}\n---\n\nSystem Description:\n{system_description} [/INST]"
    else:
        prompt = f"<s>[INST] {instruction}\n\nSystem Description:\n{system_description} [/INST]"

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
    
    # Initialize generation pipeline
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens_to_use,
        temperature=temp_to_use,
        top_p=TOP_P,
        do_sample=True if temp_to_use > 0 else False
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