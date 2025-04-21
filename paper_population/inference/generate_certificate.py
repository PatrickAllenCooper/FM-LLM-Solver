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
from paper_population.utils.config_loader import load_config, DEFAULT_CONFIG_PATH # Import config loader
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

# Get config values (Paths and Models will be derived from loaded cfg)
KB_DIR = cfg.paths.kb_output_dir
VECTOR_STORE_FILENAME = cfg.paths.kb_vector_store_filename
METADATA_FILENAME = cfg.paths.kb_metadata_filename
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
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                 if line.strip():
                     data = json.loads(line)
                     chunk_id = data.get('chunk_id')
                     if chunk_id is not None:
                         metadata_map[chunk_id] = data # Store full object
                     else:
                         py_logging.warning(f"Skipping metadata line without 'chunk_id': {line.strip()}")

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
    """Retrieves top-k relevant text chunks from the knowledge base."""
    if index is None or metadata_map is None:
        return ""
    try:
        query_embedding = embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        distances, indices = index.search(query_embedding, k)

        context = ""
        py_logging.info(f"Retrieved chunk indices: {indices[0]} distances: {distances[0]}")
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                chunk_data = metadata_map.get(idx) # Get full chunk object
                if chunk_data:
                    meta = chunk_data.get('metadata', {}) # Get nested metadata
                    context += f"--- Context Chunk {i+1} (Source: {meta.get('source', 'N/A')}, Pages: {meta.get('pages', 'N/A')}) ---\n"
                    context += chunk_data.get('text', '[Error retrieving text]') + "\n\n"
                else:
                    py_logging.warning(f"Metadata object not found for retrieved index {idx}.")
        return context.strip()
    except Exception as e:
        py_logging.error(f"Error during context retrieval: {e}")
        return ""

def format_prompt_with_context(system_description, context):
    """Formats the prompt for the LLM, including the retrieved context."""
    # This template should ideally match the one used implicitly or explicitly
    # during fine-tuning (e.g., the one in formatting_prompts_func).
    # Adapt if your fine-tuning used a different structure.
    instruction = ("Given the autonomous system described below and potentially relevant context "
                   "from research papers, propose a suitable barrier certificate function B(x) "
                   "and briefly outline the conditions it must satisfy.")

    if context:
        prompt = f"<s>[INST] {instruction}\n\nRelevant Context from Papers:\n{context}\n---\n\nSystem Description:\n{system_description} [/INST]" # Llama 3 format
    else:
        prompt = f"<s>[INST] {instruction}\n\nSystem Description:\n{system_description} [/INST]" # Llama 3 format

    return prompt

# --- Main Execution --- #

if __name__ == "__main__":
    # Keep system_description as mandatory runtime argument
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
    # Add overrides for model/adapter/kb paths if needed, though config is preferred
    # parser.add_argument("--base_model", type=str, help="Override base model name.")
    # parser.add_argument("--adapter", type=str, help="Override adapter path.")
    # parser.add_argument("--kb_dir", type=str, help="Override knowledge base directory.")

    args = parser.parse_args()

    # --- Determine final parameters (Config + CLI Overrides) ---
    k_to_use = args.k if args.k is not None else NUM_CONTEXT_CHUNKS
    max_tokens_to_use = args.max_tokens if args.max_tokens is not None else MAX_NEW_TOKENS
    temp_to_use = args.temp if args.temp is not None else TEMPERATURE
    # Example override checks (if added to argparse):
    # base_model_to_use = args.base_model if args.base_model else BASE_MODEL_NAME
    # adapter_to_use = args.adapter if args.adapter else ADAPTER_PATH
    # kb_dir_to_use = args.kb_dir if args.kb_dir else KB_DIR
    base_model_to_use = BASE_MODEL_NAME
    adapter_to_use = ADAPTER_PATH
    kb_dir_to_use = KB_DIR
    embedding_model_to_use = EMBEDDING_MODEL_NAME
    vector_store_to_use = VECTOR_STORE_FILENAME
    metadata_to_use = METADATA_FILENAME

    print("--- Initializing RAG + Fine-tuned LLM Pipeline ---")
    print(f"Using Config: {args.config}")
    print(f"Runtime Params: k={k_to_use}, max_tokens={max_tokens_to_use}, temp={temp_to_use}")
    print(f"Paths/Models: KB={kb_dir_to_use}, Base={base_model_to_use}, Adapter={adapter_to_use}")

    # 1. Load Knowledge Base Components
    print(f"Loading Knowledge Base from {kb_dir_to_use}...")
    faiss_index, metadata = load_knowledge_base(kb_dir_to_use, vector_store_to_use, metadata_to_use)
    if faiss_index is None or metadata is None:
        sys.exit(1)

    print(f"Loading Embedding Model: {embedding_model_to_use}...")
    try:
        embed_model = SentenceTransformer(embedding_model_to_use)
    except Exception as e:
        print(f"Error loading embedding model '{embedding_model_to_use}': {e}")
        sys.exit(1)

    # 2. Load Fine-tuned LLM
    print(f"Loading Fine-tuned LLM (Base: {base_model_to_use}, Adapter: {adapter_to_use})...")
    # Pass the loaded config object (cfg) to the model loader
    model, tokenizer = load_finetuned_model(base_model_to_use, adapter_to_use, cfg)
    if model is None or tokenizer is None:
        sys.exit(1)

    # 3. Retrieve Context (RAG)
    print(f"Retrieving top-{k_to_use} context chunks for the query...")
    context = retrieve_context(args.system_description, embed_model, faiss_index, metadata, k_to_use)
    if context:
        print("--- Retrieved Context ---")
        print(context)
        print("-----------------------")
    else:
        print("No relevant context retrieved or KB is empty.")

    # 4. Format Prompt
    prompt = format_prompt_with_context(args.system_description, context)
    print("--- Formatted Prompt for LLM ---")
    print(prompt)
    print("------------------------------")

    # 5. Generate Certificate
    print("Generating barrier certificate candidate...")
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens_to_use, # Use determined value
        temperature=temp_to_use,          # Use determined value
        top_p=TOP_P,                      # Use value from config
        do_sample=True if temp_to_use > 0 else False # Enable sampling if temp > 0
    )

    try:
        result = pipe(prompt)
        generated_text = result[0]['generated_text']

        # Extract only the generated part after the prompt
        # Find the end of the prompt marker [/INST]
        prompt_end_marker = "[/INST]"
        output_start_index = generated_text.find(prompt_end_marker)
        if output_start_index != -1:
             llm_output = generated_text[output_start_index + len(prompt_end_marker):].strip()
        else:
             # Fallback if marker not found (shouldn't happen with this prompt format)
             llm_output = generated_text # Or handle differently

        print("\n--- Generated Output (Barrier Certificate Candidate) ---")
        print(llm_output)
        print("--------------------------------------------------------")

    except Exception as e:
        print(f"\nError during text generation: {e}")

    print("--- Pipeline Finished ---") 