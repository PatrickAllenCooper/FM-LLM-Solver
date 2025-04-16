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

# --- Configuration ---
warnings.filterwarnings("ignore")
# Reduce transformers logging verbosity
# Use basicConfig from logging directly
import logging as py_logging
py_logging.basicConfig(level=py_logging.INFO)
logging.set_verbosity_error()

# Paths and Models (Should match previous steps)
# BASE_DIR is now the inference directory
BASE_DIR = os.path.dirname(__file__)
# Project root is one level up
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Knowledge Base (Relative to PROJECT_ROOT)
KB_DIR = os.path.join(PROJECT_ROOT, "knowledge_base", "knowledge_base_enhanced")
VECTOR_STORE_FILENAME = "paper_index_enhanced.faiss"
METADATA_FILENAME = "paper_metadata_enhanced.json"
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2' # Must match the one used for building KB

# Fine-tuned Model (Relative to PROJECT_ROOT)
# Base model used during fine-tuning
BASE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct" # Or the model you actually fine-tuned
# Path to the saved LoRA adapter weights
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "results_barrier_certs", "final_adapter")

# RAG Parameters
NUM_CONTEXT_CHUNKS = 3 # How many chunks to retrieve from KB

# Generation Parameters
MAX_NEW_TOKENS = 512 # Max tokens for the generated certificate
TEMPERATURE = 0.6 # Controls randomness (lower is more deterministic)
TOP_P = 0.9       # Nucleus sampling

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
        py_logging.info(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_map = {int(k): v for k, v in json.load(f).items()}
        py_logging.info(f"Knowledge base loaded: {index.ntotal} vectors, {len(metadata_map)} metadata entries.")
        if index.ntotal != len(metadata_map):
             py_logging.warning("Mismatch between index size and metadata size.")
        return index, metadata_map
    except Exception as e:
        py_logging.error(f"Failed to load knowledge base: {e}")
        return None, None

def load_finetuned_model(base_model_name, adapter_path):
    """Loads the base model with 4-bit quantization and merges the LoRA adapter."""
    # Configure quantization (must match fine-tuning config)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, # Or bfloat16 if used
        bnb_4bit_use_double_quant=False,
    )

    try:
        py_logging.info(f"Loading base model: {base_model_name}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map={"": 0}, # Load on GPU 0
            trust_remote_code=True
        )
        py_logging.info("Base model loaded.")

        py_logging.info(f"Loading adapter from: {adapter_path}...")
        # Load the LoRA adapter and merge it into the base model
        model = PeftModel.from_pretrained(base_model, adapter_path)
        py_logging.info("Adapter loaded.")

        # Important: Merge the adapter into the base model for optimized inference
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

    except FileNotFoundError:
        py_logging.error(f"Adapter not found at {adapter_path}. Did fine-tuning complete successfully?")
        return None, None
    except Exception as e:
        py_logging.error(f"Failed to load fine-tuned model or tokenizer: {e}")
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
                metadata = metadata_map.get(idx)
                if metadata:
                    context += f"--- Context Chunk {i+1} (Source: {metadata.get('source', 'N/A')}, Pages: {metadata.get('pages', 'N/A')}) ---\n"
                    context += metadata.get('text', '[Error retrieving text]') + "\n\n"
                else:
                    py_logging.warning(f"Metadata not found for retrieved index {idx}.")
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
    parser = argparse.ArgumentParser(description="Generate barrier certificate using RAG + Fine-tuned LLM.")
    parser.add_argument("system_description", type=str,
                        help="Text description of the autonomous system (dynamics, constraints, sets). Put in quotes.")
    parser.add_argument("-k", type=int, default=NUM_CONTEXT_CHUNKS,
                        help=f"Number of context chunks to retrieve (default: {NUM_CONTEXT_CHUNKS}).")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL_NAME,
                        help=f"Base model name used for fine-tuning (default: {BASE_MODEL_NAME}).")
    # Allow specifying adapter path relative to project root or absolute
    parser.add_argument("--adapter", type=str, default=ADAPTER_PATH,
                        help=f"Path to the fine-tuned LoRA adapter (default: {ADAPTER_PATH}).")
    # Allow specifying KB dir relative to project root or absolute
    parser.add_argument("--kb_dir", type=str, default=KB_DIR,
                         help=f"Path to the knowledge base directory (default: {KB_DIR})")
    parser.add_argument("--max_tokens", type=int, default=MAX_NEW_TOKENS,
                        help=f"Maximum new tokens for generation (default: {MAX_NEW_TOKENS}).")
    parser.add_argument("--temp", type=float, default=TEMPERATURE,
                        help=f"Generation temperature (default: {TEMPERATURE}).")
    args = parser.parse_args()

    print("--- Initializing RAG + Fine-tuned LLM Pipeline ---")

    # 1. Load Knowledge Base Components
    print(f"Loading Knowledge Base from {args.kb_dir}...")
    faiss_index, metadata = load_knowledge_base(args.kb_dir, VECTOR_STORE_FILENAME, METADATA_FILENAME)
    if faiss_index is None or metadata is None:
        exit(1)

    print("Loading Embedding Model...")
    try:
        embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"Error loading embedding model '{EMBEDDING_MODEL_NAME}': {e}")
        exit(1)

    # 2. Load Fine-tuned LLM
    print(f"Loading Fine-tuned LLM (Base: {args.base_model}, Adapter: {args.adapter})...")
    model, tokenizer = load_finetuned_model(args.base_model, args.adapter)
    if model is None or tokenizer is None:
        exit(1)

    # 3. Retrieve Context (RAG)
    print(f"Retrieving top-{args.k} context chunks for the query...")
    context = retrieve_context(args.system_description, embed_model, faiss_index, metadata, args.k)
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
    # Use Hugging Face pipeline for simpler generation
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_tokens,
        temperature=args.temp,
        top_p=TOP_P,
        do_sample=True # Important for temperature/top_p to have effect
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