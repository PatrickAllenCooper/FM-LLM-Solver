import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import argparse

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directories and filenames (should match the builder script output)
BASE_DIR = os.path.dirname(__file__)
KB_DIR = os.path.join(BASE_DIR, "knowledge_base_enhanced") # Directory containing the index and metadata
VECTOR_STORE_FILENAME = "paper_index_enhanced.faiss"
METADATA_FILENAME = "paper_metadata_enhanced.json"

# Embedding Model (must match the one used for building)
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'

# --- Functions ---

def load_knowledge_base(kb_dir, index_filename, metadata_filename):
    """Loads the FAISS index and metadata map."""
    index_path = os.path.join(kb_dir, index_filename)
    metadata_path = os.path.join(kb_dir, metadata_filename)

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        logging.error(f"Knowledge base files not found in {kb_dir}.")
        logging.error("Please run the knowledge_base_builder.py script first.")
        return None, None

    try:
        logging.info(f"Loading FAISS index from {index_path}...")
        index = faiss.read_index(index_path)
        logging.info(f"Index loaded with {index.ntotal} vectors.")
    except Exception as e:
        logging.error(f"Failed to load FAISS index: {e}")
        return None, None

    try:
        logging.info(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            # Load keys as integers for direct indexing later
            metadata_map = {int(k): v for k, v in json.load(f).items()}
        logging.info(f"Metadata loaded for {len(metadata_map)} chunks.")
    except Exception as e:
        logging.error(f"Failed to load metadata JSON: {e}")
        return None, None

    if index.ntotal != len(metadata_map):
        logging.warning(f"Mismatch between index size ({index.ntotal}) and metadata size ({len(metadata_map)}).")
        # Decide how to handle: error out, or proceed with caution?
        # For now, proceed but log warning.

    return index, metadata_map

def search_kb(query, model, index, metadata_map, k=5):
    """Embeds the query and searches the FAISS index."""
    if index is None or metadata_map is None:
        logging.error("Knowledge base not loaded. Cannot search.")
        return []

    logging.info(f"Embedding query: '{query[:100]}...'") # Log truncated query
    try:
        query_embedding = model.encode([query]) # Pass query as a list
        query_embedding = np.array(query_embedding).astype('float32')
    except Exception as e:
        logging.error(f"Failed to embed query: {e}")
        return []

    logging.info(f"Searching index for top {k} results...")
    try:
        distances, indices = index.search(query_embedding, k) # Perform the search
        results = []
        if len(indices) > 0:
            for i, idx in enumerate(indices[0]):
                if idx != -1: # FAISS returns -1 for invalid indices
                    metadata = metadata_map.get(idx) # Use .get for safety
                    if metadata:
                        results.append({
                            'index_id': idx,
                            'distance': float(distances[0][i]), # L2 distance
                            'metadata': metadata
                        })
                    else:
                        logging.warning(f"Metadata not found for index {idx}.")

        logging.info(f"Found {len(results)} relevant chunks.")
        return results
    except Exception as e:
        logging.error(f"Error during FAISS search: {e}")
        return []

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the paper knowledge base.")
    parser.add_argument("query", type=str, help="The search query to run against the knowledge base.")
    parser.add_argument("-k", type=int, default=3, help="Number of results to retrieve (default: 3).")
    args = parser.parse_args()

    # 1. Load Embedding Model
    logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        logging.error(f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}")
        sys.exit(1)

    # 2. Load Knowledge Base
    index, metadata_map = load_knowledge_base(KB_DIR, VECTOR_STORE_FILENAME, METADATA_FILENAME)
    if index is None or metadata_map is None:
        sys.exit(1)

    # 3. Search
    search_results = search_kb(args.query, model, index, metadata_map, k=args.k)

    # 4. Print Results
    print(f"\n--- Search Results for: '{args.query}' ---")
    if not search_results:
        print("No relevant chunks found.")
    else:
        for i, result in enumerate(search_results):
            meta = result['metadata']
            print(f"\nResult {i+1} (Distance: {result['distance']:.4f}):")
            print(f"  Source: {meta.get('source', 'N/A')}")
            print(f"  Title (heuristic): {meta.get('potential_title', 'N/A')}")
            print(f"  Pages: {meta.get('pages', 'N/A')}")
            print(f"--- Chunk Text (Index: {result['index_id']}) --- ")
            print(meta.get('text', '[Error retrieving text]'))
            print("---------------------------------")

    logging.info("Test script finished.") 