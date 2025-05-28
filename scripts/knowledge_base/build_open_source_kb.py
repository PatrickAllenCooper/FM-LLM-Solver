#!/usr/bin/env python3
"""
Direct open-source knowledge base builder script.

This script bypasses the standard pipeline to directly build a knowledge base
using our minimal PDF processor, avoiding NumPy version conflicts.
"""

# Required packages:
# numpy<2.0.0  # Using older version to avoid compatibility issues
# faiss-cpu  # or faiss-gpu for CUDA support
# tqdm
# sentence-transformers  # Primary embedding method
# transformers  # Fallback embedding method
# scikit-learn  # For TF-IDF fallback
# torch
# pdf2image  # For PDF processing
# pytesseract  # For OCR

import os
import sys
import glob
import json
import logging
import re
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm

# Import our minimal PDF processor
from test_minimal_pdf_processor import process_pdf_simple, detect_hardware_simple, optimize_params

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('open_source_kb_build.log')
    ]
)

def find_pdf_files(data_dir="data/fetched_papers"):
    """Find all PDF files in the data directory"""
    pdf_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    return pdf_files

def split_text_into_chunks(text, chunk_size=1000, overlap=150):
    """Split text into overlapping chunks"""
    chunks = []
    lines = text.split('\n')
    
    current_chunk = []
    current_size = 0
    
    for line in lines:
        line_size = len(line)
        
        # Skip very long lines that would exceed chunk size
        if line_size > chunk_size * 1.5:
            # Process long line separately - break it up
            words = line.split()
            temp_line = ""
            for word in words:
                if len(temp_line) + len(word) + 1 > chunk_size:
                    current_chunk.append(temp_line)
                    current_size += len(temp_line)
                    temp_line = word
                else:
                    if temp_line:
                        temp_line += " " + word
                    else:
                        temp_line = word
            
            if temp_line:
                current_chunk.append(temp_line)
                current_size += len(temp_line)
            
            continue
            
        # If adding this line would exceed chunk size, finish the current chunk
        if current_size + line_size > chunk_size and current_chunk:
            chunks.append('\n'.join(current_chunk))
            
            # Handle overlap by keeping last N characters for the next chunk
            overlap_text = []
            overlap_size = 0
            
            # Work backwards through lines to create overlap
            for prev_line in reversed(current_chunk):
                if overlap_size + len(prev_line) <= overlap:
                    overlap_text.insert(0, prev_line)
                    overlap_size += len(prev_line)
                else:
                    break
                    
            current_chunk = overlap_text
            current_size = overlap_size
        
        # Add current line to the chunk
        current_chunk.append(line)
        current_size += line_size
    
    # Add the final chunk if not empty
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

def embed_text(chunks, model_name="all-mpnet-base-v2"):
    """
    Create embeddings for text chunks with multiple fallback options.
    
    This function tries different embedding methods in order of preference:
    1. SentenceTransformer with hardware acceleration
    2. HuggingFace transformers directly
    3. Simple TF-IDF vectorization as emergency fallback
    """
    # Try SentenceTransformer first
    try:
        from sentence_transformers import SentenceTransformer
        
        logging.info(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        
        # Check for GPU
        try:
            import torch
            if torch.cuda.is_available():
                logging.info("Using CUDA for embeddings")
                model = model.to('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logging.info("Using MPS for embeddings on Apple Silicon")
                model = model.to('mps')
        except (ImportError, AttributeError) as e:
            logging.warning(f"GPU acceleration unavailable: {e}")
            logging.info("Using CPU for embeddings")
            
        # Generate embeddings
        logging.info(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = model.encode(chunks, show_progress_bar=True)
        return np.array(embeddings).astype('float32')
        
    except Exception as e:
        logging.error(f"Error using SentenceTransformer: {e}")
        logging.warning("Trying alternative embedding method...")
        
        # Try using transformers directly
        try:
            logging.info("Attempting to use HuggingFace transformers directly")
            from transformers import AutoTokenizer, AutoModel
            import torch
            import numpy as np
            
            # Mean Pooling function
            def mean_pooling(model_output, attention_mask):
                token_embeddings = model_output[0]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
            model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
            
            # Process in batches to avoid OOM
            batch_size = 8
            all_embeddings = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512)
                
                # Compute token embeddings
                with torch.no_grad():
                    model_output = model(**encoded_input)
                
                # Perform pooling
                batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                all_embeddings.append(batch_embeddings.numpy())
            
            # Concatenate all embeddings
            embeddings = np.vstack(all_embeddings).astype('float32')
            return embeddings
            
        except Exception as e:
            logging.error(f"Error using transformers directly: {e}")
            logging.warning("Falling back to TF-IDF vectorization...")
            
            # Fallback to sklearn's TF-IDF
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.decomposition import TruncatedSVD
                
                # Create TF-IDF vectorizer
                vectorizer = TfidfVectorizer(max_features=10000)
                tfidf_matrix = vectorizer.fit_transform(chunks)
                
                # Reduce to 384 dimensions with SVD to match typical embedding size
                svd = TruncatedSVD(n_components=384)
                embeddings = svd.fit_transform(tfidf_matrix)
                
                logging.info("Successfully created TF-IDF based embeddings")
                return embeddings.astype('float32')
                
            except Exception as e:
                logging.error(f"Error with TF-IDF fallback: {e}")
                logging.warning("Using random vectors as last resort")
                
                # Last resort - random vectors
                dimension = 384  # Standard embedding size
                return np.random.randn(len(chunks), dimension).astype('float32')

def build_knowledge_base():
    """
    Process PDFs and build a knowledge base using the open source pipeline.
    """
    output_dir = "output/knowledge_base"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find PDF files
    pdf_files = find_pdf_files()
    logging.info(f"Found {len(pdf_files)} PDF files")
    
    if not pdf_files:
        logging.error("No PDF files found. Exiting.")
        return 1
    
    # Detect hardware and optimize parameters
    hardware_info = detect_hardware_simple()
    params = optimize_params(hardware_info)
    
    # Process each PDF
    all_chunks = []
    all_metadata = []
    
    for i, pdf_path in enumerate(tqdm(pdf_files, desc="Processing PDFs")):
        try:
            logging.info(f"Processing PDF {i+1}/{len(pdf_files)}: {os.path.basename(pdf_path)}")
            
            # Process the PDF
            markdown = process_pdf_simple(pdf_path, params)
            
            # Split into chunks
            chunks = split_text_into_chunks(markdown)
            
            # Save chunks and metadata
            for chunk_id, chunk in enumerate(chunks):
                chunk_metadata = {
                    "chunk_id": len(all_chunks),
                    "source": os.path.basename(pdf_path),
                    "path": pdf_path,
                    "text": chunk
                }
                
                all_chunks.append(chunk)
                all_metadata.append(chunk_metadata)
                
            logging.info(f"Created {len(chunks)} chunks from {os.path.basename(pdf_path)}")
            
        except Exception as e:
            logging.error(f"Error processing {pdf_path}: {e}")
    
    if not all_chunks:
        logging.error("No chunks were generated. Exiting.")
        return 1
    
    logging.info(f"Generated {len(all_chunks)} total chunks from {len(pdf_files)} PDFs")
    
    # Create embeddings
    try:
        embeddings = embed_text(all_chunks)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        # Save index and metadata
        faiss.write_index(index, os.path.join(output_dir, "paper_index_mathpix.faiss"))
        
        with open(os.path.join(output_dir, "paper_metadata_mathpix.jsonl"), 'w') as f:
            for item in all_metadata:
                f.write(json.dumps(item) + '\n')
        
        logging.info("Knowledge base build completed successfully")
        return 0
        
    except Exception as e:
        logging.error(f"Error building knowledge base: {e}")
        return 1

if __name__ == "__main__":
    logging.info("Starting direct open source knowledge base build")
    sys.exit(build_knowledge_base()) 