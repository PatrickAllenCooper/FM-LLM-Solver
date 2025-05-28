#!/usr/bin/env python
"""
Knowledge Base Builder with simplified debugging.
This script runs the embedding process without tqdm, with explicit print statements
at key points to track progress and see where it might be getting stuck.
"""

import os
import sys
import time
import platform
from pathlib import Path
import argparse
from datetime import datetime, timedelta

# Make sure Python's output is unbuffered
os.environ['PYTHONUNBUFFERED'] = '1'

def main():
    parser = argparse.ArgumentParser(description="Debug knowledge base builder with direct output")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--force", action="store_true", help="Force rebuilding knowledge base even if files exist")
    args = parser.parse_args()
    
    # Ensure necessary directories exist
    data_dir = Path("data")
    output_dir = Path("output")
    kb_dir = output_dir / "knowledge_base"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(kb_dir, exist_ok=True)
    
    # Import and run KB builder module directly
    start_time = time.time()
    
    print("\n")
    print("=" * 80)
    print("STARTING KNOWLEDGE BASE BUILDING PROCESS - DEBUG MODE")
    print("=" * 80)
    print(f"Running on: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python: {sys.version}")
    print("This will insert explicit print statements to track progress")
    print("=" * 80)
    print("\n")
    
    # Import and run KB builder with our modifications
    try:
        # Add project root to path
        sys.path.insert(0, os.path.abspath('.'))
        
        # Import config loader
        from utils.config_loader import load_config, DEFAULT_CONFIG_PATH
        config_path = args.config if args.config else DEFAULT_CONFIG_PATH
        cfg = load_config(config_path)
        
        # Force lower memory usage settings
        cfg.knowledge_base.low_memory_mode = False
        cfg.knowledge_base.gpu_memory_limit = 4096
        
        # Force rebuild if requested
        if args.force:
            # Delete existing KB files if they exist
            vector_path = kb_dir / cfg.paths.kb_vector_store_filename
            metadata_path = kb_dir / cfg.paths.kb_metadata_filename
            
            if vector_path.exists():
                print(f"Removing existing vector store: {vector_path}")
                vector_path.unlink()
                
            if metadata_path.exists():
                print(f"Removing existing metadata: {metadata_path}")
                metadata_path.unlink()
        
        # Import internal modules with direct logging
        print("\nLoading modules...")
        import torch
        import faiss
        from sentence_transformers import SentenceTransformer
        from knowledge_base.alternative_pdf_processor import process_pdf, detect_hardware
        import logging
        
        # Monkey-patch SentenceTransformer.encode to add debugging
        original_encode = SentenceTransformer.encode
        
        def debug_encode(self, sentences, *args, **kwargs):
            batch_size = kwargs.get('batch_size', 32)
            print(f"\n[DEBUG] Encoding {len(sentences) if hasattr(sentences, '__len__') else 'unknown'} sentences with batch_size={batch_size}")
            start = time.time()
            result = original_encode(self, sentences, *args, **kwargs)
            end = time.time()
            print(f"[DEBUG] Encoding completed in {end-start:.2f} seconds")
            return result
            
        SentenceTransformer.encode = debug_encode
        
        # Implement our own step-by-step process to find where it's stuck
        print("\nStep 1: Scanning for PDFs...")
        pdf_files = []
        for root, dirs, files in os.walk(cfg.paths.pdf_input_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        print(f"Found {len(pdf_files)} PDF files")
        
        if not pdf_files:
            print("No PDF files found. Exiting.")
            return 1
            
        print("\nStep 2: Setting up embedding model...")
        hardware_info = detect_hardware()
        if hardware_info["has_gpu"]:
            print("CUDA GPU detected! Using GPU acceleration")
        else:
            print("No GPU detected. Using CPU")
        
        # Initialize embedding model with debug output
        embedding_model = cfg.embeddings.model_name
        print(f"Loading embedding model: {embedding_model}")
        
        model = SentenceTransformer(embedding_model)
        
        # Setup model on GPU with safeguards
        if hardware_info["has_gpu"]:
            try:
                # Limit GPU memory usage
                gpu_limit = cfg.knowledge_base.gpu_memory_limit
                print(f"Limiting GPU memory usage to {gpu_limit}MB")
                
                # Explicitly set device and memory limit
                device = torch.device('cuda')
                
                # Get memory info
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # MB
                print(f"GPU has {total_mem:.0f}MB total memory")
                
                # Set memory limit as fraction
                mem_fraction = min(gpu_limit / total_mem, 0.9)  # Cap at 90%
                print(f"Setting memory fraction to {mem_fraction:.2f}")
                torch.cuda.set_per_process_memory_fraction(mem_fraction)
                
                # Move model to GPU
                model = model.to(device)
                print("Model successfully moved to GPU")
            except Exception as e:
                print(f"Error setting up GPU: {e}")
                print("Falling back to CPU")
        
        # Process PDFs one by one with detailed reporting
        metadata_list = []
        all_embeddings = []
        
        # Process single PDF first as test
        print("\nStep 3: Processing first PDF as test...")
        first_pdf = pdf_files[0]
        print(f"Processing {os.path.basename(first_pdf)}")
        
        try:
            # Process the PDF
            mmd_content = process_pdf(first_pdf, cfg)
            
            if not mmd_content:
                print("No content extracted. Exiting.")
                return 1
                
            print(f"Successfully extracted {len(mmd_content)} characters of text")
            
            # Split content into chunks
            from knowledge_base.alternative_pdf_processor import split_into_chunks
            chunk_size = cfg.embeddings.chunk_size
            overlap = cfg.embeddings.chunk_overlap
            
            print(f"Splitting text into chunks (size={chunk_size}, overlap={overlap})...")
            chunks = split_into_chunks(mmd_content, chunk_size, overlap)
            
            print(f"Generated {len(chunks)} chunks")
            
            # Create metadata
            base_metadata = {
                "source": os.path.basename(first_pdf),
                "path": first_pdf
            }
            
            # Process chunks with detailed reporting
            print("\nStep 4: Generating embeddings (most intensive step)...")
            batch_size = cfg.knowledge_base.get('embedding_batch_size', 32)
            if not batch_size:
                batch_size = cfg.embeddings.get('batch_size', 32)
                
            print(f"Using batch_size={batch_size}")
            
            # Take only first few chunks for testing
            test_chunks = chunks[:min(5, len(chunks))]
            print(f"Testing with first {len(test_chunks)} chunks...")
            
            # Process in batches with direct output
            for chunk_id, chunk in enumerate(test_chunks):
                print(f"Processing chunk {chunk_id+1}/{len(test_chunks)} (length: {len(chunk)})")
                chunk_metadata = base_metadata.copy()
                chunk_metadata["chunk_id"] = chunk_id
                chunk_metadata["text"] = chunk
                
                # Generate embedding (one at a time for debugging)
                print(f"Generating embedding for chunk {chunk_id+1}...")
                embedding = model.encode(chunk, show_progress_bar=True)
                
                # Store results
                metadata_list.append(chunk_metadata)
                all_embeddings.append(embedding)
                
                print(f"Successfully generated embedding (shape: {embedding.shape})")
                
                # Monitor memory usage
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
                    max_allocated = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                    print(f"GPU memory: current={allocated:.1f}MB, peak={max_allocated:.1f}MB")
            
            # Success!
            print("\nTest completed successfully!")
            
            print("\nStep 5: FAISS Index Creation Testing...")
            try:
                # Convert embeddings to numpy array
                import numpy as np
                embeddings_array = np.array(all_embeddings).astype(np.float32)
                
                # Create and train the index
                dimension = embeddings_array.shape[1]
                index = faiss.IndexFlatL2(dimension)
                
                # Add vectors
                index.add(embeddings_array)
                print(f"Successfully added {index.ntotal} vectors to FAISS index")
                
                # Test the index
                print("Testing FAISS index with a simple query...")
                distance, idx = index.search(embeddings_array[:1], k=2)
                print(f"Search results: indices={idx}, distances={distance}")
                
                print("FAISS index created and tested successfully!")
                
            except Exception as e:
                print(f"Error in FAISS testing: {e}")
                return 1
            
        except Exception as e:
            import traceback
            print(f"Error processing PDF: {e}")
            traceback.print_exc()
            return 1
            
        # Display result
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n")
        print("=" * 80)
        print(f"DEBUG TEST COMPLETED IN {duration:.2f} SECONDS")
        print("=" * 80)
        
        if all_embeddings:
            print("\nDiagnostic Summary:")
            print(f"- PDF processing works: {'Yes' if mmd_content else 'No'}")
            print(f"- Chunk generation works: {'Yes' if chunks else 'No'}")
            print(f"- Embedding generation works: {'Yes' if all_embeddings else 'No'}")
            print(f"- FAISS index creation works: {'Yes' if 'index' in locals() else 'No'}")
            print("\nNow you can run the full process with: python run_kb_build_with_monitor.py --use-gpu")
            
        return 0
        
    except Exception as e:
        import traceback
        print(f"Error in debug process: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 