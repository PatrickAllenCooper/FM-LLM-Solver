#!/usr/bin/env python
"""
Batch Knowledge Base Builder - Processes PDFs in small batches to avoid memory issues.
This script processes a few PDFs at a time, saves intermediate results, then continues.
"""

import os
import sys
import time
import shutil
import argparse
from pathlib import Path
import json
import logging

# Make sure Python's output is unbuffered
os.environ['PYTHONUNBUFFERED'] = '1'

# Set PyTorch CUDA allocation options to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.6'

def setup_logging():
    """Configure logging to file and console"""
    log_dir = Path("output/logs")
    log_dir.mkdir(exist_ok=True, parents=True)
    
    log_file = log_dir / f"batch_kb_build_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

def get_processed_pdfs(metadata_path):
    """Get list of PDFs that have already been processed"""
    processed = set()
    
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if 'source' in data:
                            processed.add(data['source'])
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logging.error(f"Error reading metadata file: {e}")
    
    return processed

def merge_kb_files(temp_vector_path, temp_metadata_path, final_vector_path, final_metadata_path):
    """Merge temporary KB files with existing KB files"""
    # For metadata, simply append the new content
    if temp_metadata_path.exists():
        with open(temp_metadata_path, 'r') as src:
            if final_metadata_path.exists():
                with open(final_metadata_path, 'a') as dst:
                    # Append temp metadata to final file
                    dst.write(src.read())
            else:
                # Create new final metadata file
                shutil.copy2(temp_metadata_path, final_metadata_path)
    
    # For vector index, we need to use FAISS-specific merging
    if temp_vector_path.exists():
        try:
            import faiss
            
            if final_vector_path.exists():
                # Load both indices
                temp_index = faiss.read_index(str(temp_vector_path))
                final_index = faiss.read_index(str(final_vector_path))
                
                # Merge indices
                logging.info(f"Merging index with {temp_index.ntotal} vectors into existing index with {final_index.ntotal} vectors")
                
                # Get vectors from temp index
                temp_vectors = faiss.extract_index_vector(temp_index).at(0).numpy()
                
                # Add vectors to final index
                final_index.add(temp_vectors)
                
                # Save merged index
                faiss.write_index(final_index, str(final_vector_path))
                logging.info(f"Successfully merged indices. Final index now has {final_index.ntotal} vectors")
            else:
                # Just copy the temp index as the final index
                shutil.copy2(temp_vector_path, final_vector_path)
        except Exception as e:
            logging.error(f"Error merging vector indices: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Build knowledge base in small batches")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of PDFs to process in each batch")
    parser.add_argument("--force", action="store_true", help="Force rebuilding entire knowledge base")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU usage even if GPU is available")
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging()
    
    logging.info("=" * 80)
    logging.info("BATCH KNOWLEDGE BASE BUILDER")
    logging.info("=" * 80)
    logging.info(f"Will process PDFs in batches of {args.batch_size}")
    
    # Ensure we can import from the project root
    sys.path.insert(0, os.path.abspath('.'))
    
    # Import necessary modules
    try:
        from utils.config_loader import load_config, DEFAULT_CONFIG_PATH
        from knowledge_base.optimized_chunker import optimized_split_into_chunks
        import knowledge_base.alternative_pdf_processor as alt_pdf
        import knowledge_base.knowledge_base_builder as kb
        
        # First, patch the chunking function
        logging.info("Patching chunking function with optimized version...")
        alt_pdf.split_into_chunks = optimized_split_into_chunks
        
        # Patch process_pdf for memory management
        original_process_pdf = alt_pdf.process_pdf
        
        def clear_gpu_memory():
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    logging.info("CUDA cache cleared")
            except:
                pass
        
        def patched_process_pdf(pdf_path, cfg=None):
            # Clear GPU memory before processing
            clear_gpu_memory()
            
            # Try to process the PDF
            try:
                result = original_process_pdf(pdf_path, cfg)
                # Clear GPU memory after processing
                clear_gpu_memory()
                return result
            except Exception as e:
                # If CUDA out of memory, try to recover
                if "CUDA out of memory" in str(e):
                    logging.warning(f"CUDA out of memory on {os.path.basename(pdf_path)}, attempting fallback...")
                    # Force garbage collection
                    import gc
                    gc.collect()
                    
                    # Try again with CPU mode
                    if cfg is not None:
                        cfg.knowledge_base.low_memory_mode = True  # Force CPU mode
                        logging.info("Switched to CPU mode for this PDF")
                        try:
                            result = original_process_pdf(pdf_path, cfg)
                            clear_gpu_memory()
                            return result
                        except Exception as inner_e:
                            logging.error(f"Fallback also failed: {inner_e}")
                
                # Re-raise the original exception if recovery failed
                raise
        
        # Apply the patched function
        alt_pdf.process_pdf = patched_process_pdf
        
        # Load configuration
        config_path = args.config if args.config else DEFAULT_CONFIG_PATH
        cfg = load_config(config_path)
        
        # Set optimal memory settings
        if args.cpu_only:
            logging.info("Forcing CPU-only mode as requested")
            cfg.knowledge_base.low_memory_mode = True
        else:
            cfg.knowledge_base.low_memory_mode = False
            cfg.knowledge_base.gpu_memory_limit = 2048  # Reduce to 2GB to be safer
        
        # Set up paths
        pdf_input_dir = Path(cfg.paths.pdf_input_dir)
        kb_output_dir = Path(cfg.paths.kb_output_dir)
        temp_kb_dir = kb_output_dir / "temp"
        temp_pdf_dir = Path("data/temp_pdf_batch")
        
        # Final KB files
        final_vector_path = kb_output_dir / cfg.paths.kb_vector_store_filename
        final_metadata_path = kb_output_dir / cfg.paths.kb_metadata_filename
        
        # Temporary KB files (for this batch)
        temp_vector_path = temp_kb_dir / cfg.paths.kb_vector_store_filename
        temp_metadata_path = temp_kb_dir / cfg.paths.kb_metadata_filename
        
        # Create necessary directories
        kb_output_dir.mkdir(exist_ok=True, parents=True)
        temp_kb_dir.mkdir(exist_ok=True)
        
        # If force is specified, clean existing KB files
        if args.force:
            logging.info("Force rebuild requested - cleaning existing KB files")
            if final_vector_path.exists():
                final_vector_path.unlink()
            if final_metadata_path.exists():
                final_metadata_path.unlink()
        
        # Get list of PDF files
        pdf_files = []
        logging.info(f"Searching for PDFs in {pdf_input_dir} (including subdirectories)")
        
        # Recursively search for PDFs in all subdirectories
        for pdf_path in pdf_input_dir.glob("**/*.pdf"):
            if pdf_path.is_file():
                # Store both the relative path and the filename
                rel_path = pdf_path.relative_to(pdf_input_dir)
                pdf_files.append((pdf_path, pdf_path.name))
        
        if not pdf_files:
            logging.error(f"No PDF files found in {pdf_input_dir} or its subdirectories")
            return 1
        
        logging.info(f"Found {len(pdf_files)} PDF files")
        
        # Get already processed PDFs (using filename only, not path)
        processed_pdfs = get_processed_pdfs(final_metadata_path)
        logging.info(f"Found {len(processed_pdfs)} already processed PDFs")
        
        # Filter out already processed PDFs
        remaining_pdfs = [pdf_tuple for pdf_tuple in pdf_files if pdf_tuple[1] not in processed_pdfs]
        logging.info(f"{len(remaining_pdfs)} PDFs remaining to process")
        
        # Process PDFs in batches
        batch_size = args.batch_size
        total_batches = (len(remaining_pdfs) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(remaining_pdfs))
            current_batch = remaining_pdfs[batch_start:batch_end]
            
            logging.info(f"Processing batch {batch_idx+1}/{total_batches} with {len(current_batch)} PDFs")
            logging.info(f"PDFs in this batch: {', '.join([pdf[1] for pdf in current_batch])}")
            
            # Prepare temporary directory with just this batch of PDFs
            # Copy files from their original paths to the temp directory
            temp_pdf_dir.mkdir(exist_ok=True, parents=True)
            
            # Clear any existing files
            for file in temp_pdf_dir.glob("*.pdf"):
                file.unlink()
            
            # Copy the batch PDFs
            for pdf_path, pdf_name in current_batch:
                dst = temp_pdf_dir / pdf_name
                shutil.copy2(pdf_path, dst)
            
            # Update config to use temporary directories
            batch_cfg = cfg.copy()
            batch_cfg.paths.pdf_input_dir = str(temp_pdf_dir)
            batch_cfg.paths.kb_output_dir = str(temp_kb_dir)
            
            # Clear any existing temp KB files
            if temp_vector_path.exists():
                temp_vector_path.unlink()
            if temp_metadata_path.exists():
                temp_metadata_path.unlink()
            
            # Process this batch
            logging.info(f"Building knowledge base for batch {batch_idx+1}")
            try:
                batch_start_time = time.time()
                success = kb.build_knowledge_base(batch_cfg)
                batch_end_time = time.time()
                
                if success:
                    logging.info(f"Batch {batch_idx+1} completed in {batch_end_time - batch_start_time:.2f} seconds")
                    
                    # Merge results into main knowledge base
                    logging.info("Merging batch results into main knowledge base")
                    merge_success = merge_kb_files(temp_vector_path, temp_metadata_path, 
                                                 final_vector_path, final_metadata_path)
                    
                    if merge_success:
                        logging.info(f"Successfully merged batch {batch_idx+1} into main knowledge base")
                    else:
                        logging.error(f"Failed to merge batch {batch_idx+1} - will try to continue with next batch")
                else:
                    logging.error(f"Batch {batch_idx+1} failed - will try to continue with next batch")
            except Exception as e:
                logging.error(f"Error processing batch {batch_idx+1}: {e}")
                import traceback
                traceback.print_exc()
                logging.info("Will try to continue with next batch")
            
            # Force cleanup
            clear_gpu_memory()
            import gc
            gc.collect()
            
            # Small delay to ensure resources are released
            time.sleep(2)
        
        logging.info("=" * 80)
        logging.info("KNOWLEDGE BASE BATCH PROCESSING COMPLETE")
        logging.info("=" * 80)
        logging.info(f"Log file saved to: {log_file}")
        
        return 0
        
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 