#!/usr/bin/env python
"""
FM-LLM-Solver Knowledge Base Builder - Consolidated Version

This script provides a robust solution for building a knowledge base from PDF files.
Features:
- Memory-optimized processing to avoid GPU out-of-memory errors
- Batch processing with resume capability
- Progress monitoring with detailed logging
- Optimized chunking algorithm
- Support for nested directory structures

Usage:
  python kb_builder.py [--batch-size N] [--force] [--cpu-only] [--config PATH]
"""

import os
import sys
import time
import shutil
import argparse
import threading
import gc
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Make sure Python's output is unbuffered
os.environ['PYTHONUNBUFFERED'] = '1'

# Set PyTorch CUDA allocation options to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.6'

# Global state for monitoring
monitor_state = {
    'start_time': time.time(),
    'running': True,
    'total_pdfs': 0,
    'processed_pdfs': 0,
    'total_chunks': 0,
    'processed_chunks': 0,
    'last_activity': time.time(),
    'current_file': '',
    'current_phase': 'initializing',
    'batch_number': 0,
    'total_batches': 0
}

# Lock for thread-safe updates
state_lock = threading.Lock()

def update_state(**kwargs):
    """Thread-safe update of the monitor state"""
    with state_lock:
        for key, value in kwargs.items():
            if key in monitor_state:
                monitor_state[key] = value
        # Always update the last activity time when state is updated
        monitor_state['last_activity'] = time.time()

def format_time(seconds):
    """Format seconds into a human-readable duration"""
    return str(timedelta(seconds=int(seconds)))

def progress_monitor_thread():
    """Thread function to monitor and report progress periodically"""
    check_interval = 5  # seconds between progress reports
    inactivity_threshold = 60  # seconds of no activity before showing a special message
    
    print("\n[MONITOR] Starting progress monitor. You'll see updates every 5 seconds.\n")
    
    while monitor_state['running']:
        with state_lock:
            now = time.time()
            elapsed = now - monitor_state['start_time']
            inactivity_time = now - monitor_state['last_activity']
            
            # Calculate rates and estimates
            pdfs_per_sec = monitor_state['processed_pdfs'] / elapsed if elapsed > 0 else 0
            chunks_per_sec = monitor_state['processed_chunks'] / elapsed if elapsed > 0 else 0
            
            # Estimate remaining time
            if monitor_state['total_pdfs'] > 0 and pdfs_per_sec > 0:
                pdf_remaining = monitor_state['total_pdfs'] - monitor_state['processed_pdfs'] 
                pdf_time_remaining = pdf_remaining / pdfs_per_sec
                pdf_eta = datetime.now() + timedelta(seconds=pdf_time_remaining)
            else:
                pdf_time_remaining = 0
                pdf_eta = datetime.now()
                
            # Build progress message
            message = f"\n[MONITOR] Progress Report (running for {format_time(elapsed)}):\n"
            
            if monitor_state['batch_number'] > 0:
                message += f"  Current batch: {monitor_state['batch_number']}/{monitor_state['total_batches']}\n"
                
            message += f"  Current phase: {monitor_state['current_phase']}\n"
            message += f"  Current file: {monitor_state['current_file']}\n"
            
            if monitor_state['total_pdfs'] > 0:
                pdf_percent = (monitor_state['processed_pdfs'] / monitor_state['total_pdfs']) * 100
                message += f"  PDFs: {monitor_state['processed_pdfs']}/{monitor_state['total_pdfs']} ({pdf_percent:.1f}%)\n"
                message += f"  Rate: {pdfs_per_sec*60:.2f} PDFs/min\n"
                
                if pdf_time_remaining > 0:
                    message += f"  Estimated time remaining: {format_time(pdf_time_remaining)}\n"
                    message += f"  Estimated completion: {pdf_eta.strftime('%H:%M:%S')}\n"
            
            if monitor_state['processed_chunks'] > 0:
                message += f"  Chunks processed: {monitor_state['processed_chunks']}\n"
                message += f"  Chunk processing rate: {chunks_per_sec*60:.1f} chunks/min\n"
                
            # Add inactivity warning if needed
            if inactivity_time > inactivity_threshold:
                message += f"\n  WARNING: No activity detected for {format_time(inactivity_time)}!\n"
                message += "  The process may be working on a large computation or stuck.\n"
                message += "  If no progress is shown for several minutes, consider restarting.\n"
                
            print(message)
            sys.stdout.flush()  # Force immediate output
            
        time.sleep(check_interval)
    
    print("\n[MONITOR] Process completed. Shutting down monitor.\n")

def setup_logging():
    """Configure logging to file and console"""
    log_dir = Path("output/logs")
    log_dir.mkdir(exist_ok=True, parents=True)
    
    log_file = log_dir / f"kb_build_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
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

def clear_gpu_memory():
    """Force clearing of GPU memory cache"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logging.info("CUDA cache cleared")
    except Exception as e:
        logging.warning(f"Failed to clear GPU memory: {e}")

def merge_kb_files(temp_vector_path, temp_metadata_path, final_vector_path, final_metadata_path):
    """Merge temporary KB files with existing KB files"""
    logging.info(f"DEBUG: Starting merge operation")
    logging.info(f"DEBUG: Temp vector: {temp_vector_path} (exists: {temp_vector_path.exists()})")
    logging.info(f"DEBUG: Temp metadata: {temp_metadata_path} (exists: {temp_metadata_path.exists()})")
    logging.info(f"DEBUG: Final vector: {final_vector_path} (exists: {final_vector_path.exists()})")
    logging.info(f"DEBUG: Final metadata: {final_metadata_path} (exists: {final_metadata_path.exists()})")
    
    # For metadata, simply append the new content
    if temp_metadata_path.exists():
        logging.info(f"DEBUG: Merging metadata files")
        metadata_start = time.time()
        try:
            with open(temp_metadata_path, 'r') as src:
                if final_metadata_path.exists():
                    with open(final_metadata_path, 'a') as dst:
                        # Append temp metadata to final file
                        content = src.read()
                        logging.info(f"DEBUG: Appending {len(content)} bytes to metadata file")
                        dst.write(content)
                else:
                    # Create new final metadata file
                    logging.info(f"DEBUG: Creating new metadata file (copying)")
                    shutil.copy2(temp_metadata_path, final_metadata_path)
            metadata_end = time.time()
            logging.info(f"DEBUG: Metadata merge completed in {metadata_end-metadata_start:.2f}s")
        except Exception as e:
            logging.error(f"DEBUG: Error merging metadata: {e}")
            import traceback
            traceback.print_exc()
    
    # For vector index, we need to use FAISS-specific merging
    if temp_vector_path.exists():
        logging.info(f"DEBUG: Starting FAISS vector index merge")
        vector_start = time.time()
        try:
            import faiss
            
            if final_vector_path.exists():
                # Load both indices
                logging.info(f"DEBUG: Loading temp index from {temp_vector_path}")
                temp_index_start = time.time()
                temp_index = faiss.read_index(str(temp_vector_path))
                temp_index_end = time.time()
                logging.info(f"DEBUG: Temp index loaded in {temp_index_end-temp_index_start:.2f}s, contains {temp_index.ntotal} vectors")
                
                logging.info(f"DEBUG: Loading final index from {final_vector_path}")
                final_index_start = time.time() 
                final_index = faiss.read_index(str(final_vector_path))
                final_index_end = time.time()
                logging.info(f"DEBUG: Final index loaded in {final_index_end-final_index_start:.2f}s, contains {final_index.ntotal} vectors")
                
                # Merge indices
                logging.info(f"DEBUG: Extracting vectors from temp index")
                extract_start = time.time()
                temp_vectors = faiss.extract_index_vector(temp_index).at(0).numpy()
                extract_end = time.time()
                logging.info(f"DEBUG: Vector extraction completed in {extract_end-extract_start:.2f}s")
                
                logging.info(f"DEBUG: Adding {len(temp_vectors)} vectors to final index")
                add_start = time.time()
                final_index.add(temp_vectors)
                add_end = time.time()
                logging.info(f"DEBUG: Vector addition completed in {add_end-add_start:.2f}s")
                
                logging.info(f"DEBUG: Saving merged index to {final_vector_path}")
                save_start = time.time()
                faiss.write_index(final_index, str(final_vector_path))
                save_end = time.time()
                logging.info(f"DEBUG: Index saved in {save_end-save_start:.2f}s")
                
                logging.info(f"Successfully merged indices. Final index now has {final_index.ntotal} vectors")
            else:
                # Just copy the temp index as the final index
                logging.info(f"DEBUG: No existing index, copying temp index to final location")
                copy_start = time.time()
                shutil.copy2(temp_vector_path, final_vector_path)
                copy_end = time.time()
                logging.info(f"DEBUG: Index copy completed in {copy_end-copy_start:.2f}s")
            
            vector_end = time.time()
            logging.info(f"DEBUG: Vector index merge completed in {vector_end-vector_start:.2f}s")
        except Exception as e:
            logging.error(f"DEBUG: Error merging vector indices: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

def monkey_patch_kb_modules():
    """Patch knowledge base modules for optimized processing and monitoring"""
    # Ensure we can import from the project root
    sys.path.insert(0, os.path.abspath('.'))
    
    try:
        # Import modules to patch
        from knowledge_base.optimized_chunker import optimized_split_into_chunks
        import knowledge_base.alternative_pdf_processor as alt_pdf
        import knowledge_base.knowledge_base_builder as kb
        
        # 1. Patch the chunking function with our optimized version
        logging.info("Patching chunking function with optimized version...")
        
        # Store original chunking function for debugging
        original_split = alt_pdf.split_into_chunks
        
        # Create a debug wrapper for chunking function
        def debug_split_into_chunks(text, chunk_size=1000, overlap=200):
            logging.info(f"DEBUG: Starting chunking of text ({len(text)} chars)")
            start_time = time.time()
            try:
                chunks = optimized_split_into_chunks(text, chunk_size, overlap)
                end_time = time.time()
                logging.info(f"DEBUG: Chunking completed in {end_time-start_time:.2f}s - produced {len(chunks)} chunks")
                return chunks
            except Exception as e:
                logging.error(f"DEBUG: Error during chunking: {e}")
                # Fall back to original in case of error
                logging.info("DEBUG: Falling back to original chunking method")
                try:
                    return original_split(text, chunk_size, overlap)
                except Exception as fallback_e:
                    logging.error(f"DEBUG: Fallback chunking also failed: {fallback_e}")
                    # Return minimal chunks to avoid complete failure
                    return [text]
        
        # Use our debug chunking function
        alt_pdf.split_into_chunks = debug_split_into_chunks
        
        # 2. Store original process_pdf function for patching
        original_process_pdf = alt_pdf.process_pdf
        
        # 3. Create a wrapper to add memory management
        def patched_process_pdf(pdf_path, cfg=None):
            # Update monitor state
            update_state(current_file=os.path.basename(pdf_path), current_phase='processing PDF')
            
            # Clear GPU memory before processing
            clear_gpu_memory()
            
            # Try to process the PDF
            try:
                logging.info(f"DEBUG: Starting PDF processing for {os.path.basename(pdf_path)}")
                start_time = time.time()
                result = original_process_pdf(pdf_path, cfg)
                end_time = time.time()
                logging.info(f"DEBUG: PDF processing completed in {end_time-start_time:.2f}s")
                # Clear GPU memory after processing
                clear_gpu_memory()
                return result
            except Exception as e:
                # If CUDA out of memory, try to recover
                if "CUDA out of memory" in str(e):
                    logging.warning(f"CUDA out of memory on {os.path.basename(pdf_path)}, attempting fallback...")
                    # Force garbage collection
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
        
        # 4. Patch build_knowledge_base function to track progress
        original_build_kb = kb.build_knowledge_base
        
        def patched_build_kb(cfg, *args, **kwargs):
            logging.info("DEBUG: Starting knowledge base build")
            update_state(current_phase='building knowledge base')
            
            # Patch the SentenceTransformer encode method to track embedding progress
            try:
                from sentence_transformers import SentenceTransformer
                original_encode = SentenceTransformer.encode
                
                def patched_encode(self, sentences, *args, **kwargs):
                    logging.info(f"DEBUG: Starting embedding generation for {len(sentences)} texts")
                    update_state(current_phase=f'generating embeddings for {len(sentences)} texts')
                    start_time = time.time()
                    try:
                        result = original_encode(self, sentences, *args, **kwargs)
                        end_time = time.time()
                        logging.info(f"DEBUG: Embedding generation completed in {end_time-start_time:.2f}s")
                        return result
                    except Exception as e:
                        logging.error(f"DEBUG: Error during embedding generation: {e}")
                        raise
                
                SentenceTransformer.encode = patched_encode
                logging.info("DEBUG: Successfully patched SentenceTransformer.encode for monitoring")
            except Exception as e:
                logging.warning(f"DEBUG: Could not patch SentenceTransformer.encode: {e}")
            
            # Now call the original build function
            try:
                result = original_build_kb(cfg, *args, **kwargs)
                logging.info("DEBUG: Knowledge base build completed")
                return result
            except Exception as e:
                logging.error(f"DEBUG: Error during knowledge base build: {e}")
                raise
        
        # Replace the function
        kb.build_knowledge_base = patched_build_kb
        
        # 5. Try to patch tqdm for progress reporting
        try:
            import tqdm as tqdm_module
            from tqdm import tqdm as original_tqdm
            
            def patched_tqdm(*args, **kwargs):
                # Log the tqdm initialization
                if 'desc' in kwargs:
                    logging.info(f"DEBUG: Starting progress tracking: {kwargs['desc']}")
                
                # Extract total items if available
                if args and len(args) > 0 and hasattr(args[0], '__len__'):
                    total_items = len(args[0])
                    if kwargs.get('desc', '').startswith('Processing PDFs'):
                        update_state(total_pdfs=total_items)
                        logging.info(f"DEBUG: Starting processing of {total_items} PDFs")
                elif 'total' in kwargs:
                    total_items = kwargs['total']
                    if kwargs.get('desc', '').startswith('Generating embeddings'):
                        logging.info(f"DEBUG: Tracking embedding generation for {total_items} chunks")
                
                # Create the original tqdm instance
                tqdm_instance = original_tqdm(*args, **kwargs)
                
                # Store the original update method
                original_update = tqdm_instance.update
                
                # Create a patched update method
                def patched_update(n=1):
                    # Call the original update
                    result = original_update(n)
                    
                    # Update our monitor based on the progress bar type
                    if tqdm_instance.desc and tqdm_instance.desc.startswith('Processing PDFs'):
                        update_state(processed_pdfs=tqdm_instance.n, current_phase='processing PDFs')
                        # Log every 10 PDFs
                        if tqdm_instance.n % 10 == 0:
                            logging.info(f"DEBUG: Processed {tqdm_instance.n} PDFs")
                    elif tqdm_instance.desc and tqdm_instance.desc.startswith('Generating'):
                        update_state(processed_chunks=monitor_state['processed_chunks'] + n, 
                                    current_phase='generating embeddings')
                        # Log every 100 chunks
                        if monitor_state['processed_chunks'] % 100 == 0:
                            logging.info(f"DEBUG: Generated embeddings for {monitor_state['processed_chunks']} chunks")
                    
                    # Force flush stdout to ensure progress is visible
                    sys.stdout.flush()
                    
                    return result
                
                # Replace the update method
                tqdm_instance.update = patched_update
                
                # Store the original close method
                original_close = tqdm_instance.close
                
                # Create a patched close method to log completion
                def patched_close():
                    if tqdm_instance.desc:
                        logging.info(f"DEBUG: Completed progress tracking: {tqdm_instance.desc}")
                    return original_close()
                
                # Replace the close method
                tqdm_instance.close = patched_close
                
                return tqdm_instance
            
            # Add write function to our patched function
            patched_tqdm.write = original_tqdm.write
            
            # Replace tqdm in the kb module
            kb.tqdm = patched_tqdm
            
            # Also add to tqdm module if it can access it
            try:
                tqdm_module.tqdm = patched_tqdm
            except AttributeError:
                pass
                
        except ImportError:
            logging.warning("tqdm not available, using basic progress monitoring only")
    
        return True
    except ImportError as e:
        logging.error(f"Error patching KB modules: {e}")
        return False

def process_in_batches(args, cfg, log_file):
    """Process PDFs in batches to manage memory usage"""
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
    
    # Get list of PDF files (recursively search subdirectories)
    pdf_files = []
    logging.info(f"Searching for PDFs in {pdf_input_dir} (including subdirectories)")
    search_start = time.time()
    
    # Recursively search for PDFs in all subdirectories
    for pdf_path in pdf_input_dir.glob("**/*.pdf"):
        if pdf_path.is_file():
            pdf_files.append((pdf_path, pdf_path.name))
    
    search_end = time.time()
    logging.info(f"DEBUG: PDF search completed in {search_end-search_start:.2f}s")
    
    if not pdf_files:
        logging.error(f"No PDF files found in {pdf_input_dir} or its subdirectories")
        return False
    
    logging.info(f"Found {len(pdf_files)} PDF files")
    
    # Get already processed PDFs
    logging.info(f"DEBUG: Checking for already processed PDFs in {final_metadata_path}")
    processed_pdfs = get_processed_pdfs(final_metadata_path)
    logging.info(f"Found {len(processed_pdfs)} already processed PDFs")
    
    # Filter out already processed PDFs
    remaining_pdfs = [pdf_tuple for pdf_tuple in pdf_files if pdf_tuple[1] not in processed_pdfs]
    logging.info(f"{len(remaining_pdfs)} PDFs remaining to process")
    
    # Process PDFs in batches
    batch_size = args.batch_size
    total_batches = (len(remaining_pdfs) + batch_size - 1) // batch_size
    
    # Update monitor state with batch information
    update_state(total_pdfs=len(remaining_pdfs), total_batches=total_batches)
    
    # Import knowledge base builder after patching
    from knowledge_base.knowledge_base_builder import build_knowledge_base
    
    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(remaining_pdfs))
        current_batch = remaining_pdfs[batch_start:batch_end]
        
        # Update monitor state
        update_state(batch_number=batch_idx+1, 
                    current_phase=f"preparing batch {batch_idx+1}")
        
        logging.info(f"Processing batch {batch_idx+1}/{total_batches} with {len(current_batch)} PDFs")
        logging.info(f"PDFs in this batch: {', '.join([pdf[1] for pdf in current_batch])}")
        
        # Prepare temporary directory with just this batch of PDFs
        # Create temporary directory
        logging.info(f"DEBUG: Preparing temp directory for batch {batch_idx+1}")
        temp_pdf_dir.mkdir(exist_ok=True, parents=True)
        
        # Clear any existing files
        for file in temp_pdf_dir.glob("*.pdf"):
            file.unlink()
        
        # Copy the batch PDFs
        logging.info(f"DEBUG: Copying {len(current_batch)} PDFs to temp directory")
        copy_start = time.time()
        for pdf_path, pdf_name in current_batch:
            dst = temp_pdf_dir / pdf_name
            shutil.copy2(pdf_path, dst)
        copy_end = time.time()
        logging.info(f"DEBUG: PDF copying completed in {copy_end-copy_start:.2f}s")
        
        # Update config to use temporary directories
        batch_cfg = cfg.copy()
        batch_cfg.paths.pdf_input_dir = str(temp_pdf_dir)
        batch_cfg.paths.kb_output_dir = str(temp_kb_dir)
        
        # Clear any existing temp KB files
        if temp_vector_path.exists():
            logging.info(f"DEBUG: Removing existing temp vector store: {temp_vector_path}")
            temp_vector_path.unlink()
        if temp_metadata_path.exists():
            logging.info(f"DEBUG: Removing existing temp metadata: {temp_metadata_path}")
            temp_metadata_path.unlink()
        
        # Process this batch
        logging.info(f"Building knowledge base for batch {batch_idx+1}")
        try:
            batch_start_time = time.time()
            logging.info(f"DEBUG: Starting build_knowledge_base call for batch {batch_idx+1}")
            success = build_knowledge_base(batch_cfg)
            batch_end_time = time.time()
            
            logging.info(f"DEBUG: build_knowledge_base call completed in {batch_end_time-batch_start_time:.2f}s")
            
            if success:
                logging.info(f"Batch {batch_idx+1} completed in {batch_end_time - batch_start_time:.2f} seconds")
                
                # Merge results into main knowledge base
                logging.info(f"DEBUG: Starting merge of batch {batch_idx+1} results")
                logging.info("Merging batch results into main knowledge base")
                
                # Check if temp files exist
                if not temp_vector_path.exists():
                    logging.error(f"DEBUG: Expected temp vector file not found: {temp_vector_path}")
                if not temp_metadata_path.exists():
                    logging.error(f"DEBUG: Expected temp metadata file not found: {temp_metadata_path}")
                
                merge_start = time.time()
                merge_success = merge_kb_files(temp_vector_path, temp_metadata_path, 
                                            final_vector_path, final_metadata_path)
                merge_end = time.time()
                logging.info(f"DEBUG: Merge operation completed in {merge_end-merge_start:.2f}s")
                
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
        logging.info(f"DEBUG: Performing memory cleanup after batch {batch_idx+1}")
        clear_gpu_memory()
        gc.collect()
        
        # Small delay to ensure resources are released
        logging.info(f"DEBUG: Waiting 2 seconds before next batch")
        time.sleep(2)
    
    logging.info("=" * 80)
    logging.info("KNOWLEDGE BASE BATCH PROCESSING COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Log file saved to: {log_file}")
    
    return True

def start_watchdog_thread():
    """Start a watchdog thread to monitor for hangs and provide diagnostics"""
    def watchdog_thread():
        """Thread function to detect hangs and log diagnostics"""
        last_activity_time = time.time()
        long_inactivity_threshold = 120  # 2 minutes of no activity is considered a hang
        check_interval = 30  # Check every 30 seconds
        
        while monitor_state['running']:
            current_time = time.time()
            current_inactivity = current_time - monitor_state['last_activity']
            
            # Check if we're seeing a long period of inactivity
            if current_inactivity > long_inactivity_threshold:
                # We might be hanging - log detailed system info
                logging.warning(f"WATCHDOG: Potential hang detected - no activity for {current_inactivity:.1f} seconds")
                
                # Log current state
                logging.warning(f"WATCHDOG: Current phase: {monitor_state['current_phase']}")
                logging.warning(f"WATCHDOG: Current file: {monitor_state['current_file']}")
                logging.warning(f"WATCHDOG: Current batch: {monitor_state['batch_number']}/{monitor_state['total_batches']}")
                
                # Try to get GPU memory info
                try:
                    import torch
                    if torch.cuda.is_available():
                        # Get memory stats
                        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                        reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
                        max_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
                        
                        logging.warning(f"WATCHDOG: GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {max_mem:.2f}GB")
                        
                        # Get detailed memory stats by device if available
                        for i in range(torch.cuda.device_count()):
                            logging.warning(f"WATCHDOG: Device {i} memory: {torch.cuda.memory_allocated(i) / (1024 ** 3):.2f}GB allocated")
                except Exception as e:
                    logging.warning(f"WATCHDOG: Could not get GPU memory info: {e}")
                
                # Try to log CPU and system memory 
                try:
                    import psutil
                    process = psutil.Process()
                    cpu_percent = process.cpu_percent(interval=1)
                    ram_used = process.memory_info().rss / (1024 ** 3)  # GB
                    system_ram = psutil.virtual_memory().total / (1024 ** 3)  # GB
                    
                    logging.warning(f"WATCHDOG: CPU Usage: {cpu_percent}%, RAM Used: {ram_used:.2f}GB, System RAM: {system_ram:.2f}GB")
                except ImportError:
                    logging.warning("WATCHDOG: psutil not available, can't get CPU/RAM info")
                except Exception as e:
                    logging.warning(f"WATCHDOG: Error getting CPU/RAM info: {e}")
                
                # Try to log active threads
                try:
                    import threading
                    active_threads = threading.enumerate()
                    logging.warning(f"WATCHDOG: {len(active_threads)} active threads:")
                    for thread in active_threads:
                        logging.warning(f"WATCHDOG: Thread {thread.name} - Daemon: {thread.daemon}, Alive: {thread.is_alive()}")
                except Exception as e:
                    logging.warning(f"WATCHDOG: Could not enumerate threads: {e}")
                
                # Force a garbage collection to see if it helps
                try:
                    gc.collect()
                    logging.warning("WATCHDOG: Forced garbage collection")
                except Exception as e:
                    logging.warning(f"WATCHDOG: Error during forced GC: {e}")
            
            # Sleep for a while before checking again
            time.sleep(check_interval)
    
    # Start the watchdog thread
    watchdog = threading.Thread(target=watchdog_thread, daemon=True, name="Watchdog")
    watchdog.start()
    return watchdog

def main():
    parser = argparse.ArgumentParser(description="FM-LLM-Solver Knowledge Base Builder")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--batch-size", type=int, default=3, help="Number of PDFs to process in each batch")
    parser.add_argument("--force", action="store_true", help="Force rebuilding entire knowledge base")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU usage even if GPU is available")
    parser.add_argument("--skip-monitor", action="store_true", help="Skip the progress monitor thread")
    parser.add_argument("--debug", action="store_true", help="Enable additional debug logging")
    args = parser.parse_args()
    
    # Set up logging
    log_file = setup_logging()
    
    # Print welcome message
    print("\n")
    print("=" * 80)
    print("FM-LLM-SOLVER KNOWLEDGE BASE BUILDER")
    print("=" * 80)
    print("This script will build the knowledge base and index PDF documents.")
    print(f"Processing in batches of {args.batch_size} PDFs at a time.")
    print("If the process crashes, you can run it again to continue from where it left off.")
    print("=" * 80)
    print("\n")
    
    # Enable debug logging if requested
    if args.debug:
        logging.info("Debug logging enabled - you'll see detailed progress information")
    
    # Start the monitor thread if not skipped
    monitor_thread = None
    if not args.skip_monitor:
        monitor_thread = threading.Thread(target=progress_monitor_thread, daemon=True, name="ProgressMonitor")
        monitor_thread.start()
    
    # Start the watchdog thread to detect hangs
    watchdog_thread = start_watchdog_thread()
    logging.info("Watchdog thread started to monitor for potential hangs")
    
    try:
        # Apply all the necessary patches and optimizations
        patch_success = monkey_patch_kb_modules()
        if not patch_success:
            logging.error("Failed to apply optimizations. Aborting.")
            return 1
        
        # Import configuration
        from utils.config_loader import load_config, DEFAULT_CONFIG_PATH
        config_path = args.config if args.config else DEFAULT_CONFIG_PATH
        cfg = load_config(config_path)
        
        # Set optimal memory settings
        if args.cpu_only:
            logging.info("Forcing CPU-only mode as requested")
            cfg.knowledge_base.low_memory_mode = True
        else:
            logging.info("Using GPU with memory limits")
            cfg.knowledge_base.low_memory_mode = False
            cfg.knowledge_base.gpu_memory_limit = 2048  # Reduce to 2GB to be safer
        
        # Process PDFs in batches
        success = process_in_batches(args, cfg, log_file)
        
        # Stop the monitor
        update_state(running=False)
        if monitor_thread:
            monitor_thread.join(1.0)  # Wait up to 1 second for it to finish
        
        return 0 if success else 1
        
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Stop the monitor
        update_state(running=False)
        if monitor_thread:
            monitor_thread.join(1.0)
            
        return 1
    finally:
        # Make sure monitor is stopped
        update_state(running=False)

if __name__ == "__main__":
    sys.exit(main()) 