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
        alt_pdf.split_into_chunks = optimized_split_into_chunks
        
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
                result = original_process_pdf(pdf_path, cfg)
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
            update_state(current_phase='building knowledge base')
            return original_build_kb(cfg, *args, **kwargs)
        
        # Replace the function
        kb.build_knowledge_base = patched_build_kb
        
        # 5. Try to patch tqdm for progress reporting
        try:
            import tqdm as tqdm_module
            from tqdm import tqdm as original_tqdm
            
            def patched_tqdm(*args, **kwargs):
                # Extract total items if available
                if args and len(args) > 0 and hasattr(args[0], '__len__'):
                    total_items = len(args[0])
                    if kwargs.get('desc', '').startswith('Processing PDFs'):
                        update_state(total_pdfs=total_items)
                elif 'total' in kwargs:
                    total_items = kwargs['total']
                    if kwargs.get('desc', '').startswith('Generating embeddings'):
                        pass  # Chunk embedding progress
                
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
                    elif tqdm_instance.desc and tqdm_instance.desc.startswith('Generating'):
                        update_state(processed_chunks=monitor_state['processed_chunks'] + n, 
                                    current_phase='generating embeddings')
                    
                    # Force flush stdout to ensure progress is visible
                    sys.stdout.flush()
                    
                    return result
                
                # Replace the update method
                tqdm_instance.update = patched_update
                
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
    
    # Recursively search for PDFs in all subdirectories
    for pdf_path in pdf_input_dir.glob("**/*.pdf"):
        if pdf_path.is_file():
            pdf_files.append((pdf_path, pdf_path.name))
    
    if not pdf_files:
        logging.error(f"No PDF files found in {pdf_input_dir} or its subdirectories")
        return False
    
    logging.info(f"Found {len(pdf_files)} PDF files")
    
    # Get already processed PDFs
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
            success = build_knowledge_base(batch_cfg)
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
        gc.collect()
        
        # Small delay to ensure resources are released
        time.sleep(2)
    
    logging.info("=" * 80)
    logging.info("KNOWLEDGE BASE BATCH PROCESSING COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Log file saved to: {log_file}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="FM-LLM-Solver Knowledge Base Builder")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--batch-size", type=int, default=3, help="Number of PDFs to process in each batch")
    parser.add_argument("--force", action="store_true", help="Force rebuilding entire knowledge base")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU usage even if GPU is available")
    parser.add_argument("--skip-monitor", action="store_true", help="Skip the progress monitor thread")
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
    
    # Start the monitor thread if not skipped
    monitor_thread = None
    if not args.skip_monitor:
        monitor_thread = threading.Thread(target=progress_monitor_thread, daemon=True)
        monitor_thread.start()
    
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