#!/usr/bin/env python
"""
Knowledge Base Builder with forced progress monitoring.
This script uses a separate thread to regularly print progress,
ensuring progress is always visible even if tqdm or normal progress bars don't work.
"""

import os
import sys
import time
import threading
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Make sure Python's output is unbuffered
os.environ['PYTHONUNBUFFERED'] = '1'

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
    'current_phase': 'initializing'
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
    inactivity_threshold = 30  # seconds of no activity before showing a special message
    
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
            message += f"  Current phase: {monitor_state['current_phase']}\n"
            message += f"  Current file: {monitor_state['current_file']}\n"
            
            if monitor_state['total_pdfs'] > 0:
                pdf_percent = (monitor_state['processed_pdfs'] / monitor_state['total_pdfs']) * 100
                message += f"  PDFs: {monitor_state['processed_pdfs']}/{monitor_state['total_pdfs']} ({pdf_percent:.1f}%)\n"
                message += f"  Rate: {pdfs_per_sec:.2f} PDFs/min\n"
                
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

# Monkey-patch the knowledge_base_builder.py functions to update our monitor
def monkey_patch_kb_builder():
    """Patch the knowledge base builder functions to update our monitor"""
    try:
        import knowledge_base.knowledge_base_builder as kb
        
        # Store the original build_knowledge_base function
        original_build_kb = kb.build_knowledge_base
        
        # Create a patched version that updates our monitor
        def patched_build_kb(cfg, *args, **kwargs):
            update_state(current_phase='building knowledge base')
            return original_build_kb(cfg, *args, **kwargs)
        
        # Replace the function
        kb.build_knowledge_base = patched_build_kb
        
        # Also patch the process_pdf function
        from knowledge_base.alternative_pdf_processor import process_pdf as original_process_pdf
        
        def patched_process_pdf(pdf_path, cfg=None):
            update_state(current_file=os.path.basename(pdf_path), current_phase='processing PDF')
            return original_process_pdf(pdf_path, cfg)
            
        import knowledge_base.alternative_pdf_processor as alt_pdf
        alt_pdf.process_pdf = patched_process_pdf
        
        # Import tqdm and monkey-patch it
        try:
            from tqdm import tqdm as original_tqdm
            
            # Create a wrapper that updates our monitor
            def patched_tqdm(*args, **kwargs):
                # Extract total items if available
                if args and len(args) > 0 and hasattr(args[0], '__len__'):
                    total_items = len(args[0])
                    if kwargs.get('desc', '').startswith('Processing PDFs'):
                        update_state(total_pdfs=total_items)
                elif 'total' in kwargs:
                    total_items = kwargs['total']
                    if kwargs.get('desc', '').startswith('Generating embeddings'):
                        # This is a chunk embedding progress bar
                        pass
                
                # Create the original tqdm instance
                tqdm_instance = original_tqdm(*args, **kwargs)
                
                # Store the original update method
                original_update = tqdm_instance.update
                
                # Create a patched update method
                def patched_update(n=1):
                    # Call the original update
                    result = original_update(n)
                    
                    # Update our monitor based on the progress bar type
                    if tqdm_instance.desc.startswith('Processing PDFs'):
                        update_state(processed_pdfs=tqdm_instance.n, current_phase='processing PDFs')
                    elif tqdm_instance.desc.startswith('Generating'):
                        update_state(processed_chunks=monitor_state['processed_chunks'] + n, 
                                     current_phase='generating embeddings')
                    
                    # Force flush stdout to ensure progress is visible
                    sys.stdout.flush()
                    
                    return result
                
                # Replace the update method
                tqdm_instance.update = patched_update
                
                return tqdm_instance
            
            # Replace tqdm in the kb module
            import tqdm as tqdm_module
            tqdm_module.tqdm = patched_tqdm
            kb.tqdm = patched_tqdm
            
            # Also patch the tqdm.write function
            original_write = tqdm_module.tqdm.write
            
            def patched_write(s, *args, **kwargs):
                # Call the original write function
                result = original_write(s, *args, **kwargs)
                
                # Also update our monitor based on the content
                # Detect specific messages
                if "Generated" in s and "chunks" in s:
                    # Extract number of chunks
                    import re
                    chunks_match = re.search(r'Generated (\d+) chunks', s)
                    if chunks_match:
                        chunk_count = int(chunks_match.group(1))
                        update_state(total_chunks=monitor_state['total_chunks'] + chunk_count,
                                     current_phase='preparing to generate embeddings')
                
                # Force flush stdout
                sys.stdout.flush()
                
                return result
            
            # Replace tqdm.write
            tqdm_module.tqdm.write = patched_write
            
        except ImportError:
            print("[MONITOR] tqdm not available, using basic progress monitoring only")
    
    except ImportError as e:
        print(f"[MONITOR] Error patching KB builder: {e}")
        pass

def main():
    parser = argparse.ArgumentParser(description="Build knowledge base with reliable progress monitoring")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--force", action="store_true", help="Force rebuilding knowledge base even if files exist")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU with memory limits instead of CPU-only mode")
    parser.add_argument("--skip-monitor", action="store_true", help="Skip the progress monitor thread")
    args = parser.parse_args()
    
    # Ensure necessary directories exist
    data_dir = Path("data")
    output_dir = Path("output")
    kb_dir = output_dir / "knowledge_base"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(kb_dir, exist_ok=True)
    
    # Apply monkey patches
    monkey_patch_kb_builder()
    
    # Modify config directly if use-gpu flag is set
    config_arg = None
    tmp_config_path = None
    
    if args.use_gpu:
        print("Setting config to use GPU with memory limits")
        # Import here to avoid circular imports
        from utils.config_loader import load_config, DEFAULT_CONFIG_PATH
        
        # Load config file
        config_path = args.config if args.config else DEFAULT_CONFIG_PATH
        cfg = load_config(config_path)
        
        # Modify memory settings
        cfg.knowledge_base.low_memory_mode = False
        
        # Save temporary config
        from omegaconf import OmegaConf
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False, mode='w') as tmp:
            tmp_config_path = tmp.name
            yaml_str = OmegaConf.to_yaml(cfg)
            tmp.write(yaml_str)
        
        # Use the temporary config
        config_arg = tmp_config_path
    else:
        # Use original config
        config_arg = args.config
    
    # Start the monitor thread if not skipped
    monitor_thread = None
    if not args.skip_monitor:
        monitor_thread = threading.Thread(target=progress_monitor_thread, daemon=True)
        monitor_thread.start()
    
    # Import KB builder module and call it directly
    start_time = time.time()
    update_state(start_time=start_time)
    
    print("\n")
    print("=" * 80)
    print("STARTING KNOWLEDGE BASE BUILDING PROCESS")
    print("=" * 80)
    print("This will run directly (not as a subprocess) to ensure progress is visible")
    print("A separate monitor thread will report progress every 5 seconds")
    print("=" * 80)
    print("\n")
    
    # Import and run KB builder
    try:
        # Add project root to path
        sys.path.insert(0, os.path.abspath('.'))
        
        # Import KB builder
        from knowledge_base.knowledge_base_builder import build_knowledge_base
        
        # Import config loader if needed
        if not config_arg:
            from utils.config_loader import load_config, DEFAULT_CONFIG_PATH
            cfg = load_config(DEFAULT_CONFIG_PATH)
        else:
            from utils.config_loader import load_config
            cfg = load_config(config_arg)
        
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
        
        # Update state before running the builder
        update_state(current_phase='starting knowledge base build')
        
        # Run the KB builder
        success = build_knowledge_base(cfg)
        
        # Clean up temporary config if created
        if args.use_gpu and tmp_config_path and os.path.exists(tmp_config_path):
            os.unlink(tmp_config_path)
        
        # Display result
        end_time = time.time()
        duration = end_time - start_time
        
        # Stop the monitor
        update_state(running=False)
        if monitor_thread:
            monitor_thread.join(1.0)  # Wait up to 1 second for it to finish
        
        print("\n")
        print("=" * 80)
        if success:
            print(f"KNOWLEDGE BASE SUCCESSFULLY BUILT IN {duration:.2f} SECONDS")
        else:
            print("KNOWLEDGE BASE BUILD FAILED")
        print("=" * 80)
        
        return 0 if success else 1
        
    except Exception as e:
        import traceback
        print(f"Error building knowledge base: {e}")
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