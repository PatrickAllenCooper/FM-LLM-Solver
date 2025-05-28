#!/usr/bin/env python
"""
Script to optimize the knowledge base builder by patching the chunking function
and running the process.
"""

import os
import sys
import time
from pathlib import Path
import argparse

# Make sure Python's output is unbuffered
os.environ['PYTHONUNBUFFERED'] = '1'

# Set PyTorch CUDA allocation options to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.6'

def main():
    parser = argparse.ArgumentParser(description="Optimize and run knowledge base builder")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--force", action="store_true", help="Force rebuilding knowledge base even if files exist")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU usage even if GPU is available")
    args = parser.parse_args()
    
    print("\n")
    print("=" * 80)
    print("OPTIMIZING KNOWLEDGE BASE BUILDER AND RUNNING")
    print("=" * 80)
    print("This script will patch the chunking function with an optimized version")
    print("=" * 80)
    print("\n")
    
    # Ensure we can import from the project root
    sys.path.insert(0, os.path.abspath('.'))
    
    # Import our optimized chunker
    try:
        print("Step 1: Patching chunking function with optimized version...")
        
        # Import original module for patching
        from knowledge_base.alternative_pdf_processor import split_into_chunks as original_split
        import knowledge_base.alternative_pdf_processor as alt_pdf
        
        # Import our optimized version
        from knowledge_base.optimized_chunker import optimized_split_into_chunks
        
        # Patch the module
        print("Replacing original split_into_chunks with optimized version...")
        alt_pdf.split_into_chunks = optimized_split_into_chunks
        
        print("Chunking function patched successfully!")
        
        # Memory management patch - add this to the knowledge_base_builder module
        print("Adding memory management patches...")
        
        # Monkey-patch for aggressive cache clearing
        import knowledge_base.knowledge_base_builder as kb
        
        # Store original functions
        original_process_pdf = alt_pdf.process_pdf
        
        # Create a wrapper to clear CUDA cache after each PDF
        def clear_gpu_memory():
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    print("CUDA cache cleared")
            except:
                pass
        
        # Patch the process_pdf function to clear cache after each file
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
                    print(f"CUDA out of memory on {os.path.basename(pdf_path)}, attempting fallback...")
                    # Force garbage collection
                    import gc
                    gc.collect()
                    
                    # Try again with CPU mode
                    if cfg is not None:
                        cfg.knowledge_base.low_memory_mode = True  # Force CPU mode
                        print("Switched to CPU mode for this PDF")
                        try:
                            result = original_process_pdf(pdf_path, cfg)
                            clear_gpu_memory()
                            return result
                        except Exception as inner_e:
                            print(f"Fallback also failed: {inner_e}")
                
                # Re-raise the original exception if recovery failed
                raise
        
        # Apply the patched function
        alt_pdf.process_pdf = patched_process_pdf
        
        # Now run the knowledge base builder directly
        print("\nStep 2: Running knowledge base builder with optimized chunking...")
        
        # Import config
        from utils.config_loader import load_config, DEFAULT_CONFIG_PATH
        config_path = args.config if args.config else DEFAULT_CONFIG_PATH
        cfg = load_config(config_path)
        
        # Set optimal memory settings
        if args.cpu_only:
            print("Forcing CPU-only mode as requested")
            cfg.knowledge_base.low_memory_mode = True
        else:
            cfg.knowledge_base.low_memory_mode = False
            cfg.knowledge_base.gpu_memory_limit = 2048  # Reduce to 2GB to be safer
        
        # Clean existing KB files if requested
        if args.force:
            kb_dir = Path(cfg.paths.kb_output_dir)
            vector_path = kb_dir / cfg.paths.kb_vector_store_filename
            metadata_path = kb_dir / cfg.paths.kb_metadata_filename
            
            if vector_path.exists():
                print(f"Removing existing vector store: {vector_path}")
                vector_path.unlink()
                
            if metadata_path.exists():
                print(f"Removing existing metadata: {metadata_path}")
                metadata_path.unlink()
        
        # Run the builder
        from knowledge_base.knowledge_base_builder import build_knowledge_base
        start_time = time.time()
        success = build_knowledge_base(cfg)
        end_time = time.time()
        
        print("\n")
        print("=" * 80)
        if success:
            print(f"KNOWLEDGE BASE SUCCESSFULLY BUILT IN {end_time - start_time:.2f} SECONDS")
        else:
            print("KNOWLEDGE BASE BUILD FAILED")
        print("=" * 80)
        
        return 0 if success else 1
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 