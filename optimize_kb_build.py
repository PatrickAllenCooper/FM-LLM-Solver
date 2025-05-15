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

def main():
    parser = argparse.ArgumentParser(description="Optimize and run knowledge base builder")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--force", action="store_true", help="Force rebuilding knowledge base even if files exist")
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
        
        # Now run the knowledge base builder directly
        print("\nStep 2: Running knowledge base builder with optimized chunking...")
        
        # Import config
        from utils.config_loader import load_config, DEFAULT_CONFIG_PATH
        config_path = args.config if args.config else DEFAULT_CONFIG_PATH
        cfg = load_config(config_path)
        
        # Set optimal memory settings
        cfg.knowledge_base.low_memory_mode = False
        cfg.knowledge_base.gpu_memory_limit = 4096
        
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