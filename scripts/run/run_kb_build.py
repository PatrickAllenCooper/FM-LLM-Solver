#!/usr/bin/env python
"""
Knowledge Base Builder wrapper script.
This script directly builds the knowledge base without using subprocesses,
ensuring all progress bars and real-time updates are displayed properly.
"""

import os
import sys
from pathlib import Path
import argparse
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Make sure Python's output is unbuffered
os.environ['PYTHONUNBUFFERED'] = '1'

def main():
    parser = argparse.ArgumentParser(description="Build knowledge base with progress bars")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--force", action="store_true", help="Force rebuilding knowledge base even if files exist")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU with memory limits instead of CPU-only mode")
    args = parser.parse_args()
    
    # Ensure necessary directories exist
    data_dir = Path("data")
    output_dir = Path("output")
    kb_dir = output_dir / "knowledge_base"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(kb_dir, exist_ok=True)
    
    # Modify config directly if use-gpu flag is set
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
    
    # Import KB builder module and call it directly
    start_time = time.time()
    print("\n")
    print("=" * 80)
    print("STARTING KNOWLEDGE BASE BUILDING PROCESS")
    print("=" * 80)
    print("This will run directly (not as a subprocess) to ensure progress is visible")
    print("You should see real-time progress bars for the embedding process")
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
        
        # Run the KB builder
        success = build_knowledge_base(cfg)
        
        # Clean up temporary config if created
        if args.use_gpu and os.path.exists(tmp_config_path):
            os.unlink(tmp_config_path)
        
        # Display result
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n")
        print("=" * 80)
        if success:
            print(f"KNOWLEDGE BASE SUCCESSFULLY BUILT IN {duration:.2f} SECONDS")
        else:
            print("KNOWLEDGE BASE BUILD FAILED")
        print("=" * 80)
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Error building knowledge base: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 