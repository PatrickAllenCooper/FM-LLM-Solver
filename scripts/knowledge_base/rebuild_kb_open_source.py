#!/usr/bin/env python3
"""
Force rebuild of the knowledge base using the open source pipeline.

This script:
1. Updates the config to use the open source pipeline
2. Removes existing knowledge base files
3. Runs the knowledge base builder
"""

import os
import sys
import logging
import shutil
import subprocess
from omegaconf import OmegaConf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rebuild_kb.log')
    ]
)

def load_config(config_path):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    try:
        cfg = OmegaConf.load(config_path)
        logging.info(f"Loaded configuration from {config_path}")
        return cfg
    except Exception as e:
        logging.error(f"Failed to load config: {str(e)}")
        sys.exit(1)

def update_config(cfg):
    """Update config to use open source pipeline"""
    cfg.knowledge_base.pipeline = "open_source"
    logging.info("Updated configuration to use open source pipeline")
    return cfg

def save_config(cfg, config_path):
    """Save updated configuration"""
    try:
        OmegaConf.save(cfg, config_path)
        logging.info(f"Saved updated configuration to {config_path}")
    except Exception as e:
        logging.error(f"Failed to save config: {str(e)}")
        sys.exit(1)

def clean_knowledge_base(cfg):
    """Remove existing knowledge base files"""
    kb_dir = cfg.output.knowledge_base_dir
    
    if os.path.exists(kb_dir):
        logging.info(f"Cleaning knowledge base directory: {kb_dir}")
        
        # Remove the specific files rather than the whole directory
        for filename in os.listdir(kb_dir):
            file_path = os.path.join(kb_dir, filename)
            if os.path.isfile(file_path):
                try:
                    os.unlink(file_path)
                    logging.info(f"Removed {filename}")
                except Exception as e:
                    logging.error(f"Error removing {filename}: {str(e)}")
    else:
        logging.info(f"Creating knowledge base directory: {kb_dir}")
        os.makedirs(kb_dir, exist_ok=True)

def run_kb_build():
    """Run knowledge base builder"""
    logging.info("Starting knowledge base build process...")
    
    try:
        # Run with only KB build flag
        cmd = [sys.executable, "run_experiments.py", "--only-kb-build"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logging.info("Knowledge base build completed successfully")
        logging.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Knowledge base build failed: {str(e)}")
        logging.error(e.stderr)
        return False

def main():
    logging.info("=== Starting Knowledge Base Rebuild with Open Source Pipeline ===")
    
    # Load configuration
    config_path = "config.yaml"
    cfg = load_config(config_path)
    
    # Update configuration
    cfg = update_config(cfg)
    save_config(cfg, config_path)
    
    # Clean knowledge base directory
    clean_knowledge_base(cfg)
    
    # Run KB build
    success = run_kb_build()
    
    if success:
        logging.info("=== Knowledge Base Rebuild Completed Successfully ===")
    else:
        logging.error("=== Knowledge Base Rebuild Failed ===")
        sys.exit(1)

if __name__ == "__main__":
    main() 