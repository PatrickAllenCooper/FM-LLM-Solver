#!/usr/bin/env python
"""
Script to run the KB builder with the Mathpix pipeline.
This script loads Mathpix API credentials from a .env file.
"""

import os
import sys
import argparse
import subprocess
import yaml
from pathlib import Path

def load_env_file(env_path=".env"):
    """Load environment variables from a .env file"""
    env_path = Path(env_path)
    if not env_path.exists():
        print(f"Warning: .env file not found at {env_path}")
        return False
    
    # Simple .env file parser
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ[key] = value
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Run KB builder with Mathpix pipeline")
    parser.add_argument("--app-id", help="Mathpix App ID (overrides .env)")
    parser.add_argument("--app-key", help="Mathpix App Key (overrides .env)")
    parser.add_argument("--env-file", default=".env", help="Path to .env file with Mathpix credentials")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of PDFs to process in each batch")
    parser.add_argument("--force", action="store_true", help="Force rebuild the entire knowledge base")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug logging")
    args = parser.parse_args()
    
    # Load from .env file first
    load_env_file(args.env_file)
    
    # Override with command-line arguments if provided
    if args.app_id:
        os.environ["MATHPIX_APP_ID"] = args.app_id
    if args.app_key:
        os.environ["MATHPIX_APP_KEY"] = args.app_key
    
    # Check if credentials are set
    app_id = os.environ.get("MATHPIX_APP_ID")
    app_key = os.environ.get("MATHPIX_APP_KEY")
    
    if not app_id or not app_key:
        print("Error: Mathpix credentials not found.")
        print("Please either:")
        print("  1. Create a .env file with MATHPIX_APP_ID and MATHPIX_APP_KEY")
        print("  2. Provide --app-id and --app-key as command-line arguments")
        return 1
    
    # Ensure PYTHONUNBUFFERED is set
    os.environ["PYTHONUNBUFFERED"] = "1"
    
    # Create a temporary config file with mathpix pipeline
    config_path = Path("config.yaml")
    temp_config_path = Path("mathpix_config.yaml")
    
    try:
        # Load the config
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        # Set pipeline to mathpix
        cfg["knowledge_base"]["pipeline"] = "mathpix"
        
        # Write temporary config
        with open(temp_config_path, "w") as f:
            yaml.dump(cfg, f)
        
        print("=" * 80)
        print("RUNNING KNOWLEDGE BASE BUILDER WITH MATHPIX PIPELINE")
        print("=" * 80)
        print(f"Using Mathpix App ID: {app_id[:4]}..." if len(app_id) > 4 else app_id)
        print(f"Batch size: {args.batch_size}")
        print("=" * 80)
        
        # Build command
        cmd = [
            sys.executable,  # Current Python interpreter
            "kb_builder.py",
            "--config", str(temp_config_path),
            "--batch-size", str(args.batch_size)
        ]
        
        if args.force:
            cmd.append("--force")
        
        if args.debug:
            cmd.append("--debug")
        
        # Run the KB builder
        result = subprocess.run(cmd)
        
        # Return the same exit code
        return result.returncode
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    finally:
        # Clean up temporary config
        if temp_config_path.exists():
            temp_config_path.unlink()

if __name__ == "__main__":
    sys.exit(main()) 