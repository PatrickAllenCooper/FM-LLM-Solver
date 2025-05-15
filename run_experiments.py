#!/usr/bin/env python3
"""
Unified experiment runner for FMLLMSolver.

This script provides a streamlined interface for running barrier certificate generation experiments.
It handles the entire pipeline from data fetching to evaluation with a single command.
"""

import os
import sys
import argparse
import logging
import time
import json
import subprocess
from dotenv import load_dotenv
from pathlib import Path
from utils.config_loader import load_config, DEFAULT_CONFIG_PATH
from omegaconf import OmegaConf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("experiment_run.log"),
    ],
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
dotenv_path = Path(".env")
if dotenv_path.exists():
    logger.info(f"Loading environment variables from {dotenv_path}")
    load_dotenv(dotenv_path)
else:
    logger.info("No .env file found. Using system environment variables or values from config.")

def setup_experiment_env(cfg, args):
    """
    Set up the experiment environment based on config and command-line arguments.
    Creates necessary directories and sets environment variables.
    
    Parameters
    ----------
    cfg : OmegaConf.DictConfig
        The loaded configuration object
    args : argparse.Namespace
        Command-line arguments
    """
    # Create necessary directories
    logger.info("Creating necessary directories...")
    os.makedirs(cfg.paths.data_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.paths.data_dir, "fetched_papers"), exist_ok=True)
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    os.makedirs(cfg.paths.kb_output_dir, exist_ok=True)
    os.makedirs(cfg.paths.ft_output_dir, exist_ok=True)
    
    # Check PyTorch CUDA support if fine-tuning is required
    if not args.skip_finetuning:
        logger.info("Verifying PyTorch CUDA installation...")
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if not cuda_available:
                # Check if we have CPU-only PyTorch but CUDA hardware is available
                try:
                    # Try to check if CUDA hardware is available with nvidia-smi
                    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if result.returncode == 0:
                        # We have CUDA hardware but PyTorch doesn't see it - need to reinstall
                        logger.warning("CUDA hardware detected but PyTorch was installed without CUDA support!")
                        if not args.skip_deps_check:
                            logger.info("Running environment setup script to fix PyTorch installation...")
                            setup_cmd = [sys.executable, "setup_environment.py", "--force-reinstall"]
                            subprocess.run(setup_cmd, check=True)
                            
                            # Verify again
                            import importlib
                            importlib.reload(torch)
                            if not torch.cuda.is_available():
                                logger.error("Failed to install PyTorch with CUDA support. Fine-tuning will likely fail.")
                except (subprocess.SubprocessError, FileNotFoundError):
                    # CUDA hardware not available
                    logger.warning("PyTorch installed without CUDA support, and no CUDA hardware detected.")
                    logger.warning("Fine-tuning will be slow without GPU acceleration.")
            else:
                logger.info(f"PyTorch CUDA support verified. Using device: {torch.cuda.get_device_name(0)}")
        except ImportError:
            logger.error("PyTorch not installed. Please run setup_environment.py first.")
            if not args.skip_deps_check:
                logger.info("Running environment setup script...")
                setup_cmd = [sys.executable, "setup_environment.py"]
                subprocess.run(setup_cmd, check=True)
    
    # Set environment variables from config
    if args.env_from_config:
        logger.info("Setting environment variables from config...")
        if 'env_vars' in cfg:
            for key, value in cfg.env_vars.items():
                if value and str(value).strip():
                    os.environ[key] = str(value)
                    logger.info(f"Set environment variable: {key}")
        else:
            logger.warning("No env_vars section found in config.")

def fetch_data(cfg, args):
    """
    Execute the data fetching step based on config.
    
    Parameters
    ----------
    cfg : OmegaConf.DictConfig
        The loaded configuration object
    args : argparse.Namespace
        Command-line arguments
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    if args.skip_data_fetching:
        logger.info("Skipping data fetching step.")
        return True
        
    logger.info("Starting data fetching...")
    
    if 'UNPAYWALL_EMAIL' not in os.environ:
        logger.error("UNPAYWALL_EMAIL environment variable not set. Required for data fetching.")
        return False
        
    try:
        cmd = [sys.executable, "data_fetching/paper_fetcher.py"]
        config_path = args.config if args.config else DEFAULT_CONFIG_PATH
        cmd.extend(["--config", config_path])
            
        logger.info(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info("Data fetching completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Data fetching failed with error code {e.returncode}.")
        return False

def build_knowledge_base(cfg, args):
    """
    Execute the knowledge base building step based on config.
    Checks if KB files already exist before running the builder.
    
    Parameters
    ----------
    cfg : OmegaConf.DictConfig
        The loaded configuration object
    args : argparse.Namespace
        Command-line arguments
        
    Returns
    -------
    bool
        True if successful or skipped, False otherwise
    """
    if args.skip_kb_building:
        logger.info("Skipping knowledge base building step.")
        return True
        
    # Check if knowledge base files already exist
    kb_dir = cfg.paths.kb_output_dir
    vector_store_path = Path(kb_dir) / Path(cfg.paths.kb_vector_store_filename)
    metadata_path = Path(kb_dir) / Path(cfg.paths.kb_metadata_filename)
    
    if vector_store_path.exists() and metadata_path.exists():
        logger.info(f"Knowledge base files found ({vector_store_path.name}, {metadata_path.name}). Skipping build.")
        return True
    else:
        logger.info("Knowledge base files not found. Proceeding with build...")
        
    logger.info("Starting knowledge base building...")
    
    # Check for required environment variables if using Mathpix
    if cfg.knowledge_base.pipeline.lower() == "mathpix" and ('MATHPIX_APP_ID' not in os.environ or 'MATHPIX_APP_KEY' not in os.environ):
        logger.error("MATHPIX_APP_ID and/or MATHPIX_APP_KEY environment variables not set. Required for Mathpix pipeline.")
        return False
        
    try:
        cmd = [sys.executable, "knowledge_base/knowledge_base_builder.py"]
        config_path = args.config if args.config else DEFAULT_CONFIG_PATH
        cmd.extend(["--config", config_path])
            
        logger.info(f"Executing: {' '.join(cmd)}")
        
        # Detect Windows and use a different approach for subprocess management
        if os.name == 'nt':  # Windows
            logger.info("Using Windows-specific subprocess handling for real-time output")
            import msvcrt
            
            # Windows-specific: run the command directly
            # We don't use Popen with read loops as it often hangs in Windows
            os.environ['PYTHONUNBUFFERED'] = '1'  # Force Python to unbuffer output
            
            # Run the subprocess with direct output
            result = subprocess.run(
                cmd, 
                check=False,  # Don't raise exception on non-zero exit
                env=dict(os.environ, PYTHONUNBUFFERED='1')  # Ensure unbuffered output
            )
            
            exit_code = result.returncode
        else:
            # Unix/Linux/Mac approach - use Popen with real-time output
            with subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,  # Line buffered
                env=dict(os.environ, PYTHONUNBUFFERED='1')  # Ensure unbuffered output
            ) as process:
                # Read output in real-time and print it
                for line in process.stdout:
                    # Display the line without extra newlines
                    print(line.rstrip(), flush=True)
                    
                # Wait for completion
                exit_code = process.wait()
            
        # Process result
        if exit_code == 0:
            logger.info("Knowledge base building script finished successfully.")
            return True
        else:
            logger.error(f"Knowledge base building script failed with exit code {exit_code}.")
            return False
    except Exception as e:
        logger.error(f"Error executing knowledge base building script: {e}", exc_info=True)
        return False

def prepare_finetune_data(cfg, args):
    """
    Prepare the fine-tuning data by combining datasets.
    
    Parameters
    ----------
    cfg : OmegaConf.DictConfig
        The loaded configuration object
    args : argparse.Namespace
        Command-line arguments
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    if args.skip_data_prep:
        logger.info("Skipping fine-tuning data preparation step.")
        return True
        
    logger.info("Starting fine-tuning data preparation...")
    
    try:
        cmd = [sys.executable, "fine_tuning/combine_datasets.py"]
        config_path = args.config if args.config else DEFAULT_CONFIG_PATH
        cmd.extend(["--config", config_path])
            
        logger.info(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info("Fine-tuning data preparation completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Fine-tuning data preparation failed with error code {e.returncode}.")
        return False

def finetune_model(cfg, args):
    """
    Execute the model fine-tuning step based on config.
    
    Parameters
    ----------
    cfg : OmegaConf.DictConfig
        The loaded configuration object
    args : argparse.Namespace
        Command-line arguments
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    if args.skip_finetuning:
        logger.info("Skipping model fine-tuning step.")
        return True
        
    logger.info("Starting model fine-tuning...")
    
    try:
        cmd = [sys.executable, "fine_tuning/finetune_llm.py"]
        config_path = args.config if args.config else DEFAULT_CONFIG_PATH
        cmd.extend(["--config", config_path])
            
        logger.info(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info("Model fine-tuning completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Model fine-tuning failed with error code {e.returncode}.")
        return False

def run_experiment(cfg, args):
    """
    Run the full experiment using the evaluation pipeline.
    
    Parameters
    ----------
    cfg : OmegaConf.DictConfig
        The loaded configuration object
    args : argparse.Namespace
        Command-line arguments
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    logger.info("Starting experiment evaluation...")
    
    try:
        cmd = [sys.executable, "evaluation/evaluate_pipeline.py"]
        config_path = args.config if args.config else DEFAULT_CONFIG_PATH
        cmd.extend(["--config", config_path])
        
        # Add additional arguments from command line
        if args.rag_k:
            cmd.extend(["-k", str(args.rag_k)])
        if args.results_file:
            cmd.extend(["--results", args.results_file])
        if args.benchmark_file:
            cmd.extend(["--benchmark", args.benchmark_file])
            
        logger.info(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info("Experiment evaluation completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Experiment evaluation failed with error code {e.returncode}.")
        return False

def full_pipeline(cfg, args):
    """Run the full experiment pipeline."""
    start_time = time.time()
    
    if args.test_example:
        # Run a single test example
        logger.info("Running test example...")
        test_cmd = [
            sys.executable, 
            "inference/generate_certificate.py",
            "System Dynamics: dx/dt = -x**3 - y, dy/dt = x - y**3. Initial Set: x**2+y**2 <= 0.1. Unsafe Set: x >= 1.5"
        ]
        
        # Use default config path if not specified
        config_path = args.config if args.config else DEFAULT_CONFIG_PATH
        test_cmd.extend(["--config", config_path])
        
        # Add --skip-kb-check option if requested
        if args.skip_kb_check:
            test_cmd.append("--skip-kb-check")
            
        # Add --skip-adapter option if requested
        if args.skip_adapter:
            test_cmd.append("--skip-adapter")
        
        logger.info(f"Executing: {' '.join(test_cmd)}")
        subprocess.run(test_cmd)
        logger.info("Test example completed.")
        return
    
    # Step 1: Set up environment
    setup_experiment_env(cfg, args)
    
    # Step 2: Execute pipeline steps based on args
    steps = [
        ("Data fetching", fetch_data),
        ("Knowledge base building", build_knowledge_base),
        ("Fine-tuning data preparation", prepare_finetune_data),
        ("Model fine-tuning", finetune_model),
        ("Experiment evaluation", run_experiment)
    ]
    
    success = True
    for step_name, step_func in steps:
        logger.info(f"=== Starting step: {step_name} ===")
        step_success = step_func(cfg, args)
        if not step_success:
            logger.error(f"Step '{step_name}' failed. Pipeline will continue but results may be affected.")
            success = False
        logger.info(f"=== Completed step: {step_name} ===")
    
    # Log completion
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Full experiment pipeline completed in {duration:.2f} seconds.")
    
    if success:
        logger.info("All steps completed successfully.")
    else:
        logger.warning("Some steps encountered errors. Check the logs for details.")

def create_env_template():
    """
    Create a template .env file if it doesn't exist.
    """
    env_path = Path(".env")
    if not env_path.exists():
        with open(env_path, "w") as f:
            f.write("""# Environment variables for FMLLMSolver

# API keys for MathPix (required for knowledge base construction with 'mathpix' pipeline)
MATHPIX_APP_ID=YOUR_APP_ID_HERE
MATHPIX_APP_KEY=YOUR_APP_KEY_HERE

# Email for Unpaywall (required for data fetching)
UNPAYWALL_EMAIL=YOUR_EMAIL_HERE

# Optional: API key for Semantic Scholar (enhances paper fetching)
SEMANTIC_SCHOLAR_API_KEY=YOUR_SEMANTIC_SCHOLAR_KEY_HERE
""")
        logger.info(f"Created template .env file at {env_path.absolute()}")
        logger.info("Please edit this file to add your API keys and credentials.")
    else:
        logger.info(".env file already exists.")

def save_experiment_config(cfg, args, experiment_name):
    """
    Save the current experiment configuration.
    
    Parameters
    ----------
    cfg : OmegaConf.DictConfig
        The loaded configuration object
    args : argparse.Namespace
        Command-line arguments
    experiment_name : str
        Name for this experiment run
    """
    if not experiment_name:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    # Create experiments directory if it doesn't exist
    experiments_dir = Path("experiments")
    experiments_dir.mkdir(exist_ok=True)
    
    # Save args and config
    experiment_dir = experiments_dir / experiment_name
    experiment_dir.mkdir(exist_ok=True)
    
    # Save experiment metadata
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command_line_args": vars(args),
        "description": args.description or "No description provided"
    }
    
    with open(experiment_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Copy current config
    with open(experiment_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    logger.info(f"Experiment configuration saved to {experiment_dir}")
    return experiment_dir

def main():
    parser = argparse.ArgumentParser(description="Unified experiment runner for FMLLMSolver")
    
    # Configuration options
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--env-from-config", action="store_true", help="Load environment variables from config")
    parser.add_argument("--create-env-template", action="store_true", help="Create a template .env file")
    
    # Experiment identification
    parser.add_argument("--experiment-name", type=str, help="Name for this experiment run")
    parser.add_argument("--description", type=str, help="Description of this experiment")
    
    # Skip steps flags
    parser.add_argument("--skip-data-fetching", action="store_true", help="Skip data fetching step")
    parser.add_argument("--skip-kb-building", action="store_true", help="Skip knowledge base building step")
    parser.add_argument("--skip-data-prep", action="store_true", help="Skip fine-tuning data preparation step")
    parser.add_argument("--skip-finetuning", action="store_true", help="Skip model fine-tuning step")
    parser.add_argument("--skip-deps-check", action="store_true", help="Skip dependency checking (PyTorch CUDA)")
    
    # Run only specific steps
    parser.add_argument("--only-data-fetch", action="store_true", help="Only run data fetching")
    parser.add_argument("--only-kb-build", action="store_true", help="Only run knowledge base building")
    parser.add_argument("--only-finetune", action="store_true", help="Only run model fine-tuning")
    parser.add_argument("--only-evaluate", action="store_true", help="Only run evaluation")
    parser.add_argument("--test-example", action="store_true", help="Only run a single test example")
    parser.add_argument("--skip-kb-check", action="store_true", help="Skip knowledge base check when running test example")
    parser.add_argument("--skip-adapter", action="store_true", help="Skip loading the adapter and use only the base model")
    
    # Experiment parameters
    parser.add_argument("--rag-k", type=int, help="Number of context chunks to retrieve")
    parser.add_argument("--results-file", type=str, help="Path to save results CSV")
    parser.add_argument("--benchmark-file", type=str, help="Path to benchmark systems JSON")
    
    args = parser.parse_args()
    
    # Create .env template if requested
    if args.create_env_template:
        create_env_template()
        return
    
    # Load configuration
    config_path = args.config if args.config else DEFAULT_CONFIG_PATH
    logger.info(f"Loading configuration from: {config_path}")
    cfg = load_config(config_path)
    
    # Handle "only run specific step" flags
    if args.only_data_fetch:
        args.skip_kb_building = True
        args.skip_data_prep = True
        args.skip_finetuning = True
        args.test_example = False
    elif args.only_kb_build:
        args.skip_data_fetching = True
        args.skip_data_prep = True
        args.skip_finetuning = True
        args.test_example = False
    elif args.only_finetune:
        args.skip_data_fetching = True
        args.skip_kb_building = True
        args.skip_data_prep = False
        args.test_example = False
    elif args.only_evaluate:
        args.skip_data_fetching = True
        args.skip_kb_building = True
        args.skip_data_prep = True
        args.skip_finetuning = True
        args.test_example = False
    
    # Save experiment configuration if a name is provided
    if args.experiment_name:
        experiment_dir = save_experiment_config(cfg, args, args.experiment_name)
    
    # Run the full pipeline
    full_pipeline(cfg, args)

if __name__ == "__main__":
    main() 