#!/usr/bin/env python
"""
Environment setup script for FM-LLM-Solver.
This script ensures all dependencies are properly installed, including PyTorch with CUDA support.
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('setup.log')
    ]
)

def detect_hardware():
    """Detect the hardware platform"""
    hardware_info = {
        "is_apple_silicon": False,
        "has_gpu": False,
        "cpu_cores": os.cpu_count() or 1,
        "system": platform.system(),
        "machine": platform.machine()
    }
    
    # Check for Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        hardware_info["is_apple_silicon"] = True
        logging.info("Detected Apple Silicon (M-series chip)")
    
    # Check for CUDA on non-Apple systems
    if not hardware_info["is_apple_silicon"]:
        try:
            # Try to use nvidia-smi to check for GPU
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                hardware_info["has_gpu"] = True
                logging.info("NVIDIA GPU detected")
        except (FileNotFoundError, subprocess.SubprocessError):
            # Try fallback to check for CUDA in Python
            try:
                import torch
                if torch.cuda.is_available():
                    hardware_info["has_gpu"] = True
                    logging.info(f"CUDA capable GPU detected: {torch.cuda.get_device_name(0)}")
            except (ImportError, AttributeError):
                pass
    
    return hardware_info

def create_env_template():
    """Create a template .env file with instructions"""
    if os.path.exists('.env'):
        logging.info(".env file already exists. Not overwriting.")
        return
    
    template = """# API Credentials
# Replace these with your actual credentials

# Mathpix API credentials
MATHPIX_APP_ID=YOUR_APP_ID_HERE
MATHPIX_APP_KEY=YOUR_APP_KEY_HERE

# OpenAI API key (for fine-tuning/evaluation)
OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
"""
    
    with open('.env', 'w') as f:
        f.write(template)
    
    logging.info("Created .env template file. Please edit it with your actual API credentials.")

def install_dependencies(hardware_info):
    """Install dependencies based on hardware"""
    requirements_file = "requirements.txt"
    
    # Additional packages based on hardware
    apple_silicon_packages = [
        "tensorflow-macos>=2.9.0",     # TensorFlow for Mac
        "tensorflow-metal>=0.5.0",     # Metal acceleration for TensorFlow
    ]
    
    gpu_packages = [
        "tensorflow",       # Modern TensorFlow includes GPU support when available
        # Using PyTorch pip command instead of direct package specification due to +cu118 syntax issues
    ]
    
    # Install base requirements
    logging.info(f"Installing base requirements from {requirements_file}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_file])
    
    # Install hardware-specific packages
    if hardware_info["is_apple_silicon"]:
        logging.info("Installing Apple Silicon optimized packages")
        for package in apple_silicon_packages:
            try:
                logging.info(f"Installing {package}")
                subprocess.run([sys.executable, "-m", "pip", "install", package])
            except subprocess.SubprocessError as e:
                logging.warning(f"Failed to install {package}: {e}")
    
    elif hardware_info["has_gpu"]:
        logging.info("Installing GPU optimized packages")
        for package in gpu_packages:
            try:
                logging.info(f"Installing {package}")
                subprocess.run([sys.executable, "-m", "pip", "install", package])
            except subprocess.SubprocessError as e:
                logging.warning(f"Failed to install {package}: {e}")
        
        # Install PyTorch with CUDA support using the recommended command
        try:
            logging.info("Installing PyTorch with CUDA support")
            subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"])
        except subprocess.SubprocessError as e:
            logging.warning(f"Failed to install PyTorch with CUDA: {e}")

def prepare_directories():
    """Create necessary directories"""
    dirs = [
        "data/fetched_papers",
        "output/knowledge_base",
        "output/finetuning_results"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Created directory: {dir_path}")

def main():
    parser = argparse.ArgumentParser(description="Setup environment for FM-LLM-Solver")
    parser.add_argument("--create-env-template", action="store_true", help="Create a template .env file")
    parser.add_argument("--skip-dependencies", action="store_true", help="Skip installing dependencies")
    args = parser.parse_args()
    
    logging.info("Starting environment setup")
    
    # Create .env template if requested
    if args.create_env_template:
        create_env_template()
    
    # Create directories
    prepare_directories()
    
    # Detect hardware and install dependencies
    if not args.skip_dependencies:
        hardware_info = detect_hardware()
        install_dependencies(hardware_info)
    
    logging.info("Environment setup complete")
    logging.info(f"System: {platform.system()} {platform.release()} {platform.machine()}")
    logging.info("Run 'python run_experiments.py' to start the pipeline")
    
if __name__ == "__main__":
    main() 