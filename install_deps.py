#!/usr/bin/env python
"""
Installation script for tqdm progress bar and other dependencies.
This script detects your Python environment and installs tqdm.
"""

import os
import sys
import subprocess
import importlib.util
import argparse

def is_package_installed(package_name):
    """Check if a package is installed"""
    return importlib.util.find_spec(package_name) is not None

def get_active_environment():
    """Detect the active Python environment (conda, venv, etc.)"""
    env_info = {
        "type": "system",
        "path": sys.executable
    }

    # Check for conda environment
    if "CONDA_PREFIX" in os.environ:
        env_info["type"] = "conda"
        env_info["name"] = os.environ.get("CONDA_DEFAULT_ENV", "base")
        env_info["prefix"] = os.environ.get("CONDA_PREFIX")
        print(f"Detected conda environment: {env_info['name']}")
    
    # Check for virtualenv/venv
    elif sys.prefix != sys.base_prefix:
        env_info["type"] = "venv"
        env_info["prefix"] = sys.prefix
        print(f"Detected Python virtual environment at: {env_info['prefix']}")
    
    else:
        print(f"Using system Python at: {env_info['path']}")
    
    return env_info

def install_package(package_name, env_info):
    """Install a package using the appropriate method for the detected environment"""
    print(f"Installing {package_name}...")
    try:
        if env_info["type"] == "conda":
            # Conda has priority - try conda install first, then pip if it fails
            try:
                cmd = ["conda", "install", "-y", package_name]
                print(f"Running: {' '.join(cmd)}")
                subprocess.check_call(cmd)
                return True
            except subprocess.CalledProcessError:
                print(f"Conda install failed, trying pip...")
                cmd = [sys.executable, "-m", "pip", "install", package_name]
                print(f"Running: {' '.join(cmd)}")
                subprocess.check_call(cmd)
                return True
        else:
            # Use pip for venv or system Python
            cmd = [sys.executable, "-m", "pip", "install", package_name]
            print(f"Running: {' '.join(cmd)}")
            subprocess.check_call(cmd)
            return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Install tqdm and other dependencies for FM-LLM-Solver")
    parser.add_argument("--force", action="store_true", help="Force reinstall even if already installed")
    args = parser.parse_args()
    
    # Detect environment
    env_info = get_active_environment()
    
    # Check if tqdm is already installed
    if is_package_installed("tqdm") and not args.force:
        print("tqdm is already installed.")
    else:
        if args.force:
            print("Forcing reinstall of tqdm...")
        # Try to install tqdm
        if install_package("tqdm", env_info):
            print("Progress bars will now be available for the knowledge base builder.")
        else:
            print("\nCouldn't install tqdm automatically.")
            print("To manually install, try one of these commands:")
            
            if env_info["type"] == "conda":
                print(f"    conda install -y tqdm")
            print(f"    {sys.executable} -m pip install tqdm")
            
            print("\nThe script will still run without progress bars using a fallback.")
    
    print("\nSetup complete - you can now run the knowledge base builder with progress bars.")

if __name__ == "__main__":
    main() 