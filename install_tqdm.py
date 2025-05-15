#!/usr/bin/env python3
"""
Simple script to ensure tqdm is installed.
This will try to install tqdm if it's not already available.
"""

import sys
import subprocess
import importlib.util

def is_package_installed(package_name):
    """Check if a package is installed"""
    return importlib.util.find_spec(package_name) is not None

def install_package(package_name):
    """Install a package using pip"""
    print(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Successfully installed {package_name}!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package_name}: {e}")
        return False

if __name__ == "__main__":
    # Check if tqdm is already installed
    if is_package_installed("tqdm"):
        print("tqdm is already installed.")
    else:
        # Try to install tqdm
        if install_package("tqdm"):
            print("Progress bars will now be available for the knowledge base builder.")
        else:
            print("\nCouldn't install tqdm automatically.")
            print("To manually install, try running:")
            print("    pip install tqdm")
            print("or:")
            print("    python -m pip install tqdm")
            print("\nThe script will still run without progress bars.")
    
    print("\nNow you can run the knowledge base builder with progress bars enabled.") 