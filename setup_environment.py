#!/usr/bin/env python
"""
Environment setup script for FM-LLM-Solver.
This script ensures all dependencies are properly installed, including PyTorch with CUDA support.
"""

import os
import sys
import subprocess
import platform

def check_cuda():
    """Check if CUDA is available on the system."""
    try:
        # Simple command to check if nvidia-smi is available
        subprocess.run(['nvidia-smi'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ CUDA-compatible GPU detected")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("⚠️ No CUDA-compatible GPU detected or drivers not installed")
        return False

def install_pytorch(cuda_available):
    """Install PyTorch with or without CUDA support."""
    if cuda_available:
        # Install PyTorch with CUDA 11.8 support
        cmd = [sys.executable, '-m', 'pip', 'install', 
               'torch>=2.0', 'torchvision', 'torchaudio', 
               '--index-url', 'https://download.pytorch.org/whl/cu118']
        print("📦 Installing PyTorch with CUDA 11.8 support...")
    else:
        # Install CPU-only PyTorch
        cmd = [sys.executable, '-m', 'pip', 'install', 
               'torch>=2.0', 'torchvision', 'torchaudio']
        print("📦 Installing CPU-only PyTorch (no GPU acceleration)...")
    
    subprocess.run(cmd, check=True)

def install_requirements():
    """Install other requirements excluding PyTorch packages."""
    print("📦 Installing other dependencies from requirements.txt...")
    
    # Install everything except torch, torchvision, torchaudio
    cmd = [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt']
    subprocess.run(cmd, check=True)

def verify_installation():
    """Verify PyTorch is installed with CUDA if available."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print("\n=== PyTorch Installation Summary ===")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA: Not available")
            
        if 'cpu' in torch.__version__ and check_cuda():
            print("\n⚠️ WARNING: CUDA-compatible GPU detected, but PyTorch was installed without CUDA support")
            print("   Recommendation: Run this script again with the --force-reinstall flag")
            
        return cuda_available
    except ImportError:
        print("❌ Failed to import PyTorch. Installation may have failed.")
        return False

def main():
    """Main setup function."""
    print("=== FM-LLM-Solver Environment Setup ===")
    
    # Check if --force-reinstall flag is provided
    force_reinstall = "--force-reinstall" in sys.argv
    
    # Check CUDA availability
    cuda_available = check_cuda()
    
    # Uninstall existing PyTorch if forced
    if force_reinstall:
        print("🔄 Removing existing PyTorch installation...")
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torch', 'torchvision', 'torchaudio'])
    
    # Install PyTorch with appropriate CUDA support
    install_pytorch(cuda_available)
    
    # Install other requirements
    install_requirements()
    
    # Verify installation
    success = verify_installation()
    
    print("\n=== Setup Complete ===")
    if success and cuda_available:
        print("✅ Environment successfully set up with GPU acceleration")
    elif success:
        print("✅ Environment successfully set up (CPU only)")
    else:
        print("⚠️ Setup completed with warnings. Please check the logs above.")

if __name__ == "__main__":
    main() 