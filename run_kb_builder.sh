#!/bin/bash
# Script to set up a compatible environment and run the KB builder

# Exit on error
set -e

echo "==============================================="
echo "Setting up environment for Open Source KB Builder"
echo "==============================================="

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Conda found, creating environment..."
    
    # Create a new conda environment for KB building
    conda create -n kb_builder python=3.10 -y
    
    # Activate the environment
    eval "$(conda shell.bash hook)"
    conda activate kb_builder
    
    # Install requirements
    pip install -r open_source_kb_requirements.txt
    
    echo "Environment setup complete!"
else
    echo "Conda not found, using pip in current environment"
    echo "WARNING: This might not solve NumPy version conflicts"
    echo "Installing requirements..."
    
    pip install -r open_source_kb_requirements.txt
fi

echo "==============================================="
echo "Running Open Source KB Builder"
echo "==============================================="

# Run the KB builder script
python build_open_source_kb.py

echo "==============================================="
echo "KB Builder finished"
echo "===============================================" 