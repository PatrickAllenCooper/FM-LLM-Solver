import os
import tempfile
import re
import subprocess
import platform
import sys
from typing import List, Dict, Tuple, Optional
import fitz  # PyMuPDF for PDF processing
import numpy as np
from PIL import Image
import logging
import io

# Hardware detection
def detect_hardware() -> Dict[str, bool]:
    """
    Detect hardware capabilities and return a dictionary of hardware features.
    """
    hardware_info = {
        "is_apple_silicon": False,
        "has_gpu": False,
        "cpu_cores": os.cpu_count() or 1
    }
    
    # Check for Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        hardware_info["is_apple_silicon"] = True
        logging.info("Apple Silicon detected, will optimize processing accordingly")
        # Don't try to use torch on Apple Silicon, as it may have compatibility issues
        return hardware_info
        
    # Only try to check for CUDA on non-Apple systems
    try:
        import torch
        hardware_info["has_gpu"] = torch.cuda.is_available() if hasattr(torch, 'cuda') else False
        if hardware_info["has_gpu"]:
            logging.info(f"CUDA-capable GPU detected")
    except (ImportError, AttributeError):
        # If torch isn't available or doesn't have CUDA, assume no GPU
        pass
        
    return hardware_info

def optimize_for_hardware(hardware_info: Dict[str, bool]) -> Dict[str, any]:
    """
    Return optimized parameters based on hardware detection.
    """
    params = {
        "dpi": 300,  # Default DPI for PDF conversion
        "batch_size": 1,  # Default batch size for processing
        "threads": 1,  # Default thread count
        "use_mps": False  # Whether to use Metal Performance Shaders
    }
    
    if hardware_info["is_apple_silicon"]:
        logging.info("Apple Silicon detected, optimizing for M-series")
        params["threads"] = min(hardware_info["cpu_cores"], 4)  # Use up to 4 cores on M-series
        params["use_mps"] = True
        # Lower DPI slightly to improve performance on M-series
        params["dpi"] = 250
    elif hardware_info["has_gpu"]:
        logging.info("GPU detected, optimizing for CUDA")
        params["batch_size"] = 4  # Increase batch size for GPU
        params["threads"] = min(hardware_info["cpu_cores"], 8)
    else:
        logging.info("Standard CPU detected")
        params["threads"] = min(hardware_info["cpu_cores"], 2)
    
    return params

def process_pdf(pdf_path: str) -> str:
    """
    Alternative open-source pipeline to extract text from PDF without external dependencies.
    Uses PyMuPDF (fitz) to extract text directly from the PDF.
    
    Hardware-aware: Optimizes for Apple Silicon or GPU systems.
    """
    # Detect hardware and get optimized parameters
    hardware_info = detect_hardware()
    params = optimize_for_hardware(hardware_info)
    
    # Initialize output string
    full_mmd = ""

    # Log hardware detection results
    logging.info(f"Processing PDF with hardware-aware settings: {params}")
    
    # Open the PDF file with PyMuPDF
    try:
        doc = fitz.open(pdf_path)
        logging.info(f"Opened PDF with {len(doc)} pages")
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {str(e)}")
    
    # Process each page
    for i, page in enumerate(doc):
        logging.info(f"Processing page {i+1}/{len(doc)}")
        
        try:
            # Extract text from the page
            text = page.get_text()
            
            # Extract images if needed for better processing
            if len(text.strip()) < 100:  # If very little text was extracted, try image extraction
                logging.info(f"Page {i+1} has limited text, attempting image-based extraction")
                page_text = extract_text_from_page_images(page, params)
                if page_text.strip():
                    text = page_text
            
            # Add to full document
            full_mmd += f"\n## Page {i+1}\n\n{text}\n\n"
            
        except Exception as e:
            logging.error(f"Error processing page {i+1}: {str(e)}")
            full_mmd += f"\n## Page {i+1}\n\n[PDF Processing Error: {str(e)}]\n\n"
    
    # Close the document
    doc.close()
    
    # Post-process the full document
    full_mmd = post_process_document(full_mmd)
    
    return full_mmd

def extract_text_from_page_images(page, params):
    """
    Extract text from images on a page when direct text extraction fails.
    This function would ideally use OCR, but can be simplified for dependency-free operation.
    """
    # If OCR is available, we'd use it here
    # For now, we'll just return a message to indicate image content
    images = page.get_images(full=True)
    if images:
        return f"[This page contains {len(images)} images. Text extraction limited without OCR.]"
    return ""

def post_process_document(mmd_text: str) -> str:
    """
    Clean up the generated MMD document.
    """
    # Remove consecutive blank lines
    mmd_text = re.sub(r'\n{3,}', '\n\n', mmd_text)
    
    # Fix common issues in mathematical notation
    fixes = {
        r'([0-9])\s+\+\s+([0-9])': r'\1+\2',  # Fix spacing in additions
        r'([0-9])\s+\-\s+([0-9])': r'\1-\2',  # Fix spacing in subtractions
        r'([0-9])\s+\*\s+([0-9])': r'\1*\2',  # Fix spacing in multiplications
        r'([0-9])\s+\/\s+([0-9])': r'\1/\2',  # Fix spacing in divisions
    }
    
    for pattern, replacement in fixes.items():
        mmd_text = re.sub(pattern, replacement, mmd_text)
    
    return mmd_text

# For testing
if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        logging.basicConfig(level=logging.INFO)
        result = process_pdf(pdf_path)
        print(result)
    else:
        print("Usage: python alternative_pdf_processor.py <path_to_pdf>") 