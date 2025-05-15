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

def optimize_for_hardware(hardware_info: Dict[str, bool], cfg=None) -> Dict[str, any]:
    """
    Return optimized parameters based on hardware detection and configuration.
    
    Parameters
    ----------
    hardware_info : Dict[str, bool]
        Hardware detection results
    cfg : omegaconf.dictconfig.DictConfig, optional
        Configuration object with memory optimization settings
    """
    params = {
        "dpi": 300,  # Default DPI for PDF conversion
        "batch_size": 1,  # Default batch size for processing
        "threads": 1,  # Default thread count
        "use_mps": False,  # Whether to use Metal Performance Shaders
        "low_memory_mode": False  # Conservative memory usage
    }
    
    # Check if config is provided
    if cfg is not None:
        # Get low memory mode from config
        if hasattr(cfg, 'knowledge_base') and hasattr(cfg.knowledge_base, 'low_memory_mode'):
            params["low_memory_mode"] = cfg.knowledge_base.low_memory_mode
            
        # Check if we have GPU memory limit from config
        if hasattr(cfg, 'knowledge_base') and hasattr(cfg.knowledge_base, 'gpu_memory_limit'):
            params["gpu_memory_limit"] = cfg.knowledge_base.gpu_memory_limit
    
    # If low memory mode, adjust parameters regardless of hardware
    if params.get("low_memory_mode", False):
        logging.info("Low memory mode enabled, using conservative settings")
        params["dpi"] = 150  # Lower DPI for faster processing and less memory
        params["batch_size"] = 1  # Process one page at a time
        params["threads"] = 1  # Single thread to limit memory
        params["use_mps"] = False  # Disable GPU acceleration
        return params
    
    # Normal hardware-based optimization
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

def process_pdf(pdf_path: str, cfg=None) -> str:
    """
    Alternative open-source pipeline to extract text from PDF without external dependencies.
    Uses PyMuPDF (fitz) to extract text directly from the PDF.
    
    Hardware-aware: Optimizes for Apple Silicon or GPU systems.
    Memory-aware: Respects low_memory_mode setting from config.
    
    Parameters
    ----------
    pdf_path : str
        Path to the PDF file
    cfg : omegaconf.dictconfig.DictConfig, optional
        Configuration object with memory optimization settings
    """
    # Detect hardware and get optimized parameters
    hardware_info = detect_hardware()
    params = optimize_for_hardware(hardware_info, cfg)
    
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
    
    # Get memory mode
    low_memory_mode = params.get("low_memory_mode", False)
    
    # In low memory mode, explicitly free memory during processing
    if low_memory_mode:
        # Process each page and clear memory after each one
        for i in range(len(doc)):
            logging.info(f"Processing page {i+1}/{len(doc)} (low memory mode)")
            
            try:
                # Get the page
                page = doc[i]
                
                # Extract text from the page
                text = page.get_text()
                
                # Extract images if really needed and not too many
                if len(text.strip()) < 100 and i < 20:  # Only try for the first 20 pages to save memory
                    logging.info(f"Page {i+1} has limited text, attempting image-based extraction")
                    page_text = extract_text_from_page_images(page, params)
                    if page_text.strip():
                        text = page_text
                
                # Add to full document
                full_mmd += f"\n## Page {i+1}\n\n{text}\n\n"
                
                # Clear page (garbage collection)
                page = None
                
            except Exception as e:
                logging.error(f"Error processing page {i+1}: {str(e)}")
                full_mmd += f"\n## Page {i+1}\n\n[PDF Processing Error: {str(e)}]\n\n"
            
            # Force Python garbage collection
            try:
                import gc
                gc.collect()
            except:
                pass
    else:
        # Process each page normally (more efficient but uses more memory)
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
    
    # Force Python garbage collection
    try:
        import gc
        gc.collect()
    except:
        pass
    
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

def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for embedding.
    
    Parameters
    ----------
    text : str
        The input text to split
    chunk_size : int, optional
        Target size (in characters) for each chunk, by default 1000
    overlap : int, optional
        Number of characters to overlap between chunks, by default 200
        
    Returns
    -------
    List[str]
        List of text chunks
    """
    # If text is shorter than chunk_size, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    # Try to split on paragraph boundaries when possible
    while start < len(text):
        # Calculate the potential end of this chunk
        end = min(start + chunk_size, len(text))
        
        # If we're not at the end of the text, try to find a paragraph break
        if end < len(text):
            # Look for paragraph breaks (double newline) within the last 1/4 of the chunk
            search_start = max(start + (chunk_size * 3) // 4, start)
            search_text = text[search_start:end]
            
            # Find the last paragraph break in this range
            para_break = search_text.rfind('\n\n')
            
            if para_break != -1:
                # Adjust end to break at the paragraph
                end = search_start + para_break + 2  # +2 to include the newlines
            else:
                # If no paragraph break, try to break at a sentence
                sentence_end = re.search(r'[.!?]\s+', search_text[::-1])
                if sentence_end is not None:
                    # Calculate position from the end
                    sentence_pos = len(search_text) - sentence_end.start()
                    end = search_start + sentence_pos
        
        # Add this chunk to our list
        # Ensure the chunk is not empty, which shouldn't happen if start < end.
        current_chunk_text = text[start:end]
        if not current_chunk_text and start < len(text):
            # This is unexpected if start < end. As a safeguard, advance start by 1 to prevent potential hang.
            start += 1
            continue
        if current_chunk_text: # Only add non-empty chunks
            chunks.append(current_chunk_text)
        
        # Calculate the start of the next chunk
        if end >= len(text): # If the current chunk reached the end of the text
            start = len(text) # Move start to the end to terminate the loop
        else:
            # Normal case: current chunk did not reach end of text
            prev_start = start
            # The desired start of the next overlapping chunk
            potential_next_start = end - overlap
            
            # If the desired next start doesn't make progress (i.e., it's <= prev_start),
            # it means the chunk just processed was too short relative to the overlap
            # (specifically, end - prev_start <= overlap).
            # Force progress to prevent an infinite loop.
            if potential_next_start <= prev_start:
                start = prev_start + 1 
            else:
                start = potential_next_start
    
    return chunks

# For testing
if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        logging.basicConfig(level=logging.INFO)
        result = process_pdf(pdf_path)
        print(result)
    else:
        print("Usage: python alternative_pdf_processor.py <path_to_pdf>") 