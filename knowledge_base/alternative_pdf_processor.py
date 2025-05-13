import os
import tempfile
import re
import subprocess
import platform
import sys
from typing import List, Dict, Tuple, Optional
from pdf2image import convert_from_path
import pytesseract
from pytesseract import Output
import logging

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
        
    # Try to check for GPU availability without requiring torch
    try:
        import torch
        hardware_info["has_gpu"] = torch.cuda.is_available() if hasattr(torch, 'cuda') else False
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
    Alternative open-source pipeline to convert PDF to MMD-like markdown.
    Steps:
    1. Convert PDF pages to images.
    2. Extract plain text via Tesseract OCR.
    3. Detect and convert formula regions (naively).
    4. Merge text and LaTeX formulas into a markdown string.
    
    Hardware-aware: Optimizes for Apple Silicon or GPU systems.
    """
    # Detect hardware and get optimized parameters
    hardware_info = detect_hardware()
    params = optimize_for_hardware(hardware_info)
    
    # Initialize output string
    full_mmd = ""

    # Log hardware detection results
    logging.info(f"Processing PDF with hardware-aware settings: {params}")
    
    # Convert PDF pages to images (one per page)
    try:
        logging.info(f"Converting PDF to images with DPI={params['dpi']}")
        pages = convert_from_path(
            pdf_path, 
            dpi=params['dpi'],
            thread_count=params['threads']
        )
    except Exception as e:
        raise RuntimeError(f"Failed to convert PDF to images: {str(e)}")
    
    logging.info(f"Extracted {len(pages)} pages from PDF")
    
    # Process each page
    for i, page in enumerate(pages):
        logging.info(f"Processing page {i+1}/{len(pages)}")
        
        # Save page to temporary image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
            temp_img_path = temp_img.name
            page.save(temp_img_path, 'PNG')
        
        try:
            # Extract text using Tesseract OCR
            ocr_config = f"--oem 1 --psm 6"
            if hardware_info["is_apple_silicon"]:
                # Adjust OCR settings for Apple Silicon
                ocr_config += " -c tessedit_do_invert=0"
                
            text_data = pytesseract.image_to_data(
                temp_img_path, 
                config=ocr_config,
                output_type=Output.DICT
            )
            
            # Extract and process page text
            page_text = process_ocr_data(text_data, hardware_info)
            
            # Add to full document
            full_mmd += f"\n## Page {i+1}\n\n{page_text}\n\n"
            
        except Exception as e:
            logging.error(f"Error processing page {i+1}: {str(e)}")
            full_mmd += f"\n## Page {i+1}\n\n[OCR Processing Error: {str(e)}]\n\n"
        finally:
            # Clean up
            os.unlink(temp_img_path)
    
    # Post-process the full document
    full_mmd = post_process_document(full_mmd)
    
    return full_mmd

def process_ocr_data(text_data: Dict, hardware_info: Dict[str, bool]) -> str:
    """
    Process OCR data and detect potential mathematical formulas.
    """
    page_text = ""
    line_text = ""
    in_potential_formula = False
    formula_buffer = ""
    
    # Simple heuristics to detect possible math formulas
    math_indicators = [
        r"[=+\-*/^]", r"\([a-z0-9]+\)", r"\[[a-z0-9]+\]", 
        r"\\sum", r"\\int", r"\\frac", r"\$", r"\{"
    ]
    
    line_num = -1
    for i, text in enumerate(text_data['text']):
        if not text.strip():
            continue
            
        # Track line changes
        if text_data['line_num'][i] != line_num:
            if line_text:
                page_text += line_text + "\n"
                line_text = ""
            line_num = text_data['line_num'][i]
            
        # Check if this text block might be a formula
        is_formula = False
        for pattern in math_indicators:
            if re.search(pattern, text):
                is_formula = True
                break
        
        # Handle formula detection state
        if is_formula and not in_potential_formula:
            in_potential_formula = True
            formula_buffer = text
        elif is_formula and in_potential_formula:
            formula_buffer += " " + text
        elif not is_formula and in_potential_formula:
            # End of formula - convert using pix2tex if available, otherwise use as-is
            in_potential_formula = False
            formula = process_formula(formula_buffer, hardware_info)
            line_text += f" {formula} "
            formula_buffer = ""
            line_text += text + " "
        else:
            line_text += text + " "
    
    # Add any remaining text
    if line_text:
        page_text += line_text + "\n"
    
    return page_text

def process_formula(formula_text: str, hardware_info: Dict[str, bool]) -> str:
    """
    Process a detected formula area - attempt to convert to LaTeX.
    Basic implementation - could be enhanced with pix2tex or similar.
    """
    # Clean up the formula text
    formula_text = formula_text.strip()
    
    # For the naive implementation, we just wrap suspected formulas in LaTeX delimiters
    # In a real implementation, you would use pix2tex or a similar model here
    try:
        # Check if pix2tex is available and use it if possible
        latex_formula = convert_to_latex(formula_text, hardware_info)
        return f"${latex_formula}$"
    except (ImportError, Exception) as e:
        logging.warning(f"Could not process formula with pix2tex: {str(e)}")
        # Fallback: Just use naive formula detection
        return f"${formula_text}$" 

def convert_to_latex(formula_text: str, hardware_info: Dict[str, bool]) -> str:
    """
    Attempt to convert formula text to LaTeX using pix2tex if available.
    Simplified implementation - actual usage would need image extraction.
    """
    # This is a placeholder - actual implementation would use pix2tex
    # with an image of the formula region
    return formula_text

def post_process_document(mmd_text: str) -> str:
    """
    Clean up the generated MMD document.
    """
    # Remove consecutive blank lines
    mmd_text = re.sub(r'\n{3,}', '\n\n', mmd_text)
    
    # Fix common OCR errors in mathematical notation
    fixes = {
        r'\$([^$]*?)([0-9])\s+\+\s+([0-9])([^$]*?)\$': r'$\1\2+\3\4$',  # Fix spacing in additions
        r'\$([^$]*?)([0-9])\s+\-\s+([0-9])([^$]*?)\$': r'$\1\2-\3\4$',  # Fix spacing in subtractions
        r'\$([^$]*?)([0-9])\s+\*\s+([0-9])([^$]*?)\$': r'$\1\2*\3\4$',  # Fix spacing in multiplications
        r'\$([^$]*?)([0-9])\s+\/\s+([0-9])([^$]*?)\$': r'$\1\2/\3\4$',  # Fix spacing in divisions
    }
    
    for pattern, replacement in fixes.items():
        mmd_text = re.sub(pattern, replacement, mmd_text)
    
    return mmd_text

# For testing
if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        logging.basicConfig(level=logging.INFO)
        hardware_info = detect_hardware()
        print(f"Detected hardware: {hardware_info}")
        print(f"Optimized parameters: {optimize_for_hardware(hardware_info)}")
        result = process_pdf(pdf_path)
        print(f"Generated {len(result)} characters of markdown")
    else:
        print("Please provide a PDF path as argument") 