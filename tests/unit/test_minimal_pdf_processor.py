#!/usr/bin/env python3
"""
Minimal PDF processor test script without torch or complex NumPy dependencies.
"""

import os
import sys
import re
import logging
import platform
import tempfile
from pdf2image import convert_from_path
import pytesseract

logging.basicConfig(level=logging.INFO)


def detect_hardware_simple():
    """Simplified hardware detection without torch dependencies"""
    return {
        "is_apple_silicon": platform.system() == "Darwin" and platform.machine() == "arm64",
        "cpu_cores": os.cpu_count() or 1,
        "system": platform.system(),
        "machine": platform.machine(),
    }


def optimize_params(hardware_info):
    """Get optimized parameters for PDF processing"""
    params = {"dpi": 300, "threads": 1}  # Default DPI for PDF conversion  # Default thread count

    if hardware_info["is_apple_silicon"]:
        logging.info("Apple Silicon detected, optimizing for M-series")
        params["threads"] = min(hardware_info["cpu_cores"], 4)
        params["dpi"] = 250  # Lower DPI slightly on M-series for better performance
    else:
        logging.info("Standard CPU detected")
        params["threads"] = min(hardware_info["cpu_cores"], 2)

    return params


def process_pdf_simple(pdf_path, params):
    """Process a PDF file to extract text - simplified version"""
    full_text = f"# Processed PDF: {os.path.basename(pdf_path)}\n\n"

    try:
        logging.info(f"Converting PDF to images with DPI={params['dpi']}")
        # Convert PDF pages to images
        pages = convert_from_path(pdf_path, dpi=params["dpi"], thread_count=params["threads"])
        logging.info(f"Extracted {len(pages)} pages from PDF")

        # Process each page
        for i, page in enumerate(pages):
            logging.info(f"Processing page {i+1}/{len(pages)}")

            # Save page to temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
                temp_img_path = temp_img.name
                page.save(temp_img_path, "PNG")

            try:
                # OCR configuration
                ocr_config = "--oem 1 --psm 6"

                # Extract text using Tesseract OCR
                page_text = pytesseract.image_to_string(temp_img_path, config=ocr_config)

                # Add to output
                full_text += f"\n## Page {i+1}\n\n{page_text}\n\n"

            except Exception as e:
                logging.error(f"Error processing page {i+1}: {str(e)}")
                full_text += f"\n## Page {i+1}\n\n[OCR Processing Error: {str(e)}]\n\n"
            finally:
                # Clean up temp file
                if os.path.exists(temp_img_path):
                    os.unlink(temp_img_path)

            # Process only the first 3 pages for quick testing
            # if i >= 2:
            #     full_text += "\n## Remaining pages omitted for quick testing\n"
            #     break

        return full_text
    except Exception as e:
        logging.error(f"Error processing PDF: {str(e)}")
        return f"# Error processing PDF\n\nError: {str(e)}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_minimal_pdf_processor.py <pdf_file>")
        return 1

    pdf_path = sys.argv[1]
    output_dir = "output/test_pipeline"
    os.makedirs(output_dir, exist_ok=True)

    hardware_info = detect_hardware_simple()
    print(f"Hardware detected: {hardware_info}")

    params = optimize_params(hardware_info)
    print(f"Using parameters: {params}")

    try:
        markdown_content = process_pdf_simple(pdf_path, params)
        output_path = os.path.join(output_dir, os.path.basename(pdf_path) + ".md")

        with open(output_path, "w") as f:
            f.write(markdown_content)

        print(f"Successfully processed PDF and saved output to {output_path}")
        print(f"Generated {len(markdown_content)} characters of markdown")
        return 0
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
