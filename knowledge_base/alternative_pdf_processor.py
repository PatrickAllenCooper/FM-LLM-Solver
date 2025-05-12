import os
import tempfile
import re
import subprocess
from typing import List
from pdf2image import convert_from_path
import pytesseract
from pytesseract import Output

def process_pdf(pdf_path: str) -> str:
    """
    Alternative open-source pipeline to convert PDF to MMD-like markdown.
    Steps:
    1. Convert PDF pages to images.
    2. Extract plain text via Tesseract OCR.
    3. Naively detect formula regions and convert with pix2tex.
    4. Merge text and LaTeX formulas into a markdown string.
    """
    # Initialize output string
    full_mmd = ""

    # Convert PDF pages to images (one per page)
    try:
        pages = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        raise RuntimeError(f"Failed to convert PDF to images: {e}")

    for page_num, page_image in enumerate(pages, start=1):
        # Page marker
        full_mmd += f"<!-- Page {page_num} -->\n\n"

        # Extract plain text from page image
        try:
            page_text = pytesseract.image_to_string(page_image)
        except Exception:
            page_text = ""
        if page_text:
            full_mmd += page_text + "\n\n"

        # Perform OCR layout analysis to detect candidate formula regions
        try:
            data = pytesseract.image_to_data(page_image, output_type=Output.DICT)
        except Exception:
            data = {'text': []}

        for i, word in enumerate(data.get('text', [])):
            if not word or not word.strip():
                continue
            # Naive heuristic: treat words containing non-alphanumeric chars as formulas
            if re.search(r"[^A-Za-z0-9]", word):
                x, y, w, h = (
                    data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                )
                formula_image = page_image.crop((x, y, x + w, y + h))
                # Save region to temporary file
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                    formula_image.save(tmp_path)
                try:
                    # Call pix2tex CLI to get LaTeX code
                    result = subprocess.run(
                        ["pix2tex", "--model", "im2latex", tmp_path],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    latex = result.stdout.strip()
                    if latex:
                        full_mmd += f"${latex}$\n\n"
                except Exception:
                    # Ignore failures
                    pass
                finally:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

    return full_mmd 