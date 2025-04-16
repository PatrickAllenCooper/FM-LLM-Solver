import os
import fitz  # PyMuPDF
import pytesseract # For OCR
from PIL import Image # For handling images for OCR
import io # For image bytes handling
import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import spacy # For sentence splitting
import logging
import sys

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directories
BASE_DIR = os.path.dirname(__file__) # Directory of the script
PDF_INPUT_DIR = os.path.join(BASE_DIR, "recent_papers_all_sources_v2") # Output from paper_fetcher.py
OUTPUT_DIR = os.path.join(BASE_DIR, "knowledge_base_enhanced") # New output dir

# Embedding Model
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2' # Good general-purpose starting model

# Vector Store
VECTOR_STORE_FILENAME = "paper_index_enhanced.faiss"
METADATA_FILENAME = "paper_metadata_enhanced.json"

# Chunking Parameters (now based on sentences)
# Target number of sentences per chunk (approximate) - adjust as needed
CHUNK_TARGET_SENTENCE_COUNT = 10
# Number of overlapping sentences between chunks
CHUNK_OVERLAP_SENTENCE_COUNT = 2

# --- SpaCy Model Loading ---
# Attempt to load the spaCy model. Provide user guidance if it fails.
SPACY_MODEL_NAME = "en_core_web_sm"
nlp = None
try:
    nlp = spacy.load(SPACY_MODEL_NAME)
    # Increase max length if needed for very long documents/paragraphs
    # nlp.max_length = 2000000 # Example: 2 million characters
    logging.info(f"SpaCy model '{SPACY_MODEL_NAME}' loaded successfully.")
except OSError:
    logging.error(f"SpaCy model '{SPACY_MODEL_NAME}' not found.")
    logging.error("Please download it by running:")
    logging.error(f"python -m spacy download {SPACY_MODEL_NAME}")
    sys.exit(1) # Exit if essential NLP model is missing
except Exception as e:
    logging.error(f"An error occurred loading the SpaCy model: {e}")
    sys.exit(1)

# --- Tesseract Configuration ---
# Optional: Specify Tesseract command path if not in system PATH
# pytesseract.pytesseract.tesseract_cmd = r'/path/to/tesseract'

# --- Helper Functions ---

def clean_text(text):
    """Basic text cleaning."""
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove hyphenation across lines (more careful approach)
    text = re.sub(r'-\s*\n\s*', '', text)
    # Remove single newlines that likely break sentences incorrectly
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Replace multiple newlines with a single one (paragraph breaks)
    text = re.sub(r'\n\n+', '\n', text)
    return text

def extract_title_heuristic(doc):
    """Attempts to extract the title from the first page based on font size."""
    title = "Unknown Title"
    max_font_size = 0
    try:
        if doc.page_count > 0:
            first_page = doc.load_page(0)
            # Get text blocks with font information
            blocks = first_page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]
            for b in blocks:
                if "lines" in b:
                    for l in b["lines"]:
                        if "spans" in l:
                            for s in l["spans"]:
                                font_size = s.get("size", 0)
                                if font_size > max_font_size:
                                    max_font_size = font_size
                                    # Simple cleaning for potential title text
                                    potential_title = s.get("text", "").strip()
                                    if len(potential_title) > 5: # Basic check
                                        title = potential_title
    except Exception as e:
        logging.warning(f"Could not extract title heuristically: {e}")
    # Further clean the extracted title if needed
    title = clean_text(title.replace('\n', ' '))
    logging.info(f"Heuristically extracted title: '{title}' (Max font size: {max_font_size:.2f})")
    return title

def extract_content_from_pdf(pdf_path):
    """Extracts text and OCR content from PDF pages."""
    pages_content = {} # Store content per page number {page_num: {'text': str, 'ocr': str}}
    potential_title = "Unknown Title"
    try:
        doc = fitz.open(pdf_path)
        logging.info(f"Processing PDF: {os.path.basename(pdf_path)} with {doc.page_count} pages.")
        potential_title = extract_title_heuristic(doc)

        for page_num in range(doc.page_count):
            page_content = {'text': "", 'ocr': ""}
            page = doc.load_page(page_num)

            # 1. Extract text using PyMuPDF
            try:
                text = page.get_text("text")
                if text:
                    page_content['text'] = clean_text(text)
            except Exception as e:
                logging.warning(f"PyMuPDF text extraction failed on page {page_num+1} of {os.path.basename(pdf_path)}: {e}")

            # 2. Perform OCR using Tesseract
            try:
                # Render page to an image for OCR
                pix = page.get_pixmap(dpi=300) # Higher DPI often improves OCR
                img_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_bytes))
                ocr_text = pytesseract.image_to_string(img)
                if ocr_text:
                    page_content['ocr'] = clean_text(ocr_text)
            except Exception as e:
                logging.warning(f"OCR failed on page {page_num+1} of {os.path.basename(pdf_path)}: {e}")
                # Check if Tesseract is installed and configured correctly
                if "TesseractNotFoundError" in str(e):
                     logging.error("Tesseract executable not found. Please ensure Tesseract is installed and in your PATH, or configure pytesseract.pytesseract.tesseract_cmd.")

            # Store combined or individual content as needed
            pages_content[page_num + 1] = page_content # Use 1-based indexing for pages

        doc.close()
    except Exception as e:
        logging.error(f"Failed to process {pdf_path}: {e}")
    return potential_title, pages_content

def chunk_document_sentences(potential_title, pages_content, target_sentences, overlap_sentences, source_pdf):
    """Chunks the document based on sentences using SpaCy."""
    chunks = []
    all_sentences = []

    logging.info(f"Splitting content into sentences for {os.path.basename(source_pdf)}...")
    # Combine text and OCR results (simple concatenation here, could be smarter)
    full_text = ""
    page_map = [] # Keep track of which sentence belongs to which page(s)
    for page_num in sorted(pages_content.keys()):
        page_data = pages_content[page_num]
        combined_page_text = page_data['text'] + "\n\n" + page_data['ocr'] # Add separator
        cleaned_page_text = clean_text(combined_page_text)

        if cleaned_page_text:
            try:
                # Process potentially large text page by page
                doc = nlp(cleaned_page_text)
                page_sentences = list(doc.sents)
                all_sentences.extend(page_sentences)
                page_map.extend([page_num] * len(page_sentences)) # Map sentences to this page
                # Add page break marker? Maybe not necessary if chunking handles it.
            except Exception as e:
                logging.warning(f"SpaCy processing failed for page {page_num} of {os.path.basename(source_pdf)}: {e}")

    if not all_sentences:
        logging.warning(f"No sentences extracted after processing {os.path.basename(source_pdf)}")
        return chunks

    logging.info(f"Total sentences extracted: {len(all_sentences)}")

    # Chunking logic
    current_chunk_sentences = []
    current_chunk_pages = set()
    start_sentence_index = 0

    for i, sentence in enumerate(all_sentences):
        current_chunk_sentences.append(sentence.text.strip())
        current_chunk_pages.add(page_map[i]) # Add page number for this sentence

        # Check if chunk is full
        if len(current_chunk_sentences) >= target_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            metadata = {
                'source': os.path.basename(source_pdf),
                'potential_title': potential_title,
                'pages': sorted(list(current_chunk_pages)),
                'start_sentence_index': start_sentence_index,
                'end_sentence_index': i
            }
            chunks.append({'text': chunk_text, 'metadata': metadata})

            # Prepare for the next chunk with overlap
            start_sentence_index = max(0, i - overlap_sentences + 1)
            # Ensure start_sentence_index is valid
            start_sentence_index = min(start_sentence_index, i)

            # Sentences for overlap. Ensure we don't go out of bounds.
            overlap_slice = all_sentences[start_sentence_index : i + 1]
            current_chunk_sentences = [s.text.strip() for s in overlap_slice[-(overlap_sentences+1):] ] # Grab last 'overlap' sentences for next chunk start, prevent IndexError

            # Reset pages based on the actual start index for the overlap
            current_chunk_pages = set(page_map[start_sentence_index : i + 1])

    # Add the last remaining chunk if any sentences are left
    if current_chunk_sentences and start_sentence_index < len(all_sentences):
         chunk_text = " ".join(current_chunk_sentences)
         metadata = {
            'source': os.path.basename(source_pdf),
            'potential_title': potential_title,
            'pages': sorted(list(current_chunk_pages)),
            'start_sentence_index': start_sentence_index,
            'end_sentence_index': len(all_sentences) - 1
         }
         chunks.append({'text': chunk_text, 'metadata': metadata})

    logging.info(f"Created {len(chunks)} sentence-based chunks for {os.path.basename(source_pdf)}")

    # Add placeholder comment about where more advanced extraction could integrate
    # TODO: Integrate Math/Equation extraction here (e.g., using MathPix API)
    # TODO: Integrate Structure Recognition here (e.g., using GROBID)

    return chunks

# --- Main Logic ---

def build_knowledge_base():
    """Main function to build the vector store and metadata."""
    if nlp is None:
         logging.error("SpaCy model not loaded. Cannot proceed.")
         return # Exit if SpaCy failed to load

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_chunks_with_metadata = []
    processed_files = 0

    # 1. Iterate through PDFs, Extract (Text+OCR), Chunk by Sentences
    logging.info(f"Starting PDF processing from: {PDF_INPUT_DIR}")
    all_pdf_paths = []
    for root, _, files in os.walk(PDF_INPUT_DIR):
        for file in files:
            if file.lower().endswith(".pdf"):
                all_pdf_paths.append(os.path.join(root, file))

    logging.info(f"Found {len(all_pdf_paths)} PDF files.")

    for pdf_path in all_pdf_paths:
        logging.info(f"--- Processing: {os.path.basename(pdf_path)} ---")
        potential_title, pages_content = extract_content_from_pdf(pdf_path)
        if not pages_content:
            logging.warning(f"No content extracted from {os.path.basename(pdf_path)}")
            continue

        doc_chunks = chunk_document_sentences(
            potential_title,
            pages_content,
            CHUNK_TARGET_SENTENCE_COUNT,
            CHUNK_OVERLAP_SENTENCE_COUNT,
            pdf_path
        )
        all_chunks_with_metadata.extend(doc_chunks)
        processed_files += 1

    if not all_chunks_with_metadata:
        logging.error("No text chunks were generated from any PDF. Exiting.")
        return

    logging.info(f"Total chunks created: {len(all_chunks_with_metadata)} from {processed_files} PDFs.")

    # 2. Embed Chunks
    try:
        logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        # Consider using a device='cuda' if GPU is available and configured
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        logging.error(f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}")
        return

    logging.info("Embedding chunks...")
    chunk_texts = [chunk['text'] for chunk in all_chunks_with_metadata]
    try:
        # Adjust batch_size based on available RAM/VRAM
        embeddings = model.encode(chunk_texts, show_progress_bar=True, batch_size=32)
        embeddings = np.array(embeddings).astype('float32') # FAISS requires float32
        logging.info(f"Embeddings generated with shape: {embeddings.shape}")
    except Exception as e:
        logging.error(f"Failed during embedding generation: {e}")
        return

    # 3. Build and Save Vector Store (FAISS Example)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) # Simple L2 index
    # Consider IndexIVFFlat for large datasets (see previous version comments)

    try:
        index.add(embeddings)
        logging.info(f"Added {index.ntotal} vectors to FAISS index.")

        index_path = os.path.join(OUTPUT_DIR, VECTOR_STORE_FILENAME)
        faiss.write_index(index, index_path)
        logging.info(f"Saved FAISS index to {index_path}")
    except Exception as e:
        logging.error(f"Failed to build or save FAISS index: {e}")
        return

    # 4. Save Metadata
    metadata_map = {
        i: {**chunk['metadata'], 'text': chunk['text']}
        for i, chunk in enumerate(all_chunks_with_metadata)
    }

    metadata_path = os.path.join(OUTPUT_DIR, METADATA_FILENAME)
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            # Use indent=None for smallest file size, or indent=2 for readability
            json.dump(metadata_map, f, indent=None)
        logging.info(f"Saved metadata mapping to {metadata_path}")
    except Exception as e:
        logging.error(f"Failed to save metadata JSON: {e}")

if __name__ == "__main__":
    # Basic dependency check (already done for SpaCy at import time)
    try:
        import fitz
        import sentence_transformers
        import faiss
        import numpy
        import PIL
        import pytesseract
    except ImportError as e:
        # SpaCy import errors handled earlier
        if "spacy" not in str(e).lower():
            print(f"Error: Missing dependency - {e}", file=sys.stderr)
            print("Please install required packages:", file=sys.stderr)
            print("pip install -r requirements.txt", file=sys.stderr)
            # Also remind about Tesseract system dependency
            print("Ensure Tesseract OCR engine is installed.", file=sys.stderr)
            sys.exit(1)

    build_knowledge_base()
    logging.info("Enhanced knowledge base build process finished.") 