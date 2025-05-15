#!/usr/bin/env python
"""
Special debug script focused on the text splitting phase.
This focuses just on the text chunking that seems to be causing problems.
"""

import os
import sys
import time
import platform
from pathlib import Path
import argparse

# Make sure Python's output is unbuffered
os.environ['PYTHONUNBUFFERED'] = '1'

def debug_split_into_chunks(text, chunk_size=1000, overlap=200):
    """
    Debug version of the split_into_chunks function with more logging.
    """
    # Start timing
    start_time = time.time()
    
    print(f"\nDEBUG: Splitting text of length {len(text)} into chunks (size={chunk_size}, overlap={overlap})")
    
    # If text is shorter than chunk_size, return it as a single chunk
    if len(text) <= chunk_size:
        print(f"Text is shorter than chunk_size ({len(text)} <= {chunk_size}), returning as single chunk")
        return [text]
    
    chunks = []
    start = 0
    
    # Debug timers
    search_time = 0
    extraction_time = 0
    
    print("Starting chunk extraction...")
    
    # Try to split on paragraph boundaries when possible
    while start < len(text):
        # Calculate the potential end of this chunk
        end = min(start + chunk_size, len(text))
        print(f"Processing chunk from positions {start} to {end} (length: {end-start})")
        
        # If we're not at the end of the text, try to find a paragraph break
        if end < len(text):
            search_start_time = time.time()
            
            # Look for paragraph breaks (double newline) within the last 1/4 of the chunk
            search_start = max(start + (chunk_size * 3) // 4, start)
            search_text = text[search_start:end]
            
            print(f"Searching for paragraph break in positions {search_start}-{end} (length: {len(search_text)})")
            
            # Find the last paragraph break in this range
            para_break = search_text.rfind('\n\n')
            
            if para_break != -1:
                # Adjust end to break at the paragraph
                end = search_start + para_break + 2  # +2 to include the newlines
                print(f"Found paragraph break, adjusted end to {end}")
            else:
                # If no paragraph break, try to break at a sentence
                import re
                print("No paragraph break found, looking for sentence break...")
                sentence_end = re.search(r'[.!?]\s+', search_text[::-1])
                if sentence_end is not None:
                    # Calculate position from the end
                    sentence_pos = len(search_text) - sentence_end.start()
                    end = search_start + sentence_pos
                    print(f"Found sentence break, adjusted end to {end}")
                else:
                    print("No sentence break found, using original end position")
            
            search_time += time.time() - search_start_time
        
        # Extract chunk
        extraction_start_time = time.time()
        chunk = text[start:end]
        chunks.append(chunk)
        print(f"Extracted chunk {len(chunks)} (length: {len(chunk)})")
        extraction_time += time.time() - extraction_start_time
        
        # Calculate the start of the next chunk with overlap
        start = max(start, end - overlap)
        print(f"Next chunk will start at position {start}")
        
        # Ensure we're making progress
        if start >= end:
            print("WARNING: No progress being made (start >= end), forcing progress")
            start = end
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nChunking completed in {total_time:.2f} seconds")
    print(f"- Search time: {search_time:.2f} seconds ({search_time/total_time*100:.1f}%)")
    print(f"- Extraction time: {extraction_time:.2f} seconds ({extraction_time/total_time*100:.1f}%)")
    print(f"- Created {len(chunks)} chunks")
    
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Debug knowledge base chunking with direct output")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    args = parser.parse_args()
    
    # Import and run KB builder module directly
    start_time = time.time()
    
    print("\n")
    print("=" * 80)
    print("DEBUGGING TEXT CHUNKING PROCESS")
    print("=" * 80)
    print(f"Running on: {platform.system()} {platform.release()} ({platform.machine()})")
    print("This will focus on the chunking process that seems to be getting stuck")
    print("=" * 80)
    print("\n")
    
    # Import and run KB builder with our modifications
    try:
        # Add project root to path
        sys.path.insert(0, os.path.abspath('.'))
        
        # Import config loader
        from utils.config_loader import load_config, DEFAULT_CONFIG_PATH
        config_path = args.config if args.config else DEFAULT_CONFIG_PATH
        cfg = load_config(config_path)
        
        # Import internal modules
        print("\nLoading modules...")
        from knowledge_base.alternative_pdf_processor import process_pdf
        import logging
        
        # Implement our own step-by-step process to find where it's stuck
        print("\nStep 1: Scanning for PDFs...")
        pdf_files = []
        for root, dirs, files in os.walk(cfg.paths.pdf_input_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        print(f"Found {len(pdf_files)} PDF files")
        
        if not pdf_files:
            print("No PDF files found. Exiting.")
            return 1
        
        # Process single PDF first as test
        print("\nStep 2: Processing first PDF to extract text...")
        first_pdf = pdf_files[0]
        print(f"Processing {os.path.basename(first_pdf)}")
        
        try:
            # Process the PDF
            start_pdf_time = time.time()
            mmd_content = process_pdf(first_pdf, cfg)
            end_pdf_time = time.time()
            
            if not mmd_content:
                print("No content extracted. Exiting.")
                return 1
                
            print(f"Successfully extracted {len(mmd_content)} characters of text in {end_pdf_time-start_pdf_time:.2f} seconds")
            
            # Try a very simple split first to test basic functionality
            print("\nStep 3: Testing simple text splitting (this should be fast)...")
            start_simple_split = time.time()
            simple_chunks = [mmd_content[i:i+1000] for i in range(0, len(mmd_content), 800)]
            end_simple_split = time.time()
            print(f"Simple splitting completed in {end_simple_split-start_simple_split:.2f} seconds, created {len(simple_chunks)} chunks")
            
            # Now try the actual splitting function
            print("\nStep 4: Testing the actual chunking function...")
            
            # Get parameters
            chunk_size = cfg.embeddings.chunk_size
            overlap = cfg.embeddings.chunk_overlap
            
            # First with a tiny sample
            print("First testing with just a small section of text...")
            sample_text = mmd_content[:5000]  # Just take first 5000 chars
            start_sample_time = time.time()
            sample_chunks = debug_split_into_chunks(sample_text, chunk_size, overlap)
            end_sample_time = time.time()
            print(f"Sample chunking completed in {end_sample_time-start_sample_time:.2f} seconds, created {len(sample_chunks)} chunks")
            
            # Then with the full text
            print("\nNow testing with the full text (this is where it usually hangs)...")
            
            # Start a watchdog timer thread to detect hangs
            import threading
            hang_detected = False
            
            def watchdog():
                nonlocal hang_detected
                time_limit = 30  # seconds
                start = time.time()
                while time.time() - start < time_limit:
                    time.sleep(1)
                # If we get here, the time limit was reached
                hang_detected = True
                print("\nWARNING: Watchdog timer expired! The chunking process appears to be hanging.")
                print("This confirms that the chunking function is the source of the issue.")
                
            watchdog_thread = threading.Thread(target=watchdog, daemon=True)
            watchdog_thread.start()
            
            try:
                start_full_time = time.time()
                full_chunks = debug_split_into_chunks(mmd_content, chunk_size, overlap)
                end_full_time = time.time()
                print(f"Full chunking completed in {end_full_time-start_full_time:.2f} seconds, created {len(full_chunks)} chunks")
            except Exception as e:
                import traceback
                print(f"Error during chunking: {e}")
                traceback.print_exc()
                
            if hang_detected:
                print("\nThe chunking function appears to be hanging. We need to optimize it.")
                
                # Implement a simpler chunking function as a workaround
                print("\nTrying a simpler chunking function...")
                
                start_workaround_time = time.time()
                
                # Simple chunking with minimal overlap handling
                workaround_chunks = []
                for i in range(0, len(mmd_content), chunk_size - overlap):
                    chunk = mmd_content[i:i+chunk_size]
                    if chunk:  # Only add non-empty chunks
                        workaround_chunks.append(chunk)
                        
                end_workaround_time = time.time()
                print(f"Workaround chunking completed in {end_workaround_time-start_workaround_time:.2f} seconds, created {len(workaround_chunks)} chunks")
            
            # Display result
            end_time = time.time()
            duration = end_time - start_time
            
            print("\n")
            print("=" * 80)
            print(f"CHUNKING DIAGNOSTIC COMPLETED IN {duration:.2f} SECONDS")
            print("=" * 80)
            
            return 0
            
        except Exception as e:
            import traceback
            print(f"Error processing PDF: {e}")
            traceback.print_exc()
            return 1
            
    except Exception as e:
        import traceback
        print(f"Error in debug process: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 