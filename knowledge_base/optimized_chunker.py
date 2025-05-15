"""
Optimized text chunking functions for the knowledge base.
These are designed to be more efficient and avoid hanging on regex operations.
"""

import time
import logging

def optimized_split_into_chunks(text, chunk_size=1000, overlap=200):
    """
    Split text into chunks with overlap, optimized for performance.
    This version avoids expensive regex operations that can hang.
    
    Parameters
    ----------
    text : str
        The text to split into chunks
    chunk_size : int
        The target size of each chunk
    overlap : int
        The number of characters to overlap between chunks
    
    Returns
    -------
    list
        A list of text chunks
    """
    # Start timing
    start_time = time.time()
    
    # If text is shorter than chunk_size, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    
    # Simple chunking with overlap - much more reliable
    step_size = chunk_size - overlap
    
    # Add safety checks
    if step_size <= 0:
        logging.warning(f"Invalid step size ({step_size}). Using chunk_size/2.")
        step_size = chunk_size // 2
    
    # Create chunks
    for i in range(0, len(text), step_size):
        chunk = text[i:i+chunk_size]
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
    
    # Log performance
    end_time = time.time()
    processing_time = end_time - start_time
    logging.info(f"Optimized chunking completed in {processing_time:.2f} seconds. Created {len(chunks)} chunks.")
    
    return chunks

def paragraph_aware_split(text, chunk_size=1000, overlap=200, max_search_time=2.0):
    """
    Split text into chunks with overlap, attempting to break at paragraph boundaries
    but with a time limit on the search to prevent hanging.
    
    Parameters
    ----------
    text : str
        The text to split into chunks
    chunk_size : int
        The target size of each chunk
    overlap : int
        The number of characters to overlap between chunks
    max_search_time : float
        Maximum number of seconds to spend searching for paragraph/sentence breaks
        
    Returns
    -------
    list
        A list of text chunks
    """
    # Start timing
    start_time = time.time()
    
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
            search_start_time = time.time()
            
            # Look for paragraph breaks (double newline) within the last 1/4 of the chunk
            search_start = max(start + (chunk_size * 3) // 4, start)
            search_text = text[search_start:end]
            
            # Find the last paragraph break in this range
            para_break = search_text.rfind('\n\n')
            
            # Time check - if we've spent too long searching, skip optimization
            if time.time() - search_start_time > max_search_time:
                logging.warning(f"Paragraph break search taking too long, using simple boundary")
                # Just use the original end
            elif para_break != -1:
                # Adjust end to break at the paragraph
                end = search_start + para_break + 2  # +2 to include the newlines
            else:
                # If no paragraph break, try to break at a sentence with a reasonable time limit
                search_start_time = time.time()
                # Simple sentence end detection (faster than regex)
                for i in range(len(search_text)-1, 0, -1):
                    # Check for sentence ending punctuation followed by whitespace
                    if (search_text[i-1] in '.!?') and (search_text[i] in ' \t\n'):
                        # Found a sentence end
                        end = search_start + i + 1  # +1 to include the whitespace
                        break
                    
                    # Time check inside the loop
                    if time.time() - search_start_time > max_search_time:
                        logging.warning(f"Sentence break search taking too long, using simple boundary")
                        break
        
        # Add this chunk to our list
        chunks.append(text[start:end])
        
        # Calculate the start of the next chunk with overlap
        start = max(start, end - overlap)
        
        # Ensure we're making progress
        if start >= end:
            start = end
    
    # Log performance
    end_time = time.time()
    processing_time = end_time - start_time
    logging.info(f"Paragraph-aware chunking completed in {processing_time:.2f} seconds. Created {len(chunks)} chunks.")
    
    return chunks 