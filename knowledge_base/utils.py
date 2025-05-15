"""
Utility functions for knowledge base operations.
"""

import time
from datetime import datetime, timedelta

def format_time_remaining(total_items, completed_items, elapsed_time, item_name="item"):
    """
    Calculate and format the estimated time remaining based on progress.
    
    Parameters
    ----------
    total_items : int
        Total number of items to process
    completed_items : int
        Number of items already completed
    elapsed_time : float
        Time elapsed so far in seconds
    item_name : str, optional
        Name of the item type for display purposes
        
    Returns
    -------
    str
        Formatted string with time estimates
    """
    if completed_items == 0:
        return "Calculating time estimate..."
    
    items_per_second = completed_items / elapsed_time
    remaining_items = total_items - completed_items
    
    # Avoid division by zero
    if items_per_second <= 0:
        return "Time estimate unavailable"
    
    estimated_seconds_remaining = remaining_items / items_per_second
    
    # Format as HH:MM:SS
    time_remaining = timedelta(seconds=int(estimated_seconds_remaining))
    eta = datetime.now() + time_remaining
    
    # Return a formatted string
    return (
        f"Progress: {completed_items}/{total_items} {item_name}s | "
        f"Rate: {items_per_second:.2f} {item_name}s/sec | "
        f"Est. time remaining: {str(time_remaining)} | "
        f"ETA: {eta.strftime('%H:%M:%S')}"
    )

def print_time_estimate(start_time, total_items, completed_items, item_name="item"):
    """
    Print a time estimate based on current progress.
    
    Parameters
    ----------
    start_time : float
        Start time in seconds (from time.time())
    total_items : int
        Total number of items to process
    completed_items : int
        Number of items already completed
    item_name : str, optional
        Name of the item type for display purposes
    """
    elapsed_time = time.time() - start_time
    estimate = format_time_remaining(total_items, completed_items, elapsed_time, item_name)
    print(estimate)
    return estimate 