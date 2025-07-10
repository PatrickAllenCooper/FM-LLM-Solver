#!/usr/bin/env python3
"""Certificate Pipeline Integration Test"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.certificate_extraction import extract_certificate_from_llm_output

def test_pipeline():
    """Test certificate pipeline integration"""
    print("Testing certificate pipeline integration...")
    
    # Test extraction
    test_input = "B(x,y) = x**2 + y**2 - 1.0"
    result = extract_certificate_from_llm_output(test_input, ["x", "y"])
    cert = result[0] if isinstance(result, tuple) else result
    
    if cert == "x**2 + y**2 - 1.0":
        print("Pipeline test passed!")
        return True
    else:
        print(f"Pipeline test failed: expected 'x**2 + y**2 - 1.0', got '{cert}'")
        return False

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1) 