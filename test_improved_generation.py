#!/usr/bin/env python3
"""
Test the improved generation system with mathematical guidance
"""

import os
import sys
import time
import logging

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils.config_loader import load_config
from web_interface.certificate_generator import CertificateGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_improved_generation():
    """Test the improved generation with mathematical guidance"""
    logger.info("🧪 TESTING IMPROVED GENERATION SYSTEM")
    logger.info("=" * 60)
    
    # Test the problematic case with improved guidance
    system_desc = "System Dynamics: dx/dt = -x, dy/dt = -y\nInitial Set: x**2 + y**2 <= 0.25\nUnsafe Set: x**2 + y**2 >= 4.0"
    domain_bounds = {"x": [-3, 3], "y": [-3, 3]}
    
    logger.info(f"📋 System: {system_desc}")
    logger.info(f"🎯 Expected improvement: Better constant selection (between 0.25 and 4.0)")
    
    try:
        config = load_config()
        generator = CertificateGenerator(config)
        
        logger.info("🚀 Testing with mathematical guidance...")
        start_time = time.time()
        
        result = generator.generate_certificate(
            system_desc,
            "base", 
            rag_k=0,
            domain_bounds=domain_bounds
        )
        
        generation_time = time.time() - start_time
        
        logger.info("\n🔍 IMPROVED RESULTS:")
        logger.info("-" * 40)
        logger.info(f"✅ Generation successful: {result['success']}")
        logger.info(f"⏱️  Generation time: {generation_time:.1f} seconds")
        logger.info(f"🔧 Generated certificate: {result.get('certificate', 'None')}")
        logger.info(f"📝 LLM output preview: {result.get('llm_output', '')[:200]}...")
        
        # Analyze the certificate
        certificate = result.get('certificate', '')
        
        if certificate:
            # Check if it's a quadratic of the form x**2 + y**2 - c
            if 'x**2 + y**2' in certificate and '-' in certificate:
                # Extract the constant
                try:
                    parts = certificate.split('-')
                    if len(parts) == 2:
                        constant = float(parts[1].strip())
                        logger.info(f"📊 Extracted constant: {constant}")
                        
                        # Check if it's in the expected range (0.25 to 4.0)
                        if 0.25 <= constant <= 4.0:
                            logger.info(f"✅ EXCELLENT: Constant {constant} is in optimal range [0.25, 4.0]")
                            if abs(constant - 2.125) < 1.0:  # Close to midpoint
                                logger.info(f"🎯 PERFECT: Constant is near optimal midpoint!")
                        else:
                            logger.warning(f"⚠️  Constant {constant} is outside optimal range")
                            
                except Exception as e:
                    logger.warning(f"Could not parse constant: {e}")
            
            # Check for improvements
            improvements = []
            if not any(p in certificate for p in ['α', 'β', 'γ', 'C', 'a', 'b', 'c']):
                improvements.append("No placeholder variables")
            if len(certificate.strip()) > 5:
                improvements.append("Complete expression")
            if generation_time < 400:
                improvements.append("Fast generation")
                
            logger.info(f"🎉 IMPROVEMENTS: {', '.join(improvements)}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return None

if __name__ == "__main__":
    test_improved_generation() 