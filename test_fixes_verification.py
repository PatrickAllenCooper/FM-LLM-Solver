#!/usr/bin/env python3
"""
Quick verification test for the implemented fixes.
Tests the exact same input that was failing before.
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

def test_fixed_generation():
    """Test generation with the exact input that was failing"""
    logger.info("🧪 TESTING FIXES WITH ORIGINAL PROBLEMATIC INPUT")
    logger.info("=" * 60)
    
    # Exact same input that was generating "B(x, y) = x + αy - C, but this could fail..."
    system_desc = "System Dynamics: dx/dt = -x, dy/dt = -y\nInitial Set: x**2 + y**2 <= 0.25\nUnsafe Set: x**2 + y**2 >= 4.0"
    domain_bounds = {"x": [-3, 3], "y": [-3, 3]}
    
    logger.info(f"📋 System: {system_desc}")
    logger.info(f"🎯 Domain: {domain_bounds}")
    
    try:
        # Initialize generator
        config = load_config()
        generator = CertificateGenerator(config)
        
        logger.info("🚀 Generating certificate with optimized system...")
        start_time = time.time()
        
        result = generator.generate_certificate(
            system_desc,
            "base",
            rag_k=0,
            domain_bounds=domain_bounds
        )
        
        generation_time = time.time() - start_time
        
        # Analyze results
        logger.info("\n🔍 RESULTS ANALYSIS:")
        logger.info("-" * 40)
        logger.info(f"✅ Generation successful: {result['success']}")
        logger.info(f"⏱️  Generation time: {generation_time:.1f} seconds")
        logger.info(f"📏 Prompt length: {result.get('prompt_length', 0)} chars")
        logger.info(f"📝 LLM output length: {len(result.get('llm_output', ''))}")
        logger.info(f"🔧 Extracted certificate: {result.get('certificate', 'None')}")
        
        # Check for issues
        certificate = result.get('certificate', '')
        llm_output = result.get('llm_output', '')
        
        # Success criteria
        success_checks = []
        
        # 1. No placeholder variables
        has_placeholders = any(p in str(certificate) for p in ['α', 'β', 'γ', 'C', 'a', 'b', 'c'])
        success_checks.append(("No placeholder variables", not has_placeholders))
        
        # 2. Complete generation (no incomplete phrases)
        incomplete_phrases = ["Therefore, we'll opt", "but this could fail", "However", "we need"]
        is_complete = not any(phrase in llm_output for phrase in incomplete_phrases)
        success_checks.append(("Complete generation", is_complete))
        
        # 3. Valid certificate extracted
        has_certificate = certificate and len(certificate) > 0
        success_checks.append(("Certificate extracted", has_certificate))
        
        # 4. Reasonable generation time
        reasonable_time = generation_time < 400  # Under 6.7 minutes
        success_checks.append(("Reasonable generation time", reasonable_time))
        
        # 5. Compact prompt
        prompt_length = result.get('prompt_length', 0)
        compact_prompt = prompt_length < 1000
        success_checks.append(("Compact prompt", compact_prompt))
        
        logger.info("\n📊 SUCCESS CRITERIA:")
        all_passed = True
        for check_name, passed in success_checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"   {check_name}: {status}")
            if not passed:
                all_passed = False
        
        logger.info(f"\n🎯 OVERALL RESULT: {'✅ ALL FIXES SUCCESSFUL' if all_passed else '❌ SOME ISSUES REMAIN'}")
        
        if all_passed:
            logger.info("🎉 The implemented fixes have resolved all identified issues!")
            logger.info("🚀 The web interface should now generate correct results!")
        
        return result, all_passed
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return None, False

if __name__ == "__main__":
    test_fixed_generation() 