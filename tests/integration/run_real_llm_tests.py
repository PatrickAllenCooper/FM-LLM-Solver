#!/usr/bin/env python3
"""
Runner for Real LLM GPU Tests

This script runs comprehensive testing with actual GPU-accelerated LLM inference
to validate that our filtering and numerical checking works on real model outputs.
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('real_llm_test.log')
    ]
)

def main():
    """Main runner."""
    print("🚀 FM-LLM Solver: Real GPU LLM Testing")
    print("="*50)
    
    try:
        # Import and run tests
        from tests.gpu_real_llm_tests import RealLLMTestSuite
        
        print("🔧 Initializing test suite...")
        test_suite = RealLLMTestSuite()
        
        print("🎯 Running comprehensive real LLM tests...")
        print("⏳ This will take 5-10 minutes with real GPU inference...")
        
        # Setup and run
        test_suite.setup()
        analysis = test_suite.run_comprehensive_tests()
        results_file = test_suite.save_results()
        
        # Print summary
        print("\n" + "="*60)
        print("🎉 REAL LLM TESTING COMPLETED!")
        print("="*60)
        print(f"📊 Overall Success Rate: {analysis['overall_success_rate']:.1%}")
        print(f"🤖 Generation Success: {analysis['generation_success_rate']:.1%}")
        print(f"🔍 Extraction Success: {analysis['extraction_success_rate']:.1%}")
        print(f"🚫 Template Rejection: {analysis['template_rejection_rate']:.1%}")
        print(f"🔢 Numerical Verification: {analysis['numerical_success_rate']:.1%}")
        print(f"⏱️  Average Generation Time: {analysis['avg_generation_time']:.2f}s")
        print(f"💾 Results saved to: {results_file}")
        
        # Recommendations
        if analysis['overall_success_rate'] >= 0.8:
            print("\n✅ STATUS: EXCELLENT - Real LLM pipeline is working correctly!")
            print("   Your filtering and numerical checking is robust.")
        elif analysis['overall_success_rate'] >= 0.6:
            print("\n⚠️  STATUS: GOOD - Minor improvements needed")
            print("   Consider tuning extraction patterns or verification thresholds.")
        else:
            print("\n❌ STATUS: NEEDS IMPROVEMENT")
            print("   Significant issues found with real LLM output processing.")
            print("   Review extraction logic and numerical verification.")
        
        return 0 if analysis['overall_success_rate'] >= 0.6 else 1
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Try: pip install torch transformers accelerate bitsandbytes")
        return 2
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"❌ GPU Error: {e}")
            print("   Make sure CUDA is available and GPU has enough memory")
        else:
            print(f"❌ Runtime Error: {e}")
        return 3
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        logging.exception("Test suite failed")
        return 4
    finally:
        try:
            test_suite.cleanup()
        except:
            pass

if __name__ == "__main__":
    exit(main()) 