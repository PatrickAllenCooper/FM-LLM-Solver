#!/usr/bin/env python3
"""
Simple Test Runner for FM-LLM-Solver
====================================

A clean, unified interface for running all tests with proper environment detection
and GPU acceleration support.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def run_quick_check():
    """Run a quick system check"""
    print("Quick System Check...")
    
    # Check key components
    components = [
        ("utils.config_loader", "Configuration"),
        ("utils.certificate_extraction", "Certificate Extraction"),
        ("utils.verification_helpers", "Verification Helpers"),
        ("utils.numerical_checks", "Numerical Checks"),
        ("torch", "PyTorch (GPU Support)"),
    ]
    
    all_good = True
    for module, name in components:
        try:
            __import__(module)
            print(f"  OK {name}")
        except ImportError:
            print(f"  FAIL {name}")
            all_good = False
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  GPU: {gpu_name}")
        else:
            print("  GPU: Not available")
    except ImportError:
        print("  GPU: PyTorch not available")
    
    return all_good

def run_unit_tests():
    """Run unit tests"""
    print("Running Unit Tests...")
    import subprocess
    
    try:
        result = subprocess.run([
            "python", "-m", "pytest", "tests/unit/", "-v", "--tb=short"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("  Unit tests passed")
            return True
        else:
            print("  Unit tests failed")
            print(result.stdout)
            return False
    except subprocess.TimeoutExpired:
        print("  Unit tests timed out")
        return False
    except Exception as e:
        print(f"  Unit tests error: {e}")
        return False

def run_unified_suite():
    """Run the unified test suite"""
    print("Running Unified Test Suite...")
    import subprocess
    
    try:
        result = subprocess.run([
            "python", "tests/unified_test_suite.py"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("  Unified test suite passed")
            return True
        else:
            print("  Unified test suite failed")
            print(result.stdout)
            return False
    except subprocess.TimeoutExpired:
        print("  Unified test suite timed out")
        return False
    except Exception as e:
        print(f"  Unified test suite error: {e}")
        return False

def run_gpu_tests():
    """Run GPU-specific tests"""
    print("Running GPU Tests...")
    import subprocess
    
    try:
        result = subprocess.run([
            "python", "tests/unit/test_gpu_accelerated_generation.py"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("  GPU tests passed")
            return True
        else:
            print("  GPU tests failed")
            print(result.stdout)
            return False
    except subprocess.TimeoutExpired:
        print("  GPU tests timed out")
        return False
    except Exception as e:
        print(f"  GPU tests error: {e}")
        return False

def show_summary():
    """Show test summary"""
    print("Test Summary...")
    import subprocess
    
    try:
        result = subprocess.run([
            "python", "tests/test_summary.py"
        ], capture_output=True, text=True, timeout=30)
        
        print(result.stdout)
    except Exception as e:
        print(f"  Summary error: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="FM-LLM-Solver Test Runner")
    parser.add_argument("--quick", action="store_true", help="Run quick system check")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--unified", action="store_true", help="Run unified test suite")
    parser.add_argument("--gpu", action="store_true", help="Run GPU tests")
    parser.add_argument("--summary", action="store_true", help="Show test summary")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    print("FM-LLM-Solver Test Runner")
    print("=" * 40)
    
    if args.quick or not any([args.unit, args.unified, args.gpu, args.summary, args.all]):
        run_quick_check()
    
    if args.unit or args.all:
        run_unit_tests()
    
    if args.unified or args.all:
        run_unified_suite()
    
    if args.gpu or args.all:
        run_gpu_tests()
    
    if args.summary or args.all:
        show_summary()
    
    print("\nTest runner completed!")

if __name__ == "__main__":
    main() 