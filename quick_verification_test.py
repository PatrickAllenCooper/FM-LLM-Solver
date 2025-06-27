#!/usr/bin/env python3
"""
Quick Verification Fix Test - With Immediate Output
Tests the critical boundary condition fix with real-time progress.
"""

import sys
import os
import time
from pathlib import Path

print("🚀 STARTING VERIFICATION FIX TEST")
print("=" * 50)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
print(f"✅ Project root: {PROJECT_ROOT}")

try:
    print("📦 Loading configuration...")
    from utils.config_loader import load_config
    config = load_config("config.yaml")
    print("✅ Configuration loaded successfully")
    
    print("🔧 Initializing verification service...")
    from web_interface.verification_service import VerificationService
    verification_service = VerificationService(config)
    print("✅ Verification service initialized")
    
except Exception as e:
    print(f"❌ INITIALIZATION FAILED: {e}")
    sys.exit(1)

def test_simple_case():
    """Test the simplest possible case with immediate feedback."""
    print("\n🧪 TESTING SIMPLE CASE")
    print("-" * 30)
    
    # Simplest test case
    certificate = "x**2 + y**2"
    system = """System Dynamics: dx/dt = -x, dy/dt = -y
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 4.0"""
    
    print(f"📋 Certificate: {certificate}")
    print(f"📋 System: Simple stable linear system")
    print(f"🎯 Expected: SHOULD PASS (Perfect Lyapunov function)")
    
    print("\n⏳ Starting verification (minimal samples for speed)...")
    start_time = time.time()
    
    try:
        # Use minimal samples for quick test
        result = verification_service.verify_certificate(
            certificate,
            system,
            param_overrides={
                'num_samples_lie': 10,          # Very small for quick test
                'num_samples_boundary': 5,      # Very small for quick test
                'numerical_tolerance': 1e-6,
                'attempt_sos': False,           # Skip SOS for speed
                'attempt_optimization': False   # Skip optimization for speed
            }
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Verification completed in {elapsed:.2f}s")
        
        # Check results
        overall = result.get('overall_success', False)
        numerical = result.get('numerical_passed', False)
        
        print(f"\n📊 RESULTS:")
        print(f"   Overall: {'✅ PASS' if overall else '❌ FAIL'}")
        print(f"   Numerical: {'✅ PASS' if numerical else '❌ FAIL'}")
        
        # Get reason
        details = result.get('details', {})
        if 'numerical' in details:
            reason = details['numerical'].get('reason', 'No reason provided')
            print(f"   Reason: {reason}")
        
        if numerical:
            print(f"\n🎉 SUCCESS! The boundary condition fix is working!")
            print(f"✅ Numerical verification now PASSES for correct certificates")
            return True
        else:
            print(f"\n⚠️ Still failing - may need more investigation")
            return False
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Test failed after {elapsed:.2f}s: {e}")
        return False

def test_system_parsing():
    """Test just the system parsing to ensure that works."""
    print("\n🔍 TESTING SYSTEM PARSING")
    print("-" * 30)
    
    system = """System Dynamics: dx/dt = -x, dy/dt = -y
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 4.0"""
    
    try:
        parsed = verification_service.parse_system_description(system)
        print(f"✅ Variables: {parsed.get('variables', 'Not found')}")
        print(f"✅ Dynamics: {parsed.get('dynamics', 'Not found')}")
        print(f"✅ Initial set: {parsed.get('initial_set', 'Not found')}")
        print(f"✅ Unsafe set: {parsed.get('unsafe_set', 'Not found')}")
        return True
    except Exception as e:
        print(f"❌ System parsing failed: {e}")
        return False

def main():
    """Run the quick verification tests."""
    print(f"\n🎯 QUICK VERIFICATION FIX VALIDATION")
    print(f"📅 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test 1: System parsing
    print(f"\n{'='*50}")
    parsing_ok = test_system_parsing()
    
    if not parsing_ok:
        print(f"\n❌ FAILED: System parsing not working")
        return False
    
    # Test 2: Simple verification
    print(f"\n{'='*50}")
    verification_ok = test_simple_case()
    
    # Summary
    print(f"\n{'='*50}")
    print(f"🏁 TEST SUMMARY")
    print(f"📊 System Parsing: {'✅ PASS' if parsing_ok else '❌ FAIL'}")
    print(f"📊 Verification Fix: {'✅ PASS' if verification_ok else '❌ FAIL'}")
    
    if verification_ok:
        print(f"\n🎉 OVERALL: VERIFICATION FIX SUCCESSFUL!")
        print(f"✅ The boundary condition fix is working correctly")
        print(f"🚀 Ready for production testing")
        return True
    else:
        print(f"\n⚠️ OVERALL: Additional debugging needed")
        print(f"🔧 The fix may need refinement")
        return False

if __name__ == "__main__":
    try:
        success = main()
        print(f"\n🏁 Exit Code: {'0 (SUCCESS)' if success else '1 (NEEDS WORK)'}")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n⏸️ Test interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n💥 Test crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3) 