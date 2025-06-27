#!/usr/bin/env python3
"""
Quick test to validate the verification boundary condition fix.
Tests the known correct barrier certificate that was previously failing.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import load_config
from web_interface.verification_service import VerificationService

def test_verification_fix():
    """Test the critical verification fix with a known correct certificate."""
    print("🔧 TESTING VERIFICATION BOUNDARY CONDITION FIX")
    print("=" * 60)
    
    # Load configuration
    config = load_config("config.yaml")
    verification_service = VerificationService(config)
    
    # Test case: Perfect Lyapunov function that should pass
    test_certificate = "x**2 + y**2"
    test_system = """System Dynamics: dx/dt = -x, dy/dt = -y
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 4.0"""
    
    print(f"🧪 Testing Certificate: {test_certificate}")
    print(f"📋 System: Stable linear system with circular initial set")
    print(f"📊 Theoretical Result: SHOULD PASS (Perfect Lyapunov function)")
    print()
    
    # Run verification
    try:
        result = verification_service.verify_certificate(
            test_certificate,
            test_system,
            param_overrides={
                'num_samples_lie': 100,
                'num_samples_boundary': 50,
                'numerical_tolerance': 1e-6
            }
        )
        
        # Analyze results
        overall_success = result.get('overall_success', False)
        numerical_passed = result.get('numerical_passed', False)
        sos_passed = result.get('sos_passed', False)
        
        print("📊 VERIFICATION RESULTS:")
        print(f"   Overall Success: {'✅ PASS' if overall_success else '❌ FAIL'}")
        print(f"   Numerical: {'✅ PASS' if numerical_passed else '❌ FAIL'}")
        print(f"   SOS: {'✅ PASS' if sos_passed else '❌ FAIL'}")
        
        # Check if the fix worked
        if numerical_passed:
            print("\n🎉 SUCCESS: Numerical verification now PASSES!")
            print("✅ The boundary condition fix is working correctly!")
            print("🔧 Set-relative tolerance logic successfully implemented")
        else:
            print("\n⚠️ PARTIAL: Numerical verification still failing")
            print("🔍 May need additional debugging")
        
        if overall_success:
            print("\n🏆 COMPLETE SUCCESS: Certificate verification PASSED!")
            print("✅ The systematic rejection issue has been RESOLVED!")
        else:
            print(f"\n⚠️ PARTIAL SUCCESS: Overall verification result: {overall_success}")
        
        # Show detailed feedback
        details = result.get('details', {})
        if 'numerical' in details:
            numerical_details = details['numerical']
            reason = numerical_details.get('reason', 'No details available')
            print(f"\n📝 Numerical Details: {reason}")
        
        return overall_success
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_verification_fix()
    if success:
        print("\n🎯 CONCLUSION: Verification fix SUCCESSFUL!")
        print("✅ Ready for production deployment")
    else:
        print("\n🔧 CONCLUSION: May need additional fixes")
        print("⚠️ Investigate remaining issues")
    
    sys.exit(0 if success else 1) 