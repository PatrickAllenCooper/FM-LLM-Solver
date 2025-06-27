#!/usr/bin/env python3
"""Quick integration test runner with immediate feedback - no heavy ML loading."""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("🚀 Starting Quick Integration Tests...")
    print("=" * 50)
    
    try:
        # Import lightweight components
        print("📋 Testing component imports...")
        from utils.config_loader import load_config
        from web_interface.verification_service import VerificationService
        from unittest.mock import Mock
        
        # Test 1: Configuration Loading
        print("🔧 Testing configuration loading...")
        config = load_config("config.yaml")
        config_success = config is not None
        print(f"   ✅ Config loading: {'PASS' if config_success else 'FAIL'}")
        
        # Test 2: Verification Service (lightweight)
        print("🔍 Testing verification service...")
        verification_service = VerificationService(config)
        
        # Test parsing
        test_system = """System Dynamics: dx/dt = -x**3 - y, dy/dt = x - y**3
Initial Set: x**2 + y**2 <= 0.1
Unsafe Set: x >= 1.5"""
        
        parsed = verification_service.parse_system_description(test_system)
        bounds = verification_service.create_sampling_bounds(parsed)
        
        parsing_success = (len(parsed.get('variables', [])) > 0 and 
                          len(bounds) > 0)
        print(f"   ✅ System parsing: {'PASS' if parsing_success else 'FAIL'}")
        
        # Test 3: Certificate Generator (import only, no loading)
        print("🎯 Testing certificate generator import...")
        try:
            from web_interface.certificate_generator import CertificateGenerator
            # Don't initialize - just test import and basic methods
            cert_gen_import = True
            print("   ✅ Certificate Generator import: PASS")
        except Exception as e:
            cert_gen_import = False
            print(f"   ❌ Certificate Generator import: FAIL - {e}")
        
        # Test 4: Text processing (using actual methods without ML loading)
        if cert_gen_import:
            print("📝 Testing certificate extraction patterns...")
            # Create instance with mock config to avoid ML loading
            mock_config = Mock()
            mock_config.fine_tuning = Mock()
            mock_config.fine_tuning.base_model_name = "mock"
            mock_config.paths = Mock()
            mock_config.paths.ft_output_dir = "/mock/path"
            mock_config.knowledge_base = Mock()
            mock_config.knowledge_base.barrier_certificate_type = "discrete"
            
            try:
                cert_gen = CertificateGenerator.__new__(CertificateGenerator)
                cert_gen.config = mock_config
                cert_gen.models = {}
                cert_gen.knowledge_bases = {}
                cert_gen.embedding_model = None
                
                # Test extraction without loading models
                test_output = """BARRIER_CERTIFICATE_START
B(x, y) = x**2 + y**2
BARRIER_CERTIFICATE_END"""
                
                extracted = cert_gen.extract_certificate_from_output(test_output)
                extraction_success = extracted is not None
                print(f"   ✅ Certificate extraction: {'PASS' if extraction_success else 'FAIL'}")
                
                # Test template detection
                template_output = """BARRIER_CERTIFICATE_START
B(x, y) = ax**2 + bxy + cy**2 + dx + ey + f
BARRIER_CERTIFICATE_END"""
                
                template_extracted = cert_gen.extract_certificate_from_output(template_output)
                template_rejected = (template_extracted is None or 
                                   cert_gen._is_template_expression(template_extracted))
                print(f"   ✅ Template rejection: {'PASS' if template_rejected else 'FAIL'}")
                
            except Exception as e:
                extraction_success = False
                template_rejected = False
                print(f"   ❌ Certificate processing: FAIL - {e}")
        else:
            extraction_success = False
            template_rejected = False
        
        # Test 5: Verification Integration (lightweight)
        print("⚖️ Testing verification integration...")
        try:
            test_certificate = "x**2 + y**2"
            # Test with minimal samples to avoid long computation
            result = verification_service.verify_certificate(
                test_certificate,
                test_system,
                param_overrides={'num_samples_lie': 50, 'num_samples_boundary': 25}
            )
            
            verification_success = (result is not None and 
                                  isinstance(result, dict) and 
                                  'overall_success' in result)
            print(f"   ✅ Verification workflow: {'PASS' if verification_success else 'FAIL'}")
            
        except Exception as e:
            verification_success = False
            print(f"   ❌ Verification workflow: FAIL - {e}")
        
        # Calculate results
        total_tests = 6
        passed_tests = sum([
            config_success,
            parsing_success,
            cert_gen_import,
            extraction_success,
            template_rejected,
            verification_success
        ])
        
        success_rate = passed_tests / total_tests
        
        # Display results
        print("\n" + "=" * 50)
        print("🎯 QUICK INTEGRATION TEST RESULTS")
        print("=" * 50)
        print(f"✅ Tests Passed: {passed_tests}")
        print(f"❌ Tests Failed: {total_tests - passed_tests}")
        print(f"📈 Success Rate: {success_rate:.1%}")
        
        # Determine readiness level
        if success_rate >= 0.8:
            readiness_level = "INTEGRATION_READY"
            print("🎚️  Readiness Level: INTEGRATION_READY")
            print("\n💡 Key Insights:")
            print("   1. Core components are working well")
            print("   2. Text processing and parsing functional")
            print("   3. Ready for full ML model integration")
            status_code = 0
        elif success_rate >= 0.6:
            readiness_level = "BASIC_FUNCTIONAL"
            print("🎚️  Readiness Level: BASIC_FUNCTIONAL")
            print("\n💡 Key Insights:")
            print("   1. Basic functionality working")
            print("   2. Some components need attention")
            print("   3. Focus on fixing failed tests")
            status_code = 1
        else:
            readiness_level = "NEEDS_WORK"
            print("🎚️  Readiness Level: NEEDS_WORK")
            print("\n💡 Key Insights:")
            print("   1. Multiple components failing")
            print("   2. Focus on basic functionality first")
            print("   3. Check dependencies and configuration")
            status_code = 1
        
        if success_rate >= 0.8:
            print("\n🎉 System is ready for advanced testing!")
            print("   • Consider running full ML model tests")
            print("   • Deploy testbench for continuous validation")
            print("   • Begin user acceptance testing")
        else:
            print(f"\n⚠️  System needs improvement (Level: {readiness_level})")
            print("   • Fix failing component tests")
            print("   • Review error messages above")
            print("   • Ensure all dependencies are installed")
        
        return status_code
            
    except Exception as e:
        print(f"\n❌ Quick integration testing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 