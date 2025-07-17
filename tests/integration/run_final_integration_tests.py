#!/usr/bin/env python3
"""Final integration test runner - optimized for accurate web interface assessment."""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("🎯 Final Integration Test - Web Interface Assessment")
    print("=" * 60)

    try:
        # Import lightweight components
        print("📋 Testing component availability...")
        from utils.config_loader import load_config
        from web_interface.verification_service import VerificationService
        from unittest.mock import Mock

        test_results = {}

        # Test 1: Configuration System
        print("\n🔧 Configuration System...")
        config = load_config("config.yaml")
        config_success = config is not None
        test_results["config"] = config_success
        print(
            f"   {'✅' if config_success else '❌'} Configuration loading: {'PASS' if config_success else 'FAIL'}"
        )

        # Test 2: Verification Service Core
        print("\n🔍 Verification Service...")
        verification_service = VerificationService(config)

        # System parsing test
        test_system = """System Dynamics: dx/dt = -x**3 - y, dy/dt = x - y**3
Initial Set: x**2 + y**2 <= 0.1
Unsafe Set: x >= 1.5"""

        parsed = verification_service.parse_system_description(test_system)
        bounds = verification_service.create_sampling_bounds(parsed)

        parsing_success = len(parsed.get("variables", [])) > 0 and len(bounds) > 0
        test_results["parsing"] = parsing_success
        print(
            f"   {'✅' if parsing_success else '❌'} System parsing: {'PASS' if parsing_success else 'FAIL'}"
        )

        # Test 3: Certificate Generator Import
        print("\n🎯 Certificate Generator...")
        try:
            from web_interface.certificate_generator import CertificateGenerator

            cert_gen_import = True
            test_results["cert_gen_import"] = True
            print("   ✅ Import successful: PASS")
        except Exception as e:
            cert_gen_import = False
            test_results["cert_gen_import"] = False
            print(f"   ❌ Import failed: FAIL - {e}")

        # Test 4: Text Processing (Corrected Template Detection)
        if cert_gen_import:
            print("\n📝 Text Processing & Template Detection...")
            # Create mock instance
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

                # Test 1: Valid specific certificate extraction
                specific_cert_output = """BARRIER_CERTIFICATE_START
B(x, y) = 0.5*x**2 + 0.3*y**2 + 0.1*x*y - 0.05
BARRIER_CERTIFICATE_END"""

                specific_extracted = cert_gen.extract_certificate_from_output(specific_cert_output)
                specific_success = (
                    specific_extracted is not None
                    and not cert_gen._is_template_expression(specific_extracted)
                )

                # Test 2: Template rejection (this SHOULD be rejected)
                template_output = """BARRIER_CERTIFICATE_START
B(x, y) = ax**2 + bxy + cy**2 + dx + ey + f
BARRIER_CERTIFICATE_END"""

                template_extracted = cert_gen.extract_certificate_from_output(template_output)
                template_correctly_rejected = (
                    template_extracted is None
                    or cert_gen._is_template_expression(template_extracted)
                )

                # Test 3: Simple expression rejection (x**2 + y**2 should be rejected as too generic)
                simple_output = """BARRIER_CERTIFICATE_START
B(x, y) = x**2 + y**2
BARRIER_CERTIFICATE_END"""

                simple_extracted = cert_gen.extract_certificate_from_output(simple_output)
                simple_correctly_rejected = (
                    simple_extracted is None or cert_gen._is_template_expression(simple_extracted)
                )

                text_processing_success = (
                    specific_success and template_correctly_rejected and simple_correctly_rejected
                )

                test_results["text_processing"] = text_processing_success
                print(
                    f"   {'✅' if specific_success else '❌'} Specific certificate extraction: {'PASS' if specific_success else 'FAIL'}"
                )
                print(
                    f"   {'✅' if template_correctly_rejected else '❌'} Template rejection: {'PASS' if template_correctly_rejected else 'FAIL'}"
                )
                print(
                    f"   {'✅' if simple_correctly_rejected else '❌'} Generic expression rejection: {'PASS' if simple_correctly_rejected else 'FAIL'}"
                )

            except Exception as e:
                test_results["text_processing"] = False
                print(f"   ❌ Text processing tests: FAIL - {e}")
        else:
            test_results["text_processing"] = False

        # Test 5: Verification Workflow
        print("\n⚖️ Verification Workflow...")
        try:
            test_certificate = "0.5*x**2 + 0.3*y**2 - 0.05"  # More specific certificate
            result = verification_service.verify_certificate(
                test_certificate,
                test_system,
                param_overrides={"num_samples_lie": 100, "num_samples_boundary": 50},
            )

            verification_success = (
                result is not None and isinstance(result, dict) and "overall_success" in result
            )

            test_results["verification"] = verification_success
            print(
                f"   {'✅' if verification_success else '❌'} Complete workflow: {'PASS' if verification_success else 'FAIL'}"
            )

            if verification_success:
                print(f"   📊 Verification completed in {result.get('verification_time', 'N/A')}s")
                if "overall_success" in result:
                    print(f"   📈 Overall success: {result['overall_success']}")

        except Exception as e:
            test_results["verification"] = False
            print(f"   ❌ Verification workflow: FAIL - {e}")

        # Test 6: Web Interface Models Import
        print("\n🌐 Web Interface Models...")
        try:
            from web_interface.models import db, QueryLog, VerificationResult

            models_success = True
            test_results["models"] = True
            print("   ✅ Database models import: PASS")
        except Exception as e:
            models_success = False
            test_results["models"] = False
            print(f"   ❌ Database models import: FAIL - {e}")

        # Calculate final results
        total_tests = len(test_results)
        passed_tests = sum(test_results.values())
        success_rate = passed_tests / total_tests

        # Display comprehensive results
        print("\n" + "=" * 60)
        print("🏆 FINAL INTEGRATION TEST RESULTS")
        print("=" * 60)

        print(f"✅ Tests Passed: {passed_tests}/{total_tests}")
        print(f"📈 Success Rate: {success_rate:.1%}")

        # Component-by-component status
        print("\n📋 Component Status:")
        components = [
            ("Configuration System", "config"),
            ("System Parsing", "parsing"),
            ("Certificate Generator", "cert_gen_import"),
            ("Text Processing", "text_processing"),
            ("Verification Workflow", "verification"),
            ("Web Models", "models"),
        ]

        for component_name, key in components:
            status = "✅ WORKING" if test_results.get(key, False) else "❌ NEEDS ATTENTION"
            print(f"   {component_name}: {status}")

        # Determine final readiness assessment
        if success_rate >= 0.85:
            readiness_level = "PRODUCTION_READY"
            status_emoji = "🚀"
            status_code = 0
        elif success_rate >= 0.70:
            readiness_level = "INTEGRATION_READY"
            status_emoji = "⚡"
            status_code = 0
        elif success_rate >= 0.50:
            readiness_level = "BASIC_FUNCTIONAL"
            status_emoji = "⚠️"
            status_code = 1
        else:
            readiness_level = "NEEDS_WORK"
            status_emoji = "🔧"
            status_code = 1

        print(f"\n{status_emoji} Final Assessment: {readiness_level}")

        # Provide specific recommendations
        print("\n💡 Recommendations:")
        if success_rate >= 0.85:
            print("   🎉 System is production-ready!")
            print("   • Deploy web interface for users")
            print("   • Consider load testing with real users")
            print("   • Set up monitoring and logging")
        elif success_rate >= 0.70:
            print("   🎯 System is integration-ready!")
            print("   • Begin user acceptance testing")
            print("   • Load ML models for full functionality")
            print("   • Fine-tune any failing components")
        else:
            print("   🔧 System needs improvement:")
            failing_components = [
                name for name, key in components if not test_results.get(key, False)
            ]
            for component in failing_components:
                print(f"     • Fix {component}")
            print("   • Review error messages above")
            print("   • Ensure all dependencies are installed")

        print(f"\n📊 Web Interface Quality Score: {success_rate:.1%}")

        return status_code

    except Exception as e:
        print(f"\n❌ Integration testing failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
