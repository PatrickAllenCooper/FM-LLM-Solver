#!/usr/bin/env python3
"""
Barrier Certificate Theory Fix

Fixes the fundamental barrier certificate design by creating certificates that:
1. B(x) ≤ 0 on initial set
2. dB/dt ≤ 0 in safe region
3. B(x) ≥ 0 on unsafe set (THIS WAS THE MISSING PIECE!)

This ensures proper barrier certificate theory compliance.
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import load_config
from web_interface.verification_service import VerificationService

logger = logging.getLogger(__name__)


class BarrierCertificateTheoryFixer:
    """Fixes barrier certificate theory violations for proper verification."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the theory fixer."""
        self.config = load_config(config_path or "config.yaml")
        self.verification_service = VerificationService(self.config)

    def create_theory_correct_certificates(self) -> List[Dict[str, Any]]:
        """Create barrier certificates that satisfy proper theory requirements."""
        return [
            # === SEPARATION-BASED CERTIFICATES ===
            {
                "name": "radial_separation_barrier",
                "system": """System Dynamics: dx/dt = -0.5*x, dy/dt = -0.5*y
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 4.0""",
                "certificate": "x**2 + y**2 - 2.0",  # Negative on initial (≤0.25), positive on unsafe (≥4.0)
                "theory_explanation": "B(x)=-1.75 on initial boundary, B(x)=2.0 on unsafe boundary",
                "expected_success": True,
            },
            {
                "name": "scaled_radial_barrier",
                "system": """System Dynamics: dx/dt = -x, dy/dt = -2*y
Initial Set: x**2 + y**2 <= 0.5
Unsafe Set: x**2 + y**2 >= 2.0""",
                "certificate": "x**2 + y**2 - 1.25",  # Midpoint between 0.5 and 2.0
                "theory_explanation": "B(x)=-0.75 on initial boundary, B(x)=0.75 on unsafe boundary",
                "expected_success": True,
            },
            # === ELLIPTICAL BARRIERS ===
            {
                "name": "elliptical_separation",
                "system": """System Dynamics: dx/dt = -2*x, dy/dt = -y
Initial Set: 4*x**2 + y**2 <= 1.0
Unsafe Set: x**2 + y**2 >= 4.0""",
                "certificate": "x**2 + y**2 - 2.5",  # Separates elliptical initial from circular unsafe
                "theory_explanation": "Separates elliptical initial set from circular unsafe set",
                "expected_success": True,
            },
            # === LINEAR SEPARATION BARRIERS ===
            {
                "name": "linear_halfspace_barrier",
                "system": """System Dynamics: dx/dt = -x - 0.1*y, dy/dt = -0.1*x - y
Initial Set: x <= -1.0 and y**2 <= 0.25
Unsafe Set: x >= 1.0""",
                "certificate": "x",  # B(x) ≤ -1 on initial, B(x) ≥ 1 on unsafe
                "theory_explanation": "Linear barrier separating left halfspace from right halfspace",
                "expected_success": True,
            },
            # === DISCRETE TIME BARRIERS ===
            {
                "name": "discrete_contractive_correct",
                "system": """System Dynamics: x{k+1} = 0.8*x{k}, y{k+1} = 0.8*y{k}
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 1.0""",
                "certificate": "x**2 + y**2 - 0.625",  # Midpoint: (0.25 + 1.0)/2 = 0.625
                "theory_explanation": "Quadratic barrier for contractive discrete system",
                "expected_success": True,
            },
            # === COMPLEX DYNAMICS ===
            {
                "name": "nonlinear_polynomial_barrier",
                "system": """System Dynamics: dx/dt = -x**3, dy/dt = -y**3
Initial Set: x**2 + y**2 <= 0.1
Unsafe Set: x**2 + y**2 >= 1.0""",
                "certificate": "x**2 + y**2 - 0.55",  # Well-separated barrier
                "theory_explanation": "Quadratic barrier for cubic dynamics",
                "expected_success": True,
            },
        ]

    def verify_theory_correct_certificates(self) -> Dict[str, Any]:
        """Test verification with theory-correct certificates."""
        test_certificates = self.create_theory_correct_certificates()
        results = {}
        successful_verifications = 0

        # Use optimized parameters from previous optimization
        optimal_params = {
            "num_samples_lie": 200,
            "num_samples_boundary": 100,
            "tolerance": 1e-6,
            "optimization_timeout": 15,
        }

        logger.info(f"Testing {len(test_certificates)} theory-correct certificates...")

        for cert_test in test_certificates:
            try:
                start_time = time.time()

                result = self.verification_service.verify_certificate(
                    cert_test["certificate"], cert_test["system"], param_overrides=optimal_params
                )

                verification_time = time.time() - start_time

                if result and isinstance(result, dict):
                    overall_success = result.get("overall_success", False)
                    numerical_passed = result.get("numerical_passed", False)
                    symbolic_passed = result.get("symbolic_passed", False)
                    sos_passed = result.get("sos_passed", False)

                    # Theory correctness: should pass if designed correctly
                    theory_correct = overall_success == cert_test["expected_success"]

                    results[cert_test["name"]] = {
                        "overall_success": overall_success,
                        "theory_correct": theory_correct,
                        "verification_time": verification_time,
                        "numerical_passed": numerical_passed,
                        "symbolic_passed": symbolic_passed,
                        "sos_passed": sos_passed,
                        "certificate": cert_test["certificate"],
                        "theory_explanation": cert_test["theory_explanation"],
                        "expected_success": cert_test["expected_success"],
                        "verification_reason": result.get("reason", "N/A"),
                    }

                    if theory_correct:
                        successful_verifications += 1
                        if overall_success:
                            logger.info(f"✅ {cert_test['name']}: SUCCESS (as expected)")
                        else:
                            logger.info(f"✅ {cert_test['name']}: CORRECTLY FAILED (as expected)")
                    else:
                        if overall_success:
                            logger.warning(f"❌ {cert_test['name']}: UNEXPECTED SUCCESS")
                        else:
                            logger.warning(f"❌ {cert_test['name']}: UNEXPECTED FAILURE")

            except Exception as e:
                results[cert_test["name"]] = {
                    "overall_success": False,
                    "theory_correct": False,
                    "error": str(e),
                    "verification_time": 0.0,
                }
                logger.error(f"❌ {cert_test['name']}: ERROR - {str(e)}")

        return {
            "total_tests": len(test_certificates),
            "successful_verifications": successful_verifications,
            "success_rate": successful_verifications / len(test_certificates),
            "detailed_results": results,
        }

    def create_certificate_design_guidelines(self) -> Dict[str, Any]:
        """Create guidelines for proper barrier certificate design."""
        return {
            "fundamental_requirements": {
                "initial_set_condition": "B(x) ≤ 0 for all x in Initial Set",
                "unsafe_set_condition": "B(x) ≥ 0 for all x in Unsafe Set",
                "lie_derivative_condition": "dB/dt ≤ 0 in Safe Region",
                "separation_requirement": "Certificate must separate initial and unsafe sets",
            },
            "design_patterns": {
                "radial_barriers": {
                    "form": "||x||² - c",
                    "design_rule": "Choose c such that: max(B on Initial) < 0 < min(B on Unsafe)",
                    "example": "For Initial: ||x||² ≤ r₁², Unsafe: ||x||² ≥ r₂², use B = ||x||² - (r₁² + r₂²)/2",
                },
                "linear_barriers": {
                    "form": "aᵀx - c",
                    "design_rule": "Choose normal vector a and offset c to separate sets",
                    "example": "For Initial: x ≤ -1, Unsafe: x ≥ 1, use B = x",
                },
                "elliptical_barriers": {
                    "form": "xᵀPx - c",
                    "design_rule": "Matrix P shapes the barrier to match set geometries",
                    "example": "Match barrier curvature to initial set shape",
                },
            },
            "common_mistakes": {
                "wrong_sign": "Using B = ||x||² for separation (always positive)",
                "no_separation": "Certificate doesn't separate initial from unsafe",
                "boundary_violations": "B = 0 on boundaries instead of proper inequalities",
                "scale_mismatch": "Certificate scale doesn't match system dynamics",
            },
            "verification_interpretation": {
                "sos_success": "Symbolic conditions verified (most reliable)",
                "numerical_success": "Sampling found no counterexamples",
                "boundary_failures": "Check initial/unsafe set constraint violations",
                "optimization_counterexamples": "Found points violating barrier conditions",
            },
        }

    def generate_theory_fix_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive report on barrier certificate theory fixes."""
        report = []
        report.append("🎓 BARRIER CERTIFICATE THEORY FIX REPORT")
        report.append("=" * 55)

        success_rate = results["success_rate"]
        report.append(f"\n🎯 THEORY-CORRECT SUCCESS RATE: {success_rate:.1%}")

        # Determine theory compliance level
        if success_rate >= 0.9:
            level = "🏆 THEORY COMPLIANT"
            status = "Excellent barrier certificate theory implementation"
        elif success_rate >= 0.7:
            level = "✅ MOSTLY COMPLIANT"
            status = "Good theory implementation with minor issues"
        elif success_rate >= 0.5:
            level = "⚠️ PARTIALLY COMPLIANT"
            status = "Some theory violations need addressing"
        else:
            level = "❌ THEORY VIOLATIONS"
            status = "Major barrier certificate theory issues"

        report.append(f"📈 Theory Compliance: {level}")
        report.append(f"📋 Assessment: {status}")

        # Individual test analysis
        report.append("\n📋 THEORY-CORRECT TEST RESULTS:")

        detailed = results["detailed_results"]
        working_certificates = []
        failing_certificates = []

        for test_name, test_result in detailed.items():
            success = test_result.get("overall_success", False)
            theory_correct = test_result.get("theory_correct", False)

            if success and theory_correct:
                emoji = "🟢"
                working_certificates.append(test_name)
            elif theory_correct:
                emoji = "🟡"  # Correctly failed as expected
            else:
                emoji = "🔴"
                failing_certificates.append(test_name)

            report.append(f"   {emoji} {test_name}: {'SUCCESS' if success else 'FAIL'}")

            if "theory_explanation" in test_result:
                report.append(f"      Theory: {test_result['theory_explanation']}")

            if "verification_reason" in test_result and not success:
                reason = test_result["verification_reason"]
                if reason != "N/A":
                    report.append(f"      Reason: {reason}")

        # Success analysis
        if working_certificates:
            report.append("\n✅ WORKING CERTIFICATE PATTERNS:")
            for cert in working_certificates:
                cert_info = detailed[cert]
                report.append(f"   • {cert}: {cert_info.get('certificate', 'N/A')}")
                report.append(
                    f"     Verification: SOS={cert_info.get('sos_passed', False)}, "
                    f"Numerical={cert_info.get('numerical_passed', False)}"
                )

        # Design guidelines
        guidelines = self.create_certificate_design_guidelines()

        report.append("\n🎓 BARRIER CERTIFICATE DESIGN GUIDELINES:")
        report.append("   📌 Fundamental Requirements:")
        for req, desc in guidelines["fundamental_requirements"].items():
            report.append(f"      • {desc}")

        report.append("\n   📐 Design Patterns:")
        for pattern, info in guidelines["design_patterns"].items():
            report.append(f"      • {pattern.replace('_', ' ').title()}: {info['form']}")
            report.append(f"        Rule: {info['design_rule']}")

        # Optimization recommendations
        report.append("\n🎯 OPTIMIZATION RECOMMENDATIONS:")

        if success_rate >= 0.9:
            report.append("   🎉 Excellent barrier certificate theory implementation!")
            report.append("   • Ready for complex system certificate generation")
            report.append("   • Use as template for automatic certificate design")
            report.append("   • Deploy for production barrier certificate validation")
        elif success_rate >= 0.7:
            report.append("   📈 Good theory compliance with room for improvement!")
            report.append("   • Fix remaining theory violation cases")
            report.append("   • Enhance certificate design automation")
            report.append("   • Test with more complex certificate forms")
        else:
            report.append("   🔧 Major theory fixes needed:")
            report.append("   • Review fundamental barrier certificate definitions")
            report.append("   • Fix certificate-set separation logic")
            report.append("   • Implement proper boundary condition checking")
            report.append("   • Test with simple, well-understood cases first")

        # Certificate automation insights
        if success_rate >= 0.7:
            report.append("\n🤖 CERTIFICATE AUTOMATION READINESS:")
            report.append("   • System understands proper barrier certificate theory")
            report.append("   • Ready for automated certificate suggestion algorithms")
            report.append("   • Can guide LLM certificate generation with theory constraints")
            report.append("   • Suitable for iterative certificate refinement")

        return "\n".join(report)


def main():
    """Run barrier certificate theory fix optimization."""
    try:
        print("🎓 Running Barrier Certificate Theory Fix...")
        print("=" * 50)

        fixer = BarrierCertificateTheoryFixer()
        results = fixer.verify_theory_correct_certificates()

        # Generate and display report
        report = fixer.generate_theory_fix_report(results)
        print("\n" + report)

        # Save results
        with open("barrier_certificate_theory_fix_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        print("\n💾 Results saved to barrier_certificate_theory_fix_results.json")

        # Return exit code based on success rate
        success_rate = results["success_rate"]
        return 0 if success_rate >= 0.7 else 1

    except Exception as e:
        print(f"❌ Theory fix optimization failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
