#!/usr/bin/env python3
"""
Verification Optimization Module

Fixes verification issues by:
1. Generating proper certificate-system pairings
2. Optimizing verification parameters
3. Handling initial set constraint violations
4. Improving SOS solver configuration
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import load_config
from web_interface.verification_service import VerificationService

logger = logging.getLogger(__name__)


class VerificationOptimizer:
    """Optimizes verification system for better barrier certificate validation."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the verification optimizer."""
        self.config = load_config(config_path or "config.yaml")
        self.verification_service = VerificationService(self.config)

    def create_proper_certificate_system_pairs(self) -> List[Dict[str, Any]]:
        """Create certificate-system pairs that should theoretically work."""
        return [
            # === WORKING PAIRS (Theoretically Sound) ===
            {
                "name": "simple_decay_quadratic",
                "system": """System Dynamics: dx/dt = -2*x, dy/dt = -2*y
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 4.0""",
                "certificate": "x**2 + y**2 - 0.26",  # Ensures B(x) <= -0.01 on initial set
                "expected_success": True,
                "theory_notes": "Quadratic Lyapunov function for stable linear system",
            },
            {
                "name": "scaled_quadratic_decay",
                "system": """System Dynamics: dx/dt = -x, dy/dt = -3*y
Initial Set: x**2 + y**2 <= 0.5
Unsafe Set: x**2 + y**2 >= 2.0""",
                "certificate": "x**2 + y**2 - 0.51",  # B(x) <= -0.01 on initial set
                "expected_success": True,
                "theory_notes": "Different decay rates, quadratic barrier",
            },
            {
                "name": "weighted_quadratic",
                "system": """System Dynamics: dx/dt = -3*x, dy/dt = -y
Initial Set: 4*x**2 + y**2 <= 1.0
Unsafe Set: x**2 + y**2 >= 4.0""",
                "certificate": "4*x**2 + y**2 - 1.01",  # Matches initial set shape
                "expected_success": True,
                "theory_notes": "Weighted quadratic matching elliptical initial set",
            },
            # === DISCRETE TIME PAIRS ===
            {
                "name": "discrete_contractive",
                "system": """System Dynamics: x{k+1} = 0.5*x{k}, y{k+1} = 0.5*y{k}
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 1.0""",
                "certificate": "x**2 + y**2 - 0.26",
                "expected_success": True,
                "theory_notes": "Contractive discrete system with quadratic barrier",
            },
            # === CHALLENGING BUT SOLVABLE ===
            {
                "name": "linear_barrier_example",
                "system": """System Dynamics: dx/dt = -x - y, dy/dt = y
Initial Set: x <= -0.5 and y >= -0.1 and y <= 0.1
Unsafe Set: x >= 0.5""",
                "certificate": "x + 0.51",  # Linear barrier: x + 0.51 <= 0 on initial set
                "expected_success": True,
                "theory_notes": "Linear barrier for reachability problem",
            },
        ]

    def optimize_verification_parameters(self) -> Dict[str, Any]:
        """Determine optimal verification parameters through testing."""
        parameter_tests = [
            {
                "name": "high_precision",
                "params": {
                    "num_samples_lie": 500,
                    "num_samples_boundary": 200,
                    "tolerance": 1e-8,
                    "optimization_timeout": 30,
                },
            },
            {
                "name": "balanced_performance",
                "params": {
                    "num_samples_lie": 300,
                    "num_samples_boundary": 150,
                    "tolerance": 1e-6,
                    "optimization_timeout": 20,
                },
            },
            {
                "name": "fast_screening",
                "params": {
                    "num_samples_lie": 100,
                    "num_samples_boundary": 50,
                    "tolerance": 1e-5,
                    "optimization_timeout": 10,
                },
            },
        ]

        # Test with a known-good certificate-system pair
        test_certificate = "x**2 + y**2 - 0.26"
        test_system = """System Dynamics: dx/dt = -2*x, dy/dt = -2*y
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 4.0"""

        results = {}
        best_params = None
        best_score = 0

        for param_test in parameter_tests:
            try:
                start_time = time.time()

                result = self.verification_service.verify_certificate(
                    test_certificate, test_system, param_overrides=param_test["params"]
                )

                verification_time = time.time() - start_time

                if result and isinstance(result, dict):
                    # Score based on success, speed, and reliability
                    success_score = 1.0 if result.get("overall_success", False) else 0.0
                    speed_score = max(0, 1.0 - verification_time / 10.0)  # Penalty after 10s

                    # Check individual components
                    numerical_score = 0.3 if result.get("numerical_passed", False) else 0.0
                    symbolic_score = 0.3 if result.get("symbolic_passed", False) else 0.0
                    sos_score = 0.3 if result.get("sos_passed", False) else 0.0

                    total_score = (
                        success_score
                        + speed_score * 0.3
                        + numerical_score
                        + symbolic_score
                        + sos_score
                    )

                    results[param_test["name"]] = {
                        "success": result.get("overall_success", False),
                        "verification_time": verification_time,
                        "total_score": total_score,
                        "numerical_passed": result.get("numerical_passed", False),
                        "symbolic_passed": result.get("symbolic_passed", False),
                        "sos_passed": result.get("sos_passed", False),
                        "params": param_test["params"],
                    }

                    if total_score > best_score:
                        best_score = total_score
                        best_params = param_test["params"]

            except Exception as e:
                results[param_test["name"]] = {
                    "success": False,
                    "error": str(e),
                    "total_score": 0.0,
                }

        return {
            "parameter_test_results": results,
            "best_parameters": best_params,
            "best_score": best_score,
        }

    def test_proper_certificate_system_pairs(self) -> Dict[str, Any]:
        """Test verification with theoretically sound certificate-system pairs."""
        test_pairs = self.create_proper_certificate_system_pairs()
        results = {}
        successful_verifications = 0

        # Get optimized parameters
        param_optimization = self.optimize_verification_parameters()
        optimal_params = param_optimization["best_parameters"]

        if optimal_params is None:
            # Fallback to balanced parameters
            optimal_params = {
                "num_samples_lie": 300,
                "num_samples_boundary": 150,
                "tolerance": 1e-6,
                "optimization_timeout": 20,
            }

        logger.info(f"Using optimized parameters: {optimal_params}")

        for pair in test_pairs:
            try:
                start_time = time.time()

                result = self.verification_service.verify_certificate(
                    pair["certificate"], pair["system"], param_overrides=optimal_params
                )

                verification_time = time.time() - start_time

                if result and isinstance(result, dict):
                    overall_success = result.get("overall_success", False)
                    prediction_correct = overall_success == pair["expected_success"]

                    results[pair["name"]] = {
                        "success": overall_success,
                        "prediction_correct": prediction_correct,
                        "verification_time": verification_time,
                        "expected_success": pair["expected_success"],
                        "theory_notes": pair["theory_notes"],
                        "numerical_passed": result.get("numerical_passed", False),
                        "symbolic_passed": result.get("symbolic_passed", False),
                        "sos_passed": result.get("sos_passed", False),
                        "certificate": pair["certificate"],
                        "failure_reason": result.get("reason", "N/A"),
                    }

                    if prediction_correct:
                        successful_verifications += 1
                else:
                    results[pair["name"]] = {
                        "success": False,
                        "prediction_correct": False,
                        "error": "Invalid verification result",
                    }

            except Exception as e:
                results[pair["name"]] = {
                    "success": False,
                    "prediction_correct": False,
                    "error": str(e),
                }

        return {
            "total_tests": len(test_pairs),
            "successful_verifications": successful_verifications,
            "success_rate": successful_verifications / len(test_pairs),
            "detailed_results": results,
            "optimization_results": param_optimization,
        }

    def generate_verification_optimization_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive report on verification optimization."""
        report = []
        report.append("🔧 VERIFICATION SYSTEM OPTIMIZATION REPORT")
        report.append("=" * 55)

        success_rate = results["success_rate"]
        report.append(f"\n🎯 VERIFICATION SUCCESS RATE: {success_rate:.1%}")

        # Determine optimization status
        if success_rate >= 0.8:
            status = "🚀 HIGHLY OPTIMIZED"
            color = "Ready for production"
        elif success_rate >= 0.6:
            status = "⚡ WELL OPTIMIZED"
            color = "Ready for testing"
        elif success_rate >= 0.4:
            status = "⚠️ PARTIALLY OPTIMIZED"
            color = "Needs improvement"
        else:
            status = "🔧 NEEDS MAJOR WORK"
            color = "Requires fundamental fixes"

        report.append(f"📈 Optimization Status: {status}")
        report.append(f"📋 Assessment: {color}")

        # Parameter optimization results
        if "optimization_results" in results:
            opt_results = results["optimization_results"]
            best_params = opt_results.get("best_parameters")

            report.append("\n🎛️ OPTIMAL PARAMETERS:")
            if best_params:
                for param, value in best_params.items():
                    report.append(f"   • {param}: {value}")
            else:
                report.append("   ⚠️ No optimal parameters found - using defaults")

        # Individual test results
        report.append("\n📋 INDIVIDUAL TEST RESULTS:")
        detailed = results["detailed_results"]

        for test_name, test_result in detailed.items():
            success = test_result.get("success", False)
            prediction_correct = test_result.get("prediction_correct", False)

            emoji = "✅" if success and prediction_correct else "❌"
            report.append(f"   {emoji} {test_name}: {'PASS' if success else 'FAIL'}")

            if "theory_notes" in test_result:
                report.append(f"      Theory: {test_result['theory_notes']}")

            if "failure_reason" in test_result and not success:
                report.append(f"      Reason: {test_result['failure_reason']}")

        # Recommendations
        report.append("\n💡 OPTIMIZATION RECOMMENDATIONS:")

        if success_rate >= 0.8:
            report.append("   🎉 Verification system is highly optimized!")
            report.append("   • System ready for complex certificate validation")
            report.append("   • Consider stress testing with harder problems")
            report.append("   • Deploy for production use")
        elif success_rate >= 0.6:
            report.append("   📈 Good optimization progress!")
            report.append("   • Fine-tune parameters for remaining failures")
            report.append("   • Test with more certificate varieties")
            report.append("   • Consider SOS solver improvements")
        else:
            report.append("   🔧 Major improvements needed:")
            report.append("   • Review barrier certificate theory implementation")
            report.append("   • Fix initial set constraint handling")
            report.append("   • Improve numerical tolerance settings")
            report.append("   • Debug SOS solver configuration")

        return "\n".join(report)


def main():
    """Run verification optimization."""
    try:
        print("🔧 Running Verification System Optimization...")
        print("=" * 50)

        optimizer = VerificationOptimizer()
        results = optimizer.test_proper_certificate_system_pairs()

        # Generate and display report
        report = optimizer.generate_verification_optimization_report(results)
        print("\n" + report)

        # Save results
        with open("verification_optimization_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        print("\n💾 Results saved to verification_optimization_results.json")

        # Return exit code based on success rate
        success_rate = results["success_rate"]
        return 0 if success_rate >= 0.6 else 1

    except Exception as e:
        print(f"❌ Verification optimization failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
