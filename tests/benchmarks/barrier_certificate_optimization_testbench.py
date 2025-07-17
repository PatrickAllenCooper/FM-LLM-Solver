#!/usr/bin/env python3
"""
Barrier Certificate Optimization Testbench

Comprehensive testing framework for optimizing barrier certificate generation
and verification across various certificate types and system complexities.
Focuses on base LLM performance without fine-tuning.
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from unittest.mock import Mock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import load_config
from web_interface.verification_service import VerificationService
from web_interface.certificate_generator import CertificateGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CertificateTestCase:
    """Test case for barrier certificate generation and verification."""

    name: str
    system_description: str
    certificate_type: str  # quadratic, polynomial, rational, linear, custom
    expected_variables: List[str]
    difficulty: str  # easy, medium, hard, extreme
    system_type: str  # continuous, discrete
    domain_bounds: Optional[Dict[str, List[float]]] = None
    known_good_certificate: Optional[str] = None
    expected_verification: Optional[bool] = None


@dataclass
class OptimizationResult:
    """Result from optimization iteration."""

    test_name: str
    success: bool
    certificate_extracted: bool
    certificate_valid: bool
    verification_passed: bool
    processing_time: float
    issues: List[str]
    suggestions: List[str]
    certificate: Optional[str] = None
    verification_details: Optional[Dict] = None


class BarrierCertificateOptimizationTestbench:
    """Comprehensive testbench for optimizing barrier certificate generation."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the optimization testbench."""
        self.config = load_config(config_path or "config.yaml")
        self.verification_service = VerificationService(self.config)
        self.certificate_generator = None  # Will be initialized with mock for base LLM testing
        self.results = []

        # Test cases covering various certificate types and complexities
        self.test_cases = self._create_comprehensive_test_suite()

    def _create_comprehensive_test_suite(self) -> List[CertificateTestCase]:
        """Create comprehensive test suite covering various certificate types."""
        return [
            # === QUADRATIC CERTIFICATES (EASY) ===
            CertificateTestCase(
                name="simple_quadratic_continuous",
                system_description="""System Dynamics: dx/dt = -x, dy/dt = -y
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 4.0""",
                certificate_type="quadratic",
                expected_variables=["x", "y"],
                difficulty="easy",
                system_type="continuous",
                domain_bounds={"x": [-3, 3], "y": [-3, 3]},
                known_good_certificate="x**2 + y**2",
                expected_verification=True,
            ),
            CertificateTestCase(
                name="quadratic_discrete_linear",
                system_description="""System Dynamics: x{k+1} = 0.9*x{k}, y{k+1} = 0.8*y{k}
Initial Set: x**2 + y**2 <= 0.5
Unsafe Set: abs(x) >= 2.0 or abs(y) >= 2.0""",
                certificate_type="quadratic",
                expected_variables=["x", "y"],
                difficulty="easy",
                system_type="discrete",
                domain_bounds={"x": [-3, 3], "y": [-3, 3]},
                known_good_certificate="x**2 + y**2",
            ),
            # === POLYNOMIAL CERTIFICATES (MEDIUM) ===
            CertificateTestCase(
                name="polynomial_continuous_nonlinear",
                system_description="""System Dynamics: dx/dt = -x**3 - y, dy/dt = x - y**3
Initial Set: x**2 + y**2 <= 0.1
Unsafe Set: x >= 1.5""",
                certificate_type="polynomial",
                expected_variables=["x", "y"],
                difficulty="medium",
                system_type="continuous",
                domain_bounds={"x": [-2, 2], "y": [-2, 2]},
                expected_verification=False,  # Complex system, may not have simple polynomial barrier
            ),
            CertificateTestCase(
                name="polynomial_van_der_pol",
                system_description="""System Dynamics: dx/dt = y, dy/dt = -x + (1 - x**2)*y
Initial Set: x**2 + y**2 <= 0.1
Unsafe Set: x**2 + y**2 >= 4.0""",
                certificate_type="polynomial",
                expected_variables=["x", "y"],
                difficulty="medium",
                system_type="continuous",
                domain_bounds={"x": [-3, 3], "y": [-3, 3]},
            ),
            # === MIXED CERTIFICATES (MEDIUM-HARD) ===
            CertificateTestCase(
                name="mixed_quadratic_with_cross_terms",
                system_description="""System Dynamics: dx/dt = -x + y**2, dy/dt = -y + x*y
Initial Set: (x-0.5)**2 + y**2 <= 0.25
Unsafe Set: x <= -1.5 or y >= 2.0""",
                certificate_type="quadratic",
                expected_variables=["x", "y"],
                difficulty="medium",
                system_type="continuous",
                domain_bounds={"x": [-2, 2], "y": [-2, 3]},
            ),
            # === HIGH-DIMENSIONAL SYSTEMS (HARD) ===
            CertificateTestCase(
                name="three_dimensional_linear",
                system_description="""System Dynamics: dx/dt = -x + 0.1*y, dy/dt = -y + 0.1*z, dz/dt = -z
Initial Set: x**2 + y**2 + z**2 <= 0.5
Unsafe Set: x >= 2.0 or y >= 2.0 or z >= 2.0""",
                certificate_type="quadratic",
                expected_variables=["x", "y", "z"],
                difficulty="hard",
                system_type="continuous",
                domain_bounds={"x": [-3, 3], "y": [-3, 3], "z": [-3, 3]},
                known_good_certificate="x**2 + y**2 + z**2",
            ),
            # === COMPLEX CONSTRAINTS (HARD) ===
            CertificateTestCase(
                name="complex_constraint_system",
                system_description="""System Dynamics: dx/dt = -x*y, dy/dt = x**2 - y
Initial Set: (x-1)**2 + (y-1)**2 <= 0.25 and x >= 0 and y >= 0
Unsafe Set: x**2 + y**2 >= 9.0 or x <= -0.5""",
                certificate_type="polynomial",
                expected_variables=["x", "y"],
                difficulty="hard",
                system_type="continuous",
                domain_bounds={"x": [-1, 4], "y": [-1, 4]},
            ),
            # === DISCRETE TIME COMPLEXITY (HARD) ===
            CertificateTestCase(
                name="discrete_nonlinear_dynamics",
                system_description="""System Dynamics: x{k+1} = 0.8*x{k} + 0.1*y{k}**2, y{k+1} = 0.9*y{k} - 0.1*x{k}*y{k}
Initial Set: x**2 + y**2 <= 0.5
Unsafe Set: abs(x) >= 3.0 or abs(y) >= 3.0""",
                certificate_type="quadratic",
                expected_variables=["x", "y"],
                difficulty="hard",
                system_type="discrete",
                domain_bounds={"x": [-4, 4], "y": [-4, 4]},
            ),
            # === EXTREME COMPLEXITY (EXTREME) ===
            CertificateTestCase(
                name="extreme_lorenz_system",
                system_description="""System Dynamics: dx/dt = 10*(y - x), dy/dt = x*(28 - z) - y, dz/dt = x*y - (8/3)*z
Initial Set: x**2 + y**2 + z**2 <= 1.0
Unsafe Set: x**2 + y**2 + z**2 >= 100.0""",
                certificate_type="polynomial",
                expected_variables=["x", "y", "z"],
                difficulty="extreme",
                system_type="continuous",
                domain_bounds={"x": [-15, 15], "y": [-20, 20], "z": [0, 50]},
            ),
        ]

    def _initialize_certificate_generator_for_base_llm(self):
        """Initialize certificate generator optimized for base LLM testing."""
        if self.certificate_generator is None:
            # Create mock certificate generator for testing text processing without heavy ML loading
            mock_config = Mock()
            mock_config.fine_tuning = Mock()
            mock_config.fine_tuning.base_model_name = "base_llm_test"
            mock_config.paths = Mock()
            mock_config.paths.ft_output_dir = "/mock/path"
            mock_config.knowledge_base = Mock()
            mock_config.knowledge_base.barrier_certificate_type = "unified"

            self.certificate_generator = CertificateGenerator.__new__(CertificateGenerator)
            self.certificate_generator.config = mock_config
            self.certificate_generator.models = {}
            self.certificate_generator.knowledge_bases = {}
            self.certificate_generator.embedding_model = None

    def test_certificate_extraction_varieties(self) -> Dict[str, Any]:
        """Test certificate extraction with various formats and types."""
        self._initialize_certificate_generator_for_base_llm()

        test_outputs = [
            # Standard quadratic forms
            {
                "name": "standard_quadratic",
                "output": """BARRIER_CERTIFICATE_START
B(x, y) = 0.7*x**2 + 1.2*y**2 - 0.3
BARRIER_CERTIFICATE_END""",
                "expected_type": "quadratic",
                "should_extract": True,
            },
            # Cross-term quadratic
            {
                "name": "cross_term_quadratic",
                "output": """BARRIER_CERTIFICATE_START
B(x, y) = 0.5*x**2 + 0.8*y**2 + 0.3*x*y - 0.1
BARRIER_CERTIFICATE_END""",
                "expected_type": "quadratic",
                "should_extract": True,
            },
            # Polynomial forms
            {
                "name": "polynomial_degree_4",
                "output": """BARRIER_CERTIFICATE_START
B(x, y) = 0.2*x**4 + 0.3*y**4 + 0.1*x**2*y**2 - 0.05
BARRIER_CERTIFICATE_END""",
                "expected_type": "polynomial",
                "should_extract": True,
            },
            # Three-dimensional
            {
                "name": "three_dimensional",
                "output": """BARRIER_CERTIFICATE_START
B(x, y, z) = 0.4*x**2 + 0.6*y**2 + 0.5*z**2 + 0.1*x*z - 0.2
BARRIER_CERTIFICATE_END""",
                "expected_type": "quadratic",
                "should_extract": True,
            },
            # With rational coefficients
            {
                "name": "rational_coefficients",
                "output": """BARRIER_CERTIFICATE_START
B(x, y) = (1/2)*x**2 + (3/4)*y**2 - 1/10
BARRIER_CERTIFICATE_END""",
                "expected_type": "quadratic",
                "should_extract": True,
            },
            # Complex polynomial
            {
                "name": "complex_polynomial",
                "output": """BARRIER_CERTIFICATE_START
B(x, y) = 0.1*x**6 + 0.2*y**6 + 0.15*x**2*y**4 + 0.12*x**4*y**2 - 0.05*x**2 - 0.03*y**2 + 0.01
BARRIER_CERTIFICATE_END""",
                "expected_type": "polynomial",
                "should_extract": True,
            },
            # Template expressions (should be rejected)
            {
                "name": "template_generic",
                "output": """BARRIER_CERTIFICATE_START
B(x, y) = ax**2 + by**2 + cxy + d
BARRIER_CERTIFICATE_END""",
                "expected_type": "template",
                "should_extract": False,
            },
            # Too simple (should be evaluated carefully)
            {
                "name": "very_simple",
                "output": """BARRIER_CERTIFICATE_START
B(x, y) = x + y
BARRIER_CERTIFICATE_END""",
                "expected_type": "linear",
                "should_extract": True,  # Linear can be valid
            },
        ]

        results = {}
        successful_extractions = 0

        for test in test_outputs:
            try:
                extracted = self.certificate_generator.extract_certificate_from_output(
                    test["output"]
                )
                extraction_successful = extracted is not None
                correctly_handled = extraction_successful == test["should_extract"]

                results[test["name"]] = {
                    "extraction_successful": extraction_successful,
                    "correctly_handled": correctly_handled,
                    "extracted_certificate": extracted,
                    "expected_type": test["expected_type"],
                    "should_extract": test["should_extract"],
                }

                if correctly_handled:
                    successful_extractions += 1

            except Exception as e:
                results[test["name"]] = {
                    "extraction_successful": False,
                    "correctly_handled": False,
                    "error": str(e),
                }

        return {
            "total_tests": len(test_outputs),
            "successful_extractions": successful_extractions,
            "success_rate": successful_extractions / len(test_outputs),
            "detailed_results": results,
        }

    def test_system_parsing_robustness(self) -> Dict[str, Any]:
        """Test system parsing with various formats and complexities."""
        parsing_tests = [
            {
                "name": "standard_continuous",
                "description": """System Dynamics: dx/dt = -x**2 - y, dy/dt = x - y**2
Initial Set: x**2 + y**2 <= 0.5
Unsafe Set: x >= 2.0""",
                "expected_variables": ["x", "y"],
                "expected_dynamics": 2,
            },
            {
                "name": "discrete_with_subscripts",
                "description": """System Dynamics: x_{k+1} = 0.8*x_k + 0.1*y_k, y_{k+1} = 0.9*y_k - 0.1*x_k
Initial Set: x**2 + y**2 <= 1.0
Unsafe Set: abs(x) >= 3.0 or abs(y) >= 3.0""",
                "expected_variables": ["x", "y"],
                "expected_dynamics": 2,
            },
            {
                "name": "three_dimensional",
                "description": """System Dynamics: dx/dt = -x + y*z, dy/dt = -y + x*z, dz/dt = -z + x*y
Initial Set: x**2 + y**2 + z**2 <= 0.25
Unsafe Set: x**2 + y**2 + z**2 >= 4.0""",
                "expected_variables": ["x", "y", "z"],
                "expected_dynamics": 3,
            },
            {
                "name": "complex_constraints",
                "description": """System Dynamics: dx/dt = -x**3 + y, dy/dt = -y**3 - x
Initial Set: (x-1)**2 + (y+0.5)**2 <= 0.1 and x >= 0
Unsafe Set: x**2 + y**2 >= 9.0 or x <= -2.0
Domain: x ∈ [-3, 3], y ∈ [-3, 3]""",
                "expected_variables": ["x", "y"],
                "expected_dynamics": 2,
            },
        ]

        results = {}
        successful_parsing = 0

        for test in parsing_tests:
            try:
                parsed = self.verification_service.parse_system_description(test["description"])
                bounds = self.verification_service.create_sampling_bounds(parsed)

                variables_correct = len(parsed.get("variables", [])) == len(
                    test["expected_variables"]
                )
                dynamics_correct = len(parsed.get("dynamics", [])) == test["expected_dynamics"]
                bounds_created = len(bounds) > 0

                parsing_successful = variables_correct and dynamics_correct and bounds_created

                results[test["name"]] = {
                    "parsing_successful": parsing_successful,
                    "variables_found": parsed.get("variables", []),
                    "dynamics_count": len(parsed.get("dynamics", [])),
                    "bounds_created": bounds_created,
                    "sampling_bounds": bounds,
                    "variables_correct": variables_correct,
                    "dynamics_correct": dynamics_correct,
                }

                if parsing_successful:
                    successful_parsing += 1

            except Exception as e:
                results[test["name"]] = {"parsing_successful": False, "error": str(e)}

        return {
            "total_tests": len(parsing_tests),
            "successful_parsing": successful_parsing,
            "success_rate": successful_parsing / len(parsing_tests),
            "detailed_results": results,
        }

    def test_verification_robustness(self) -> Dict[str, Any]:
        """Test verification with various certificate and system combinations."""
        verification_tests = [
            {
                "name": "simple_stable_system",
                "certificate": "x**2 + y**2",
                "system": """System Dynamics: dx/dt = -x, dy/dt = -y
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 4.0""",
                "should_pass": True,
            },
            {
                "name": "quadratic_with_coefficients",
                "certificate": "0.5*x**2 + 0.8*y**2",
                "system": """System Dynamics: dx/dt = -2*x, dy/dt = -3*y
Initial Set: x**2 + y**2 <= 0.5
Unsafe Set: x**2 + y**2 >= 2.0""",
                "should_pass": True,
            },
            {
                "name": "cross_term_certificate",
                "certificate": "x**2 + y**2 + 0.1*x*y",
                "system": """System Dynamics: dx/dt = -x - 0.1*y, dy/dt = -y - 0.1*x
Initial Set: x**2 + y**2 <= 0.1
Unsafe Set: x**2 + y**2 >= 1.0""",
                "should_pass": None,  # Unknown, let's see
            },
        ]

        results = {}
        successful_verifications = 0

        for test in verification_tests:
            try:
                start_time = time.time()

                result = self.verification_service.verify_certificate(
                    test["certificate"],
                    test["system"],
                    param_overrides={"num_samples_lie": 200, "num_samples_boundary": 100},
                )

                verification_time = time.time() - start_time
                verification_completed = result is not None and isinstance(result, dict)

                if verification_completed:
                    overall_success = result.get("overall_success", False)
                    prediction_correct = (
                        test["should_pass"] is None or overall_success == test["should_pass"]
                    )

                    results[test["name"]] = {
                        "verification_completed": True,
                        "overall_success": overall_success,
                        "verification_time": verification_time,
                        "prediction_correct": prediction_correct,
                        "expected_result": test["should_pass"],
                        "numerical_passed": result.get("numerical_passed", False),
                        "symbolic_passed": result.get("symbolic_passed", False),
                        "sos_passed": result.get("sos_passed", False),
                    }

                    if prediction_correct:
                        successful_verifications += 1
                else:
                    results[test["name"]] = {
                        "verification_completed": False,
                        "error": "Verification returned invalid result",
                    }

            except Exception as e:
                results[test["name"]] = {"verification_completed": False, "error": str(e)}

        return {
            "total_tests": len(verification_tests),
            "successful_verifications": successful_verifications,
            "success_rate": successful_verifications / len(verification_tests),
            "detailed_results": results,
        }

    def run_comprehensive_optimization_tests(self) -> Dict[str, Any]:
        """Run comprehensive optimization tests across all components."""
        print("🔬 Running Comprehensive Barrier Certificate Optimization Tests")
        print("=" * 70)

        results = {}

        # Test 1: Certificate Extraction Varieties
        print("\n📝 Testing Certificate Extraction Varieties...")
        extraction_results = self.test_certificate_extraction_varieties()
        results["extraction"] = extraction_results
        print(f"   Success Rate: {extraction_results['success_rate']:.1%}")

        # Test 2: System Parsing Robustness
        print("\n🔍 Testing System Parsing Robustness...")
        parsing_results = self.test_system_parsing_robustness()
        results["parsing"] = parsing_results
        print(f"   Success Rate: {parsing_results['success_rate']:.1%}")

        # Test 3: Verification Robustness
        print("\n⚖️ Testing Verification Robustness...")
        verification_results = self.test_verification_robustness()
        results["verification"] = verification_results
        print(f"   Success Rate: {verification_results['success_rate']:.1%}")

        # Calculate overall optimization score
        overall_score = (
            extraction_results["success_rate"] * 0.4  # 40% weight on extraction
            + parsing_results["success_rate"] * 0.3  # 30% weight on parsing
            + verification_results["success_rate"] * 0.3  # 30% weight on verification
        )

        results["overall"] = {
            "optimization_score": overall_score,
            "component_scores": {
                "extraction": extraction_results["success_rate"],
                "parsing": parsing_results["success_rate"],
                "verification": verification_results["success_rate"],
            },
        }

        return results

    def generate_optimization_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive optimization report with actionable insights."""
        report = []
        report.append("📊 BARRIER CERTIFICATE OPTIMIZATION REPORT")
        report.append("=" * 50)

        overall_score = results["overall"]["optimization_score"]
        report.append(f"\n🎯 OVERALL OPTIMIZATION SCORE: {overall_score:.1%}")

        # Determine optimization level
        if overall_score >= 0.9:
            level = "🚀 HIGHLY OPTIMIZED"
            status = "Ready for production deployment"
        elif overall_score >= 0.75:
            level = "⚡ WELL OPTIMIZED"
            status = "Ready for advanced testing"
        elif overall_score >= 0.6:
            level = "⚠️ PARTIALLY OPTIMIZED"
            status = "Needs targeted improvements"
        else:
            level = "🔧 NEEDS OPTIMIZATION"
            status = "Requires significant improvements"

        report.append(f"📈 Optimization Level: {level}")
        report.append(f"📋 Status: {status}")

        # Component breakdown
        report.append("\n📋 COMPONENT PERFORMANCE:")
        components = results["overall"]["component_scores"]
        for component, score in components.items():
            emoji = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "❌"
            report.append(f"   {emoji} {component.capitalize()}: {score:.1%}")

        # Detailed insights
        report.append("\n💡 OPTIMIZATION INSIGHTS:")

        # Extraction insights
        extraction = results["extraction"]
        if extraction["success_rate"] < 0.8:
            report.append("   🔧 Certificate Extraction needs improvement:")
            report.append("      • Enhance template detection patterns")
            report.append("      • Improve parsing of complex mathematical expressions")
            report.append("      • Add support for more certificate varieties")
        else:
            report.append("   ✅ Certificate Extraction performing well")

        # Parsing insights
        parsing = results["parsing"]
        if parsing["success_rate"] < 0.8:
            report.append("   🔧 System Parsing needs improvement:")
            report.append("      • Enhance dynamic system pattern recognition")
            report.append("      • Improve constraint parsing robustness")
            report.append("      • Add better bounds detection")
        else:
            report.append("   ✅ System Parsing performing well")

        # Verification insights
        verification = results["verification"]
        if verification["success_rate"] < 0.8:
            report.append("   🔧 Verification System needs improvement:")
            report.append("      • Optimize numerical verification parameters")
            report.append("      • Enhance SOS solver configuration")
            report.append("      • Improve timeout and error handling")
        else:
            report.append("   ✅ Verification System performing well")

        # Recommendations
        report.append("\n🎯 OPTIMIZATION RECOMMENDATIONS:")
        if overall_score >= 0.9:
            report.append("   🎉 System is highly optimized!")
            report.append("   • Focus on edge case handling")
            report.append("   • Consider performance optimizations")
            report.append("   • Deploy for production use")
        elif overall_score >= 0.75:
            report.append("   📈 System is well optimized!")
            report.append("   • Address remaining component weaknesses")
            report.append("   • Test with more complex systems")
            report.append("   • Prepare for production deployment")
        else:
            report.append("   🔧 Priority optimizations needed:")
            lowest_component = min(components.items(), key=lambda x: x[1])
            report.append(f"      • Focus on {lowest_component[0]} ({lowest_component[1]:.1%})")
            report.append("      • Run targeted optimization iterations")
            report.append("      • Test improvements systematically")

        return "\n".join(report)


def main():
    """Run the barrier certificate optimization testbench."""
    try:
        testbench = BarrierCertificateOptimizationTestbench()
        results = testbench.run_comprehensive_optimization_tests()

        # Generate and display report
        report = testbench.generate_optimization_report(results)
        print("\n" + report)

        # Save results for iteration tracking
        with open("optimization_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        print("\n💾 Results saved to optimization_results.json")

        # Return appropriate exit code
        overall_score = results["overall"]["optimization_score"]
        return 0 if overall_score >= 0.75 else 1

    except Exception as e:
        print(f"❌ Optimization testbench failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
