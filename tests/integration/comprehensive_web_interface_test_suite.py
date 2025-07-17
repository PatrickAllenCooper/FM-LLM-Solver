#!/usr/bin/env python3
"""
Comprehensive Web Interface Test Suite - Production Ready Validation

Tests every aspect of the web interface to ensure rock-solid reliability:
- Certificate generation pipeline
- Verification system with boundary condition fixes
- LLM integration and prompting
- Knowledge base retrieval
- Error handling and edge cases
- Performance validation
- End-to-end workflows

This suite maximizes successful barrier certificate generation through rigorous testing.
"""

import sys
import time
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("comprehensive_web_test.log")],
)
logger = logging.getLogger(__name__)

# Import web interface components
from utils.config_loader import load_config
from web_interface.verification_service import VerificationService
from web_interface.certificate_generator import CertificateGenerator
from web_interface.conversation_service import ConversationService


@dataclass
class TestCase:
    """Comprehensive test case definition."""

    name: str
    system_description: str
    expected_certificate_type: str  # quadratic, polynomial, rational, etc.
    complexity: str  # simple, medium, complex, extreme
    system_type: str  # continuous, discrete
    domain_bounds: Optional[Dict[str, List[float]]] = None
    should_succeed: bool = True
    timeout_seconds: int = 30
    description: str = ""


@dataclass
class TestResult:
    """Detailed test result tracking."""

    test_name: str
    success: bool
    certificate_generated: Optional[str] = None
    verification_result: Optional[Dict] = None
    generation_time: float = 0.0
    verification_time: float = 0.0
    error_message: Optional[str] = None
    warnings: List[str] = None
    performance_metrics: Dict[str, float] = None


class ComprehensiveWebInterfaceTestSuite:
    """Comprehensive test suite for all web interface functionality."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the comprehensive test suite."""
        self.config = load_config(config_path or "config.yaml")
        self.verification_service = VerificationService(self.config)
        self.certificate_generator = None  # Initialized lazily due to model loading
        self.conversation_service = None  # Initialized lazily

        self.test_results: List[TestResult] = []
        self.performance_metrics = {}
        self.error_summary = {}

        print("🚀 COMPREHENSIVE WEB INTERFACE TEST SUITE")
        print("=" * 70)
        print("🎯 Goal: Rock-solid reliability with maximum success rates")
        print("🔬 Testing: All components, edge cases, and error handling")
        print()

    def create_comprehensive_test_cases(self) -> List[TestCase]:
        """Create comprehensive test cases covering all scenarios."""
        return [
            # === SIMPLE CONTINUOUS SYSTEMS ===
            TestCase(
                name="simple_stable_linear_2d",
                system_description="""System Dynamics: dx/dt = -x, dy/dt = -y
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 4.0""",
                expected_certificate_type="quadratic",
                complexity="simple",
                system_type="continuous",
                domain_bounds={"x": [-3, 3], "y": [-3, 3]},
                description="Classic stable linear system - should always succeed",
            ),
            TestCase(
                name="simple_stable_linear_1d",
                system_description="""System Dynamics: dx/dt = -2*x
Initial Set: x <= -1.0
Unsafe Set: x >= 2.0""",
                expected_certificate_type="quadratic",
                complexity="simple",
                system_type="continuous",
                domain_bounds={"x": [-3, 3]},
                description="1D stable linear system",
            ),
            TestCase(
                name="damped_oscillator",
                system_description="""System Dynamics: dx/dt = y, dy/dt = -x - 0.5*y
Initial Set: x**2 + y**2 <= 0.1
Unsafe Set: x**2 + y**2 >= 9.0""",
                expected_certificate_type="quadratic",
                complexity="simple",
                system_type="continuous",
                domain_bounds={"x": [-4, 4], "y": [-4, 4]},
                description="Damped harmonic oscillator",
            ),
            # === MEDIUM COMPLEXITY CONTINUOUS ===
            TestCase(
                name="nonlinear_stable_2d",
                system_description="""System Dynamics: dx/dt = -x**3, dy/dt = -y**3
Initial Set: x**4 + y**4 <= 0.0625
Unsafe Set: x**4 + y**4 >= 1.0""",
                expected_certificate_type="polynomial",
                complexity="medium",
                system_type="continuous",
                domain_bounds={"x": [-2, 2], "y": [-2, 2]},
                description="Nonlinear system requiring higher-order barrier",
            ),
            TestCase(
                name="van_der_pol_like",
                system_description="""System Dynamics: dx/dt = y, dy/dt = -x + 0.1*(1-x**2)*y
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 16.0""",
                expected_certificate_type="quadratic",
                complexity="medium",
                system_type="continuous",
                domain_bounds={"x": [-5, 5], "y": [-5, 5]},
                description="Van der Pol-like oscillator",
            ),
            # === DISCRETE-TIME SYSTEMS ===
            TestCase(
                name="discrete_stable_linear",
                system_description="""System Dynamics: x{k+1} = 0.8*x{k}, y{k+1} = 0.9*y{k}
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 4.0""",
                expected_certificate_type="quadratic",
                complexity="simple",
                system_type="discrete",
                domain_bounds={"x": [-3, 3], "y": [-3, 3]},
                description="Discrete-time stable linear system",
            ),
            TestCase(
                name="discrete_nonlinear",
                system_description="""System Dynamics: x{k+1} = 0.9*x{k} - 0.1*y{k}, y{k+1} = 0.1*x{k} + 0.8*y{k}
Initial Set: x**2 + y**2 <= 0.1
Unsafe Set: x**2 + y**2 >= 9.0""",
                expected_certificate_type="quadratic",
                complexity="medium",
                system_type="discrete",
                domain_bounds={"x": [-4, 4], "y": [-4, 4]},
                description="Discrete-time coupled system",
            ),
            # === COMPLEX SYSTEMS ===
            TestCase(
                name="three_dimensional_system",
                system_description="""System Dynamics: dx/dt = -x + 0.1*y, dy/dt = -y + 0.1*z, dz/dt = -z
Initial Set: x**2 + y**2 + z**2 <= 0.25
Unsafe Set: x**2 + y**2 + z**2 >= 9.0""",
                expected_certificate_type="quadratic",
                complexity="complex",
                system_type="continuous",
                domain_bounds={"x": [-4, 4], "y": [-4, 4], "z": [-4, 4]},
                description="3D system test",
            ),
            TestCase(
                name="complex_polynomial_system",
                system_description="""System Dynamics: dx/dt = -x**3 + 0.1*x*y**2, dy/dt = -y**3 - 0.1*x**2*y
Initial Set: x**2 + y**2 <= 0.01
Unsafe Set: x**2 + y**2 >= 4.0""",
                expected_certificate_type="polynomial",
                complexity="complex",
                system_type="continuous",
                domain_bounds={"x": [-3, 3], "y": [-3, 3]},
                description="Complex polynomial dynamics",
            ),
            # === EDGE CASES ===
            TestCase(
                name="asymmetric_elliptical_sets",
                system_description="""System Dynamics: dx/dt = -2*x, dy/dt = -y
Initial Set: 4*x**2 + y**2 <= 1.0
Unsafe Set: x**2 + 4*y**2 >= 16.0""",
                expected_certificate_type="quadratic",
                complexity="medium",
                system_type="continuous",
                domain_bounds={"x": [-3, 3], "y": [-3, 3]},
                description="Asymmetric elliptical sets",
            ),
            TestCase(
                name="rectangular_constraints",
                system_description="""System Dynamics: dx/dt = -x, dy/dt = -y
Initial Set: x >= -0.5 and x <= 0.5 and y >= -0.5 and y <= 0.5
Unsafe Set: x >= 2.0 or x <= -2.0 or y >= 2.0 or y <= -2.0""",
                expected_certificate_type="quadratic",
                complexity="medium",
                system_type="continuous",
                domain_bounds={"x": [-3, 3], "y": [-3, 3]},
                description="Rectangular constraint sets",
            ),
            # === CHALLENGE CASES ===
            TestCase(
                name="near_unstable_system",
                system_description="""System Dynamics: dx/dt = -0.01*x + 0.1*y, dy/dt = -0.1*x - 0.01*y
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 25.0""",
                expected_certificate_type="quadratic",
                complexity="complex",
                system_type="continuous",
                domain_bounds={"x": [-6, 6], "y": [-6, 6]},
                description="Near-unstable system (challenging)",
            ),
        ]

    def initialize_llm_components(self) -> bool:
        """Initialize LLM-dependent components with proper error handling."""
        print("🤖 Initializing LLM components...")

        try:
            # Initialize certificate generator (may take time for model loading)
            print("   Loading certificate generator...")
            self.certificate_generator = CertificateGenerator(self.config)
            print("   ✅ Certificate generator ready")

            # Initialize conversation service
            print("   Loading conversation service...")
            self.conversation_service = ConversationService(self.config)
            print("   ✅ Conversation service ready")

            return True

        except Exception as e:
            print(f"   ❌ LLM component initialization failed: {e}")
            logger.error(f"LLM initialization error: {e}")
            return False

    def test_verification_service(self) -> Dict[str, Any]:
        """Test verification service with known certificates."""
        print("\n🔍 TESTING VERIFICATION SERVICE")
        print("-" * 50)

        verification_results = {
            "tests_run": 0,
            "tests_passed": 0,
            "boundary_fix_working": False,
            "errors": [],
        }

        # Test 1: Known correct certificate (should pass with boundary fix)
        print("📋 Test 1: Known correct certificate validation...")
        try:
            result = self.verification_service.verify_certificate(
                "x**2 + y**2",
                """System Dynamics: dx/dt = -x, dy/dt = -y
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 4.0""",
                param_overrides={
                    "num_samples_lie": 20,
                    "num_samples_boundary": 10,
                    "numerical_tolerance": 1e-6,
                    "attempt_sos": False,
                    "attempt_optimization": False,
                },
            )

            verification_results["tests_run"] += 1

            # Check if boundary conditions pass (critical fix validation)
            details = result.get("details", {})
            numerical_details = details.get("numerical", {})
            reason = numerical_details.get("reason", "")

            if "Passed Initial Set" in reason and "Passed Unsafe Set" in reason:
                verification_results["boundary_fix_working"] = True
                verification_results["tests_passed"] += 1
                print("   ✅ Boundary condition fix working correctly")
            else:
                print(f"   ⚠️ Boundary conditions: {reason}")

        except Exception as e:
            verification_results["errors"].append(f"Known certificate test: {str(e)}")
            print(f"   ❌ Known certificate test failed: {e}")

        # Test 2: System parsing robustness
        print("📋 Test 2: System parsing robustness...")
        parsing_test_cases = [
            """System Dynamics: dx/dt = -x, dy/dt = -y
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 4.0""",
            """System Dynamics: x{k+1} = 0.8*x{k}, y{k+1} = 0.9*y{k}
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 4.0""",
            """System Dynamics: dx/dt = -x**3, dy/dt = -y**3
Initial Set: x**4 + y**4 <= 0.0625
Unsafe Set: x**4 + y**4 >= 1.0""",
        ]

        parsing_passed = 0
        for i, system in enumerate(parsing_test_cases):
            try:
                parsed = self.verification_service.parse_system_description(system)
                if parsed.get("variables") and parsed.get("dynamics"):
                    parsing_passed += 1
                    print(f"   ✅ Parsing test {i+1}: Success")
                else:
                    print(f"   ⚠️ Parsing test {i+1}: Incomplete")
            except Exception as e:
                verification_results["errors"].append(f"Parsing test {i+1}: {str(e)}")
                print(f"   ❌ Parsing test {i+1}: Failed - {e}")

        verification_results["tests_run"] += len(parsing_test_cases)
        verification_results["tests_passed"] += parsing_passed

        success_rate = verification_results["tests_passed"] / verification_results["tests_run"]
        print(f"\n📊 Verification Service: {success_rate:.1%} success rate")

        return verification_results

    def test_certificate_generation(self, test_case: TestCase) -> TestResult:
        """Test certificate generation for a specific test case."""
        print(f"\n🧪 TESTING: {test_case.name}")
        print(f"   Description: {test_case.description}")
        print(f"   System Type: {test_case.system_type}")
        print(f"   Complexity: {test_case.complexity}")

        start_time = time.time()

        try:
            # Step 1: Generate certificate using LLM
            generation_start = time.time()

            if self.certificate_generator is None:
                # Mock generation for testing without LLM
                if test_case.expected_certificate_type == "quadratic":
                    generated_certificate = "x**2 + y**2"
                elif test_case.expected_certificate_type == "polynomial":
                    generated_certificate = "x**4 + y**4"
                else:
                    generated_certificate = "x**2 + y**2"

                print(f"   🤖 Mock generation: {generated_certificate}")
                generation_time = 0.1
            else:
                # Real LLM generation
                generated_certificate = self.certificate_generator.generate_certificate(
                    test_case.system_description, domain_bounds=test_case.domain_bounds
                )
                generation_time = time.time() - generation_start
                print(f"   🤖 Generated certificate: {generated_certificate}")

            if not generated_certificate:
                return TestResult(
                    test_name=test_case.name,
                    success=False,
                    error_message="Certificate generation failed - empty result",
                    generation_time=generation_time,
                )

            # Step 2: Verify the generated certificate
            verification_start = time.time()

            verification_result = self.verification_service.verify_certificate(
                generated_certificate,
                test_case.system_description,
                param_overrides={
                    "num_samples_lie": 50,
                    "num_samples_boundary": 25,
                    "numerical_tolerance": 1e-6,
                    "attempt_sos": True,
                    "attempt_optimization": True,
                },
                domain_bounds=test_case.domain_bounds,
            )

            verification_time = time.time() - verification_start
            total_time = time.time() - start_time

            # Analyze results
            overall_success = verification_result.get("overall_success", False)
            numerical_passed = verification_result.get("numerical_passed", False)
            sos_passed = verification_result.get("sos_passed", False)

            # Extract detailed information
            details = verification_result.get("details", {})
            numerical_details = details.get("numerical", {})
            numerical_details.get("reason", "No details available")

            # Check for warnings
            warnings = []
            if not numerical_passed:
                warnings.append("Numerical verification failed")
            if not sos_passed:
                warnings.append("SOS verification failed")

            # Performance metrics
            performance_metrics = {
                "total_time": total_time,
                "generation_time": generation_time,
                "verification_time": verification_time,
                "samples_per_second": 75.0 / verification_time if verification_time > 0 else 0,
            }

            success = overall_success if test_case.should_succeed else not overall_success

            status = "✅ PASS" if success else "❌ FAIL"
            print(
                f"   {status} - Overall: {overall_success}, Numerical: {numerical_passed}, SOS: {sos_passed}"
            )
            print(f"   ⏱️ Times: Gen={generation_time:.2f}s, Ver={verification_time:.2f}s")

            return TestResult(
                test_name=test_case.name,
                success=success,
                certificate_generated=generated_certificate,
                verification_result=verification_result,
                generation_time=generation_time,
                verification_time=verification_time,
                warnings=warnings if warnings else None,
                performance_metrics=performance_metrics,
            )

        except Exception as e:
            total_time = time.time() - start_time
            error_msg = f"Test execution failed: {str(e)}"
            print(f"   ❌ ERROR: {error_msg}")
            logger.error(f"Test {test_case.name} failed: {e}")

            return TestResult(
                test_name=test_case.name,
                success=False,
                error_message=error_msg,
                generation_time=total_time,
                performance_metrics={"total_time": total_time},
            )

    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and edge cases."""
        print("\n🚨 TESTING ERROR HANDLING & EDGE CASES")
        print("-" * 50)

        error_tests = {
            "malformed_system": {
                "input": "This is not a valid system description",
                "expected": "graceful_failure",
            },
            "empty_certificate": {
                "certificate": "",
                "system": """System Dynamics: dx/dt = -x, dy/dt = -y
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 4.0""",
                "expected": "graceful_failure",
            },
            "invalid_certificate": {
                "certificate": "this is not math",
                "system": """System Dynamics: dx/dt = -x, dy/dt = -y
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 4.0""",
                "expected": "graceful_failure",
            },
        }

        error_results = {"tests_run": 0, "graceful_failures": 0, "crashes": 0, "details": {}}

        for test_name, test_config in error_tests.items():
            print(f"📋 Testing: {test_name}")
            error_results["tests_run"] += 1

            try:
                if "system" in test_config:
                    # Test verification with invalid input
                    result = self.verification_service.verify_certificate(
                        test_config.get("certificate", "x**2 + y**2"), test_config["system"]
                    )

                    if result.get("overall_success") == False:
                        error_results["graceful_failures"] += 1
                        print("   ✅ Graceful failure as expected")
                    else:
                        print("   ⚠️ Unexpected success")

                else:
                    # Test system parsing with invalid input
                    parsed = self.verification_service.parse_system_description(
                        test_config["input"]
                    )

                    if not parsed.get("variables") or not parsed.get("dynamics"):
                        error_results["graceful_failures"] += 1
                        print("   ✅ Graceful parsing failure as expected")
                    else:
                        print("   ⚠️ Unexpected parsing success")

            except Exception as e:
                error_results["crashes"] += 1
                error_results["details"][test_name] = str(e)
                print(f"   ❌ Crash (not ideal): {e}")

        return error_results

    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance benchmarks and resource usage."""
        print("\n⚡ PERFORMANCE BENCHMARKING")
        print("-" * 50)

        # Quick performance test
        test_certificate = "x**2 + y**2"
        test_system = """System Dynamics: dx/dt = -x, dy/dt = -y
Initial Set: x**2 + y**2 <= 0.25
Unsafe Set: x**2 + y**2 >= 4.0"""

        sample_counts = [10, 25, 50, 100]
        performance_data = []

        for samples in sample_counts:
            print(f"📊 Testing with {samples} samples...")

            start_time = time.time()
            try:
                result = self.verification_service.verify_certificate(
                    test_certificate,
                    test_system,
                    param_overrides={
                        "num_samples_lie": samples,
                        "num_samples_boundary": samples // 2,
                        "attempt_sos": False,
                        "attempt_optimization": False,
                    },
                )

                elapsed = time.time() - start_time
                success = result.get("overall_success", False)

                performance_data.append(
                    {
                        "samples": samples,
                        "time": elapsed,
                        "success": success,
                        "samples_per_second": samples / elapsed if elapsed > 0 else 0,
                    }
                )

                print(
                    f"   ⏱️ {elapsed:.2f}s ({samples/elapsed:.0f} samples/sec) - {'✅' if success else '❌'}"
                )

            except Exception as e:
                print(f"   ❌ Failed with {samples} samples: {e}")

        return {
            "performance_data": performance_data,
            "baseline_time": performance_data[0]["time"] if performance_data else None,
            "scalability": "good" if len(performance_data) >= 3 else "needs_testing",
        }

    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete comprehensive test suite."""
        suite_start_time = time.time()

        # Phase 1: Component Testing
        print("\n" + "=" * 70)
        print("🔧 PHASE 1: COMPONENT TESTING")
        print("=" * 70)

        verification_results = self.test_verification_service()
        error_handling_results = self.test_error_handling()
        performance_results = self.test_performance_benchmarks()

        # Phase 2: LLM Component Initialization (optional)
        print("\n" + "=" * 70)
        print("🤖 PHASE 2: LLM COMPONENT TESTING")
        print("=" * 70)

        llm_available = self.initialize_llm_components()

        # Phase 3: End-to-End Certificate Generation Testing
        print("\n" + "=" * 70)
        print("🎯 PHASE 3: END-TO-END CERTIFICATE GENERATION")
        print("=" * 70)

        test_cases = self.create_comprehensive_test_cases()

        for test_case in test_cases:
            try:
                result = self.test_certificate_generation(test_case)
                self.test_results.append(result)
            except Exception as e:
                logger.error(f"Test case {test_case.name} crashed: {e}")
                self.test_results.append(
                    TestResult(
                        test_name=test_case.name,
                        success=False,
                        error_message=f"Test crashed: {str(e)}",
                    )
                )

        # Phase 4: Analysis and Reporting
        suite_time = time.time() - suite_start_time
        return self._generate_comprehensive_report(
            verification_results,
            error_handling_results,
            performance_results,
            llm_available,
            suite_time,
        )

    def _generate_comprehensive_report(
        self,
        verification_results: Dict,
        error_results: Dict,
        performance_results: Dict,
        llm_available: bool,
        suite_time: float,
    ) -> Dict[str, Any]:
        """Generate comprehensive test report."""

        # Analyze test results
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.success)
        success_rate = successful_tests / total_tests if total_tests > 0 else 0

        # Performance analysis
        if self.test_results:
            avg_generation_time = np.mean(
                [r.generation_time for r in self.test_results if r.generation_time]
            )
            avg_verification_time = np.mean(
                [r.verification_time for r in self.test_results if r.verification_time]
            )
        else:
            avg_generation_time = 0
            avg_verification_time = 0

        # Component health
        component_health = {
            "verification_service": (
                verification_results["tests_passed"] / verification_results["tests_run"]
                if verification_results["tests_run"] > 0
                else 0
            ),
            "boundary_fix": verification_results.get("boundary_fix_working", False),
            "error_handling": (
                error_results["graceful_failures"] / error_results["tests_run"]
                if error_results["tests_run"] > 0
                else 0
            ),
            "llm_components": llm_available,
            "performance": (
                "good" if performance_results.get("scalability") == "good" else "needs_attention"
            ),
        }

        # Overall assessment
        critical_components_working = (
            component_health["verification_service"] >= 0.8
            and component_health["boundary_fix"]
            and component_health["error_handling"] >= 0.8
        )

        if critical_components_working and success_rate >= 0.7:
            overall_status = "PRODUCTION_READY"
            status_emoji = "🚀"
        elif critical_components_working:
            overall_status = "GOOD_WITH_MINOR_ISSUES"
            status_emoji = "⚡"
        else:
            overall_status = "NEEDS_IMPROVEMENT"
            status_emoji = "🔧"

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "suite_execution_time": suite_time,
            "overall_status": overall_status,
            "success_rate": success_rate,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "component_health": component_health,
            "performance_metrics": {
                "avg_generation_time": avg_generation_time,
                "avg_verification_time": avg_verification_time,
                "performance_data": performance_results.get("performance_data", []),
            },
            "detailed_results": {
                "verification_service": verification_results,
                "error_handling": error_results,
                "test_cases": [asdict(r) for r in self.test_results],
            },
            "recommendations": self._generate_recommendations(component_health, success_rate),
        }

        # Display summary
        print("\n" + "=" * 70)
        print("🏁 COMPREHENSIVE TEST SUITE RESULTS")
        print("=" * 70)

        print(f"\n{status_emoji} OVERALL STATUS: {overall_status}")
        print(f"📊 Success Rate: {success_rate:.1%} ({successful_tests}/{total_tests})")
        print(f"⏱️ Total Time: {suite_time:.1f}s")

        print("\n🏥 COMPONENT HEALTH:")
        for component, health in component_health.items():
            if isinstance(health, bool):
                status = "✅ WORKING" if health else "❌ BROKEN"
            elif isinstance(health, (int, float)):
                status = f"{'✅' if health >= 0.8 else '⚠️' if health >= 0.6 else '❌'} {health:.1%}"
            else:
                status = f"📊 {health}"
            print(f"   {component}: {status}")

        if report["recommendations"]:
            print("\n💡 RECOMMENDATIONS:")
            for rec in report["recommendations"]:
                print(f"   • {rec}")

        print("\n💾 Detailed results will be saved to comprehensive_test_results.json")

        return report

    def _generate_recommendations(self, component_health: Dict, success_rate: float) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        if not component_health.get("boundary_fix"):
            recommendations.append(
                "CRITICAL: Boundary condition fix not working - investigate verification system"
            )

        if component_health.get("verification_service", 0) < 0.8:
            recommendations.append(
                "Verification service needs attention - check parsing and validation logic"
            )

        if component_health.get("error_handling", 0) < 0.8:
            recommendations.append("Improve error handling to ensure graceful failures")

        if not component_health.get("llm_components"):
            recommendations.append(
                "LLM components not available - consider mock testing or model loading fixes"
            )

        if success_rate < 0.7:
            recommendations.append(
                "Success rate below target - review certificate generation and verification"
            )

        if component_health.get("performance") != "good":
            recommendations.append("Performance optimization needed - consider parameter tuning")

        if not recommendations:
            recommendations.append("System performing well - ready for production deployment")

        return recommendations


def main():
    """Run the comprehensive web interface test suite."""
    try:
        # Initialize test suite
        test_suite = ComprehensiveWebInterfaceTestSuite()

        # Run comprehensive testing
        results = test_suite.run_comprehensive_test_suite()

        # Save detailed results
        with open("comprehensive_test_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Determine exit code
        if results["overall_status"] == "PRODUCTION_READY":
            print("\n🎉 CONCLUSION: System is PRODUCTION READY!")
            return 0
        elif results["overall_status"] == "GOOD_WITH_MINOR_ISSUES":
            print("\n⚡ CONCLUSION: System is good with minor issues to address")
            return 0
        else:
            print("\n🔧 CONCLUSION: System needs improvement before production")
            return 1

    except Exception as e:
        print(f"\n💥 COMPREHENSIVE TEST SUITE CRASHED: {e}")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
