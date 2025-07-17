"""
Comprehensive LLM Generation Testbench for FM-LLM-Solver
Tests real LLM generation, extraction, parsing, and verification
using the same code as the web interface.
"""

import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.config_loader import load_config
from web_interface.certificate_generator import CertificateGenerator
from web_interface.verification_service import VerificationService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Represents a single test case for LLM generation."""

    name: str
    system_description: str
    domain_bounds: Dict[str, List[float]]
    expected_properties: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Results from a single test execution."""

    test_case: TestCase
    model_config: str
    rag_k: int
    generation_success: bool = False
    certificate: Optional[str] = None
    llm_output: Optional[str] = None
    extraction_success: bool = False
    verification_results: Dict[str, Any] = field(default_factory=dict)
    generation_time: float = 0.0
    verification_time: float = 0.0
    error: Optional[str] = None
    attempt_number: int = 1


class LLMGenerationTestbench:
    """Comprehensive testbench for LLM barrier certificate generation."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the testbench with configuration."""
        self.config = load_config(config_path)
        self.certificate_generator = CertificateGenerator(self.config)
        self.verification_service = VerificationService(self.config)
        self.results: List[TestResult] = []
        self.test_cases: List[TestCase] = []

        # Statistics tracking
        self.stats = defaultdict(
            lambda: {
                "total": 0,
                "generation_success": 0,
                "extraction_success": 0,
                "verification_success": 0,
                "errors": [],
            }
        )

    def add_test_case(self, test_case: TestCase) -> None:
        """Add a test case to the suite."""
        self.test_cases.append(test_case)

    def setup_default_test_cases(self) -> None:
        """Set up a comprehensive set of default test cases."""
        # Linear discrete-time systems
        self.add_test_case(
            TestCase(
                name="Linear Stable System",
                system_description="Discrete-time linear system: x[k+1] = 0.9*x[k] + 0.1*y[k], y[k+1] = -0.1*x[k] + 0.8*y[k]",
                domain_bounds={"x": [-5, 5], "y": [-5, 5]},
                expected_properties={"type": "quadratic", "positive_definite": True},
                tags=["linear", "stable", "basic"],
            )
        )

        self.add_test_case(
            TestCase(
                name="Linear System with Coupling",
                system_description="Discrete-time system: x[k+1] = 0.95*x[k] - 0.2*y[k], y[k+1] = 0.1*x[k] + 0.85*y[k]",
                domain_bounds={"x": [-10, 10], "y": [-10, 10]},
                expected_properties={"type": "quadratic"},
                tags=["linear", "coupled"],
            )
        )

        # Nonlinear systems
        self.add_test_case(
            TestCase(
                name="Nonlinear Polynomial System",
                system_description="Discrete-time nonlinear system: x[k+1] = 0.8*x[k] - 0.1*x[k]*y[k], y[k+1] = 0.9*y[k] + 0.05*x[k]**2",
                domain_bounds={"x": [-2, 2], "y": [-2, 2]},
                expected_properties={"type": "polynomial"},
                tags=["nonlinear", "polynomial"],
            )
        )

        # Systems with specific safety regions
        self.add_test_case(
            TestCase(
                name="System with Safety Constraint",
                system_description="Discrete-time system x[k+1] = 0.7*x[k] + 0.2*y[k], y[k+1] = -0.3*x[k] + 0.8*y[k] with safety region x^2 + y^2 <= 4",
                domain_bounds={"x": [-3, 3], "y": [-3, 3]},
                expected_properties={"includes_safety": True},
                tags=["safety", "constrained"],
            )
        )

        # Edge cases
        self.add_test_case(
            TestCase(
                name="Simple Decoupled System",
                system_description="Discrete-time decoupled system: x[k+1] = 0.9*x[k], y[k+1] = 0.85*y[k]",
                domain_bounds={"x": [-1, 1], "y": [-1, 1]},
                expected_properties={"type": "separable"},
                tags=["simple", "decoupled"],
            )
        )

        # Systems with different domain sizes
        self.add_test_case(
            TestCase(
                name="Large Domain System",
                system_description="Discrete-time system: x[k+1] = 0.99*x[k] - 0.01*y[k], y[k+1] = 0.01*x[k] + 0.99*y[k]",
                domain_bounds={"x": [-100, 100], "y": [-100, 100]},
                expected_properties={"large_domain": True},
                tags=["large_domain"],
            )
        )

        self.add_test_case(
            TestCase(
                name="Asymmetric Domain System",
                system_description="Discrete-time system: x[k+1] = 0.8*x[k] + 0.1*y[k], y[k+1] = -0.2*x[k] + 0.9*y[k]",
                domain_bounds={"x": [-5, 10], "y": [-2, 8]},
                expected_properties={"asymmetric_domain": True},
                tags=["asymmetric"],
            )
        )

    def run_single_test(
        self, test_case: TestCase, model_config: str, rag_k: int, attempt: int = 1
    ) -> TestResult:
        """Run a single test case and return results."""
        logger.info(
            f"Running test '{test_case.name}' with model '{model_config}', RAG k={rag_k}, attempt {attempt}"
        )

        result = TestResult(
            test_case=test_case,
            model_config=model_config,
            rag_k=rag_k,
            attempt_number=attempt,
        )

        try:
            # Generation phase
            gen_start = time.time()
            generation_result = self.certificate_generator.generate_certificate(
                system_description=test_case.system_description,
                model_key=model_config,
                rag_k=rag_k,
                domain_bounds=test_case.domain_bounds,
            )
            result.generation_time = time.time() - gen_start

            # Check generation success
            result.generation_success = generation_result.get("success", False)
            result.llm_output = generation_result.get("llm_output", "")
            result.certificate = generation_result.get("certificate", None)

            if not result.generation_success:
                result.error = generation_result.get(
                    "error", "Unknown generation error"
                )
                logger.error(f"Generation failed: {result.error}")
                return result

            # Check extraction success
            result.extraction_success = result.certificate is not None
            if not result.extraction_success:
                result.error = "Failed to extract valid certificate from LLM output"
                logger.warning(
                    f"Extraction failed. LLM output: {result.llm_output[:200]}..."
                )
                return result

            logger.info(f"Successfully extracted certificate: {result.certificate}")

            # Normalize variable names for verification (x -> x_, y -> y_)
            # This is needed because the verification system expects underscored variables
            normalized_certificate = result.certificate
            if normalized_certificate:
                # Replace standalone x and y with x_ and y_
                import re

                normalized_certificate = re.sub(r"\bx\b", "x_", normalized_certificate)
                normalized_certificate = re.sub(r"\by\b", "y_", normalized_certificate)
                logger.info(
                    f"Normalized certificate for verification: {normalized_certificate}"
                )

            # Verification phase
            verif_start = time.time()

            # Use verification parameters from config with some overrides for testing
            verif_params = {
                "num_samples_lie": 1000,
                "num_samples_boundary": 500,
                "numerical_tolerance": 1e-6,
                "attempt_sos": True,
                "attempt_optimization": True,
                "optimization_max_iter": 100,
                "optimization_pop_size": 15,
            }

            verification_result = self.verification_service.verify_certificate(
                normalized_certificate,  # certificate_str - positional (with normalized variables)
                test_case.system_description,  # system_description - positional
                verif_params,  # param_overrides - positional
                test_case.domain_bounds,  # domain_bounds - positional
            )
            result.verification_time = time.time() - verif_start

            # Store verification results
            result.verification_results = {
                "numerical_passed": verification_result.get("numerical_passed", False),
                "symbolic_passed": verification_result.get("symbolic_passed", False),
                "sos_passed": verification_result.get("sos_passed", False),
                "overall_success": verification_result.get("overall_success", False),
                "details": verification_result.get("details", {}),
            }

            logger.info(
                f"Verification complete - Overall: {result.verification_results['overall_success']}"
            )

        except Exception as e:
            result.error = f"Exception during test: {str(e)}"
            logger.error(f"Test failed with exception: {e}", exc_info=True)

        return result

    def run_test_suite(
        self,
        model_configs: List[str] = None,
        rag_k_values: List[int] = None,
        max_attempts: int = 3,
    ) -> None:
        """Run the complete test suite with specified configurations."""
        if not self.test_cases:
            logger.info("No test cases defined, setting up defaults...")
            self.setup_default_test_cases()

        if model_configs is None:
            # Get all available models
            available_models = self.certificate_generator.get_available_models()
            model_configs = [m["key"] for m in available_models if m["available"]]
            logger.info(f"Testing all available models: {model_configs}")

        if rag_k_values is None:
            rag_k_values = [0, 3]  # Test with and without RAG

        total_tests = len(self.test_cases) * len(model_configs) * len(rag_k_values)
        logger.info(f"Starting test suite: {total_tests} total configurations")

        test_count = 0
        for test_case in self.test_cases:
            for model_config in model_configs:
                for rag_k in rag_k_values:
                    test_count += 1
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Test {test_count}/{total_tests}")
                    logger.info(f"{'='*60}")

                    # Run test with retries
                    best_result = None
                    for attempt in range(1, max_attempts + 1):
                        result = self.run_single_test(
                            test_case, model_config, rag_k, attempt
                        )
                        self.results.append(result)

                        # Update statistics
                        key = f"{model_config}_k{rag_k}"
                        self.stats[key]["total"] += 1

                        if result.generation_success:
                            self.stats[key]["generation_success"] += 1
                        if result.extraction_success:
                            self.stats[key]["extraction_success"] += 1
                        if result.verification_results.get("overall_success", False):
                            self.stats[key]["verification_success"] += 1

                        if result.error:
                            self.stats[key]["errors"].append(
                                {
                                    "test": test_case.name,
                                    "error": result.error,
                                    "attempt": attempt,
                                }
                            )

                        # If we got a successful verification, stop retrying
                        if result.verification_results.get("overall_success", False):
                            best_result = result
                            logger.info(f"✓ Test passed on attempt {attempt}")
                            break
                        else:
                            logger.warning(f"✗ Test failed on attempt {attempt}")

                    if best_result is None:
                        logger.error(f"Test failed after {max_attempts} attempts")

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze test results and generate insights."""
        analysis = {
            "summary": {},
            "by_model": {},
            "by_test_case": {},
            "common_errors": {},
            "performance": {},
        }

        # Overall summary
        total_tests = len(self.results)
        successful_generations = sum(1 for r in self.results if r.generation_success)
        successful_extractions = sum(1 for r in self.results if r.extraction_success)
        successful_verifications = sum(
            1
            for r in self.results
            if r.verification_results.get("overall_success", False)
        )

        analysis["summary"] = {
            "total_tests": total_tests,
            "generation_success_rate": (
                successful_generations / total_tests if total_tests > 0 else 0
            ),
            "extraction_success_rate": (
                successful_extractions / total_tests if total_tests > 0 else 0
            ),
            "verification_success_rate": (
                successful_verifications / total_tests if total_tests > 0 else 0
            ),
        }

        # Analysis by model configuration
        model_results = defaultdict(lambda: {"tests": [], "success_count": 0})
        for result in self.results:
            key = f"{result.model_config}_k{result.rag_k}"
            model_results[key]["tests"].append(result)
            if result.verification_results.get("overall_success", False):
                model_results[key]["success_count"] += 1

        for key, data in model_results.items():
            test_count = len(data["tests"])
            analysis["by_model"][key] = {
                "total_tests": test_count,
                "success_rate": (
                    data["success_count"] / test_count if test_count > 0 else 0
                ),
                "avg_generation_time": np.mean(
                    [r.generation_time for r in data["tests"]]
                ),
                "avg_verification_time": np.mean(
                    [r.verification_time for r in data["tests"]]
                ),
            }

        # Analysis by test case
        test_case_results = defaultdict(lambda: {"attempts": 0, "successes": 0})
        for result in self.results:
            test_name = result.test_case.name
            test_case_results[test_name]["attempts"] += 1
            if result.verification_results.get("overall_success", False):
                test_case_results[test_name]["successes"] += 1

        for test_name, data in test_case_results.items():
            analysis["by_test_case"][test_name] = {
                "success_rate": (
                    data["successes"] / data["attempts"] if data["attempts"] > 0 else 0
                ),
                "total_attempts": data["attempts"],
            }

        # Common error patterns
        error_counts = defaultdict(int)
        for result in self.results:
            if result.error:
                # Categorize errors
                if "placeholder" in result.error.lower():
                    error_counts["placeholder_variables"] += 1
                elif "extract" in result.error.lower():
                    error_counts["extraction_failure"] += 1
                elif "timeout" in result.error.lower():
                    error_counts["timeout"] += 1
                else:
                    error_counts["other"] += 1

        analysis["common_errors"] = dict(error_counts)

        return analysis

    def generate_report(self, output_file: str = "testbench_report.json") -> None:
        """Generate a comprehensive test report."""
        logger.info("\nGenerating test report...")

        # Analyze results
        analysis = self.analyze_results()

        # Create report
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_test_cases": len(self.test_cases),
                "models_tested": list(set(r.model_config for r in self.results)),
                "rag_k_values": list(set(r.rag_k for r in self.results)),
            },
            "summary": analysis["summary"],
            "model_performance": analysis["by_model"],
            "test_case_analysis": analysis["by_test_case"],
            "error_analysis": analysis["common_errors"],
            "detailed_results": [],
        }

        # Add detailed results for failed tests
        for result in self.results:
            if not result.verification_results.get("overall_success", False):
                report["detailed_results"].append(
                    {
                        "test_case": result.test_case.name,
                        "model": result.model_config,
                        "rag_k": result.rag_k,
                        "attempt": result.attempt_number,
                        "certificate": result.certificate,
                        "error": result.error,
                        "verification_details": result.verification_results.get(
                            "details", {}
                        ),
                    }
                )

        # Save report
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUITE SUMMARY")
        print("=" * 60)
        print(f"Total Tests Run: {analysis['summary']['total_tests']}")
        print(
            f"Generation Success Rate: {analysis['summary']['generation_success_rate']:.2%}"
        )
        print(
            f"Extraction Success Rate: {analysis['summary']['extraction_success_rate']:.2%}"
        )
        print(
            f"Verification Success Rate: {analysis['summary']['verification_success_rate']:.2%}"
        )
        print("\nModel Performance:")
        for model, stats in analysis["by_model"].items():
            print(f"  {model}: {stats['success_rate']:.2%} success rate")
        print(f"\nDetailed report saved to: {output_file}")

    def run_focused_test(
        self,
        system_description: str,
        domain_bounds: Dict[str, List[float]],
        model_config: str = "base",
        rag_k: int = 3,
    ) -> TestResult:
        """Run a focused test on a specific system for debugging."""
        test_case = TestCase(
            name="Focused Test",
            system_description=system_description,
            domain_bounds=domain_bounds,
            tags=["focused", "debug"],
        )

        result = self.run_single_test(test_case, model_config, rag_k)

        # Print detailed output for debugging
        print("\n" + "=" * 60)
        print("FOCUSED TEST RESULTS")
        print("=" * 60)
        print(f"System: {system_description}")
        print(f"Domain: {domain_bounds}")
        print(f"Model: {model_config}, RAG k={rag_k}")
        print("-" * 60)
        print(f"Generation Success: {result.generation_success}")
        print(f"Extraction Success: {result.extraction_success}")
        print(f"Certificate: {result.certificate}")
        print(
            f"Verification: {result.verification_results.get('overall_success', False)}"
        )
        if result.error:
            print(f"Error: {result.error}")
        print("-" * 60)
        print("LLM Output:")
        print(
            result.llm_output[:500] + "..."
            if len(result.llm_output) > 500
            else result.llm_output
        )

        return result


def main():
    """Main entry point for the testbench."""
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM Generation Testbench for FM-LLM-Solver"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument("--models", nargs="+", help="Specific models to test")
    parser.add_argument(
        "--rag-k", nargs="+", type=int, default=[0, 3], help="RAG k values to test"
    )
    parser.add_argument(
        "--max-attempts", type=int, default=3, help="Maximum attempts per test"
    )
    parser.add_argument(
        "--output", default="testbench_report.json", help="Output report file"
    )
    parser.add_argument("--focused", action="store_true", help="Run focused test mode")
    parser.add_argument("--system", help="System description for focused test")
    parser.add_argument("--domain", help="Domain bounds for focused test (JSON format)")

    args = parser.parse_args()

    # Create testbench
    testbench = LLMGenerationTestbench(args.config)

    if args.focused:
        # Run focused test
        if not args.system:
            system = "Discrete-time system: x[k+1] = 0.9*x[k] + 0.1*y[k], y[k+1] = -0.1*x[k] + 0.8*y[k]"
        else:
            system = args.system

        if not args.domain:
            domain = {"x": [-5, 5], "y": [-5, 5]}
        else:
            domain = json.loads(args.domain)

        testbench.run_focused_test(
            system, domain, args.models[0] if args.models else "base", args.rag_k[0]
        )
    else:
        # Run full test suite
        testbench.run_test_suite(
            model_configs=args.models,
            rag_k_values=args.rag_k,
            max_attempts=args.max_attempts,
        )

        # Generate report
        testbench.generate_report(args.output)


if __name__ == "__main__":
    main()
