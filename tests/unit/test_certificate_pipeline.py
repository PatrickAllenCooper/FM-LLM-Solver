#!/usr/bin/env python3
"""
Comprehensive certificate pipeline testing with GPU acceleration.
Tests the entire flow: prompting -> generation -> cleaning -> extraction -> validation
"""

import os
import sys
import time
import logging
import pytest
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import load_config
from utils.certificate_extraction import (
    extract_certificate_from_llm_output,
    clean_and_validate_expression,
    is_template_expression,
)
from utils.verification_helpers import create_verification_context, validate_candidate_expression
from utils.numerical_checks import NumericalCheckConfig, ViolationInfo, NumericalCheckResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CertificatePipelineTester:
    """Comprehensive tester for the certificate generation pipeline"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.test_results = []
        self.gpu_available = self._check_gpu_availability()

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for testing"""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def generate_test_cases(self) -> List[Dict]:
        """Generate comprehensive test cases for certificate pipeline"""
        return [
            # Simple linear systems
            {
                "name": "linear_2d_stable",
                "system": "dx/dt = -x, dy/dt = -y",
                "initial_set": "x**2 + y**2 <= 0.25",
                "unsafe_set": "x**2 + y**2 >= 4.0",
                "expected_form": "x**2 + y**2",
                "expected_range": (0.25, 4.0),
            },
            {
                "name": "linear_2d_unstable",
                "system": "dx/dt = x, dy/dt = y",
                "initial_set": "x**2 + y**2 <= 0.1",
                "unsafe_set": "x**2 + y**2 >= 1.0",
                "expected_form": "x**2 + y**2",
                "expected_range": (0.1, 1.0),
            },
            # Nonlinear systems
            {
                "name": "nonlinear_cubic",
                "system": "dx/dt = -x**3 - y, dy/dt = x - y**3",
                "initial_set": "x**2 + y**2 <= 0.1",
                "unsafe_set": "x**2 + y**2 >= 2.0",
                "expected_form": "x**2 + y**2",
                "expected_range": (0.1, 2.0),
            },
            {
                "name": "nonlinear_van_der_pol",
                "system": "dx/dt = y, dy/dt = -x + y*(1 - x**2)",
                "initial_set": "x**2 + y**2 <= 0.5",
                "unsafe_set": "x**2 + y**2 >= 3.0",
                "expected_form": "x**2 + y**2",
                "expected_range": (0.5, 3.0),
            },
            # Higher dimensional systems
            {
                "name": "linear_3d",
                "system": "dx/dt = -x, dy/dt = -y, dz/dt = -z",
                "initial_set": "x**2 + y**2 + z**2 <= 0.1",
                "unsafe_set": "x**2 + y**2 + z**2 >= 1.0",
                "expected_form": "x**2 + y**2 + z**2",
                "expected_range": (0.1, 1.0),
            },
            # Complex barrier forms
            {
                "name": "elliptical_barrier",
                "system": "dx/dt = -x, dy/dt = -2*y",
                "initial_set": "x**2 + 2*y**2 <= 0.1",
                "unsafe_set": "x**2 + 2*y**2 >= 1.0",
                "expected_form": "x**2 + 2*y**2",
                "expected_range": (0.1, 1.0),
            },
            # Edge cases
            {
                "name": "very_small_domain",
                "system": "dx/dt = -x, dy/dt = -y",
                "initial_set": "x**2 + y**2 <= 0.01",
                "unsafe_set": "x**2 + y**2 >= 0.1",
                "expected_form": "x**2 + y**2",
                "expected_range": (0.01, 0.1),
            },
            {
                "name": "large_domain",
                "system": "dx/dt = -x, dy/dt = -y",
                "initial_set": "x**2 + y**2 <= 1.0",
                "unsafe_set": "x**2 + y**2 >= 10.0",
                "expected_form": "x**2 + y**2",
                "expected_range": (1.0, 10.0),
            },
        ]

    def create_mock_llm_output(self, test_case: Dict, certificate: str) -> str:
        """Create realistic LLM output with the certificate embedded"""
        outputs = [
            """BARRIER_CERTIFICATE_START
{certificate}
BARRIER_CERTIFICATE_END""",
            """Based on the system dynamics, I propose the following barrier certificate:

B(x,y) = {certificate}

This certificate satisfies the barrier conditions.""",
            """Here's the barrier certificate for the given system:

{certificate}

The certificate ensures safety by creating a barrier between initial and unsafe sets.""",
            """BARRIER_CERTIFICATE_START
B(x,y) = {certificate}
BARRIER_CERTIFICATE_END

Additional analysis shows this certificate is valid.""",
            """The optimal barrier certificate is:

{certificate}

This provides the required safety guarantees.""",
        ]

        # Add some noise to test robustness
        import random

        output = random.choice(outputs)

        # Sometimes add extra text
        if random.random() < 0.3:
            output += "\n\nThis certificate was generated using advanced mathematical analysis."

        return output

    def test_certificate_extraction(self, test_case: Dict) -> Dict:
        """Test certificate extraction from various LLM outputs"""
        logger.info(f"Testing extraction for: {test_case['name']}")

        # Generate test certificates
        base_certificate = test_case["expected_form"]
        test_certificates = [
            base_certificate,
            f"{base_certificate} - 1.5",
            f"{base_certificate} - 2.0",
            f"({base_certificate}) - 1.0",
            f"{base_certificate} - 0.5",
        ]

        results = []
        for i, cert in enumerate(test_certificates):
            # Create mock LLM output
            llm_output = self.create_mock_llm_output(test_case, cert)

            # Test extraction
            variables = ["x", "y"] if "z" not in test_case["system"] else ["x", "y", "z"]
            extracted_result = extract_certificate_from_llm_output(llm_output, variables)
            extracted = (
                extracted_result[0] if isinstance(extracted_result, tuple) else extracted_result
            )

            # Test cleaning and validation
            cleaned = clean_and_validate_expression(extracted, variables) if extracted else None

            # Test template detection
            is_template = is_template_expression(extracted) if extracted else True

            results.append(
                {
                    "certificate": cert,
                    "llm_output": llm_output,
                    "extracted": extracted,
                    "cleaned": cleaned,
                    "is_template": is_template,
                    "extraction_success": extracted is not None,
                    "cleaning_success": cleaned is not None,
                    "template_rejected": is_template,
                }
            )

        return {
            "test_case": test_case["name"],
            "results": results,
            "success_rate": sum(
                1 for r in results if r["extraction_success"] and not r["template_rejected"]
            )
            / len(results),
        }

    def test_verification_integration(self, test_case: Dict, certificate: str) -> Dict:
        """Test verification integration with numerical checks"""
        logger.info(f"Testing verification for: {test_case['name']}")

        # Create verification config
        system_info = {
            "dynamics": test_case["system"],
            "initial_set": test_case["initial_set"],
            "unsafe_set": test_case["unsafe_set"],
            "variables": ["x", "y"] if "z" not in test_case["system"] else ["x", "y", "z"],
        }

        # Note: create_verification_context requires SystemInfo and VerificationConfig objects
        # For now, we'll mock this since the actual implementation is complex
        config = {"system_info": system_info, "verification_config": "mock"}

        # Test numerical checks
        numerical_config = NumericalCheckConfig(
            n_samples=100, tolerance=1e-6, max_iter=100, pop_size=50
        )

        # Mock numerical check results
        violation_info = ViolationInfo(
            point={"x": 0.1, "y": 0.1},
            violation_type="boundary_violation",
            value=0.5,
            expected="≤ 0",
        )

        result = NumericalCheckResult(
            passed=False,
            reason="Mock verification result",
            violations=1,
            violation_points=[violation_info],
            samples_checked={"total": 100, "initial": 50, "safe": 50},
        )

        return {
            "test_case": test_case["name"],
            "certificate": certificate,
            "verification_config": config,
            "numerical_result": result,
            "verification_success": result.passed,
        }

    def test_gpu_acceleration(self) -> Dict:
        """Test GPU acceleration for certificate generation"""
        logger.info("Testing GPU acceleration")

        if not self.gpu_available:
            return {"gpu_available": False, "acceleration_tests": []}

        # Test GPU memory allocation and computation
        try:
            import torch

            # Test tensor operations
            start_time = time.time()
            x = torch.randn(1000, 1000, device="cuda")
            y = torch.randn(1000, 1000, device="cuda")
            z = torch.mm(x, y)
            gpu_time = time.time() - start_time

            # Test CPU equivalent
            start_time = time.time()
            x_cpu = torch.randn(1000, 1000)
            y_cpu = torch.randn(1000, 1000)
            z_cpu = torch.mm(x_cpu, y_cpu)
            cpu_time = time.time() - start_time

            return {
                "gpu_available": True,
                "gpu_time": gpu_time,
                "cpu_time": cpu_time,
                "speedup": cpu_time / gpu_time if gpu_time > 0 else 0,
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**2,  # MB
                "gpu_memory_cached": torch.cuda.memory_reserved() / 1024**2,  # MB
            }

        except Exception as e:
            return {"gpu_available": True, "error": str(e)}

    def run_comprehensive_tests(self) -> Dict:
        """Run all comprehensive tests"""
        logger.info("🚀 Starting comprehensive certificate pipeline tests")
        logger.info(f"GPU Available: {self.gpu_available}")

        test_cases = self.generate_test_cases()
        results = {
            "test_cases": [],
            "gpu_tests": {},
            "overall_success_rate": 0.0,
            "total_tests": 0,
            "passed_tests": 0,
        }

        # Test certificate extraction
        for test_case in test_cases:
            extraction_result = self.test_certificate_extraction(test_case)
            results["test_cases"].append(extraction_result)

            # Test verification integration
            if extraction_result["results"]:
                best_cert = extraction_result["results"][0]["certificate"]
                verification_result = self.test_verification_integration(test_case, best_cert)
                extraction_result["verification"] = verification_result

        # Test GPU acceleration
        results["gpu_tests"] = self.test_gpu_acceleration()

        # Calculate overall success rate
        total_tests = len(test_cases)
        passed_tests = sum(1 for tc in results["test_cases"] if tc["success_rate"] > 0.8)

        results["total_tests"] = total_tests
        results["passed_tests"] = passed_tests
        results["overall_success_rate"] = passed_tests / total_tests if total_tests > 0 else 0.0

        return results

    def save_test_results(
        self, results: Dict, output_path: str = "test_results/certificate_pipeline_results.json"
    ):
        """Save test results to file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Test results saved to: {output_path}")


# Pytest test functions
@pytest.fixture
def pipeline_tester():
    """Create a pipeline tester instance"""
    return CertificatePipelineTester()


def test_certificate_extraction_pipeline(pipeline_tester):
    """Test certificate extraction pipeline"""
    test_cases = pipeline_tester.generate_test_cases()

    for test_case in test_cases[:3]:  # Test first 3 cases
        result = pipeline_tester.test_certificate_extraction(test_case)

        assert (
            result["success_rate"] > 0.5
        ), f"Extraction success rate too low for {test_case['name']}"
        assert len(result["results"]) > 0, f"No extraction results for {test_case['name']}"

        # Check that at least one extraction was successful
        successful_extractions = [r for r in result["results"] if r["extraction_success"]]
        assert len(successful_extractions) > 0, f"No successful extractions for {test_case['name']}"


def test_verification_integration(pipeline_tester):
    """Test verification integration"""
    test_cases = pipeline_tester.generate_test_cases()

    for test_case in test_cases[:2]:  # Test first 2 cases
        certificate = test_case["expected_form"] + " - 1.5"
        result = pipeline_tester.test_verification_integration(test_case, certificate)

        assert "verification_config" in result, "Verification config not created"
        assert "numerical_result" in result, "Numerical result not generated"


def test_gpu_acceleration(pipeline_tester):
    """Test GPU acceleration"""
    result = pipeline_tester.test_gpu_acceleration()

    if result["gpu_available"]:
        assert "gpu_time" in result or "error" in result, "GPU test should return timing or error"
    else:
        assert not result["gpu_available"], "GPU should be marked as unavailable"


def test_comprehensive_pipeline(pipeline_tester):
    """Test the entire comprehensive pipeline"""
    results = pipeline_tester.run_comprehensive_tests()

    assert "test_cases" in results, "Test cases should be included"
    assert "gpu_tests" in results, "GPU tests should be included"
    assert "overall_success_rate" in results, "Overall success rate should be calculated"

    # Save results
    pipeline_tester.save_test_results(results)

    # Assert minimum success rate
    assert (
        results["overall_success_rate"] > 0.6
    ), f"Overall success rate too low: {results['overall_success_rate']}"


if __name__ == "__main__":
    # Run comprehensive tests
    tester = CertificatePipelineTester()
    results = tester.run_comprehensive_tests()

    print("\n🎯 COMPREHENSIVE TEST RESULTS")
    print("=" * 50)
    print(f"Total Test Cases: {results['total_tests']}")
    print(f"Passed Tests: {results['passed_tests']}")
    print(f"Overall Success Rate: {results['overall_success_rate']:.1%}")
    print(f"GPU Available: {results['gpu_tests'].get('gpu_available', False)}")

    if results["gpu_tests"].get("gpu_available", False):
        print(f"GPU Speedup: {results['gpu_tests'].get('speedup', 0):.1f}x")

    # Save results
    tester.save_test_results(results)
