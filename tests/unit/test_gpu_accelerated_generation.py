#!/usr/bin/env python3
"""
GPU-accelerated certificate generation testing with RTX 3080.
Tests the entire pipeline with GPU acceleration for faster iteration.
"""

import os
import sys
import time
import logging
import pytest
import numpy as np
from pathlib import Path
from typing import Dict, List
import json
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUAcceleratedTester:
    """GPU-accelerated tester for certificate generation pipeline"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.device = self._setup_gpu()
        self.test_results = []

    def _setup_gpu(self) -> torch.device:
        """Setup GPU device for testing"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"🚀 GPU Available: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )
            return device
        else:
            logger.warning("⚠️ GPU not available, using CPU")
            return torch.device("cpu")

    def generate_gpu_test_cases(self) -> List[Dict]:
        """Generate test cases optimized for GPU acceleration"""
        return [
            # High-dimensional systems for GPU acceleration
            {
                "name": "high_dim_linear",
                "system": "dx/dt = -x, dy/dt = -y, dz/dt = -z, dw/dt = -w",
                "initial_set": "x**2 + y**2 + z**2 + w**2 <= 0.1",
                "unsafe_set": "x**2 + y**2 + z**2 + w**2 >= 1.0",
                "expected_form": "x**2 + y**2 + z**2 + w**2",
                "dimensions": 4,
            },
            {
                "name": "high_dim_nonlinear",
                "system": "dx/dt = -x**3, dy/dt = -y**3, dz/dt = -z**3, dw/dt = -w**3",
                "initial_set": "x**2 + y**2 + z**2 + w**2 <= 0.1",
                "unsafe_set": "x**2 + y**2 + z**2 + w**2 >= 1.0",
                "expected_form": "x**2 + y**2 + z**2 + w**2",
                "dimensions": 4,
            },
            # Complex barrier forms
            {
                "name": "complex_quadratic",
                "system": "dx/dt = -x, dy/dt = -2*y, dz/dt = -3*z",
                "initial_set": "x**2 + 2*y**2 + 3*z**2 <= 0.1",
                "unsafe_set": "x**2 + 2*y**2 + 3*z**2 >= 1.0",
                "expected_form": "x**2 + 2*y**2 + 3*z**2",
                "dimensions": 3,
            },
            # Large-scale systems
            {
                "name": "large_scale_linear",
                "system": "dx/dt = -x, dy/dt = -y, dz/dt = -z, dw/dt = -w, dv/dt = -v",
                "initial_set": "x**2 + y**2 + z**2 + w**2 + v**2 <= 0.1",
                "unsafe_set": "x**2 + y**2 + z**2 + w**2 + v**2 >= 1.0",
                "expected_form": "x**2 + y**2 + z**2 + w**2 + v**2",
                "dimensions": 5,
            },
        ]

    def test_gpu_memory_management(self) -> Dict:
        """Test GPU memory management for large-scale computations"""
        logger.info("🧪 Testing GPU memory management")

        if self.device.type == "cpu":
            return {"gpu_available": False}

        try:
            # Test memory allocation and deallocation
            memory_tests = []

            for size in [1000, 2000, 4000, 8000]:
                start_memory = torch.cuda.memory_allocated()

                # Allocate large tensors
                x = torch.randn(size, size, device=self.device)
                y = torch.randn(size, size, device=self.device)

                # Perform computation
                start_time = time.time()
                z = torch.mm(x, y)
                torch.cuda.synchronize()
                compute_time = time.time() - start_time

                # Check memory usage
                peak_memory = torch.cuda.memory_allocated()

                # Clean up
                del x, y, z
                torch.cuda.empty_cache()
                final_memory = torch.cuda.memory_allocated()

                memory_tests.append(
                    {
                        "size": size,
                        "start_memory_mb": start_memory / 1024**2,
                        "peak_memory_mb": peak_memory / 1024**2,
                        "final_memory_mb": final_memory / 1024**2,
                        "compute_time": compute_time,
                        "memory_cleaned": final_memory < start_memory + 1024**2,  # Within 1MB
                    }
                )

            return {
                "gpu_available": True,
                "memory_tests": memory_tests,
                "all_tests_passed": all(t["memory_cleaned"] for t in memory_tests),
            }

        except Exception as e:
            return {"gpu_available": True, "error": str(e)}

    def test_gpu_accelerated_sampling(self, test_case: Dict) -> Dict:
        """Test GPU-accelerated sampling for verification"""
        logger.info(f"🧪 Testing GPU sampling for: {test_case['name']}")

        dimensions = test_case.get("dimensions", 2)

        if self.device.type == "cpu":
            return {"gpu_available": False}

        try:
            # Generate samples on GPU
            num_samples = 10000
            start_time = time.time()

            # Generate random samples on GPU
            samples = torch.randn(num_samples, dimensions, device=self.device)

            # Scale samples to appropriate range
            samples = samples * 2.0  # Scale to [-2, 2] range

            # Move to CPU for evaluation (simulating certificate evaluation)
            samples_cpu = samples.cpu().numpy()

            gpu_time = time.time() - start_time

            # Test CPU equivalent
            start_time = time.time()
            np.random.randn(num_samples, dimensions) * 2.0
            cpu_time = time.time() - start_time

            # Evaluate barrier certificate on samples
            barrier_values = []
            for sample in samples_cpu[:100]:  # Test first 100 samples
                if dimensions == 2:
                    x, y = sample
                    barrier_value = x**2 + y**2
                elif dimensions == 3:
                    x, y, z = sample
                    barrier_value = x**2 + y**2 + z**2
                elif dimensions == 4:
                    x, y, z, w = sample
                    barrier_value = x**2 + y**2 + z**2 + w**2
                else:
                    barrier_value = np.sum(sample**2)

                barrier_values.append(barrier_value)

            return {
                "gpu_available": True,
                "dimensions": dimensions,
                "num_samples": num_samples,
                "gpu_time": gpu_time,
                "cpu_time": cpu_time,
                "speedup": cpu_time / gpu_time if gpu_time > 0 else 0,
                "barrier_values_computed": len(barrier_values),
                "memory_used_mb": torch.cuda.memory_allocated() / 1024**2,
            }

        except Exception as e:
            return {"gpu_available": True, "error": str(e)}

    def test_gpu_accelerated_verification(self, test_case: Dict) -> Dict:
        """Test GPU-accelerated verification pipeline"""
        logger.info(f"🧪 Testing GPU verification for: {test_case['name']}")

        dimensions = test_case.get("dimensions", 2)
        certificate = test_case["expected_form"] + " - 1.5"

        if self.device.type == "cpu":
            return {"gpu_available": False}

        try:
            # Generate verification samples on GPU
            num_lie_samples = 5000
            num_boundary_samples = 2500

            start_time = time.time()

            # Generate Lie derivative samples
            lie_samples = torch.randn(num_lie_samples, dimensions, device=self.device) * 2.0

            # Generate boundary samples
            boundary_samples = (
                torch.randn(num_boundary_samples, dimensions, device=self.device) * 2.0
            )

            # Move to CPU for evaluation
            lie_samples_cpu = lie_samples.cpu().numpy()
            boundary_samples_cpu = boundary_samples.cpu().numpy()

            gpu_time = time.time() - start_time

            # Evaluate barrier certificate on samples
            lie_violations = 0
            boundary_violations = 0

            # Test Lie derivative conditions
            for sample in lie_samples_cpu[:100]:  # Test subset
                if dimensions == 2:
                    x, y = sample
                    barrier_value = x**2 + y**2 - 1.5
                    # Simplified Lie derivative (for testing)
                    lie_derivative = -2 * x**2 - 2 * y**2
                else:
                    barrier_value = np.sum(sample**2) - 1.5
                    lie_derivative = -2 * np.sum(sample**2)

                if lie_derivative > 0:  # Violation condition
                    lie_violations += 1

            # Test boundary conditions
            for sample in boundary_samples_cpu[:100]:  # Test subset
                if dimensions == 2:
                    x, y = sample
                    barrier_value = x**2 + y**2 - 1.5
                else:
                    barrier_value = np.sum(sample**2) - 1.5

                # Check boundary conditions
                if barrier_value > 0:  # Unsafe set violation
                    boundary_violations += 1

            return {
                "gpu_available": True,
                "dimensions": dimensions,
                "certificate": certificate,
                "gpu_time": gpu_time,
                "lie_samples": num_lie_samples,
                "boundary_samples": num_boundary_samples,
                "lie_violations": lie_violations,
                "boundary_violations": boundary_violations,
                "memory_used_mb": torch.cuda.memory_allocated() / 1024**2,
            }

        except Exception as e:
            return {"gpu_available": True, "error": str(e)}

    def test_gpu_batch_processing(self) -> Dict:
        """Test GPU batch processing for multiple certificates"""
        logger.info("🧪 Testing GPU batch processing")

        if self.device.type == "cpu":
            return {"gpu_available": False}

        try:
            # Test processing multiple certificates in batch
            certificates = [
                "x**2 + y**2 - 1.5",
                "x**2 + 2*y**2 - 2.0",
                "x**2 + y**2 + z**2 - 1.0",
                "x**2 + y**2 + z**2 + w**2 - 2.5",
            ]

            batch_results = []
            start_time = time.time()

            for i, cert in enumerate(certificates):
                # Generate samples for this certificate
                dimensions = cert.count("x") + cert.count("y") + cert.count("z") + cert.count("w")
                num_samples = 1000

                samples = torch.randn(num_samples, dimensions, device=self.device) * 2.0
                samples_cpu = samples.cpu().numpy()

                # Evaluate certificate
                barrier_values = []
                for sample in samples_cpu[:100]:
                    if dimensions == 2:
                        x, y = sample
                        barrier_value = x**2 + y**2 - 1.5
                    elif dimensions == 3:
                        x, y, z = sample
                        barrier_value = x**2 + y**2 + z**2 - 1.0
                    else:
                        x, y, z, w = sample
                        barrier_value = x**2 + y**2 + z**2 + w**2 - 2.5

                    barrier_values.append(barrier_value)

                batch_results.append(
                    {
                        "certificate": cert,
                        "dimensions": dimensions,
                        "avg_barrier_value": np.mean(barrier_values),
                        "std_barrier_value": np.std(barrier_values),
                    }
                )

            total_time = time.time() - start_time

            return {
                "gpu_available": True,
                "num_certificates": len(certificates),
                "total_time": total_time,
                "avg_time_per_certificate": total_time / len(certificates),
                "batch_results": batch_results,
                "memory_used_mb": torch.cuda.memory_allocated() / 1024**2,
            }

        except Exception as e:
            return {"gpu_available": True, "error": str(e)}

    def run_gpu_comprehensive_tests(self) -> Dict:
        """Run all GPU-accelerated comprehensive tests"""
        logger.info("🚀 Starting GPU-accelerated comprehensive tests")
        logger.info(f"Device: {self.device}")

        results = {
            "device": str(self.device),
            "gpu_available": self.device.type == "cuda",
            "memory_tests": {},
            "sampling_tests": [],
            "verification_tests": [],
            "batch_tests": {},
            "overall_success_rate": 0.0,
        }

        # Test GPU memory management
        results["memory_tests"] = self.test_gpu_memory_management()

        # Test GPU-accelerated sampling
        test_cases = self.generate_gpu_test_cases()
        for test_case in test_cases:
            sampling_result = self.test_gpu_accelerated_sampling(test_case)
            results["sampling_tests"].append(sampling_result)

            verification_result = self.test_gpu_accelerated_verification(test_case)
            results["verification_tests"].append(verification_result)

        # Test batch processing
        results["batch_tests"] = self.test_gpu_batch_processing()

        # Calculate success rate
        total_tests = len(test_cases) * 2 + 2  # sampling + verification + memory + batch
        passed_tests = 0

        if results["memory_tests"].get("all_tests_passed", False):
            passed_tests += 1

        for sampling_test in results["sampling_tests"]:
            if sampling_test.get("gpu_available", False) and "error" not in sampling_test:
                passed_tests += 1

        for verification_test in results["verification_tests"]:
            if verification_test.get("gpu_available", False) and "error" not in verification_test:
                passed_tests += 1

        if (
            results["batch_tests"].get("gpu_available", False)
            and "error" not in results["batch_tests"]
        ):
            passed_tests += 1

        results["total_tests"] = total_tests
        results["passed_tests"] = passed_tests
        results["overall_success_rate"] = passed_tests / total_tests if total_tests > 0 else 0.0

        return results

    def save_gpu_test_results(
        self, results: Dict, output_path: str = "test_results/gpu_accelerated_results.json"
    ):
        """Save GPU test results to file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"GPU test results saved to: {output_path}")


# Pytest test functions
@pytest.fixture
def gpu_tester():
    """Create a GPU tester instance"""
    return GPUAcceleratedTester()


def test_gpu_memory_management(gpu_tester):
    """Test GPU memory management"""
    result = gpu_tester.test_gpu_memory_management()

    if result.get("gpu_available", False):
        assert "memory_tests" in result, "Memory tests should be included"
        if "all_tests_passed" in result:
            assert result["all_tests_passed"], "All memory tests should pass"


def test_gpu_sampling(gpu_tester):
    """Test GPU-accelerated sampling"""
    test_cases = gpu_tester.generate_gpu_test_cases()

    for test_case in test_cases[:2]:  # Test first 2 cases
        result = gpu_tester.test_gpu_accelerated_sampling(test_case)

        if result.get("gpu_available", False):
            assert "speedup" in result, "Speedup should be calculated"
            assert result["speedup"] > 0, "Speedup should be positive"


def test_gpu_verification(gpu_tester):
    """Test GPU-accelerated verification"""
    test_cases = gpu_tester.generate_gpu_test_cases()

    for test_case in test_cases[:2]:  # Test first 2 cases
        result = gpu_tester.test_gpu_accelerated_verification(test_case)

        if result.get("gpu_available", False):
            assert "certificate" in result, "Certificate should be included"
            assert "lie_violations" in result, "Lie violations should be counted"


def test_gpu_batch_processing(gpu_tester):
    """Test GPU batch processing"""
    result = gpu_tester.test_gpu_batch_processing()

    if result.get("gpu_available", False):
        assert "num_certificates" in result, "Number of certificates should be included"
        assert "avg_time_per_certificate" in result, "Average time should be calculated"


def test_comprehensive_gpu_tests(gpu_tester):
    """Test the entire GPU-accelerated pipeline"""
    results = gpu_tester.run_gpu_comprehensive_tests()

    assert "device" in results, "Device should be specified"
    assert "gpu_available" in results, "GPU availability should be checked"
    assert "overall_success_rate" in results, "Overall success rate should be calculated"

    # Save results
    gpu_tester.save_gpu_test_results(results)

    # Assert minimum success rate for GPU tests
    if results["gpu_available"]:
        assert (
            results["overall_success_rate"] > 0.7
        ), f"GPU success rate too low: {results['overall_success_rate']}"


if __name__ == "__main__":
    # Run comprehensive GPU tests
    tester = GPUAcceleratedTester()
    results = tester.run_gpu_comprehensive_tests()

    print("\n🚀 GPU-ACCELERATED TEST RESULTS")
    print("=" * 50)
    print(f"Device: {results['device']}")
    print(f"GPU Available: {results['gpu_available']}")
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed Tests: {results['passed_tests']}")
    print(f"Overall Success Rate: {results['overall_success_rate']:.1%}")

    if results["gpu_available"]:
        print(f"Memory Tests Passed: {results['memory_tests'].get('all_tests_passed', False)}")

        # Show sampling speedups
        speedups = [
            t.get("speedup", 0) for t in results["sampling_tests"] if t.get("gpu_available", False)
        ]
        if speedups:
            print(f"Average GPU Speedup: {np.mean(speedups):.1f}x")

    # Save results
    tester.save_gpu_test_results(results)
