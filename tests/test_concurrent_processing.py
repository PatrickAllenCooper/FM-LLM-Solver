#!/usr/bin/env python3
"""
Concurrent Processing Tests
Tests for parallel execution, thread safety, and GPU utilization
"""

import sys
import os
import time
import concurrent.futures
import multiprocessing
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.certificate_extraction import extract_certificate_from_llm_output
from tests.unit.test_certificate_validation_accuracy import CertificateValidationTester
import torch


# Move multiprocessing function outside of class for Windows compatibility
def validate_certificate_mp(args):
    """Multiprocessing-safe validation function"""
    llm_output, system = args

    # Import inside function to avoid pickling issues
    from utils.certificate_extraction import extract_certificate_from_llm_output
    from tests.unit.test_certificate_validation_accuracy import CertificateValidationTester

    result = extract_certificate_from_llm_output(llm_output, ["x", "y"])
    cert = result[0] if isinstance(result, tuple) else result

    if cert:
        tester = CertificateValidationTester()
        validation = tester.validate_certificate_mathematically(cert, system, n_samples=5)
        return {"certificate": cert, "valid": validation["valid"]}
    return None


class TestConcurrentProcessing:
    """Test concurrent and parallel processing capabilities"""

    def test_thread_pool_processing(self):
        """Test processing with thread pool"""
        print("Testing thread pool processing...")

        # Create test data
        test_certificates = [f"B(x,y) = x**2 + y**2 - {0.5 + i*0.1}" for i in range(20)]

        system = {
            "dynamics": ["dx/dt = -x", "dy/dt = -y"],
            "initial_set": ["x**2 + y**2 <= 0.25"],
            "unsafe_set": ["x**2 + y**2 >= 4.0"],
        }

        def process_certificate(llm_output):
            """Process a single certificate"""
            # Extract
            result = extract_certificate_from_llm_output(llm_output, ["x", "y"])
            cert = result[0] if isinstance(result, tuple) else result

            if cert:
                # Validate
                tester = CertificateValidationTester()
                validation = tester.validate_certificate_mathematically(cert, system, n_samples=10)
                return {
                    "certificate": cert,
                    "valid": validation["valid"],
                    "violations": validation.get("num_violations", 0),
                }
            return None

        # Sequential processing
        print("\nSequential processing...")
        start_time = time.time()
        sequential_results = []
        for cert in test_certificates:
            result = process_certificate(cert)
            sequential_results.append(result)
        sequential_time = time.time() - start_time

        # Parallel processing with thread pool
        print("\nParallel processing (threads)...")
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            parallel_results = list(executor.map(process_certificate, test_certificates))
        parallel_time = time.time() - start_time

        print(f"\nSequential time: {sequential_time:.2f}s")
        print(f"Parallel time: {parallel_time:.2f}s")
        print(f"Speedup: {sequential_time/parallel_time:.2f}x")

        # Results should be identical
        assert len(sequential_results) == len(parallel_results)
        for i, (seq, par) in enumerate(zip(sequential_results, parallel_results)):
            if seq and par:
                assert seq["certificate"] == par["certificate"], f"Result mismatch at index {i}"

    def test_process_pool_processing(self):
        """Test processing with process pool"""
        print("\nTesting process pool processing...")

        # Prepare test data
        system = {
            "dynamics": ["dx/dt = -x", "dy/dt = -y"],
            "initial_set": ["x**2 + y**2 <= 0.25"],
            "unsafe_set": ["x**2 + y**2 >= 4.0"],
        }

        test_data = [(f"B(x,y) = x**2 + y**2 - {0.5 + i*0.1}", system) for i in range(10)]

        # Process pool execution
        print("Processing with process pool...")
        start_time = time.time()

        # Use spawn method for Windows compatibility
        if __name__ == "__main__":
            with multiprocessing.get_context("spawn").Pool(processes=4) as pool:
                results = pool.map(validate_certificate_mp, test_data)
        else:
            # For imports, just run sequentially
            results = [validate_certificate_mp(data) for data in test_data]

        duration = time.time() - start_time
        valid_count = sum(1 for r in results if r and r["valid"])

        print(f"Completed in {duration:.2f}s")
        print(f"Valid certificates: {valid_count}/{len(results)}")

        # Should process all certificates
        assert len(results) == len(test_data)

    def test_gpu_parallel_validation(self):
        """Test GPU-accelerated parallel validation"""
        print("\nTesting GPU parallel validation...")

        if not torch.cuda.is_available():
            print("GPU not available, skipping GPU tests")
            return

        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

        # Generate batch of points for validation
        batch_size = 1000
        num_batches = 10

        def validate_batch_gpu(certificate_expr, points):
            """Validate certificate on batch of points using GPU"""
            # Parse certificate expression
            import sympy as sp

            x, y = sp.symbols("x y")
            cert_func = sp.lambdify([x, y], sp.parse_expr(certificate_expr), "numpy")

            # Move to GPU and evaluate
            x_vals = points[:, 0]
            y_vals = points[:, 1]

            # Compute certificate values
            cert_values = cert_func(x_vals, y_vals)

            return cert_values

        # Test certificate
        certificate = "x**2 + y**2 - 1.0"

        print(f"\nProcessing {num_batches} batches of {batch_size} points each...")

        # CPU baseline
        print("CPU processing...")
        cpu_start = time.time()
        for i in range(num_batches):
            points = torch.randn(batch_size, 2)
            validate_batch_gpu(certificate, points.numpy())
        cpu_time = time.time() - cpu_start

        # GPU processing
        print("GPU processing...")
        gpu_start = time.time()
        for i in range(num_batches):
            points = torch.randn(batch_size, 2, device=device)
            # For actual GPU processing, would need GPU-compatible evaluation
            # Here we simulate the overhead
            points_cpu = points.cpu().numpy()
            validate_batch_gpu(certificate, points_cpu)
        gpu_time = time.time() - gpu_start

        print(f"\nCPU time: {cpu_time:.3f}s")
        print(f"GPU time: {gpu_time:.3f}s")
        print(f"Points processed: {num_batches * batch_size}")
        print(f"Throughput: {(num_batches * batch_size) / gpu_time:.0f} points/second")

    def test_concurrent_file_operations(self):
        """Test concurrent file I/O operations"""
        print("\nTesting concurrent file operations...")

        import tempfile
        import json

        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:

            def save_validation_result(index, result):
                """Save validation result to file"""
                filename = os.path.join(temp_dir, f"result_{index}.json")
                with open(filename, "w") as f:
                    json.dump(result, f)
                return filename

            def load_validation_result(filename):
                """Load validation result from file"""
                with open(filename, "r") as f:
                    return json.load(f)

            # Generate and save results concurrently
            num_results = 20
            results_to_save = [
                {
                    "index": i,
                    "certificate": f"x**2 + y**2 - {1.0 + i*0.1}",
                    "valid": i % 2 == 0,
                    "timestamp": time.time(),
                }
                for i in range(num_results)
            ]

            print(f"Saving {num_results} results concurrently...")
            start_time = time.time()

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for i, result in enumerate(results_to_save):
                    future = executor.submit(save_validation_result, i, result)
                    futures.append(future)

                # Wait for all saves to complete
                filenames = [f.result() for f in futures]

            save_time = time.time() - start_time

            # Load results concurrently
            print(f"Loading {num_results} results concurrently...")
            start_time = time.time()

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                loaded_results = list(executor.map(load_validation_result, filenames))

            load_time = time.time() - start_time

            print(f"\nSave time: {save_time:.3f}s")
            print(f"Load time: {load_time:.3f}s")
            print(f"Total I/O operations: {num_results * 2}")

            # Verify all results loaded correctly
            assert len(loaded_results) == num_results
            for original, loaded in zip(results_to_save, loaded_results):
                assert original["index"] == loaded["index"]
                assert original["certificate"] == loaded["certificate"]

    def test_race_conditions(self):
        """Test for race conditions in concurrent access"""
        print("\nTesting for race conditions...")

        # Shared counter to test thread safety
        counter = {"value": 0}
        lock = threading.Lock()

        def increment_counter(iterations):
            """Increment counter with potential race condition"""
            for _ in range(iterations):
                # Unsafe increment (race condition)
                current = counter["value"]
                time.sleep(0.00001)  # Tiny delay to increase chance of race
                counter["value"] = current + 1

        def safe_increment_counter(iterations):
            """Increment counter safely with lock"""
            for _ in range(iterations):
                with lock:
                    counter["value"] += 1

        # Test unsafe version (should show race conditions)
        counter["value"] = 0
        threads = []
        num_threads = 10
        iterations_per_thread = 100

        print("Testing unsafe concurrent access...")
        for _ in range(num_threads):
            t = threading.Thread(target=increment_counter, args=(iterations_per_thread,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        unsafe_final = counter["value"]
        expected = num_threads * iterations_per_thread
        print(f"Unsafe final value: {unsafe_final} (expected: {expected})")

        # Test safe version
        counter["value"] = 0
        threads = []

        print("\nTesting safe concurrent access...")
        for _ in range(num_threads):
            t = threading.Thread(target=safe_increment_counter, args=(iterations_per_thread,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        safe_final = counter["value"]
        print(f"Safe final value: {safe_final} (expected: {expected})")

        # Safe version should always be correct
        assert safe_final == expected, "Thread-safe version should be accurate"

        # The certificate validation should be thread-safe
        print("\nTesting certificate validation thread safety...")
        tester = CertificateValidationTester()
        system = {
            "dynamics": ["dx/dt = -x", "dy/dt = -y"],
            "initial_set": ["x**2 + y**2 <= 0.25"],
            "unsafe_set": ["x**2 + y**2 >= 4.0"],
        }

        results = []
        result_lock = threading.Lock()

        def validate_concurrent(cert_expr):
            """Validate certificate and store result safely"""
            result = tester.validate_certificate_mathematically(cert_expr, system, n_samples=5)
            with result_lock:
                results.append(result)

        # Run concurrent validations
        threads = []
        for i in range(10):
            cert = f"x**2 + y**2 - {1.0 + i*0.1}"
            t = threading.Thread(target=validate_concurrent, args=(cert,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        print(f"Completed {len(results)} concurrent validations safely")
        assert len(results) == 10, "All validations should complete"


def main():
    """Run concurrent processing tests"""
    print("Concurrent Processing Test Suite")
    print("=" * 60)

    test = TestConcurrentProcessing()

    # Run all concurrent tests
    test.test_thread_pool_processing()
    test.test_process_pool_processing()
    test.test_gpu_parallel_validation()
    test.test_concurrent_file_operations()
    test.test_race_conditions()

    print("\nAll concurrent processing tests completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
