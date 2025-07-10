#!/usr/bin/env python3
"""Performance benchmarks for certificate validation pipeline"""

import sys
import os
import time
import statistics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.certificate_extraction import extract_certificate_from_llm_output
from tests.unit.test_certificate_validation_accuracy import CertificateValidationTester

def benchmark_extraction():
    """Benchmark certificate extraction performance"""
    print("Benchmarking certificate extraction...")
    
    test_inputs = [
        ("BARRIER_CERTIFICATE_START\nx**2 + y**2 - 1.5\nBARRIER_CERTIFICATE_END", ["x", "y"]),
        ("B(x,y) = x**2 + y**2 - 1.0", ["x", "y"]),
        ("Certificate: x**2 + y**2 + z**2 - 2.0", ["x", "y", "z"]),
        ("The barrier certificate is B(x,y) = 2*x**2 + 3*y**2 - 1.5", ["x", "y"]),
    ] * 25  # 100 total extractions
    
    times = []
    for input_text, variables in test_inputs:
        start = time.perf_counter()
        extract_certificate_from_llm_output(input_text, variables)
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = statistics.mean(times) * 1000  # Convert to ms
    std_time = statistics.stdev(times) * 1000
    max_time = max(times) * 1000
    
    print(f"  Extractions: {len(test_inputs)}")
    print(f"  Average time: {avg_time:.2f}ms")
    print(f"  Std dev: {std_time:.2f}ms")
    print(f"  Max time: {max_time:.2f}ms")
    
    # Performance threshold: average should be < 10ms
    return avg_time < 10.0

def benchmark_validation():
    """Benchmark certificate validation performance"""
    print("\nBenchmarking certificate validation...")
    
    tester = CertificateValidationTester()
    system = {
        "dynamics": ["dx/dt = -x", "dy/dt = -y"],
        "initial_set": ["x**2 + y**2 <= 0.25"],
        "unsafe_set": ["x**2 + y**2 >= 4.0"]
    }
    
    certificates = [
        "x**2 + y**2 - 1.0",
        "x**2 + y**2 - 1.5",
        "2*x**2 + 3*y**2 - 2.0",
    ] * 10  # 30 total validations
    
    times = []
    for cert in certificates:
        start = time.perf_counter()
        tester.validate_certificate_mathematically(cert, system, n_samples=20)
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = statistics.mean(times) * 1000  # Convert to ms
    std_time = statistics.stdev(times) * 1000
    max_time = max(times) * 1000
    
    print(f"  Validations: {len(certificates)}")
    print(f"  Average time: {avg_time:.2f}ms")
    print(f"  Std dev: {std_time:.2f}ms")
    print(f"  Max time: {max_time:.2f}ms")
    
    # Performance threshold: average should be < 100ms
    return avg_time < 100.0

def benchmark_end_to_end():
    """Benchmark end-to-end pipeline performance"""
    print("\nBenchmarking end-to-end pipeline...")
    
    tester = CertificateValidationTester()
    system = {
        "dynamics": ["dx/dt = -x", "dy/dt = -y"],
        "initial_set": ["x**2 + y**2 <= 0.25"],
        "unsafe_set": ["x**2 + y**2 >= 4.0"]
    }
    
    llm_outputs = [
        "BARRIER_CERTIFICATE_START\nx**2 + y**2 - 1.0\nBARRIER_CERTIFICATE_END",
        "B(x,y) = x**2 + y**2 - 1.5",
        "Certificate: 2*x**2 + 2*y**2 - 3.0",
    ] * 10  # 30 total
    
    times = []
    for output in llm_outputs:
        start = time.perf_counter()
        
        # Extract
        extracted = extract_certificate_from_llm_output(output, ["x", "y"])
        cert = extracted[0] if isinstance(extracted, tuple) else extracted
        
        # Validate
        if cert:
            tester.validate_certificate_mathematically(cert, system, n_samples=20)
        
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = statistics.mean(times) * 1000  # Convert to ms
    std_time = statistics.stdev(times) * 1000
    max_time = max(times) * 1000
    
    print(f"  Pipeline runs: {len(llm_outputs)}")
    print(f"  Average time: {avg_time:.2f}ms")
    print(f"  Std dev: {std_time:.2f}ms")
    print(f"  Max time: {max_time:.2f}ms")
    
    # Performance threshold: average should be < 150ms
    return avg_time < 150.0

def main():
    """Run all performance benchmarks"""
    print("Performance Benchmark Suite")
    print("="*60)
    
    results = []
    
    # Run benchmarks
    results.append(("Extraction", benchmark_extraction()))
    results.append(("Validation", benchmark_validation()))
    results.append(("End-to-End", benchmark_end_to_end()))
    
    # Summary
    print("\n" + "="*60)
    print("Performance Summary:")
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        print("\nAll performance benchmarks passed!")
        print("The pipeline meets performance requirements for production use.")
        return 0
    else:
        print("\nSome performance benchmarks failed!")
        print("Performance optimization needed before production deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 