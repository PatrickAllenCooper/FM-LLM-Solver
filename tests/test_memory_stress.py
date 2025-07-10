#!/usr/bin/env python3
"""
Memory and Stress Tests
Tests for memory usage, resource limits, and stress conditions
"""

import sys
import os
import time
import psutil
import gc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.certificate_extraction import extract_certificate_from_llm_output
from tests.unit.test_certificate_validation_accuracy import CertificateValidationTester

class TestMemoryStress:
    """Test memory usage and stress conditions"""
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def test_memory_leak(self):
        """Test for memory leaks in repeated operations"""
        print("Testing for memory leaks...")
        
        initial_memory = self.get_memory_usage()
        print(f"Initial memory: {initial_memory:.1f} MB")
        
        tester = CertificateValidationTester()
        system = {
            "dynamics": ["dx/dt = -x", "dy/dt = -y"],
            "initial_set": ["x**2 + y**2 <= 0.25"],
            "unsafe_set": ["x**2 + y**2 >= 4.0"]
        }
        
        # Perform many iterations
        iterations = 100
        memory_samples = []
        
        for i in range(iterations):
            # Extract and validate
            result = extract_certificate_from_llm_output(
                f"B(x,y) = x**2 + y**2 - {1.0 + i*0.01}", ["x", "y"]
            )
            cert = result[0] if isinstance(result, tuple) else result
            
            if cert:
                tester.validate_certificate_mathematically(cert, system, n_samples=5)
            
            # Sample memory every 10 iterations
            if i % 10 == 0:
                gc.collect()  # Force garbage collection
                current_memory = self.get_memory_usage()
                memory_samples.append(current_memory)
                print(f"  Iteration {i}: {current_memory:.1f} MB")
        
        # Check for memory growth
        memory_growth = memory_samples[-1] - memory_samples[0]
        growth_rate = memory_growth / len(memory_samples)
        
        print(f"\nMemory growth: {memory_growth:.1f} MB")
        print(f"Growth rate: {growth_rate:.3f} MB/sample")
        
        # Should not grow more than 50MB total
        assert memory_growth < 50, f"Memory leak detected: {memory_growth:.1f} MB growth"
    
    def test_large_expressions(self):
        """Test handling of very large certificate expressions"""
        print("\nTesting large expressions...")
        
        # Generate increasingly complex expressions
        test_cases = [
            # Standard size
            ("x**2 + y**2 - 1.0", "Standard"),
            
            # Many terms
            (" + ".join([f"{i}*x**2" for i in range(1, 11)]) + " + y**2 - 1.0", "10 terms"),
            
            # Deep nesting
            ("((((x**2 + y**2) - 0.5) + 0.5) - 0.5) + 0.5 - 1.0", "Deep nesting"),
            
            # Long variable names (edge case)
            ("very_long_variable_name_x**2 + very_long_variable_name_y**2 - 1.0", "Long names"),
        ]
        
        for expr, description in test_cases:
            print(f"\nTesting {description}...")
            start_mem = self.get_memory_usage()
            
            try:
                # Adjust variables for long name case
                if "very_long" in expr:
                    vars = ["very_long_variable_name_x", "very_long_variable_name_y"]
                    formatted_expr = f"B({vars[0]},{vars[1]}) = {expr}"
                else:
                    vars = ["x", "y"]
                    formatted_expr = f"B(x,y) = {expr}"
                
                result = extract_certificate_from_llm_output(formatted_expr, vars)
                cert = result[0] if isinstance(result, tuple) else result
                
                if cert:
                    print(f"  Extracted successfully")
                    # Validate to ensure it works end-to-end
                    if not "very_long" in expr:  # Skip validation for long names
                        tester = CertificateValidationTester()
                        system = {
                            "dynamics": ["dx/dt = -x", "dy/dt = -y"],
                            "initial_set": ["x**2 + y**2 <= 0.25"],
                            "unsafe_set": ["x**2 + y**2 >= 4.0"]
                        }
                        validation = tester.validate_certificate_mathematically(
                            cert, system, n_samples=5
                        )
                        print(f"  Validation: {'Valid' if validation['valid'] else 'Invalid'}")
                else:
                    print(f"  Extraction failed")
                
                end_mem = self.get_memory_usage()
                print(f"  Memory used: {end_mem - start_mem:.1f} MB")
                
            except Exception as e:
                print(f"  Exception: {type(e).__name__}")
    
    def test_concurrent_stress(self):
        """Test system under concurrent load"""
        print("\nTesting concurrent stress...")
        
        import threading
        import queue
        
        # Shared queue for results
        result_queue = queue.Queue()
        error_queue = queue.Queue()
        
        def worker(worker_id, num_operations):
            """Worker thread function"""
            tester = CertificateValidationTester()
            system = {
                "dynamics": ["dx/dt = -x", "dy/dt = -y"],
                "initial_set": ["x**2 + y**2 <= 0.25"],
                "unsafe_set": ["x**2 + y**2 >= 4.0"]
            }
            
            try:
                for i in range(num_operations):
                    # Generate unique certificate
                    cert_value = 0.5 + (worker_id * 0.1) + (i * 0.01)
                    llm_output = f"B(x,y) = x**2 + y**2 - {cert_value}"
                    
                    # Extract
                    result = extract_certificate_from_llm_output(llm_output, ["x", "y"])
                    cert = result[0] if isinstance(result, tuple) else result
                    
                    if cert:
                        # Validate
                        validation = tester.validate_certificate_mathematically(
                            cert, system, n_samples=5
                        )
                        result_queue.put((worker_id, i, validation['valid']))
                    else:
                        result_queue.put((worker_id, i, False))
                        
            except Exception as e:
                error_queue.put((worker_id, str(e)))
        
        # Start multiple worker threads
        num_threads = 5
        operations_per_thread = 20
        threads = []
        
        start_time = time.time()
        start_mem = self.get_memory_usage()
        
        print(f"Starting {num_threads} threads, {operations_per_thread} operations each...")
        
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i, operations_per_thread))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        duration = time.time() - start_time
        end_mem = self.get_memory_usage()
        
        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        errors = []
        while not error_queue.empty():
            errors.append(error_queue.get())
        
        print(f"\nCompleted in {duration:.2f}s")
        print(f"Total operations: {len(results)}")
        print(f"Errors: {len(errors)}")
        print(f"Memory used: {end_mem - start_mem:.1f} MB")
        print(f"Operations/second: {len(results)/duration:.1f}")
        
        # Should complete all operations without errors
        assert len(errors) == 0, f"Had {len(errors)} errors in concurrent execution"
        assert len(results) == num_threads * operations_per_thread
    
    def test_resource_limits(self):
        """Test behavior at resource limits"""
        print("\nTesting resource limits...")
        
        # Test with very high sampling density
        print("\nHigh sampling density test...")
        tester = CertificateValidationTester()
        system = {
            "dynamics": ["dx/dt = -x", "dy/dt = -y"],
            "initial_set": ["x**2 + y**2 <= 0.25"],
            "unsafe_set": ["x**2 + y**2 >= 4.0"]
        }
        
        # Test with increasing sample sizes
        sample_sizes = [10, 50, 100, 200]
        for n_samples in sample_sizes:
            start_time = time.time()
            start_mem = self.get_memory_usage()
            
            try:
                result = tester.validate_certificate_mathematically(
                    "x**2 + y**2 - 1.0", system, n_samples=n_samples
                )
                
                duration = time.time() - start_time
                mem_used = self.get_memory_usage() - start_mem
                
                print(f"  n_samples={n_samples}: {duration:.2f}s, {mem_used:.1f} MB")
                
                # Should complete even with high sampling
                assert result is not None
                
            except Exception as e:
                print(f"  n_samples={n_samples}: Failed - {type(e).__name__}")
                # High sample counts might fail, which is acceptable
                if n_samples <= 100:
                    raise  # But reasonable counts should work
    
    def test_input_size_limits(self):
        """Test limits on input sizes"""
        print("\nTesting input size limits...")
        
        # Test very long LLM outputs
        sizes = [100, 1000, 10000, 100000]
        
        for size in sizes:
            # Generate long output with certificate embedded
            padding = "This is padding text. " * (size // 20)
            llm_output = f"{padding}\nB(x,y) = x**2 + y**2 - 1.0\n{padding}"
            
            print(f"\nTesting {size} character input...")
            start_time = time.time()
            
            try:
                result = extract_certificate_from_llm_output(llm_output, ["x", "y"])
                cert = result[0] if isinstance(result, tuple) else result
                
                duration = time.time() - start_time
                
                if cert:
                    print(f"  Extracted in {duration:.3f}s")
                else:
                    print(f"  Failed to extract")
                
                # Should handle reasonable sizes
                if size <= 10000:
                    assert cert is not None, f"Should extract from {size} char input"
                    
            except Exception as e:
                print(f"  Exception: {type(e).__name__}")
                # Very large inputs might fail
                if size <= 10000:
                    raise

def main():
    """Run memory and stress tests"""
    print("Memory and Stress Test Suite")
    print("=" * 60)
    
    test = TestMemoryStress()
    
    # Run all stress tests
    test.test_memory_leak()
    test.test_large_expressions()
    test.test_concurrent_stress()
    test.test_resource_limits()
    test.test_input_size_limits()
    
    print("\nAll memory and stress tests completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 