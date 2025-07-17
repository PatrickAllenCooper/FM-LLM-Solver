"""
Performance tests for FM-LLM-Solver components.

Tests system performance under load and measures response times.
"""

import time
import pytest
import asyncio
import concurrent.futures
from unittest.mock import Mock, patch

from fm_llm_solver.core.config_manager import ConfigurationManager
from fm_llm_solver.core.cache_manager import CacheManager
from fm_llm_solver.core.monitoring import MonitoringManager
from fm_llm_solver.services.certificate_generator import CertificateGenerator


class TestPerformance:
    """Performance test suite."""

    @pytest.fixture
    def performance_config_manager(self):
        """Mock config manager for performance tests."""
        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: {
            "cache.backend": "memory",
            "cache.max_size": 10000,
            "cache.default_ttl": 300,
            "monitoring.enabled": True,
            "monitoring.metrics.prometheus_enabled": True,
        }.get(key, default)
        return mock_config

    def test_cache_performance(self, performance_config_manager):
        """Test cache performance under load."""
        cache_manager = CacheManager(performance_config_manager)

        # Measure write performance
        start_time = time.time()
        for i in range(1000):
            cache_manager.set(f"key_{i}", f"value_{i}")
        write_time = time.time() - start_time

        # Measure read performance
        start_time = time.time()
        for i in range(1000):
            cache_manager.get(f"key_{i}")
        read_time = time.time() - start_time

        # Performance assertions
        assert write_time < 1.0, f"Cache writes took too long: {write_time}s"
        assert read_time < 0.5, f"Cache reads took too long: {read_time}s"

        # Check hit rate
        stats = cache_manager.get_stats()
        assert stats.get("hits", 0) >= 1000

    def test_concurrent_cache_access(self, performance_config_manager):
        """Test cache performance under concurrent access."""
        cache_manager = CacheManager(performance_config_manager)

        def cache_operations(worker_id):
            """Perform cache operations for a worker."""
            for i in range(100):
                key = f"worker_{worker_id}_key_{i}"
                value = f"worker_{worker_id}_value_{i}"
                cache_manager.set(key, value)
                retrieved = cache_manager.get(key)
                assert retrieved == value
            return True

        # Run concurrent operations
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(cache_operations, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        total_time = time.time() - start_time

        # All operations should succeed
        assert all(results)

        # Should complete within reasonable time
        assert total_time < 5.0, f"Concurrent operations took too long: {total_time}s"

    def test_monitoring_metrics_performance(self, performance_config_manager):
        """Test monitoring metrics performance."""
        monitoring_manager = MonitoringManager(performance_config_manager)

        # Test metric recording performance
        start_time = time.time()
        for i in range(1000):
            monitoring_manager.record_metric(
                f"test_metric_{i % 10}", float(i), {"worker": str(i % 5)}
            )

        metric_time = time.time() - start_time

        # Should record metrics quickly
        assert metric_time < 2.0, f"Metric recording took too long: {metric_time}s"

        # Test metric retrieval performance
        start_time = time.time()
        metrics = monitoring_manager.get_metrics()
        retrieval_time = time.time() - start_time

        assert retrieval_time < 0.5, f"Metric retrieval took too long: {retrieval_time}s"
        assert len(metrics) > 0

    @pytest.mark.slow
    def test_system_under_load(self, performance_config_manager):
        """Test system performance under sustained load."""
        cache_manager = CacheManager(performance_config_manager)
        monitoring_manager = MonitoringManager(performance_config_manager)

        # Simulate sustained load
        operations = 0
        errors = 0
        start_time = time.time()

        # Run for 10 seconds
        while time.time() - start_time < 10:
            try:
                # Cache operations
                cache_manager.set(f"load_key_{operations}", f"load_value_{operations}")
                cache_manager.get(f"load_key_{operations}")

                # Monitoring operations
                monitoring_manager.record_metric(
                    "load_test_operations", operations, {"type": "load"}
                )

                operations += 1
            except Exception:
                errors += 1

        total_time = time.time() - start_time
        ops_per_second = operations / total_time
        error_rate = errors / operations if operations > 0 else 0

        # Performance requirements
        assert ops_per_second > 100, f"Operations per second too low: {ops_per_second}"
        assert error_rate < 0.01, f"Error rate too high: {error_rate}"

    def test_memory_usage(self, performance_config_manager):
        """Test memory usage under load."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        cache_manager = CacheManager(performance_config_manager)

        # Fill cache with data
        for i in range(5000):
            cache_manager.set(f"memory_test_key_{i}", f"memory_test_value_{i}" * 100)

        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB)
        assert (
            memory_increase < 100 * 1024 * 1024
        ), f"Memory usage too high: {memory_increase} bytes"

        # Clear cache and check memory cleanup
        cache_manager.clear()

        # Give time for garbage collection
        import gc

        gc.collect()

        final_memory = process.memory_info().rss
        memory_after_clear = final_memory - initial_memory

        # Memory should be mostly reclaimed
        assert (
            memory_after_clear < memory_increase * 0.5
        ), "Memory not properly reclaimed after cache clear"


class TestLoadTesting:
    """Load testing scenarios."""

    @pytest.fixture
    def load_test_config(self):
        """Configuration for load testing."""
        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: {
            "cache.backend": "memory",
            "cache.max_size": 50000,
            "monitoring.enabled": True,
            "web_interface.host": "127.0.0.1",
            "web_interface.port": 5000,
        }.get(key, default)
        return mock_config

    @pytest.mark.slow
    def test_certificate_generation_load(self, load_test_config):
        """Test certificate generation under load."""
        with patch(
            "fm_llm_solver.services.certificate_generator.CertificateGenerator"
        ) as mock_cert_gen:
            # Mock the certificate generator
            mock_instance = Mock()
            mock_cert_gen.return_value = mock_instance

            def mock_generate(problem):
                # Simulate processing time
                time.sleep(0.1)
                return {
                    "certificate": f"certificate_for_{problem}",
                    "status": "success",
                    "processing_time": 0.1,
                }

            mock_instance.generate_certificate.side_effect = mock_generate

            cert_generator = CertificateGenerator(load_test_config)

            # Test concurrent certificate generation
            def generate_certificate(problem_id):
                return cert_generator.generate_certificate(f"problem_{problem_id}")

            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(generate_certificate, i) for i in range(20)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]

            total_time = time.time() - start_time

            # All generations should succeed
            assert all(result["status"] == "success" for result in results)

            # Should complete within reasonable time (with concurrency)
            assert total_time < 10.0, f"Load test took too long: {total_time}s"

    @pytest.mark.slow
    @pytest.mark.skipif(not pytest.importorskip("requests"), reason="requests not available")
    def test_web_interface_load(self, load_test_config):
        """Test web interface under load."""
        import requests
        from threading import Thread
        import queue

        # This test assumes the web interface is running
        # In a real scenario, you'd start the app in test mode

        base_url = "http://127.0.0.1:5000"
        results_queue = queue.Queue()

        def make_request(request_id):
            """Make a single request to the web interface."""
            try:
                start_time = time.time()
                response = requests.get(f"{base_url}/health", timeout=5)
                duration = time.time() - start_time

                results_queue.put(
                    {
                        "request_id": request_id,
                        "status_code": response.status_code,
                        "duration": duration,
                        "success": response.status_code == 200,
                    }
                )
            except Exception as e:
                results_queue.put({"request_id": request_id, "error": str(e), "success": False})

        # Skip if web interface is not running
        try:
            requests.get(f"{base_url}/health", timeout=1)
        except requests.exceptions.RequestException:
            pytest.skip("Web interface not running")

        # Create multiple threads to make concurrent requests
        threads = []
        num_requests = 50

        start_time = time.time()
        for i in range(num_requests):
            thread = Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all requests to complete
        for thread in threads:
            thread.join()

        total_time = time.time() - start_time

        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        # Analyze results
        successful_requests = [r for r in results if r.get("success", False)]
        failed_requests = [r for r in results if not r.get("success", False)]

        success_rate = len(successful_requests) / len(results)
        avg_response_time = (
            sum(r["duration"] for r in successful_requests) / len(successful_requests)
            if successful_requests
            else 0
        )

        # Performance assertions
        assert success_rate > 0.95, f"Success rate too low: {success_rate}"
        assert avg_response_time < 1.0, f"Average response time too high: {avg_response_time}s"
        assert total_time < 20.0, f"Load test took too long: {total_time}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
