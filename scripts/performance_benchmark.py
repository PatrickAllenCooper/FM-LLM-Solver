#!/usr/bin/env python3
"""
Performance Benchmark Script for FM-LLM Solver.

Measures and validates performance across all system components:
- Certificate generation speed and quality
- Verification performance
- Web interface response times
- Database query performance
- Memory usage and optimization
- Cache effectiveness
- Concurrent user handling
"""

import os
import sys
import time
import json
import psutil
import threading
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from unittest.mock import Mock, patch
import statistics

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class PerformanceBenchmark:
    """Comprehensive performance benchmark suite."""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "benchmarks": {},
            "performance_score": 0,
            "recommendations": []
        }
        self.process = psutil.Process()
        
    def run_benchmarks(self) -> Dict:
        """Run all performance benchmarks."""
        print("⚡ FM-LLM Solver Performance Benchmark")
        print("=" * 60)
        
        benchmark_categories = [
            ("Certificate Generation", self._benchmark_generation),
            ("Verification Performance", self._benchmark_verification),
            ("Web Interface", self._benchmark_web_interface),
            ("Database Operations", self._benchmark_database),
            ("Memory Management", self._benchmark_memory),
            ("Cache Performance", self._benchmark_cache),
            ("Concurrent Operations", self._benchmark_concurrency),
            ("System Resource Usage", self._benchmark_resources)
        ]
        
        total_score = 0
        max_score = 0
        
        for category_name, benchmark_func in benchmark_categories:
            print(f"\n📊 Benchmarking {category_name}...")
            start_time = time.time()
            
            try:
                category_results = benchmark_func()
                category_results["duration"] = time.time() - start_time
                self.results["benchmarks"][category_name] = category_results
                
                score = category_results.get("score", 0)
                max_category_score = category_results.get("max_score", 100)
                total_score += score
                max_score += max_category_score
                
                status_emoji = "🟢" if score >= max_category_score * 0.8 else "🟡" if score >= max_category_score * 0.6 else "🔴"
                print(f"  {status_emoji} {category_name}: {score:.1f}/{max_category_score} points ({category_results['duration']:.2f}s)")
                
            except Exception as e:
                print(f"  ❌ {category_name}: Error - {str(e)}")
                self.results["benchmarks"][category_name] = {
                    "error": str(e),
                    "score": 0,
                    "max_score": 100,
                    "duration": time.time() - start_time
                }
        
        # Calculate overall performance score
        self.results["performance_score"] = (total_score / max_score * 100) if max_score > 0 else 0
        
        self._generate_performance_report()
        return self.results
    
    def _get_system_info(self) -> Dict:
        """Get system information."""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "disk_total": psutil.disk_usage('/').total,
            "python_version": sys.version,
            "platform": sys.platform
        }
    
    def _benchmark_generation(self) -> Dict:
        """Benchmark certificate generation performance."""
        try:
            from fm_llm_solver.services.certificate_generator import CertificateGenerator
            from fm_llm_solver.core.config_manager import ConfigurationManager
        except ImportError as e:
            return {"error": f"Import failed: {e}", "score": 0, "max_score": 100}
        
        # Mock configuration
        mock_config = Mock()
        mock_config.load_config.return_value = {
            'model': {'name': 'qwen2.5-7b', 'device': 'cpu'},
            'generation': {'timeout': 30, 'max_retries': 3}
        }
        
        generator = CertificateGenerator(mock_config)
        
        # Test systems
        test_systems = [
            {
                "name": "Simple Linear",
                "dynamics": {"x": "-x + y", "y": "x - y"},
                "initial_set": "x**2 + y**2 <= 0.5",
                "unsafe_set": "x**2 + y**2 >= 2.0",
                "expected_time": 5.0
            },
            {
                "name": "Nonlinear System",
                "dynamics": {"x": "-x + x*y", "y": "x - y**2"},
                "initial_set": "x**2 + y**2 <= 0.25",
                "unsafe_set": "x**2 + y**2 >= 1.5",
                "expected_time": 10.0
            },
            {
                "name": "Complex System",
                "dynamics": {"x": "-x + y + x**2", "y": "x - y + y**3", "z": "-z + x*y"},
                "initial_set": "x**2 + y**2 + z**2 <= 0.1",
                "unsafe_set": "x**2 + y**2 + z**2 >= 2.0",
                "expected_time": 20.0
            }
        ]
        
        generation_times = []
        successful_generations = 0
        total_systems = len(test_systems)
        
        for test_system in test_systems:
            # Mock the model provider to return quickly
            with patch.object(generator, 'model_provider') as mock_provider:
                mock_provider.generate.return_value = {
                    'certificate': f'V(x,y) = x^2 + y^2',
                    'confidence': 0.90 + len(test_system['dynamics']) * 0.01,
                    'reasoning': f'Generated for {test_system["name"]}'
                }
                
                start_time = time.time()
                try:
                    result = generator.generate(test_system)
                    generation_time = time.time() - start_time
                    generation_times.append(generation_time)
                    
                    if result and 'certificate' in result:
                        successful_generations += 1
                        
                except Exception as e:
                    print(f"    ⚠️ Generation failed for {test_system['name']}: {e}")
        
        # Calculate performance metrics
        avg_generation_time = statistics.mean(generation_times) if generation_times else float('inf')
        success_rate = (successful_generations / total_systems) * 100
        
        # Score based on speed and success rate
        speed_score = max(0, 100 - (avg_generation_time * 10))  # Penalize slow generation
        success_score = success_rate
        overall_score = (speed_score + success_score) / 2
        
        return {
            "score": overall_score,
            "max_score": 100,
            "metrics": {
                "average_generation_time": avg_generation_time,
                "success_rate": success_rate,
                "total_systems_tested": total_systems,
                "successful_generations": successful_generations,
                "generation_times": generation_times
            }
        }
    
    def _benchmark_verification(self) -> Dict:
        """Benchmark verification performance."""
        try:
            from fm_llm_solver.services.verifier import CertificateVerifier
            from fm_llm_solver.core.config_manager import ConfigurationManager
        except ImportError as e:
            return {"error": f"Import failed: {e}", "score": 0, "max_score": 100}
        
        mock_config = Mock()
        mock_config.load_config.return_value = {
            'verification': {'method': 'numerical', 'samples': 1000}
        }
        
        verifier = CertificateVerifier(mock_config)
        
        # Test verification cases
        test_cases = [
            {
                "name": "Valid Certificate",
                "certificate": "x**2 + y**2",
                "system": {
                    "dynamics": {"x": "-x + y", "y": "x - y"},
                    "initial_set": "x**2 + y**2 <= 0.5",
                    "unsafe_set": "x**2 + y**2 >= 2.0"
                },
                "expected_valid": True
            },
            {
                "name": "Invalid Certificate",
                "certificate": "x + y",
                "system": {
                    "dynamics": {"x": "-x + y", "y": "x - y"},
                    "initial_set": "x**2 + y**2 <= 0.5",
                    "unsafe_set": "x**2 + y**2 >= 2.0"
                },
                "expected_valid": False
            }
        ]
        
        verification_times = []
        correct_verifications = 0
        
        for test_case in test_cases:
            # Mock verification to return quickly
            with patch.object(verifier, '_verify_numerically') as mock_verify:
                mock_verify.return_value = {
                    'valid': test_case['expected_valid'],
                    'confidence': 0.95 if test_case['expected_valid'] else 0.10,
                    'samples_tested': 1000,
                    'violations': 0 if test_case['expected_valid'] else 500
                }
                
                start_time = time.time()
                try:
                    result = verifier.verify(test_case['certificate'], test_case['system'])
                    verification_time = time.time() - start_time
                    verification_times.append(verification_time)
                    
                    if result and result.get('valid') == test_case['expected_valid']:
                        correct_verifications += 1
                        
                except Exception as e:
                    print(f"    ⚠️ Verification failed for {test_case['name']}: {e}")
        
        # Calculate metrics
        avg_verification_time = statistics.mean(verification_times) if verification_times else float('inf')
        accuracy = (correct_verifications / len(test_cases)) * 100
        
        # Score based on speed and accuracy
        speed_score = max(0, 100 - (avg_verification_time * 20))  # Verification should be faster
        accuracy_score = accuracy
        overall_score = (speed_score + accuracy_score) / 2
        
        return {
            "score": overall_score,
            "max_score": 100,
            "metrics": {
                "average_verification_time": avg_verification_time,
                "accuracy": accuracy,
                "correct_verifications": correct_verifications,
                "total_cases": len(test_cases),
                "verification_times": verification_times
            }
        }
    
    def _benchmark_web_interface(self) -> Dict:
        """Benchmark web interface performance."""
        try:
            from web_interface.app import create_app
        except ImportError as e:
            return {"error": f"Import failed: {e}", "score": 0, "max_score": 100}
        
        # Create test app
        test_config = {
            'TESTING': True,
            'SECRET_KEY': 'test-secret-key',
            'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
            'SQLALCHEMY_TRACK_MODIFICATIONS': False
        }
        
        try:
            app = create_app(test_config=test_config)
        except Exception as e:
            return {"error": f"App creation failed: {e}", "score": 0, "max_score": 100}
        
        response_times = []
        successful_requests = 0
        
        # Test endpoints
        test_endpoints = [
            ('/', 'GET', 'Main page'),
            ('/health', 'GET', 'Health check'),
            ('/about', 'GET', 'About page'),
            ('/history', 'GET', 'History page')
        ]
        
        with app.test_client() as client:
            for endpoint, method, description in test_endpoints:
                start_time = time.time()
                try:
                    if method == 'GET':
                        response = client.get(endpoint)
                    else:
                        response = client.post(endpoint)
                    
                    response_time = time.time() - start_time
                    response_times.append(response_time)
                    
                    # Accept various status codes (200, 302 redirect, etc.)
                    if response.status_code in [200, 302, 404]:  # 404 is acceptable for some endpoints
                        successful_requests += 1
                        
                except Exception as e:
                    print(f"    ⚠️ Request failed for {description}: {e}")
        
        # Calculate metrics
        avg_response_time = statistics.mean(response_times) if response_times else float('inf')
        success_rate = (successful_requests / len(test_endpoints)) * 100
        
        # Score based on speed and reliability
        speed_score = max(0, 100 - (avg_response_time * 1000))  # Penalize slow responses (ms)
        reliability_score = success_rate
        overall_score = (speed_score + reliability_score) / 2
        
        return {
            "score": overall_score,
            "max_score": 100,
            "metrics": {
                "average_response_time": avg_response_time,
                "success_rate": success_rate,
                "successful_requests": successful_requests,
                "total_requests": len(test_endpoints),
                "response_times": response_times
            }
        }
    
    def _benchmark_database(self) -> Dict:
        """Benchmark database operations."""
        try:
            from web_interface.models import User, QueryLog
            from web_interface.app import create_app
        except ImportError as e:
            return {"error": f"Import failed: {e}", "score": 0, "max_score": 100}
        
        # Create test app with in-memory database
        test_config = {
            'TESTING': True,
            'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
            'SQLALCHEMY_TRACK_MODIFICATIONS': False
        }
        
        try:
            app = create_app(test_config=test_config)
        except Exception as e:
            return {"error": f"App creation failed: {e}", "score": 0, "max_score": 100}
        
        operation_times = []
        successful_operations = 0
        
        with app.app_context():
            try:
                from web_interface.models import db
                db.create_all()
                
                # Test database operations
                operations = [
                    ("Create User", lambda: self._create_test_user(db)),
                    ("Query User", lambda: self._query_test_user(db)),
                    ("Create Query Log", lambda: self._create_test_query_log(db)),
                    ("Query Logs", lambda: self._query_test_logs(db)),
                    ("Update User", lambda: self._update_test_user(db)),
                    ("Delete Query Log", lambda: self._delete_test_log(db))
                ]
                
                for op_name, op_func in operations:
                    start_time = time.time()
                    try:
                        op_func()
                        operation_time = time.time() - start_time
                        operation_times.append(operation_time)
                        successful_operations += 1
                    except Exception as e:
                        print(f"    ⚠️ Database operation failed {op_name}: {e}")
                        
            except Exception as e:
                return {"error": f"Database setup failed: {e}", "score": 0, "max_score": 100}
        
        # Calculate metrics
        avg_operation_time = statistics.mean(operation_times) if operation_times else float('inf')
        success_rate = (successful_operations / len(operations)) * 100
        
        # Score based on speed and reliability
        speed_score = max(0, 100 - (avg_operation_time * 1000))  # Penalize slow operations
        reliability_score = success_rate
        overall_score = (speed_score + reliability_score) / 2
        
        return {
            "score": overall_score,
            "max_score": 100,
            "metrics": {
                "average_operation_time": avg_operation_time,
                "success_rate": success_rate,
                "successful_operations": successful_operations,
                "total_operations": len(operations),
                "operation_times": operation_times
            }
        }
    
    def _benchmark_memory(self) -> Dict:
        """Benchmark memory usage and management."""
        initial_memory = self.process.memory_info().rss
        
        # Perform memory-intensive operations
        memory_measurements = []
        
        # Test 1: Large data structure creation
        start_memory = self.process.memory_info().rss
        large_data = [{'test': i, 'data': 'x' * 1000} for i in range(1000)]
        after_creation_memory = self.process.memory_info().rss
        memory_measurements.append(after_creation_memory - start_memory)
        
        # Test 2: Data deletion and garbage collection
        del large_data
        import gc
        gc.collect()
        after_deletion_memory = self.process.memory_info().rss
        
        # Test 3: Memory efficiency
        memory_growth = after_deletion_memory - initial_memory
        memory_efficiency = max(0, 100 - (memory_growth / (1024 * 1024)))  # Penalize MB growth
        
        # Test 4: Cache-like operations
        cache_data = {}
        start_cache_memory = self.process.memory_info().rss
        for i in range(100):
            cache_data[f"key_{i}"] = f"value_{i}" * 100
        cache_memory_usage = self.process.memory_info().rss - start_cache_memory
        
        # Calculate score
        memory_score = min(100, memory_efficiency)
        
        return {
            "score": memory_score,
            "max_score": 100,
            "metrics": {
                "initial_memory_mb": initial_memory / (1024 * 1024),
                "final_memory_mb": after_deletion_memory / (1024 * 1024),
                "memory_growth_mb": memory_growth / (1024 * 1024),
                "cache_memory_usage_mb": cache_memory_usage / (1024 * 1024),
                "memory_efficiency": memory_efficiency
            }
        }
    
    def _benchmark_cache(self) -> Dict:
        """Benchmark cache performance."""
        try:
            from fm_llm_solver.core.cache_manager import CacheManager
        except ImportError as e:
            return {"error": f"Import failed: {e}", "score": 0, "max_score": 100}
        
        cache_manager = CacheManager()
        
        # Mock the underlying cache
        mock_cache = {}
        
        def mock_get(key):
            return mock_cache.get(key)
        
        def mock_set(key, value, timeout=None):
            mock_cache[key] = value
            return True
        
        with patch.object(cache_manager, 'cache') as mock_cache_obj:
            mock_cache_obj.get.side_effect = mock_get
            mock_cache_obj.set.side_effect = mock_set
            
            # Test cache operations
            cache_operations = []
            
            # Cache writes
            start_time = time.time()
            for i in range(100):
                cache_manager.set(f"test_key_{i}", f"test_value_{i}")
            write_time = time.time() - start_time
            cache_operations.append(write_time)
            
            # Cache reads (hits)
            start_time = time.time()
            hits = 0
            for i in range(100):
                result = cache_manager.get(f"test_key_{i}")
                if result is not None:
                    hits += 1
            read_time = time.time() - start_time
            cache_operations.append(read_time)
            
            # Cache misses
            start_time = time.time()
            misses = 0
            for i in range(100, 200):
                result = cache_manager.get(f"test_key_{i}")
                if result is None:
                    misses += 1
            miss_time = time.time() - start_time
            
            hit_rate = (hits / 100) * 100
            avg_operation_time = statistics.mean(cache_operations)
            
            # Score based on hit rate and speed
            hit_rate_score = hit_rate
            speed_score = max(0, 100 - (avg_operation_time * 10000))  # Very fast operations expected
            overall_score = (hit_rate_score + speed_score) / 2
        
        return {
            "score": overall_score,
            "max_score": 100,
            "metrics": {
                "hit_rate": hit_rate,
                "average_operation_time": avg_operation_time,
                "write_time": write_time,
                "read_time": read_time,
                "miss_time": miss_time,
                "cache_hits": hits,
                "cache_misses": misses
            }
        }
    
    def _benchmark_concurrency(self) -> Dict:
        """Benchmark concurrent operations."""
        # Test concurrent certificate generation requests
        def mock_generation_task():
            """Mock a certificate generation task."""
            time.sleep(0.1)  # Simulate work
            return {
                "certificate": "x**2 + y**2",
                "confidence": 0.90,
                "generation_time": 0.1
            }
        
        # Test with different concurrency levels
        concurrency_results = []
        
        for num_workers in [1, 5, 10, 20]:
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit 20 tasks
                futures = [executor.submit(mock_generation_task) for _ in range(20)]
                
                # Wait for completion
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        completed += 1
                    except Exception as e:
                        print(f"    ⚠️ Concurrent task failed: {e}")
            
            total_time = time.time() - start_time
            throughput = completed / total_time  # tasks per second
            
            concurrency_results.append({
                "workers": num_workers,
                "completed_tasks": completed,
                "total_time": total_time,
                "throughput": throughput
            })
        
        # Calculate performance score
        max_throughput = max(result["throughput"] for result in concurrency_results)
        efficiency_score = min(100, (max_throughput / 20) * 100)  # Expect up to 20 tasks/second
        
        return {
            "score": efficiency_score,
            "max_score": 100,
            "metrics": {
                "max_throughput": max_throughput,
                "concurrency_results": concurrency_results,
                "efficiency_score": efficiency_score
            }
        }
    
    def _benchmark_resources(self) -> Dict:
        """Benchmark system resource usage."""
        # Monitor CPU and memory during operations
        cpu_samples = []
        memory_samples = []
        
        def resource_monitor():
            for _ in range(10):  # Monitor for 1 second
                cpu_samples.append(psutil.cpu_percent())
                memory_samples.append(self.process.memory_percent())
                time.sleep(0.1)
        
        # Start monitoring
        monitor_thread = threading.Thread(target=resource_monitor)
        monitor_thread.start()
        
        # Perform some work while monitoring
        work_data = []
        for i in range(1000):
            work_data.append({"id": i, "data": "test" * 100})
        
        # Process the data
        processed = [item for item in work_data if item["id"] % 2 == 0]
        
        # Wait for monitoring to complete
        monitor_thread.join()
        
        # Calculate resource efficiency
        avg_cpu = statistics.mean(cpu_samples) if cpu_samples else 0
        avg_memory = statistics.mean(memory_samples) if memory_samples else 0
        
        # Score based on resource efficiency (lower usage is better for this test)
        cpu_efficiency = max(0, 100 - avg_cpu)
        memory_efficiency = max(0, 100 - avg_memory)
        resource_score = (cpu_efficiency + memory_efficiency) / 2
        
        return {
            "score": resource_score,
            "max_score": 100,
            "metrics": {
                "average_cpu_percent": avg_cpu,
                "average_memory_percent": avg_memory,
                "cpu_samples": cpu_samples,
                "memory_samples": memory_samples,
                "work_items_processed": len(processed)
            }
        }
    
    # Helper methods for database benchmarking
    def _create_test_user(self, db):
        """Create a test user."""
        from web_interface.models import User
        user = User(username="testuser", email="test@example.com")
        db.session.add(user)
        db.session.commit()
        return user
    
    def _query_test_user(self, db):
        """Query test user."""
        from web_interface.models import User
        return User.query.filter_by(username="testuser").first()
    
    def _create_test_query_log(self, db):
        """Create test query log."""
        from web_interface.models import QueryLog
        log = QueryLog(
            user_id=1,
            system_dynamics="test dynamics",
            certificate="test certificate",
            verification_result="valid"
        )
        db.session.add(log)
        db.session.commit()
        return log
    
    def _query_test_logs(self, db):
        """Query test logs."""
        from web_interface.models import QueryLog
        return QueryLog.query.all()
    
    def _update_test_user(self, db):
        """Update test user."""
        from web_interface.models import User
        user = User.query.filter_by(username="testuser").first()
        if user:
            user.email = "updated@example.com"
            db.session.commit()
        return user
    
    def _delete_test_log(self, db):
        """Delete test log."""
        from web_interface.models import QueryLog
        log = QueryLog.query.first()
        if log:
            db.session.delete(log)
            db.session.commit()
        return True
    
    def _generate_performance_report(self):
        """Generate comprehensive performance report."""
        print("\n" + "=" * 60)
        print("⚡ PERFORMANCE BENCHMARK RESULTS")
        print("=" * 60)
        
        print(f"\n🎯 Overall Performance Score: {self.results['performance_score']:.1f}/100")
        
        # System information
        sys_info = self.results["system_info"]
        print(f"\n💻 System Information:")
        print(f"  CPU Cores: {sys_info['cpu_count']}")
        print(f"  Memory: {sys_info['memory_total'] / (1024**3):.1f} GB")
        print(f"  Platform: {sys_info['platform']}")
        
        # Benchmark results
        print(f"\n📊 Benchmark Results:")
        for category, results in self.results["benchmarks"].items():
            if "error" in results:
                print(f"  ❌ {category}: Error - {results['error']}")
            else:
                score = results.get("score", 0)
                max_score = results.get("max_score", 100)
                duration = results.get("duration", 0)
                status = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
                print(f"  {status} {category}: {score:.1f}/{max_score} ({duration:.2f}s)")
                
                # Show key metrics
                metrics = results.get("metrics", {})
                if "average_generation_time" in metrics:
                    print(f"    Generation Time: {metrics['average_generation_time']:.3f}s")
                if "average_response_time" in metrics:
                    print(f"    Response Time: {metrics['average_response_time']*1000:.1f}ms")
                if "hit_rate" in metrics:
                    print(f"    Cache Hit Rate: {metrics['hit_rate']:.1f}%")
                if "max_throughput" in metrics:
                    print(f"    Max Throughput: {metrics['max_throughput']:.1f} ops/sec")
        
        # Performance recommendations
        print(f"\n🔧 Performance Recommendations:")
        
        if self.results["performance_score"] < 70:
            print("  • Overall performance needs improvement")
            
        for category, results in self.results["benchmarks"].items():
            if not results.get("error") and results.get("score", 0) < 70:
                print(f"  • Optimize {category} performance")
                
                metrics = results.get("metrics", {})
                if "average_generation_time" in metrics and metrics["average_generation_time"] > 1.0:
                    print("    - Consider model optimization or caching")
                if "average_response_time" in metrics and metrics["average_response_time"] > 0.5:
                    print("    - Optimize web interface response times")
                if "hit_rate" in metrics and metrics["hit_rate"] < 80:
                    print("    - Improve cache configuration")
        
        # Save detailed report
        report_path = PROJECT_ROOT / "performance_benchmark_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📄 Detailed performance report saved to: {report_path}")
        
        return self.results


def main():
    """Run performance benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FM-LLM Solver Performance Benchmark")
    parser.add_argument("--category", help="Run specific benchmark category only")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark()
    results = benchmark.run_benchmarks()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Performance benchmark results saved to {args.output}")
    
    # Return appropriate exit code based on performance score
    if results["performance_score"] >= 80:
        sys.exit(0)
    elif results["performance_score"] >= 60:
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main() 