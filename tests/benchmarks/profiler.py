"""
Performance Profiler for Barrier Certificate Validation (Phase 1 Day 8)
Analyzes performance bottlenecks and generates optimization recommendations
"""

import time
import cProfile
import pstats
import json
import os
import sys
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import tracemalloc
import gc

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.level_set_tracker import BarrierCertificateValidator
from evaluation.verify_certificate import verify_barrier_certificate
from omegaconf import DictConfig


@dataclass
class ProfileResult:
    """Results from a single profiling run"""

    test_name: str
    total_time: float
    memory_peak: float
    memory_allocated: float
    function_times: Dict[str, float]
    call_counts: Dict[str, int]
    bottlenecks: List[str]
    recommendations: List[str]


class BarrierCertificateProfiler:
    """Performance profiler for barrier certificate validation"""

    def __init__(self, output_dir: str = "profiling_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []

    def profile_validation(
        self,
        certificate: str,
        system_info: Dict[str, Any],
        config: DictConfig,
        test_name: str = "unnamed",
    ) -> ProfileResult:
        """Profile a single validation run"""

        # Start memory tracking
        tracemalloc.start()
        gc.collect()

        # Create profiler
        profiler = cProfile.Profile()

        # Run validation with profiling
        start_time = time.time()
        profiler.enable()

        try:
            validator = BarrierCertificateValidator(certificate, system_info, config)
            validator.validate()
        except Exception as e:
            print(f"Error during validation: {e}")

        profiler.disable()
        total_time = time.time() - start_time

        # Get memory stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Analyze profiling results
        stats = pstats.Stats(profiler)

        # Extract function times and call counts
        function_times = {}
        call_counts = {}

        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            func_name = f"{func[0]}:{func[1]}:{func[2]}"
            function_times[func_name] = ct  # cumulative time
            call_counts[func_name] = cc  # call count

        # Identify bottlenecks (top 10 time consumers)
        sorted_funcs = sorted(function_times.items(), key=lambda x: x[1], reverse=True)[:10]
        bottlenecks = [
            f"{name}: {time:.3f}s ({call_counts.get(name, 0)} calls)" for name, time in sorted_funcs
        ]

        # Generate recommendations
        recommendations = self._generate_recommendations(
            function_times, call_counts, total_time, peak
        )

        # Create result
        profile_result = ProfileResult(
            test_name=test_name,
            total_time=total_time,
            memory_peak=peak / 1024 / 1024,  # Convert to MB
            memory_allocated=current / 1024 / 1024,
            function_times=dict(sorted_funcs[:20]),  # Top 20 functions
            call_counts={k: call_counts[k] for k, _ in sorted_funcs[:20]},
            bottlenecks=bottlenecks,
            recommendations=recommendations,
        )

        self.results.append(profile_result)

        # Save detailed stats
        self._save_detailed_stats(stats, test_name)

        return profile_result

    def _generate_recommendations(
        self,
        function_times: Dict[str, float],
        call_counts: Dict[str, int],
        total_time: float,
        memory_peak: int,
    ) -> List[str]:
        """Generate optimization recommendations based on profiling data"""
        recommendations = []

        # Check for expensive functions
        for func, time in function_times.items():
            if time > 0.5 * total_time:  # Function takes >50% of total time
                recommendations.append(
                    f"CRITICAL: Function '{func}' consumes {100*time/total_time:.1f}% "
                    "of total execution time. Consider optimization."
                )

        # Check for frequently called functions
        for func, count in call_counts.items():
            if count > 10000:
                time_per_call = function_times.get(func, 0) / count if count > 0 else 0
                if time_per_call > 0.0001:  # More than 0.1ms per call
                    recommendations.append(
                        f"Function '{func}' called {count} times with "
                        f"{time_per_call*1000:.3f}ms per call. Consider caching or vectorization."
                    )

        # Memory recommendations
        if memory_peak > 500:  # More than 500MB
            recommendations.append(
                f"High memory usage detected: {memory_peak:.1f}MB peak. "
                "Consider reducing sample sizes or using memory-efficient data structures."
            )

        # Sampling recommendations
        if "generate_samples" in str(function_times):
            sample_time = sum(t for f, t in function_times.items() if "generate_samples" in f)
            if sample_time > 0.2 * total_time:
                recommendations.append(
                    "Sampling takes significant time. Consider using quasi-random sequences "
                    "or adaptive sampling strategies."
                )

        # Symbolic computation recommendations
        if "sympy" in str(function_times):
            sympy_time = sum(t for f, t in function_times.items() if "sympy" in f)
            if sympy_time > 0.3 * total_time:
                recommendations.append(
                    "SymPy operations are expensive. Consider caching symbolic computations "
                    "or using numerical approximations where appropriate."
                )

        return recommendations

    def _save_detailed_stats(self, stats: pstats.Stats, test_name: str):
        """Save detailed profiling statistics"""
        output_file = os.path.join(self.output_dir, f"{test_name}_detailed.txt")

        with open(output_file, "w") as f:
            # Redirect stats output to file
            stats.stream = f
            f.write(f"Detailed Profiling Results for {test_name}\n")
            f.write("=" * 80 + "\n\n")

            stats.sort_stats("cumulative")
            stats.print_stats(50)  # Top 50 functions

            f.write("\n\nCallers Information:\n")
            f.write("=" * 80 + "\n")
            stats.print_callers(20)  # Top 20 callers

    def run_benchmark_suite(self):
        """Run a comprehensive benchmark suite"""
        print("Running barrier certificate validation benchmarks...")

        # Define test cases with varying complexity
        test_cases = [
            {
                "name": "simple_2d",
                "certificate": "x**2 + y**2 - 1.0",
                "system": {
                    "variables": ["x", "y"],
                    "dynamics": ["-x", "-y"],
                    "initial_set_conditions": ["x**2 + y**2 <= 0.25"],
                    "unsafe_set_conditions": ["x**2 + y**2 >= 4.0"],
                    "safe_set_conditions": [],
                    "sampling_bounds": {"x": (-3, 3), "y": (-3, 3)},
                },
                "samples": [1000, 5000, 10000],
            },
            {
                "name": "complex_3d",
                "certificate": "x**2 + y**2 + z**2 + 0.1*x*y - 2.0",
                "system": {
                    "variables": ["x", "y", "z"],
                    "dynamics": ["-x + 0.1*y", "-y - 0.1*x", "-z"],
                    "initial_set_conditions": ["x**2 + y**2 + z**2 <= 0.5"],
                    "unsafe_set_conditions": ["x**2 + y**2 + z**2 >= 9.0"],
                    "safe_set_conditions": [],
                    "sampling_bounds": {"x": (-4, 4), "y": (-4, 4), "z": (-4, 4)},
                },
                "samples": [1000, 5000, 10000],
            },
            {
                "name": "polynomial_4th",
                "certificate": "x**4 + y**4 + x**2 + y**2 - 3.0",
                "system": {
                    "variables": ["x", "y"],
                    "dynamics": ["-x - x**3", "-y - y**3"],
                    "initial_set_conditions": ["x**2 + y**2 <= 0.25"],
                    "unsafe_set_conditions": ["x**2 + y**2 >= 4.0"],
                    "safe_set_conditions": [],
                    "sampling_bounds": {"x": (-3, 3), "y": (-3, 3)},
                },
                "samples": [1000, 5000],
            },
        ]

        # Run benchmarks
        for test_case in test_cases:
            for num_samples in test_case["samples"]:
                config = DictConfig(
                    {
                        "numerical_tolerance": 1e-6,
                        "num_samples_boundary": num_samples,
                        "num_samples_lie": num_samples * 2,
                        "optimization_maxiter": 50,
                        "optimization_popsize": 15,
                    }
                )

                test_name = f"{test_case['name']}_s{num_samples}"
                print(f"\nProfiling {test_name}...")

                result = self.profile_validation(
                    test_case["certificate"], test_case["system"], config, test_name
                )

                # Print summary
                print(f"  Total time: {result.total_time:.3f}s")
                print(f"  Memory peak: {result.memory_peak:.1f}MB")
                print(
                    f"  Top bottleneck: {result.bottlenecks[0] if result.bottlenecks else 'None'}"
                )

    def generate_report(self):
        """Generate a comprehensive profiling report"""
        report_file = os.path.join(self.output_dir, "profiling_report.json")

        # Convert results to JSON-serializable format
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "results": [asdict(r) for r in self.results],
            "summary": self._generate_summary(),
        }

        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)

        print(f"\nProfiling report saved to: {report_file}")

        # Also generate human-readable report
        self._generate_text_report()

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not self.results:
            return {}

        times = [r.total_time for r in self.results]
        memories = [r.memory_peak for r in self.results]

        return {
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "avg_memory": sum(memories) / len(memories),
            "max_memory": max(memories),
            "common_bottlenecks": self._find_common_bottlenecks(),
        }

    def _find_common_bottlenecks(self) -> List[str]:
        """Find functions that are bottlenecks across multiple tests"""
        bottleneck_counts = {}

        for result in self.results:
            for bottleneck in result.bottlenecks[:5]:  # Top 5 from each test
                func_name = bottleneck.split(":")[0]
                bottleneck_counts[func_name] = bottleneck_counts.get(func_name, 0) + 1

        # Return functions that appear in multiple tests
        common = [(name, count) for name, count in bottleneck_counts.items() if count > 1]
        common.sort(key=lambda x: x[1], reverse=True)

        return [f"{name} (appears in {count} tests)" for name, count in common[:5]]

    def _generate_text_report(self):
        """Generate human-readable text report"""
        report_file = os.path.join(self.output_dir, "profiling_summary.txt")

        with open(report_file, "w") as f:
            f.write("BARRIER CERTIFICATE VALIDATION PROFILING REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Tests: {len(self.results)}\n\n")

            # Summary statistics
            summary = self._generate_summary()
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average execution time: {summary.get('avg_time', 0):.3f}s\n")
            f.write(
                f"Min/Max time: {summary.get('min_time', 0):.3f}s / {summary.get('max_time', 0):.3f}s\n"
            )
            f.write(f"Average memory usage: {summary.get('avg_memory', 0):.1f}MB\n")
            f.write(f"Peak memory usage: {summary.get('max_memory', 0):.1f}MB\n\n")

            # Common bottlenecks
            f.write("COMMON BOTTLENECKS\n")
            f.write("-" * 40 + "\n")
            for bottleneck in summary.get("common_bottlenecks", []):
                f.write(f"  • {bottleneck}\n")
            f.write("\n")

            # Individual test results
            f.write("INDIVIDUAL TEST RESULTS\n")
            f.write("=" * 80 + "\n\n")

            for result in self.results:
                f.write(f"Test: {result.test_name}\n")
                f.write(f"Time: {result.total_time:.3f}s | Memory: {result.memory_peak:.1f}MB\n")
                f.write("Top bottlenecks:\n")
                for i, bottleneck in enumerate(result.bottlenecks[:3]):
                    f.write(f"  {i+1}. {bottleneck}\n")
                f.write("Recommendations:\n")
                for rec in result.recommendations[:3]:
                    f.write(f"  • {rec}\n")
                f.write("-" * 80 + "\n\n")

        print(f"Summary report saved to: {report_file}")


def compare_validators():
    """Compare performance of different validators"""
    profiler = BarrierCertificateProfiler(output_dir="validator_comparison")

    # Test case
    certificate = "x**2 + y**2 - 1.0"
    system_info = {
        "variables": ["x", "y"],
        "dynamics": ["-x", "-y"],
        "initial_set_conditions": ["x**2 + y**2 <= 0.25"],
        "unsafe_set_conditions": ["x**2 + y**2 >= 4.0"],
        "safe_set_conditions": [],
        "sampling_bounds": {"x": (-3, 3), "y": (-3, 3)},
    }
    config = DictConfig(
        {
            "numerical_tolerance": 1e-6,
            "num_samples_boundary": 5000,
            "num_samples_lie": 10000,
            "optimization_maxiter": 100,
            "optimization_popsize": 30,
        }
    )

    print("Comparing validator implementations...")

    # Profile new validator
    print("\nProfiling new BarrierCertificateValidator...")
    new_result = profiler.profile_validation(certificate, system_info, config, "new_validator")

    # Profile old validator (if available)
    print("\nProfiling old verify_barrier_certificate...")
    old_profiler = cProfile.Profile()
    old_start = time.time()
    old_profiler.enable()

    try:
        verify_barrier_certificate(certificate, system_info, config)
    except Exception as e:
        print(f"Error in old validator: {e}")

    old_profiler.disable()
    old_time = time.time() - old_start

    # Compare results
    print("\nCOMPARISON RESULTS")
    print("=" * 50)
    print(f"New validator time: {new_result.total_time:.3f}s")
    print(f"Old validator time: {old_time:.3f}s")
    print(f"Speedup: {old_time/new_result.total_time:.2f}x" if new_result.total_time > 0 else "N/A")
    print(f"Memory difference: {new_result.memory_peak:.1f}MB (new)")

    profiler.generate_report()


def main():
    """Main entry point for profiler"""
    import argparse

    parser = argparse.ArgumentParser(description="Profile barrier certificate validation")
    parser.add_argument("--benchmark", action="store_true", help="Run full benchmark suite")
    parser.add_argument(
        "--compare", action="store_true", help="Compare different validator implementations"
    )
    parser.add_argument(
        "--output", default="profiling_results", help="Output directory for results"
    )

    args = parser.parse_args()

    if args.compare:
        compare_validators()
    else:
        profiler = BarrierCertificateProfiler(output_dir=args.output)

        if args.benchmark:
            profiler.run_benchmark_suite()
        else:
            # Run a single test
            print("Running single profiling test...")
            result = profiler.profile_validation(
                "x**2 + y**2 - 1.0",
                {
                    "variables": ["x", "y"],
                    "dynamics": ["-x", "-y"],
                    "initial_set_conditions": ["x**2 + y**2 <= 0.25"],
                    "unsafe_set_conditions": ["x**2 + y**2 >= 4.0"],
                    "safe_set_conditions": [],
                    "sampling_bounds": {"x": (-3, 3), "y": (-3, 3)},
                },
                DictConfig(
                    {
                        "numerical_tolerance": 1e-6,
                        "num_samples_boundary": 5000,
                        "num_samples_lie": 10000,
                        "optimization_maxiter": 100,
                        "optimization_popsize": 30,
                    }
                ),
                "single_test",
            )

            print(f"\nTotal time: {result.total_time:.3f}s")
            print(f"Memory peak: {result.memory_peak:.1f}MB")
            print("\nTop bottlenecks:")
            for i, bottleneck in enumerate(result.bottlenecks[:5]):
                print(f"  {i+1}. {bottleneck}")

        profiler.generate_report()


if __name__ == "__main__":
    main()
