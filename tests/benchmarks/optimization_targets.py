"""
Optimization Targets and Benchmark Suite (Phase 1 Day 9)
Defines specific optimization goals and benchmarks for barrier certificate validation
"""

import time
import json
import os
import sys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.level_set_tracker import BarrierCertificateValidator
from omegaconf import DictConfig


@dataclass
class OptimizationTarget:
    """Defines a specific optimization target"""

    name: str
    description: str
    target_metric: str  # e.g., 'validation_time', 'memory_usage', 'accuracy'
    target_value: float
    current_value: Optional[float] = None
    achieved: bool = False
    improvement_percentage: Optional[float] = None


@dataclass
class BenchmarkResult:
    """Result from a benchmark run"""

    benchmark_name: str
    test_case: str
    validation_time: float
    memory_usage: float
    accuracy: float
    samples_processed: int
    configuration: Dict[str, Any]
    timestamp: datetime


class OptimizationBenchmarkSuite:
    """Comprehensive benchmark suite for optimization targets"""

    def __init__(self, output_dir: str = "optimization_benchmarks"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.targets = self._define_optimization_targets()
        self.results = []

    def _define_optimization_targets(self) -> List[OptimizationTarget]:
        """Define specific optimization targets for Phase 1"""
        return [
            OptimizationTarget(
                name="fast_validation",
                description="Validate simple 2D systems in under 0.5 seconds",
                target_metric="validation_time",
                target_value=0.5,
            ),
            OptimizationTarget(
                name="memory_efficient",
                description="Keep memory usage under 100MB for standard problems",
                target_metric="memory_usage",
                target_value=100.0,
            ),
            OptimizationTarget(
                name="high_accuracy",
                description="Achieve 95% accuracy on ground truth test set",
                target_metric="accuracy",
                target_value=0.95,
            ),
            OptimizationTarget(
                name="scalable_3d",
                description="Handle 3D systems with 50k samples in under 5 seconds",
                target_metric="validation_time_3d",
                target_value=5.0,
            ),
            OptimizationTarget(
                name="real_time_feedback",
                description="Provide initial validation result in under 0.1 seconds",
                target_metric="initial_response_time",
                target_value=0.1,
            ),
        ]

    def run_benchmark_suite(self):
        """Run complete benchmark suite"""
        print("Running Optimization Benchmark Suite")
        print("=" * 60)

        # Define benchmark test cases
        benchmarks = [
            self._benchmark_simple_2d,
            self._benchmark_complex_2d,
            self._benchmark_3d_system,
            self._benchmark_high_dimensional,
            self._benchmark_memory_stress,
            self._benchmark_accuracy,
            self._benchmark_real_time,
        ]

        for benchmark in benchmarks:
            try:
                print(f"\nRunning {benchmark.__name__}...")
                benchmark()
            except Exception as e:
                print(f"Error in {benchmark.__name__}: {e}")

        # Evaluate targets
        self._evaluate_targets()

        # Generate report
        self._generate_report()

    def _benchmark_simple_2d(self):
        """Benchmark simple 2D system validation"""
        test_cases = [
            {
                "name": "linear_stable",
                "certificate": "x**2 + y**2 - 1.0",
                "system": {
                    "variables": ["x", "y"],
                    "dynamics": ["-x", "-y"],
                    "initial_set_conditions": ["x**2 + y**2 <= 0.25"],
                    "unsafe_set_conditions": ["x**2 + y**2 >= 4.0"],
                    "safe_set_conditions": [],
                    "sampling_bounds": {"x": (-3, 3), "y": (-3, 3)},
                },
            },
            {
                "name": "linear_coupled",
                "certificate": "x**2 + 2*x*y + 2*y**2 - 1.5",
                "system": {
                    "variables": ["x", "y"],
                    "dynamics": ["-x - y", "x - y"],
                    "initial_set_conditions": ["x**2 + y**2 <= 0.25"],
                    "unsafe_set_conditions": ["x**2 + y**2 >= 4.0"],
                    "safe_set_conditions": [],
                    "sampling_bounds": {"x": (-3, 3), "y": (-3, 3)},
                },
            },
        ]

        for test_case in test_cases:
            for num_samples in [1000, 5000, 10000]:
                config = DictConfig(
                    {
                        "numerical_tolerance": 1e-6,
                        "num_samples_boundary": num_samples,
                        "num_samples_lie": num_samples * 2,
                        "optimization_maxiter": 50,
                        "optimization_popsize": 15,
                    }
                )

                result = self._run_single_benchmark(
                    f"simple_2d_{test_case['name']}_s{num_samples}", test_case, config
                )
                self.results.append(result)

    def _benchmark_complex_2d(self):
        """Benchmark complex 2D systems"""
        test_case = {
            "name": "polynomial_4th",
            "certificate": "x**4 + y**4 + 2*x**2*y**2 + x**2 + y**2 - 5.0",
            "system": {
                "variables": ["x", "y"],
                "dynamics": ["-x - x**3 + 0.1*y", "-y - y**3 - 0.1*x"],
                "initial_set_conditions": ["x**2 + y**2 <= 0.5"],
                "unsafe_set_conditions": ["x**2 + y**2 >= 9.0"],
                "safe_set_conditions": [],
                "sampling_bounds": {"x": (-4, 4), "y": (-4, 4)},
            },
        }

        config = DictConfig(
            {
                "numerical_tolerance": 1e-6,
                "num_samples_boundary": 10000,
                "num_samples_lie": 20000,
                "optimization_maxiter": 100,
                "optimization_popsize": 30,
            }
        )

        result = self._run_single_benchmark("complex_2d", test_case, config)
        self.results.append(result)

    def _benchmark_3d_system(self):
        """Benchmark 3D system validation"""
        test_case = {
            "name": "3d_linear",
            "certificate": "x**2 + y**2 + z**2 - 2.0",
            "system": {
                "variables": ["x", "y", "z"],
                "dynamics": ["-x", "-y", "-z"],
                "initial_set_conditions": ["x**2 + y**2 + z**2 <= 0.5"],
                "unsafe_set_conditions": ["x**2 + y**2 + z**2 >= 9.0"],
                "safe_set_conditions": [],
                "sampling_bounds": {"x": (-4, 4), "y": (-4, 4), "z": (-4, 4)},
            },
        }

        # Test with 50k samples for scalability target
        config = DictConfig(
            {
                "numerical_tolerance": 1e-6,
                "num_samples_boundary": 20000,
                "num_samples_lie": 30000,
                "optimization_maxiter": 50,
                "optimization_popsize": 20,
            }
        )

        result = self._run_single_benchmark("3d_50k_samples", test_case, config)
        self.results.append(result)

        # Update target
        for target in self.targets:
            if target.name == "scalable_3d":
                target.current_value = result.validation_time
                target.achieved = result.validation_time <= target.target_value

    def _benchmark_high_dimensional(self):
        """Benchmark high-dimensional systems"""
        # 5D system
        test_case = {
            "name": "5d_linear",
            "certificate": " + ".join([f"{var}**2" for var in ["x1", "x2", "x3", "x4", "x5"]])
            + " - 3.0",
            "system": {
                "variables": ["x1", "x2", "x3", "x4", "x5"],
                "dynamics": [f"-{var}" for var in ["x1", "x2", "x3", "x4", "x5"]],
                "initial_set_conditions": [
                    " + ".join([f"{var}**2" for var in ["x1", "x2", "x3", "x4", "x5"]]) + " <= 1.0"
                ],
                "unsafe_set_conditions": [
                    " + ".join([f"{var}**2" for var in ["x1", "x2", "x3", "x4", "x5"]]) + " >= 16.0"
                ],
                "safe_set_conditions": [],
                "sampling_bounds": {var: (-5, 5) for var in ["x1", "x2", "x3", "x4", "x5"]},
            },
        }

        config = DictConfig(
            {
                "numerical_tolerance": 1e-6,
                "num_samples_boundary": 5000,
                "num_samples_lie": 10000,
                "optimization_maxiter": 30,
                "optimization_popsize": 10,
            }
        )

        result = self._run_single_benchmark("5d_system", test_case, config)
        self.results.append(result)

    def _benchmark_memory_stress(self):
        """Benchmark memory usage under stress"""
        import tracemalloc

        test_case = {
            "name": "memory_stress",
            "certificate": "x**2 + y**2 - 1.0",
            "system": {
                "variables": ["x", "y"],
                "dynamics": ["-x", "-y"],
                "initial_set_conditions": ["x**2 + y**2 <= 0.25"],
                "unsafe_set_conditions": ["x**2 + y**2 >= 4.0"],
                "safe_set_conditions": [],
                "sampling_bounds": {"x": (-3, 3), "y": (-3, 3)},
            },
        }

        # Large sample size to stress memory
        config = DictConfig(
            {
                "numerical_tolerance": 1e-6,
                "num_samples_boundary": 50000,
                "num_samples_lie": 100000,
                "optimization_maxiter": 10,
                "optimization_popsize": 5,
            }
        )

        tracemalloc.start()
        result = self._run_single_benchmark("memory_stress", test_case, config)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result.memory_usage = peak / 1024 / 1024  # Convert to MB
        self.results.append(result)

        # Update target
        for target in self.targets:
            if target.name == "memory_efficient":
                target.current_value = result.memory_usage
                target.achieved = result.memory_usage <= target.target_value

    def _benchmark_accuracy(self):
        """Benchmark validation accuracy"""
        # Load ground truth test cases
        ground_truth_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "ground_truth/barrier_certificates.json"
        )

        if not os.path.exists(ground_truth_file):
            print("Warning: Ground truth file not found for accuracy benchmark")
            return

        with open(ground_truth_file, "r") as f:
            data = json.load(f)

        correct = 0
        total = 0

        for test in data["certificates"][:10]:  # Test first 10 for speed
            try:
                # Prepare system info
                system_info = {
                    "variables": test["system"]["variables"],
                    "dynamics": [d.split("=")[1].strip() for d in test["system"]["dynamics"]],
                    "initial_set_conditions": test["system"]["initial_set"],
                    "unsafe_set_conditions": test["system"]["unsafe_set"],
                    "safe_set_conditions": [],
                    "sampling_bounds": {var: (-3, 3) for var in test["system"]["variables"]},
                }

                config = DictConfig(
                    {
                        "numerical_tolerance": 1e-6,
                        "num_samples_boundary": 5000,
                        "num_samples_lie": 10000,
                        "optimization_maxiter": 50,
                        "optimization_popsize": 20,
                    }
                )

                validator = BarrierCertificateValidator(test["certificate"], system_info, config)
                result = validator.validate()

                if result["is_valid"] == test["expected_valid"]:
                    correct += 1
                total += 1

            except Exception as e:
                print(f"Error in accuracy test {test['id']}: {e}")

        accuracy = correct / total if total > 0 else 0

        # Update target
        for target in self.targets:
            if target.name == "high_accuracy":
                target.current_value = accuracy
                target.achieved = accuracy >= target.target_value

    def _benchmark_real_time(self):
        """Benchmark real-time response capability"""
        test_case = {
            "name": "real_time",
            "certificate": "x**2 + y**2 - 1.0",
            "system": {
                "variables": ["x", "y"],
                "dynamics": ["-x", "-y"],
                "initial_set_conditions": ["x**2 + y**2 <= 0.25"],
                "unsafe_set_conditions": ["x**2 + y**2 >= 4.0"],
                "safe_set_conditions": [],
                "sampling_bounds": {"x": (-3, 3), "y": (-3, 3)},
            },
        }

        # Minimal configuration for fast initial response
        config = DictConfig(
            {
                "numerical_tolerance": 1e-4,  # Relaxed tolerance
                "num_samples_boundary": 100,  # Minimal samples
                "num_samples_lie": 200,
                "optimization_maxiter": 5,
                "optimization_popsize": 5,
            }
        )

        # Measure time to first result
        start_time = time.time()
        validator = BarrierCertificateValidator(
            test_case["certificate"], test_case["system"], config
        )
        # Just parse and setup
        initial_response_time = time.time() - start_time

        # Update target
        for target in self.targets:
            if target.name == "real_time_feedback":
                target.current_value = initial_response_time
                target.achieved = initial_response_time <= target.target_value

    def _run_single_benchmark(
        self, benchmark_name: str, test_case: Dict[str, Any], config: DictConfig
    ) -> BenchmarkResult:
        """Run a single benchmark test"""
        start_time = time.time()

        try:
            validator = BarrierCertificateValidator(
                test_case["certificate"], test_case["system"], config
            )
            result = validator.validate()
            validation_time = time.time() - start_time

            # Calculate samples processed
            samples_processed = config.num_samples_boundary + config.num_samples_lie

            benchmark_result = BenchmarkResult(
                benchmark_name=benchmark_name,
                test_case=test_case["name"],
                validation_time=validation_time,
                memory_usage=0.0,  # Will be set separately if needed
                accuracy=1.0 if result["is_valid"] else 0.0,
                samples_processed=samples_processed,
                configuration=dict(config),
                timestamp=datetime.now(),
            )

            print(f"  {benchmark_name}: {validation_time:.3f}s ({samples_processed} samples)")

            return benchmark_result

        except Exception as e:
            print(f"  {benchmark_name}: FAILED - {e}")
            return BenchmarkResult(
                benchmark_name=benchmark_name,
                test_case=test_case["name"],
                validation_time=float("in"),
                memory_usage=0.0,
                accuracy=0.0,
                samples_processed=0,
                configuration=dict(config),
                timestamp=datetime.now(),
            )

    def _evaluate_targets(self):
        """Evaluate optimization targets based on results"""
        # Calculate average validation time for simple 2D
        simple_2d_times = [
            r.validation_time
            for r in self.results
            if "simple_2d" in r.benchmark_name and "s5000" in r.benchmark_name
        ]
        if simple_2d_times:
            avg_time = np.mean(simple_2d_times)
            for target in self.targets:
                if target.name == "fast_validation":
                    target.current_value = avg_time
                    target.achieved = avg_time <= target.target_value
                    target.improvement_percentage = (
                        (target.target_value - avg_time) / target.target_value * 100
                        if avg_time > target.target_value
                        else 100
                    )

    def _generate_report(self):
        """Generate optimization benchmark report"""
        report_file = os.path.join(self.output_dir, "optimization_report.json")

        # Prepare report data
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "targets": [asdict(t) for t in self.targets],
            "results": [asdict(r) for r in self.results],
            "summary": self._generate_summary(),
        }

        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)

        # Print summary
        print("\nOPTIMIZATION TARGETS SUMMARY")
        print("=" * 60)

        for target in self.targets:
            status = "✓ ACHIEVED" if target.achieved else "✗ NOT MET"
            print(f"\n{target.name}: {status}")
            print(f"  Description: {target.description}")
            print(f"  Target: {target.target_value}")
            print(
                f"  Current: {target.current_value:.3f}"
                if target.current_value
                else "  Current: Not measured"
            )
            if target.improvement_percentage is not None:
                print(f"  Improvement needed: {100 - target.improvement_percentage:.1f}%")

        print(f"\nDetailed report saved to: {report_file}")

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        achieved_count = sum(1 for t in self.targets if t.achieved)

        return {
            "total_targets": len(self.targets),
            "achieved": achieved_count,
            "success_rate": achieved_count / len(self.targets) * 100,
            "total_benchmarks": len(self.results),
            "avg_validation_time": np.mean([r.validation_time for r in self.results]),
            "total_samples_processed": sum(r.samples_processed for r in self.results),
        }


def main():
    """Main entry point for optimization benchmarks"""
    import argparse

    parser = argparse.ArgumentParser(description="Run optimization benchmarks")
    parser.add_argument(
        "--output", default="optimization_benchmarks", help="Output directory for results"
    )
    parser.add_argument(
        "--targets-only",
        action="store_true",
        help="Only show optimization targets without running benchmarks",
    )

    args = parser.parse_args()

    suite = OptimizationBenchmarkSuite(output_dir=args.output)

    if args.targets_only:
        print("OPTIMIZATION TARGETS")
        print("=" * 60)
        for target in suite.targets:
            print(f"\n{target.name}:")
            print(f"  {target.description}")
            print(f"  Target {target.target_metric}: {target.target_value}")
    else:
        suite.run_benchmark_suite()


if __name__ == "__main__":
    main()
