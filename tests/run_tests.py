#!/usr/bin/env python3
"""
Unified Test Runner for FM-LLM-Solver
Consolidates all test execution with configurable options.
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


class UnifiedTestRunner:
    """Unified test runner for all FM-LLM-Solver tests."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the test runner."""
        self.project_root = PROJECT_ROOT
        self.results = {
            'unit': {},
            'integration': {},
            'benchmarks': {},
            'summary': {}
        }
        self.config_path = config_path
        
    def run_unit_tests(self, pattern: str = "*test*.py", verbose: bool = False) -> Dict[str, Any]:
        """Run unit tests."""
        print("\n🧪 Running Unit Tests...")
        print("=" * 60)
        
        unit_dir = self.project_root / "tests" / "unit"
        results = {'passed': 0, 'failed': 0, 'errors': 0, 'tests': []}
        
        for test_file in unit_dir.glob(pattern):
            if test_file.name == "__init__.py":
                continue
                
            print(f"  Running {test_file.name}...")
            
            try:
                # Import and run the test
                module_name = test_file.stem
                spec = __import__(f"tests.unit.{module_name}", fromlist=[module_name])
                
                # Look for a main() function or test runner
                if hasattr(spec, 'main'):
                    result = spec.main()
                    if result == 0:
                        results['passed'] += 1
                        print(f"    ✅ PASSED")
                    else:
                        results['failed'] += 1
                        print(f"    ❌ FAILED")
                else:
                    # Try to run pytest on the file
                    import subprocess
                    cmd = [sys.executable, "-m", "pytest", str(test_file), "-v" if verbose else "-q"]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        results['passed'] += 1
                        print(f"    ✅ PASSED")
                    else:
                        results['failed'] += 1
                        print(f"    ❌ FAILED")
                        if verbose:
                            print(result.stdout)
                            print(result.stderr)
                            
            except Exception as e:
                results['errors'] += 1
                print(f"    💥 ERROR: {str(e)}")
                
        self.results['unit'] = results
        return results
        
    def run_integration_tests(self, quick: bool = False) -> Dict[str, Any]:
        """Run integration tests."""
        print("\n🔗 Running Integration Tests...")
        print("=" * 60)
        
        results = {'passed': 0, 'failed': 0, 'errors': 0, 'duration': 0}
        start_time = time.time()
        
        try:
            if quick:
                # Run quick integration tests
                from tests.integration.run_quick_integration_tests import main as quick_main
                print("  Running quick integration tests...")
                result = quick_main()
            else:
                # Run full integration tests
                from tests.integration.run_final_integration_tests import AdvancedIntegrationTester
                print("  Running full integration tests...")
                tester = AdvancedIntegrationTester()
                test_results = tester.run_all_integration_tests()
                report = tester.generate_report()
                
                results['passed'] = report['summary']['passed']
                results['failed'] = report['summary']['failed']
                results['errors'] = report['summary']['errors']
                results['report'] = report
                
        except Exception as e:
            results['errors'] += 1
            print(f"  💥 Integration test error: {str(e)}")
            
        results['duration'] = time.time() - start_time
        self.results['integration'] = results
        return results
        
    def run_benchmarks(self, specific_benchmark: Optional[str] = None) -> Dict[str, Any]:
        """Run benchmark tests."""
        print("\n📊 Running Benchmarks...")
        print("=" * 60)
        
        benchmarks_dir = self.project_root / "tests" / "benchmarks"
        results = {'benchmarks': {}, 'total_duration': 0}
        
        # List of benchmark modules
        benchmark_modules = [
            'llm_generation_testbench',
            'barrier_certificate_optimization_testbench',
            'verification_boundary_fix_testbench',
            'web_interface_testbench'
        ]
        
        if specific_benchmark:
            benchmark_modules = [bm for bm in benchmark_modules if specific_benchmark in bm]
            
        for module_name in benchmark_modules:
            print(f"\n  Running {module_name}...")
            start_time = time.time()
            
            try:
                # Import the benchmark module
                spec = __import__(f"tests.benchmarks.{module_name}", fromlist=[module_name])
                
                # Run the benchmark
                if module_name == 'llm_generation_testbench':
                    from tests.benchmarks.llm_generation_testbench import LLMGenerationTestbench
                    testbench = LLMGenerationTestbench(self.config_path)
                    testbench.setup_default_test_cases()
                    testbench.run_test_suite(
                        model_configs=["base"],
                        rag_k_values=[0],
                        max_attempts=1
                    )
                    analysis = testbench.analyze_results()
                    results['benchmarks'][module_name] = {
                        'success': True,
                        'summary': analysis['summary'],
                        'duration': time.time() - start_time
                    }
                else:
                    # Generic benchmark execution
                    if hasattr(spec, 'run_benchmark'):
                        result = spec.run_benchmark()
                        results['benchmarks'][module_name] = {
                            'success': True,
                            'result': result,
                            'duration': time.time() - start_time
                        }
                    else:
                        results['benchmarks'][module_name] = {
                            'success': False,
                            'error': 'No run_benchmark function found',
                            'duration': time.time() - start_time
                        }
                        
            except Exception as e:
                results['benchmarks'][module_name] = {
                    'success': False,
                    'error': str(e),
                    'duration': time.time() - start_time
                }
                print(f"    💥 Benchmark error: {str(e)}")
                
        results['total_duration'] = sum(b['duration'] for b in results['benchmarks'].values())
        self.results['benchmarks'] = results
        return results
        
    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        print("\n📝 Generating Test Report...")
        print("=" * 60)
        
        # Calculate summary statistics
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_errors = 0
        
        # Unit tests
        if 'unit' in self.results:
            unit = self.results['unit']
            total_tests += unit.get('passed', 0) + unit.get('failed', 0) + unit.get('errors', 0)
            total_passed += unit.get('passed', 0)
            total_failed += unit.get('failed', 0)
            total_errors += unit.get('errors', 0)
            
        # Integration tests
        if 'integration' in self.results:
            integration = self.results['integration']
            total_tests += integration.get('passed', 0) + integration.get('failed', 0) + integration.get('errors', 0)
            total_passed += integration.get('passed', 0)
            total_failed += integration.get('failed', 0)
            total_errors += integration.get('errors', 0)
            
        # Summary
        self.results['summary'] = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed': total_passed,
            'failed': total_failed,
            'errors': total_errors,
            'success_rate': (total_passed / total_tests * 100) if total_tests > 0 else 0
        }
        
        # Display summary
        print(f"\n{'='*60}")
        print("📊 TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {total_passed}")
        print(f"❌ Failed: {total_failed}")
        print(f"💥 Errors: {total_errors}")
        print(f"📈 Success Rate: {self.results['summary']['success_rate']:.1f}%")
        
        # Save report if requested
        if output_file:
            output_path = self.project_root / "results" / output_file
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"\n💾 Report saved to: {output_path}")
            
        return self.results
        
    def run_all(self, skip_benchmarks: bool = False) -> int:
        """Run all tests."""
        print("\n🚀 FM-LLM-Solver Unified Test Runner")
        print("=" * 60)
        
        overall_start = time.time()
        
        # Run tests in order
        self.run_unit_tests()
        self.run_integration_tests(quick=True)
        
        if not skip_benchmarks:
            self.run_benchmarks()
            
        # Generate report
        report = self.generate_report(f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        total_duration = time.time() - overall_start
        print(f"\n⏱️  Total test duration: {total_duration:.2f}s")
        
        # Return exit code based on results
        if report['summary']['failed'] > 0 or report['summary']['errors'] > 0:
            return 1
        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Unified Test Runner for FM-LLM-Solver")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--benchmarks", action="store_true", help="Run only benchmarks")
    parser.add_argument("--benchmark", type=str, help="Run specific benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--skip-benchmarks", action="store_true", help="Skip benchmark tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", type=str, help="Output report file name")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    # Create runner
    runner = UnifiedTestRunner(args.config)
    
    # Determine what to run
    if args.unit:
        runner.run_unit_tests(verbose=args.verbose)
    elif args.integration:
        runner.run_integration_tests(quick=args.quick)
    elif args.benchmarks or args.benchmark:
        runner.run_benchmarks(specific_benchmark=args.benchmark)
    else:
        # Run all
        return runner.run_all(skip_benchmarks=args.skip_benchmarks)
        
    # Generate report
    runner.generate_report(args.output)
    
    # Check for failures
    summary = runner.results.get('summary', {})
    if summary.get('failed', 0) > 0 or summary.get('errors', 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main()) 