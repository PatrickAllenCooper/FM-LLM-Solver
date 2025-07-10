"""
Automated Test Harness for Barrier Certificate Validation (Phase 1 Day 6-7)

This harness:
1. Loads ground truth certificates from JSON
2. Runs validation using both old and new validators
3. Compares results and identifies discrepancies
4. Generates detailed reports
"""

import json
import sys
import os
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.level_set_tracker import BarrierCertificateValidator, LevelSetTracker
from utils.set_membership import SetMembershipTester
from utils.adaptive_tolerance import AdaptiveTolerance
from evaluation.verify_certificate import verify_barrier_certificate
from evaluation.verify_certificate_fixed import numerical_check_all_conditions_fixed
from omegaconf import DictConfig
import sympy
import numpy as np


@dataclass
class TestResult:
    """Result of a single test case"""
    test_id: str
    system_name: str
    certificate: str
    expected_valid: bool
    
    # Validation results
    new_validator_result: Optional[bool] = None
    old_validator_result: Optional[bool] = None
    fixed_validator_result: Optional[bool] = None
    
    # Details
    level_sets_computed: Optional[Dict[str, float]] = None
    level_sets_expected: Optional[Dict[str, float]] = None
    level_set_match: Optional[bool] = None
    
    # Timing
    new_validator_time: Optional[float] = None
    old_validator_time: Optional[float] = None
    
    # Errors
    new_validator_error: Optional[str] = None
    old_validator_error: Optional[str] = None
    
    # Analysis
    agreement: Optional[bool] = None
    correct: Optional[bool] = None
    notes: Optional[str] = None


class BarrierCertificateTestHarness:
    """Comprehensive test harness for barrier certificate validation"""
    
    def __init__(self, ground_truth_file: str = "tests/ground_truth/barrier_certificates.json"):
        self.ground_truth_file = ground_truth_file
        self.test_cases = []
        self.results = []
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("TestHarness")
        logger.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        fh = logging.FileHandler(f'test_harness_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
    
    def load_test_cases(self):
        """Load test cases from ground truth JSON"""
        self.logger.info(f"Loading test cases from {self.ground_truth_file}")
        
        with open(self.ground_truth_file, 'r') as f:
            data = json.load(f)
            
        self.test_cases = data['certificates']
        self.logger.info(f"Loaded {len(self.test_cases)} test cases")
        
    def run_all_tests(self, subset: Optional[List[str]] = None):
        """Run all test cases or a subset"""
        if not self.test_cases:
            self.load_test_cases()
            
        test_list = self.test_cases
        if subset:
            test_list = [tc for tc in self.test_cases if tc['id'] in subset]
            
        self.logger.info(f"Running {len(test_list)} tests...")
        
        for i, test_case in enumerate(test_list):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Test {i+1}/{len(test_list)}: {test_case['id']}")
            self.logger.info(f"System: {test_case['system']['name']}")
            
            result = self.run_single_test(test_case)
            self.results.append(result)
            
            # Log immediate result
            if result.correct:
                self.logger.info(f"✓ PASS: Result matches expectation")
            else:
                self.logger.warning(f"✗ FAIL: Result does not match expectation")
                
    def run_single_test(self, test_case: Dict[str, Any]) -> TestResult:
        """Run a single test case through all validators"""
        result = TestResult(
            test_id=test_case['id'],
            system_name=test_case['system']['name'],
            certificate=test_case['certificate'],
            expected_valid=test_case['expected_valid'],
            level_sets_expected=test_case.get('level_sets'),
            notes=test_case.get('notes', '')
        )
        
        # Prepare system info
        system_info = self._prepare_system_info(test_case['system'])
        
        # Run new validator
        self.logger.info("Running new validator...")
        try:
            start_time = time.time()
            new_result = self._run_new_validator(
                test_case['certificate'],
                system_info
            )
            result.new_validator_time = time.time() - start_time
            result.new_validator_result = new_result['is_valid']
            result.level_sets_computed = new_result.get('level_sets')
            
            # Check level set match
            if result.level_sets_expected and result.level_sets_computed:
                c1_match = abs(result.level_sets_computed.get('c1', 0) - 
                             result.level_sets_expected.get('c1', 0)) < 0.1
                c2_match = abs(result.level_sets_computed.get('c2', 0) - 
                             result.level_sets_expected.get('c2', 0)) < 0.1
                result.level_set_match = c1_match and c2_match
                
        except Exception as e:
            self.logger.error(f"New validator error: {str(e)}")
            result.new_validator_error = str(e)
            result.new_validator_result = None
            
        # Run old validator
        self.logger.info("Running old validator...")
        try:
            start_time = time.time()
            old_result = self._run_old_validator(
                test_case['certificate'],
                system_info
            )
            result.old_validator_time = time.time() - start_time
            result.old_validator_result = old_result.get('overall_success', False)
            
        except Exception as e:
            self.logger.error(f"Old validator error: {str(e)}")
            result.old_validator_error = str(e)
            result.old_validator_result = None
            
        # Run fixed numerical validator
        self.logger.info("Running fixed numerical validator...")
        try:
            fixed_result = self._run_fixed_validator(
                test_case['certificate'],
                system_info
            )
            result.fixed_validator_result = fixed_result
            
        except Exception as e:
            self.logger.error(f"Fixed validator error: {str(e)}")
            result.fixed_validator_result = None
            
        # Analyze results
        result.agreement = (result.new_validator_result == result.old_validator_result 
                          if None not in [result.new_validator_result, result.old_validator_result]
                          else None)
        
        # Check correctness (prioritize new validator)
        if result.new_validator_result is not None:
            result.correct = (result.new_validator_result == result.expected_valid)
        elif result.fixed_validator_result is not None:
            result.correct = (result.fixed_validator_result == result.expected_valid)
        elif result.old_validator_result is not None:
            result.correct = (result.old_validator_result == result.expected_valid)
        else:
            result.correct = None
            
        return result
        
    def _prepare_system_info(self, system: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare system info for validators"""
        variables = system['variables']
        
        # Parse dynamics
        dynamics = []
        for dyn_str in system['dynamics']:
            # Handle both continuous and discrete time
            if 'dt' in dyn_str:
                # Continuous: dx/dt = f(x)
                dynamics.append(dyn_str.split('=')[1].strip())
            elif '[k+1]' in dyn_str:
                # Discrete: x[k+1] = f(x[k])
                dynamics.append(dyn_str.split('=')[1].strip())
            else:
                dynamics.append(dyn_str)
                
        # Create bounds
        bounds = {}
        for var in variables:
            bounds[var] = (-3.0, 3.0)  # Default bounds
            
        return {
            'variables': variables,
            'dynamics': dynamics,
            'initial_set_conditions': system['initial_set'],
            'unsafe_set_conditions': system['unsafe_set'],
            'safe_set_conditions': [],  # Will be auto-generated
            'sampling_bounds': bounds,
            'is_discrete': system.get('time_type') == 'discrete'
        }
        
    def _run_new_validator(self, certificate: str, system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run the new BarrierCertificateValidator"""
        config = DictConfig({
            'numerical_tolerance': 1e-6,
            'num_samples_boundary': 5000,
            'num_samples_lie': 10000,
            'optimization_maxiter': 100,
            'optimization_popsize': 30
        })
        
        validator = BarrierCertificateValidator(
            certificate_str=certificate,
            system_info=system_info,
            config=config
        )
        
        return validator.validate()
        
    def _run_old_validator(self, certificate: str, system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run the old verify_barrier_certificate"""
        config = DictConfig({
            'numerical_tolerance': 1e-6,
            'num_samples_boundary': 5000,
            'num_samples_lie': 10000,
            'optimization_maxiter': 100,
            'optimization_popsize': 30
        })
        
        return verify_barrier_certificate(
            certificate,
            system_info,
            config
        )
        
    def _run_fixed_validator(self, certificate: str, system_info: Dict[str, Any]) -> bool:
        """Run the fixed numerical validator"""
        # Parse certificate
        variables = system_info['variables']
        var_symbols = [sympy.Symbol(v) for v in variables]
        
        try:
            B_expr = sympy.parse_expr(certificate)
            B_func = sympy.lambdify(var_symbols, B_expr, 'numpy')
            
            # Compute Lie derivative
            lie_derivative = 0
            for i, var in enumerate(var_symbols):
                dB_dvar = sympy.diff(B_expr, var)
                if system_info.get('is_discrete'):
                    # For discrete systems, use ΔB = B(f(x)) - B(x)
                    # This is simplified - would need full implementation
                    lie_derivative += dB_dvar * sympy.parse_expr(system_info['dynamics'][i])
                else:
                    lie_derivative += dB_dvar * sympy.parse_expr(system_info['dynamics'][i])
                    
            dB_dt_func = sympy.lambdify(var_symbols, lie_derivative, 'numpy')
            
            # Parse set conditions
            from evaluation.utils import parse_relationals
            initial_relationals = parse_relationals(
                system_info['initial_set_conditions'], var_symbols
            )
            unsafe_relationals = parse_relationals(
                system_info['unsafe_set_conditions'], var_symbols
            )
            safe_relationals = []  # Auto-generated
            
            # Run fixed validator
            passed, details = numerical_check_all_conditions_fixed(
                B_func, dB_dt_func,
                system_info['sampling_bounds'],
                var_symbols,
                initial_relationals,
                unsafe_relationals,
                safe_relationals,
                5000, 10000, 1e-6
            )
            
            return passed
            
        except Exception as e:
            self.logger.error(f"Error in fixed validator: {e}")
            return None
            
    def generate_report(self, output_file: str = "test_harness_report.txt"):
        """Generate a comprehensive test report"""
        self.logger.info(f"\nGenerating report to {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("BARRIER CERTIFICATE TEST HARNESS REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Tests: {len(self.results)}\n\n")
            
            # Summary statistics
            passed = sum(1 for r in self.results if r.correct == True)
            failed = sum(1 for r in self.results if r.correct == False)
            errors = sum(1 for r in self.results if r.correct is None)
            
            f.write("SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Passed: {passed}/{len(self.results)} ({100*passed/len(self.results):.1f}%)\n")
            f.write(f"Failed: {failed}/{len(self.results)} ({100*failed/len(self.results):.1f}%)\n")
            f.write(f"Errors: {errors}/{len(self.results)} ({100*errors/len(self.results):.1f}%)\n\n")
            
            # Agreement statistics
            agreements = [r for r in self.results if r.agreement is not None]
            agree_count = sum(1 for r in agreements if r.agreement)
            if agreements:
                f.write(f"Validator Agreement: {agree_count}/{len(agreements)} ({100*agree_count/len(agreements):.1f}%)\n\n")
            
            # Detailed results
            f.write("DETAILED RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            for result in self.results:
                f.write(f"Test ID: {result.test_id}\n")
                f.write(f"System: {result.system_name}\n")
                f.write(f"Certificate: {result.certificate}\n")
                f.write(f"Expected Valid: {result.expected_valid}\n")
                
                f.write("\nResults:\n")
                f.write(f"  New Validator: {result.new_validator_result}")
                if result.new_validator_error:
                    f.write(f" (ERROR: {result.new_validator_error})")
                if result.new_validator_time:
                    f.write(f" [{result.new_validator_time:.3f}s]")
                f.write("\n")
                
                f.write(f"  Old Validator: {result.old_validator_result}")
                if result.old_validator_error:
                    f.write(f" (ERROR: {result.old_validator_error})")
                if result.old_validator_time:
                    f.write(f" [{result.old_validator_time:.3f}s]")
                f.write("\n")
                
                f.write(f"  Fixed Validator: {result.fixed_validator_result}\n")
                
                if result.level_sets_computed:
                    f.write(f"\nLevel Sets:\n")
                    f.write(f"  Computed: c1={result.level_sets_computed.get('c1', 'N/A'):.3f}, "
                           f"c2={result.level_sets_computed.get('c2', 'N/A'):.3f}\n")
                    if result.level_sets_expected:
                        f.write(f"  Expected: c1={result.level_sets_expected.get('c1', 'N/A'):.3f}, "
                               f"c2={result.level_sets_expected.get('c2', 'N/A'):.3f}\n")
                        f.write(f"  Match: {result.level_set_match}\n")
                
                f.write(f"\nStatus: {'PASS' if result.correct else 'FAIL' if result.correct is not None else 'ERROR'}")
                if result.agreement is not None:
                    f.write(f" | Agreement: {'YES' if result.agreement else 'NO'}")
                f.write("\n")
                
                if result.notes:
                    f.write(f"Notes: {result.notes}\n")
                    
                f.write("-" * 80 + "\n\n")
                
            # Failed tests summary
            failed_tests = [r for r in self.results if r.correct == False]
            if failed_tests:
                f.write("\nFAILED TESTS SUMMARY\n")
                f.write("=" * 80 + "\n")
                for result in failed_tests:
                    f.write(f"{result.test_id}: Expected {result.expected_valid}, "
                           f"Got New={result.new_validator_result}, Old={result.old_validator_result}\n")
                    
            # Error tests summary
            error_tests = [r for r in self.results if r.correct is None]
            if error_tests:
                f.write("\nERROR TESTS SUMMARY\n")
                f.write("=" * 80 + "\n")
                for result in error_tests:
                    f.write(f"{result.test_id}: ")
                    if result.new_validator_error:
                        f.write(f"New error: {result.new_validator_error[:100]}... ")
                    if result.old_validator_error:
                        f.write(f"Old error: {result.old_validator_error[:100]}...")
                    f.write("\n")
                    
        self.logger.info(f"Report saved to {output_file}")
        
    def export_results_json(self, output_file: str = "test_harness_results.json"):
        """Export results to JSON for further analysis"""
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(self.results),
            'summary': {
                'passed': sum(1 for r in self.results if r.correct == True),
                'failed': sum(1 for r in self.results if r.correct == False),
                'errors': sum(1 for r in self.results if r.correct is None)
            },
            'results': [asdict(r) for r in self.results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
            
        self.logger.info(f"Results exported to {output_file}")


def main():
    """Main entry point for test harness"""
    harness = BarrierCertificateTestHarness()
    
    # Run all tests
    harness.run_all_tests()
    
    # Or run subset
    # harness.run_all_tests(subset=['linear_stable_2d_basic', 'invalid_no_separation'])
    
    # Generate reports
    harness.generate_report()
    harness.export_results_json()
    
    # Print summary
    passed = sum(1 for r in harness.results if r.correct == True)
    total = len(harness.results)
    print(f"\nTest Summary: {passed}/{total} passed ({100*passed/total:.1f}%)")


if __name__ == "__main__":
    main() 