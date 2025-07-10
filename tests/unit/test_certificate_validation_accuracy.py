#!/usr/bin/env python3
"""
Certificate Validation Accuracy Test
==================================

Tests the mathematical correctness of generated barrier certificates
with real validation against system dynamics.
"""

import os
import sys
import time
import logging
import numpy as np
import sympy
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import load_config
from utils.certificate_extraction import extract_certificate_from_llm_output
from utils.verification_helpers import validate_candidate_expression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CertificateValidationTester:
    """Comprehensive tester for certificate validation accuracy"""
    
    def __init__(self):
        self.validation_results = []
        
    def generate_test_systems(self) -> List[Dict]:
        """Generate test systems with known valid certificates"""
        return [
            # Simple linear system - should have valid certificates
            {
                "name": "linear_stable_2d",
                "dynamics": ["dx/dt = -x", "dy/dt = -y"],
                "initial_set": ["x**2 + y**2 <= 0.25"],
                "unsafe_set": ["x**2 + y**2 >= 4.0"],
                "valid_certificates": [
                    "x**2 + y**2 - 1.0",  # Should be valid (separates initial from unsafe)
                    "x**2 + y**2 - 0.75",  # Should be valid (separates initial from unsafe)
                    "x**2 + y**2 - 1.5",  # Should be invalid (violates unsafe set boundary)
                    "x**2 + y**2 - 2.0",  # Should be invalid (violates unsafe set boundary)
                    "x**2 + y**2 - 0.5",  # Should be invalid (too small)
                    "x**2 + y**2 - 5.0",  # Should be invalid (too large)
                ],
                "expected_valid": [True, True, False, False, False, False]
            },
            
            # Unstable system - should reject most certificates
            {
                "name": "linear_unstable_2d",
                "dynamics": ["dx/dt = x", "dy/dt = y"],
                "initial_set": ["x**2 + y**2 <= 0.1"],
                "unsafe_set": ["x**2 + y**2 >= 1.0"],
                "valid_certificates": [
                    "x**2 + y**2 - 0.5",  # Should be invalid (unstable system)
                    "x**2 + y**2 - 1.5",  # Should be invalid (unstable system)
                ],
                "expected_valid": [False, False]
            },
            
            # Nonlinear system - more complex validation
            {
                "name": "nonlinear_cubic_2d",
                "dynamics": ["dx/dt = -x**3 - y", "dy/dt = x - y**3"],
                "initial_set": ["x**2 + y**2 <= 0.1"],
                "unsafe_set": ["x**2 + y**2 >= 2.0"],
                "valid_certificates": [
                    "x**2 + y**2 - 0.5",  # Should be invalid (Lie derivative positive for nonlinear)
                    "x**2 + y**2 - 0.75",  # Should be invalid (Lie derivative positive for nonlinear)
                    "x**2 + y**2 - 1.0",  # Should be invalid (Lie derivative positive for nonlinear)
                    "x**2 + y**2 - 1.5",  # Should be invalid (Lie derivative positive for nonlinear)
                    "x**2 + y**2 - 0.05", # Should be invalid (too small)
                ],
                "expected_valid": [False, False, False, False, False]
            },
            
            # 3D system
            {
                "name": "linear_3d",
                "dynamics": ["dx/dt = -x", "dy/dt = -y", "dz/dt = -z"],
                "initial_set": ["x**2 + y**2 + z**2 <= 0.1"],
                "unsafe_set": ["x**2 + y**2 + z**2 >= 1.0"],
                "valid_certificates": [
                    "x**2 + y**2 + z**2 - 0.5",  # Should be valid (separates initial from unsafe)
                    "x**2 + y**2 + z**2 - 0.75",  # Should be valid (separates initial from unsafe)
                    "x**2 + y**2 + z**2 - 0.3",  # Should be valid (separates initial from unsafe)
                    "x**2 + y**2 + z**2 - 1.5",  # Should be invalid (violates unsafe set)
                    "x**2 + y**2 + z**2 - 0.05", # Should be invalid (too small)
                ],
                "expected_valid": [True, True, True, False, False]
            }
        ]
    
    def validate_certificate_mathematically(self, certificate: str, system: Dict, n_samples: int = 10) -> Dict:
        """Validate a certificate mathematically with robust sampling"""
        try:
            # Parse certificate
            x, y = sympy.symbols('x y')
            if 'z' in certificate:
                z = sympy.symbols('z')
                vars_sympy = [x, y, z]
            else:
                vars_sympy = [x, y]
            
            # Parse certificate expression
            B = sympy.parse_expr(certificate)
            
            # Parse dynamics
            dynamics = []
            for dyn in system['dynamics']:
                if '=' in dyn:
                    rhs = dyn.split('=')[1].strip()
                    dynamics.append(sympy.parse_expr(rhs))
                else:
                    dynamics.append(sympy.parse_expr(dyn))
            
            # Calculate Lie derivative (for continuous systems)
            if len(dynamics) == 2:  # 2D system
                dB_dx = sympy.diff(B, x)
                dB_dy = sympy.diff(B, y)
                lie_derivative = dB_dx * dynamics[0] + dB_dy * dynamics[1]
            elif len(dynamics) == 3:  # 3D system
                dB_dx = sympy.diff(B, x)
                dB_dy = sympy.diff(B, y)
                dB_dz = sympy.diff(B, z)
                lie_derivative = dB_dx * dynamics[0] + dB_dy * dynamics[1] + dB_dz * dynamics[2]
            else:
                return {"valid": False, "error": "Unsupported system dimension"}
            
            # Parse set conditions
            initial_condition = sympy.parse_expr(system['initial_set'][0].replace('<=', '<='))
            unsafe_condition = sympy.parse_expr(system['unsafe_set'][0].replace('>=', '>='))
            
            # Check barrier conditions
            violations = []
            
            # --- Improved: Grid and random sampling for initial set boundary ---
            rng = np.random.default_rng(42)
            if len(vars_sympy) == 3:
                # 3D grid and random points
                grid = np.linspace(-0.5, 0.5, n_samples)
                for xi in grid:
                    for yi in grid:
                        for zi in grid:
                            point = {'x': float(xi), 'y': float(yi), 'z': float(zi)}
                            if initial_condition.subs(point):
                                B_val = B.subs(point)
                                if B_val > 0:
                                    violations.append(f"Initial set violation at {point}: B={B_val}")
                # Random points
                for _ in range(n_samples):
                    point = {'x': float(rng.uniform(-0.5, 0.5)), 'y': float(rng.uniform(-0.5, 0.5)), 'z': float(rng.uniform(-0.5, 0.5))}
                    if initial_condition.subs(point):
                        B_val = B.subs(point)
                        if B_val > 0:
                            violations.append(f"Initial set violation at {point}: B={B_val}")
            else:
                # 2D grid and random points
                grid = np.linspace(-0.5, 0.5, n_samples)
                for xi in grid:
                    for yi in grid:
                        point = {'x': float(xi), 'y': float(yi)}
                        if initial_condition.subs(point):
                            B_val = B.subs(point)
                            if B_val > 0:
                                violations.append(f"Initial set violation at {point}: B={B_val}")
                for _ in range(n_samples):
                    point = {'x': float(rng.uniform(-0.5, 0.5)), 'y': float(rng.uniform(-0.5, 0.5))}
                    if initial_condition.subs(point):
                        B_val = B.subs(point)
                        if B_val > 0:
                            violations.append(f"Initial set violation at {point}: B={B_val}")
            
            # --- Improved: Grid and random sampling for unsafe set boundary ---
            if len(vars_sympy) == 3:
                grid = np.linspace(-3.0, 3.0, n_samples)
                for xi in grid:
                    for yi in grid:
                        for zi in grid:
                            point = {'x': float(xi), 'y': float(yi), 'z': float(zi)}
                            if unsafe_condition.subs(point):  # Point is IN unsafe set
                                B_val = B.subs(point)
                                if B_val <= 0:  # Should be > 0 in unsafe set
                                    violations.append(f"Unsafe set violation at {point}: B={B_val}")
                for _ in range(n_samples):
                    point = {'x': float(rng.uniform(-3.0, 3.0)), 'y': float(rng.uniform(-3.0, 3.0)), 'z': float(rng.uniform(-3.0, 3.0))}
                    if unsafe_condition.subs(point):  # Point is IN unsafe set
                        B_val = B.subs(point)
                        if B_val <= 0:  # Should be > 0 in unsafe set
                            violations.append(f"Unsafe set violation at {point}: B={B_val}")
            else:
                grid = np.linspace(-3.0, 3.0, n_samples)
                for xi in grid:
                    for yi in grid:
                        point = {'x': float(xi), 'y': float(yi)}
                        if unsafe_condition.subs(point):  # Point is IN unsafe set
                            B_val = B.subs(point)
                            if B_val <= 0:  # Should be > 0 in unsafe set
                                violations.append(f"Unsafe set violation at {point}: B={B_val}")
                for _ in range(n_samples):
                    point = {'x': float(rng.uniform(-3.0, 3.0)), 'y': float(rng.uniform(-3.0, 3.0))}
                    if unsafe_condition.subs(point):  # Point is IN unsafe set
                        B_val = B.subs(point)
                        if B_val <= 0:  # Should be > 0 in unsafe set
                            violations.append(f"Unsafe set violation at {point}: B={B_val}")
            
            # --- Improved: Grid and random sampling for Lie derivative ---
            if len(vars_sympy) == 3:
                grid = np.linspace(-1.0, 1.0, n_samples)
                for xi in grid:
                    for yi in grid:
                        for zi in grid:
                            point = {'x': float(xi), 'y': float(yi), 'z': float(zi)}
                            lie_val = lie_derivative.subs(point)
                            if lie_val > 0:
                                violations.append(f"Lie derivative violation at {point}: dB/dt={lie_val}")
                for _ in range(n_samples):
                    point = {'x': float(rng.uniform(-1.0, 1.0)), 'y': float(rng.uniform(-1.0, 1.0)), 'z': float(rng.uniform(-1.0, 1.0))}
                    lie_val = lie_derivative.subs(point)
                    if lie_val > 0:
                        violations.append(f"Lie derivative violation at {point}: dB/dt={lie_val}")
            else:
                grid = np.linspace(-1.0, 1.0, n_samples)
                for xi in grid:
                    for yi in grid:
                        point = {'x': float(xi), 'y': float(yi)}
                        lie_val = lie_derivative.subs(point)
                        if lie_val > 0:
                            violations.append(f"Lie derivative violation at {point}: dB/dt={lie_val}")
                for _ in range(n_samples):
                    point = {'x': float(rng.uniform(-1.0, 1.0)), 'y': float(rng.uniform(-1.0, 1.0))}
                    lie_val = lie_derivative.subs(point)
                    if lie_val > 0:
                        violations.append(f"Lie derivative violation at {point}: dB/dt={lie_val}")
            
            is_valid = len(violations) == 0
            
            return {
                "valid": is_valid,
                "violations": violations,
                "certificate": certificate,
                "lie_derivative": str(lie_derivative),
                "num_violations": len(violations)
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "certificate": certificate
            }
    
    def test_certificate_extraction_accuracy(self) -> Dict:
        """Test certificate extraction accuracy"""
        logger.info("🧪 Testing certificate extraction accuracy...")
        
        # Test cases with known expected results
        test_cases = [
            {
                "input": "BARRIER_CERTIFICATE_START\nx**2 + y**2 - 1.5\nBARRIER_CERTIFICATE_END",
                "expected": "x**2 + y**2 - 1.5",
                "variables": ["x", "y"]
            },
            {
                "input": "B(x,y) = x**2 + y**2 - 1.5",
                "expected": "x**2 + y**2 - 1.5",
                "variables": ["x", "y"]
            },
            {
                "input": "Certificate: x**2 + y**2 - 1.5",
                "expected": "x**2 + y**2 - 1.5",
                "variables": ["x", "y"]
            },
            {
                "input": "Invalid format with no certificate",
                "expected": None,
                "variables": ["x", "y"]
            },
            {
                "input": "BARRIER_CERTIFICATE_START\nax**2 + by**2 + c\nBARRIER_CERTIFICATE_END",
                "expected": None,  # Template should be rejected
                "variables": ["x", "y"]
            }
        ]
        
        correct_extractions = 0
        results = []
        
        for i, test_case in enumerate(test_cases):
            try:
                extracted_result = extract_certificate_from_llm_output(
                    test_case["input"], 
                    test_case["variables"]
                )
                extracted = extracted_result[0] if isinstance(extracted_result, tuple) else extracted_result
                
                # Check if extraction matches expected
                if extracted == test_case["expected"]:
                    correct_extractions += 1
                    status = "✅ CORRECT"
                else:
                    status = "❌ INCORRECT"
                
                results.append({
                    "test_case": i + 1,
                    "input": test_case["input"][:50] + "..." if len(test_case["input"]) > 50 else test_case["input"],
                    "expected": test_case["expected"],
                    "extracted": extracted,
                    "correct": extracted == test_case["expected"],
                    "status": status
                })
                
            except Exception as e:
                results.append({
                    "test_case": i + 1,
                    "input": test_case["input"][:50] + "..." if len(test_case["input"]) > 50 else test_case["input"],
                    "expected": test_case["expected"],
                    "extracted": None,
                    "correct": False,
                    "status": f"❌ ERROR: {str(e)}"
                })
        
        accuracy = correct_extractions / len(test_cases)
        
        return {
            "accuracy": accuracy,
            "correct_extractions": correct_extractions,
            "total_tests": len(test_cases),
            "results": results
        }
    
    def test_certificate_validation_accuracy(self) -> Dict:
        """Test certificate validation accuracy"""
        logger.info("🧪 Testing certificate validation accuracy...")
        
        systems = self.generate_test_systems()
        all_results = []
        
        for system in systems:
            logger.info(f"Testing system: {system['name']}")
            system_results = []
            
            for i, certificate in enumerate(system['valid_certificates']):
                expected_valid = system['expected_valid'][i]
                
                # Validate certificate mathematically
                validation_result = self.validate_certificate_mathematically(certificate, system)
                
                # Check if validation matches expectation
                actual_valid = validation_result.get('valid', False)
                validation_correct = (actual_valid == expected_valid)
                
                system_results.append({
                    "certificate": certificate,
                    "expected_valid": expected_valid,
                    "actual_valid": actual_valid,
                    "validation_correct": validation_correct,
                    "violations": validation_result.get('violations', []),
                    "lie_derivative": validation_result.get('lie_derivative', ''),
                    "num_violations": validation_result.get('num_violations', 0)
                })
            
            # Calculate accuracy for this system
            correct_validations = sum(1 for r in system_results if r['validation_correct'])
            system_accuracy = correct_validations / len(system_results)
            
            all_results.append({
                "system_name": system['name'],
                "accuracy": system_accuracy,
                "correct_validations": correct_validations,
                "total_certificates": len(system_results),
                "results": system_results
            })
        
        # Calculate overall accuracy
        total_correct = sum(r['correct_validations'] for r in all_results)
        total_tests = sum(r['total_certificates'] for r in all_results)
        overall_accuracy = total_correct / total_tests if total_tests > 0 else 0
        
        return {
            "overall_accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_tests": total_tests,
            "system_results": all_results
        }
    
    def test_end_to_end_accuracy(self) -> Dict:
        """Test end-to-end accuracy from LLM output to validation"""
        logger.info("🧪 Testing end-to-end accuracy...")
        
        # Simulate LLM outputs with certificates
        llm_outputs = [
            "BARRIER_CERTIFICATE_START\nx**2 + y**2 - 1.0\nBARRIER_CERTIFICATE_END",
            "B(x,y) = x**2 + y**2 - 0.75",
            "Certificate: x**2 + y**2 - 0.3",
            "BARRIER_CERTIFICATE_START\nax**2 + by**2 + c\nBARRIER_CERTIFICATE_END",  # Template
        ]
        
        system = {
            "dynamics": ["dx/dt = -x", "dy/dt = -y"],
            "initial_set": ["x**2 + y**2 <= 0.25"],
            "unsafe_set": ["x**2 + y**2 >= 4.0"]
        }
        
        expected_results = [True, True, False, False]  # First two should be valid, last two invalid
        
        correct_end_to_end = 0
        results = []
        
        for i, llm_output in enumerate(llm_outputs):
            try:
                # Extract certificate
                extracted_result = extract_certificate_from_llm_output(llm_output, ["x", "y"])
                extracted = extracted_result[0] if isinstance(extracted_result, tuple) else extracted_result
                
                if extracted:
                    # Validate certificate
                    validation_result = self.validate_certificate_mathematically(extracted, system)
                    actual_valid = validation_result.get('valid', False)
                else:
                    actual_valid = False
                
                expected_valid = expected_results[i]
                end_to_end_correct = (actual_valid == expected_valid)
                
                if end_to_end_correct:
                    correct_end_to_end += 1
                
                results.append({
                    "llm_output": llm_output[:50] + "..." if len(llm_output) > 50 else llm_output,
                    "extracted": extracted,
                    "expected_valid": expected_valid,
                    "actual_valid": actual_valid,
                    "end_to_end_correct": end_to_end_correct,
                    "validation_details": validation_result if extracted else {"valid": False, "error": "No extraction"}
                })
                
            except Exception as e:
                results.append({
                    "llm_output": llm_output[:50] + "..." if len(llm_output) > 50 else llm_output,
                    "extracted": None,
                    "expected_valid": expected_results[i],
                    "actual_valid": False,
                    "end_to_end_correct": False,
                    "validation_details": {"valid": False, "error": str(e)}
                })
        
        end_to_end_accuracy = correct_end_to_end / len(llm_outputs)
        
        return {
            "end_to_end_accuracy": end_to_end_accuracy,
            "correct_end_to_end": correct_end_to_end,
            "total_tests": len(llm_outputs),
            "results": results
        }
    
    def run_comprehensive_accuracy_tests(self) -> Dict:
        """Run all accuracy tests"""
        logger.info("🚀 Starting comprehensive accuracy tests...")
        
        start_time = time.time()
        
        # Run all accuracy tests
        extraction_results = self.test_certificate_extraction_accuracy()
        validation_results = self.test_certificate_validation_accuracy()
        end_to_end_results = self.test_end_to_end_accuracy()
        
        total_time = time.time() - start_time
        
        # Compile comprehensive results
        comprehensive_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_time": total_time,
            "extraction_accuracy": extraction_results["accuracy"],
            "validation_accuracy": validation_results["overall_accuracy"],
            "end_to_end_accuracy": end_to_end_results["end_to_end_accuracy"],
            "overall_accuracy": (extraction_results["accuracy"] + 
                               validation_results["overall_accuracy"] + 
                               end_to_end_results["end_to_end_accuracy"]) / 3,
            "detailed_results": {
                "extraction": extraction_results,
                "validation": validation_results,
                "end_to_end": end_to_end_results
            }
        }
        
        return comprehensive_results
    
    def save_accuracy_results(self, results: Dict, output_path: str = "test_results/certificate_accuracy_results.json"):
        """Save accuracy test results"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Accuracy results saved to: {output_path}")
    
    def generate_accuracy_report(self, results: Dict) -> str:
        """Generate human-readable accuracy report"""
        report = []
        report.append("🎯 CERTIFICATE ACCURACY REPORT")
        report.append("=" * 50)
        report.append(f"Timestamp: {results['timestamp']}")
        report.append(f"Total Time: {results['total_time']:.1f} seconds")
        report.append("")
        
        # Overall accuracy
        overall_acc = results['overall_accuracy']
        report.append(f"📊 OVERALL ACCURACY: {overall_acc:.1%}")
        
        if overall_acc >= 0.95:
            report.append("✅ EXCELLENT: Near-perfect accuracy achieved!")
        elif overall_acc >= 0.85:
            report.append("✅ GOOD: High accuracy achieved")
        elif overall_acc >= 0.70:
            report.append("⚠️ MODERATE: Some accuracy issues detected")
        else:
            report.append("❌ POOR: Significant accuracy issues detected")
        
        report.append("")
        
        # Component accuracies
        report.append("🔍 COMPONENT ACCURACIES:")
        report.append(f"  Extraction: {results['extraction_accuracy']:.1%}")
        report.append(f"  Validation: {results['validation_accuracy']:.1%}")
        report.append(f"  End-to-End: {results['end_to_end_accuracy']:.1%}")
        
        report.append("")
        
        # Detailed results
        report.append("📋 DETAILED RESULTS:")
        
        # Extraction results
        ext_results = results['detailed_results']['extraction']
        report.append(f"  Extraction Tests: {ext_results['correct_extractions']}/{ext_results['total_tests']} correct")
        
        # Validation results
        val_results = results['detailed_results']['validation']
        report.append(f"  Validation Tests: {val_results['total_correct']}/{val_results['total_tests']} correct")
        
        # End-to-end results
        e2e_results = results['detailed_results']['end_to_end']
        report.append(f"  End-to-End Tests: {e2e_results['correct_end_to_end']}/{e2e_results['total_tests']} correct")
        
        return "\n".join(report)


def main():
    """Main function to run accuracy tests"""
    tester = CertificateValidationTester()
    
    print("Starting Certificate Accuracy Tests...")
    print("=" * 50)
    
    # Run comprehensive accuracy tests
    results = tester.run_comprehensive_accuracy_tests()
    
    # Generate and display report
    report = tester.generate_accuracy_report(results)
    print(report)
    
    # Save results
    tester.save_accuracy_results(results)
    
    # Return appropriate exit code
    if results['overall_accuracy'] >= 0.95:
        print("\n🎉 Near-perfect accuracy achieved!")
        return 0
    elif results['overall_accuracy'] >= 0.85:
        print("\n✅ High accuracy achieved")
        return 0
    elif results['overall_accuracy'] >= 0.70:
        print("\n⚠️ Moderate accuracy - improvements needed")
        return 1
    else:
        print("\n❌ Poor accuracy - significant improvements needed")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 