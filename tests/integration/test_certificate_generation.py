#!/usr/bin/env python3
"""
Test script for barrier certificate generation using code generation models.

This script tests all available models with various barrier certificate problems,
using quantization for GPU memory management and robust output parsing.
"""

import os
import sys
import time
import json
import re
import traceback
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import subprocess

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import gc
from fm_llm_solver.services.model_manager import get_model_manager
from fm_llm_solver.services.model_downloader import get_model_downloader
from fm_llm_solver.core.logging import get_logger
from utils.config_loader import load_config

logger = get_logger(__name__)


class CertificateExtractor:
    """Robust certificate extraction from model outputs."""
    
    def __init__(self):
        self.function_patterns = [
            # Python function definitions
            r'def\s+(\w+)\s*\([^)]*\):\s*\n((?:\s+.*\n?)*)',
            # Single line function
            r'(\w+)\s*=\s*lambda\s+[^:]*:\s*(.+)',
            # Mathematical expression
            r'(?:barrier|certificate|function).*?[:=]\s*([^,\n]+)',
            # Direct mathematical expressions
            r'(?:x|y|\w+)\s*[+\-*/]\s*(?:x|y|\w+)(?:\s*[+\-*/]\s*(?:x|y|\w+))*'
        ]
        
        # Common variable substitutions for robustness
        self.variable_mappings = {
            'x1': 'x', 'x2': 'y', 'x_1': 'x', 'x_2': 'y',
            'state[0]': 'x', 'state[1]': 'y',
            'vars[0]': 'x', 'vars[1]': 'y'
        }

    def extract_python_function(self, text: str) -> Optional[str]:
        """Extract Python function from text."""
        # Look for function definitions
        func_pattern = r'def\s+\w+\s*\([^)]*\):\s*\n((?:\s+.*\n?)*)'
        match = re.search(func_pattern, text, re.MULTILINE)
        if match:
            return match.group(0).strip()
        
        # Look for lambda functions
        lambda_pattern = r'(\w+\s*=\s*lambda\s+[^:]*:\s*.+)'
        match = re.search(lambda_pattern, text)
        if match:
            return match.group(1).strip()
        
        return None

    def extract_mathematical_expression(self, text: str) -> Optional[str]:
        """Extract mathematical expression from text."""
        # Common patterns for barrier functions
        patterns = [
            r'(?:return\s+|=\s*)([x+y\-\*\/\^\(\)\s\d\.]+)',
            r'barrier.*?[:=]\s*([^,\n]+)',
            r'V\s*[:=]\s*([^,\n]+)',
            r'(?:x|y)\s*[\+\-\*\/]\s*(?:x|y)(?:\s*[\+\-\*\/]\s*(?:x|y|\d+))*'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                expr = matches[0].strip()
                # Clean up the expression
                expr = re.sub(r'\s+', ' ', expr)
                if self.is_valid_expression(expr):
                    return expr
        
        return None

    def is_valid_expression(self, expr: str) -> bool:
        """Check if expression is a valid mathematical expression."""
        if not expr:
            return False
        
        # Check for required variables
        has_x = 'x' in expr
        has_y = 'y' in expr
        
        # Check for mathematical operators or functions
        has_math = any(op in expr for op in ['+', '-', '*', '/', '**', '^', 'sqrt', 'sin', 'cos'])
        
        return (has_x or has_y) and has_math

    def normalize_expression(self, expr: str) -> str:
        """Normalize mathematical expression."""
        # Replace common variations
        for old, new in self.variable_mappings.items():
            expr = expr.replace(old, new)
        
        # Handle power notation
        expr = expr.replace('^', '**')
        
        # Clean up spaces
        expr = re.sub(r'\s+', ' ', expr).strip()
        
        return expr

    def extract_certificate(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract barrier certificate from model output."""
        result = {
            'function': None,
            'expression': None,
            'raw_output': text,
            'extraction_method': None
        }
        
        # Try to extract Python function first
        func = self.extract_python_function(text)
        if func:
            result['function'] = func
            result['extraction_method'] = 'python_function'
            
            # Also try to extract the mathematical expression from within function
            expr = self.extract_mathematical_expression(func)
            if expr:
                result['expression'] = self.normalize_expression(expr)
            
            return result
        
        # Try to extract mathematical expression
        expr = self.extract_mathematical_expression(text)
        if expr:
            result['expression'] = self.normalize_expression(expr)
            result['extraction_method'] = 'mathematical_expression'
            return result
        
        # If nothing found, return None
        return None


class CertificateTestCase:
    """A test case for barrier certificate generation."""
    
    def __init__(self, name: str, system: str, unsafe_region: str, 
                 expected_type: str = "polynomial", description: str = ""):
        self.name = name
        self.system = system
        self.unsafe_region = unsafe_region
        self.expected_type = expected_type
        self.description = description

    def generate_prompt(self) -> str:
        """Generate a prompt for the model."""
        return f"""Generate a barrier certificate function for the following dynamical system:

System dynamics: {self.system}
Unsafe region: {self.unsafe_region}

Please provide a Python function that computes a barrier certificate V(x, y) such that:
1. V(x, y) > 0 in the unsafe region
2. The Lie derivative ∇V · f ≤ 0 along system trajectories

Example format:
```python
def barrier_certificate(x, y):
    return x**2 + y**2 - 1  # Example barrier function
```

Barrier certificate:"""


class ModelCertificateTest:
    """Test framework for certificate generation with various models."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.model_manager = get_model_manager()
        self.downloader = get_model_downloader()
        self.extractor = CertificateExtractor()
        
        # Test configuration
        self.test_cases = self._create_test_cases()
        self.results = {}
        
        # GPU memory management
        self.max_memory_gb = 7.5  # Leave some headroom
        
    def _create_test_cases(self) -> List[CertificateTestCase]:
        """Create test cases for certificate generation."""
        return [
            CertificateTestCase(
                name="Linear System",
                system="x' = x + y, y' = -x + y",
                unsafe_region="x^2 + y^2 > 4",
                expected_type="quadratic",
                description="Simple linear system with circular unsafe region"
            ),
            CertificateTestCase(
                name="Nonlinear Polynomial",
                system="x' = x^2 - y, y' = -x + y^2",
                unsafe_region="x^2 + y^2 > 1",
                expected_type="polynomial",
                description="Nonlinear polynomial system"
            ),
            CertificateTestCase(
                name="Van der Pol Oscillator",
                system="x' = y, y' = μ(1 - x^2)y - x",
                unsafe_region="x > 2",
                expected_type="polynomial",
                description="Classic Van der Pol oscillator with half-plane unsafe region"
            ),
            CertificateTestCase(
                name="Predator-Prey",
                system="x' = x(a - by), y' = y(cx - d)",
                unsafe_region="x < 0.1 or y < 0.1",
                expected_type="rational",
                description="Predator-prey model with extinction regions"
            ),
            CertificateTestCase(
                name="Quadratic System",
                system="x' = -x + xy, y' = -y + x^2",
                unsafe_region="x^2 + y^2 < 0.1",
                expected_type="polynomial",
                description="Quadratic system with small circular unsafe region"
            )
        ]

    def get_available_models(self) -> List[str]:
        """Get list of models suitable for testing."""
        available_models = self.model_manager.get_available_models()
        
        # Sort by estimated memory usage (smallest first)
        model_order = [
            "qwen2.5-coder-0.5b-instruct",
            "qwen2.5-coder-1.5b-instruct", 
            "opencoder-1.5b",
            "codegemma-2b",
            "qwen2.5-coder-3b-instruct",
            "starcoder2-3b",
            "codegemma-7b-instruct",
            "qwen2.5-coder-7b-instruct",
            "codellama-7b-instruct",
            "starcoder2-7b",
            "opencoder-8b",
            "deepseek-coder-v2-lite-instruct"
        ]
        
        # Filter to only available models
        return [model_id for model_id in model_order if model_id in available_models]

    def estimate_model_memory(self, model_id: str) -> float:
        """Estimate model memory usage in GB."""
        available_models = self.model_manager.get_available_models()
        if model_id not in available_models:
            return float('inf')
        
        model_config = available_models[model_id]
        
        # Simple heuristic based on parameter count
        params = model_config.get('parameters', '0B')
        if 'B' in params:
            param_count = float(params.replace('B', ''))
            # Rough estimate: 2 bytes per parameter (FP16) + overhead
            return param_count * 2.5
        
        return 8.0  # Default conservative estimate

    def clear_gpu_memory(self):
        """Clear GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def download_model_if_needed(self, model_id: str) -> bool:
        """Download model if not already available."""
        if self.downloader.is_model_downloaded(model_id):
            self.logger.info(f"Model {model_id} already downloaded")
            return True
        
        try:
            available_models = self.model_manager.get_available_models()
            if model_id not in available_models:
                self.logger.error(f"Model {model_id} not in configuration")
                return False
            
            model_config = available_models[model_id]
            self.logger.info(f"Downloading model {model_id}...")
            
            cache_path = self.downloader.download_model(model_id, model_config)
            self.logger.info(f"Downloaded {model_id} to {cache_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download {model_id}: {e}")
            return False

    def test_model_on_case(self, model_id: str, test_case: CertificateTestCase) -> Dict[str, Any]:
        """Test a specific model on a test case."""
        start_time = time.time()
        result = {
            'model_id': model_id,
            'test_case': test_case.name,
            'success': False,
            'execution_time': 0.0,
            'error': None,
            'certificate': None,
            'generated_text': None
        }
        
        try:
            # Generate prompt
            prompt = test_case.generate_prompt()
            
            # Generate with model
            self.logger.info(f"Testing {model_id} on {test_case.name}")
            
            generated_text = self.model_manager.generate_text(
                prompt=prompt,
                max_tokens=512,
                temperature=0.1  # Low temperature for consistent results
            )
            
            result['generated_text'] = generated_text
            
            # Extract certificate
            certificate = self.extractor.extract_certificate(generated_text)
            
            if certificate:
                result['certificate'] = certificate
                result['success'] = True
                self.logger.info(f"Successfully extracted certificate for {test_case.name}")
            else:
                result['error'] = "Could not extract valid certificate from output"
                self.logger.warning(f"Failed to extract certificate for {test_case.name}")
            
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error testing {model_id} on {test_case.name}: {e}")
            
        result['execution_time'] = time.time() - start_time
        return result

    def test_model(self, model_id: str, use_quantization: bool = False) -> Dict[str, Any]:
        """Test a model on all test cases."""
        model_results = {
            'model_id': model_id,
            'quantized': use_quantization,
            'test_results': [],
            'summary': {
                'total_tests': len(self.test_cases),
                'successful_tests': 0,
                'success_rate': 0.0,
                'average_time': 0.0,
                'total_time': 0.0
            }
        }
        
        try:
            # Download model if needed
            if not self.download_model_if_needed(model_id):
                model_results['error'] = f"Failed to download model {model_id}"
                return model_results
            
            # Clear GPU memory before loading
            self.clear_gpu_memory()
            
            # Load model with optional quantization
            self.logger.info(f"Loading model {model_id} (quantized: {use_quantization})")
            
            # Configure quantization if needed
            if use_quantization:
                # We'll modify the model config to use quantization
                available_models = self.model_manager.get_available_models()
                model_config = available_models[model_id].copy()
                model_config['quantization'] = '4bit'  # Use 4-bit quantization
            
            success = self.model_manager.switch_model(model_id)
            
            if not success:
                model_results['error'] = f"Failed to load model {model_id}"
                return model_results
            
            # Run tests
            total_time = 0.0
            successful_tests = 0
            
            for test_case in self.test_cases:
                try:
                    test_result = self.test_model_on_case(model_id, test_case)
                    model_results['test_results'].append(test_result)
                    
                    total_time += test_result['execution_time']
                    if test_result['success']:
                        successful_tests += 1
                        
                except Exception as e:
                    self.logger.error(f"Error in test case {test_case.name}: {e}")
                    model_results['test_results'].append({
                        'model_id': model_id,
                        'test_case': test_case.name,
                        'success': False,
                        'error': str(e),
                        'execution_time': 0.0
                    })
            
            # Calculate summary
            model_results['summary'].update({
                'successful_tests': successful_tests,
                'success_rate': successful_tests / len(self.test_cases) if self.test_cases else 0.0,
                'average_time': total_time / len(self.test_cases) if self.test_cases else 0.0,
                'total_time': total_time
            })
            
        except Exception as e:
            model_results['error'] = f"Model testing failed: {e}"
            self.logger.error(f"Failed to test model {model_id}: {e}")
            traceback.print_exc()
        
        finally:
            # Clean up GPU memory
            try:
                self.model_manager.unload_model(model_id)
            except:
                pass
            self.clear_gpu_memory()
        
        return model_results

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test across all available models."""
        self.logger.info("Starting comprehensive certificate generation test")
        
        available_models = self.get_available_models()
        self.logger.info(f"Testing {len(available_models)} models")
        
        test_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_models': len(available_models),
            'model_results': [],
            'summary': {
                'models_tested': 0,
                'models_successful': 0,
                'average_success_rate': 0.0,
                'best_model': None,
                'fastest_model': None
            }
        }
        
        for model_id in available_models:
            try:
                # Estimate memory requirements
                estimated_memory = self.estimate_model_memory(model_id)
                use_quantization = estimated_memory > self.max_memory_gb
                
                if use_quantization:
                    self.logger.info(f"Using quantization for {model_id} (estimated {estimated_memory:.1f}GB)")
                
                # Test the model
                model_result = self.test_model(model_id, use_quantization)
                test_results['model_results'].append(model_result)
                
                # Log results
                summary = model_result.get('summary', {})
                success_rate = summary.get('success_rate', 0.0)
                avg_time = summary.get('average_time', 0.0)
                
                self.logger.info(f"Model {model_id}: {success_rate:.1%} success rate, {avg_time:.2f}s avg time")
                
            except Exception as e:
                self.logger.error(f"Failed to test model {model_id}: {e}")
                test_results['model_results'].append({
                    'model_id': model_id,
                    'error': str(e),
                    'summary': {'success_rate': 0.0, 'average_time': 0.0}
                })
        
        # Calculate overall summary
        successful_models = 0
        total_success_rate = 0.0
        best_model = None
        fastest_model = None
        best_success_rate = 0.0
        fastest_time = float('inf')
        
        for model_result in test_results['model_results']:
            if 'error' not in model_result:
                successful_models += 1
                summary = model_result['summary']
                success_rate = summary['success_rate']
                avg_time = summary['average_time']
                
                total_success_rate += success_rate
                
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_model = model_result['model_id']
                
                if avg_time < fastest_time and success_rate > 0:
                    fastest_time = avg_time
                    fastest_model = model_result['model_id']
        
        test_results['summary'].update({
            'models_tested': successful_models,
            'models_successful': successful_models,
            'average_success_rate': total_success_rate / successful_models if successful_models > 0 else 0.0,
            'best_model': best_model,
            'fastest_model': fastest_model
        })
        
        return test_results

    def save_results(self, results: Dict[str, Any], filename: str = "certificate_test_results.json"):
        """Save test results to file."""
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Results saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")

    def print_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of test results."""
        print("\n" + "="*80)
        print("BARRIER CERTIFICATE GENERATION TEST RESULTS")
        print("="*80)
        
        summary = results['summary']
        print(f"Total Models Tested: {summary['models_tested']}")
        print(f"Average Success Rate: {summary['average_success_rate']:.1%}")
        print(f"Best Model: {summary['best_model']}")
        print(f"Fastest Model: {summary['fastest_model']}")
        
        print("\nDetailed Results:")
        print("-"*80)
        
        for model_result in results['model_results']:
            if 'error' in model_result:
                print(f"{model_result['model_id']}: ERROR - {model_result['error']}")
                continue
            
            model_id = model_result['model_id']
            summary = model_result['summary']
            quantized = model_result.get('quantized', False)
            
            print(f"{model_id} {'(4-bit)' if quantized else ''}:")
            print(f"  Success Rate: {summary['success_rate']:.1%} ({summary['successful_tests']}/{summary['total_tests']})")
            print(f"  Average Time: {summary['average_time']:.2f}s")
            print(f"  Total Time: {summary['total_time']:.2f}s")
            
            # Show successful extractions
            successful_cases = [r for r in model_result['test_results'] if r['success']]
            if successful_cases:
                print(f"  Successful Cases: {', '.join(r['test_case'] for r in successful_cases)}")
            print()


def main():
    """Main test execution."""
    print("Starting Certificate Generation Testing with Code Generation Models")
    print("="*80)
    
    # Initialize test framework
    tester = ModelCertificateTest()
    
    try:
        # Run comprehensive test
        results = tester.run_comprehensive_test()
        
        # Save results
        tester.save_results(results)
        
        # Print summary
        tester.print_summary(results)
        
        return results
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main() 