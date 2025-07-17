#!/usr/bin/env python3
"""
Comprehensive certificate generation test with multiple models.
Tests various barrier certificate scenarios with robust extraction.
"""

import os
import sys
import time
import json
import re
import traceback
from typing import Dict, List, Any, Optional

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    import torch  # type: ignore
    from fm_llm_solver.core.types import ModelConfig, ModelProvider
    from fm_llm_solver.services.model_provider import ModelProviderFactory  # type: ignore
    from fm_llm_solver.core.logging import get_logger
    from utils.config_loader import load_config

    logger = get_logger(__name__)
    
except Exception as e:
    print(f"Error importing: {e}")
    sys.exit(1)


class RobustCertificateExtractor:
    """Enhanced certificate extraction with multiple strategies."""
    
    def __init__(self):
        self.extraction_patterns = [
            # Python function with various names
            (r'def\s+(barrier_certificate|compute_barrier|barrier_function|certificate)\s*\([^)]*\):\s*\n((?:\s+.*\n?)*)', 'python_function'),
            # Lambda functions
            (r'(\w+\s*=\s*lambda\s+[^:]*:\s*[^,\n]+)', 'lambda_function'),
            # Return statements
            (r'return\s+([x+y\-\*\/\^\(\)\s\d\.]+(?:\s*[+\-*/]\s*[x+y\-\*\/\^\(\)\s\d\.]+)*)', 'return_expression'),
            # Direct expressions with assignment
            (r'V\s*\([^)]*\)\s*=\s*([^,\n]+)', 'assignment_expression'),
            # Mathematical expressions
            (r'([x+y]\s*\*\*?\s*\d+(?:\s*[+\-]\s*[x+y]\s*\*\*?\s*\d+)*(?:\s*[+\-]\s*\d+)?)', 'math_expression'),
        ]

    def extract_certificate(self, text: str) -> Dict[str, Any]:
        """Extract certificate using multiple strategies."""
        results = {
            'success': False,
            'function': None,
            'expression': None,
            'method': None,
            'confidence': 0.0,
            'raw_output': text
        }
        
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Try each extraction pattern
        for pattern, method in self.extraction_patterns:
            match = re.search(pattern, cleaned_text, re.MULTILINE | re.IGNORECASE)
            if match:
                extracted = match.group(0).strip()
                if self.validate_certificate(extracted):
                    results.update({
                        'success': True,
                        'function': extracted if method == 'python_function' else None,
                        'expression': self.extract_expression_from_text(extracted),
                        'method': method,
                        'confidence': self.calculate_confidence(extracted, method)
                    })
                    break
        
        return results

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove code block markers
        text = re.sub(r'```\w*\n?', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_expression_from_text(self, text: str) -> Optional[str]:
        """Extract mathematical expression from text."""
        # Look for return statements
        return_match = re.search(r'return\s+([^,\n]+)', text)
        if return_match:
            return return_match.group(1).strip()
        
        # Look for lambda expressions
        lambda_match = re.search(r'lambda\s+[^:]*:\s*([^,\n]+)', text)
        if lambda_match:
            return lambda_match.group(1).strip()
        
        # Look for mathematical expressions
        math_match = re.search(r'([x+y]\s*[\+\-\*\/\^]+[^,\n]*)', text)
        if math_match:
            return math_match.group(1).strip()
        
        return None

    def validate_certificate(self, text: str) -> bool:
        """Validate if extracted text looks like a valid certificate."""
        if not text:
            return False
        
        # Must contain variables
        has_variables = any(var in text for var in ['x', 'y'])
        
        # Must contain mathematical operations
        has_math = any(op in text for op in ['+', '-', '*', '/', '**', '^'])
        
        # Should be reasonable length
        reasonable_length = 10 <= len(text) <= 500
        
        return has_variables and has_math and reasonable_length

    def calculate_confidence(self, text: str, method: str) -> float:
        """Calculate confidence score for extraction."""
        confidence = 0.5  # Base confidence
        
        # Method-based confidence
        method_scores = {
            'python_function': 0.9,
            'return_expression': 0.8,
            'lambda_function': 0.7,
            'assignment_expression': 0.6,
            'math_expression': 0.5
        }
        confidence = method_scores.get(method, 0.3)
        
        # Content-based adjustments
        if 'barrier' in text.lower():
            confidence += 0.1
        if 'def ' in text:
            confidence += 0.1
        if re.search(r'x\*\*2|y\*\*2', text):
            confidence += 0.1
        
        return min(confidence, 1.0)


class CertificateTestSuite:
    """Comprehensive test suite for certificate generation."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.extractor = RobustCertificateExtractor()
        self.test_cases = self._create_test_cases()
        self.results = []

    def _create_test_cases(self) -> List[Dict[str, Any]]:
        """Create diverse test cases."""
        return [
            {
                'name': 'Linear System - Circular',
                'system': "x' = x + y, y' = -x + y",
                'unsafe_region': "x^2 + y^2 > 4",
                'expected_pattern': r'x\*\*2.*y\*\*2',
                'difficulty': 'easy'
            },
            {
                'name': 'Nonlinear Polynomial',
                'system': "x' = x^2 - y, y' = -x + y^2",
                'unsafe_region': "x^2 + y^2 > 1",
                'expected_pattern': r'polynomial|quadratic',
                'difficulty': 'medium'
            },
            {
                'name': 'Van der Pol Oscillator',
                'system': "x' = y, y' = μ(1 - x^2)y - x where μ = 0.1",
                'unsafe_region': "x > 2 or x < -2",
                'expected_pattern': r'x|barrier',
                'difficulty': 'medium'
            },
            {
                'name': 'Stable Linear System',
                'system': "x' = -x + 2y, y' = -2x - y",
                'unsafe_region': "x^2 + y^2 < 0.1",
                'expected_pattern': r'x\*\*2.*y\*\*2',
                'difficulty': 'easy'
            },
            {
                'name': 'Simple Integrator',
                'system': "x' = y, y' = -x",
                'unsafe_region': "|x| > 3",
                'expected_pattern': r'x|abs',
                'difficulty': 'easy'
            }
        ]

    def create_prompt(self, test_case: Dict[str, Any]) -> str:
        """Create an optimized prompt for the test case."""
        return f"""Generate a barrier certificate function for this dynamical system:

System: {test_case['system']}
Unsafe region: {test_case['unsafe_region']}

Requirements:
1. V(x,y) > 0 in unsafe region
2. Lie derivative ∇V·f ≤ 0 on trajectories

Provide a Python function:
```python
def barrier_certificate(x, y):
    return  # your barrier function here
```
"""

    def test_model_on_cases(self, model_id: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Test a model on all test cases."""
        print(f"\n{'='*60}")
        print(f"TESTING MODEL: {model_id}")
        print(f"{'='*60}")
        
        results = {
            'model_id': model_id,
            'model_name': model_config['display_name'],
            'test_results': [],
            'summary': {
                'total_tests': len(self.test_cases),
                'successful_extractions': 0,
                'high_confidence_extractions': 0,
                'average_confidence': 0.0,
                'average_generation_time': 0.0
            }
        }

        try:
            # Configure model with quantization for memory efficiency
            provider_name = model_config['provider']
            
            config = ModelConfig(
                provider=ModelProvider(provider_name),  # Remove .upper() since enum values are lowercase
                name=model_config['name'],
                trust_remote_code=model_config.get('trust_remote_code', True),
                device="cuda" if torch.cuda.is_available() else "cpu",
                quantization="4bit" if torch.cuda.is_available() else None
            )
            
            print(f"Loading {model_id} with 4-bit quantization...")
            provider = ModelProviderFactory.create(provider_name, config)
            
            load_start = time.time()
            provider.load_model(config)
            load_time = time.time() - load_start
            print(f"Model loaded in {load_time:.2f}s")

            total_confidence = 0.0
            total_generation_time = 0.0
            
            # Test each case
            for i, test_case in enumerate(self.test_cases, 1):
                print(f"\nTest {i}/{len(self.test_cases)}: {test_case['name']}")
                
                prompt = self.create_prompt(test_case)
                
                # Generate with appropriate parameters
                gen_start = time.time()
                try:
                    generated_text = provider.generate_text(
                        prompt=prompt,
                        max_tokens=256,
                        temperature=0.1,  # Low for consistency
                        top_p=0.9
                    )
                    gen_time = time.time() - gen_start
                    total_generation_time += gen_time
                    
                    # Extract certificate
                    extraction = self.extractor.extract_certificate(generated_text)
                    
                    test_result = {
                        'test_case': test_case['name'],
                        'difficulty': test_case['difficulty'],
                        'success': extraction['success'],
                        'confidence': extraction['confidence'],
                        'method': extraction['method'],
                        'expression': extraction['expression'],
                        'generation_time': gen_time,
                        'generated_text': generated_text[:200] + "..." if len(generated_text) > 200 else generated_text
                    }
                    
                    results['test_results'].append(test_result)
                    
                    if extraction['success']:
                        results['summary']['successful_extractions'] += 1
                        total_confidence += extraction['confidence']
                        
                        if extraction['confidence'] > 0.7:
                            results['summary']['high_confidence_extractions'] += 1
                            
                        print(f"  ✅ Success (confidence: {extraction['confidence']:.2f})")
                        print(f"     Expression: {extraction['expression']}")
                    else:
                        print(f"  ❌ Failed to extract certificate")
                        
                except Exception as e:
                    print(f"  ❌ Generation failed: {e}")
                    test_result = {
                        'test_case': test_case['name'],
                        'difficulty': test_case['difficulty'],
                        'success': False,
                        'error': str(e),
                        'generation_time': 0.0
                    }
                    results['test_results'].append(test_result)

            # Calculate summary statistics
            successful = results['summary']['successful_extractions']
            if successful > 0:
                results['summary']['average_confidence'] = total_confidence / successful
            results['summary']['average_generation_time'] = total_generation_time / len(self.test_cases)
            
            success_rate = successful / len(self.test_cases)
            print(f"\nModel Summary:")
            print(f"  Success Rate: {success_rate:.1%} ({successful}/{len(self.test_cases)})")
            print(f"  High Confidence: {results['summary']['high_confidence_extractions']}")
            print(f"  Avg Confidence: {results['summary']['average_confidence']:.2f}")
            print(f"  Avg Gen Time: {results['summary']['average_generation_time']:.2f}s")

        except Exception as e:
            print(f"❌ Model testing failed: {e}")
            results['error'] = str(e)
        
        finally:
            # Cleanup
            try:
                provider.unload_model()
            except:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test across multiple models."""
        print("🚀 COMPREHENSIVE CERTIFICATE GENERATION TEST")
        print("="*80)
        
        # Load configuration
        config = load_config()
        available_models = config['models']['available_models']
        
        # Select models for testing (prioritize smaller ones for 8GB GPU)
        test_models = [
            'qwen2.5-coder-0.5b-instruct',
            'qwen2.5-coder-1.5b-instruct',
            'opencoder-1.5b',
            'qwen2.5-coder-3b-instruct',
            # Larger models with quantization
            'qwen2.5-coder-7b-instruct'
        ]
        
        # Filter to available models
        test_models = [m for m in test_models if m in available_models]
        
        print(f"Testing {len(test_models)} models on {len(self.test_cases)} test cases")
        print(f"Models: {', '.join(test_models)}")
        
        overall_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_models': len(test_models),
            'total_test_cases': len(self.test_cases),
            'model_results': [],
            'summary': {
                'best_model': None,
                'best_success_rate': 0.0,
                'fastest_model': None,
                'fastest_time': float('inf')
            }
        }
        
        for model_id in test_models:
            model_config = available_models[model_id]
            result = self.test_model_on_cases(model_id, model_config)
            overall_results['model_results'].append(result)
            
            # Update best performance tracking
            if 'summary' in result:
                success_rate = result['summary']['successful_extractions'] / result['summary']['total_tests']
                avg_time = result['summary']['average_generation_time']
                
                if success_rate > overall_results['summary']['best_success_rate']:
                    overall_results['summary']['best_success_rate'] = success_rate
                    overall_results['summary']['best_model'] = model_id
                
                if avg_time < overall_results['summary']['fastest_time'] and success_rate > 0:
                    overall_results['summary']['fastest_time'] = avg_time
                    overall_results['summary']['fastest_model'] = model_id
        
        return overall_results

    def save_results(self, results: Dict[str, Any], filename: str = "comprehensive_certificate_results.json"):
        """Save comprehensive results."""
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n📊 Results saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")

    def print_summary(self, results: Dict[str, Any]):
        """Print formatted summary."""
        print("\n" + "🏆 FINAL RESULTS SUMMARY" + " "*50)
        print("="*80)
        
        summary = results['summary']
        print(f"Best Performing Model: {summary['best_model']} ({summary['best_success_rate']:.1%} success)")
        print(f"Fastest Model: {summary['fastest_model']} ({summary['fastest_time']:.2f}s avg)")
        
        print(f"\nDetailed Model Performance:")
        print("-"*80)
        
        for result in results['model_results']:
            if 'error' in result:
                print(f"❌ {result['model_id']}: ERROR - {result['error']}")
                continue
            
            model_id = result['model_id']
            summary = result['summary']
            
            success_rate = summary['successful_extractions'] / summary['total_tests']
            print(f"📊 {result['model_name']} ({model_id}):")
            print(f"   Success Rate: {success_rate:.1%} ({summary['successful_extractions']}/{summary['total_tests']})")
            print(f"   High Confidence: {summary['high_confidence_extractions']}")
            print(f"   Avg Confidence: {summary['average_confidence']:.2f}")
            print(f"   Avg Generation Time: {summary['average_generation_time']:.2f}s")
            print()


def main():
    """Main test execution."""
    print("🎯 Starting Comprehensive Certificate Generation Test")
    
    try:
        # Initialize test suite
        test_suite = CertificateTestSuite()
        
        # Run comprehensive test
        results = test_suite.run_comprehensive_test()
        
        # Save and display results
        test_suite.save_results(results)
        test_suite.print_summary(results)
        
        print("\n✅ Comprehensive testing completed!")
        return results
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main() 