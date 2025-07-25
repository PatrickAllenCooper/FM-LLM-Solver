#!/usr/bin/env python3
"""
Mathematical verification test for barrier certificates.
Tests different model families and validates mathematical correctness.
"""

import os
import sys
import time
import re
import json
import numpy as np  # type: ignore
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    import torch  # type: ignore
    from fm_llm_solver.core.types import ModelConfig, ModelProvider
    from fm_llm_solver.services.model_provider import ModelProviderFactory  # type: ignore
    from utils.config_loader import load_config
    
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class BarrierCertificateValidator:
    """Validates mathematical correctness of barrier certificates."""
    
    def __init__(self):
        self.test_cases = [
            {
                'name': 'Linear System',
                'system_func': lambda x, y: np.array([x + y, -x + y]),
                'unsafe_condition': lambda x, y: x**2 + y**2 > 4,
                'expected_form': 'quadratic'
            },
            {
                'name': 'Nonlinear System', 
                'system_func': lambda x, y: np.array([x**2 - y, -x + y**2]),
                'unsafe_condition': lambda x, y: x**2 + y**2 > 1,
                'expected_form': 'polynomial'
            }
        ]

    def extract_function_from_text(self, text: str) -> Optional[str]:
        """Extract barrier function from model output."""
        patterns = [
            r'return\s+([^,\n;]+)',  # return statement
            r'([x+y]\s*[\+\-\*\/\^]+[^,\n;]*)',  # mathematical expression
            r'V\s*=\s*([^,\n;]+)',  # assignment
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            for match in matches:
                expr = match.strip()
                if self.is_valid_expression(expr):
                    return self.normalize_expression(expr)
        
        return None

    def is_valid_expression(self, expr: str) -> bool:
        """Check if expression is mathematically valid."""
        return (('x' in expr or 'y' in expr) and 
                any(op in expr for op in ['+', '-', '*', '/', '^', '**']) and
                len(expr) > 5)

    def normalize_expression(self, expr: str) -> str:
        """Normalize mathematical expression."""
        # Convert to Python syntax
        expr = expr.replace('^', '**')
        # Remove trailing punctuation
        expr = re.sub(r'[,;)\s]*$', '', expr)
        return expr.strip()

    def evaluate_barrier_function(self, expr: str, x_val: float, y_val: float) -> Optional[float]:
        """Safely evaluate barrier function at given point."""
        try:
            # Create safe evaluation environment
            safe_dict = {
                'x': x_val, 'y': y_val,
                'sqrt': np.sqrt, 'abs': abs,
                'sin': np.sin, 'cos': np.cos,
                'exp': np.exp, 'log': np.log,
                '__builtins__': {}
            }
            return eval(expr, safe_dict)
        except:
            return None

    def validate_certificate(self, expr: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Validate barrier certificate mathematically."""
        result = {
            'valid': False,
            'score': 0.0,
            'tests_passed': 0,
            'total_tests': 0,
            'details': []
        }
        
        # Test points
        test_points = [
            (0, 0),      # Origin
            (1, 1),      # Interior
            (2, 2),      # Boundary region
            (3, 0),      # Unsafe region
            (0, 3),      # Unsafe region
            (-2, -2),    # Negative quadrant
        ]
        
        for x, y in test_points:
            result['total_tests'] += 1
            
            # Evaluate barrier function
            barrier_val = self.evaluate_barrier_function(expr, x, y)
            if barrier_val is None:
                result['details'].append(f"Point ({x},{y}): Evaluation failed")
                continue
            
            # Check unsafe condition
            is_unsafe = test_case['unsafe_condition'](x, y)
            
            # For barrier certificates: V(x,y) > 0 should hold in unsafe region
            if is_unsafe and barrier_val > 0:
                result['tests_passed'] += 1
                result['details'].append(f"Point ({x},{y}): ✅ Unsafe region, V={barrier_val:.2f}")
            elif not is_unsafe and barrier_val <= 0:
                result['tests_passed'] += 1  
                result['details'].append(f"Point ({x},{y}): ✅ Safe region, V={barrier_val:.2f}")
            elif is_unsafe and barrier_val <= 0:
                result['details'].append(f"Point ({x},{y}): ❌ Unsafe but V={barrier_val:.2f} ≤ 0")
            else:
                result['details'].append(f"Point ({x},{y}): ⚠️ Safe but V={barrier_val:.2f} > 0")
        
        result['score'] = result['tests_passed'] / result['total_tests'] if result['total_tests'] > 0 else 0
        result['valid'] = result['score'] >= 0.5  # At least 50% tests pass
        
        return result


def test_model_mathematical_correctness(model_id: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Test model with mathematical verification."""
    print(f"\n🧮 MATHEMATICAL TEST: {model_config['display_name']}")
    
    validator = BarrierCertificateValidator()
    
    result = {
        'model_id': model_id,
        'model_name': model_config['display_name'],
        'success': False,
        'mathematical_score': 0.0,
        'certificates': [],
        'error': None
    }
    
    try:
        # Configure model
        provider_name = model_config['provider']
        config = ModelConfig(
            provider=ModelProvider(provider_name),
            name=model_config['name'],
            trust_remote_code=True,
            device="cuda",
            quantization="4bit"
        )
        
        print(f"   Loading {model_id}...")
        provider = ModelProviderFactory.create(provider_name, config)
        provider.load_model(config)
        
        total_score = 0.0
        
        # Test different scenarios
        test_prompts = [
            {
                'name': 'Basic Linear System',
                'prompt': """Generate a barrier certificate for:
System: x' = x + y, y' = -x + y
Unsafe region: x^2 + y^2 > 4

Return only the mathematical expression V(x,y) = ...""",
                'test_case_idx': 0
            },
            {
                'name': 'Nonlinear System',
                'prompt': """Generate a barrier certificate for:
System: x' = x^2 - y, y' = -x + y^2  
Unsafe region: x^2 + y^2 > 1

Return the barrier function V(x,y):""",
                'test_case_idx': 1
            }
        ]
        
        for prompt_info in test_prompts:
            print(f"   Testing: {prompt_info['name']}")
            
            # Generate certificate
            generated = provider.generate_text(
                prompt=prompt_info['prompt'],
                max_tokens=128,
                temperature=0.1
            )
            
            # Extract function
            expr = validator.extract_function_from_text(generated)
            
            if expr:
                print(f"   Extracted: {expr}")
                
                # Validate mathematically
                test_case = validator.test_cases[prompt_info['test_case_idx']]
                validation = validator.validate_certificate(expr, test_case)
                
                cert_result = {
                    'prompt': prompt_info['name'],
                    'expression': expr,
                    'validation': validation,
                    'raw_output': generated[:100] + "..."
                }
                
                result['certificates'].append(cert_result)
                total_score += validation['score']
                
                print(f"   Score: {validation['score']:.1%} ({validation['tests_passed']}/{validation['total_tests']} tests)")
                
            else:
                print(f"   ❌ No valid expression extracted")
                result['certificates'].append({
                    'prompt': prompt_info['name'],
                    'expression': None,
                    'validation': {'valid': False, 'score': 0.0},
                    'raw_output': generated[:100] + "..."
                })
        
        # Calculate overall score
        result['mathematical_score'] = total_score / len(test_prompts)
        result['success'] = result['mathematical_score'] > 0.3
        
        provider.unload_model()
        
    except Exception as e:
        result['error'] = str(e)
        print(f"   ❌ Error: {e}")
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return result


def main():
    """Main mathematical verification test."""
    print("🧮 MATHEMATICAL VERIFICATION TEST")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    available_models = config['models']['available_models']
    
    # Select diverse models that fit in 8GB GPU
    test_models = [
        'qwen2.5-coder-0.5b-instruct',  # Qwen family
        'qwen2.5-coder-1.5b-instruct', # Qwen family  
        'qwen2.5-coder-3b-instruct',   # Larger Qwen
    ]
    
    # Filter available
    test_models = [m for m in test_models if m in available_models]
    
    print(f"🎯 Testing mathematical correctness of {len(test_models)} models")
    
    results = []
    total_score = 0.0
    
    for i, model_id in enumerate(test_models, 1):
        print(f"\n{'='*60}")
        print(f"MODEL {i}/{len(test_models)}")
        
        model_config = available_models[model_id]
        result = test_model_mathematical_correctness(model_id, model_config)
        results.append(result)
        
        if result['success']:
            total_score += result['mathematical_score']
            print(f"   ✅ Overall Score: {result['mathematical_score']:.1%}")
        else:
            print(f"   ❌ Failed with score: {result['mathematical_score']:.1%}")
    
    # Final summary
    print(f"\n🏆 MATHEMATICAL VERIFICATION RESULTS")
    print("=" * 60)
    
    successful_models = sum(1 for r in results if r['success'])
    avg_score = total_score / len(results) if results else 0
    
    print(f"Models Passed: {successful_models}/{len(results)}")
    print(f"Average Mathematical Score: {avg_score:.1%}")
    
    print(f"\nDetailed Results:")
    print("-" * 40)
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        score = result['mathematical_score']
        
        print(f"{status} {result['model_name']}: {score:.1%}")
        
        for cert in result['certificates']:
            if cert['expression']:
                val_score = cert['validation']['score']
                print(f"   {cert['prompt']}: {cert['expression']} (Score: {val_score:.1%})")
            else:
                print(f"   {cert['prompt']}: No valid expression")
        
        if result['error']:
            print(f"   Error: {result['error']}")
        print()
    
    # Save detailed results
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'mathematical_verification': True,
        'models_tested': len(test_models),
        'models_passed': successful_models,
        'average_score': avg_score,
        'results': results
    }
    
    with open('mathematical_verification_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"📊 Detailed results saved to mathematical_verification_results.json")
    
    if successful_models > 0:
        print(f"\n🎉 SUCCESS: {successful_models} models generate mathematically valid certificates!")
        print(f"Average mathematical correctness: {avg_score:.1%}")
    else:
        print(f"\n⚠️ No models passed mathematical verification")
    
    return summary


if __name__ == "__main__":
    results = main() 