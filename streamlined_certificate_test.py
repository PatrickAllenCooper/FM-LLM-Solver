#!/usr/bin/env python3
"""
Streamlined certificate generation test for multiple models.
Tests with real-time output and progressive results.
"""

import os
import sys
import time
import re
import json
from typing import Dict, List, Any, Optional

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


def extract_certificate_robust(text: str) -> Optional[str]:
    """Robust certificate extraction."""
    patterns = [
        r'def\s+\w+\([^)]*\):\s*\n\s*return\s+([^,\n]+)',  # Function with return
        r'return\s+([x+y\*\-\/\^\(\)\s\d\.]+)',  # Direct return
        r'([x+y]\s*[\+\-\*\/\^]+[^,\n]*)',  # Math expression
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            expr = match.group(1).strip()
            if ('x' in expr or 'y' in expr) and any(op in expr for op in ['+', '-', '*', '/', '^']):
                return expr
    
    return None


def test_single_model(model_id: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Test a single model with real-time output."""
    print(f"\n🔧 TESTING: {model_config['display_name']} ({model_id})")
    print(f"   Parameters: {model_config['parameters']}")
    print(f"   GPU Memory: {model_config['recommended_gpu_memory']}")
    
    result = {
        'model_id': model_id,
        'model_name': model_config['display_name'],
        'success': False,
        'load_time': 0.0,
        'generation_time': 0.0,
        'certificate': None,
        'error': None
    }
    
    try:
        # Configure with quantization
        provider_name = model_config['provider']
        config = ModelConfig(
            provider=ModelProvider(provider_name),
            name=model_config['name'],
            trust_remote_code=model_config.get('trust_remote_code', True),
            device="cuda",
            quantization="4bit"
        )
        
        print(f"   🔄 Loading with 4-bit quantization...")
        
        # Load model
        provider = ModelProviderFactory.create(provider_name, config)
        
        load_start = time.time()
        provider.load_model(config)
        load_time = time.time() - load_start
        result['load_time'] = load_time
        
        print(f"   ✅ Loaded in {load_time:.1f}s")
        
        # Test prompt
        prompt = """Generate a barrier certificate function for this system:

System: x' = x + y, y' = -x + y  
Unsafe region: x^2 + y^2 > 4

Provide a Python function:
```python
def barrier_certificate(x, y):
    return  # your barrier function
```
"""
        
        print(f"   🧠 Generating certificate...")
        
        # Generate
        gen_start = time.time()
        generated_text = provider.generate_text(
            prompt=prompt,
            max_tokens=256,
            temperature=0.1
        )
        gen_time = time.time() - gen_start
        result['generation_time'] = gen_time
        
        print(f"   ⚡ Generated in {gen_time:.1f}s")
        
        # Extract certificate
        certificate = extract_certificate_robust(generated_text)
        
        if certificate:
            result['success'] = True
            result['certificate'] = certificate
            print(f"   ✅ SUCCESS: {certificate}")
        else:
            result['error'] = "Could not extract valid certificate"
            print(f"   ❌ FAILED: No valid certificate extracted")
            print(f"      Output: {generated_text[:100]}...")
        
        # Cleanup
        provider.unload_model()
        
    except Exception as e:
        result['error'] = str(e)
        print(f"   ❌ ERROR: {e}")
    
    finally:
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return result


def main():
    """Main test execution."""
    print("🚀 STREAMLINED CERTIFICATE GENERATION TEST")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    available_models = config['models']['available_models']
    
    # Select models for testing (from smallest to largest)
    test_models = [
        'qwen2.5-coder-0.5b-instruct',
        'qwen2.5-coder-1.5b-instruct',
        'opencoder-1.5b', 
        'qwen2.5-coder-3b-instruct',
        'qwen2.5-coder-7b-instruct'
    ]
    
    # Filter available models
    test_models = [m for m in test_models if m in available_models]
    
    print(f"📊 Testing {len(test_models)} models:")
    for i, model_id in enumerate(test_models, 1):
        print(f"   {i}. {available_models[model_id]['display_name']}")
    
    # Run tests
    results = []
    successful_tests = 0
    
    for i, model_id in enumerate(test_models, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}/{len(test_models)}")
        
        model_config = available_models[model_id]
        result = test_single_model(model_id, model_config)
        results.append(result)
        
        if result['success']:
            successful_tests += 1
        
        # Show memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   🔋 GPU Memory: {memory_used:.1f}/{memory_total:.1f} GB")
    
    # Final summary
    print(f"\n🏆 FINAL RESULTS")
    print("=" * 60)
    print(f"Success Rate: {successful_tests}/{len(test_models)} ({successful_tests/len(test_models)*100:.1f}%)")
    
    print(f"\nModel Performance:")
    print("-" * 40)
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        load_time = result['load_time']
        gen_time = result['generation_time']
        
        print(f"{status} {result['model_name']}")
        print(f"    Load: {load_time:.1f}s | Generate: {gen_time:.1f}s")
        
        if result['success']:
            print(f"    Certificate: {result['certificate']}")
        else:
            print(f"    Error: {result['error']}")
        print()
    
    # Save results
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_models': len(test_models),
        'successful_models': successful_tests,
        'success_rate': successful_tests / len(test_models),
        'results': results
    }
    
    with open('certificate_test_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"📋 Results saved to certificate_test_summary.json")
    
    if successful_tests > 0:
        print(f"\n🎉 SUCCESS: {successful_tests} models can generate barrier certificates!")
    else:
        print(f"\n⚠️  No models successfully generated certificates")
    
    return summary


if __name__ == "__main__":
    results = main() 