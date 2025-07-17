#!/usr/bin/env python3
"""
Simple test script for barrier certificate generation.
Tests with one model to verify basic functionality.
"""

import os
import sys
import time
import re
import traceback

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    import torch  # type: ignore
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except ImportError:
    print("PyTorch not available")

# Test the model loading
try:
    from fm_llm_solver.core.types import ModelConfig, ModelProvider
    from fm_llm_solver.services.model_provider import ModelProviderFactory  # type: ignore
    from fm_llm_solver.core.logging import get_logger

    logger = get_logger(__name__)
    print("Successfully imported FM-LLM Solver components")
    
except Exception as e:
    print(f"Error importing components: {e}")
    traceback.print_exc()


def extract_certificate_simple(text: str) -> str:
    """Simple certificate extraction from model output."""
    # Look for Python function definitions
    func_pattern = r'def\s+\w+\s*\([^)]*\):\s*\n((?:\s+.*\n?)*)'
    match = re.search(func_pattern, text, re.MULTILINE)
    if match:
        return match.group(0).strip()
    
    # Look for return statements with mathematical expressions
    return_pattern = r'return\s+([x+y\-\*\/\^\(\)\s\d\.]+)'
    match = re.search(return_pattern, text)
    if match:
        return f"V(x, y) = {match.group(1).strip()}"
    
    # Look for any mathematical expression with x and y
    expr_pattern = r'([x+y]\s*[\+\-\*\/\^]\s*[x+y\d\(\)]+)'
    match = re.search(expr_pattern, text)
    if match:
        return f"V(x, y) = {match.group(1).strip()}"
    
    return "No certificate found"


def test_model_availability():
    """Test which models are available."""
    print("\n" + "="*60)
    print("CHECKING MODEL AVAILABILITY")
    print("="*60)
    
    try:
        # Check if we can import and list models
        from utils.config_loader import load_config
        
        config = load_config()
        print(f"Config keys: {list(config.keys())}")
        
        if 'models' in config and 'available_models' in config['models']:
            available_models = config['models']['available_models']
            print(f"Total models configured: {len(available_models)}")
            
            for model_id, model_config in available_models.items():
                print(f"- {model_id}: {model_config.get('display_name', 'Unknown')}")
                
            return available_models
        else:
            print("No models found in configuration")
            print("Available config sections:", list(config.keys()))
            return {}
            
    except Exception as e:
        print(f"Error checking model availability: {e}")
        traceback.print_exc()
        return {}


def test_certificate_generation():
    """Test certificate generation with a simple model."""
    print("\n" + "="*60)
    print("TESTING BARRIER CERTIFICATE GENERATION")
    print("="*60)
    
    # Simple test case
    prompt = """Generate a barrier certificate function for the following dynamical system:

System dynamics: x' = x + y, y' = -x + y
Unsafe region: x^2 + y^2 > 4

Please provide a Python function that computes a barrier certificate V(x, y) such that:
1. V(x, y) > 0 in the unsafe region
2. The Lie derivative ∇V · f ≤ 0 along system trajectories

Example format:
```python
def barrier_certificate(x, y):
    return x**2 + y**2 - 1  # Example barrier function
```

Barrier certificate:"""

    try:
        # Get available models first
        available_models = test_model_availability()
        if not available_models:
            print("No models available for testing")
            return False
        
        # Choose the smallest model for testing
        test_models = [
            "qwen2.5-coder-0.5b-instruct",
            "qwen2.5-coder-1.5b-instruct", 
            "opencoder-1.5b"
        ]
        
        test_model_id = None
        for model_id in test_models:
            if model_id in available_models:
                test_model_id = model_id
                break
        
        if not test_model_id:
            print("No suitable test model found")
            print("Available models:", list(available_models.keys()))
            return False
        
        model_info = available_models[test_model_id]
        
        # Create a simple model configuration for testing
        model_config = ModelConfig(
            provider=ModelProvider.QWEN,  # Use QWEN for the test
            name=model_info['name'],
            trust_remote_code=model_info.get('trust_remote_code', True),
            device="cuda" if torch.cuda.is_available() else "cpu",
            quantization="4bit" if torch.cuda.is_available() else None  # Use quantization if on GPU
        )
        
        print(f"Testing with model: {test_model_id}")
        print(f"Model name: {model_config.name}")
        print(f"Device: {model_config.device}")
        print(f"Quantization: {model_config.quantization}")
        
        # Create model provider using factory
        provider = ModelProviderFactory.create("qwen", model_config)
        
        print("Loading model...")
        start_time = time.time()
        provider.load_model(model_config)
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Generate certificate
        print("Generating certificate...")
        start_time = time.time()
        
        generated_text = provider.generate_text(
            prompt=prompt,
            max_tokens=256,
            temperature=0.1  # Low temperature for consistency
        )
        
        generation_time = time.time() - start_time
        print(f"Generated in {generation_time:.2f} seconds")
        
        print("\nGenerated Output:")
        print("-" * 40)
        print(generated_text)
        print("-" * 40)
        
        # Extract certificate
        certificate = extract_certificate_simple(generated_text)
        print(f"\nExtracted Certificate: {certificate}")
        
        # Cleanup
        provider.unload_model()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during test: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("FM-LLM Solver Certificate Generation Test")
    print("=" * 60)
    
    # Test model availability first
    test_model_availability()
    
    # Test certificate generation
    success = test_certificate_generation()
    
    if success:
        print("\n✅ Certificate generation test PASSED")
    else:
        print("\n❌ Certificate generation test FAILED") 