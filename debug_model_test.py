#!/usr/bin/env python3
"""
Debug script to test model loading with different providers.
"""

import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    from fm_llm_solver.core.types import ModelConfig, ModelProvider
    from fm_llm_solver.services.model_provider import ModelProviderFactory
    from utils.config_loader import load_config

    print("✅ Successfully imported all components")
    
    # Test the enum values
    print(f"Available providers: {[p.value for p in ModelProvider]}")
    
    # Load configuration
    config = load_config()
    available_models = config['models']['available_models']
    
    # Test with the smallest model
    test_model_id = 'qwen2.5-coder-0.5b-instruct'
    if test_model_id in available_models:
        model_config = available_models[test_model_id]
        provider_name = model_config['provider']
        
        print(f"Testing model: {test_model_id}")
        print(f"Provider from config: {provider_name}")
        
        # Try to get the enum value
        try:
            provider_enum = ModelProvider(provider_name)
            print(f"✅ Provider enum: {provider_enum}")
            
            # Create model config
            config_obj = ModelConfig(
                provider=provider_enum,
                name=model_config['name'],
                trust_remote_code=True,
                device="cuda",
                quantization="4bit"
            )
            
            print(f"✅ Created config: {config_obj.provider}")
            
            # Try to create provider
            provider = ModelProviderFactory.create(provider_name, config_obj)
            print(f"✅ Created provider: {type(provider).__name__}")
            
        except ValueError as e:
            print(f"❌ Provider enum error: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print(f"❌ Model {test_model_id} not found in config")
        print(f"Available models: {list(available_models.keys())[:5]}...")

except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc() 