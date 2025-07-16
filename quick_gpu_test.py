#!/usr/bin/env python3
"""
Quick GPU Test - Real LLM Inference with Progress

Shows immediate progress and results from actual GPU-accelerated LLM inference
to verify that filtering and numerical checking works on real model outputs.
"""

import sys
import time
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    """Quick GPU test with real-time progress."""
    print("🚀 FM-LLM SOLVER: Quick GPU Test")
    print("="*50)
    
    # Check GPU
    print("🔍 Checking GPU availability...")
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA not available!")
        print("   This test requires an NVIDIA GPU with CUDA.")
        return 1
    
    gpu_name = torch.cuda.get_device_name()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU detected: {gpu_name}")
    print(f"🔋 GPU memory: {gpu_memory:.1f} GB")
    
    # Test system
    test_prompt = """You are an expert in control theory. Generate a barrier certificate for this system:

System: dx/dt = -x, dy/dt = -y
Initial set: x² + y² ≤ 0.25  
Unsafe set: x² + y² ≥ 4.0

Format your answer as:
BARRIER_CERTIFICATE_START
B(x,y) = [your certificate]
BARRIER_CERTIFICATE_END"""

    print("\n📝 Test prompt:")
    print(f"   '{test_prompt[:100]}...'")
    
    try:
        print("\n🤖 Loading model (Qwen 7B with 4-bit quantization)...")
        print("⏳ This takes 30-60 seconds on first load...")
        
        # Import and setup
        from fm_llm_solver.services.model_provider import QwenProvider
        from fm_llm_solver.core.types import ModelConfig, ModelProvider
        
        # Quick model config
        model_config = ModelConfig(
            provider=ModelProvider.QWEN,
            name="Qwen/Qwen2.5-7B-Instruct",
            device="cuda",
            quantization="4bit",
            temperature=0.1,
            max_tokens=256
        )
        
        # Load model with progress
        load_start = time.time()
        provider = QwenProvider()
        
        print("   📦 Loading tokenizer...")
        provider.load_model(model_config)
        load_time = time.time() - load_start
        
        memory_used = torch.cuda.memory_allocated() / 1e9
        print(f"✅ Model loaded in {load_time:.1f}s")
        print(f"🔥 GPU memory used: {memory_used:.2f} GB")
        
        print("\n🧠 Generating with real LLM...")
        print("⏳ Generating (15-30 seconds)...")
        
        # Generate
        gen_start = time.time()
        raw_output = provider.generate_text(
            prompt=test_prompt,
            max_tokens=256,
            temperature=0.1
        )
        gen_time = time.time() - gen_start
        
        print(f"✅ Generation completed in {gen_time:.1f}s")
        print(f"\n📄 Raw LLM Output:")
        print("="*50)
        print(raw_output)
        print("="*50)
        
        # Test extraction
        print("\n🔍 Testing certificate extraction...")
        
        from utils.certificate_extraction import extract_certificate_from_llm_output
        from utils.certificate_extraction import is_template_expression, clean_and_validate_expression
        
        # Extract
        try:
            extracted_result = extract_certificate_from_llm_output(raw_output, ["x", "y"])
            extracted = extracted_result[0] if isinstance(extracted_result, tuple) else extracted_result
            
            if extracted:
                print(f"✅ Extracted: '{extracted}'")
                
                # Test template detection
                is_template = is_template_expression(extracted)
                print(f"📋 Is template: {is_template}")
                
                if not is_template:
                    # Test cleaning
                    try:
                        cleaned = clean_and_validate_expression(extracted, ["x", "y"])
                        print(f"🧹 Cleaned: '{cleaned}'")
                        
                        # Success!
                        print("\n🎉 SUCCESS: Real LLM pipeline working!")
                        print("   ✅ GPU inference: Working")
                        print("   ✅ Certificate extraction: Working") 
                        print("   ✅ Template filtering: Working")
                        print("   ✅ Expression cleaning: Working")
                        
                        result_code = 0
                        
                    except Exception as e:
                        print(f"⚠️  Cleaning failed: {e}")
                        print("   Pipeline partially working, needs tuning")
                        result_code = 1
                else:
                    print("❌ Template detected - real LLM generating templates")
                    print("   Your prompting needs improvement")
                    result_code = 2
            else:
                print("❌ Extraction failed - real LLM output not parseable")
                print("   Your extraction logic needs improvement")
                result_code = 3
                
        except Exception as e:
            print(f"❌ Extraction error: {e}")
            result_code = 4
        
        # Cleanup
        print("\n🧹 Cleaning up...")
        provider.unload_model()
        torch.cuda.empty_cache()
        
        # Summary
        print(f"\n📊 Performance Summary:")
        print(f"   Model load time: {load_time:.1f}s")
        print(f"   Generation time: {gen_time:.1f}s")
        print(f"   GPU memory peak: {memory_used:.2f} GB")
        print(f"   Total test time: {time.time() - load_start:.1f}s")
        
        return result_code
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Install: pip install torch transformers accelerate bitsandbytes")
        return 5
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"❌ CUDA error: {e}")
            print("   Check GPU memory or CUDA installation")
        else:
            print(f"❌ Runtime error: {e}")
        return 6
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 7

if __name__ == "__main__":
    exit(main()) 