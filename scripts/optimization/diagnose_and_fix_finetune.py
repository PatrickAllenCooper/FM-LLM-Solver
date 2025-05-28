#!/usr/bin/env python
"""
Diagnostic tool for debugging and fixing fine-tuning issues in FM-LLM-Solver.
This script helps identify common problems with GPU configuration, model availability,
and memory usage that can cause fine-tuning to fail.
"""

import os
import sys
import gc
import time
import json
import subprocess
import torch
import psutil
import argparse
from pathlib import Path
from huggingface_hub import list_models

# Add project root to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)

# Import project modules
from utils.config_loader import load_config, DEFAULT_CONFIG_PATH

def check_gpu_configuration():
    """Check if CUDA is available and print GPU information."""
    print("\n=== GPU Configuration ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA is NOT available. Make sure you have NVIDIA drivers installed.")
        print("   Fine-tuning requires a CUDA-compatible GPU.")
        return False
    
    # GPU is available
    device_count = torch.cuda.device_count()
    print(f"✅ CUDA is available. Found {device_count} GPU(s).")
    
    for i in range(device_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # in GB
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        
        # Check compute capability for tensor cores
        major, minor = torch.cuda.get_device_capability(i)
        print(f"   Compute capability: {major}.{minor}")
        
        if major >= 7:
            print("   ✅ This GPU supports Tensor Cores (good for training)")
        elif major >= 6:
            print("   ⚠️ This GPU has Pascal architecture (can train but slower)")
        else:
            print("   ⚠️ This GPU has older architecture (training may be very slow)")
    
    # Check available GPU memory
    available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # in GB
    print(f"\n   Available GPU memory: {available_memory:.2f} GB")
    
    if available_memory < 8:
        print("   ⚠️ Less than 8GB VRAM available. This will limit model size severely.")
        print("      Consider using --use_cpu_offload option with fine-tuning.")
    elif available_memory < 12:
        print("   ⚠️ Less than 12GB VRAM available. You'll need heavy quantization.")
        print("      Make sure config uses 4-bit quantization and LoRA rank <= 8.")
    elif available_memory < 24:
        print("   ℹ️ 12-24GB VRAM available. Should work with 4-bit quantization.")
    else:
        print("   ✅ Plenty of VRAM available. Fine-tuning should work well.")
    
    return True

def check_hf_model_existence(model_name):
    """Verify if a Hugging Face model exists and is downloadable."""
    print(f"\n=== Checking Model Availability: {model_name} ===")
    
    # Try to verify model exists on HF
    try:
        # This will check if the model exists on the Hub
        from huggingface_hub import hf_hub_url, url_to_filename
        from huggingface_hub.utils import EntryNotFoundError
        
        try:
            # Test if model exists by trying to get model.safetensors URL
            # Use a more efficient method, attempting to get a model file URL
            config_file_url = hf_hub_url(model_name, "config.json")
            print(f"✅ Model '{model_name}' exists on Hugging Face Hub")
            
            # Suggest similar models if it contains version numbers that might be wrong
            if "qwen" in model_name.lower():
                print("\nAvailable Qwen models that might be useful alternatives:")
                qwen_models = [m for m in list_models(filter="Qwen") if "Instruct" in m.modelId and m.downloads > 1000]
                for i, model in enumerate(qwen_models[:5]):
                    print(f"  {i+1}. {model.modelId} ({model.downloads:,} downloads)")
            
            return True
        except EntryNotFoundError:
            print(f"❌ Model '{model_name}' does not exist on Hugging Face Hub")
            
            # Helpful suggestion for similar models
            base_name = model_name.split("/")[-1].lower()
            print("\nSuggesting similar models:")
            
            if "qwen" in base_name:
                print("Similar Qwen models:")
                models = [m for m in list_models(filter="Qwen") if "Instruct" in m.modelId]
                models = sorted(models, key=lambda x: x.downloads, reverse=True)
                for i, model in enumerate(models[:5]):
                    print(f"  {i+1}. {model.modelId} ({model.downloads:,} downloads)")
            elif "llama" in base_name:
                print("Similar Llama models:")
                models = [m for m in list_models(filter="llama") if "Instruct" in m.modelId]
                models = sorted(models, key=lambda x: x.downloads, reverse=True)
                for i, model in enumerate(models[:5]):
                    print(f"  {i+1}. {model.modelId} ({model.downloads:,} downloads)")
            
            return False
    except Exception as e:
        print(f"⚠️ Error checking model: {e}")
        return False

def check_config_compatibility():
    """Check if configuration settings are compatible with GPU memory."""
    print("\n=== Configuration Compatibility ===")
    
    cfg = load_config()
    
    # Get model name from config
    model_name = cfg.fine_tuning.base_model_name
    print(f"Model: {model_name}")
    
    # Extract model size (if available)
    model_size = None
    for size in ["7b", "8b", "14b", "15b", "70b", "72b"]:
        if size in model_name.lower():
            model_size = size
            break
    
    if model_size:
        print(f"Detected model size: {model_size}")
    else:
        print("Could not detect model size from name")
    
    # Check LoRA config
    lora_r = cfg.fine_tuning.lora.r
    print(f"LoRA rank: {lora_r}")
    
    if lora_r > 8 and (model_size in ["14b", "15b", "70b", "72b"]):
        print("⚠️ LoRA rank is high for large model. Consider reducing to 8 or lower.")
    
    # Check quantization
    using_4bit = cfg.fine_tuning.quantization.use_4bit
    print(f"Using 4-bit quantization: {using_4bit}")
    
    if not using_4bit and (model_size in ["14b", "15b", "70b", "72b"]):
        print("❌ 4-bit quantization should be enabled for large models")
    
    # Check nested quantization
    using_nested = cfg.fine_tuning.quantization.use_nested_quant
    print(f"Using nested quantization: {using_nested}")
    
    if not using_nested and model_size in ["14b", "15b", "70b", "72b"]:
        print("⚠️ Consider enabling nested quantization for large models")
    
    # Check gradient accumulation
    grad_accum = cfg.fine_tuning.training.gradient_accumulation_steps
    print(f"Gradient accumulation steps: {grad_accum}")
    
    if grad_accum < 8 and model_size in ["14b", "15b", "70b", "72b"]:
        print("⚠️ Consider increasing gradient accumulation steps to 8 or higher")
    
    # Check sequence length
    seq_length = cfg.fine_tuning.training.get("max_seq_length", "Not specified")
    print(f"Max sequence length: {seq_length}")
    
    if seq_length == "Not specified" or seq_length > 1024:
        print("⚠️ Consider setting max_seq_length to 1024 or lower to save memory")
    
    return True

def analyze_finetune_errors():
    """Parse error logs to find known issues"""
    print("\n=== Analyzing Fine-tuning Errors ===")
    
    error_log_path = os.path.join(PROJECT_ROOT, "experiment_run.log")
    
    if not os.path.exists(error_log_path):
        print(f"❌ Error log file not found at {error_log_path}")
        return False
    
    with open(error_log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
    
    # Check for common error patterns
    cuda_oom = "CUDA out of memory" in log_content
    cuda_error = "CUDA error:" in log_content
    model_not_found = "403 Client Error" in log_content or "404 Client Error" in log_content
    import_error = "ImportError" in log_content
    
    if cuda_oom:
        print("❌ CUDA Out of Memory Error Detected")
        print("   Recommended fixes:")
        print("   1. Reduce LoRA rank in config.yaml")
        print("   2. Increase gradient_accumulation_steps")
        print("   3. Use 4-bit quantization with nested_quant")
        print("   4. Reduce max_seq_length to 1024 or lower")
        print("   5. Try a smaller model (7B/8B instead of 14B/15B)")
    
    if cuda_error and not cuda_oom:
        print("❌ CUDA Error Detected (not memory related)")
        print("   Recommended fixes:")
        print("   1. Update NVIDIA drivers")
        print("   2. Check for PyTorch/CUDA version mismatch")
        print("   3. Try reinstalling PyTorch with compatible CUDA version")
    
    if model_not_found:
        print("❌ Model Not Found Error")
        print("   Recommended fixes:")
        print("   1. Check model name in config.yaml")
        print("   2. Verify model exists on Hugging Face Hub")
        print("   3. Check for proper authentication if model is gated")
    
    if import_error:
        print("❌ Import Error Detected")
        print("   Recommended fixes:")
        print("   1. Make sure all dependencies are installed")
        print("   2. Run: pip install -r requirements.txt")
        print("   3. Check for conflicting package versions")
    
    # If no specific errors found
    if not any([cuda_oom, cuda_error, model_not_found, import_error]):
        print("No specific errors detected in log file")
        print("Check experiment_run.log for details")
    
    return True

def fix_common_issues(cfg_path):
    """Attempt to fix common issues by modifying config.yaml."""
    print("\n=== Fixing Common Issues ===")
    
    # Load current config
    cfg = load_config(cfg_path)
    
    # Check if we need to make changes
    fixed_anything = False
    
    # 1. Fix model name if it's using non-existent Qwen model
    if "Qwen/Qwen2.5-15B-Instruct" in cfg.fine_tuning.base_model_name:
        print("⚠️ Detected incorrect model name: Qwen/Qwen2.5-15B-Instruct")
        print("ℹ️ Changing to correct model: Qwen/Qwen2.5-14B-Instruct")
        
        # Create backup
        backup_path = os.path.join(PROJECT_ROOT, "config.yaml.bak")
        with open(cfg_path, 'r', encoding='utf-8') as f_in:
            with open(backup_path, 'w', encoding='utf-8') as f_out:
                f_out.write(f_in.read())
        
        # Read file content
        with open(cfg_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace model name
        new_content = content.replace(
            'base_model_name: "Qwen/Qwen2.5-15B-Instruct"', 
            'base_model_name: "Qwen/Qwen2.5-14B-Instruct"'
        )
        
        # Write updated content
        with open(cfg_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ Fixed model name in config.yaml")
        print(f"ℹ️ Backup created at {backup_path}")
        fixed_anything = True
    
    # 2. Optimize memory settings for large models
    model_name = cfg.fine_tuning.base_model_name
    large_model = any(size in model_name.lower() for size in ["14b", "15b", "70b", "72b"])
    
    if large_model:
        # Create backup if not already created
        if not fixed_anything:
            backup_path = os.path.join(PROJECT_ROOT, "config.yaml.bak")
            with open(cfg_path, 'r', encoding='utf-8') as f_in:
                with open(backup_path, 'w', encoding='utf-8') as f_out:
                    f_out.write(f_in.read())
        
        # Read file content
        with open(cfg_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        memory_optimizations = []
        
        # Check and fix LoRA rank
        if cfg.fine_tuning.lora.r > 8:
            content = content.replace(
                f'r: {cfg.fine_tuning.lora.r}', 
                'r: 8'
            )
            memory_optimizations.append("Reduced LoRA rank to 8")
        
        # Check and fix quantization
        if not cfg.fine_tuning.quantization.use_4bit:
            content = content.replace(
                'use_4bit: false', 
                'use_4bit: true'
            )
            memory_optimizations.append("Enabled 4-bit quantization")
        
        # Check and fix nested quantization
        if not cfg.fine_tuning.quantization.use_nested_quant:
            content = content.replace(
                'use_nested_quant: false', 
                'use_nested_quant: true'
            )
            memory_optimizations.append("Enabled nested quantization")
        
        # Check and fix gradient accumulation
        if cfg.fine_tuning.training.gradient_accumulation_steps < 8:
            content = content.replace(
                f'gradient_accumulation_steps: {cfg.fine_tuning.training.gradient_accumulation_steps}',
                'gradient_accumulation_steps: 8'
            )
            memory_optimizations.append("Increased gradient accumulation steps to 8")
        
        # Set max sequence length if not specified or too large
        max_seq_length = cfg.fine_tuning.training.get("max_seq_length", None)
        if max_seq_length is None or max_seq_length > 1024:
            if 'max_seq_length:' in content:
                content = content.replace(
                    f'max_seq_length: {max_seq_length}',
                    'max_seq_length: 1024'
                )
            else:
                # Add max_seq_length under training section if not present
                training_section = content.find('training:')
                if training_section != -1:
                    insertion_point = content.find('\n', training_section)
                    if insertion_point != -1:
                        content = content[:insertion_point] + '\n    max_seq_length: 1024' + content[insertion_point:]
            memory_optimizations.append("Set max sequence length to 1024")
        
        # Write updated content if any changes were made
        if memory_optimizations:
            with open(cfg_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ Applied memory optimizations:")
            for opt in memory_optimizations:
                print(f"   - {opt}")
            print(f"ℹ️ Backup created at {backup_path}")
            fixed_anything = True
    
    if not fixed_anything:
        print("No issues found that require automatic fixing")
    
    return fixed_anything

def diagnostic_train_test():
    """Run a minimal test fine-tuning job to diagnose issues."""
    print("\n=== Running Diagnostic Fine-tuning Test ===")
    print("This will attempt a 10-step mini fine-tuning run to identify issues")
    
    # Create a temporary config file for the test
    temp_config_path = os.path.join(PROJECT_ROOT, "test_finetune_config.yaml")
    
    # Load current config
    cfg = load_config()
    
    # Create a minimal config for testing
    test_config = {
        "paths": {
            "project_root": ".",
            "data_dir": "data",
            "output_dir": "output/test_finetune",
            "ft_combined_data_file": cfg.paths.ft_combined_data_file,
            "ft_output_dir": "output/test_finetune"
        },
        "fine_tuning": {
            "base_model_name": "Qwen/Qwen2.5-7B-Instruct",  # Use smaller model for test
            "data_format": "instruction",
            "use_adapter": True,
            "lora": {
                "r": 4,  # Minimal rank
                "alpha": 4,
                "dropout": 0.1,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj"
                ]
            },
            "quantization": {
                "use_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_quant_type": "nf4",
                "use_nested_quant": True
            },
            "training": {
                "num_train_epochs": 1,
                "per_device_train_batch_size": 1,
                "per_device_eval_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "gradient_checkpointing": True,
                "max_grad_norm": 0.3,
                "learning_rate": 0.0002,
                "weight_decay": 0.001,
                "optim": "paged_adamw_32bit",
                "lr_scheduler_type": "cosine",
                "max_steps": 10,  # Just 10 steps for quick test
                "warmup_ratio": 0.03,
                "group_by_length": False,
                "save_steps": 10,
                "logging_steps": 1,
                "packing": False,
                "max_seq_length": 512  # Short sequences
            }
        }
    }
    
    # Create temp config file
    os.makedirs("output/test_finetune", exist_ok=True)
    with open(temp_config_path, 'w') as f:
        json.dump(test_config, f, indent=2)
    
    # Run test fine-tuning
    print("Running minimal test fine-tuning...")
    try:
        cmd = f"python fine_tuning/finetune_llm.py --config {temp_config_path}"
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Monitor memory while running
        peak_mem_gb = 0
        start_time = time.time()
        timeout = 300  # 5 minutes timeout
        
        print("\nMonitoring GPU memory usage during test:")
        
        while process.poll() is None and time.time() - start_time < timeout:
            time.sleep(2)  # Check every 2 seconds
            
            if torch.cuda.is_available():
                # Get current memory usage
                mem_used = torch.cuda.memory_reserved(0) / 1024**3  # in GB
                peak_mem_gb = max(peak_mem_gb, mem_used)
                print(f"\rCurrent GPU memory: {mem_used:.2f} GB | Peak: {peak_mem_gb:.2f} GB", end="")
        
        # Collect output
        stdout, stderr = process.communicate(timeout=10)
        
        print("\n\nTest completed.")
        print(f"Peak GPU memory usage: {peak_mem_gb:.2f} GB")
        
        if process.returncode != 0:
            print("\n❌ Test fine-tuning failed")
            print("\nError output:")
            
            # Extract key errors from output
            error_lines = []
            for line in stderr.split('\n'):
                if "Error" in line or "Exception" in line or "CUDA" in line:
                    error_lines.append(line)
            
            if error_lines:
                print("\nKey errors:")
                for line in error_lines[:5]:  # Show first 5 errors
                    print(f"  {line}")
            else:
                print("Full error output:")
                print(stderr[-500:])  # Last 500 chars of stderr
            
            # Analyze the error
            if "CUDA out of memory" in stderr:
                print("\n❌ CUDA Out of Memory Error")
                print("   Even with minimal settings, GPU memory is insufficient.")
                print("   Consider using a different GPU or trying CPU offloading.")
            elif "ImportError" in stderr:
                print("\n❌ Missing dependencies")
                print("   Try: pip install -r requirements.txt")
            elif "403 Client Error" in stderr or "404 Client Error" in stderr:
                print("\n❌ Model access error")
                print("   Model either doesn't exist or requires authentication.")
        else:
            print("\n✅ Test fine-tuning completed successfully!")
            print("This suggests your environment is properly configured.")
            print("You can now try a full fine-tuning run with your desired configuration.")
    
    except Exception as e:
        print(f"\n❌ Error running test: {e}")
    
    # Clean up
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)

def main():
    parser = argparse.ArgumentParser(description="Diagnose and fix fine-tuning issues")
    parser.add_argument("--fix", action="store_true", help="Apply automatic fixes to config.yaml")
    parser.add_argument("--test", action="store_true", help="Run a minimal fine-tuning test")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to config file")
    args = parser.parse_args()
    
    print("====== FM-LLM-Solver Fine-tuning Diagnostics ======")
    print(f"Project root: {PROJECT_ROOT}")
    
    # Always run these checks
    gpu_ok = check_gpu_configuration()
    
    if not gpu_ok:
        print("\n❌ GPU configuration issues detected. Fine-tuning may not work properly.")
    
    # Check model existence
    cfg = load_config(args.config)
    model_name = cfg.fine_tuning.base_model_name
    model_ok = check_hf_model_existence(model_name)
    
    if not model_ok:
        print(f"\n❌ Model '{model_name}' appears to be unavailable.")
    
    # Check config compatibility
    check_config_compatibility()
    
    # Analyze recent errors
    analyze_finetune_errors()
    
    # Apply fixes if requested
    if args.fix:
        fixed = fix_common_issues(args.config)
        if fixed:
            print("\n✅ Applied fixes to config.yaml")
        else:
            print("\nℹ️ No automatic fixes were necessary")
    
    # Run test if requested
    if args.test:
        diagnostic_train_test()
    
    print("\n====== Diagnostics Complete ======")
    
    if not args.fix and not args.test:
        print("\nRecommendations:")
        print("1. Run with --fix to apply automatic configuration fixes")
        print("2. Run with --test to perform a minimal fine-tuning test")
        print("   Example: python scripts/optimization/diagnose_and_fix_finetune.py --fix --test")

if __name__ == "__main__":
    main() 