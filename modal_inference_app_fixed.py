#!/usr/bin/env python3
"""
Modal inference application for FM-LLM-Solver (Fixed Version)
Deploys inference API to Modal's serverless GPU platform
"""

import modal
import os
import sys
from pathlib import Path

# Modal app configuration
app = modal.App("fm-llm-solver-inference")

# Docker image with CUDA and dependencies
image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04", add_python="3.10"
    )
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.35.0", 
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",
        "einops>=0.7.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "sympy>=1.12",
        "PyYAML>=6.0",
        "python-dotenv>=1.0.0",
        "tenacity>=8.2.0",
        "psutil>=5.9.0",
    ])
    .apt_install(["git", "curl"])
)

# Volume for model caching
volume = modal.Volume.from_name("fm-llm-models", create_if_missing=True)

@app.cls(
    gpu="A10G",  # Updated syntax
    image=image,
    volumes={"/models": volume},
    timeout=1800,  # 30 minutes
    scaledown_window=300,  # Fixed parameter name
)
class FMLLMInference:
    """FM-LLM-Solver inference service on Modal"""
    
    model_name: str = modal.parameter(default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    
    @modal.enter()
    def setup(self):
        """Initialize the inference service"""
        import torch
        import logging
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Set up model cache directory
        self.model_cache_dir = "/models"
        os.makedirs(self.model_cache_dir, exist_ok=True)
        
        # Configure quantization for memory efficiency
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load default model
        self.models = {}
        self.current_model = None
        self.load_model(self.model_name)
        
        self.logger.info("FM-LLM-Solver inference service initialized")
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def load_model(self, model_name: str = None):
        """Load a specific model"""
        if model_name is None:
            model_name = self.model_name
            
        if model_name == self.current_model and model_name in self.models:
            return self.models[model_name]
            
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            self.logger.info(f"Loading model: {model_name}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=self.model_cache_dir
            )
            
            # Load model with quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=self.quantization_config,
                cache_dir=self.model_cache_dir
            )
            
            self.models[model_name] = {
                "model": model,
                "tokenizer": tokenizer
            }
            
            self.current_model = model_name
            self.logger.info(f"Successfully loaded model: {model_name}")
            
            return self.models[model_name]
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    @modal.method()
    def generate_certificate(
        self, 
        system_description: str,
        model_name: str = None,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        do_sample: bool = True
    ):
        """Generate barrier certificate for given system"""
        try:
            # Load model if different from current
            if model_name and model_name != self.current_model:
                self.load_model(model_name)
            
            model_data = self.models[self.current_model]
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]
            
            # Create prompt for barrier certificate generation
            prompt = f"""Generate a barrier certificate for the following dynamical system:

System: {system_description}

Please provide a barrier function B(x) that satisfies the barrier certificate conditions. 
The function should be a mathematical expression using the system variables.

Barrier Certificate:"""

            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the generated part (after the prompt)
            barrier_certificate = generated_text[len(prompt):].strip()
            
            return {
                "success": True,
                "barrier_certificate": barrier_certificate,
                "system_description": system_description,
                "model_used": self.current_model,
                "metadata": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature
                }
            }
            
        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "system_description": system_description
            }
    
    @modal.method()
    def health_check(self):
        """Health check endpoint"""
        import torch
        
        return {
            "status": "healthy",
            "model_loaded": self.current_model is not None,
            "current_model": self.current_model,
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }

# Create a simple web endpoint
@app.function(
    image=image,
    timeout=300
)
@modal.web_endpoint(method="POST")
def web_generate(request_data: dict):
    """Web endpoint for certificate generation"""
    try:
        # Get inference class instance
        inference = FMLLMInference()
        
        # Extract parameters
        system_description = request_data.get("system_description", "")
        model_name = request_data.get("model_name")
        max_new_tokens = request_data.get("max_new_tokens", 512)
        temperature = request_data.get("temperature", 0.1)
        
        if not system_description:
            return {
                "success": False,
                "error": "system_description is required"
            }
        
        # Generate certificate
        result = inference.generate_certificate.remote(
            system_description=system_description,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Web endpoint error: {str(e)}"
        }

@app.function(image=image)
@modal.web_endpoint(method="GET")
def health():
    """Health check web endpoint"""
    return {"status": "healthy", "service": "fm-llm-solver-inference"}

# Entry point for deployment
if __name__ == "__main__":
    # Test locally if needed
    pass 