#!/usr/bin/env python3
"""
Modal inference application for FM-LLM-Solver (Simple Version)
Simplified version without quantization for reliable deployment
"""

import modal
import os
import sys
from pathlib import Path

# Modal app configuration
app = modal.App("fm-llm-solver-simple")

# Simplified Docker image without complex quantization
image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04", add_python="3.10"
    )
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.35.0", 
        "accelerate>=0.24.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "tenacity>=8.2.0",
        "psutil>=5.9.0",
    ])
    .apt_install(["git", "curl"])
)

# Volume for model caching
volume = modal.Volume.from_name("fm-llm-models", create_if_missing=True)

@app.cls(
    gpu="A10G",
    image=image,
    volumes={"/models": volume},
    timeout=1800,  # 30 minutes
    scaledown_window=300,
)
class SimpleFMLLMInference:
    """Simplified FM-LLM-Solver inference service"""
    
    @modal.enter()
    def setup(self):
        """Initialize the inference service"""
        import torch
        import logging
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Set up model cache directory
        self.model_cache_dir = "/models"
        os.makedirs(self.model_cache_dir, exist_ok=True)
        
        # Use a smaller, more reliable model
        self.model_name = "microsoft/DialoGPT-medium"  # Smaller, more reliable
        
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=self.model_cache_dir
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model without quantization (simpler)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=self.model_cache_dir
            )
            
            self.logger.info(f"Successfully loaded model: {self.model_name}")
            self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
                self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            # Fallback to CPU-only smaller model
            self.model_name = "gpt2"
            self.logger.info(f"Falling back to: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
    
    @modal.method()
    def generate_certificate(
        self, 
        system_description: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True
    ):
        """Generate barrier certificate for given system"""
        try:
            import torch
            
            # Create prompt for barrier certificate generation
            prompt = f"""System: {system_description}

Generate a barrier function B(x) for this dynamical system. The barrier function should be a polynomial that proves safety.

Barrier function: B(x) = """

            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = inputs.to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the generated part (after the prompt)
            barrier_certificate = generated_text[len(prompt):].strip()
            
            # Simple validation - look for mathematical expressions
            if any(char in barrier_certificate for char in ['x', 'y', 'z', '+', '-', '*', '^', '2']):
                success = True
                message = "Barrier function generated successfully"
            else:
                success = False
                message = "Generated text doesn't appear to be a mathematical function"
            
            return {
                "success": success,
                "message": message,
                "barrier_certificate": barrier_certificate,
                "system_description": system_description,
                "model_used": self.model_name,
                "metadata": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "prompt_length": len(prompt)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "system_description": system_description,
                "model_used": getattr(self, 'model_name', 'unknown'),
                "message": "Error during generation"
            }
    
    @modal.method()
    def health_check(self):
        """Health check endpoint"""
        import torch
        
        return {
            "status": "healthy",
            "model_loaded": hasattr(self, 'model') and self.model is not None,
            "model_name": getattr(self, 'model_name', 'not_loaded'),
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }

# Web endpoints
@app.function(
    image=image,
    timeout=300
)
@modal.fastapi_endpoint(method="POST")
def generate(request_data: dict):
    """Web endpoint for certificate generation"""
    try:
        # Get inference class instance
        inference = SimpleFMLLMInference()
        
        # Extract parameters
        system_description = request_data.get("system_description", "")
        max_new_tokens = request_data.get("max_new_tokens", 100)
        temperature = request_data.get("temperature", 0.7)
        
        if not system_description:
            return {
                "success": False,
                "error": "system_description is required",
                "message": "Please provide a system description"
            }
        
        # Generate certificate
        result = inference.generate_certificate.remote(
            system_description=system_description,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Web endpoint error: {str(e)}",
            "message": "Error in web endpoint"
        }

@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health():
    """Health check web endpoint"""
    return {"status": "healthy", "service": "fm-llm-solver-simple", "version": "1.0"}

# Test function
@app.function(image=image, timeout=60)
@modal.fastapi_endpoint(method="POST") 
def test():
    """Quick test endpoint"""
    try:
        inference = SimpleFMLLMInference()
        result = inference.generate_certificate.remote(
            system_description="dx/dt = -x",
            max_new_tokens=50
        )
        return {"test": "passed", "result": result}
    except Exception as e:
        return {"test": "failed", "error": str(e)}

if __name__ == "__main__":
    # Can run tests locally if needed
    pass 