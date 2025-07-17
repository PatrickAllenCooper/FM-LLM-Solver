#!/usr/bin/env python3
"""
Modal inference application for FM-LLM-Solver
Deploys inference API to Modal's serverless GPU platform
"""

import modal
import os
import sys
from pathlib import Path

# Modal app configuration
app = modal.App("fm-llm-solver-inference")

# Define GPU configuration
GPU_CONFIG = modal.gpu.A10G()  # Good balance of performance and cost

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
        "flash-attn>=2.3.0",
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
    gpu=GPU_CONFIG,
    image=image,
    volumes={"/models": volume},
    timeout=1800,  # 30 minutes
    container_idle_timeout=300,  # 5 minutes
    allow_concurrent_inputs=10,
)
class FMLLMInference:
    """FM-LLM-Solver inference service on Modal"""
    
    def __init__(self):
        self.models = {}
        self.current_model = None
        
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
        
        self.logger.info("FM-LLM-Solver inference service initialized")
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def load_model(self, model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"):
        """Load a specific model"""
        if model_name == self.current_model:
            return self.models[model_name]
            
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            self.logger.info(f"Loading model: {model_name}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.model_cache_dir,
                trust_remote_code=True
            )
            
            # Load model with quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=self.model_cache_dir,
                quantization_config=self.quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            # Clear previous model if exists
            if self.current_model and self.current_model in self.models:
                del self.models[self.current_model]
                torch.cuda.empty_cache()
            
            self.models[model_name] = {"model": model, "tokenizer": tokenizer}
            self.current_model = model_name
            
            self.logger.info(f"Successfully loaded model: {model_name}")
            return self.models[model_name]
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    @modal.method()
    def generate_certificate(
        self, 
        system_description: str,
        model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> dict:
        """Generate barrier certificate for given system"""
        try:
            import time
            import torch
            
            start_time = time.time()
            
            # Load model
            model_data = self.load_model(model_name)
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]
            
            # Create prompt
            prompt = f"""You are an expert in barrier certificate generation for dynamical systems.

System Description: {system_description}

Generate a barrier certificate B(x) for this system. The certificate should:
1. Be a polynomial function
2. Separate the safe and unsafe regions
3. Satisfy barrier conditions

Barrier Certificate:"""

            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            certificate = response.split("Barrier Certificate:")[-1].strip()
            
            generation_time = time.time() - start_time
            
            return {
                "certificate": certificate,
                "model_used": model_name,
                "generation_time": generation_time,
                "success": True,
                "prompt_tokens": len(inputs["input_ids"][0]),
                "response_tokens": len(outputs[0]) - len(inputs["input_ids"][0])
            }
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return {
                "certificate": None,
                "error": str(e),
                "success": False
            }
    
    @modal.method()
    def health_check(self) -> dict:
        """Health check endpoint"""
        import torch
        return {
            "status": "healthy",
            "cuda_available": torch.cuda.is_available(),
            "current_model": self.current_model,
            "gpu_memory_used": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        }

# FastAPI web server
@app.function(
    image=image,
    allow_concurrent_inputs=100
)
@modal.web_endpoint(method="POST", path="/generate")
def web_generate(request_data: dict):
    """Web endpoint for certificate generation"""
    inference = FMLLMInference()
    return inference.generate_certificate.remote(**request_data)

@app.function(image=image)
@modal.web_endpoint(method="GET", path="/health")
def web_health():
    """Web health check endpoint"""
    inference = FMLLMInference()
    return inference.health_check.remote()

@app.function(image=image)
@modal.web_endpoint(method="GET", path="/")
def web_root():
    """Root endpoint"""
    return {
        "service": "FM-LLM-Solver Inference API",
        "status": "online",
        "endpoints": {
            "generate": "POST /generate",
            "health": "GET /health"
        }
    }

# Local development function
@app.local_entrypoint()
def main():
    """Deploy and test the inference service"""
    print("🚀 Deploying FM-LLM-Solver to Modal...")
    
    # Test the service
    inference = FMLLMInference()
    
    print("🧪 Testing certificate generation...")
    result = inference.generate_certificate.remote(
        system_description="dx/dt = -x, dy/dt = -y, initial set: x^2 + y^2 <= 1, unsafe set: x^2 + y^2 >= 4",
        model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct"
    )
    
    print("✅ Test result:")
    print(f"   Certificate: {result.get('certificate', 'None')[:100]}...")
    print(f"   Generation time: {result.get('generation_time', 0):.2f}s")
    print(f"   Success: {result.get('success', False)}")
    
    print("\n🌐 Service deployed! Endpoints:")
    print("   - Certificate Generation: POST /generate")
    print("   - Health Check: GET /health")
    print("\n💡 Use 'modal serve modal_inference_app.py' to run locally")

if __name__ == "__main__":
    main() 