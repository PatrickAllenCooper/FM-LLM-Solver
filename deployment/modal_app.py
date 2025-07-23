"""
Modal deployment for FM-LLM Solver Inference API
Provides serverless GPU-accelerated barrier certificate generation.
"""

import modal
import os
import sys
from pathlib import Path

# Modal App
app = modal.App("fm-llm-solver")

# Define GPU-optimized container image
inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.35.0", 
        "sentence-transformers>=2.2.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.4.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "faiss-cpu>=1.7.4",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",
        "peft>=0.6.0",
        "datasets>=2.14.0",
        "aiofiles>=23.2.0",
        "python-multipart>=0.0.6",
    ])
    .pip_install("flash-attn", pre=True)
    .apt_install("git")
    .run_commands([
        "pip install 'git+https://github.com/huggingface/transformers.git'",
        "pip install 'git+https://github.com/huggingface/peft.git'",
    ])
)

# Shared volume for model storage
models_volume = modal.Volume.from_name("fm-llm-models", create_if_missing=True)
kb_volume = modal.Volume.from_name("fm-llm-kb", create_if_missing=True)

# GPU configuration
GPU_CONFIG = modal.gpu.A10G()  # 24GB GPU, good for LLMs

@app.function(
    image=inference_image,
    gpu=GPU_CONFIG,
    volumes={
        "/models": models_volume,
        "/kb": kb_volume,
    },
    container_idle_timeout=300,  # Keep warm for 5 minutes
    timeout=1800,  # 30 minute timeout for long generations
    allow_concurrent_inputs=3,  # Handle multiple requests
)
class InferenceAPI:
    """FM-LLM Solver Inference API on Modal."""
    
    def __init__(self):
        """Initialize the inference service."""
        import logging
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize state
        self.models_cache = {}
        self.kb_cache = {}
        self.embedding_model = None
        self.config = None
        
    @modal.enter()
    def load_models(self):
        """Load models and knowledge bases on container startup."""
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import json
        
        self.logger.info("🚀 Starting FM-LLM Solver inference service...")
        self.logger.info(f"GPU available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        
        # Load embedding model for RAG
        try:
            self.logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("✅ Embedding model loaded")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
        
        # Load base LLM with quantization
        try:
            self.logger.info("Loading base language model...")
            
            # Quantization config for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
            model_name = "microsoft/DialoGPT-medium"  # Smaller model for quick testing
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            
            self.models_cache["base"] = {
                "model": model,
                "tokenizer": tokenizer,
            }
            
            self.logger.info("✅ Language model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load language model: {e}")
            # Continue without model for testing
        
        # Load knowledge base if available
        try:
            kb_path = "/kb/index.json"
            if os.path.exists(kb_path):
                with open(kb_path, 'r') as f:
                    kb_data = json.load(f)
                self.kb_cache["unified"] = kb_data
                self.logger.info("✅ Knowledge base loaded")
            else:
                self.logger.info("📄 No knowledge base found, proceeding without RAG")
        except Exception as e:
            self.logger.warning(f"Knowledge base loading failed: {e}")
        
        self.logger.info("🎯 Inference service ready!")

    @modal.method()
    def health_check(self):
        """Health check endpoint."""
        import torch
        
        return {
            "status": "healthy",
            "service": "fm-llm-solver-inference",
            "gpu_available": torch.cuda.is_available(),
            "models_loaded": list(self.models_cache.keys()),
            "kb_loaded": len(self.kb_cache) > 0,
            "embedding_model_loaded": self.embedding_model is not None,
        }

    @modal.method()
    def generate_certificate(
        self,
        system_description: str,
        model_config: str = "base",
        rag_k: int = 3,
        domain_bounds: dict = None,
        request_id: str = None,
    ):
        """Generate barrier certificate for the given system."""
        import time
        import torch
        
        start_time = time.time()
        self.logger.info(f"🔥 Generating certificate for request {request_id}")
        
        try:
            # Validate model exists
            if model_config not in self.models_cache:
                # For now, use base model as fallback
                model_config = "base"
                if model_config not in self.models_cache:
                    return {
                        "success": False,
                        "error": f"No models available. Service may be starting up.",
                        "certificate": None,
                        "llm_output": "",
                        "context_chunks": 0,
                        "processing_time": time.time() - start_time,
                    }
            
            model_info = self.models_cache[model_config]
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            
            # Retrieve RAG context if available
            context = ""
            context_chunks = 0
            if rag_k > 0 and self.embedding_model and self.kb_cache:
                context = self._retrieve_context(system_description, rag_k)
                context_chunks = context.count("Chunk") if context else 0
            
            # Build prompt
            prompt = self._build_barrier_certificate_prompt(
                system_description, context, domain_bounds
            )
            
            # Generate response
            with torch.no_grad():
                inputs = tokenizer.encode(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                
                # Generate with controlled parameters
                outputs = model.generate(
                    inputs,
                    max_new_tokens=512,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # Decode response
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_text = response[len(prompt):].strip()
            
            # Extract certificate (simplified for demo)
            certificate = self._extract_certificate(generated_text)
            
            processing_time = time.time() - start_time
            self.logger.info(f"✅ Certificate generated in {processing_time:.2f}s")
            
            return {
                "success": True,
                "certificate": certificate,
                "llm_output": generated_text,
                "context_chunks": context_chunks,
                "processing_time": processing_time,
                "model_used": model_config,
                "cached": False,
            }
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "certificate": None,
                "llm_output": "",
                "context_chunks": 0,
                "processing_time": time.time() - start_time,
            }

    def _retrieve_context(self, query: str, k: int = 3):
        """Retrieve relevant context from knowledge base."""
        if not self.embedding_model or not self.kb_cache:
            return ""
        
        try:
            # Simple similarity search (would use FAISS in production)
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            kb_data = self.kb_cache.get("unified", {})
            if not kb_data:
                return ""
            
            # Get query embedding
            query_emb = self.embedding_model.encode([query])
            
            # Simplified context retrieval
            context_pieces = []
            for i, item in enumerate(kb_data.get("chunks", [])[:k]):
                context_pieces.append(f"Chunk {i+1}: {item[:200]}...")
            
            return "\n".join(context_pieces)
            
        except Exception as e:
            self.logger.warning(f"Context retrieval failed: {e}")
            return ""

    def _build_barrier_certificate_prompt(self, system_desc: str, context: str, domain_bounds: dict):
        """Build the prompt for barrier certificate generation."""
        prompt = f"""You are an expert in control theory and barrier certificates. Generate a barrier certificate for the following system.

System Description:
{system_desc}

"""
        
        if context:
            prompt += f"Relevant Context:\n{context}\n\n"
        
        if domain_bounds:
            prompt += f"Domain Bounds: {domain_bounds}\n\n"
        
        prompt += """Please provide a barrier certificate in the form V(x) that satisfies:
1. V(x) > 0 for all x in the unsafe set
2. V(x) ≤ 0 for all x in the initial set  
3. ∇V(x) · f(x) ≤ 0 for all x where V(x) = 0

Barrier Certificate:"""
        
        return prompt

    def _extract_certificate(self, text: str):
        """Extract the barrier certificate from generated text."""
        # Simple extraction - in production would use more sophisticated parsing
        lines = text.split('\n')
        for line in lines:
            if 'V(' in line or 'barrier' in line.lower():
                return line.strip()
        
        # Fallback - return first non-empty line
        for line in lines:
            if line.strip():
                return line.strip()
                
        return "V(x) = x'*P*x - 1"  # Default example

# Web endpoint for external access
@app.function(image=inference_image)
@modal.web_endpoint(method="POST", label="fm-llm-generate")
def web_generate(item: dict):
    """Web endpoint for barrier certificate generation."""
    inference_service = InferenceAPI()
    
    return inference_service.generate_certificate(
        system_description=item.get("system_description", ""),
        model_config=item.get("model_config", "base"),
        rag_k=item.get("rag_k", 3),
        domain_bounds=item.get("domain_bounds"),
        request_id=item.get("request_id", "web-request"),
    )

@app.function(image=inference_image)
@modal.web_endpoint(method="GET", label="fm-llm-health")
def web_health():
    """Health check web endpoint."""
    inference_service = InferenceAPI()
    return inference_service.health_check()

# FastAPI app for full API compatibility
@app.function(
    image=inference_image,
    gpu=GPU_CONFIG,
    volumes={"/models": models_volume, "/kb": kb_volume},
    container_idle_timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    """Full FastAPI application for compatibility."""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import Optional, Dict, Any
    
    api = FastAPI(
        title="FM-LLM Solver Inference API",
        description="Serverless GPU barrier certificate generation",
        version="1.0.0"
    )
    
    # Add CORS
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request models
    class GenerateRequest(BaseModel):
        system_description: str
        model_config: str = "base"
        rag_k: int = 3
        domain_bounds: Optional[Dict[str, Any]] = None
        request_id: Optional[str] = None
    
    # Initialize inference service
    inference_service = InferenceAPI()
    
    @api.get("/health")
    def health_check():
        return inference_service.health_check()
    
    @api.post("/generate")
    def generate_certificate(request: GenerateRequest):
        return inference_service.generate_certificate(
            system_description=request.system_description,
            model_config=request.model_config,
            rag_k=request.rag_k,
            domain_bounds=request.domain_bounds,
            request_id=request.request_id,
        )
    
    return api

if __name__ == "__main__":
    # Deploy the app
    print("🚀 Deploying FM-LLM Solver to Modal...")
    print("📡 Endpoints will be available at:")
    print("   - Health: https://your-workspace--fm-llm-health.modal.run")
    print("   - Generate: https://your-workspace--fm-llm-generate.modal.run")
    print("   - Full API: https://your-workspace--fm-llm-solver.modal.run") 