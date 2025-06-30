"""
Inference API for FM-LLM Solver
Provides high-performance barrier certificate generation with caching and queuing support.
"""

import os
import sys
import time
import logging
import asyncio
import hashlib
import json
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.config_loader import load_config
from inference.generate_certificate import (
    load_knowledge_base, load_finetuned_model, retrieve_context,
    format_prompt_with_context
)
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from knowledge_base.kb_utils import get_active_kb_paths, determine_kb_type_from_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model caching
models_cache = {}
kb_cache = {}
embedding_model = None
config = None
redis_client = None

# Request/Response models
class InferenceRequest(BaseModel):
    """Request model for barrier certificate generation."""
    system_description: str = Field(..., description="System dynamics, initial set, and unsafe set")
    model_config: str = Field("finetuned", description="Model configuration to use")
    rag_k: int = Field(3, ge=0, le=10, description="Number of RAG context chunks")
    domain_bounds: Optional[Dict[str, List[float]]] = Field(None, description="Domain bounds for variables")
    request_id: Optional[str] = Field(None, description="Optional request ID for tracking")

class InferenceResponse(BaseModel):
    """Response model for barrier certificate generation."""
    success: bool
    certificate: Optional[str] = None
    llm_output: Optional[str] = None
    context_chunks: int = 0
    error: Optional[str] = None
    request_id: Optional[str] = None
    processing_time: float = 0.0
    cached: bool = False

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    models_loaded: List[str]
    kb_loaded: bool
    cache_enabled: bool
    gpu_available: bool

# Cache utilities
def get_cache_key(request: InferenceRequest) -> str:
    """Generate cache key for request."""
    key_data = {
        'system': request.system_description,
        'model': request.model_config,
        'rag_k': request.rag_k,
        'bounds': request.domain_bounds
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()

async def get_cached_result(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get cached result if available."""
    if redis_client is None:
        return None
    
    try:
        result = await redis_client.get(cache_key)
        if result:
            return json.loads(result)
    except Exception as e:
        logger.warning(f"Cache retrieval error: {e}")
    
    return None

async def cache_result(cache_key: str, result: Dict[str, Any], ttl: int = 3600):
    """Cache result with TTL."""
    if redis_client is None:
        return
    
    try:
        await redis_client.setex(
            cache_key,
            ttl,
            json.dumps(result)
        )
    except Exception as e:
        logger.warning(f"Cache storage error: {e}")

# Model loading utilities
def load_models():
    """Load all configured models on startup."""
    global config, embedding_model
    
    logger.info("Loading models...")
    
    # Load embedding model
    try:
        embedding_model = SentenceTransformer(
            config.knowledge_base.embedding_model_name
        )
        logger.info(f"Loaded embedding model: {config.knowledge_base.embedding_model_name}")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise
    
    # Load configured model variants
    model_configs = [
        ('base', False, None),
        ('finetuned', True, os.path.join(config.paths.ft_output_dir, "final_adapter")),
        ('discrete', True, os.path.join(config.paths.ft_output_dir, "discrete_adapter")),
        ('continuous', True, os.path.join(config.paths.ft_output_dir, "continuous_adapter"))
    ]
    
    for model_key, use_adapter, adapter_path in model_configs:
        try:
            # Skip if adapter doesn't exist
            if use_adapter and adapter_path and not os.path.exists(adapter_path):
                logger.info(f"Skipping {model_key} - adapter not found at {adapter_path}")
                continue
            
            # Create temporary config for model loading
            temp_config = config.copy()
            temp_config.fine_tuning.use_adapter = use_adapter
            
            model, tokenizer = load_finetuned_model(
                config.fine_tuning.base_model_name,
                adapter_path,
                temp_config
            )
            
            if model and tokenizer:
                # Create pipeline
                pipe = pipeline(
                    task="text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=config.inference.max_new_tokens,
                    temperature=config.inference.temperature,
                    top_p=config.inference.top_p,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                models_cache[model_key] = {
                    'pipeline': pipe,
                    'tokenizer': tokenizer,
                    'use_adapter': use_adapter
                }
                logger.info(f"Loaded model: {model_key}")
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")

def load_knowledge_bases():
    """Load knowledge bases on startup."""
    global kb_cache
    
    kb_types = ['unified', 'discrete', 'continuous']
    
    for kb_type in kb_types:
        try:
            if kb_type == 'discrete':
                kb_dir = config.paths.kb_discrete_output_dir
                vector_file = config.paths.kb_discrete_vector_store_filename
                metadata_file = config.paths.kb_discrete_metadata_filename
            elif kb_type == 'continuous':
                kb_dir = config.paths.kb_continuous_output_dir
                vector_file = config.paths.kb_continuous_vector_store_filename
                metadata_file = config.paths.kb_continuous_metadata_filename
            else:
                kb_dir = config.paths.kb_output_dir
                vector_file = config.paths.kb_vector_store_filename
                metadata_file = config.paths.kb_metadata_filename
            
            index, metadata = load_knowledge_base(kb_dir, vector_file, metadata_file)
            
            if index and metadata:
                kb_cache[kb_type] = {
                    'index': index,
                    'metadata': metadata
                }
                logger.info(f"Loaded {kb_type} knowledge base with {index.ntotal} vectors")
        except Exception as e:
            logger.warning(f"Failed to load {kb_type} knowledge base: {e}")

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global config, redis_client
    
    # Startup
    logger.info("Starting FM-LLM Solver Inference API...")
    
    # Load configuration
    config_path = os.environ.get('CONFIG_PATH', 'config/config.yaml')
    config = load_config(config_path)
    
    # Initialize Redis if enabled
    if config.deployment.performance.enable_caching:
        try:
            import aioredis
            redis_client = await aioredis.create_redis_pool(
                'redis://redis:6379',
                encoding='utf-8'
            )
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.warning(f"Redis connection failed, caching disabled: {e}")
            redis_client = None
    
    # Load models and knowledge bases
    load_models()
    load_knowledge_bases()
    
    logger.info("Inference API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Inference API...")
    if redis_client:
        redis_client.close()
        await redis_client.wait_closed()

# Create FastAPI app
app = FastAPI(
    title="FM-LLM Solver Inference API",
    version="1.0.0",
    description="High-performance barrier certificate generation API",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    import torch
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        models_loaded=list(models_cache.keys()),
        kb_loaded=len(kb_cache) > 0,
        cache_enabled=redis_client is not None,
        gpu_available=torch.cuda.is_available()
    )

# Main inference endpoint
@app.post("/generate", response_model=InferenceResponse)
async def generate_certificate(request: InferenceRequest):
    """Generate barrier certificate for the given system."""
    start_time = time.time()
    
    # Check cache first
    cache_key = get_cache_key(request)
    if config.deployment.performance.enable_caching:
        cached_result = await get_cached_result(cache_key)
        if cached_result:
            return InferenceResponse(
                **cached_result,
                cached=True,
                processing_time=time.time() - start_time
            )
    
    try:
        # Validate model exists
        if request.model_config not in models_cache:
            raise HTTPException(
                status_code=400,
                detail=f"Model configuration '{request.model_config}' not available"
            )
        
        model_info = models_cache[request.model_config]
        pipe = model_info['pipeline']
        
        # Determine barrier type
        barrier_type = 'unified'  # Default
        if request.model_config in ['discrete', 'continuous']:
            barrier_type = request.model_config
        
        # Get appropriate knowledge base
        kb_info = kb_cache.get(barrier_type)
        
        # Retrieve context if RAG enabled
        context = ""
        context_chunks = 0
        if request.rag_k > 0 and kb_info and embedding_model:
            try:
                context = retrieve_context(
                    request.system_description,
                    embedding_model,
                    kb_info['index'],
                    kb_info['metadata'],
                    request.rag_k
                )
                context_chunks = context.count('--- Context Chunk') if context else 0
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
        
        # Generate prompt
        prompt = format_prompt_with_context(
            request.system_description,
            context,
            barrier_type,
            request.domain_bounds
        )
        
        # Generate certificate
        result = pipe(prompt)
        generated_text = result[0]['generated_text']
        
        # Extract output
        prompt_end_marker = "[/INST]"
        output_start_index = generated_text.find(prompt_end_marker)
        if output_start_index != -1:
            llm_output = generated_text[output_start_index + len(prompt_end_marker):].strip()
        else:
            llm_output = generated_text
        
        # Extract certificate
        from web_interface.certificate_generator import CertificateGenerator
        generator = CertificateGenerator(config)
        certificate = generator.extract_certificate_from_output(llm_output)
        
        # Prepare response
        response_data = {
            'success': certificate is not None,
            'certificate': certificate,
            'llm_output': llm_output,
            'context_chunks': context_chunks,
            'error': None if certificate else "Failed to extract valid certificate",
            'request_id': request.request_id,
            'processing_time': time.time() - start_time
        }
        
        # Cache successful results
        if certificate and config.deployment.performance.enable_caching:
            await cache_result(
                cache_key,
                response_data,
                config.deployment.performance.cache_ttl
            )
        
        return InferenceResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        return InferenceResponse(
            success=False,
            error=str(e),
            request_id=request.request_id,
            processing_time=time.time() - start_time
        )

# Batch inference endpoint
@app.post("/generate_batch", response_model=List[InferenceResponse])
async def generate_batch(requests: List[InferenceRequest]):
    """Generate certificates for multiple systems in batch."""
    # Process requests concurrently
    tasks = [generate_certificate(req) for req in requests]
    results = await asyncio.gather(*tasks)
    return results

# Model information endpoint
@app.get("/models")
async def get_available_models():
    """Get list of available models."""
    return {
        "models": [
            {
                "key": key,
                "loaded": True,
                "use_adapter": info.get('use_adapter', False)
            }
            for key, info in models_cache.items()
        ]
    }

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "inference_api.main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info"
    ) 