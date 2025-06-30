import os
import sys
import re
import json
import logging
from typing import Dict, List, Optional, Any
import requests
import time

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from inference.generate_certificate import (
    load_knowledge_base, load_finetuned_model, retrieve_context, 
    format_prompt_with_context
)
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from knowledge_base.kb_utils import get_active_kb_paths, determine_kb_type_from_config, validate_kb_config

logger = logging.getLogger(__name__)

class CertificateGenerator:
    """Service for generating barrier certificates using different model configurations."""
    
    def __init__(self, config):
        """Initialize the certificate generator with configuration."""
        self.config = config
        self.deployment_mode = config.deployment.mode
        self.inference_api_url = None
        
        # For local mode, import inference modules
        if self.deployment_mode == "local":
            from inference.generate_certificate import (
                load_knowledge_base, load_finetuned_model, retrieve_context, 
                format_prompt_with_context
            )
            from sentence_transformers import SentenceTransformer
            from transformers import pipeline
            
            # Store imported functions
            self.load_knowledge_base = load_knowledge_base
            self.load_finetuned_model = load_finetuned_model
            self.retrieve_context = retrieve_context
            self.format_prompt_with_context = format_prompt_with_context
            self.SentenceTransformer = SentenceTransformer
            self.pipeline = pipeline
            
            self.models = {}  # Cache for loaded models
            self.embedding_model = None
            self.knowledge_bases = {}  # Cache for knowledge bases
        
        # For hybrid/cloud mode, use inference API
        else:
            self.inference_api_url = os.environ.get(
                'INFERENCE_API_URL',
                config.deployment.cloud.inference_api_url or 'http://inference:8000'
            )
            logger.info(f"Using inference API at: {self.inference_api_url}")
        
        # Validate configuration (non-blocking)
        try:
            if not validate_kb_config(config):
                logger.warning("Knowledge base configuration validation failed. Some features may be limited.")
        except Exception as e:
            logger.warning(f"Could not validate knowledge base configuration: {e}. Some features may be limited.")
        
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available model configurations."""
        if self.deployment_mode != "local":
            # Query inference API for available models
            try:
                response = requests.get(f"{self.inference_api_url}/models", timeout=10)
                if response.status_code == 200:
                    api_models = response.json().get('models', [])
                    return [
                        {
                            'key': model['key'],
                            'name': f"{model['key'].title()} Model",
                            'description': f"Model configuration: {model['key']}",
                            'type': 'finetuned' if model.get('use_adapter') else 'base',
                            'barrier_type': self.config.knowledge_base.barrier_certificate_type
                        }
                        for model in api_models
                    ]
            except Exception as e:
                logger.error(f"Failed to query inference API for models: {e}")
                # Fall back to default models
        
        # Local mode or fallback
        models = []
        
        # Base model configuration
        models.append({
            'key': 'base',
            'name': 'Base Model',
            'description': f"Base model without fine-tuning: {self.config.fine_tuning.base_model_name}",
            'type': 'base',
            'barrier_type': self.config.knowledge_base.barrier_certificate_type
        })
        
        # Fine-tuned model configuration
        adapter_path = os.path.join(self.config.paths.ft_output_dir, "final_adapter")
        if os.path.exists(adapter_path) or self.deployment_mode != "local":
            models.append({
                'key': 'finetuned',
                'name': 'Fine-tuned Model',
                'description': f"Fine-tuned model with adapter: {self.config.fine_tuning.base_model_name}",
                'type': 'finetuned',
                'barrier_type': self.config.knowledge_base.barrier_certificate_type
            })
        
        # Add discrete/continuous specific models if available
        if self.config.knowledge_base.barrier_certificate_type == 'unified':
            # Check for discrete-specific models
            discrete_adapter = os.path.join(self.config.paths.ft_output_dir, "discrete_adapter")
            if os.path.exists(discrete_adapter) or self.deployment_mode != "local":
                models.append({
                    'key': 'discrete',
                    'name': 'Discrete Fine-tuned Model',
                    'description': 'Model specifically fine-tuned for discrete barrier certificates',
                    'type': 'finetuned',
                    'barrier_type': 'discrete'
                })
            
            # Check for continuous-specific models
            continuous_adapter = os.path.join(self.config.paths.ft_output_dir, "continuous_adapter")
            if os.path.exists(continuous_adapter) or self.deployment_mode != "local":
                models.append({
                    'key': 'continuous',
                    'name': 'Continuous Fine-tuned Model',
                    'description': 'Model specifically fine-tuned for continuous barrier certificates',
                    'type': 'finetuned',
                    'barrier_type': 'continuous'
                })
        
        return models
    
    def _get_model_config(self, model_key: str) -> Dict[str, Any]:
        """Get model configuration for a specific model key."""
        if model_key == 'base':
            return {
                'use_adapter': False,
                'base_model_name': self.config.fine_tuning.base_model_name,
                'adapter_path': None,
                'barrier_type': self.config.knowledge_base.barrier_certificate_type
            }
        elif model_key == 'finetuned':
            return {
                'use_adapter': True,
                'base_model_name': self.config.fine_tuning.base_model_name,
                'adapter_path': os.path.join(self.config.paths.ft_output_dir, "final_adapter"),
                'barrier_type': self.config.knowledge_base.barrier_certificate_type
            }
        elif model_key == 'discrete':
            return {
                'use_adapter': True,
                'base_model_name': self.config.fine_tuning.base_model_name,
                'adapter_path': os.path.join(self.config.paths.ft_output_dir, "discrete_adapter"),
                'barrier_type': 'discrete'
            }
        elif model_key == 'continuous':
            return {
                'use_adapter': True,
                'base_model_name': self.config.fine_tuning.base_model_name,
                'adapter_path': os.path.join(self.config.paths.ft_output_dir, "continuous_adapter"),
                'barrier_type': 'continuous'
            }
        else:
            raise ValueError(f"Unknown model key: {model_key}")
    
    def _load_embedding_model(self):
        """Load the embedding model for RAG."""
        if self.deployment_mode != "local":
            raise RuntimeError("_load_embedding_model should only be called in local mode")
        
        if self.embedding_model is None:
            try:
                self.embedding_model = self.SentenceTransformer(
                    self.config.knowledge_base.embedding_model_name
                )
                logger.info(f"Loaded embedding model: {self.config.knowledge_base.embedding_model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        return self.embedding_model
    
    def _load_knowledge_base(self, barrier_type: str):
        """Load knowledge base for the specified barrier certificate type."""
        if self.deployment_mode != "local":
            raise RuntimeError("_load_knowledge_base should only be called in local mode")
        
        if barrier_type not in self.knowledge_bases:
            try:
                # Get KB paths based on barrier type
                if barrier_type == 'discrete':
                    kb_dir = self.config.paths.kb_discrete_output_dir
                    vector_file = self.config.paths.kb_discrete_vector_store_filename
                    metadata_file = self.config.paths.kb_discrete_metadata_filename
                elif barrier_type == 'continuous':
                    kb_dir = self.config.paths.kb_continuous_output_dir
                    vector_file = self.config.paths.kb_continuous_vector_store_filename
                    metadata_file = self.config.paths.kb_continuous_metadata_filename
                else:  # unified or default
                    kb_dir = self.config.paths.kb_output_dir
                    vector_file = self.config.paths.kb_vector_store_filename
                    metadata_file = self.config.paths.kb_metadata_filename
                
                index, metadata = self.load_knowledge_base(kb_dir, vector_file, metadata_file)
                if index is None or metadata is None:
                    logger.warning(f"No knowledge base found for {barrier_type}. RAG will be disabled.")
                    self.knowledge_bases[barrier_type] = None
                    return None
                
                self.knowledge_bases[barrier_type] = {
                    'index': index,
                    'metadata': metadata
                }
                logger.info(f"Loaded {barrier_type} knowledge base with {index.ntotal} vectors")
                
            except Exception as e:
                logger.warning(f"Failed to load {barrier_type} knowledge base: {e}. RAG will be disabled.")
                self.knowledge_bases[barrier_type] = None
                return None
        
        return self.knowledge_bases[barrier_type]
    
    def _load_model(self, model_key: str):
        """Load and cache a model configuration."""
        if self.deployment_mode != "local":
            raise RuntimeError("_load_model should only be called in local mode")
        
        if model_key not in self.models:
            try:
                model_config = self._get_model_config(model_key)
                
                # Create temporary config for model loading
                temp_config = self.config.copy()
                temp_config.fine_tuning.use_adapter = model_config['use_adapter']
                
                model, tokenizer = self.load_finetuned_model(
                    model_config['base_model_name'],
                    model_config['adapter_path'],
                    temp_config
                )
                
                if model is None or tokenizer is None:
                    raise ValueError(f"Failed to load model: {model_key}")
                
                # Create optimized pipeline for barrier certificate generation
                pipe = self.pipeline(
                    task="text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=self.config.inference.max_new_tokens,
                    temperature=self.config.inference.temperature,
                    top_p=self.config.inference.top_p,
                    do_sample=self.config.inference.get('do_sample', True),
                    repetition_penalty=self.config.inference.get('repetition_penalty', 1.1),
                    # Add stop sequences to prevent incomplete generation
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                self.models[model_key] = {
                    'pipeline': pipe,
                    'config': model_config
                }
                logger.info(f"Loaded model: {model_key}")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_key}: {e}")
                raise
        
        return self.models[model_key]
    
    def extract_certificate_from_output(self, llm_output: str) -> Optional[str]:
        """Extract barrier certificate from LLM output with enhanced placeholder detection."""
        # Primary: Look for BARRIER_CERTIFICATE_START block
        pattern = r'BARRIER_CERTIFICATE_START\s*\n?.*?B\s*\([^)]+\)\s*=\s*([^\n\r]+)'
        match = re.search(pattern, llm_output, re.DOTALL | re.IGNORECASE)
        
        if match:
            expression = match.group(1).strip()
            cleaned_expr = self._clean_certificate_expression(expression)
            
            # Enhanced placeholder detection
            if self._has_placeholder_variables(cleaned_expr):
                logger.warning(f"REJECTED: Contains placeholder variables: {cleaned_expr}")
                return None
                
            if self._is_template_expression(cleaned_expr):
                logger.warning(f"REJECTED: Template expression: {cleaned_expr}")
                return None
            
            # Validate it contains actual variables
            if self._contains_state_variables(cleaned_expr):
                logger.info(f"Successfully extracted certificate: {cleaned_expr}")
                return cleaned_expr
        
        # Fallback: look for B(...) = ... pattern anywhere
        fallback_pattern = r'B\s*\([^)]+\)\s*=\s*([^\n\r]+)'
        fallback_match = re.search(fallback_pattern, llm_output, re.IGNORECASE)
        
        if fallback_match:
            expression = fallback_match.group(1).strip()
            cleaned_expr = self._clean_certificate_expression(expression)
            
            if self._has_placeholder_variables(cleaned_expr):
                logger.warning(f"REJECTED (fallback): Contains placeholder variables: {cleaned_expr}")
                return None
                
            if self._is_template_expression(cleaned_expr):
                logger.warning(f"REJECTED (fallback): Template expression: {cleaned_expr}")
                return None
                
            if self._contains_state_variables(cleaned_expr):
                logger.info(f"Successfully extracted certificate (fallback): {cleaned_expr}")
                return cleaned_expr
        
        # Enhanced fallback: look for mathematical expressions after = sign
        math_pattern = r'=\s*([x\d\.\*\+\-\s\*\*\(\)]+[^\n\r]*)'
        math_match = re.search(math_pattern, llm_output)
        if math_match:
            expression = math_match.group(1).strip()
            cleaned_expr = self._clean_certificate_expression(expression)
            
            # Check if this is actually a valid mathematical expression (not a system equation)
            if ('k]' in cleaned_expr or 'k+1]' in cleaned_expr or 
                'and' in cleaned_expr or ',' in cleaned_expr):
                # This is likely a system equation, not a certificate
                logger.debug(f"Rejected math pattern - appears to be system equation: {cleaned_expr}")
                return None
            
            # Only accept if it looks like a valid mathematical expression
            if self._contains_state_variables(cleaned_expr) and not self._has_placeholder_variables(cleaned_expr):
                logger.info(f"Successfully extracted certificate (math pattern): {cleaned_expr}")
                return cleaned_expr
        
        logger.warning("Failed to extract valid certificate from LLM output")
        return None
    
    def _clean_certificate_expression(self, expression: str) -> str:
        """Clean LaTeX artifacts and other formatting issues from certificate expressions."""
        if not expression:
            return expression
        
        # FIRST: Remove any text after the mathematical expression
        # Stop at common sentence starters or descriptive phrases
        sentence_patterns = [
            r'\s+is\s+.*$',                 # "is a valid...", "is appropriate..."
            r'\s+which\s+.*$',               # "which ensures..."
            r'\s+this\s+.*$',                # "this guarantees..."
            r'\s+that\s+.*$',                # "that proves..."
            r'\s+for\s+.*$',                 # "for the system..."
            r'\s+where\s+.*$',               # "where x and y..."
            r'\s+ensuring\s+.*$',            # "ensuring stability..."
            r'\s+guaranteeing\s+.*$',        # "guaranteeing safety..."
            r'\s+could\s+be\s+.*$',          # "could be appropriate"
            r'\s+would\s+be\s+.*$',          # "would be appropriate"
            r'\s+seems\s+.*$',               # "seems appropriate"
            r'\s+appears\s+.*$',             # "appears suitable"
            r'\s+might\s+be\s+.*$',          # "might be appropriate"
            r'\s+should\s+be\s+.*$',         # "should be suitable"
            r'\s+can\s+be\s+used\s*$',      # "can be used"
            r'\s+satisfies\s+.*$',           # "satisfies the conditions"
            r'\s+represents\s+.*$',          # "represents a barrier"
            r'\[/?INST\].*$',               # Remove [INST] or [/INST] and everything after
            r'\s+Great!.*$',                # "Great! That's correct..."
            r'\s+[Tt]hat\'s\s+correct.*$',  # "That's correct..."
        ]
        
        cleaned = expression
        for pattern in sentence_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove trailing [CUT] or similar bracketed text
        cleaned = re.sub(r'\s*\[.*?\]\s*$', '', cleaned).strip()
        
        # Remove LaTeX brackets and delimiters
        cleaned = re.sub(r'\\[\[\]()]', '', cleaned)  # Remove \[ \] \( \)
        cleaned = re.sub(r'\\[\{\}]', '', cleaned)    # Remove \{ \}
        
        # Remove standalone LaTeX brackets at the end
        cleaned = re.sub(r'\s*\\\]\s*$', '', cleaned)  # Remove trailing \]
        cleaned = re.sub(r'\s*\\\[\s*$', '', cleaned)  # Remove trailing \[
        cleaned = re.sub(r'\s*\\\)\s*$', '', cleaned)  # Remove trailing \)
        cleaned = re.sub(r'\s*\\\(\s*$', '', cleaned)  # Remove trailing \(
        
        # Convert LaTeX math operators to standard notation
        cleaned = cleaned.replace('\\cdot', '*')
        cleaned = cleaned.replace('\\times', '*')
        cleaned = cleaned.replace('\\div', '/')
        cleaned = cleaned.replace('^', '**')           # Convert exponentiation
        
        # Remove LaTeX commands that might appear
        cleaned = re.sub(r'\\[a-zA-Z]+\s*', '', cleaned)  # Remove LaTeX commands like \alpha, \beta
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        # Remove trailing punctuation that might cause parsing issues
        cleaned = cleaned.rstrip('.,;:')
        
        logger.debug(f"Cleaned certificate expression: '{expression}' -> '{cleaned}'")
        return cleaned
    
    def _has_placeholder_variables(self, expression: str) -> bool:
        """Check if expression contains placeholder variables (Greek letters, single letters, etc.)."""
        if not expression:
            return True
        
        # Greek letters and common placeholders
        placeholders = [
            'α', 'β', 'γ', 'δ', 'λ', 'μ', 'θ', 'ρ', 'σ', 'τ', 'φ', 'χ', 'ψ', 'ω',
            '\\alpha', '\\beta', '\\gamma', '\\delta', '\\lambda', '\\mu', '\\theta',
            '\\rho', '\\sigma', '\\tau', '\\phi', '\\chi', '\\psi', '\\omega'
        ]
        
        # Check for Greek letters and LaTeX placeholders
        for placeholder in placeholders:
            if placeholder in expression:
                return True
        
        # More specific patterns for template variables
        # Look for single letters that are multiplied by variables or used as coefficients
        template_patterns = [
            r'\b[a-h]\s*\*\s*x',      # a*x, b*x, etc.
            r'\b[a-h]\s*\*\s*y',      # a*y, b*y, etc.
            r'\b[a-h]\s*\*\s*z',      # a*z, b*z, etc.
            r'\b[a-h]\s*x\b',         # ax, bx (without *)
            r'\b[a-h]\s*y\b',         # ay, by (without *)
            r'\b[a-h]\s*z\b',         # az, bz (without *)
            r'\b[A-HJ-W]\s*x\b',      # Ax, Bx, etc.
            r'\b[A-HJ-W]\s*y\b',      # Ay, By, etc.
            r'\b[A-HJ-W]\s*z\b',      # Az, Bz, etc.
            r'\b[a-zA-Z]\s*_\d+',     # a_1, b_2, etc.
            r'\bc_\d+',               # c_1, c_2, etc.
            r'coeff',                 # coefficient
            r'param',                 # parameter
        ]
        
        for pattern in template_patterns:
            if re.search(pattern, expression, re.IGNORECASE):
                return True
        
        # Don't reject expressions that are just numbers with x, y, z
        # like "2.1*x**2 + 3.5*y**2 - 1.0"
        
        return False
    
    def _contains_state_variables(self, expression: str) -> bool:
        """Check if expression contains actual state variables (x, y, z)."""
        if not expression:
            return False
        
        # Look for state variables x, y, z
        state_var_pattern = r'\b[xyz]\b'
        return bool(re.search(state_var_pattern, expression, re.IGNORECASE))
    
    def _is_template_expression(self, expression: str) -> bool:
        """Check if the expression is a template/placeholder rather than a concrete barrier certificate."""
        if not expression:
            return True
        
        # If it already has placeholder variables, it's a template
        if self._has_placeholder_variables(expression):
            return True
        
        # Common template patterns that should be rejected
        template_patterns = [
            # Template variables (single letters as coefficients) - more precise matching
            r'\b[a-h]\*[xy]',  # ax, bx, cy, etc. (single letters multiplying state variables)
            r'\b[a-h]\*\*2',   # a**2, b**2, etc. (single letters as base)
            r'\b[a-h][xy]\b',  # ax, by, etc. (single letter followed by state variable)
            
            # Common placeholder naming
            r'\bc[0-9]',  # c1, c2, c3, etc.
            r'\bcoeff',   # coeff, coefficient
            r'\bparam',   # param, parameter
            
            # Standalone single letters that aren't state variables
            r'^\s*[a-h]\s*$',  # Just 'a' or 'b' etc.
            r'^\s*[a-h]\s*\+',  # Starting with single letter like 'a +'
        ]
        
        # Check for any template patterns
        for pattern in template_patterns:
            if re.search(pattern, expression, re.IGNORECASE):
                logger.debug(f"Template pattern detected: '{pattern}' in '{expression}'")
                return True
        
        # Additional heuristic: if expression has more than 3 single-letter variables 
        # (excluding x, y, z which are likely state variables), it's probably a template
        single_letters = re.findall(r'\b[a-h]\b', expression.lower())  # Only check a-h, not all letters
        
        if len(set(single_letters)) >= 3:  # 3 or more different single-letter template variables
            logger.debug(f"Too many template variables detected: {set(single_letters)} in '{expression}'")
            return True
        
        return False
    
    def generate_certificate(self, system_description: str, model_key: str, rag_k: int = 3, domain_bounds: dict = None) -> Dict[str, Any]:
        """Generate a barrier certificate for the given system description."""
        if self.deployment_mode == "local":
            return self._generate_local(system_description, model_key, rag_k, domain_bounds)
        else:
            return self._generate_remote(system_description, model_key, rag_k, domain_bounds)
    
    def _generate_remote(self, system_description: str, model_key: str, rag_k: int, domain_bounds: dict = None) -> Dict[str, Any]:
        """Generate certificate using remote inference API."""
        try:
            logger.info(f"Generating certificate via API with model: {model_key}, RAG k: {rag_k}")
            
            # Prepare request
            request_data = {
                "system_description": system_description,
                "model_config": model_key,
                "rag_k": rag_k,
                "domain_bounds": domain_bounds,
                "request_id": f"web-{int(time.time())}"
            }
            
            # Call inference API
            response = requests.post(
                f"{self.inference_api_url}/generate",
                json=request_data,
                timeout=self.config.deployment.services.inference.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': result.get('success', False),
                    'certificate': result.get('certificate'),
                    'llm_output': result.get('llm_output', ''),
                    'context_chunks': result.get('context_chunks', 0),
                    'error': result.get('error'),
                    'cached': result.get('cached', False),
                    'processing_time': result.get('processing_time', 0)
                }
            else:
                error_msg = f"Inference API error: {response.status_code}"
                if response.text:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('detail', error_msg)
                    except:
                        error_msg = response.text[:200]
                
                logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg
                }
                
        except requests.Timeout:
            logger.error("Inference API timeout")
            return {
                'success': False,
                'error': 'Inference API timeout. Please try again.'
            }
        except requests.ConnectionError:
            logger.error("Cannot connect to inference API")
            return {
                'success': False,
                'error': 'Cannot connect to inference service. Please check if the service is running.'
            }
        except Exception as e:
            logger.error(f"Remote generation error: {str(e)}")
            return {
                'success': False,
                'error': f'Failed to generate certificate: {str(e)}'
            }
    
    def _generate_local(self, system_description: str, model_key: str, rag_k: int, domain_bounds: dict = None) -> Dict[str, Any]:
        """Generate certificate using local models (original implementation)."""
        try:
            logger.info(f"Generating certificate locally with model: {model_key}, RAG k: {rag_k}")
            if domain_bounds:
                logger.info(f"Using domain bounds: {domain_bounds}")
            
            # Load required components
            model_info = self._load_model(model_key)
            kb_info = self._load_knowledge_base(model_info['config']['barrier_type'])
            
            # Retrieve context if RAG is enabled and knowledge base is available
            context = ""
            context_chunks = 0
            if rag_k > 0 and kb_info is not None:
                try:
                    embedding_model = self._load_embedding_model()
                    context = self.retrieve_context(
                        system_description,
                        embedding_model,
                        kb_info['index'],
                        kb_info['metadata'],
                        rag_k
                    )
                    context_chunks = context.count('--- Context Chunk') if context else 0
                except Exception as e:
                    logger.warning(f"RAG retrieval failed: {e}. Continuing without context.")
            elif rag_k > 0 and kb_info is None:
                logger.info("RAG requested but no knowledge base available. Continuing without context.")
            
            # Try generation with multiple attempts if needed
            max_attempts = 3
            certificate = None
            llm_output = ""
            
            for attempt in range(max_attempts):
                logger.info(f"Generation attempt {attempt + 1}/{max_attempts}")
                
                try:
                    # Use enhanced prompt if first attempt with standard prompt fails
                    if attempt == 0:
                        # Standard prompt
                        prompt = self.format_prompt_with_context(
                            system_description,
                            context,
                            model_info['config']['barrier_type'],
                            domain_bounds
                        )
                    else:
                        # Enhanced prompt with few-shot examples
                        prompt = self._create_enhanced_prompt(
                            system_description,
                            context,
                            model_info['config']['barrier_type'],
                            domain_bounds
                        )
                    
                    # Generate certificate
                    result = model_info['pipeline'](prompt)
                    
                    # Handle the result properly - it returns a list of dicts
                    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                        generated_text = result[0].get('generated_text', '')
                    else:
                        logger.error(f"Unexpected pipeline result format: {type(result)}")
                        if hasattr(result, '__iter__'):
                            logger.error(f"Result content: {result}")
                        continue
                    
                    # Extract only the generated part after the prompt
                    prompt_end_marker = "[/INST]"
                    output_start_index = generated_text.find(prompt_end_marker)
                    if output_start_index != -1:
                        llm_output = generated_text[output_start_index + len(prompt_end_marker):].strip()
                    else:
                        llm_output = generated_text
                    
                    # Extract certificate
                    certificate = self.extract_certificate_from_output(llm_output)
                    
                    if certificate:
                        logger.info(f"Successfully extracted certificate on attempt {attempt + 1}")
                        break
                    else:
                        logger.warning(f"Attempt {attempt + 1} failed to extract valid certificate")
                        
                except Exception as e:
                    logger.error(f"Error during generation attempt {attempt + 1}: {str(e)}")
                    logger.error(f"Error type: {type(e).__name__}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue
            
            return {
                'success': True,
                'llm_output': llm_output,
                'certificate': certificate,
                'context_chunks': context_chunks,
                'model_config': model_info['config'],
                'prompt_length': len(prompt),
                'domain_bounds': domain_bounds
            }
            
        except Exception as e:
            logger.error(f"Error generating certificate: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'llm_output': None,
                'certificate': None,
                'context_chunks': 0
            }
    
    def _create_enhanced_prompt(self, system_description: str, context: str, barrier_type: str, domain_bounds: dict = None) -> str:
        """Create an enhanced prompt with few-shot examples for better generation."""
        
        # State variables are always x, y as specified
        state_vars = ['x', 'y']
        var_string = ", ".join(state_vars)
        
        # Few-shot examples for DISCRETE-TIME systems
        examples = """
EXAMPLE 1:
Discrete-time system: x[k+1] = 0.9*x[k] + 0.1*y[k], y[k+1] = -0.1*x[k] + 0.8*y[k]
Domain: x ∈ [-5, 5], y ∈ [-5, 5]
BARRIER_CERTIFICATE_START
B(x,y) = 2.0*x**2 + 1.0*x*y + 3.0*y**2

EXAMPLE 2:  
Discrete-time system: x[k+1] = 0.95*x[k] - 0.2*y[k], y[k+1] = 0.1*x[k] + 0.85*y[k]
Domain: x ∈ [-10, 10], y ∈ [-10, 10]
BARRIER_CERTIFICATE_START
B(x,y) = x**2 + y**2 - 4.0

EXAMPLE 3:
Discrete-time system: x[k+1] = 0.8*x[k] + 0.3*y[k], y[k+1] = -0.2*x[k] + 0.9*y[k]
Domain: x ∈ [-3, 3], y ∈ [-3, 3]
BARRIER_CERTIFICATE_START
B(x,y) = 5.0*x**2 - 2.0*x*y + 4.0*y**2
"""
        
        # Domain bounds info
        domain_text = ""
        if domain_bounds:
            try:
                # Handle different domain bounds formats
                if isinstance(domain_bounds, dict):
                    parts = []
                    for var, bounds in domain_bounds.items():
                        if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
                            # Handle list/tuple format [min, max] - this is what frontend sends
                            parts.append(f"{var} ∈ [{bounds[0]}, {bounds[1]}]")
                        elif isinstance(bounds, dict) and 'min' in bounds and 'max' in bounds:
                            # Handle dict format with 'min' and 'max' keys
                            parts.append(f"{var} ∈ [{bounds['min']}, {bounds['max']}]")
                    if parts:
                        domain_text = f"\nDomain: {', '.join(parts)}"
            except Exception as e:
                logger.warning(f"Error formatting domain bounds: {e}")
                logger.warning(f"Domain bounds structure: {domain_bounds}")
                # Continue without domain bounds rather than failing
        
        # Context info
        context_text = ""
        if context:
            context_text = f"\n\nRelevant research context:\n{context[:500]}...\n"
        
        # Ensure domain is always specified
        if not domain_bounds or not domain_text:
            # Default domain if not specified
            domain_text = "\nDomain: x ∈ [-10, 10], y ∈ [-10, 10]"
            logger.info("No domain bounds specified, using default: x,y ∈ [-10, 10]")
        
        prompt = f"""<s>[INST] You are an expert in discrete-time systems and barrier certificates.

TASK: Generate a DISCRETE-TIME barrier certificate for the given dynamical system.

SYSTEM INFORMATION:
- State variables: x, y (always use these two variables)
- This is a DISCRETE-TIME system (uses k, k+1 notation)
- Domain bounds are explicitly defined

CRITICAL RULES:
1. Use ONLY concrete numerical values (like 1.0, 2.5, -3.0)
2. NEVER use placeholder variables (no a, b, c, α, β, etc.)
3. COMPLETE the mathematical expression - no "to be determined" values
4. Your output MUST contain exactly one line starting with "B(x,y) ="
5. The barrier certificate must be suitable for DISCRETE-TIME analysis
6. Output ONLY the barrier certificate expression, nothing else

{examples}

Now generate a discrete-time barrier certificate for:

{system_description}{domain_text}{context_text}

Remember: 
- Use ONLY concrete numbers (like 2.0, 5.0, not a, b, c)
- State variables are x and y
- This is for a DISCRETE-TIME system
- Output ONLY: B(x,y) = <expression>
- Nothing else after the expression

BARRIER_CERTIFICATE_START
B(x,y) = [/INST]"""
        
        return prompt
    
    def test_model_availability(self, model_key: str) -> Dict[str, Any]:
        """Test if a model is available and can be loaded."""
        try:
            model_config = self._get_model_config(model_key)
            
            # Check if adapter path exists (for fine-tuned models)
            if model_config['use_adapter'] and model_config['adapter_path']:
                if not os.path.exists(model_config['adapter_path']):
                    return {
                        'available': False,
                        'error': f"Adapter path not found: {model_config['adapter_path']}"
                    }
            
            # Check if knowledge base exists for the barrier type (optional)
            barrier_type = model_config['barrier_type']
            kb_available = False
            try:
                kb_info = self._load_knowledge_base(barrier_type)
                kb_available = kb_info is not None
            except Exception as e:
                logger.warning(f"Knowledge base check failed for {barrier_type}: {e}")
                kb_available = False
            
            return {
                'available': True,
                'config': model_config,
                'knowledge_base_available': kb_available
            }
            
        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            } 