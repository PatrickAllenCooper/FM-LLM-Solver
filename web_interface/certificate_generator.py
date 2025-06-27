import os
import sys
import re
import json
import logging
from typing import Dict, List, Optional, Any

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
        self.models = {}  # Cache for loaded models
        self.embedding_model = None
        self.knowledge_bases = {}  # Cache for knowledge bases
        
        # Validate configuration (non-blocking)
        try:
            if not validate_kb_config(config):
                logger.warning("Knowledge base configuration validation failed. Some features may be limited.")
        except Exception as e:
            logger.warning(f"Could not validate knowledge base configuration: {e}. Some features may be limited.")
        
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available model configurations."""
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
        if os.path.exists(adapter_path):
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
            if os.path.exists(discrete_adapter):
                models.append({
                    'key': 'discrete',
                    'name': 'Discrete Fine-tuned Model',
                    'description': 'Model specifically fine-tuned for discrete barrier certificates',
                    'type': 'finetuned',
                    'barrier_type': 'discrete'
                })
            
            # Check for continuous-specific models
            continuous_adapter = os.path.join(self.config.paths.ft_output_dir, "continuous_adapter")
            if os.path.exists(continuous_adapter):
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
        if self.embedding_model is None:
            try:
                self.embedding_model = SentenceTransformer(
                    self.config.knowledge_base.embedding_model_name
                )
                logger.info(f"Loaded embedding model: {self.config.knowledge_base.embedding_model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        return self.embedding_model
    
    def _load_knowledge_base(self, barrier_type: str):
        """Load knowledge base for the specified barrier certificate type."""
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
                
                index, metadata = load_knowledge_base(kb_dir, vector_file, metadata_file)
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
        if model_key not in self.models:
            try:
                model_config = self._get_model_config(model_key)
                
                # Create temporary config for model loading
                temp_config = self.config.copy()
                temp_config.fine_tuning.use_adapter = model_config['use_adapter']
                
                model, tokenizer = load_finetuned_model(
                    model_config['base_model_name'],
                    model_config['adapter_path'],
                    temp_config
                )
                
                if model is None or tokenizer is None:
                    raise ValueError(f"Failed to load model: {model_key}")
                
                # Create optimized pipeline for barrier certificate generation
                pipe = pipeline(
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
        
        # Remove common LaTeX artifacts
        cleaned = expression
        
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
        
        # Remove trailing text descriptions (common LLM habit)
        descriptive_patterns = [
            r'\s+where\s+.*$',
            r'\s+for\s+.*$',
            r'\s+such\s+that\s+.*$',
            r'\s+ensuring\s+.*$',
            r'\s+guaranteeing\s+.*$',
            r'\s+could\s+be\s+.*$',           # "could be appropriate", "could be suitable"
            r'\s+would\s+be\s+.*$',          # "would be appropriate"
            r'\s+seems\s+.*$',               # "seems appropriate"
            r'\s+appears\s+.*$',             # "appears suitable"
            r'\s+might\s+be\s+.*$',          # "might be appropriate"
            r'\s+should\s+be\s+.*$',         # "should be suitable"
            r'\s+is\s+appropriate\s*$',      # "is appropriate"
            r'\s+is\s+suitable\s*$',         # "is suitable"
            r'\s+works\s+well\s*$',          # "works well"
            r'\s+can\s+be\s+used\s*$'        # "can be used"
        ]
        for pattern in descriptive_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
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
        
        # Check for single letter constants (excluding x, y, z which are state variables)
        # Pattern: standalone letters a-h, j-w (excluding x, y, z, i for imaginary)
        single_letter_pattern = r'\b[a-hjkl-w]\b'
        if re.search(single_letter_pattern, expression, re.IGNORECASE):
            return True
        
        # Check for uppercase constants like C, K, A, B
        uppercase_constants = r'\b[A-HJ-W]\b'  # Exclude I (imaginary), X, Y, Z (state vars)
        if re.search(uppercase_constants, expression):
            return True
        
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
        try:
            logger.info(f"Generating certificate with model: {model_key}, RAG k: {rag_k}")
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
                    context = retrieve_context(
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
            
            # Format prompt with domain bounds
            prompt = format_prompt_with_context(
                system_description,
                context,
                model_info['config']['barrier_type'],
                domain_bounds
            )
            
            # Generate certificate
            result = model_info['pipeline'](prompt)
            generated_text = result[0]['generated_text']
            
            # Extract only the generated part after the prompt
            prompt_end_marker = "[/INST]"
            output_start_index = generated_text.find(prompt_end_marker)
            if output_start_index != -1:
                llm_output = generated_text[output_start_index + len(prompt_end_marker):].strip()
            else:
                llm_output = generated_text
            
            # Extract certificate
            certificate = self.extract_certificate_from_output(llm_output)
            
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