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
                
                # Create pipeline
                pipe = pipeline(
                    task="text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=self.config.inference.max_new_tokens,
                    temperature=self.config.inference.temperature,
                    top_p=self.config.inference.top_p,
                    do_sample=True if self.config.inference.temperature > 0 else False
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
        """Extract barrier certificate from LLM output."""
        # Look for the certificate between the markers
        pattern = r'BARRIER_CERTIFICATE_START\s*\n?(.*?)\n?\s*BARRIER_CERTIFICATE_END'
        match = re.search(pattern, llm_output, re.DOTALL | re.IGNORECASE)
        
        if match:
            certificate_block = match.group(1).strip()
            
            # Extract the mathematical expression from B(...) = ...
            b_pattern = r'B\s*\([^)]+\)\s*=\s*(.+)'
            b_match = re.search(b_pattern, certificate_block, re.IGNORECASE)
            
            if b_match:
                expression = b_match.group(1).strip()
                cleaned_expr = self._clean_certificate_expression(expression)
                
                # Check if this is a template/placeholder expression
                if self._is_template_expression(cleaned_expr):
                    logger.warning(f"Detected template expression, rejecting: {cleaned_expr}")
                    return None
                
                return cleaned_expr
        
        # Fallback: look for B(...) = ... pattern anywhere in the output
        fallback_pattern = r'B\s*\([^)]+\)\s*=\s*([^\n]+)'
        fallback_match = re.search(fallback_pattern, llm_output, re.IGNORECASE)
        
        if fallback_match:
            expression = fallback_match.group(1).strip()
            cleaned_expr = self._clean_certificate_expression(expression)
            
            # Check if this is a template/placeholder expression
            if self._is_template_expression(cleaned_expr):
                logger.warning(f"Detected template expression in fallback, rejecting: {cleaned_expr}")
                return None
            
            return cleaned_expr
        
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
    
    def _is_template_expression(self, expression: str) -> bool:
        """Check if the expression is a template/placeholder rather than a concrete barrier certificate."""
        if not expression:
            return True
        
        # Common template patterns that should be rejected
        template_patterns = [
            # Single letter variables: a, b, c, d, e, f, etc.
            r'\b[a-z]\*',  # a*, b*, c*, etc.
            r'\b[a-z]\s*\*',  # a *, b *, etc. 
            r'\b[a-z]x',  # ax, bx, cx, etc.
            r'\b[a-z]y',  # ay, by, cy, etc.
            r'\b[a-z]\*\*',  # a**, b**, etc.
            
            # Common placeholder naming
            r'\bc[0-9]',  # c1, c2, c3, etc.
            r'\bcoeff',   # coeff, coefficient
            r'\bparam',   # param, parameter
            r'\balpha\b', r'\bbeta\b', r'\bgamma\b',  # Greek letters as placeholders
            
            # The exact problematic pattern
            r'[a-z]\*\*2.*[a-z]y.*[a-z]\*\*2.*[a-z]\*.*[a-z]y.*[a-z]',  # ax**2 + bxy + cy**2 + dx + ey + f pattern
            
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
        single_letters = re.findall(r'\b[a-z]\b', expression.lower())
        non_state_letters = [letter for letter in single_letters if letter not in ['x', 'y', 'z', 'k', 't']]
        
        if len(set(non_state_letters)) >= 3:  # 3 or more different single-letter non-state variables
            logger.debug(f"Too many single-letter variables detected: {set(non_state_letters)} in '{expression}'")
            return True
        
        # Check for the specific problematic pattern from the failed result
        if 'ax**2' in expression.lower() and 'bxy' in expression.lower() and 'cy**2' in expression.lower():
            logger.debug(f"Detected specific template pattern 'ax**2 + bxy + cy**2...' in '{expression}'")
            return True
        
        return False
    
    def generate_certificate(self, system_description: str, model_key: str, rag_k: int = 3) -> Dict[str, Any]:
        """Generate a barrier certificate for the given system description."""
        try:
            logger.info(f"Generating certificate with model: {model_key}, RAG k: {rag_k}")
            
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
            
            # Format prompt
            prompt = format_prompt_with_context(
                system_description,
                context,
                model_info['config']['barrier_type']
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
                'prompt_length': len(prompt)
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