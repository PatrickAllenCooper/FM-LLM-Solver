import json
import logging
import os
import re
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

import requests

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from datetime import datetime

from flask_login import current_user
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Import from the core services architecture
try:
    from fm_llm_solver.services.certificate_generator import CertificateGenerator as CoreCertificateGenerator
    from fm_llm_solver.core.types import SystemDescription, GenerationResult
    from fm_llm_solver.core.config import Config
    from fm_llm_solver.services.model_provider import QwenProvider
    CORE_SERVICES_AVAILABLE = True
except ImportError:
    CORE_SERVICES_AVAILABLE = False

# Import generic model service
try:
    from fm_llm_solver.services.generic_models import GenericModelService
    GENERIC_MODELS_AVAILABLE = True
except ImportError:
    GENERIC_MODELS_AVAILABLE = False
    logger.warning("Generic model service not available - Claude and ChatGPT integration disabled")

# Import enhanced symbolic parser
try:
    from fm_llm_solver.services.symbolic_parser import SymbolicCertificateParser
    SYMBOLIC_PARSER_AVAILABLE = True
except ImportError:
    SYMBOLIC_PARSER_AVAILABLE = False
    logger.warning("Enhanced symbolic parser not available - using basic extraction")

# Fallback to original implementation
from inference.generate_certificate import (
    format_prompt_with_context,
    load_finetuned_model,
    load_knowledge_base,
    retrieve_context,
)
from knowledge_base.kb_utils import (
    determine_kb_type_from_config,
    get_active_kb_paths,
    validate_kb_config,
)
from web_interface.models import QueryLog, UserActivity, db

logger = logging.getLogger(__name__)


class CertificateGenerator:
    """
    Web interface certificate generator with improved architecture.
    
    This class serves as a bridge between the web interface and the core services,
    providing backward compatibility while leveraging cleaner architecture when available.
    """

    def __init__(self, config):
        """Initialize the certificate generator with configuration."""
        self.config = config
        # Default to local mode if deployment section doesn't exist
        self.deployment_mode = getattr(config, "deployment", {}).get("mode", "local")
        self.inference_api_url = getattr(config, "deployment", {}).get(
            "inference_api_url", "http://localhost:8000"
        )

        # Try to use core services architecture first
        if CORE_SERVICES_AVAILABLE:
            try:
                self.core_generator = CoreCertificateGenerator(config)
                logger.info("Core certificate generator initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize core generator: {e}")
                self.core_generator = None
        else:
            self.core_generator = None

        # Initialize generic model service if available
        if GENERIC_MODELS_AVAILABLE:
            try:
                self.generic_model_service = GenericModelService(config)
                logger.info("Generic model service initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize generic model service: {e}")
                self.generic_model_service = None
        else:
            self.generic_model_service = None

        # Initialize enhanced symbolic parser if available
        if SYMBOLIC_PARSER_AVAILABLE:
            try:
                self.symbolic_parser = SymbolicCertificateParser()
                logger.info("Enhanced symbolic parser initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize symbolic parser: {e}")
                self.symbolic_parser = None
        else:
            self.symbolic_parser = None

        # Initialize local inference components (fallback)
        self.models = {}
        self.embedding_model = None
        self.knowledge_base = None
        self.knowledge_base_service = None

        # Load knowledge base lazily
        self._kb_loaded = False
        self._kb_lock = None

    def generate_certificate_with_user_tracking(
        self,
        system_description,
        model_config=None,
        conversation_id=None,
        system_name=None,
        user_tags=None,
        domain_bounds=None,
    ):
        """Enhanced certificate generation with comprehensive user tracking."""
        self.start_time = time.time()

        # Initialize user context
        if current_user.is_authenticated:
            self.user_context = {
                "user_id": current_user.id,
                "username": current_user.username,
                "subscription_type": current_user.subscription_type,
                "session_id": current_user.get("session_id"),
                "ip_address": self._get_client_ip(),
                "user_agent": self._get_user_agent(),
            }

            # Check rate limits
            if not current_user.check_rate_limit():
                raise Exception(
                    f"Rate limit exceeded. Daily limit: {current_user.daily_request_limit}"
                )

            # Increment request count
            current_user.increment_request_count()
            db.session.commit()

        # Create enhanced query log entry
        query_log = self._create_enhanced_query_log(
            system_description,
            model_config,
            conversation_id,
            system_name,
            user_tags,
            domain_bounds,
        )

        try:
            # Log activity start
            self._log_user_activity(
                "certificate_generation_started",
                {
                    "system_name": system_name,
                    "model_config": (
                        model_config.get("name") if model_config else "default"
                    ),
                    "conversation_id": conversation_id,
                    "has_domain_bounds": bool(domain_bounds),
                    "system_complexity": self._estimate_system_complexity(
                        system_description
                    ),
                },
            )

            # Generate certificate using existing method
            result = self.generate_certificate(
                system_description, model_config or {}, conversation_id or ""
            )

            # Process and enhance the result
            enhanced_result = self._enhance_result_with_metadata(result, query_log)

            # Update query log with results
            self._update_query_log_with_results(query_log, enhanced_result)

            # Log successful generation
            self._log_certificate_generation_success(query_log, enhanced_result)

            # Update user statistics
            if current_user.is_authenticated:
                current_user.increment_certificate_count()
                db.session.commit()

            return enhanced_result

        except Exception as e:
            # Log failure
            self._log_certificate_generation_failure(query_log, str(e))

            # Update query log with error
            query_log.status = "failed"
            query_log.error_message = str(e)
            query_log.processing_end = datetime.utcnow()
            db.session.commit()

            raise e

    def _create_enhanced_query_log(
        self,
        system_description,
        model_config,
        conversation_id,
        system_name,
        user_tags,
        domain_bounds,
    ):
        """Create comprehensive query log with user tracking."""

        # Detect system properties
        system_vars = self._extract_system_variables(system_description)
        system_type = self._detect_system_type(system_description)
        system_dimension = len(system_vars) if system_vars else None

        # Prepare model configuration
        model_name = model_config.get("name", "default") if model_config else "default"
        rag_k = model_config.get("rag_k", 0) if model_config else 0
        temperature = model_config.get("temperature", 0.7) if model_config else 0.7
        max_tokens = model_config.get("max_tokens", 512) if model_config else 512

        query_log = QueryLog(
            user_id=current_user.id if current_user.is_authenticated else None,
            # System details
            system_description=system_description,
            system_name=system_name,
            system_type=system_type,
            system_dimension=system_dimension,
            variables=system_vars,
            # Model configuration
            model_config=model_config or {},
            model_name=model_name,
            rag_k=rag_k,
            temperature=temperature,
            max_tokens=max_tokens,
            # Context tracking
            conversation_id=conversation_id,
            session_id=self.user_context.get("session_id"),
            ip_address=self.user_context.get("ip_address"),
            user_agent=self.user_context.get("user_agent"),
            # User interaction
            tags=user_tags or [],
            # Domain bounds
            certificate_domain_bounds=(
                json.dumps(domain_bounds) if domain_bounds else None
            ),
            domain_description=(
                self._generate_domain_description(domain_bounds)
                if domain_bounds
                else None
            ),
            # Status and timing
            status="pending",
            processing_start=datetime.utcnow(),
            timestamp=datetime.utcnow(),
        )

        db.session.add(query_log)
        db.session.commit()
        return query_log

    def _enhance_result_with_metadata(self, result, query_log):
        """Add metadata and quality metrics to the generation result."""
        processing_time = time.time() - self.start_time

        enhanced_result = (
            result.copy()
            if isinstance(result, dict)
            else {"generated_certificate": result, "status": "completed"}
        )

        # Add performance metrics
        enhanced_result.update(
            {
                "processing_time_seconds": processing_time,
                "query_id": query_log.id,
                "timestamp": datetime.utcnow().isoformat(),
                "user_context": (
                    self.user_context if current_user.is_authenticated else None
                ),
                # Quality metrics (to be computed)
                "certificate_complexity": None,
                "confidence_score": None,
                "mathematical_soundness": None,
            }
        )

        # Analyze certificate if generated successfully
        if enhanced_result.get("generated_certificate"):
            cert_analysis = self._analyze_certificate(
                enhanced_result["generated_certificate"]
            )
            enhanced_result.update(cert_analysis)

        return enhanced_result

    def _update_query_log_with_results(self, query_log, enhanced_result):
        """Update the query log with generation results."""
        query_log.generated_certificate = enhanced_result.get("generated_certificate")
        query_log.status = enhanced_result.get("status", "completed")
        query_log.processing_end = datetime.utcnow()
        query_log.processing_time = enhanced_result.get("processing_time_seconds")

        # Certificate analysis results
        query_log.certificate_format = enhanced_result.get("certificate_format")
        query_log.certificate_complexity = enhanced_result.get("certificate_complexity")
        query_log.extraction_method = enhanced_result.get("extraction_method")
        query_log.confidence_score = enhanced_result.get("confidence_score")
        query_log.mathematical_soundness = enhanced_result.get("mathematical_soundness")

        # Token usage and cost estimation
        query_log.total_tokens_used = enhanced_result.get("total_tokens_used")
        query_log.cost_estimate = enhanced_result.get("cost_estimate")

        db.session.commit()

    def _log_certificate_generation_success(self, query_log, result):
        """Log successful certificate generation activity."""
        self._log_user_activity(
            "certificate_generated",
            {
                "query_id": query_log.id,
                "system_name": query_log.system_name,
                "model_name": query_log.model_name,
                "processing_time": result.get("processing_time_seconds"),
                "certificate_format": result.get("certificate_format"),
                "certificate_complexity": result.get("certificate_complexity"),
                "tokens_used": result.get("total_tokens_used"),
                "cost_estimate": result.get("cost_estimate"),
                "has_certificate": bool(result.get("generated_certificate")),
            },
            success=True,
            response_time_ms=int(result.get("processing_time_seconds", 0) * 1000),
        )

    def _log_certificate_generation_failure(self, query_log, error_message):
        """Log failed certificate generation activity."""
        processing_time = time.time() - self.start_time if self.start_time else 0

        self._log_user_activity(
            "certificate_generation_failed",
            {
                "query_id": query_log.id,
                "system_name": query_log.system_name,
                "model_name": query_log.model_name,
                "error_message": error_message,
                "processing_time": processing_time,
            },
            success=False,
            response_time_ms=int(processing_time * 1000),
        )

    def _log_user_activity(
        self, activity_type, details=None, success=True, response_time_ms=None
    ):
        """Log user activity if user is authenticated."""
        if current_user.is_authenticated:
            try:
                activity = UserActivity(
                    user_id=current_user.id,
                    activity_type=activity_type,
                    activity_details=details or {},
                    ip_address=self.user_context.get("ip_address"),
                    user_agent=self.user_context.get("user_agent"),
                    session_id=self.user_context.get("session_id"),
                    response_time_ms=response_time_ms,
                    success=success,
                )
                db.session.add(activity)
                db.session.commit()
            except Exception:
                # Don't let activity logging break the main functionality
                db.session.rollback()

    def _extract_system_variables(self, system_description):
        """Extract variable names from system description."""
        import re

        # Common patterns for variable extraction
        patterns = [
            r"([a-zA-Z]\w*)\s*\'",  # x', y', etc.
            r"d([a-zA-Z]\w*)/dt",  # dx/dt, dy/dt, etc.
            r"([a-zA-Z]\w*)\s*\(",  # x(t), y(t), etc.
            r"([a-zA-Z]\w*)\s*=",  # x =, y =, etc.
        ]

        variables = set()
        for pattern in patterns:
            matches = re.findall(pattern, system_description.lower())
            variables.update(matches)

        # Filter out common non-variable words
        exclude_words = {
            "dt",
            "dx",
            "dy",
            "dz",
            "sin",
            "cos",
            "exp",
            "log",
            "sqrt",
            "abs",
            "max",
            "min",
        }
        variables = [
            var for var in variables if var not in exclude_words and len(var) <= 5
        ]

        return list(variables)

    def _detect_system_type(self, system_description):
        """Detect if system is continuous, discrete, or stochastic."""
        description_lower = system_description.lower()

        if any(
            keyword in description_lower
            for keyword in ["dt", "derivative", "differential", "continuous"]
        ):
            return "continuous"
        elif any(
            keyword in description_lower
            for keyword in ["k+1", "next", "discrete", "iteration"]
        ):
            return "discrete"
        elif any(
            keyword in description_lower
            for keyword in ["noise", "stochastic", "random", "probability"]
        ):
            return "stochastic"
        else:
            return "unknown"

    def _estimate_system_complexity(self, system_description):
        """Estimate system complexity based on description."""
        # Simple heuristic based on length, math operations, etc.
        base_score = len(system_description) // 20

        math_operations = len(re.findall(r"[+\-*/^]", system_description))
        functions = len(
            re.findall(r"(sin|cos|exp|log|sqrt|abs)", system_description.lower())
        )

        complexity = base_score + math_operations + functions * 2
        return min(complexity, 100)  # Cap at 100

    def _analyze_certificate(self, certificate):
        """Analyze the generated certificate for quality metrics."""
        if not certificate:
            return {
                "certificate_format": None,
                "certificate_complexity": 0,
                "extraction_method": "none",
                "confidence_score": 0.0,
                "mathematical_soundness": 0.0,
                "symbolic_validation": {}
            }

        # Use enhanced symbolic parser if available and we have stored results
        if self.symbolic_parser and hasattr(self, 'last_parse_result') and self.last_parse_result:
            parse_result = self.last_parse_result
            
            # Extract detailed analysis from parser
            analysis = {
                "certificate_format": parse_result.format_type,
                "certificate_complexity": parse_result.complexity_score,
                "extraction_method": "enhanced_symbolic_parser",
                "confidence_score": parse_result.confidence,
                "mathematical_soundness": self._calculate_mathematical_soundness(parse_result),
                "symbolic_validation": {}
            }
            
            # Add symbolic validation if available
            if parse_result.symbolic_expression:
                try:
                    validation = self.symbolic_parser.validate_barrier_certificate(
                        certificate, parse_result.variables
                    )
                    analysis["symbolic_validation"] = validation
                    
                    # Update mathematical soundness based on validation
                    if validation.get('valid', False):
                        analysis["mathematical_soundness"] = min(
                            analysis["mathematical_soundness"] + 0.2, 1.0
                        )
                        
                except Exception as e:
                    logger.warning(f"Symbolic validation failed: {e}")
                    analysis["symbolic_validation"] = {"error": str(e)}
            
            # Add parser warnings and errors as metadata
            if parse_result.warnings:
                analysis["parser_warnings"] = parse_result.warnings
            if parse_result.errors:
                analysis["parser_errors"] = parse_result.errors
                
            return analysis
        
        # Fallback to basic analysis
        return self._analyze_certificate_basic(certificate)
    
    def _analyze_certificate_basic(self, certificate):
        """Basic certificate analysis (fallback method)."""
        cert_lower = certificate.lower()

        # Detect certificate format
        if any(op in cert_lower for op in ["^2", "**2", "x*x", "y*y"]):
            cert_format = "polynomial"
        elif any(func in cert_lower for func in ["sin", "cos", "tan"]):
            cert_format = "trigonometric"
        elif "/" in cert_lower:
            cert_format = "rational"
        else:
            cert_format = "linear"

        # Estimate complexity
        complexity = len(certificate) // 10
        complexity += len(re.findall(r"[+\-*/^]", certificate))
        complexity += len(re.findall(r"(sin|cos|exp|log|sqrt)", cert_lower)) * 2

        return {
            "certificate_format": cert_format,
            "certificate_complexity": min(complexity, 100),
            "extraction_method": "regex_pattern",
            "confidence_score": 0.5,  # Default confidence for basic analysis
            "mathematical_soundness": 0.5,  # Default soundness
            "symbolic_validation": {}
        }
    
    def _calculate_mathematical_soundness(self, parse_result) -> float:
        """Calculate mathematical soundness score based on parser results."""
        soundness = 0.3  # Base score
        
        # Successful symbolic parsing bonus
        if parse_result.symbolic_expression:
            soundness += 0.3
        
        # Format appropriateness for barrier certificates
        if parse_result.format_type in ['polynomial_quadratic', 'polynomial_multivariate']:
            soundness += 0.2
        elif parse_result.format_type in ['polynomial', 'linear']:
            soundness += 0.1
        
        # Confidence bonus
        soundness += parse_result.confidence * 0.2
        
        # Complexity penalty (very complex expressions may be less reliable)
        if parse_result.complexity_score > 75:
            soundness -= 0.1
        
        return min(max(soundness, 0.0), 1.0)  # Clamp to [0, 1]

    def _generate_domain_description(self, domain_bounds):
        """Generate human-readable domain description."""
        if not domain_bounds:
            return None

        descriptions = []
        for var, bounds in domain_bounds.items():
            if isinstance(bounds, list) and len(bounds) == 2:
                descriptions.append(f"{var} ∈ [{bounds[0]}, {bounds[1]}]")

        return ", ".join(descriptions)

    def _get_client_ip(self):
        """Get client IP address from request."""
        try:
            from flask import request

            # Check for forwarded headers (when behind proxy)
            if request.headers.get("X-Forwarded-For"):
                return request.headers.get("X-Forwarded-For").split(",")[0].strip()
            elif request.headers.get("X-Real-IP"):
                return request.headers.get("X-Real-IP")
            else:
                return request.remote_addr
        except:
            return "unknown"

    def _get_user_agent(self):
        """Get user agent from request."""
        try:
            from flask import request

            return request.headers.get("User-Agent", "")[:500]  # Limit length
        except:
            return "unknown"

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available model configurations."""
        if self.deployment_mode != "local":
            # Query inference API for available models
            try:
                response = requests.get(f"{self.inference_api_url}/models", timeout=10)
                if response.status_code == 200:
                    api_models = response.json().get("models", [])
                    return [
                        {
                            "key": model["key"],
                            "name": f"{model['key'].title()} Model",
                            "description": f"Model configuration: {model['key']}",
                            "type": "finetuned" if model.get("use_adapter") else "base",
                            "barrier_type": self.config.knowledge_base.barrier_certificate_type,
                        }
                        for model in api_models
                    ]
            except Exception as e:
                logger.error(f"Failed to query inference API for models: {e}")
                # Fall back to default models

        # Local mode or fallback
        models = []

        # Base model configuration
        models.append(
            {
                "key": "base",
                "name": "Base Model",
                "description": f"Base model without fine-tuning: {self.config.fine_tuning.base_model_name}",
                "type": "base",
                "barrier_type": self.config.knowledge_base.barrier_certificate_type,
            }
        )

        # Fine-tuned model configuration
        adapter_path = os.path.join(self.config.paths.ft_output_dir, "final_adapter")
        if os.path.exists(adapter_path) or self.deployment_mode != "local":
            models.append(
                {
                    "key": "finetuned",
                    "name": "Fine-tuned Model",
                    "description": f"Fine-tuned model with adapter: {self.config.fine_tuning.base_model_name}",
                    "type": "finetuned",
                    "barrier_type": self.config.knowledge_base.barrier_certificate_type,
                }
            )

        # Add discrete/continuous specific models if available
        if self.config.knowledge_base.barrier_certificate_type == "unified":
            # Check for discrete-specific models
            discrete_adapter = os.path.join(
                self.config.paths.ft_output_dir, "discrete_adapter"
            )
            if os.path.exists(discrete_adapter) or self.deployment_mode != "local":
                models.append(
                    {
                        "key": "discrete",
                        "name": "Discrete Fine-tuned Model",
                        "description": "Model specifically fine-tuned for discrete barrier certificates",
                        "type": "finetuned",
                        "barrier_type": "discrete",
                    }
                )

            # Check for continuous-specific models
            continuous_adapter = os.path.join(
                self.config.paths.ft_output_dir, "continuous_adapter"
            )
            if os.path.exists(continuous_adapter) or self.deployment_mode != "local":
                models.append(
                    {
                        "key": "continuous",
                        "name": "Continuous Fine-tuned Model",
                        "description": "Model specifically fine-tuned for continuous barrier certificates",
                        "type": "finetuned",
                        "barrier_type": "continuous",
                    }
                )

        # Add Claude and ChatGPT models if available
        if GENERIC_MODELS_AVAILABLE:
            try:
                # Use the generic_model_service to get available models
                available_models = self.generic_model_service.get_available_models()
                for model_info in available_models:
                    models.append(
                        {
                            "key": model_info["key"],
                            "name": model_info["name"],
                            "description": model_info["description"],
                            "type": "generic",
                            "barrier_type": "unified", # Generic models are unified
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to load generic models: {e}")

        return models

    def _get_model_config(self, model_key: str) -> Dict[str, Any]:
        """Get model configuration for a specific model key."""
        if model_key == "base":
            return {
                "use_adapter": False,
                "base_model_name": self.config.fine_tuning.base_model_name,
                "adapter_path": None,
                "barrier_type": self.config.knowledge_base.barrier_certificate_type,
            }
        elif model_key == "finetuned":
            return {
                "use_adapter": True,
                "base_model_name": self.config.fine_tuning.base_model_name,
                "adapter_path": os.path.join(
                    self.config.paths.ft_output_dir, "final_adapter"
                ),
                "barrier_type": self.config.knowledge_base.barrier_certificate_type,
            }
        elif model_key == "discrete":
            return {
                "use_adapter": True,
                "base_model_name": self.config.fine_tuning.base_model_name,
                "adapter_path": os.path.join(
                    self.config.paths.ft_output_dir, "discrete_adapter"
                ),
                "barrier_type": "discrete",
            }
        elif model_key == "continuous":
            return {
                "use_adapter": True,
                "base_model_name": self.config.fine_tuning.base_model_name,
                "adapter_path": os.path.join(
                    self.config.paths.ft_output_dir, "continuous_adapter"
                ),
                "barrier_type": "continuous",
            }
        else:
            raise ValueError(f"Unknown model key: {model_key}")

    def _load_embedding_model(self):
        """Load the embedding model for RAG."""
        if self.deployment_mode != "local":
            raise RuntimeError(
                "_load_embedding_model should only be called in local mode"
            )

        if self.embedding_model is None:
            try:
                self.embedding_model = self.SentenceTransformer(
                    self.config.knowledge_base.embedding_model_name
                )
                logger.info(
                    f"Loaded embedding model: {self.config.knowledge_base.embedding_model_name}"
                )
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        return self.embedding_model

    def _load_knowledge_base(self, barrier_type: str):
        """Load knowledge base for the specified barrier certificate type."""
        if self.deployment_mode != "local":
            raise RuntimeError(
                "_load_knowledge_base should only be called in local mode"
            )

        if barrier_type not in self.knowledge_bases:
            try:
                # Get KB paths based on barrier type
                if barrier_type == "discrete":
                    kb_dir = self.config.paths.kb_discrete_output_dir
                    vector_file = self.config.paths.kb_discrete_vector_store_filename
                    metadata_file = self.config.paths.kb_discrete_metadata_filename
                elif barrier_type == "continuous":
                    kb_dir = self.config.paths.kb_continuous_output_dir
                    vector_file = self.config.paths.kb_continuous_vector_store_filename
                    metadata_file = self.config.paths.kb_continuous_metadata_filename
                else:  # unified or default
                    kb_dir = self.config.paths.kb_output_dir
                    vector_file = self.config.paths.kb_vector_store_filename
                    metadata_file = self.config.paths.kb_metadata_filename

                index, metadata = self.load_knowledge_base(
                    kb_dir, vector_file, metadata_file
                )
                if index is None or metadata is None:
                    logger.warning(
                        f"No knowledge base found for {barrier_type}. RAG will be disabled."
                    )
                    self.knowledge_bases[barrier_type] = None
                    return None

                self.knowledge_bases[barrier_type] = {
                    "index": index,
                    "metadata": metadata,
                }
                logger.info(
                    f"Loaded {barrier_type} knowledge base with {index.ntotal} vectors"
                )

            except Exception as e:
                logger.warning(
                    f"Failed to load {barrier_type} knowledge base: {e}. RAG will be disabled."
                )
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
                temp_config.fine_tuning.use_adapter = model_config["use_adapter"]

                model, tokenizer = self.load_finetuned_model(
                    model_config["base_model_name"],
                    model_config["adapter_path"],
                    temp_config,
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
                    do_sample=self.config.inference.get("do_sample", True),
                    repetition_penalty=self.config.inference.get(
                        "repetition_penalty", 1.1
                    ),
                    # Add stop sequences to prevent incomplete generation
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                self.models[model_key] = {"pipeline": pipe, "config": model_config}
                logger.info(f"Loaded model: {model_key}")

            except Exception as e:
                logger.error(f"Failed to load model {model_key}: {e}")
                raise

        return self.models[model_key]

    def extract_certificate_from_output(self, llm_output: str) -> Optional[str]:
        """Extract barrier certificate from LLM output using shared utilities."""
        from utils.certificate_extraction import (
            clean_certificate_expression,
            contains_state_variables,
            extract_certificate_from_llm_output,
            has_placeholder_variables,
            is_template_expression,
        )

        # Use shared extraction function
        variables = ["x", "y", "z"]  # Default state variables
        extracted_expr, extraction_failed = extract_certificate_from_llm_output(
            llm_output, variables
        )

        if extraction_failed or extracted_expr is None:
            logger.warning("Failed to extract valid certificate from LLM output")
            return None

        # Clean the extracted expression
        cleaned_expr = clean_certificate_expression(extracted_expr)

        # Enhanced placeholder detection
        if has_placeholder_variables(cleaned_expr):
            logger.warning(f"REJECTED: Contains placeholder variables: {cleaned_expr}")
            return None

        if is_template_expression(cleaned_expr):
            logger.warning(f"REJECTED: Template expression: {cleaned_expr}")
            return None

        # Validate it contains actual variables
        if contains_state_variables(cleaned_expr):
            logger.info(f"Successfully extracted certificate: {cleaned_expr}")
            return cleaned_expr

        logger.warning("Extracted certificate does not contain state variables")
        return None

    # Certificate cleaning and validation methods moved to utils.certificate_extraction

    def generate_certificate(
        self,
        system_description: str,
        model_key: str,
        rag_k: int = 3,
        domain_bounds: dict = None,
    ) -> Dict[str, Any]:
        """
        Generate a barrier certificate for the given system description.
        
        This method intelligently routes to either core services, generic models,
        or legacy implementation based on availability and configuration.
        """
        try:
            # Check if this is a generic model (Claude, ChatGPT, etc.)
            if self._is_generic_model(model_key):
                return self._generate_with_generic_model(
                    system_description, model_key, rag_k, domain_bounds
                )
            
            # Use core services if available for fine-tuned models
            elif self.core_generator:
                return self._generate_with_core_services(
                    system_description, model_key, rag_k, domain_bounds
                )
            
            # Fallback to legacy implementation
            elif self.deployment_mode == "local":
                return self._generate_local(
                    system_description, model_key, rag_k, domain_bounds
                )
            else:
                return self._generate_remote(
                    system_description, model_key, rag_k, domain_bounds
                )
                
        except Exception as e:
            logger.error(f"Certificate generation failed: {e}")
            return {
                "success": False,
                "error": f"Failed to generate certificate: {str(e)}",
                "certificate": None,
                "llm_output": "",
                "context_chunks": 0
            }
    
    def _is_generic_model(self, model_key: str) -> bool:
        """Check if the model key corresponds to a generic model."""
        generic_models = ['claude-3-sonnet', 'claude-3-haiku', 'gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo']
        return model_key in generic_models
    
    def _generate_with_generic_model(
        self,
        system_description: str,
        model_key: str,
        rag_k: int = 3,
        domain_bounds: dict = None,
    ) -> Dict[str, Any]:
        """Generate certificate using generic models (Claude, ChatGPT)."""
        if not self.generic_model_service:
            return {
                "success": False,
                "error": "Generic model service not available",
                "certificate": None,
                "llm_output": "",
                "context_chunks": 0
            }
        
        try:
            start_time = time.time()
            logger.info(f"Using generic model for generation: model={model_key}, rag_k={rag_k}")
            
            # Get RAG context if requested
            rag_context = None
            context_chunks = 0
            if rag_k > 0:
                try:
                    # Load knowledge base for context retrieval
                    kb, kb_metadata = self._load_knowledge_base("unified")  # Generic models use unified KB
                    if kb and kb_metadata:
                        context_chunks_data = retrieve_context(
                            system_description, kb, kb_metadata, k=rag_k
                        )
                        if context_chunks_data:
                            rag_context = "\n\n".join(context_chunks_data)
                            context_chunks = len(context_chunks_data)
                            logger.info(f"Retrieved {context_chunks} context chunks for RAG")
                except Exception as e:
                    logger.warning(f"Failed to retrieve RAG context: {e}")
                    rag_context = None
                    context_chunks = 0
            
            # Generate certificate using generic model
            response = self.generic_model_service.generate_certificate(
                system_description=system_description,
                model_name=model_key,
                rag_context=rag_context,
                temperature=0.1,
                max_tokens=1024
            )
            
            processing_time = time.time() - start_time
            
            if response.success:
                # Extract certificate from response
                certificate = self._extract_certificate_from_text(response.content)
                
                return {
                    "success": True,
                    "certificate": certificate,
                    "llm_output": response.content,
                    "context_chunks": context_chunks,
                    "processing_time_seconds": processing_time,
                    "model_name": model_key,
                    "model_type": "generic",
                    "total_tokens_used": response.tokens_used,
                    "cost_estimate": response.cost_estimate,
                    "provider": response.metadata.get("provider") if response.metadata else None,
                    "confidence_score": self._estimate_confidence(response.content),
                    "extraction_method": "generic_text_extraction"
                }
            else:
                return {
                    "success": False,
                    "error": response.error,
                    "certificate": None,
                    "llm_output": "",
                    "context_chunks": context_chunks,
                    "processing_time_seconds": processing_time,
                    "model_name": model_key,
                    "model_type": "generic"
                }
        
        except Exception as e:
            logger.error(f"Generic model generation failed: {e}")
            return {
                "success": False,
                "error": f"Generic model generation failed: {str(e)}",
                "certificate": None,
                "llm_output": "",
                "context_chunks": 0,
                "model_name": model_key,
                "model_type": "generic"
            }
    
    def _extract_certificate_from_text(self, text: str) -> str:
        """Extract barrier certificate function from generic model response."""
        if not text:
            return ""
        
        # Use enhanced symbolic parser if available
        if self.symbolic_parser:
            try:
                parse_result = self.symbolic_parser.parse_certificate(text)
                if parse_result.success and parse_result.expression:
                    logger.info(f"Enhanced parser extracted: {parse_result.expression}")
                    logger.info(f"Format type: {parse_result.format_type}, Confidence: {parse_result.confidence:.2f}")
                    
                    # Store additional metadata for later use
                    if hasattr(self, 'last_parse_result'):
                        self.last_parse_result = parse_result
                    
                    return parse_result.expression
                else:
                    logger.warning(f"Enhanced parser failed: {parse_result.errors}")
                    # Fall back to basic extraction
            except Exception as e:
                logger.warning(f"Enhanced parser error: {e}, falling back to basic extraction")
        
        # Fallback to basic extraction methods
        return self._extract_certificate_basic(text)
    
    def _extract_certificate_basic(self, text: str) -> str:
        """Basic certificate extraction (fallback method)."""
        # Common patterns for barrier functions
        patterns = [
            r"B\(x\)\s*=\s*([^.\n]+)",  # B(x) = ...
            r"B\([^)]+\)\s*=\s*([^.\n]+)",  # B(x,y) = ...
            r"barrier function:\s*([^.\n]+)",  # barrier function: ...
            r"barrier certificate:\s*([^.\n]+)",  # barrier certificate: ...
            r"B\s*=\s*([^.\n]+)",  # B = ...
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                certificate = match.group(1).strip()
                # Clean up common artifacts
                certificate = re.sub(r'["\']', '', certificate)  # Remove quotes
                certificate = certificate.split('\n')[0]  # Take first line only
                if len(certificate) > 5:  # Reasonable minimum length
                    return certificate
        
        # Fallback: look for mathematical expressions
        math_pattern = r'([x-z]+[0-9]*[\s\+\-\*\/\^\(\)x-z0-9\s]+)'
        matches = re.findall(math_pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match.strip()) > 5 and any(var in match.lower() for var in ['x', 'y', 'z']):
                return match.strip()
        
        return ""
    
    def _estimate_confidence(self, text: str) -> float:
        """Estimate confidence in the generated certificate."""
        if not text:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Check for mathematical rigor indicators
        if any(keyword in text.lower() for keyword in ['verify', 'satisfy', 'condition', 'proof']):
            confidence += 0.2
        
        # Check for proper barrier function format
        if re.search(r'B\([x-z]+\)', text, re.IGNORECASE):
            confidence += 0.1
        
        # Check for mathematical expressions
        if re.search(r'[x-z]+[0-9]*[\+\-\*\/\^]', text):
            confidence += 0.1
        
        # Check for explanation quality
        if len(text) > 200:  # Detailed explanation
            confidence += 0.1
        
        return min(confidence, 1.0)
