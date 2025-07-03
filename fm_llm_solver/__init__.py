"""
FM-LLM Solver: Barrier Certificate Generation using Large Language Models.

A comprehensive system for generating and verifying barrier certificates
for dynamical systems using LLMs enhanced with RAG and fine-tuning.
"""

__version__ = "1.0.0"
__author__ = "Patrick Allen Cooper"
__email__ = "patrick.cooper@colorado.edu"
__license__ = "MIT"

# Core module exports
from fm_llm_solver.core.config import Config, load_config
from fm_llm_solver.core.logging import get_logger, configure_logging
from fm_llm_solver.core.exceptions import (
    FMLLMSolverError,
    ConfigurationError,
    GenerationError,
    VerificationError,
    KnowledgeBaseError
)

# Service exports
from fm_llm_solver.services.certificate_generator import CertificateGenerator
from fm_llm_solver.services.verifier import CertificateVerifier
from fm_llm_solver.services.knowledge_base import KnowledgeBase

__all__ = [
    "__version__",
    "Config",
    "load_config", 
    "get_logger",
    "configure_logging",
    "FMLLMSolverError",
    "ConfigurationError",
    "GenerationError",
    "VerificationError",
    "KnowledgeBaseError",
    "CertificateGenerator",
    "CertificateVerifier",
    "KnowledgeBase"
] 