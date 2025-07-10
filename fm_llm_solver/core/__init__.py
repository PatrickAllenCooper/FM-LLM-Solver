"""
Core components for FM-LLM Solver.

This module contains fundamental components used throughout the system:
- Configuration management
- Logging infrastructure
- Exception hierarchy
- Base classes and interfaces
"""

from fm_llm_solver.core.config import Config, load_config, validate_config
from fm_llm_solver.core.logging import get_logger, configure_logging
from fm_llm_solver.core.exceptions import (
    FMLLMSolverError,
    ConfigurationError,
    GenerationError,
    VerificationError,
    KnowledgeBaseError,
    ValidationError
)
from fm_llm_solver.core.interfaces import (
    Generator,
    Verifier,
    KnowledgeStore,
    ModelProvider
)
from fm_llm_solver.core.types import (
    SystemDescription,
    BarrierCertificate,
    VerificationResult,
    GenerationResult
)
from fm_llm_solver.core.environment_detector import get_environment_detector

__all__ = [
    # Config
    "Config",
    "load_config",
    "validate_config",
    # Logging
    "get_logger",
    "configure_logging",
    # Exceptions
    "FMLLMSolverError",
    "ConfigurationError",
    "GenerationError",
    "VerificationError",
    "KnowledgeBaseError",
    "ValidationError",
    # Interfaces
    "Generator",
    "Verifier", 
    "KnowledgeStore",
    "ModelProvider",
    # Types
    "SystemDescription",
    "BarrierCertificate",
    "VerificationResult",
    "GenerationResult",
    # Environment Detection
    "get_environment_detector"
] 