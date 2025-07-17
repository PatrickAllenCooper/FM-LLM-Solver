"""
Core components for FM-LLM Solver.

This module contains fundamental components used throughout the system:
- Configuration management
- Logging infrastructure
- Exception hierarchy
- Base classes and interfaces
"""

from fm_llm_solver.core.config import Config, load_config, validate_config
from fm_llm_solver.core.environment_detector import get_environment_detector
from fm_llm_solver.core.exceptions import (
    ConfigurationError,
    FMLLMSolverError,
    GenerationError,
    KnowledgeBaseError,
    ValidationError,
    VerificationError,
)
from fm_llm_solver.core.interfaces import (
    Generator,
    KnowledgeStore,
    ModelProvider,
    Verifier,
)
from fm_llm_solver.core.logging import configure_logging, get_logger
from fm_llm_solver.core.types import (
    BarrierCertificate,
    GenerationResult,
    SystemDescription,
    VerificationResult,
)

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
    "get_environment_detector",
]
