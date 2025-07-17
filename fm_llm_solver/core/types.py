"""
Type definitions for FM-LLM Solver.

Provides strong typing for data structures used throughout the system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from enum import Enum


class SystemType(Enum):
    """Types of dynamical systems supported."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    STOCHASTIC = "stochastic"
    HYBRID = "hybrid"


class VerificationMethod(Enum):
    """Available verification methods."""

    NUMERICAL = "numerical"
    SYMBOLIC = "symbolic"
    SOS = "sos"
    ALL = "all"


class ModelProvider(Enum):
    """Supported model providers."""

    QWEN = "qwen"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LLAMA = "llama"
    CUSTOM = "custom"


@dataclass
class DomainBounds:
    """Domain bounds for variables."""

    bounds: Dict[str, Tuple[float, float]]

    def contains(self, point: Dict[str, float]) -> bool:
        """Check if a point is within bounds."""
        for var, (low, high) in self.bounds.items():
            if var not in point or not (low <= point[var] <= high):
                return False
        return True


@dataclass
class SystemDescription:
    """Complete description of a dynamical system."""

    dynamics: Dict[str, str]  # Variable -> expression mapping
    initial_set: str
    unsafe_set: str
    system_type: SystemType = SystemType.CONTINUOUS
    domain_bounds: Optional[DomainBounds] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        """Convert to natural language description."""
        parts = []

        # System type
        if self.system_type == SystemType.CONTINUOUS:
            dynamics_str = ", ".join(f"d{var}/dt = {expr}" for var, expr in self.dynamics.items())
        elif self.system_type == SystemType.DISCRETE:
            dynamics_str = ", ".join(f"{var}[k+1] = {expr}" for var, expr in self.dynamics.items())
        else:
            dynamics_str = ", ".join(f"{var}: {expr}" for var, expr in self.dynamics.items())

        parts.append(f"System Dynamics: {dynamics_str}")
        parts.append(f"Initial Set: {self.initial_set}")
        parts.append(f"Unsafe Set: {self.unsafe_set}")

        if self.domain_bounds:
            bounds_str = ", ".join(
                f"{var} ∈ [{low}, {high}]" for var, (low, high) in self.domain_bounds.bounds.items()
            )
            parts.append(f"Domain: {bounds_str}")

        return ". ".join(parts)


@dataclass
class BarrierCertificate:
    """Represents a barrier certificate."""

    expression: str
    variables: List[str]
    certificate_type: str = "standard"  # standard, exponential, rational
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.expression


@dataclass
class VerificationCheck:
    """Result of a single verification check."""

    check_type: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Complete verification result."""

    valid: bool
    checks: List[VerificationCheck]
    computation_time: float
    method: VerificationMethod
    certificate: Optional[BarrierCertificate] = None
    error: Optional[str] = None

    @property
    def summary(self) -> Dict[str, bool]:
        """Get summary of check results."""
        return {check.check_type: check.passed for check in self.checks}


@dataclass
class GenerationResult:
    """Result of certificate generation."""

    certificate: Optional[BarrierCertificate]
    confidence: float
    rag_context: List[Dict[str, Any]] = field(default_factory=list)
    generation_time: float = 0.0
    model_name: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if generation was successful."""
        return self.certificate is not None and self.error is None


@dataclass
class RAGDocument:
    """Document retrieved from knowledge base."""

    content: str
    metadata: Dict[str, Any]
    score: float
    source: str

    def __str__(self) -> str:
        return f"{self.source} (score: {self.score:.3f}): {self.content[:100]}..."


@dataclass
class ModelConfig:
    """Configuration for a language model."""

    provider: ModelProvider
    name: str
    temperature: float = 0.7
    max_tokens: int = 1024
    quantization: Optional[str] = None
    device: str = "cuda"
    use_flash_attention: bool = True
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_length: int = 512
    use_gradient_checkpointing: bool = True
    output_dir: str = "output/finetuning_results"
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10


@dataclass
class QueryLog:
    """Log entry for a query."""

    id: str
    timestamp: datetime
    user_id: Optional[str]
    system_description: SystemDescription
    result: Optional[GenerationResult]
    verification: Optional[VerificationResult]
    processing_time: float
    status: str  # pending, completed, failed
    error: Optional[str] = None
