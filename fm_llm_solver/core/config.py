"""
Configuration management for FM-LLM Solver.

Handles loading, validation, and access to configuration settings.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import yaml
from pydantic import BaseModel, Field, field_validator, ConfigDict
import json

from fm_llm_solver.core.exceptions import ConfigurationError


class PathConfig(BaseModel):
    """Path configuration."""

    kb_output_dir: str = Field(default="kb_data", description="Knowledge base output directory")
    fetched_papers_dir: str = Field(
        default="data/fetched_papers", description="Fetched papers directory"
    )
    ft_data_dir: str = Field(default="data", description="Fine-tuning data directory")
    output_dir: str = Field(default="output", description="General output directory")
    log_dir: str = Field(default="logs", description="Log directory")
    temp_dir: str = Field(default="temp", description="Temporary files directory")

    @field_validator("*", mode="before")
    def expand_paths(cls, v):
        """Expand environment variables and user paths."""
        if isinstance(v, str):
            return os.path.expandvars(os.path.expanduser(v))
        return v


class ModelConfig(BaseModel):
    """Model configuration."""

    provider: str = Field(default="qwen", description="Model provider")
    name: str = Field(default="Qwen/Qwen2.5-14B-Instruct", description="Model name")
    use_finetuned: bool = Field(default=True, description="Use fine-tuned model if available")
    quantization: Optional[str] = Field(default=None, description="Quantization type (4bit, 8bit)")
    device: str = Field(default="cuda", description="Device to use")
    device_map: str = Field(default="auto", description="Device mapping strategy")
    torch_dtype: str = Field(default="auto", description="Torch data type")
    use_flash_attention: bool = Field(default=True, description="Use flash attention")
    use_gradient_checkpointing: bool = Field(default=True, description="Use gradient checkpointing")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Generation temperature")
    max_new_tokens: int = Field(default=1024, ge=1, description="Maximum new tokens")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")

    @field_validator("provider")
    def validate_provider(cls, v):
        """Validate model provider."""
        valid_providers = ["qwen", "openai", "anthropic", "llama", "custom"]
        if v not in valid_providers:
            raise ValueError(f"Invalid provider: {v}. Must be one of {valid_providers}")
        return v


class RAGConfig(BaseModel):
    """RAG configuration."""

    enabled: bool = Field(default=True, description="Enable RAG")
    k_retrieved: int = Field(default=3, ge=1, description="Number of documents to retrieve")
    chunk_size: int = Field(default=1000, ge=100, description="Chunk size for documents")
    chunk_overlap: int = Field(default=200, ge=0, description="Chunk overlap")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    rerank: bool = Field(default=False, description="Enable reranking")
    min_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum similarity score")


class TrainingConfig(BaseModel):
    """Training configuration."""

    num_epochs: int = Field(default=3, ge=1, description="Number of epochs")
    batch_size: int = Field(default=4, ge=1, description="Batch size")
    learning_rate: float = Field(default=2e-4, gt=0, description="Learning rate")
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    warmup_steps: int = Field(default=100, ge=0)
    max_length: int = Field(default=512, ge=1)
    save_steps: int = Field(default=500, ge=1)
    eval_steps: int = Field(default=100, ge=1)
    logging_steps: int = Field(default=10, ge=1)
    save_total_limit: int = Field(default=3, ge=1)
    fp16: bool = Field(default=True, description="Use FP16 training")
    bf16: bool = Field(default=False, description="Use BF16 training")
    gradient_checkpointing: bool = Field(default=True)
    optim: str = Field(default="adamw_torch", description="Optimizer")
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0)


class NumericalVerificationConfig(BaseModel):
    """Numerical verification configuration."""

    num_samples: int = Field(default=1000, ge=100)
    grid_density: int = Field(default=20, ge=5)
    tolerance: float = Field(default=1e-6, gt=0)


class SymbolicVerificationConfig(BaseModel):
    """Symbolic verification configuration."""

    simplify: bool = Field(default=True)
    timeout: int = Field(default=30, ge=1)


class SOSVerificationConfig(BaseModel):
    """SOS verification configuration."""

    solver: str = Field(default="mosek", description="SOS solver")
    degree: int = Field(default=4, ge=2)


class VerificationConfig(BaseModel):
    """Verification configuration."""

    numerical: NumericalVerificationConfig = Field(default_factory=NumericalVerificationConfig)
    symbolic: SymbolicVerificationConfig = Field(default_factory=SymbolicVerificationConfig)
    sos: SOSVerificationConfig = Field(default_factory=SOSVerificationConfig)
    methods: List[str] = Field(default=["numerical", "symbolic"])


class RateLimitConfig(BaseModel):
    """Rate limit configuration."""

    requests_per_day: int = Field(default=50, ge=1)
    premium_requests_per_day: int = Field(default=200, ge=1)


class SessionConfig(BaseModel):
    """Session configuration."""

    secret_key: str = Field(default="change-me-in-production")
    permanent_session_lifetime: int = Field(default=86400)  # 24 hours


class PasswordConfig(BaseModel):
    """Password configuration."""

    min_length: int = Field(default=8, ge=6)
    require_uppercase: bool = Field(default=True)
    require_lowercase: bool = Field(default=True)
    require_numbers: bool = Field(default=True)
    require_special: bool = Field(default=True)


class CORSConfig(BaseModel):
    """CORS configuration."""

    allowed_origins: List[str] = Field(default=["*"])
    allowed_methods: List[str] = Field(default=["GET", "POST", "PUT", "DELETE"])
    allowed_headers: List[str] = Field(default=["*"])


class SecurityConfig(BaseModel):
    """Security configuration."""

    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    password: PasswordConfig = Field(default_factory=PasswordConfig)
    cors: CORSConfig = Field(default_factory=CORSConfig)


class DeploymentConfig(BaseModel):
    """Deployment configuration."""

    mode: str = Field(default="local", description="Deployment mode")
    inference_api_url: Optional[str] = Field(default=None)
    redis_url: Optional[str] = Field(default=None)
    database_url: str = Field(default="sqlite:///fm_llm_solver.db")

    @field_validator("mode")
    def validate_mode(cls, v):
        """Validate deployment mode."""
        valid_modes = ["local", "hybrid", "cloud"]
        if v not in valid_modes:
            raise ValueError(f"Invalid mode: {v}. Must be one of {valid_modes}")
        return v


class CostConfig(BaseModel):
    """Cost configuration."""

    gpu_cost_per_hour: float = Field(default=0.50, ge=0)
    api_cost_per_1k: float = Field(default=0.02, ge=0)
    storage_cost_per_gb_month: float = Field(default=0.023, ge=0)
    bandwidth_cost_per_gb: float = Field(default=0.09, ge=0)


class AlertConfig(BaseModel):
    """Alert configuration."""

    error_rate_threshold: float = Field(default=0.1, ge=0, le=1)
    gpu_utilization_threshold: float = Field(default=0.9, ge=0, le=1)
    response_time_threshold: float = Field(default=10.0, gt=0)


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""

    enabled: bool = Field(default=True)
    costs: CostConfig = Field(default_factory=CostConfig)
    metrics_retention_days: int = Field(default=90, ge=1)
    alerts: AlertConfig = Field(default_factory=AlertConfig)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO")
    structured: bool = Field(default=False)
    console: bool = Field(default=True)


class Config(BaseModel):
    """Main configuration class."""

    paths: PathConfig = Field(default_factory=PathConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    debug: bool = Field(default=False)
    version: str = Field(default="1.0.0")

    model_config = ConfigDict(extra="allow")  # Allow extra fields for extensibility


def load_config(
    config_path: Optional[Union[str, Path]] = None, overrides: Optional[Dict[str, Any]] = None
) -> Config:
    """
    Load configuration from file and environment.

    Args:
        config_path: Path to configuration file
        overrides: Dictionary of config overrides

    Returns:
        Loaded and validated configuration

    Raises:
        ConfigurationError: If configuration is invalid
    """
    # Default config path
    if config_path is None:
        config_path = os.environ.get("FM_LLM_CONFIG", "config/config.yaml")

    config_path = Path(config_path)

    # Load from file
    config_dict = {}
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                if config_path.suffix == ".yaml" or config_path.suffix == ".yml":
                    config_dict = yaml.safe_load(f) or {}
                elif config_path.suffix == ".json":
                    config_dict = json.load(f)
                else:
                    raise ConfigurationError(f"Unsupported config format: {config_path.suffix}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load config from {config_path}: {e}")

    # Apply environment overrides
    config_dict = apply_env_overrides(config_dict)

    # Apply explicit overrides
    if overrides:
        config_dict = deep_merge(config_dict, overrides)

    # Create and validate config
    try:
        config = Config(**config_dict)
    except Exception as e:
        raise ConfigurationError(f"Invalid configuration: {e}")

    # Create directories
    create_directories(config)

    return config


def apply_env_overrides(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides.

    Environment variables should be prefixed with FM_LLM_
    and use double underscores for nesting.

    Example: FM_LLM_MODEL__PROVIDER=openai
    """
    prefix = "FM_LLM_"

    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Remove prefix and convert to lowercase
            key_path = key[len(prefix) :].lower().split("__")

            # Navigate to the correct position in config_dict
            current = config_dict
            for part in key_path[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set the value (with type conversion)
            current[key_path[-1]] = convert_env_value(value)

    return config_dict


def convert_env_value(value: str) -> Any:
    """Convert environment variable string to appropriate type."""
    # Boolean
    if value.lower() in ("true", "false"):
        return value.lower() == "true"

    # Numeric
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    # JSON
    if value.startswith("[") or value.startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

    # String
    return value


def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def create_directories(config: Config) -> None:
    """Create necessary directories."""
    directories = [
        config.paths.kb_output_dir,
        config.paths.fetched_papers_dir,
        config.paths.ft_data_dir,
        config.paths.output_dir,
        config.paths.log_dir,
        config.paths.temp_dir,
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def validate_config(config: Config) -> List[str]:
    """
    Validate configuration for common issues.

    Returns:
        List of warning messages
    """
    warnings = []

    # Check for default secret key
    if config.security.session.secret_key == "change-me-in-production":
        warnings.append("Using default secret key - change for production!")

    # Check model availability
    if config.model.device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                warnings.append("CUDA requested but not available")
        except ImportError:
            warnings.append("PyTorch not installed - cannot check CUDA")

    # Check deployment configuration
    if config.deployment.mode == "hybrid" and not config.deployment.inference_api_url:
        warnings.append("Hybrid mode requires inference_api_url")

    return warnings
