"""
Configuration manager for FM-LLM Solver.

Provides unified configuration management with environment-specific settings,
secret management, and comprehensive validation.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, OmegaConf

from fm_llm_solver.core.exceptions import ConfigurationError


class Environment(Enum):
    """Deployment environments."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class SecretProvider(Enum):
    """Secret providers."""

    ENVIRONMENT = "environment"
    FILE = "file"
    AWS_SECRETS = "aws_secrets"
    AZURE_KEYVAULT = "azure_keyvault"
    HASHICORP_VAULT = "hashicorp_vault"


@dataclass
class ConfigurationTemplate:
    """Template for generating configuration files."""

    environment: Environment
    template_vars: Dict[str, Any] = field(default_factory=dict)
    required_secrets: List[str] = field(default_factory=list)
    optional_secrets: List[str] = field(default_factory=list)


class ConfigurationManager:
    """
    Comprehensive configuration manager with environment support.
    """

    def __init__(
        self,
        config_dir: Optional[Union[str, Path]] = None,
        environment: Optional[Union[str, Environment]] = None,
        secret_provider: SecretProvider = SecretProvider.ENVIRONMENT,
    ):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory containing configuration files
            environment: Deployment environment
            secret_provider: Provider for secrets
        """
        self.logger = logging.getLogger(__name__)

        # Set up paths
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Determine environment
        if isinstance(environment, str):
            environment = Environment(environment)
        self.environment = environment or self._detect_environment()

        # Secret management
        self.secret_provider = secret_provider
        self._secrets_cache = {}

        # Configuration cache
        self._config_cache: Optional[DictConfig] = None

        self.logger.info(
            f"Configuration manager initialized for {self.environment.value}"
        )

    def _detect_environment(self) -> Environment:
        """Auto-detect environment from various sources."""
        # Check environment variable
        env_var = os.environ.get("FM_LLM_ENV", "").lower()
        if env_var:
            try:
                return Environment(env_var)
            except ValueError:
                pass

        # Check common environment indicators
        if os.environ.get("CI"):
            return Environment.TESTING

        if os.environ.get("KUBERNETES_SERVICE_HOST"):
            return Environment.PRODUCTION

        if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
            return Environment.PRODUCTION

        # Default to development
        return Environment.DEVELOPMENT

    def load_config(
        self,
        config_name: Optional[str] = None,
        merge_env_specific: bool = True,
        validate: bool = True,
        use_cache: bool = True,
    ) -> DictConfig:
        """
        Load configuration with environment-specific overrides.

        Args:
            config_name: Name of config file (without extension)
            merge_env_specific: Whether to merge environment-specific config
            validate: Whether to validate configuration
            use_cache: Whether to use cached configuration

        Returns:
            Configuration object
        """
        if use_cache and self._config_cache:
            return self._config_cache

        try:
            # Base configuration
            if not config_name:
                config_name = "config"

            base_config = self._load_base_config(config_name)

            # Environment-specific overrides
            if merge_env_specific:
                env_config = self._load_env_config(config_name)
                if env_config:
                    base_config = OmegaConf.merge(base_config, env_config)

            # Apply environment variable overrides
            config_with_env = self._apply_env_overrides(base_config)

            # Resolve secrets
            config_with_secrets = self._resolve_secrets(config_with_env)

            # Expand paths and variables
            final_config = self._expand_variables(config_with_secrets)

            # Validate if requested
            if validate:
                self._validate_config(final_config)

            # Cache result
            if use_cache:
                self._config_cache = final_config

            self.logger.info(
                f"Configuration loaded successfully for {self.environment.value}"
            )
            return final_config

        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}")

    def _load_base_config(self, config_name: str) -> DictConfig:
        """Load base configuration file."""
        config_path = self.config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            # Try alternative extensions
            for ext in [".yml", ".json"]:
                alt_path = self.config_dir / f"{config_name}{ext}"
                if alt_path.exists():
                    config_path = alt_path
                    break
            else:
                raise ConfigurationError(f"Configuration file not found: {config_path}")

        self.logger.info(f"Loading base config from {config_path}")
        return OmegaConf.load(config_path)

    def _load_env_config(self, config_name: str) -> Optional[DictConfig]:
        """Load environment-specific configuration."""
        env_config_path = (
            self.config_dir / f"{config_name}.{self.environment.value}.yaml"
        )

        if env_config_path.exists():
            self.logger.info(f"Loading environment config from {env_config_path}")
            return OmegaConf.load(env_config_path)

        return None

    def _apply_env_overrides(self, config: DictConfig) -> DictConfig:
        """Apply environment variable overrides."""
        env_overrides = {}

        # Look for FM_LLM_* environment variables
        for key, value in os.environ.items():
            if key.startswith("FM_LLM_"):
                # Convert FM_LLM_MODEL_NAME to model.name
                config_key = key[7:].lower().replace("_", ".")
                env_overrides[config_key] = self._convert_env_value(value)

        if env_overrides:
            self.logger.info(f"Applying {len(env_overrides)} environment overrides")
            # Convert flat keys to nested structure
            nested_overrides = self._flatten_to_nested(env_overrides)
            config = OmegaConf.merge(config, nested_overrides)

        return config

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean values
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        if value.lower() in ("false", "no", "0", "of"):
            return False

        # Numeric values
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # JSON values
        if value.startswith(("{", "[")):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass

        return value

    def _flatten_to_nested(self, flat_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat dot-notation keys to nested dictionary."""
        result = {}

        for key, value in flat_dict.items():
            keys = key.split(".")
            current = result

            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            current[keys[-1]] = value

        return result

    def _resolve_secrets(self, config: DictConfig) -> DictConfig:
        """Resolve secret references in configuration."""

        def resolve_value(value):
            if isinstance(value, str) and value.startswith("${secret:"):
                secret_name = value[9:-1]  # Remove ${secret: and }
                return self._get_secret(secret_name)
            elif isinstance(value, DictConfig):
                return OmegaConf.create({k: resolve_value(v) for k, v in value.items()})
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            return value

        return OmegaConf.create({k: resolve_value(v) for k, v in config.items()})

    def _get_secret(self, secret_name: str) -> str:
        """Get secret from configured provider."""
        # Check cache first
        if secret_name in self._secrets_cache:
            return self._secrets_cache[secret_name]

        secret_value = None

        if self.secret_provider == SecretProvider.ENVIRONMENT:
            secret_value = os.environ.get(secret_name)

        elif self.secret_provider == SecretProvider.FILE:
            secrets_file = self.config_dir / "secrets.json"
            if secrets_file.exists():
                with open(secrets_file, "r") as f:
                    secrets = json.load(f)
                secret_value = secrets.get(secret_name)

        elif self.secret_provider == SecretProvider.AWS_SECRETS:
            secret_value = self._get_aws_secret(secret_name)

        # Add other providers as needed

        if secret_value is None:
            raise ConfigurationError(f"Secret not found: {secret_name}")

        # Cache the secret
        self._secrets_cache[secret_name] = secret_value
        return secret_value

    def _get_aws_secret(self, secret_name: str) -> Optional[str]:
        """Get secret from AWS Secrets Manager."""
        try:
            import boto3
            import botocore

            session = boto3.session.Session()
            client = session.client("secretsmanager")

            response = client.get_secret_value(SecretId=secret_name)
            return response["SecretString"]

        except ImportError:
            self.logger.warning("boto3 not available for AWS secrets")
            return None
        except botocore.exceptions.ClientError as e:
            self.logger.warning(f"Failed to get AWS secret {secret_name}: {e}")
            return None

    def _expand_variables(self, config: DictConfig) -> DictConfig:
        """Expand path variables and other substitutions."""

        def expand_value(value):
            if isinstance(value, str):
                # Expand environment variables
                value = os.path.expandvars(value)
                # Expand user home directory
                value = os.path.expanduser(value)
                return value
            elif isinstance(value, DictConfig):
                return OmegaConf.create({k: expand_value(v) for k, v in value.items()})
            elif isinstance(value, list):
                return [expand_value(item) for item in value]
            return value

        return OmegaConf.create({k: expand_value(v) for k, v in config.items()})

    def _validate_config(self, config: DictConfig) -> None:
        """Validate configuration against schema."""
        # Basic validation
        required_sections = ["paths", "model", "logging"]

        for section in required_sections:
            if section not in config:
                raise ConfigurationError(
                    f"Required configuration section missing: {section}"
                )

        # Validate paths exist
        if "paths" in config:
            for path_name, path_value in config.paths.items():
                if path_name.endswith("_dir") and path_value:
                    path_obj = Path(path_value)
                    try:
                        path_obj.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        self.logger.warning(
                            f"Could not create directory {path_value}: {e}"
                        )

        self.logger.info("Configuration validation passed")

    def save_template(
        self, template: ConfigurationTemplate, output_path: Optional[Path] = None
    ) -> Path:
        """Save configuration template for environment."""
        if not output_path:
            output_path = self.config_dir / f"config.{template.environment.value}.yaml"

        # Generate template content
        template_config = self._generate_template_config(template)

        # Save to file
        OmegaConf.save(template_config, output_path)

        self.logger.info(f"Configuration template saved to {output_path}")
        return output_path

    def _generate_template_config(self, template: ConfigurationTemplate) -> DictConfig:
        """Generate configuration from template."""
        config = OmegaConf.create(
            {
                "environment": template.environment.value,
                "deployment": {
                    "mode": (
                        "local"
                        if template.environment == Environment.DEVELOPMENT
                        else "cloud"
                    )
                },
            }
        )

        # Add template variables
        for key, value in template.template_vars.items():
            OmegaConf.set(config, key, value)

        # Add secret placeholders
        for secret in template.required_secrets:
            OmegaConf.set(config, secret, f"${{secret:{secret}}}")

        return config

    def get_environment_info(self) -> Dict[str, Any]:
        """Get information about current environment."""
        return {
            "environment": self.environment.value,
            "config_dir": str(self.config_dir),
            "secret_provider": self.secret_provider.value,
            "has_cached_config": self._config_cache is not None,
            "available_configs": [
                f.stem
                for f in self.config_dir.glob("*.yaml")
                if not f.stem.endswith(
                    (".development", ".testing", ".staging", ".production")
                )
            ],
        }

    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self._config_cache = None
        self._secrets_cache.clear()
        self.logger.info("Configuration cache cleared")

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        default_logging = {
            "log_directory": "logs",
            "root_level": "INFO",
            "loggers": {
                "api": {
                    "level": "INFO",
                    "handlers": ["console", "rotating_file"],
                    "json_format": True,
                    "propagate": False,
                },
                "model_operations": {
                    "level": "INFO",
                    "handlers": ["console", "rotating_file"],
                    "json_format": True,
                    "propagate": False,
                },
                "security": {
                    "level": "WARNING",
                    "handlers": ["console", "rotating_file", "syslog"],
                    "json_format": True,
                    "propagate": False,
                },
                "performance": {
                    "level": "INFO",
                    "handlers": ["rotating_file"],
                    "json_format": True,
                    "propagate": False,
                },
                "database": {
                    "level": "INFO",
                    "handlers": ["console", "rotating_file"],
                    "json_format": True,
                    "propagate": False,
                },
                "web": {
                    "level": "INFO",
                    "handlers": ["console", "rotating_file"],
                    "json_format": True,
                    "propagate": False,
                },
            },
        }

        # Try to get logging config from loaded configuration
        try:
            config = self.load_config(use_cache=True)
            user_logging = config.get("logging", {})
        except Exception:
            # If config loading fails, use empty user config
            user_logging = {}

        # Merge configurations
        merged_config = default_logging.copy()
        merged_config.update(user_logging)

        # Merge logger configs
        if "loggers" in user_logging:
            merged_config["loggers"].update(user_logging["loggers"])

        return merged_config
