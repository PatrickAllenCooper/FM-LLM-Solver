#!/usr/bin/env python3
"""
Hierarchical Configuration Loader for FM-LLM Solver

This module provides a configuration loading system that supports:
1. Base configuration (config/base.yaml)
2. Environment-specific overrides (config/environments/{env}.yaml)
3. User-specific overrides (config/user/local.yaml)
4. Environment variable substitution

Configuration loading order (later overrides earlier):
base.yaml → environments/{environment}.yaml → user/local.yaml → environment variables

Usage:
    from utils.hierarchical_config_loader import load_hierarchical_config
    
    config = load_hierarchical_config()  # Auto-detect environment
    config = load_hierarchical_config(environment="production")  # Explicit environment
"""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Raised when configuration loading fails."""
    pass

class HierarchicalConfigLoader:
    """
    Hierarchical configuration loader that supports base, environment, and user overrides.
    """
    
    def __init__(self, config_root: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_root: Root directory containing configuration files.
                        Defaults to PROJECT_ROOT/config
        """
        if config_root is None:
            # Auto-detect project root
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent  # Go up from utils/ to project root
            config_root = project_root / "config"
        
        self.config_root = Path(config_root)
        self.base_config_path = self.config_root / "base.yaml"
        self.environments_dir = self.config_root / "environments"
        self.user_dir = self.config_root / "user"
        
        # Ensure directories exist
        self.environments_dir.mkdir(parents=True, exist_ok=True)
        self.user_dir.mkdir(parents=True, exist_ok=True)
    
    def _detect_environment(self) -> str:
        """
        Detect the current environment from various sources.
        
        Priority:
        1. FM_LLM_ENV environment variable
        2. ENVIRONMENT environment variable  
        3. NODE_ENV environment variable (for compatibility)
        4. Default to 'development'
        
        Returns:
            Environment name (development, staging, production)
        """
        env_vars = ["FM_LLM_ENV", "ENVIRONMENT", "NODE_ENV"]
        
        for env_var in env_vars:
            env_value = os.getenv(env_var)
            if env_value:
                env_value = env_value.lower()
                if env_value in ["development", "dev"]:
                    return "development"
                elif env_value in ["staging", "stage"]:
                    return "staging"
                elif env_value in ["production", "prod"]:
                    return "production"
                elif env_value in ["testing", "test"]:
                    return "testing"
                else:
                    logger.warning(f"Unknown environment '{env_value}', defaulting to development")
                    return "development"
        
        logger.info("No environment specified, defaulting to development")
        return "development"
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Load a YAML file safely.
        
        Args:
            file_path: Path to the YAML file
            
        Returns:
            Parsed YAML content as dictionary
            
        Raises:
            ConfigurationError: If file cannot be loaded or parsed
        """
        if not file_path.exists():
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
                return content if content is not None else {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML file {file_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error reading file {file_path}: {e}")
    
    def _substitute_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively substitute environment variables in configuration values.
        
        Supports patterns:
        - ${ENV:VAR_NAME}          - Required variable
        - ${ENV:VAR_NAME:default}  - Variable with default value
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with environment variables substituted
        """
        if isinstance(config, dict):
            return {key: self._substitute_environment_variables(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._substitute_environment_variables(item) for item in config]
        elif isinstance(config, str):
            return self._substitute_env_vars_in_string(config)
        else:
            return config
    
    def _substitute_env_vars_in_string(self, value: str) -> str:
        """
        Substitute environment variables in a string value.
        
        Args:
            value: String that may contain environment variable references
            
        Returns:
            String with environment variables substituted
        """
        # Pattern: ${ENV:VAR_NAME} or ${ENV:VAR_NAME:default}
        pattern = r'\$\{ENV:([^}:]+)(?::([^}]*))?\}'
        
        def replace_env_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            
            env_value = os.getenv(var_name)
            if env_value is not None:
                return env_value
            elif default_value != "":
                return default_value
            elif match.group(2) is None:
                # No default provided, this is a required variable
                raise ConfigurationError(f"Required environment variable '{var_name}' is not set")
            else:
                # Empty default provided
                return ""
        
        return re.sub(pattern, replace_env_var, value)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, with override values taking precedence.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def load_config(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """
        Load hierarchical configuration.
        
        Args:
            environment: Environment name. If None, auto-detect.
            
        Returns:
            Merged configuration dictionary
            
        Raises:
            ConfigurationError: If configuration cannot be loaded
        """
        if environment is None:
            environment = self._detect_environment()
        
        logger.info(f"Loading configuration for environment: {environment}")
        
        # 1. Load base configuration
        logger.debug(f"Loading base configuration from: {self.base_config_path}")
        config = self._load_yaml_file(self.base_config_path)
        
        if not config:
            raise ConfigurationError(f"Base configuration file not found or empty: {self.base_config_path}")
        
        # 2. Load environment-specific overrides
        env_config_path = self.environments_dir / f"{environment}.yaml"
        logger.debug(f"Loading environment configuration from: {env_config_path}")
        env_config = self._load_yaml_file(env_config_path)
        
        if env_config:
            logger.debug(f"Merging environment configuration: {environment}")
            config = self._deep_merge(config, env_config)
        else:
            logger.warning(f"No environment configuration found for: {environment}")
        
        # 3. Load user-specific overrides
        user_config_path = self.user_dir / "local.yaml"
        logger.debug(f"Loading user configuration from: {user_config_path}")
        user_config = self._load_yaml_file(user_config_path)
        
        if user_config:
            logger.debug("Merging user configuration")
            config = self._deep_merge(config, user_config)
        
        # 4. Substitute environment variables
        logger.debug("Substituting environment variables")
        config = self._substitute_environment_variables(config)
        
        # 5. Add runtime metadata
        config["_metadata"] = {
            "environment": environment,
            "loaded_at": str(Path.cwd()),
            "config_files": {
                "base": str(self.base_config_path),
                "environment": str(env_config_path) if env_config else None,
                "user": str(user_config_path) if user_config else None,
            }
        }
        
        logger.info(f"Configuration loaded successfully for environment: {environment}")
        return config
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the loaded configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if valid, raises ConfigurationError if not
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        required_sections = ["environment", "database", "web", "inference"]
        
        for section in required_sections:
            if section not in config:
                raise ConfigurationError(f"Required configuration section missing: {section}")
        
        # Validate environment settings
        env = config.get("environment", {})
        if "mode" not in env:
            raise ConfigurationError("environment.mode is required")
        
        if env["mode"] not in ["development", "staging", "production", "testing"]:
            raise ConfigurationError(f"Invalid environment.mode: {env['mode']}")
        
        # Validate database settings
        db = config.get("database", {})
        if "url" not in db:
            raise ConfigurationError("database.url is required")
        
        # Validate web settings
        web = config.get("web", {})
        if "host" not in web or "port" not in web:
            raise ConfigurationError("web.host and web.port are required")
        
        logger.debug("Configuration validation passed")
        return True

# Global loader instance
_loader = None

def get_config_loader() -> HierarchicalConfigLoader:
    """Get the global configuration loader instance."""
    global _loader
    if _loader is None:
        _loader = HierarchicalConfigLoader()
    return _loader

def load_hierarchical_config(environment: Optional[str] = None, validate: bool = True) -> Dict[str, Any]:
    """
    Load hierarchical configuration (convenience function).
    
    Args:
        environment: Environment name. If None, auto-detect.
        validate: Whether to validate the configuration
        
    Returns:
        Merged configuration dictionary
        
    Raises:
        ConfigurationError: If configuration cannot be loaded or is invalid
    """
    loader = get_config_loader()
    config = loader.load_config(environment)
    
    if validate:
        loader.validate_config(config)
    
    return config

# Backward compatibility with existing code
def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration with backward compatibility.
    
    Args:
        config_path: Ignored (for backward compatibility)
        
    Returns:
        Configuration dictionary
    """
    return load_hierarchical_config()

if __name__ == "__main__":
    # Test the configuration loader
    try:
        config = load_hierarchical_config()
        print(f"✅ Configuration loaded successfully!")
        print(f"Environment: {config['environment']['mode']}")
        print(f"Deployment mode: {config['deployment']['mode']}")
        print(f"Database: {config['database']['url']}")
    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}") 