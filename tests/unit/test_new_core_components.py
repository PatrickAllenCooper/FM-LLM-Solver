"""
Comprehensive unit tests for new core components of FM-LLM-Solver.
"""

import os
import pytest
import tempfile
from unittest.mock import Mock, patch
from pathlib import Path

# Import the components we're testing
from fm_llm_solver.core.config_manager import ConfigurationManager
from fm_llm_solver.core.database_manager import DatabaseManager
from fm_llm_solver.core.cache_manager import CacheManager


class TestConfigurationManager:
    """Test suite for ConfigurationManager."""

    def test_init_with_valid_config(self):
        """Test initialization with valid configuration."""
        sample_config = {
            "environment": "testing",
            "database": {
                "primary": {
                    "host": "localhost",
                    "port": 5432,
                    "name": "test_db",
                    "user": "test_user",
                }
            },
            "paths": {"data_dir": "./data", "logs_dir": "./logs"},
            "logging": {"level": "INFO"},
            "model": {"provider": "dummy", "name": "dummy-model"},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            config_file = config_dir / "config.yaml"
            # Ensure the config directory exists
            config_dir.mkdir(parents=True, exist_ok=True)
            import yaml

            with open(config_file, "w") as f:
                yaml.dump(sample_config, f)
            with patch.dict(
                os.environ, {"DB_PASSWORD": "secret_password", "FM_LLM_ENV": "testing"}
            ):
                config_manager = ConfigurationManager(config_dir, environment="testing")
                assert config_manager.environment.value == "testing"

                # Load the configuration and access values
                config = config_manager.load_config()
                assert config["database"]["primary"]["host"] == "localhost"


class TestDatabaseManager:
    """Test suite for DatabaseManager."""

    def test_init_with_valid_config(self):
        """Test initialization with valid configuration."""
        mock_config = Mock()
        mock_config.get.return_value = {
            "primary": {
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "username": "test_user",
                "password": "test_password",
            }
        }

        db_manager = DatabaseManager(mock_config)
        assert db_manager.config_manager == mock_config


class TestCacheManager:
    """Test suite for CacheManager."""

    def test_init_with_valid_config(self):
        """Test initialization with valid configuration."""
        mock_config = Mock()
        mock_config.get.return_value = {
            "backend": "memory",
            "max_size": 1000,
            "default_ttl": 300,
            "key_prefix": "test_",
        }

        cache_manager = CacheManager(mock_config)
        assert cache_manager.config_manager == mock_config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
