"""
Shared test configuration and fixtures for FM-LLM-Solver tests.

This file contains common fixtures, test utilities, and configuration
that can be used across all test modules.
"""

import os
import sys
import pytest
import tempfile
import logging
from pathlib import Path
from unittest.mock import Mock, patch

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our modules
from fm_llm_solver.core.config_manager import ConfigurationManager
from fm_llm_solver.core.database_manager import DatabaseManager
from fm_llm_solver.core.logging_manager import LoggingManager
from fm_llm_solver.core.error_handler import ErrorHandler
from fm_llm_solver.core.cache_manager import CacheManager
from fm_llm_solver.core.monitoring import MonitoringManager


@pytest.fixture(scope="session")
def test_environment():
    """Set up test environment variables."""
    test_env = {
        "FM_LLM_ENV": "testing",
        "SECRET_KEY": "test-secret-key",
        "DB_PASSWORD": "test-password",
        "ENCRYPTION_KEY": "test-encryption-key",
        "REDIS_URL": "redis://localhost:6379/15",  # Use a separate test database
    }

    with patch.dict(os.environ, test_env):
        yield test_env


@pytest.fixture(scope="session")
def test_config():
    """Complete test configuration."""
    return {
        "environment": "testing",
        "database": {
            "primary": {
                "host": "localhost",
                "port": 5432,
                "database": "test_fm_llm",
                "username": "test_user",
                "password": "${secret:DB_PASSWORD}",
                "pool_size": 5,
                "max_overflow": 10,
                "pool_timeout": 30,
                "echo": False,
            }
        },
        "cache": {
            "backend": "memory",
            "max_size": 1000,
            "default_ttl": 300,
            "redis_url": "${secret:REDIS_URL}",
            "key_prefix": "test_",
            "namespace": "testing",
        },
        "logging": {
            "log_directory": "/tmp/test_logs",
            "root_level": "DEBUG",
            "loggers": {
                "api": {
                    "level": "DEBUG",
                    "handlers": ["console"],
                    "json_format": False,
                    "propagate": False,
                },
                "test": {
                    "level": "DEBUG",
                    "handlers": ["console"],
                    "json_format": False,
                    "propagate": False,
                },
            },
        },
        "monitoring": {
            "enabled": True,
            "metrics": {
                "prometheus_enabled": False,  # Disable Prometheus in tests
                "custom_metrics_retention_hours": 1,
                "system_metrics_interval": 60,
            },
            "health_checks": {
                "enabled": True,
                "default_interval": 30,
                "default_timeout": 5,
                "critical_failure_threshold": 3,
            },
        },
        "error_handling": {
            "max_retries": 3,
            "retry_delay": 0.1,  # Faster retries in tests
            "exponential_backof": True,
            "circuit_breaker": {"failure_threshold": 5, "recovery_timeout": 30},
        },
        "security": {
            "rate_limit": {
                "default": "1000/minute",  # More lenient in tests
                "api_endpoints": "10000/hour",
                "auth_endpoints": "100/minute",
            },
            "cors": {
                "enabled": True,
                "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
                "max_age": 600,
            },
            "headers": {
                "force_https": False,  # Disable HTTPS requirement in tests
                "content_security_policy": False,
                "frame_options": "DENY",
            },
        },
        "web_interface": {
            "host": "127.0.0.1",
            "port": 5000,
            "debug": True,
            "testing": True,
            "cors_origins": ["http://localhost:3000"],
        },
        "fine_tuning": {
            "base_model_name": "test-model",
            "use_adapter": True,
            "quantization": {
                "use_4bit": False,  # Disable quantization in tests
                "bnb_4bit_compute_dtype": "float32",
            },
            "training": {
                "num_train_epochs": 1,
                "per_device_train_batch_size": 1,
                "max_seq_length": 512,  # Shorter sequences for testing
            },
        },
        "inference": {
            "rag_k": 3,
            "max_new_tokens": 50,  # Shorter responses in tests
            "temperature": 0.3,
            "device": "cpu",  # Force CPU in tests
            "torch_dtype": "float32",
        },
    }


@pytest.fixture
def config_manager(test_environment, test_config):
    """Create a test configuration manager."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = Path(temp_dir) / "test_config.yaml"

        import yaml

        with open(config_file, "w") as f:
            yaml.dump(test_config, f)

        config_manager = ConfigurationManager(config_file)
        yield config_manager


@pytest.fixture
def mock_config_manager():
    """Create a mock configuration manager for unit tests."""
    mock_config = Mock()

    # Default return values for common configuration keys
    config_values = {
        "environment": "testing",
        "database": {
            "primary": {
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "username": "test_user",
                "password": "test_password",
            }
        },
        "cache.backend": "memory",
        "cache.max_size": 1000,
        "cache.default_ttl": 300,
        "cache.key_prefix": "test_",
        "logging.root_level": "DEBUG",
        "logging.log_directory": "/tmp/test_logs",
        "monitoring.enabled": True,
        "monitoring.metrics.prometheus_enabled": False,
        "error_handling.max_retries": 3,
        "error_handling.retry_delay": 0.1,
        "web_interface.host": "127.0.0.1",
        "web_interface.port": 5000,
        "web_interface.debug": True,
    }

    def mock_get(key, default=None):
        if key in config_values:
            return config_values[key]

        # Handle nested keys
        keys = key.split(".")
        value = config_values
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    mock_config.get.side_effect = mock_get
    mock_config.environment = "testing"

    return mock_config


@pytest.fixture
def database_manager(config_manager):
    """Create a test database manager."""
    return DatabaseManager(config_manager)


@pytest.fixture
def cache_manager(config_manager):
    """Create a test cache manager."""
    return CacheManager(config_manager)


@pytest.fixture
def logging_manager(config_manager):
    """Create a test logging manager."""
    manager = LoggingManager(config_manager)
    manager.setup_logging()
    return manager


@pytest.fixture
def error_handler(config_manager):
    """Create a test error handler."""
    return ErrorHandler(config_manager)


@pytest.fixture
def monitoring_manager(config_manager):
    """Create a test monitoring manager."""
    return MonitoringManager(config_manager)


@pytest.fixture
def test_logger(logging_manager):
    """Get a test logger."""
    return logging_manager.get_logger("test")


@pytest.fixture(scope="function")
def clean_cache(cache_manager):
    """Ensure cache is clean before and after each test."""
    cache_manager.clear()
    yield cache_manager
    cache_manager.clear()


@pytest.fixture
def sample_problem():
    """Sample optimization problem for testing."""
    return {
        "description": "Minimize the function f(x, y) = x^2 + y^2 subject to x + y <= 10 and x, y >= 0",
        "variables": ["x", "y"],
        "constraints": ["x + y <= 10", "x >= 0", "y >= 0"],
        "objective": "minimize x^2 + y^2",
    }


@pytest.fixture
def sample_certificate():
    """Sample certificate for testing."""
    return {
        "certificate": "V(x, y) = x^2 + y^2 + 1",
        "verification_result": {"is_valid": True, "method": "numerical", "confidence": 0.95},
        "metadata": {
            "generation_time": 1.23,
            "model_used": "test-model",
            "timestamp": "2024-01-01T00:00:00Z",
        },
    }


# Test markers configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "smoke: marks tests as smoke tests")
    config.addinivalue_line("markers", "web: marks tests related to web interface")
    config.addinivalue_line("markers", "api: marks tests related to API endpoints")
    config.addinivalue_line("markers", "database: marks tests related to database functionality")
    config.addinivalue_line("markers", "cache: marks tests related to caching functionality")


# Test collection configuration
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        test_file = str(item.fspath)

        if "unit" in test_file:
            item.add_marker(pytest.mark.unit)
        elif "integration" in test_file:
            item.add_marker(pytest.mark.integration)
        elif "performance" in test_file:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)

        # Add markers based on test file name
        if "test_web" in test_file or "web_interface" in test_file:
            item.add_marker(pytest.mark.web)
        elif "test_api" in test_file:
            item.add_marker(pytest.mark.api)
        elif "test_database" in test_file or "database" in test_file:
            item.add_marker(pytest.mark.database)
        elif "test_cache" in test_file or "cache" in test_file:
            item.add_marker(pytest.mark.cache)

        # Add smoke marker for health check tests
        if "health" in item.name.lower() or "smoke" in item.name.lower():
            item.add_marker(pytest.mark.smoke)


# Session hooks
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment."""
    # Ensure test directories exist - use Windows-compatible paths
    import tempfile

    temp_dir = tempfile.gettempdir()
    test_dirs = [
        Path(temp_dir) / "test_logs",
        Path(temp_dir) / "test_cache",
        Path(temp_dir) / "test_data",
    ]

    for test_dir in test_dirs:
        test_dir.mkdir(exist_ok=True)

    # Set logging level for tests
    logging.getLogger().setLevel(logging.DEBUG)

    yield

    # Cleanup after all tests
    import shutil

    for test_dir in test_dirs:
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)


# Utility functions for tests
def assert_valid_response(response, expected_status=200):
    """Assert that a response is valid."""
    assert response.status_code == expected_status
    assert response.headers.get("Content-Type") is not None


def assert_valid_json_response(response, expected_status=200):
    """Assert that a response is valid JSON."""
    assert_valid_response(response, expected_status)
    assert "application/json" in response.headers.get("Content-Type", "")
    assert response.json() is not None


def create_test_certificate():
    """Create a test certificate for testing."""
    return {
        "certificate": "V(x) = x^2 + 1",
        "variables": ["x"],
        "constraints": ["x >= 0"],
        "verification_result": {"is_valid": True, "method": "symbolic", "confidence": 1.0},
    }
