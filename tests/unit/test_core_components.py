"""
Unit tests for FM-LLM Solver core components.

Tests configuration, logging, exceptions, and types.
"""

import json

import pytest
import yaml

from fm_llm_solver.core.config import (
    Config,
    apply_env_overrides,
    convert_env_value,
    load_config,
    validate_config,
)
from fm_llm_solver.core.exceptions import (
    AuthenticationError,
    FMLLMSolverError,
    ModelError,
    ValidationError,
)
from fm_llm_solver.core.logging import (
    StructuredFormatter,
    configure_logging,
    get_logger,
)
from fm_llm_solver.core.types import (
    BarrierCertificate,
    DomainBounds,
    GenerationResult,
    SystemDescription,
    SystemType,
    VerificationCheck,
    VerificationResult,
)


class TestConfig:
    """Test configuration management."""

    def test_default_config(self):
        """Test loading default configuration."""
        config = Config()

        # Check defaults
        assert config.model.provider == "qwen"
        assert config.model.device == "cuda"
        assert config.rag.enabled is True
        assert config.security.rate_limit.requests_per_day == 50
        assert config.deployment.mode == "local"

    def test_load_from_yaml(self, tmp_path):
        """Test loading configuration from YAML file."""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "model": {"provider": "openai", "name": "gpt-4", "temperature": 0.5},
            "rag": {"enabled": False, "k_retrieved": 5},
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_file)

        assert config.model.provider == "openai"
        assert config.model.name == "gpt-4"
        assert config.model.temperature == 0.5
        assert config.rag.enabled is False
        assert config.rag.k_retrieved == 5

    def test_environment_overrides(self, monkeypatch):
        """Test environment variable overrides."""
        monkeypatch.setenv("FM_LLM_MODEL__PROVIDER", "anthropic")
        monkeypatch.setenv("FM_LLM_MODEL__TEMPERATURE", "0.8")
        monkeypatch.setenv("FM_LLM_RAG__ENABLED", "false")
        monkeypatch.setenv("FM_LLM_SECURITY__RATE_LIMIT__REQUESTS_PER_DAY", "100")

        config_dict = {}
        config_dict = apply_env_overrides(config_dict)

        assert config_dict["model"]["provider"] == "anthropic"
        assert config_dict["model"]["temperature"] == 0.8
        assert config_dict["rag"]["enabled"] is False
        assert config_dict["security"]["rate_limit"]["requests_per_day"] == 100

    def test_convert_env_value(self):
        """Test environment value conversion."""
        assert convert_env_value("true") is True
        assert convert_env_value("false") is False
        assert convert_env_value("42") == 42
        assert convert_env_value("3.14") == 3.14
        assert convert_env_value('["a", "b"]') == ["a", "b"]
        assert convert_env_value('{"key": "value"}') == {"key": "value"}
        assert convert_env_value("string") == "string"

    def test_validate_config(self):
        """Test configuration validation."""
        config = Config()
        warnings = validate_config(config)

        # Should warn about default secret key
        assert any("secret key" in w for w in warnings)

    def test_invalid_provider(self):
        """Test invalid model provider."""
        with pytest.raises(ValueError):
            Config(model={"provider": "invalid"})

    def test_path_expansion(self, monkeypatch):
        """Test path expansion with environment variables."""
        monkeypatch.setenv("TEST_DIR", "/test/path")

        config = Config(paths={"kb_output_dir": "$TEST_DIR/kb"})
        assert config.paths.kb_output_dir == "/test/path/kb"


class TestExceptions:
    """Test exception hierarchy."""

    def test_base_exception(self):
        """Test base exception functionality."""
        error = FMLLMSolverError(
            "Test error", error_code="TEST_ERROR", details={"key": "value"}
        )

        assert str(error) == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.details == {"key": "value"}

        # Test to_dict
        error_dict = error.to_dict()
        assert error_dict["error"] == "TEST_ERROR"
        assert error_dict["message"] == "Test error"
        assert error_dict["details"] == {"key": "value"}

    def test_validation_error(self):
        """Test validation error with field."""
        error = ValidationError("Invalid value", field="temperature")

        assert error.details["field"] == "temperature"

    def test_model_error(self):
        """Test model error with model name."""
        error = ModelError("Model failed", model_name="test-model")

        assert error.details["model"] == "test-model"

    def test_api_errors(self):
        """Test API error types."""
        auth_error = AuthenticationError()
        assert auth_error.status_code == 401

        auth_error = AuthenticationError("Custom message")
        assert str(auth_error) == "Custom message"


class TestTypes:
    """Test type definitions."""

    def test_system_type_enum(self):
        """Test SystemType enum."""
        assert SystemType.CONTINUOUS.value == "continuous"
        assert SystemType.DISCRETE.value == "discrete"
        assert SystemType.STOCHASTIC.value == "stochastic"

    def test_domain_bounds(self):
        """Test DomainBounds functionality."""
        bounds = DomainBounds(bounds={"x": (-1.0, 1.0), "y": (0.0, 2.0)})

        # Test contains
        assert bounds.contains({"x": 0.0, "y": 1.0})
        assert not bounds.contains({"x": 2.0, "y": 1.0})
        assert not bounds.contains({"x": 0.0})  # Missing variable

    def test_system_description(self):
        """Test SystemDescription to_text conversion."""
        system = SystemDescription(
            dynamics={"x": "-x + y", "y": "x - y"},
            initial_set="x**2 + y**2 <= 0.5",
            unsafe_set="x**2 + y**2 >= 2.0",
            system_type=SystemType.CONTINUOUS,
            domain_bounds=DomainBounds(bounds={"x": (-2, 2), "y": (-2, 2)}),
        )

        text = system.to_text()
        assert "dx/dt = -x + y" in text
        assert "dy/dt = x - y" in text
        assert "Initial Set: x**2 + y**2 <= 0.5" in text
        assert "Unsafe Set: x**2 + y**2 >= 2.0" in text
        assert "x ∈ [-2, 2]" in text

    def test_barrier_certificate(self):
        """Test BarrierCertificate."""
        cert = BarrierCertificate(
            expression="x**2 + y**2 - 1",
            variables=["x", "y"],
            certificate_type="standard",
        )

        assert str(cert) == "x**2 + y**2 - 1"
        assert cert.variables == ["x", "y"]

    def test_verification_result(self):
        """Test VerificationResult."""
        checks = [
            VerificationCheck("initial_set", True, "Initial set satisfied"),
            VerificationCheck("unsafe_set", True, "Unsafe set satisfied"),
            VerificationCheck("invariance", False, "Invariance not satisfied"),
        ]

        result = VerificationResult(
            valid=False, checks=checks, computation_time=1.23, method="numerical"
        )

        summary = result.summary
        assert summary["initial_set"] is True
        assert summary["unsafe_set"] is True
        assert summary["invariance"] is False

    def test_generation_result(self):
        """Test GenerationResult."""
        cert = BarrierCertificate("x**2 + y**2", ["x", "y"])

        result = GenerationResult(
            certificate=cert,
            confidence=0.85,
            generation_time=2.5,
            model_name="test-model",
        )

        assert result.success is True

        # Test failed result
        failed_result = GenerationResult(
            certificate=None, confidence=0.0, error="Generation failed"
        )

        assert failed_result.success is False


class TestLogging:
    """Test logging configuration."""

    def test_configure_logging(self, tmp_path):
        """Test logging configuration."""
        log_dir = tmp_path / "logs"

        configure_logging(
            level="DEBUG", log_dir=str(log_dir), console=True, structured=False
        )

        # Check log files created
        assert log_dir.exists()

        # Test logging
        logger = get_logger("test")
        logger.info("Test message")
        logger.error("Test error")

    def test_get_logger_with_context(self):
        """Test getting logger with context."""
        context = {"user_id": "123", "request_id": "abc"}
        logger = get_logger("test", context)

        # Logger should be an adapter
        assert hasattr(logger, "extra")
        assert logger.extra == context

    def test_structured_formatter(self):
        """Test structured JSON formatter."""
        import logging

        formatter = StructuredFormatter()

        # Create a log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Add extra fields
        record.user_id = "123"
        record.request_id = "abc"

        formatted = formatter.format(record)

        # Should be valid JSON
        parsed = json.loads(formatted)
        assert parsed["message"] == "Test message"
        assert parsed["level"] == "INFO"
        assert parsed["user_id"] == "123"
        assert parsed["request_id"] == "abc"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
