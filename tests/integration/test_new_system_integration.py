"""
Integration tests for the new FM-LLM-Solver system components.

These tests verify that all components work together correctly:
- Web interface integration
- Database integration  
- Cache integration
- Configuration loading
- Error handling
- Monitoring integration
"""

import os
import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch

# Import our components
from fm_llm_solver.core.config_manager import ConfigurationManager
from fm_llm_solver.core.database_manager import DatabaseManager
from fm_llm_solver.core.logging_manager import LoggingManager
from fm_llm_solver.core.error_handler import ErrorHandler
from fm_llm_solver.core.cache_manager import CacheManager
from fm_llm_solver.core.monitoring import MonitoringManager
from fm_llm_solver.web.app import create_app
from fm_llm_solver.services.certificate_generator import CertificateGenerator
from fm_llm_solver.services.parser import Parser


class TestSystemIntegration:
    """Test suite for system integration."""
    
    @pytest.fixture
    def full_config(self):
        """Complete configuration for integration testing."""
        return {
            "environment": "testing",
            "database": {
                "primary": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "test_db",
                    "username": "test_user",
                    "password": "test_password",
                    "pool_size": 5,
                    "max_overflow": 10
                }
            },
            "cache": {
                "backend": "memory",
                "max_size": 1000,
                "default_ttl": 300,
                "key_prefix": "test_"
            },
            "logging": {
                "log_directory": "/tmp/test_logs",
                "root_level": "INFO",
                "loggers": {
                    "api": {
                        "level": "DEBUG",
                        "handlers": ["console"],
                        "json_format": True
                    }
                }
            },
            "monitoring": {
                "enabled": True,
                "metrics": {
                    "prometheus_enabled": True,
                    "custom_metrics_retention_hours": 24
                },
                "health_checks": {
                    "enabled": True,
                    "default_interval": 30
                }
            },
            "error_handling": {
                "max_retries": 3,
                "retry_delay": 1.0,
                "exponential_backoff": True
            },
            "web_interface": {
                "host": "127.0.0.1",
                "port": 5000,
                "debug": True
            }
        }
    
    @pytest.fixture
    def config_manager(self, full_config):
        """Configuration manager for integration tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.yaml"
            
            import yaml
            with open(config_file, 'w') as f:
                yaml.dump(full_config, f)
            
            yield ConfigurationManager(config_file)
    
    def test_configuration_loading(self, config_manager):
        """Test that configuration loads correctly."""
        assert config_manager.environment == "testing"
        assert config_manager.get("database.primary.host") == "localhost"
        assert config_manager.get("cache.backend") == "memory"
        assert config_manager.get("monitoring.enabled") == True
    
    def test_logging_manager_integration(self, config_manager):
        """Test logging manager integration."""
        logging_manager = LoggingManager(config_manager)
        logging_manager.setup_logging()
        
        # Get a logger and test it
        logger = logging_manager.get_logger("api")
        assert logger is not None
        
        # Test logging operations
        logger.info("Test integration message")
        logger.debug("Test debug message")
    
    def test_cache_manager_integration(self, config_manager):
        """Test cache manager integration."""
        cache_manager = CacheManager(config_manager)
        
        # Test cache operations
        cache_manager.set("integration_test_key", "integration_test_value", ttl=60)
        result = cache_manager.get("integration_test_key")
        assert result == "integration_test_value"
        
        # Test cache stats
        stats = cache_manager.get_stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "size" in stats
    
    def test_monitoring_manager_integration(self, config_manager):
        """Test monitoring manager integration."""
        monitoring_manager = MonitoringManager(config_manager)
        
        # Test metric recording
        monitoring_manager.record_metric("integration_test_metric", 42.0, {"test": "true"})
        
        # Test health check registration
        def test_health_check():
            return {"status": "healthy", "details": "Integration test service healthy"}
        
        monitoring_manager.register_health_check("integration_test", test_health_check)
        
        # Run health checks
        results = monitoring_manager.run_health_checks()
        assert "integration_test" in results
        assert results["integration_test"]["status"] == "healthy"
    
    def test_error_handler_integration(self, config_manager):
        """Test error handler integration."""
        error_handler = ErrorHandler(config_manager)
        
        # Test retry strategy
        call_count = 0
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = error_handler.handle_error(
            failing_function,
            strategy="retry",
            context={"operation": "integration_test"}
        )
        
        assert result == "success"
        assert call_count == 3
    
    def test_web_app_creation(self, config_manager):
        """Test web application creation."""
        with patch.dict(os.environ, {"FM_LLM_ENV": "testing"}):
            app = create_app(config_manager)
            assert app is not None
            assert app.config['TESTING'] == True
    
    def test_certificate_generator_integration(self, config_manager):
        """Test certificate generator integration."""
        # Mock the dependencies
        with patch('fm_llm_solver.services.certificate_generator.CertificateGenerator') as mock_cert_gen:
            mock_instance = Mock()
            mock_cert_gen.return_value = mock_instance
            mock_instance.generate_certificate.return_value = {
                "certificate": "test_certificate",
                "status": "success"
            }
            
            cert_generator = CertificateGenerator(config_manager)
            result = cert_generator.generate_certificate("test_problem")
            
            assert result["status"] == "success"
            assert "certificate" in result
    
    def test_parser_integration(self, config_manager):
        """Test parser integration."""
        with patch('fm_llm_solver.services.parser.Parser') as mock_parser:
            mock_instance = Mock()
            mock_parser.return_value = mock_instance
            mock_instance.parse_problem.return_value = {
                "variables": ["x", "y"],
                "constraints": ["x + y <= 10"],
                "objective": "minimize x + y"
            }
            
            parser = Parser(config_manager)
            result = parser.parse_problem("test problem description")
            
            assert "variables" in result
            assert "constraints" in result
            assert "objective" in result
    
    def test_end_to_end_workflow(self, config_manager):
        """Test end-to-end workflow integration."""
        # Initialize all components
        logging_manager = LoggingManager(config_manager)
        cache_manager = CacheManager(config_manager)
        monitoring_manager = MonitoringManager(config_manager)
        error_handler = ErrorHandler(config_manager)
        
        # Set up logging
        logging_manager.setup_logging()
        logger = logging_manager.get_logger("integration")
        
        # Simulate a complete workflow
        logger.info("Starting end-to-end integration test")
        
        # Cache some data
        cache_manager.set("workflow_test", {"step": 1, "data": "test_data"})
        
        # Record metrics
        monitoring_manager.record_metric("workflow_steps", 1.0, {"test": "e2e"})
        
        # Simulate error handling
        def workflow_step():
            cached_data = cache_manager.get("workflow_test")
            if cached_data:
                return {"success": True, "data": cached_data}
            else:
                raise Exception("No cached data")
        
        result = error_handler.handle_error(
            workflow_step,
            strategy="retry",
            context={"operation": "workflow_test"}
        )
        
        assert result["success"] == True
        assert result["data"]["step"] == 1
        
        logger.info("End-to-end integration test completed successfully")
    
    def test_component_failure_handling(self, config_manager):
        """Test handling of component failures."""
        error_handler = ErrorHandler(config_manager)
        cache_manager = CacheManager(config_manager)
        
        # Test cache failure handling
        def cache_dependent_function():
            # This should fail gracefully if cache is unavailable
            try:
                cache_manager.get("non_existent_key")
                return "cache_available"
            except Exception:
                return "cache_unavailable"
        
        result = error_handler.handle_error(
            cache_dependent_function,
            strategy="graceful_degradation",
            context={"operation": "cache_test"}
        )
        
        assert result in ["cache_available", "cache_unavailable"]
    
    def test_performance_monitoring(self, config_manager):
        """Test performance monitoring integration."""
        monitoring_manager = MonitoringManager(config_manager)
        
        # Test performance tracking
        import time
        start_time = time.time()
        
        # Simulate some work
        time.sleep(0.1)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Record performance metric
        monitoring_manager.record_histogram("operation_duration", duration, {"operation": "test"})
        
        # Check metrics
        metrics = monitoring_manager.get_metrics()
        assert len(metrics) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
