"""
Comprehensive logging manager for FM-LLM-Solver.

Provides structured logging with JSON output, proper log levels, and integration
with log aggregation systems like ELK stack or Prometheus.
"""

import json
import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .config_manager import ConfigurationManager


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process': record.process,
            'thread': record.thread,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add extra fields if present
        if self.include_extra and hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        # Add any custom attributes
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'getMessage',
                          'extra']:
                log_data[key] = value
        
        return json.dumps(log_data, default=str)


class LoggingManager:
    """
    Comprehensive logging manager that provides structured logging with JSON output.
    
    Features:
    - JSON formatted logs for machine parsing
    - Multiple handlers (console, file, rotating file)
    - Configurable log levels per component
    - Integration with monitoring systems
    - Performance logging capabilities
    - Security audit logging
    """
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        self.config_manager = config_manager or ConfigurationManager()
        self.loggers: Dict[str, logging.Logger] = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging_config = self.config_manager.get_logging_config()
        
        # Create logs directory
        log_dir = Path(logging_config.get('log_directory', 'logs'))
        log_dir.mkdir(exist_ok=True)
        
        # Set root logger level
        root_level = logging_config.get('root_level', 'INFO')
        logging.getLogger().setLevel(getattr(logging, root_level))
        
        # Configure specific loggers
        logger_configs = logging_config.get('loggers', {})
        for logger_name, config in logger_configs.items():
            self._configure_logger(logger_name, config, log_dir)
    
    def _configure_logger(self, name: str, config: Dict[str, Any], log_dir: Path):
        """Configure a specific logger."""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, config.get('level', 'INFO')))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Add handlers based on configuration
        handlers = config.get('handlers', ['console'])
        
        for handler_type in handlers:
            handler = self._create_handler(handler_type, config, log_dir, name)
            if handler:
                logger.addHandler(handler)
        
        # Prevent propagation to root logger if specified
        logger.propagate = config.get('propagate', False)
        
        self.loggers[name] = logger
    
    def _create_handler(self, handler_type: str, config: Dict[str, Any], 
                       log_dir: Path, logger_name: str) -> Optional[logging.Handler]:
        """Create a logging handler."""
        handler = None
        
        if handler_type == 'console':
            handler = logging.StreamHandler(sys.stdout)
            if config.get('json_format', True):
                handler.setFormatter(JSONFormatter())
            else:
                handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
        
        elif handler_type == 'file':
            log_file = log_dir / f"{logger_name}.log"
            handler = logging.FileHandler(log_file)
            handler.setFormatter(JSONFormatter())
        
        elif handler_type == 'rotating_file':
            log_file = log_dir / f"{logger_name}.log"
            max_bytes = config.get('max_file_size', 10 * 1024 * 1024)  # 10MB
            backup_count = config.get('backup_count', 5)
            handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
            handler.setFormatter(JSONFormatter())
        
        elif handler_type == 'syslog':
            handler = logging.handlers.SysLogHandler()
            handler.setFormatter(JSONFormatter())
        
        if handler:
            handler.setLevel(getattr(logging, config.get('level', 'INFO')))
        
        return handler
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger by name."""
        if name not in self.loggers:
            # Create default logger configuration
            config = {
                'level': 'INFO',
                'handlers': ['console', 'rotating_file'],
                'json_format': True,
                'propagate': False
            }
            log_dir = Path(self.config_manager.get_logging_config().get('log_directory', 'logs'))
            log_dir.mkdir(exist_ok=True)
            self._configure_logger(name, config, log_dir)
        
        return self.loggers[name]
    
    def log_performance(self, logger_name: str, operation: str, 
                       duration: float, metadata: Optional[Dict[str, Any]] = None):
        """Log performance metrics."""
        logger = self.get_logger(logger_name)
        
        perf_data = {
            'operation': operation,
            'duration_seconds': duration,
            'performance_log': True
        }
        
        if metadata:
            perf_data.update(metadata)
        
        logger.info(f"Performance: {operation} completed in {duration:.3f}s", extra=perf_data)
    
    def log_security_event(self, event_type: str, user_id: Optional[str] = None,
                          ip_address: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Log security-related events."""
        logger = self.get_logger('security')
        
        security_data = {
            'event_type': event_type,
            'user_id': user_id,
            'ip_address': ip_address,
            'security_log': True
        }
        
        if details:
            security_data.update(details)
        
        logger.warning(f"Security event: {event_type}", extra=security_data)
    
    def log_model_operation(self, model_name: str, operation: str, 
                           success: bool, duration: Optional[float] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        """Log model operations."""
        logger = self.get_logger('model_operations')
        
        model_data = {
            'model_name': model_name,
            'operation': operation,
            'success': success,
            'model_operation_log': True
        }
        
        if duration is not None:
            model_data['duration_seconds'] = duration
        
        if metadata:
            model_data.update(metadata)
        
        level = logging.INFO if success else logging.ERROR
        message = f"Model {operation}: {'success' if success else 'failed'}"
        logger.log(level, message, extra=model_data)
    
    def log_api_request(self, endpoint: str, method: str, status_code: int,
                       duration: float, user_id: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None):
        """Log API requests."""
        logger = self.get_logger('api')
        
        api_data = {
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'duration_seconds': duration,
            'user_id': user_id,
            'api_log': True
        }
        
        if metadata:
            api_data.update(metadata)
        
        logger.info(f"API {method} {endpoint} - {status_code}", extra=api_data)
    
    def create_context_logger(self, name: str, context: Dict[str, Any]) -> 'ContextLogger':
        """Create a context-aware logger."""
        return ContextLogger(self.get_logger(name), context)


class ContextLogger:
    """Logger that automatically includes context information."""
    
    def __init__(self, logger: logging.Logger, context: Dict[str, Any]):
        self.logger = logger
        self.context = context
    
    def _log_with_context(self, level: int, message: str, *args, **kwargs):
        """Log with context information."""
        extra = kwargs.get('extra', {})
        extra.update(self.context)
        kwargs['extra'] = extra
        self.logger.log(level, message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs):
        self._log_with_context(logging.DEBUG, message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        self._log_with_context(logging.INFO, message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        self._log_with_context(logging.WARNING, message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        self._log_with_context(logging.ERROR, message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        self._log_with_context(logging.CRITICAL, message, *args, **kwargs)


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, logging_manager: LoggingManager, logger_name: str, 
                 operation: str, metadata: Optional[Dict[str, Any]] = None):
        self.logging_manager = logging_manager
        self.logger_name = logger_name
        self.operation = operation
        self.metadata = metadata or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            success = exc_type is None
            
            metadata = self.metadata.copy()
            metadata['success'] = success
            
            if exc_type:
                metadata['exception_type'] = exc_type.__name__
                metadata['exception_message'] = str(exc_val)
            
            self.logging_manager.log_performance(
                self.logger_name, self.operation, duration, metadata
            )


# Global logging manager instance
_logging_manager = None


def get_logging_manager() -> LoggingManager:
    """Get the global logging manager instance."""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager()
    return _logging_manager


def get_logger(name: str) -> logging.Logger:
    """Get a logger by name."""
    return get_logging_manager().get_logger(name)


def log_performance(logger_name: str, operation: str, duration: float, 
                   metadata: Optional[Dict[str, Any]] = None):
    """Log performance metrics."""
    return get_logging_manager().log_performance(logger_name, operation, duration, metadata)


def timer(logger_name: str, operation: str, metadata: Optional[Dict[str, Any]] = None):
    """Performance timer decorator/context manager."""
    return PerformanceTimer(get_logging_manager(), logger_name, operation, metadata) 