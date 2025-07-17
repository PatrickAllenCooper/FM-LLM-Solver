"""
Logging configuration for FM-LLM Solver.

Provides structured logging with different levels and formats.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json
from logging.handlers import RotatingFileHandler


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        if hasattr(record, "user_id"):
            log_obj["user_id"] = record.user_id
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        if hasattr(record, "error_code"):
            log_obj["error_code"] = record.error_code

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        return super().format(record)


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter for adding context to logs."""

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process log message and add context."""
        extra = kwargs.get("extra", {})

        # Add context from adapter
        for key, value in self.extra.items():
            if key not in extra:
                extra[key] = value

        kwargs["extra"] = extra
        return msg, kwargs


def configure_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    console: bool = True,
    structured: bool = False,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (None for no file logging)
        console: Whether to log to console
        structured: Use structured (JSON) logging
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
    """
    # Create logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))

        if structured:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_format = ColoredFormatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(console_format)

        root_logger.addHandler(console_handler)

    # File handler
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Main log file
        log_file = log_path / f"fm_llm_solver_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file

        if structured:
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_format = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_format)

        root_logger.addHandler(file_handler)

        # Error log file
        error_file = log_path / f"fm_llm_solver_errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = RotatingFileHandler(
            error_file, maxBytes=max_bytes, backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(error_handler)

    # Suppress noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    # Log configuration
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging configured: level={level}, console={console}, "
        f"structured={structured}, log_dir={log_dir}"
    )


def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Get a logger instance with optional context.

    Args:
        name: Logger name (usually __name__)
        context: Optional context to add to all logs

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    if context:
        return LoggerAdapter(logger, context)

    return logger


def log_performance(logger: logging.Logger, operation: str):
    """
    Decorator for logging operation performance.

    Args:
        logger: Logger instance
        operation: Operation name
    """
    import functools
    import time

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time

                logger.info(
                    f"{operation} completed",
                    extra={
                        "operation": operation,
                        "duration_seconds": elapsed,
                        "status": "success",
                    },
                )

                return result

            except Exception as e:
                elapsed = time.time() - start_time

                logger.error(
                    f"{operation} failed",
                    extra={
                        "operation": operation,
                        "duration_seconds": elapsed,
                        "status": "failed",
                        "error": str(e),
                    },
                    exc_info=True,
                )

                raise

        return wrapper

    return decorator


def log_api_request(logger: logging.Logger):
    """
    Decorator for logging API requests.

    Args:
        logger: Logger instance
    """
    import functools
    from flask import request, g

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Log request
            logger.info(
                f"API request: {request.method} {request.path}",
                extra={
                    "method": request.method,
                    "path": request.path,
                    "ip": request.remote_addr,
                    "user_id": getattr(g, "user_id", None),
                    "request_id": getattr(g, "request_id", None),
                },
            )

            try:
                result = func(*args, **kwargs)

                # Log response
                logger.info(
                    f"API response: {request.method} {request.path}",
                    extra={
                        "method": request.method,
                        "path": request.path,
                        "status_code": getattr(result, "status_code", 200),
                        "request_id": getattr(g, "request_id", None),
                    },
                )

                return result

            except Exception as e:
                logger.error(
                    f"API error: {request.method} {request.path}",
                    extra={
                        "method": request.method,
                        "path": request.path,
                        "error": str(e),
                        "request_id": getattr(g, "request_id", None),
                    },
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator
