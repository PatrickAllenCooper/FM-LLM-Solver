"""
Comprehensive error handling system for FM-LLM-Solver.

Provides error recovery, graceful degradation, retry mechanisms, and structured
error reporting throughout the system.
"""

import functools
import time
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Type, Generator
from dataclasses import dataclass, field
from enum import Enum

from .exceptions import (
    FMLLMSolverError,
    RetryableError,
    NonRetryableError,
)
from .logging_manager import get_logger


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Error recovery strategies."""

    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"
    FAIL_FAST = "fail_fast"


@dataclass
class ErrorContext:
    """Context information for error handling."""

    operation: str
    component: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: List[Type[Exception]] = field(default_factory=list)
    non_retryable_exceptions: List[Type[Exception]] = field(default_factory=list)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: Type[Exception] = Exception


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker implementation for error handling."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.next_attempt_time = 0
        self.logger = get_logger(__name__)

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == CircuitBreakerState.CLOSED:
            return True

        if self.state == CircuitBreakerState.OPEN:
            if time.time() >= self.next_attempt_time:
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            return False

        # Half-open state
        return True

    def on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
        self.logger.info("Circuit breaker reset to closed state")

    def on_failure(self, exception: Exception):
        """Handle failed execution."""
        if isinstance(exception, self.config.expected_exception):
            self.failure_count += 1

            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self.next_attempt_time = time.time() + self.config.recovery_timeout
                self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                self.next_attempt_time = time.time() + self.config.recovery_timeout
                self.logger.warning("Circuit breaker returned to open state")


class ErrorHandler:
    """
    Comprehensive error handling system.

    Features:
    - Retry mechanisms with exponential backoff
    - Circuit breaker pattern
    - Graceful degradation
    - Error recovery strategies
    - Structured error reporting
    - Context-aware error handling
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        self.error_counts: Dict[str, int] = {}
        self.last_errors: Dict[str, Exception] = {}

    def register_fallback(self, operation: str, handler: Callable):
        """Register a fallback handler for an operation."""
        self.fallback_handlers[operation] = handler
        self.logger.info(f"Registered fallback handler for {operation}")

    def register_circuit_breaker(self, operation: str, config: CircuitBreakerConfig):
        """Register a circuit breaker for an operation."""
        self.circuit_breakers[operation] = CircuitBreaker(config)
        self.logger.info(f"Registered circuit breaker for {operation}")

    def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    ) -> Dict[str, Any]:
        """
        Handle an error with the specified strategy.

        Args:
            error: The exception that occurred
            context: Context information
            strategy: Recovery strategy to use
            severity: Error severity level

        Returns:
            Error handling result
        """
        error_id = f"{context.component}:{context.operation}"

        # Track error occurrence
        self.error_counts[error_id] = self.error_counts.get(error_id, 0) + 1
        self.last_errors[error_id] = error

        # Log the error
        self._log_error(error, context, severity)

        # Determine handling strategy
        if isinstance(error, NonRetryableError):
            strategy = RecoveryStrategy.FAIL_FAST
        elif isinstance(error, RetryableError):
            strategy = RecoveryStrategy.RETRY

        # Execute strategy
        result = self._execute_strategy(error, context, strategy)

        # Report to monitoring if critical
        if severity == ErrorSeverity.CRITICAL:
            self._report_critical_error(error, context)

        return result

    def _log_error(self, error: Exception, context: ErrorContext, severity: ErrorSeverity):
        """Log error with context information."""
        log_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "operation": context.operation,
            "component": context.component,
            "severity": severity.value,
            "user_id": context.user_id,
            "session_id": context.session_id,
            "request_id": context.request_id,
            "metadata": context.metadata,
            "traceback": traceback.format_exc(),
        }

        if isinstance(error, FMLLMSolverError):
            log_data.update(error.details)

        # Log at appropriate level
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical("Critical error occurred", extra=log_data)
        elif severity == ErrorSeverity.HIGH:
            self.logger.error("High severity error occurred", extra=log_data)
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning("Medium severity error occurred", extra=log_data)
        else:
            self.logger.info("Low severity error occurred", extra=log_data)

    def _execute_strategy(
        self, error: Exception, context: ErrorContext, strategy: RecoveryStrategy
    ) -> Dict[str, Any]:
        """Execute the specified recovery strategy."""
        operation_key = f"{context.component}:{context.operation}"

        if strategy == RecoveryStrategy.RETRY:
            return {"action": "retry", "retry_recommended": True}

        elif strategy == RecoveryStrategy.FALLBACK:
            if operation_key in self.fallback_handlers:
                try:
                    fallback_result = self.fallback_handlers[operation_key]()
                    return {"action": "fallback", "success": True, "result": fallback_result}
                except Exception as fallback_error:
                    self.logger.error(f"Fallback failed for {operation_key}: {fallback_error}")
                    return {"action": "fallback", "success": False}
            else:
                return {"action": "fallback", "success": False, "reason": "no_handler"}

        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return self._graceful_degradation(error, context)

        elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            if operation_key in self.circuit_breakers:
                circuit_breaker = self.circuit_breakers[operation_key]
                circuit_breaker.on_failure(error)
                return {"action": "circuit_breaker", "state": circuit_breaker.state.value}
            else:
                return {"action": "circuit_breaker", "success": False, "reason": "no_breaker"}

        elif strategy == RecoveryStrategy.FAIL_FAST:
            return {"action": "fail_fast", "should_propagate": True}

        return {"action": "unknown"}

    def _graceful_degradation(self, error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Implement graceful degradation."""
        # Default graceful degradation strategies
        if "model" in context.component.lower():
            return {
                "action": "graceful_degradation",
                "degraded_service": "basic_response",
                "message": "AI model unavailable, providing basic response",
            }

        elif "knowledge_base" in context.component.lower():
            return {
                "action": "graceful_degradation",
                "degraded_service": "no_rag",
                "message": "Knowledge base unavailable, proceeding without RAG",
            }

        elif "verification" in context.component.lower():
            return {
                "action": "graceful_degradation",
                "degraded_service": "basic_verification",
                "message": "Advanced verification unavailable, using basic checks",
            }

        return {
            "action": "graceful_degradation",
            "degraded_service": "minimal",
            "message": "Service degraded, providing minimal functionality",
        }

    def _report_critical_error(self, error: Exception, context: ErrorContext):
        """Report critical errors to monitoring systems."""
        # This would integrate with monitoring systems like Sentry, DataDog, etc.
        self.logger.critical(
            f"CRITICAL ERROR REPORTED: {type(error).__name__} in {context.component}",
            extra={
                "alert": True,
                "component": context.component,
                "operation": context.operation,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_counts": self.error_counts.copy(),
            "circuit_breaker_states": {
                op: cb.state.value for op, cb in self.circuit_breakers.items()
            },
            "last_errors": {
                op: {"type": type(error).__name__, "message": str(error)}
                for op, error in self.last_errors.items()
            },
        }

    def reset_error_counts(self):
        """Reset error counts."""
        self.error_counts.clear()
        self.last_errors.clear()
        self.logger.info("Error counts reset")


def with_error_handling(
    operation: str,
    component: str,
    strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    context_data: Optional[Dict[str, Any]] = None,
):
    """
    Decorator for automatic error handling.

    Args:
        operation: Operation name
        component: Component name
        strategy: Recovery strategy
        severity: Error severity
        context_data: Additional context data
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = get_error_handler()
            context = ErrorContext(
                operation=operation, component=component, metadata=context_data or {}
            )

            try:
                return func(*args, **kwargs)
            except Exception as e:
                result = error_handler.handle_error(e, context, strategy, severity)

                if result.get("should_propagate", False):
                    raise

                if result.get("action") == "fallback" and result.get("success"):
                    return result["result"]

                # For other strategies, re-raise the exception
                raise

        return wrapper

    return decorator


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[List[Type[Exception]]] = None,
    non_retryable_exceptions: Optional[List[Type[Exception]]] = None,
):
    """
    Decorator for automatic retry with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        jitter: Whether to add jitter to delays
        retryable_exceptions: Exceptions that should be retried
        non_retryable_exceptions: Exceptions that should not be retried
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                jitter=jitter,
                retryable_exceptions=retryable_exceptions or [RetryableError],
                non_retryable_exceptions=non_retryable_exceptions or [NonRetryableError],
            )

            return _retry_with_config(func, config, *args, **kwargs)

        return wrapper

    return decorator


def _retry_with_config(func: Callable, config: RetryConfig, *args, **kwargs):
    """Execute function with retry logic."""
    logger = get_logger(__name__)

    for attempt in range(config.max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Check if exception should not be retried
            if config.non_retryable_exceptions and any(
                isinstance(e, exc_type) for exc_type in config.non_retryable_exceptions
            ):
                logger.info(f"Non-retryable exception {type(e).__name__}, not retrying")
                raise

            # Check if exception should be retried
            if config.retryable_exceptions and not any(
                isinstance(e, exc_type) for exc_type in config.retryable_exceptions
            ):
                logger.info(f"Exception {type(e).__name__} not in retryable list, not retrying")
                raise

            if attempt == config.max_retries:
                logger.error(f"Max retries ({config.max_retries}) exceeded for {func.__name__}")
                raise

            # Calculate delay
            delay = min(config.base_delay * (config.exponential_base**attempt), config.max_delay)

            if config.jitter:
                import random

                delay *= 0.5 + random.random() * 0.5  # Add 0-50% jitter

            logger.info(
                f"Retry {attempt + 1}/{config.max_retries} for {func.__name__} "
                f"after {delay:.2f}s delay. Error: {e}"
            )

            time.sleep(delay)

    # This should never be reached
    raise RuntimeError("Retry logic error")


@contextmanager
def error_boundary(
    operation: str, component: str, fallback_value: Any = None, suppress_errors: bool = False
) -> Generator[ErrorContext, None, None]:
    """
    Context manager for error boundary handling.

    Args:
        operation: Operation name
        component: Component name
        fallback_value: Value to return if error occurs
        suppress_errors: Whether to suppress errors
    """
    error_handler = get_error_handler()
    context = ErrorContext(operation=operation, component=component)

    try:
        yield context
    except Exception as e:
        result = error_handler.handle_error(e, context, RecoveryStrategy.GRACEFUL_DEGRADATION)

        if suppress_errors:
            return fallback_value

        if result.get("action") == "graceful_degradation":
            logger = get_logger(__name__)
            logger.warning(f"Error boundary activated for {operation}: {result.get('message')}")
            return fallback_value

        raise


# Global error handler instance
_error_handler = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def configure_error_handling(
    fallback_handlers: Optional[Dict[str, Callable]] = None,
    circuit_breakers: Optional[Dict[str, CircuitBreakerConfig]] = None,
):
    """Configure global error handling settings."""
    error_handler = get_error_handler()

    if fallback_handlers:
        for operation, handler in fallback_handlers.items():
            error_handler.register_fallback(operation, handler)

    if circuit_breakers:
        for operation, config in circuit_breakers.items():
            error_handler.register_circuit_breaker(operation, config)
