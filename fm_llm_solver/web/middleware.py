"""
Middleware for FM-LLM Solver web interface.

Handles request logging, error handling, and security headers.
"""

import time
import uuid
from functools import wraps

from flask import Flask, g, jsonify, request
from werkzeug.exceptions import HTTPException

from fm_llm_solver.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    FMLLMSolverError,
    RateLimitError,
    ValidationError,
)
from fm_llm_solver.core.logging import get_logger


def setup_request_logging(app: Flask) -> None:
    """Setup request logging middleware."""
    logger = get_logger(__name__)

    @app.before_request
    def log_request():
        """Log incoming requests."""
        # Generate request ID
        g.request_id = str(uuid.uuid4())
        g.start_time = time.time()

        # Log request
        logger.info(
            f"Request started: {request.method} {request.path}",
            extra={
                "request_id": g.request_id,
                "method": request.method,
                "path": request.path,
                "remote_addr": request.remote_addr,
                "user_agent": request.headers.get("User-Agent", "Unknown"),
            },
        )

    @app.after_request
    def log_response(response):
        """Log outgoing responses."""
        if hasattr(g, "start_time"):
            elapsed = time.time() - g.start_time

            logger.info(
                f"Request completed: {request.method} {request.path}",
                extra={
                    "request_id": getattr(g, "request_id", "unknown"),
                    "method": request.method,
                    "path": request.path,
                    "status_code": response.status_code,
                    "duration_seconds": elapsed,
                },
            )

        # Add request ID to response headers
        if hasattr(g, "request_id"):
            response.headers["X-Request-ID"] = g.request_id

        return response


def setup_error_handlers(app: Flask) -> None:
    """Setup error handling middleware."""
    logger = get_logger(__name__)

    @app.errorhandler(ValidationError)
    def handle_validation_error(e: ValidationError):
        """Handle validation errors."""
        logger.warning(
            f"Validation error: {e}",
            extra={"request_id": getattr(g, "request_id", None)},
        )
        return jsonify(e.to_dict()), 400

    @app.errorhandler(AuthenticationError)
    def handle_authentication_error(e: AuthenticationError):
        """Handle authentication errors."""
        logger.warning(
            f"Authentication error: {e}",
            extra={"request_id": getattr(g, "request_id", None)},
        )
        return jsonify(e.to_dict()), 401

    @app.errorhandler(AuthorizationError)
    def handle_authorization_error(e: AuthorizationError):
        """Handle authorization errors."""
        logger.warning(
            f"Authorization error: {e}",
            extra={"request_id": getattr(g, "request_id", None)},
        )
        return jsonify(e.to_dict()), 403

    @app.errorhandler(RateLimitError)
    def handle_rate_limit_error(e: RateLimitError):
        """Handle rate limit errors."""
        logger.warning(
            f"Rate limit error: {e}",
            extra={"request_id": getattr(g, "request_id", None)},
        )
        return jsonify(e.to_dict()), 429

    @app.errorhandler(FMLLMSolverError)
    def handle_application_error(e: FMLLMSolverError):
        """Handle general application errors."""
        logger.error(
            f"Application error: {e}",
            extra={"request_id": getattr(g, "request_id", None)},
        )
        return jsonify(e.to_dict()), 500

    @app.errorhandler(HTTPException)
    def handle_http_exception(e: HTTPException):
        """Handle HTTP exceptions."""
        logger.warning(
            f"HTTP exception: {e}", extra={"request_id": getattr(g, "request_id", None)}
        )
        return (
            jsonify({"error": e.name, "message": e.description, "status_code": e.code}),
            e.code,
        )

    @app.errorhandler(Exception)
    def handle_unexpected_error(e: Exception):
        """Handle unexpected errors."""
        logger.error(
            f"Unexpected error: {e}",
            extra={"request_id": getattr(g, "request_id", None)},
            exc_info=True,
        )

        # Don't expose internal errors in production
        if app.debug:
            message = str(e)
        else:
            message = "An unexpected error occurred"

        return (
            jsonify(
                {
                    "error": "InternalServerError",
                    "message": message,
                    "request_id": getattr(g, "request_id", None),
                }
            ),
            500,
        )


def setup_security_headers(app: Flask) -> None:
    """Setup security headers middleware."""

    @app.after_request
    def add_security_headers(response):
        """Add security headers to response."""
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosnif"

        # Enable XSS protection
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "font-src 'self' https://cdn.jsdelivr.net; "
            "img-src 'self' data:; "
            "connect-src 'self'"
        )

        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Feature Policy
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )

        return response


def require_json(f):
    """Decorator to require JSON content type."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            raise ValidationError(
                "Content-Type must be application/json", field="content-type"
            )
        return f(*args, **kwargs)

    return decorated_function


def validate_request(schema):
    """Decorator to validate request data against a schema."""

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # Validate request data
                data = schema(**request.get_json())
                g.validated_data = data
            except Exception as e:
                raise ValidationError(f"Invalid request data: {e}")

            return f(*args, **kwargs)

        return decorated_function

    return decorator


def track_usage(operation: str):
    """Decorator to track API usage."""

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Track start time
            start_time = time.time()

            try:
                # Execute function
                result = f(*args, **kwargs)

                # Log usage
                if hasattr(g, "current_user") and g.current_user:
                    from flask import current_app

                    current_app.monitoring_service.track_usage(
                        user_id=g.current_user.id,
                        operation=operation,
                        duration=time.time() - start_time,
                        success=True,
                    )

                return result

            except Exception as e:
                # Log failed usage
                if hasattr(g, "current_user") and g.current_user:
                    from flask import current_app

                    current_app.monitoring_service.track_usage(
                        user_id=g.current_user.id,
                        operation=operation,
                        duration=time.time() - start_time,
                        success=False,
                        error=str(e),
                    )

                raise

        return decorated_function

    return decorator
