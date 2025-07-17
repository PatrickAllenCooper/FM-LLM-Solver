"""
Utility functions for FM-LLM Solver web interface.

Provides authentication, rate limiting, input validation, and other helper functions.
"""

import re
import secrets
import functools
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable

from flask import request, jsonify, current_app, session, Flask
from flask_login import current_user
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
import hashlib
import hmac
import bleach
from urllib.parse import urlparse
import ipaddress

from fm_llm_solver.core.logging import get_logger
from fm_llm_solver.core.exceptions import ValidationError, SecurityError, AuthenticationError
from fm_llm_solver.core.cache_manager import get_cache_manager

logger = get_logger(__name__)

# Rate limiting storage (in production, use Redis)
_rate_limit_store: Dict[str, Dict[str, Any]] = {}


def rate_limit(max_requests: int = 50, window_seconds: int = 86400):
    """
    Rate limiting decorator.

    Args:
        max_requests: Maximum requests allowed
        window_seconds: Time window in seconds (default: 24 hours)
    """

    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            # Get client identifier
            if current_user.is_authenticated:
                client_id = f"user:{current_user.id}"
                # Use user's daily limit if available
                if hasattr(current_user, "get_daily_limit"):
                    user_limit = current_user.get_daily_limit()
                    if user_limit > max_requests:
                        max_requests = user_limit
            else:
                client_id = f"ip:{request.remote_addr}"

            now = datetime.utcnow()
            window_start = now - timedelta(seconds=window_seconds)

            # Clean old entries
            if client_id in _rate_limit_store:
                _rate_limit_store[client_id]["requests"] = [
                    req_time
                    for req_time in _rate_limit_store[client_id]["requests"]
                    if req_time > window_start
                ]
            else:
                _rate_limit_store[client_id] = {"requests": []}

            # Check rate limit
            current_requests = len(_rate_limit_store[client_id]["requests"])
            if current_requests >= max_requests:
                logger.warning(
                    f"Rate limit exceeded for {client_id}: {current_requests}/{max_requests}"
                )
                return jsonify({"error": "Rate limit exceeded", "retry_after": window_seconds}), 429

            # Record request
            _rate_limit_store[client_id]["requests"].append(now)

            # Record in user model if authenticated
            if current_user.is_authenticated and hasattr(current_user, "record_request"):
                current_user.record_request()

            return f(*args, **kwargs)

        return decorated_function

    return decorator


def require_api_key(f: Callable) -> Callable:
    """Decorator to require API key for API endpoints."""

    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("X-API-Key") or request.args.get("api_key")

        if not api_key:
            return jsonify({"error": "API key required"}), 401

        # In production, validate against database
        # For now, check against config
        valid_keys = getattr(current_app.fm_config.security, "api_keys", [])
        if api_key not in valid_keys:
            logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
            return jsonify({"error": "Invalid API key"}), 401

        return f(*args, **kwargs)

    return decorated_function


def validate_input(input_text: str, max_length: int = 10000) -> str:
    """
    Validate and sanitize user input.

    Args:
        input_text: Input text to validate
        max_length: Maximum allowed length

    Returns:
        Sanitized input text

    Raises:
        ValidationError: If input is invalid
    """
    if not input_text or not input_text.strip():
        raise ValidationError("Input cannot be empty")

    input_text = input_text.strip()

    if len(input_text) > max_length:
        raise ValidationError(f"Input too long (max {max_length} characters)")

    # Check for suspicious content
    suspicious_patterns = [
        r"<script[^>]*>.*?</script>",  # Script tags
        r"javascript:",  # JavaScript URLs
        r"on\w+\s*=",  # Event handlers
        r"data:.*?base64",  # Data URLs
    ]

    for pattern in suspicious_patterns:
        if re.search(pattern, input_text, re.IGNORECASE | re.DOTALL):
            logger.warning(f"Suspicious input detected: {pattern}")
            raise ValidationError("Input contains suspicious content")

    return input_text


def generate_csrf_token() -> str:
    """Generate CSRF token for forms."""
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(16)
    return session["csrf_token"]


def validate_csrf_token(token: str) -> bool:
    """Validate CSRF token."""
    expected_token = session.get("csrf_token")
    if not expected_token:
        return False
    return secrets.compare_digest(expected_token, token)


def require_csrf(f: Callable) -> Callable:
    """Decorator to require CSRF token for POST requests."""

    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method == "POST":
            token = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
            if not token or not validate_csrf_token(token):
                logger.warning(f"CSRF token validation failed for {request.endpoint}")
                return jsonify({"error": "CSRF token invalid"}), 403

        return f(*args, **kwargs)

    return decorated_function


def admin_required(f: Callable) -> Callable:
    """Decorator to require admin role."""

    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({"error": "Authentication required"}), 401

        if not current_user.is_admin:
            return jsonify({"error": "Admin access required"}), 403

        return f(*args, **kwargs)

    return decorated_function


def premium_required(f: Callable) -> Callable:
    """Decorator to require premium or admin role."""

    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({"error": "Authentication required"}), 401

        if not current_user.is_premium:
            return jsonify({"error": "Premium access required"}), 403

        return f(*args, **kwargs)

    return decorated_function


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove/replace unsafe characters
    filename = re.sub(r"[^\w\-_\.]", "_", filename)
    # Remove multiple consecutive dots/underscores
    filename = re.sub(r"[_\.]{2,}", "_", filename)
    # Ensure reasonable length
    if len(filename) > 255:
        name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
        filename = f"{name[:250]}.{ext}" if ext else filename[:255]

    return filename


def format_time_ago(dt: datetime) -> str:
    """Format datetime as 'time ago' string."""
    now = datetime.utcnow()
    diff = now - dt

    if diff.days > 0:
        return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    else:
        return "Just now"


def parse_domain_bounds(bounds_str: str) -> Optional[Dict[str, tuple]]:
    """
    Parse domain bounds from string format.

    Expected format: "x ∈ [-2, 2], y ∈ [-1, 1]" or "x in [-2, 2], y in [-1, 1]"
    """
    if not bounds_str:
        return None

    bounds = {}

    # Pattern to match: variable ∈ [min, max] or variable in [min, max]
    pattern = r"(\w+)\s*(?:∈|in)\s*\[\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*\]"
    matches = re.findall(pattern, bounds_str)

    for var, min_val, max_val in matches:
        try:
            bounds[var] = (float(min_val), float(max_val))
        except ValueError:
            logger.warning(f"Failed to parse bounds for variable {var}: {min_val}, {max_val}")
            continue

    return bounds if bounds else None


def jsonify_error(message: str, status_code: int = 400, **kwargs) -> tuple:
    """Create standardized JSON error response."""
    error_data = {"error": message, "status_code": status_code, **kwargs}
    return jsonify(error_data), status_code


def log_user_action(action: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Log user action for audit trail."""
    user_id = current_user.id if current_user.is_authenticated else None
    ip_address = request.remote_addr
    user_agent = request.headers.get("User-Agent", "")

    log_data = {
        "action": action,
        "user_id": user_id,
        "ip_address": ip_address,
        "user_agent": user_agent,
        "timestamp": datetime.utcnow().isoformat(),
        "endpoint": request.endpoint,
        "method": request.method,
        "details": details or {},
    }

    logger.info(f"User action: {action}", extra={"audit_log": log_data})


class APIResponse:
    """Standardized API response helper."""

    @staticmethod
    def success(data: Any = None, message: str = "Success") -> Dict[str, Any]:
        """Create success response."""
        response = {"success": True, "message": message}
        if data is not None:
            response["data"] = data
        return response

    @staticmethod
    def error(message: str, code: str = "GENERAL_ERROR", details: Any = None) -> Dict[str, Any]:
        """Create error response."""
        response = {"success": False, "error": {"message": message, "code": code}}
        if details is not None:
            response["error"]["details"] = details
        return response

    @staticmethod
    def paginated(data: list, page: int, per_page: int, total: int) -> Dict[str, Any]:
        """Create paginated response."""
        return {
            "success": True,
            "data": data,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "pages": (total + per_page - 1) // per_page,
                "has_next": page * per_page < total,
                "has_prev": page > 1,
            },
        }


# Enhanced Security Features


def setup_security_headers(app: Flask) -> None:
    """Setup security headers for the Flask app."""

    @app.after_request
    def set_security_headers(response):
        """Set security headers on all responses."""
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosnif"

        # Enable XSS protection
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Force HTTPS (in production)
        if app.config.get("ENV") == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # Content Security Policy
        csp = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'",  # Adjust as needed
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "font-src 'self'",
            "connect-src 'self'",
            "frame-ancestors 'none'",
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp)

        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions Policy
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

        return response


def setup_rate_limiting(app: Flask) -> Limiter:
    """Setup rate limiting for the Flask app."""
    # Get rate limit configuration
    config = getattr(app, "fm_config", {})
    default_limit = config.get("security", {}).get("rate_limit", {}).get("default", "100/day")

    limiter = Limiter(
        app,
        key_func=get_remote_address,
        default_limits=[default_limit],
        storage_uri=app.config.get("RATELIMIT_STORAGE_URI", "memory://"),
        strategy="fixed-window",
    )

    return limiter


def setup_cors(app: Flask) -> None:
    """Setup CORS for the Flask app."""
    config = getattr(app, "fm_config", {})
    cors_config = config.get("web_interface", {}).get("cors_origins", ["http://localhost:3000"])

    CORS(
        app,
        origins=cors_config,
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Requested-With", "X-CSRF-Token"],
        supports_credentials=True,
    )


def validate_ip_address(ip_address: str) -> bool:
    """Validate if IP address is allowed."""
    try:
        ip = ipaddress.ip_address(ip_address)

        # Block private networks in production (adjust as needed)
        if current_app.config.get("ENV") == "production":
            if ip.is_private or ip.is_loopback:
                return False

        # Check against blocklist (implement as needed)
        # This could check against a database or external service

        return True
    except ValueError:
        return False


def sanitize_html_content(content: str, allowed_tags: Optional[list] = None) -> str:
    """Sanitize HTML content to prevent XSS."""
    if allowed_tags is None:
        allowed_tags = [
            "p",
            "br",
            "strong",
            "em",
            "u",
            "ol",
            "ul",
            "li",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        ]

    return bleach.clean(content, tags=allowed_tags, attributes={}, strip=True)


def validate_json_input(data: dict, required_fields: list, max_depth: int = 5) -> dict:
    """Validate JSON input with security checks."""

    def check_depth(obj, current_depth=0):
        """Check JSON depth to prevent attacks."""
        if current_depth > max_depth:
            raise ValidationError(f"JSON depth exceeds maximum allowed ({max_depth})")

        if isinstance(obj, dict):
            for value in obj.values():
                check_depth(value, current_depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                check_depth(item, current_depth + 1)

    # Check depth
    check_depth(data)

    # Check required fields
    for field in required_fields:
        if field not in data:
            raise ValidationError(f"Required field missing: {field}")

    # Validate field types and content
    for key, value in data.items():
        if isinstance(value, str):
            # Check for suspicious content
            data[key] = validate_input(value)
        elif isinstance(value, dict):
            data[key] = validate_json_input(value, [], max_depth - 1)

    return data


def generate_api_key(user_id: int, scope: str = "general") -> str:
    """Generate secure API key for user."""
    secret = current_app.config.get("SECRET_KEY", "fallback-secret")
    timestamp = str(int(datetime.utcnow().timestamp()))

    # Create payload
    payload = f"{user_id}:{scope}:{timestamp}"

    # Generate HMAC
    signature = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

    # Combine payload and signature
    api_key = f"{payload}:{signature}"

    # Encode to make it more opaque
    import base64

    return base64.b64encode(api_key.encode()).decode()


def validate_api_key(api_key: str) -> Optional[dict]:
    """Validate API key and return user info."""
    try:
        import base64

        decoded = base64.b64decode(api_key.encode()).decode()
        parts = decoded.split(":")

        if len(parts) != 4:
            return None

        user_id, scope, timestamp, signature = parts

        # Verify signature
        secret = current_app.config.get("SECRET_KEY", "fallback-secret")
        payload = f"{user_id}:{scope}:{timestamp}"
        expected_signature = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

        if not hmac.compare_digest(signature, expected_signature):
            return None

        # Check if key is not too old (e.g., 1 year)
        key_age = datetime.utcnow().timestamp() - float(timestamp)
        if key_age > 365 * 24 * 3600:  # 1 year
            return None

        return {
            "user_id": int(user_id),
            "scope": scope,
            "created_at": datetime.fromtimestamp(float(timestamp)),
        }

    except Exception as e:
        logger.warning(f"API key validation failed: {e}")
        return None


def require_valid_origin(f: Callable) -> Callable:
    """Decorator to validate request origin."""

    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        origin = request.headers.get("Origin")
        referer = request.headers.get("Referer")

        # Get allowed origins from config
        config = getattr(current_app, "fm_config", {})
        allowed_origins = config.get("web_interface", {}).get("cors_origins", [])

        # Check origin
        if origin:
            if origin not in allowed_origins:
                logger.warning(f"Request from disallowed origin: {origin}")
                raise SecurityError("Invalid origin", violation_type="origin_validation")

        # Check referer for additional validation
        if referer:
            parsed_referer = urlparse(referer)
            referer_origin = f"{parsed_referer.scheme}://{parsed_referer.netloc}"
            if referer_origin not in allowed_origins:
                logger.warning(f"Request from disallowed referer: {referer}")

        return f(*args, **kwargs)

    return decorated_function


def detect_suspicious_activity(
    user_id: Optional[int] = None, ip_address: Optional[str] = None
) -> bool:
    """Detect suspicious activity patterns."""
    cache = get_cache_manager()

    # Track request patterns
    if user_id:
        key = f"activity:user:{user_id}"
    else:
        key = f"activity:ip:{ip_address or request.remote_addr}"

    # Get recent activity
    activity = cache.get(key) or {"requests": [], "suspicious_count": 0}

    current_time = datetime.utcnow().timestamp()

    # Clean old activity (last hour)
    activity["requests"] = [
        req_time for req_time in activity["requests"] if current_time - req_time < 3600
    ]

    # Add current request
    activity["requests"].append(current_time)

    # Check for suspicious patterns
    suspicious = False

    # Too many requests in short time
    if len(activity["requests"]) > 100:  # 100 requests per hour
        suspicious = True
        activity["suspicious_count"] += 1
        logger.warning(f"High request rate detected for {key}")

    # Burst detection (more than 10 requests in 1 minute)
    recent_requests = [
        req_time for req_time in activity["requests"] if current_time - req_time < 60
    ]
    if len(recent_requests) > 10:
        suspicious = True
        activity["suspicious_count"] += 1
        logger.warning(f"Request burst detected for {key}")

    # Update cache
    cache.set(key, activity, ttl=3600)

    return suspicious


def log_security_event(event_type: str, severity: str = "medium", **kwargs) -> None:
    """Log security-related events."""
    event_data = {
        "event_type": event_type,
        "severity": severity,
        "ip_address": get_client_ip(),
        "user_agent": request.headers.get("User-Agent", ""),
        "endpoint": request.endpoint,
        "method": request.method,
        "timestamp": datetime.utcnow().isoformat(),
        **kwargs,
    }

    if current_user.is_authenticated:
        event_data["user_id"] = current_user.id

    logger.warning(f"Security event: {event_type}", extra={"security_event": event_data})


def get_client_ip() -> str:
    """Get the real client IP address."""
    # Check for forwarded headers (be careful with these in production)
    forwarded_headers = [
        "X-Forwarded-For",
        "X-Real-IP",
        "CF-Connecting-IP",  # Cloudflare
        "X-Forwarded",
    ]

    for header in forwarded_headers:
        if header in request.headers:
            # Take the first IP if multiple are present
            ip = request.headers[header].split(",")[0].strip()
            if validate_ip_address(ip):
                return ip

    return request.remote_addr or "127.0.0.1"


def handle_error_response(error: Exception) -> tuple:
    """Handle errors and return appropriate response."""
    if isinstance(error, ValidationError):
        return jsonify(APIResponse.error(str(error), "VALIDATION_ERROR")), 400
    elif isinstance(error, AuthenticationError):
        return jsonify(APIResponse.error(str(error), "AUTHENTICATION_ERROR")), 401
    elif isinstance(error, SecurityError):
        log_security_event("security_violation", severity="high", error=str(error))
        return jsonify(APIResponse.error("Security violation detected", "SECURITY_ERROR")), 403
    else:
        logger.error(f"Unhandled error: {error}")
        return jsonify(APIResponse.error("Internal server error", "INTERNAL_ERROR")), 500


def encrypt_sensitive_data(data: str, key: Optional[str] = None) -> str:
    """Encrypt sensitive data for storage."""
    from cryptography.fernet import Fernet
    import base64

    if key is None:
        key = current_app.config.get("ENCRYPTION_KEY")
        if not key:
            # Generate a key from the secret key
            import hashlib

            secret = current_app.config.get("SECRET_KEY", "fallback-secret")
            key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())

    if isinstance(key, str):
        key = key.encode()

    fernet = Fernet(key)
    encrypted = fernet.encrypt(data.encode())
    return base64.b64encode(encrypted).decode()


def decrypt_sensitive_data(encrypted_data: str, key: Optional[str] = None) -> str:
    """Decrypt sensitive data."""
    from cryptography.fernet import Fernet
    import base64

    if key is None:
        key = current_app.config.get("ENCRYPTION_KEY")
        if not key:
            import hashlib

            secret = current_app.config.get("SECRET_KEY", "fallback-secret")
            key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())

    if isinstance(key, str):
        key = key.encode()

    fernet = Fernet(key)
    encrypted = base64.b64decode(encrypted_data.encode())
    decrypted = fernet.decrypt(encrypted)
    return decrypted.decode()
