"""
Utility functions for FM-LLM Solver web interface.

Provides authentication, rate limiting, input validation, and other helper functions.
"""

import re
import secrets
import functools
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable

from flask import request, jsonify, g, current_app, session
from flask_login import current_user

from fm_llm_solver.core.logging import get_logger
from fm_llm_solver.core.exceptions import ValidationError

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
                if hasattr(current_user, 'get_daily_limit'):
                    user_limit = current_user.get_daily_limit()
                    if user_limit > max_requests:
                        max_requests = user_limit
            else:
                client_id = f"ip:{request.remote_addr}"
            
            now = datetime.utcnow()
            window_start = now - timedelta(seconds=window_seconds)
            
            # Clean old entries
            if client_id in _rate_limit_store:
                _rate_limit_store[client_id]['requests'] = [
                    req_time for req_time in _rate_limit_store[client_id]['requests']
                    if req_time > window_start
                ]
            else:
                _rate_limit_store[client_id] = {'requests': []}
            
            # Check rate limit
            current_requests = len(_rate_limit_store[client_id]['requests'])
            if current_requests >= max_requests:
                logger.warning(f"Rate limit exceeded for {client_id}: {current_requests}/{max_requests}")
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'retry_after': window_seconds
                }), 429
            
            # Record request
            _rate_limit_store[client_id]['requests'].append(now)
            
            # Record in user model if authenticated
            if current_user.is_authenticated and hasattr(current_user, 'record_request'):
                current_user.record_request()
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def require_api_key(f: Callable) -> Callable:
    """Decorator to require API key for API endpoints."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        # In production, validate against database
        # For now, check against config
        valid_keys = getattr(current_app.fm_config.security, 'api_keys', [])
        if api_key not in valid_keys:
            logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
            return jsonify({'error': 'Invalid API key'}), 401
        
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
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',                # JavaScript URLs
        r'on\w+\s*=',                 # Event handlers
        r'data:.*?base64',            # Data URLs
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, input_text, re.IGNORECASE | re.DOTALL):
            logger.warning(f"Suspicious input detected: {pattern}")
            raise ValidationError("Input contains suspicious content")
    
    return input_text


def generate_csrf_token() -> str:
    """Generate CSRF token for forms."""
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(16)
    return session['csrf_token']


def validate_csrf_token(token: str) -> bool:
    """Validate CSRF token."""
    expected_token = session.get('csrf_token')
    if not expected_token:
        return False
    return secrets.compare_digest(expected_token, token)


def require_csrf(f: Callable) -> Callable:
    """Decorator to require CSRF token for POST requests."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method == 'POST':
            token = request.form.get('csrf_token') or request.headers.get('X-CSRF-Token')
            if not token or not validate_csrf_token(token):
                logger.warning(f"CSRF token validation failed for {request.endpoint}")
                return jsonify({'error': 'CSRF token invalid'}), 403
        
        return f(*args, **kwargs)
    
    return decorated_function


def admin_required(f: Callable) -> Callable:
    """Decorator to require admin role."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({'error': 'Authentication required'}), 401
        
        if not current_user.is_admin:
            return jsonify({'error': 'Admin access required'}), 403
        
        return f(*args, **kwargs)
    
    return decorated_function


def premium_required(f: Callable) -> Callable:
    """Decorator to require premium or admin role."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({'error': 'Authentication required'}), 401
        
        if not current_user.is_premium:
            return jsonify({'error': 'Premium access required'}), 403
        
        return f(*args, **kwargs)
    
    return decorated_function


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove/replace unsafe characters
    filename = re.sub(r'[^\w\-_\.]', '_', filename)
    # Remove multiple consecutive dots/underscores
    filename = re.sub(r'[_\.]{2,}', '_', filename)
    # Ensure reasonable length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
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
    pattern = r'(\w+)\s*(?:∈|in)\s*\[\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*\]'
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
    error_data = {
        'error': message,
        'status_code': status_code,
        **kwargs
    }
    return jsonify(error_data), status_code


def log_user_action(action: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Log user action for audit trail."""
    user_id = current_user.id if current_user.is_authenticated else None
    ip_address = request.remote_addr
    user_agent = request.headers.get('User-Agent', '')
    
    log_data = {
        'action': action,
        'user_id': user_id,
        'ip_address': ip_address,
        'user_agent': user_agent,
        'timestamp': datetime.utcnow().isoformat(),
        'endpoint': request.endpoint,
        'method': request.method,
        'details': details or {}
    }
    
    logger.info(f"User action: {action}", extra={'audit_log': log_data})


class APIResponse:
    """Standardized API response helper."""
    
    @staticmethod
    def success(data: Any = None, message: str = 'Success') -> Dict[str, Any]:
        """Create success response."""
        response = {
            'success': True,
            'message': message
        }
        if data is not None:
            response['data'] = data
        return response
    
    @staticmethod
    def error(message: str, code: str = 'GENERAL_ERROR', details: Any = None) -> Dict[str, Any]:
        """Create error response."""
        response = {
            'success': False,
            'error': {
                'message': message,
                'code': code
            }
        }
        if details is not None:
            response['error']['details'] = details
        return response
    
    @staticmethod
    def paginated(data: list, page: int, per_page: int, total: int) -> Dict[str, Any]:
        """Create paginated response."""
        return {
            'success': True,
            'data': data,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': (total + per_page - 1) // per_page,
                'has_next': page * per_page < total,
                'has_prev': page > 1
            }
        } 