"""
Authentication and security middleware for FM-LLM Solver web interface.
"""

import functools
import time
from datetime import datetime, timedelta
from flask import request, jsonify, redirect, url_for, flash, g, abort
from flask_login import LoginManager, current_user, login_required
from werkzeug.exceptions import TooManyRequests
from web_interface.models import db, User, RateLimitLog, IPBlacklist, SecurityLog
import re
import ipaddress
import hashlib
import secrets

login_manager = LoginManager()

def init_auth(app):
    """Initialize authentication system."""
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
    # Register security middleware
    app.before_request(security_checks)
    app.after_request(security_headers)

def security_checks():
    """Run security checks before each request."""
    # Check IP blacklist
    client_ip = get_client_ip()
    
    # Check if IP is blacklisted
    blocked_ip = IPBlacklist.query.filter_by(
        ip_address=client_ip,
        is_active=True
    ).first()
    
    if blocked_ip and blocked_ip.is_blocked():
        log_security_event('blocked_ip_attempt', severity='medium', 
                          description=f'Blocked IP {client_ip} attempted access')
        abort(403, 'Access denied')
    
    # Store IP in request context
    g.client_ip = client_ip
    g.request_start = time.time()

def security_headers(response):
    """Add security headers to response."""
    # Prevent clickjacking
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    
    # Prevent MIME type sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'
    
    # Enable XSS protection
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    # Content Security Policy
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: https:; "
        "connect-src 'self';"
    )
    
    # Referrer Policy
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    
    # Permissions Policy
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
    
    return response

def rate_limit(max_requests=50, window='day', by='user'):
    """
    Rate limiting decorator.
    
    Args:
        max_requests: Maximum number of requests allowed
        window: Time window ('hour', 'day')
        by: Rate limit by 'user' or 'ip'
    """
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            # Check if user is authenticated
            if by == 'user' and not current_user.is_authenticated:
                return redirect(url_for('auth.login'))
            
            # Get identifier
            if by == 'user':
                identifier = current_user.id
                rate_limiter = current_user
            else:
                identifier = get_client_ip()
                rate_limiter = None
            
            # Check rate limit
            if by == 'user' and current_user.is_authenticated:
                # User-based rate limiting
                if not current_user.check_rate_limit():
                    # Log rate limit violation
                    log_rate_limit_violation(current_user.id, request.endpoint, 
                                           current_user.daily_request_count)
                    
                    if request.is_json:
                        return jsonify({
                            'error': f'Rate limit exceeded. Maximum {current_user.daily_request_limit} requests per day.',
                            'requests_today': current_user.daily_request_count,
                            'limit': current_user.daily_request_limit
                        }), 429
                    else:
                        flash(f'Rate limit exceeded. Maximum {current_user.daily_request_limit} requests per day.', 'error')
                        return redirect(url_for('index'))
                
                # Increment request count
                current_user.increment_request_count()
                db.session.commit()
            
            else:
                # IP-based rate limiting for non-authenticated users
                # Simple in-memory rate limiting (consider using Redis for production)
                ip = get_client_ip()
                key = f"rate_limit:{ip}:{request.endpoint}"
                
                # This is a simplified version - in production, use Redis or similar
                # For now, we'll just allow non-authenticated users limited access
                pass
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator

def require_api_key(f):
    """Decorator to require API key for programmatic access."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        user = User.query.filter_by(api_key=api_key).first()
        if not user or not user.is_active:
            log_security_event('invalid_api_key', severity='medium',
                             description=f'Invalid API key attempt: {api_key[:8]}...')
            return jsonify({'error': 'Invalid API key'}), 401
        
        # Check rate limit
        if not user.check_rate_limit():
            return jsonify({
                'error': f'Rate limit exceeded. Maximum {user.daily_request_limit} requests per day.',
                'requests_today': user.daily_request_count,
                'limit': user.daily_request_limit
            }), 429
        
        # Set current user for the request
        g.current_api_user = user
        
        # Increment request count
        user.increment_request_count()
        db.session.commit()
        
        return f(*args, **kwargs)
    
    return decorated_function

def validate_input(patterns):
    """
    Decorator to validate input against patterns.
    
    Args:
        patterns: Dict of field_name: regex_pattern
    """
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            data = request.get_json() if request.is_json else request.form
            
            for field, pattern in patterns.items():
                value = data.get(field, '')
                if not re.match(pattern, str(value)):
                    log_security_event('invalid_input', severity='low',
                                     description=f'Invalid input for field {field}')
                    return jsonify({'error': f'Invalid input for field: {field}'}), 400
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator

def admin_required(f):
    """Decorator to require admin role."""
    @functools.wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if current_user.role != 'admin':
            abort(403, 'Admin access required')
        return f(*args, **kwargs)
    
    return decorated_function

# Utility functions

def get_client_ip():
    """Get client IP address, considering proxies."""
    if request.environ.get('HTTP_X_FORWARDED_FOR'):
        # Behind proxy
        ip = request.environ['HTTP_X_FORWARDED_FOR'].split(',')[0].strip()
    else:
        ip = request.environ.get('REMOTE_ADDR', '0.0.0.0')
    
    # Validate IP
    try:
        ipaddress.ip_address(ip)
        return ip
    except ValueError:
        return '0.0.0.0'

def log_security_event(event_type, severity='low', description='', user_id=None):
    """Log security-related events."""
    log = SecurityLog(
        event_type=event_type,
        severity=severity,
        description=description,
        user_id=user_id or (current_user.id if current_user.is_authenticated else None),
        username=current_user.username if current_user.is_authenticated else None,
        ip_address=get_client_ip(),
        user_agent=request.headers.get('User-Agent', '')[:500],
        endpoint=request.endpoint
    )
    db.session.add(log)
    db.session.commit()

def log_rate_limit_violation(user_id, endpoint, requests_today):
    """Log rate limit violations."""
    log = RateLimitLog(
        user_id=user_id,
        endpoint=endpoint,
        method=request.method,
        ip_address=get_client_ip(),
        was_blocked=True,
        requests_today=requests_today,
        limit_exceeded_by=requests_today - current_user.daily_request_limit
    )
    db.session.add(log)
    db.session.commit()

def generate_api_key():
    """Generate a secure API key."""
    return secrets.token_urlsafe(48)

def check_password_strength(password):
    """
    Check password strength.
    Returns (is_valid, message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not re.search(r"[a-z]", password):
        return False, "Password must contain lowercase letters"
    
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain uppercase letters"
    
    if not re.search(r"\d", password):
        return False, "Password must contain numbers"
    
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain special characters"
    
    return True, "Password is strong"

def detect_brute_force(username, ip_address, window_minutes=15, max_attempts=5):
    """
    Detect potential brute force attacks.
    Returns True if suspicious activity detected.
    """
    time_threshold = datetime.utcnow() - timedelta(minutes=window_minutes)
    
    # Check failed login attempts
    failed_attempts = SecurityLog.query.filter(
        SecurityLog.event_type == 'login_failed',
        SecurityLog.timestamp > time_threshold,
        db.or_(
            SecurityLog.username == username,
            SecurityLog.ip_address == ip_address
        )
    ).count()
    
    return failed_attempts >= max_attempts

def block_ip(ip_address, reason, duration_hours=24):
    """Block an IP address."""
    blocked_until = datetime.utcnow() + timedelta(hours=duration_hours) if duration_hours else None
    
    # Check if already blocked
    existing = IPBlacklist.query.filter_by(ip_address=ip_address).first()
    if existing:
        existing.is_active = True
        existing.blocked_until = blocked_until
        existing.reason = reason
    else:
        block = IPBlacklist(
            ip_address=ip_address,
            reason=reason,
            blocked_until=blocked_until
        )
        db.session.add(block)
    
    db.session.commit()

# CSRF Protection
def generate_csrf_token():
    """Generate CSRF token."""
    if '_csrf_token' not in g:
        g._csrf_token = secrets.token_urlsafe(32)
    return g._csrf_token

def validate_csrf_token(token):
    """Validate CSRF token."""
    return token == g.get('_csrf_token') 