"""
Authentication routes for FM-LLM Solver web interface.
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, session, abort
from flask_login import login_user, logout_user, login_required, current_user
from web_interface.models import db, User, SecurityLog, UserActivity, UserSession
from web_interface.auth import (
    check_password_strength, detect_brute_force, block_ip, 
    log_security_event, generate_api_key, get_client_ip
)
from datetime import datetime
import re
import secrets

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login page and handler."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember', False)
        
        # Input validation
        if not username or not password:
            flash('Please enter both username and password.', 'error')
            return render_template('auth/login.html')
        
        # Check for brute force
        ip_address = get_client_ip()
        if detect_brute_force(username, ip_address):
            # Block IP for repeated failures
            block_ip(ip_address, 'Brute force login attempts', duration_hours=1)
            log_security_event('brute_force_detected', severity='high',
                             description=f'Brute force detected for username: {username}')
            flash('Too many failed login attempts. Please try again later.', 'error')
            return render_template('auth/login.html'), 429
        
        # Find user
        user = User.query.filter(
            db.or_(User.username == username, User.email == username)
        ).first()
        
        if user and user.check_password(password):
            if not user.is_active:
                log_security_event('login_inactive_account', severity='medium',
                                 description=f'Login attempt for inactive account: {username}')
                flash('Your account has been deactivated. Please contact support.', 'error')
                return render_template('auth/login.html')
            
            # Successful login
            login_user(user, remember=remember)
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            log_security_event('login_success', severity='low',
                             description=f'Successful login for user: {username}')
            
            # Redirect to next page or index
            next_page = request.args.get('next')
            if next_page and next_page.startswith('/'):
                return redirect(next_page)
            return redirect(url_for('index'))
        else:
            # Failed login
            log_security_event('login_failed', severity='medium',
                             description=f'Failed login attempt for username: {username}')
            flash('Invalid username or password.', 'error')
    
    return render_template('auth/login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page and handler."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Validation
        errors = []
        
        # Username validation
        if not username or len(username) < 3:
            errors.append('Username must be at least 3 characters long.')
        elif not re.match(r'^[a-zA-Z0-9_]+$', username):
            errors.append('Username can only contain letters, numbers, and underscores.')
        elif User.query.filter_by(username=username).first():
            errors.append('Username already exists.')
        
        # Email validation
        if not email or not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            errors.append('Please enter a valid email address.')
        elif User.query.filter_by(email=email).first():
            errors.append('Email already registered.')
        
        # Password validation
        if password != confirm_password:
            errors.append('Passwords do not match.')
        else:
            is_valid, message = check_password_strength(password)
            if not is_valid:
                errors.append(message)
        
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('auth/register.html')
        
        # Create user
        user = User(
            username=username,
            email=email,
            is_active=True,
            is_verified=False  # Email verification can be added later
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        log_security_event('user_registered', severity='low',
                         description=f'New user registered: {username}')
        
        # Auto-login after registration
        login_user(user)
        flash('Registration successful! Welcome to FM-LLM Solver.', 'success')
        
        return redirect(url_for('index'))
    
    return render_template('auth/register.html')

@auth_bp.route('/logout')
@login_required
def logout():
    """User logout."""
    username = current_user.username
    logout_user()
    
    log_security_event('logout', severity='low',
                     description=f'User logged out: {username}')
    
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@auth_bp.route('/profile')
@login_required
def profile():
    """User profile page."""
    return render_template('auth/profile.html', user=current_user)

@auth_bp.route('/profile/update', methods=['POST'])
@login_required
def update_profile():
    """Update user profile."""
    email = request.form.get('email', '').strip().lower()
    
    # Validate email
    if email and email != current_user.email:
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            flash('Please enter a valid email address.', 'error')
        elif User.query.filter(User.email == email, User.id != current_user.id).first():
            flash('Email already in use.', 'error')
        else:
            current_user.email = email
            db.session.commit()
            flash('Email updated successfully.', 'success')
    
    return redirect(url_for('auth.profile'))

@auth_bp.route('/profile/change-password', methods=['POST'])
@login_required
def change_password():
    """Change user password."""
    current_password = request.form.get('current_password', '')
    new_password = request.form.get('new_password', '')
    confirm_password = request.form.get('confirm_password', '')
    
    if not current_user.check_password(current_password):
        flash('Current password is incorrect.', 'error')
    elif new_password != confirm_password:
        flash('New passwords do not match.', 'error')
    else:
        is_valid, message = check_password_strength(new_password)
        if not is_valid:
            flash(message, 'error')
        else:
            current_user.set_password(new_password)
            db.session.commit()
            
            log_security_event('password_changed', severity='medium',
                             description=f'Password changed for user: {current_user.username}')
            
            flash('Password changed successfully.', 'success')
    
    return redirect(url_for('auth.profile'))

@auth_bp.route('/api/key', methods=['GET', 'POST'])
@login_required
def api_key():
    """Generate or regenerate API key."""
    if request.method == 'POST':
        # Generate new API key
        current_user.api_key = generate_api_key()
        current_user.api_key_created = datetime.utcnow()
        db.session.commit()
        
        log_security_event('api_key_generated', severity='medium',
                         description=f'API key generated for user: {current_user.username}')
        
        if request.is_json:
            return jsonify({
                'success': True,
                'api_key': current_user.api_key,
                'created': current_user.api_key_created.isoformat()
            })
        else:
            flash('API key generated successfully.', 'success')
            return redirect(url_for('auth.profile'))
    
    # GET request - show current API key status
    if request.is_json:
        return jsonify({
            'has_key': bool(current_user.api_key),
            'created': current_user.api_key_created.isoformat() if current_user.api_key_created else None
        })
    else:
        return render_template('auth/api_key.html', user=current_user)

@auth_bp.route('/api/key/revoke', methods=['POST'])
@login_required
def revoke_api_key():
    """Revoke API key."""
    current_user.api_key = None
    current_user.api_key_created = None
    db.session.commit()
    
    log_security_event('api_key_revoked', severity='medium',
                     description=f'API key revoked for user: {current_user.username}')
    
    if request.is_json:
        return jsonify({'success': True, 'message': 'API key revoked'})
    else:
        flash('API key revoked successfully.', 'success')
        return redirect(url_for('auth.profile'))

# Admin routes
@auth_bp.route('/admin/users')
@login_required
def admin_users():
    """Admin user management page."""
    if current_user.role != 'admin':
        abort(403, 'Admin access required')
    
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template('auth/admin_users.html', users=users)

@auth_bp.route('/admin/security-logs')
@login_required
def admin_security_logs():
    """View security logs."""
    if current_user.role != 'admin':
        abort(403, 'Admin access required')
    
    page = request.args.get('page', 1, type=int)
    logs = SecurityLog.query.order_by(SecurityLog.timestamp.desc()).paginate(
        page=page, per_page=50, error_out=False
    )
    
    return render_template('auth/security_logs.html', logs=logs) 

def log_user_activity(activity_type, details=None, success=True, response_time_ms=None):
    """Log user activity for analytics and security."""
    if current_user.is_authenticated:
        try:
            activity = UserActivity(
                user_id=current_user.id,
                activity_type=activity_type,
                activity_details=details or {},
                ip_address=get_client_ip(),
                user_agent=request.headers.get('User-Agent', ''),
                session_id=session.get('_id', ''),
                response_time_ms=response_time_ms,
                success=success
            )
            db.session.add(activity)
            db.session.commit()
        except Exception:
            # Don't let activity logging break the main functionality
            db.session.rollback()

def track_login_activity(user):
    """Track login activity and update user statistics."""
    user.login_count = (user.login_count or 0) + 1
    user.last_login = datetime.utcnow()
    
    # Create or update session record
    session_token = session.get('_id', secrets.token_urlsafe(32))
    session['_id'] = session_token
    
    user_session = UserSession(
        user_id=user.id,
        session_token=session_token,
        ip_address=get_client_ip(),
        user_agent=request.headers.get('User-Agent', ''),
        device_type=detect_device_type(request.headers.get('User-Agent', '')),
        browser=detect_browser(request.headers.get('User-Agent', '')),
        os=detect_os(request.headers.get('User-Agent', '')),
        login_method='password',
        is_remembered=bool(request.form.get('remember'))
    )
    db.session.add(user_session)
    
    log_user_activity('login', {
        'ip_address': get_client_ip(),
        'user_agent': request.headers.get('User-Agent', ''),
        'remember_me': bool(request.form.get('remember'))
    })

def detect_device_type(user_agent):
    """Detect device type from user agent."""
    user_agent = user_agent.lower()
    if 'mobile' in user_agent or 'android' in user_agent or 'iphone' in user_agent:
        return 'mobile'
    elif 'tablet' in user_agent or 'ipad' in user_agent:
        return 'tablet'
    else:
        return 'desktop'

def detect_browser(user_agent):
    """Detect browser from user agent."""
    user_agent = user_agent.lower()
    if 'chrome' in user_agent:
        return 'Chrome'
    elif 'firefox' in user_agent:
        return 'Firefox'
    elif 'safari' in user_agent:
        return 'Safari'
    elif 'edge' in user_agent:
        return 'Edge'
    else:
        return 'Unknown'

def detect_os(user_agent):
    """Detect operating system from user agent."""
    user_agent = user_agent.lower()
    if 'windows' in user_agent:
        return 'Windows'
    elif 'macintosh' in user_agent or 'mac os' in user_agent:
        return 'macOS'
    elif 'linux' in user_agent:
        return 'Linux'
    elif 'android' in user_agent:
        return 'Android'
    elif 'ios' in user_agent:
        return 'iOS'
    else:
        return 'Unknown' 