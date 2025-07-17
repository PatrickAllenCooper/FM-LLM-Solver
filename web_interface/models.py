import json
from datetime import datetime, timedelta

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()


def init_db(app):
    """Initialize database with Flask app."""
    db.init_app(app)
    return db


class User(UserMixin, db.Model):
    """Enhanced model for user authentication and management."""

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    # Enhanced user profile
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    organization = db.Column(db.String(200))
    job_title = db.Column(db.String(100))
    bio = db.Column(db.Text)
    website = db.Column(db.String(255))
    location = db.Column(db.String(100))
    timezone = db.Column(db.String(50), default="UTC")

    # User status and verification
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)
    is_premium = db.Column(db.Boolean, default=False)
    email_verified = db.Column(db.Boolean, default=False)
    email_verification_token = db.Column(db.String(255))
    password_reset_token = db.Column(db.String(255))
    password_reset_expires = db.Column(db.DateTime)

    # Account timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    last_active = db.Column(db.DateTime, default=datetime.utcnow)
    profile_updated_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Subscription and billing
    subscription_type = db.Column(
        db.String(20), default="free"
    )  # free, basic, premium, enterprise
    subscription_start = db.Column(db.DateTime)
    subscription_end = db.Column(db.DateTime)
    billing_email = db.Column(db.String(120))

    # Rate limiting and usage
    daily_request_count = db.Column(db.Integer, default=0)
    monthly_request_count = db.Column(db.Integer, default=0)
    total_request_count = db.Column(db.Integer, default=0)
    last_request_date = db.Column(db.Date)
    last_request_month = db.Column(db.String(7))  # YYYY-MM format

    # Usage limits based on subscription
    daily_request_limit = db.Column(db.Integer, default=50)
    monthly_request_limit = db.Column(db.Integer, default=1000)
    max_concurrent_requests = db.Column(db.Integer, default=3)

    # API access
    api_key = db.Column(db.String(64), unique=True)
    api_key_created = db.Column(db.DateTime)
    api_key_last_used = db.Column(db.DateTime)
    api_requests_count = db.Column(db.Integer, default=0)

    # User preferences
    preferred_models = db.Column(db.JSON)  # List of preferred model names
    default_rag_k = db.Column(db.Integer, default=3)
    email_notifications = db.Column(db.Boolean, default=True)
    marketing_emails = db.Column(db.Boolean, default=False)
    theme_preference = db.Column(db.String(20), default="light")  # light, dark, auto

    # User role and permissions
    role = db.Column(db.String(20), default="user")  # user, premium, admin, researcher
    permissions = db.Column(db.JSON)  # Custom permissions list

    # Privacy and security
    profile_visibility = db.Column(
        db.String(20), default="private"
    )  # public, private, contacts
    two_factor_enabled = db.Column(db.Boolean, default=False)
    two_factor_secret = db.Column(db.String(32))
    backup_codes = db.Column(db.JSON)  # List of backup codes

    # Activity tracking
    login_count = db.Column(db.Integer, default=0)
    certificates_generated = db.Column(db.Integer, default=0)
    successful_verifications = db.Column(db.Integer, default=0)
    favorite_systems = db.Column(db.JSON)  # List of favorite system IDs

    # Relationships
    queries = db.relationship("QueryLog", backref="user", lazy=True)
    conversations = db.relationship("Conversation", backref="user", lazy=True)
    rate_limit_logs = db.relationship(
        "RateLimitLog", backref="user", lazy=True, cascade="all, delete-orphan"
    )
    user_activities = db.relationship(
        "UserActivity", backref="user", lazy=True, cascade="all, delete-orphan"
    )
    certificate_favorites = db.relationship(
        "CertificateFavorite", backref="user", lazy=True, cascade="all, delete-orphan"
    )
    user_sessions = db.relationship(
        "UserSession", backref="user", lazy=True, cascade="all, delete-orphan"
    )

    def set_password(self, password):
        """Set password hash."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Check password against hash."""
        return check_password_hash(self.password_hash, password)

    def check_rate_limit(self):
        """Check if user has exceeded rate limit."""
        today = datetime.utcnow().date()
        current_month = datetime.utcnow().strftime("%Y-%m")

        # Reset daily counter
        if self.last_request_date != today:
            self.daily_request_count = 0
            self.last_request_date = today

        # Reset monthly counter
        if self.last_request_month != current_month:
            self.monthly_request_count = 0
            self.last_request_month = current_month

        # Check both daily and monthly limits
        return (
            self.daily_request_count < self.daily_request_limit
            and self.monthly_request_count < self.monthly_request_limit
        )

    def increment_request_count(self):
        """Increment daily and monthly request counts."""
        today = datetime.utcnow().date()
        current_month = datetime.utcnow().strftime("%Y-%m")

        if self.last_request_date != today:
            self.daily_request_count = 1
            self.last_request_date = today
        else:
            self.daily_request_count += 1

        if self.last_request_month != current_month:
            self.monthly_request_count = 1
            self.last_request_month = current_month
        else:
            self.monthly_request_count += 1

        self.total_request_count = (self.total_request_count or 0) + 1
        self.last_active = datetime.utcnow()

        if self.api_key:
            self.api_requests_count = (self.api_requests_count or 0) + 1
            self.api_key_last_used = datetime.utcnow()

    def increment_certificate_count(self):
        """Increment successful certificate generation count."""
        self.certificates_generated = (self.certificates_generated or 0) + 1

    def increment_verification_count(self):
        """Increment successful verification count."""
        self.successful_verifications = (self.successful_verifications or 0) + 1

    def get_subscription_status(self):
        """Get current subscription status."""
        if not self.subscription_end:
            return {"active": False, "type": "free", "days_remaining": None}

        if datetime.utcnow() > self.subscription_end:
            return {
                "active": False,
                "type": self.subscription_type,
                "days_remaining": 0,
            }

        days_remaining = (self.subscription_end - datetime.utcnow()).days
        return {
            "active": True,
            "type": self.subscription_type,
            "days_remaining": days_remaining,
        }

    def get_usage_stats(self):
        """Get user usage statistics."""
        return {
            "daily_requests": self.daily_request_count,
            "daily_limit": self.daily_request_limit,
            "monthly_requests": self.monthly_request_count,
            "monthly_limit": self.monthly_request_limit,
            "total_requests": self.total_request_count,
            "certificates_generated": self.certificates_generated,
            "successful_verifications": self.successful_verifications,
            "api_requests": self.api_requests_count,
            "daily_usage_percent": round(
                ((self.daily_request_count or 0) / (self.daily_request_limit or 1))
                * 100,
                1,
            ),
            "monthly_usage_percent": round(
                ((self.monthly_request_count or 0) / (self.monthly_request_limit or 1))
                * 100,
                1,
            ),
        }

    @property
    def full_name(self):
        """Get user's full name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.username

    @property
    def is_admin(self):
        """Check if user is admin."""
        return self.role == "admin"

    @property
    def display_name(self):
        """Get display name for UI."""
        return self.full_name if (self.first_name or self.last_name) else self.username

    def to_dict(self, include_sensitive=False):
        """Convert user to dictionary for API responses."""
        data = {
            "id": self.id,
            "username": self.username,
            "email": self.email if include_sensitive else None,
            "display_name": self.display_name,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "organization": self.organization,
            "job_title": self.job_title,
            "bio": self.bio,
            "website": self.website,
            "location": self.location,
            "is_verified": self.is_verified,
            "is_premium": self.is_premium,
            "role": self.role,
            "subscription_type": self.subscription_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "certificates_generated": self.certificates_generated,
            "successful_verifications": self.successful_verifications,
            "profile_visibility": self.profile_visibility,
        }

        if include_sensitive:
            data.update(
                {
                    "usage_stats": self.get_usage_stats(),
                    "subscription_status": self.get_subscription_status(),
                    "email_notifications": self.email_notifications,
                    "theme_preference": self.theme_preference,
                    "api_key_created": (
                        self.api_key_created.isoformat()
                        if self.api_key_created
                        else None
                    ),
                    "two_factor_enabled": self.two_factor_enabled,
                }
            )

        return data

    def __repr__(self):
        return f"<User {self.username}>"


class UserActivity(db.Model):
    """Track detailed user activities for analytics and security."""

    __tablename__ = "user_activities"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    # Activity details
    activity_type = db.Column(
        db.String(50), nullable=False
    )  # login, logout, certificate_generated, verification_run, etc.
    activity_details = db.Column(db.JSON)  # Additional details about the activity

    # Request context
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    session_id = db.Column(db.String(255))

    # Performance tracking
    response_time_ms = db.Column(db.Integer)
    success = db.Column(db.Boolean, default=True)
    error_message = db.Column(db.Text)

    def __repr__(self):
        return f"<UserActivity {self.activity_type} by {self.user_id}>"


class UserSession(db.Model):
    """Track user sessions for security and analytics."""

    __tablename__ = "user_sessions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    session_token = db.Column(db.String(255), unique=True, nullable=False)

    # Session details
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_activity = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)

    # Device/browser info
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    device_type = db.Column(db.String(50))  # desktop, mobile, tablet
    browser = db.Column(db.String(100))
    os = db.Column(db.String(100))

    # Security
    login_method = db.Column(db.String(20))  # password, api_key, oauth
    is_remembered = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f"<UserSession {self.user_id} - {self.created_at}>"


class CertificateFavorite(db.Model):
    """Track user's favorite certificates and systems."""

    __tablename__ = "certificate_favorites"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    query_id = db.Column(db.Integer, db.ForeignKey("query_logs.id"), nullable=False)

    # Favorite details
    name = db.Column(db.String(200))  # Custom name for the favorite
    notes = db.Column(db.Text)  # User's notes about this certificate
    tags = db.Column(db.JSON)  # User-defined tags

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_accessed = db.Column(db.DateTime, default=datetime.utcnow)

    # Visibility
    is_public = db.Column(db.Boolean, default=False)  # Whether other users can see this

    def __repr__(self):
        return f"<CertificateFavorite {self.name} by {self.user_id}>"


class RateLimitLog(db.Model):
    """Model for tracking rate limit violations and patterns."""

    __tablename__ = "rate_limit_logs"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    # Request details
    endpoint = db.Column(db.String(200))
    method = db.Column(db.String(10))
    ip_address = db.Column(db.String(45))

    # Rate limit status
    was_blocked = db.Column(db.Boolean, default=False)
    requests_today = db.Column(db.Integer)
    limit_exceeded_by = db.Column(db.Integer, default=0)

    def __repr__(self):
        return f"<RateLimitLog {self.user_id} at {self.timestamp}>"


class IPBlacklist(db.Model):
    """Model for tracking blocked IP addresses."""

    __tablename__ = "ip_blacklist"

    id = db.Column(db.Integer, primary_key=True)
    ip_address = db.Column(db.String(45), unique=True, nullable=False)
    reason = db.Column(db.String(200))
    blocked_at = db.Column(db.DateTime, default=datetime.utcnow)
    blocked_until = db.Column(db.DateTime)  # None means permanent
    is_active = db.Column(db.Boolean, default=True)

    # Tracking
    request_count = db.Column(db.Integer, default=0)
    last_request = db.Column(db.DateTime)

    def is_blocked(self):
        """Check if IP is currently blocked."""
        if not self.is_active:
            return False
        if self.blocked_until and datetime.utcnow() > self.blocked_until:
            self.is_active = False
            return False
        return True

    def __repr__(self):
        return f"<IPBlacklist {self.ip_address}>"


class SecurityLog(db.Model):
    """Model for security-related events logging."""

    __tablename__ = "security_logs"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    event_type = db.Column(db.String(50))  # login_failed, suspicious_activity, etc.

    # User info (if applicable)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    username = db.Column(db.String(80))

    # Request info
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    endpoint = db.Column(db.String(200))

    # Event details
    description = db.Column(db.Text)
    severity = db.Column(db.String(20))  # low, medium, high, critical

    def __repr__(self):
        return f"<SecurityLog {self.event_type} at {self.timestamp}>"


class Conversation(db.Model):
    """Model for tracking ongoing conversations with the LLM."""

    __tablename__ = "conversation"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(
        db.String(100), unique=True, nullable=False
    )  # For frontend tracking
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # User tracking
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    # Conversation settings
    model_config = db.Column(db.String(100), nullable=False)
    rag_k = db.Column(db.Integer, default=3)

    # Current state
    status = db.Column(
        db.String(20), default="active"
    )  # active, generating, completed, archived
    system_description = db.Column(db.Text)  # Latest understood system description
    ready_to_generate = db.Column(
        db.Boolean, default=False
    )  # User's readiness indicator

    # Domain bounds for certificate validity (extracted from conversation)
    domain_bounds = db.Column(db.Text)  # JSON string: {"x": [-2, 2], "y": [-1, 1]}
    domain_conditions = db.Column(db.Text)  # JSON string: ["x >= -2", "x <= 2", ...]

    # Relationship to messages
    messages = db.relationship(
        "ConversationMessage",
        backref="conversation",
        lazy=True,
        cascade="all, delete-orphan",
        order_by="ConversationMessage.timestamp",
    )

    def __repr__(self):
        return f"<Conversation {self.id}: {self.status}>"

    @property
    def message_count(self):
        """Get the number of messages in this conversation."""
        return len(self.messages)

    @property
    def last_message_time(self):
        """Get the timestamp of the last message."""
        if self.messages:
            return max(msg.timestamp for msg in self.messages)
        return self.created_at

    def get_domain_bounds_dict(self):
        """Get domain bounds as a dictionary."""
        if self.domain_bounds:
            try:
                return json.loads(self.domain_bounds)
            except json.JSONDecodeError:
                return {}
        return {}

    def set_domain_bounds_dict(self, bounds_dict):
        """Set domain bounds from a dictionary."""
        if bounds_dict:
            self.domain_bounds = json.dumps(bounds_dict)
        else:
            self.domain_bounds = None


class ConversationMessage(db.Model):
    """Model for individual messages in a conversation."""

    __tablename__ = "conversation_message"

    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(
        db.Integer, db.ForeignKey("conversation.id"), nullable=False
    )
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Message content
    role = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)

    # Metadata
    message_type = db.Column(
        db.String(30), default="chat"
    )  # 'chat', 'system_clarification', 'generation_request'
    processing_time_seconds = db.Column(db.Float)  # For assistant messages
    context_chunks_used = db.Column(db.Integer, default=0)  # For RAG context

    def __repr__(self):
        return f"<ConversationMessage {self.id}: {self.role}>"

    def to_dict(self):
        """Convert message to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "message_type": self.message_type,
            "processing_time_seconds": self.processing_time_seconds,
            "context_chunks_used": self.context_chunks_used,
        }


class QueryLog(db.Model):
    """Enhanced model to track all queries and their results with comprehensive user data."""

    __tablename__ = "query_logs"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    # System and query details
    system_description = db.Column(db.Text, nullable=False)
    system_name = db.Column(db.String(200))  # User-provided name for the system
    system_type = db.Column(db.String(50))  # continuous, discrete, stochastic
    system_dimension = db.Column(db.Integer)  # Detected or specified dimension
    variables = db.Column(db.JSON)  # List of system variables

    # Model configuration and generation
    model_config = db.Column(db.JSON)
    model_name = db.Column(db.String(200))  # Name of the model used
    model_version = db.Column(db.String(50))  # Version of the model
    rag_k = db.Column(db.Integer, default=0)  # Number of RAG documents used
    temperature = db.Column(db.Float, default=0.7)  # Generation temperature
    max_tokens = db.Column(db.Integer, default=512)  # Max tokens generated

    # Results
    generated_certificate = db.Column(db.Text)
    certificate_format = db.Column(
        db.String(50)
    )  # polynomial, trigonometric, rational, etc.
    certificate_complexity = db.Column(db.Integer)  # Estimated complexity score
    extraction_method = db.Column(
        db.String(50)
    )  # How certificate was extracted from LLM output

    # Status and performance
    status = db.Column(
        db.String(50), default="pending"
    )  # pending, completed, failed, verified
    error_message = db.Column(db.Text)
    processing_time = db.Column(db.Float)  # seconds
    processing_start = db.Column(db.DateTime)  # Start time of processing
    processing_end = db.Column(db.DateTime)  # End time of processing
    total_tokens_used = db.Column(db.Integer)  # Total tokens consumed
    cost_estimate = db.Column(db.Float)  # Estimated cost in USD

    # Context and tracking
    conversation_id = db.Column(db.String(36))
    session_id = db.Column(db.String(255))
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    # User interaction
    user_rating = db.Column(db.Integer)  # 1-5 star rating from user
    user_feedback = db.Column(db.Text)  # User's comments about the result
    is_favorite = db.Column(db.Boolean, default=False)
    is_public = db.Column(db.Boolean, default=False)  # Whether user shared publicly
    tags = db.Column(db.JSON)  # User-defined tags

    # Domain bounds for certificate validity
    certificate_domain_bounds = db.Column(
        db.Text
    )  # JSON string: {"x": [-2, 2], "y": [-1, 1]}
    domain_bounds_conditions = db.Column(
        db.Text
    )  # JSON string: ["x >= -2", "x <= 2", ...]
    domain_description = db.Column(db.Text)  # Human-readable domain description

    # Verification tracking
    verification_requested = db.Column(db.Boolean, default=False)
    verification_completed = db.Column(db.Boolean, default=False)
    verification_success = db.Column(db.Boolean, default=False)
    verification_attempts = db.Column(db.Integer, default=0)

    # Quality metrics
    confidence_score = db.Column(db.Float)  # Model's confidence in the result
    mathematical_soundness = db.Column(
        db.Float
    )  # Automated assessment of mathematical validity

    # Relationships
    verification_result = db.relationship(
        "VerificationResult", backref="query", uselist=False
    )
    certificate_favorite = db.relationship(
        "CertificateFavorite", backref="query_log", uselist=False
    )

    def __repr__(self):
        return f"<QueryLog {self.id}: {self.status}>"

    @property
    def processing_time(self):
        """Calculate processing time in seconds."""
        if self.processing_start and self.processing_end:
            return (self.processing_end - self.processing_start).total_seconds()
        return None

    def get_domain_bounds_dict(self):
        """Get domain bounds as a dictionary."""
        if self.certificate_domain_bounds:
            try:
                return json.loads(self.certificate_domain_bounds)
            except json.JSONDecodeError:
                return {}
        return {}

    def set_domain_bounds_dict(self, bounds_dict):
        """Set domain bounds from a dictionary."""
        if bounds_dict:
            self.certificate_domain_bounds = json.dumps(bounds_dict)
        else:
            self.certificate_domain_bounds = None

    def get_domain_conditions(self):
        """Get domain bounds conditions as a list."""
        if self.domain_bounds_conditions:
            try:
                return json.loads(self.domain_bounds_conditions)
            except json.JSONDecodeError:
                return []
        return []

    def set_domain_conditions(self, conditions_list):
        """Set domain bounds conditions from a list."""
        if conditions_list:
            self.domain_bounds_conditions = json.dumps(conditions_list)
        else:
            self.domain_bounds_conditions = None


class VerificationResult(db.Model):
    """Model for detailed verification results."""

    __tablename__ = "verification_result"

    id = db.Column(db.Integer, primary_key=True)
    query_id = db.Column(db.Integer, db.ForeignKey("query_logs.id"), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Individual verification checks
    numerical_check_passed = db.Column(db.Boolean, default=False)
    symbolic_check_passed = db.Column(db.Boolean, default=False)
    sos_check_passed = db.Column(db.Boolean, default=False)

    # Domain bounds verification
    domain_bounds_check_passed = db.Column(
        db.Boolean, default=True
    )  # True if no bounds specified
    domain_bounds_violations = db.Column(
        db.Integer, default=0
    )  # Number of violations found

    # Overall verification result
    overall_success = db.Column(db.Boolean, default=False)

    # Detailed verification information (JSON string)
    verification_details = db.Column(
        db.Text
    )  # JSON with detailed results, error messages, etc.

    # Verification metadata
    verification_time_seconds = db.Column(db.Float)
    samples_used = db.Column(db.Integer)
    tolerance_used = db.Column(db.Float)

    def __repr__(self):
        return f"<VerificationResult {self.id}: overall={self.overall_success}>"


class ModelConfiguration(db.Model):
    """Model for storing different model configurations."""

    __tablename__ = "model_configuration"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text)

    # Model settings
    base_model_name = db.Column(db.String(200), nullable=False)
    adapter_path = db.Column(db.String(500))
    barrier_certificate_type = db.Column(db.String(50))  # discrete, continuous, unified

    # Configuration JSON
    config_json = db.Column(db.Text)  # JSON string with full configuration

    # Status
    is_active = db.Column(db.Boolean, default=True)
    is_available = db.Column(db.Boolean, default=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    def __repr__(self):
        return f"<ModelConfiguration {self.name}>"


class SystemBenchmark(db.Model):
    """Model for storing benchmark systems for testing."""

    __tablename__ = "system_benchmark"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)

    # System definition
    system_dynamics = db.Column(db.Text, nullable=False)
    initial_set = db.Column(db.Text)
    unsafe_set = db.Column(db.Text)
    safe_set = db.Column(db.Text)
    state_variables = db.Column(db.String(200))

    # Domain bounds for barrier certificate validity
    certificate_domain_bounds = db.Column(
        db.Text
    )  # JSON string: {"x": [-2, 2], "y": [-1, 1]}
    domain_bounds_description = db.Column(
        db.Text
    )  # Human-readable description of domain

    # Expected results (if known)
    expected_certificate = db.Column(db.Text)
    expected_verification = db.Column(db.Boolean)

    # Metadata
    difficulty_level = db.Column(db.String(20))  # easy, medium, hard
    system_type = db.Column(db.String(50))  # discrete, continuous, hybrid
    dimension = db.Column(db.Integer)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

    def __repr__(self):
        return f"<SystemBenchmark {self.name}>"

    def to_system_description(self):
        """Convert to system description format used by the inference engine."""
        description = f"System Dynamics: {self.system_dynamics}"

        if self.state_variables:
            description += f"\nState Variables: {self.state_variables}"

        if self.initial_set:
            description += f"\nInitial Set: {self.initial_set}"

        if self.unsafe_set:
            description += f"\nUnsafe Set: {self.unsafe_set}"

        if self.safe_set:
            description += f"\nSafe Set: {self.safe_set}"

        # Add domain bounds information
        if self.certificate_domain_bounds:
            try:
                bounds_dict = json.loads(self.certificate_domain_bounds)
                bounds_desc = ", ".join(
                    [
                        f"{var} ∈ [{bounds[0]}, {bounds[1]}]"
                        for var, bounds in bounds_dict.items()
                    ]
                )
                description += f"\nDomain Bounds: {bounds_desc}"
            except json.JSONDecodeError:
                pass

        if self.domain_bounds_description:
            description += f"\nDomain Description: {self.domain_bounds_description}"

        return description

    def get_domain_bounds_dict(self):
        """Get domain bounds as a dictionary."""
        if self.certificate_domain_bounds:
            try:
                return json.loads(self.certificate_domain_bounds)
            except json.JSONDecodeError:
                return {}
        return {}

    def set_domain_bounds_dict(self, bounds_dict):
        """Set domain bounds from a dictionary."""
        if bounds_dict:
            self.certificate_domain_bounds = json.dumps(bounds_dict)
        else:
            self.certificate_domain_bounds = None
