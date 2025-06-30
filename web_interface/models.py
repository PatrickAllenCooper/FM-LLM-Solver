from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

db = SQLAlchemy()

def init_db(app):
    """Initialize database with Flask app."""
    db.init_app(app)
    return db

class User(UserMixin, db.Model):
    """Model for user authentication and management."""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    
    # User status
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Rate limiting
    daily_request_count = db.Column(db.Integer, default=0)
    last_request_date = db.Column(db.Date)
    daily_request_limit = db.Column(db.Integer, default=50)  # Customizable per user
    
    # API key for programmatic access (optional)
    api_key = db.Column(db.String(64), unique=True)
    api_key_created = db.Column(db.DateTime)
    
    # User role
    role = db.Column(db.String(20), default='user')  # user, premium, admin
    
    # Relationships
    queries = db.relationship('QueryLog', backref='user', lazy=True)
    rate_limit_logs = db.relationship('RateLimitLog', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Set password hash."""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash."""
        return check_password_hash(self.password_hash, password)
    
    def check_rate_limit(self):
        """Check if user has exceeded rate limit."""
        today = datetime.utcnow().date()
        if self.last_request_date != today:
            # Reset counter for new day
            self.daily_request_count = 0
            self.last_request_date = today
        
        return self.daily_request_count < self.daily_request_limit
    
    def increment_request_count(self):
        """Increment daily request count."""
        today = datetime.utcnow().date()
        if self.last_request_date != today:
            self.daily_request_count = 1
            self.last_request_date = today
        else:
            self.daily_request_count += 1
    
    def __repr__(self):
        return f'<User {self.username}>'

class RateLimitLog(db.Model):
    """Model for tracking rate limit violations and patterns."""
    __tablename__ = 'rate_limit_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
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
        return f'<RateLimitLog {self.user_id} at {self.timestamp}>'

class IPBlacklist(db.Model):
    """Model for tracking blocked IP addresses."""
    __tablename__ = 'ip_blacklist'
    
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
        return f'<IPBlacklist {self.ip_address}>'

class SecurityLog(db.Model):
    """Model for security-related events logging."""
    __tablename__ = 'security_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    event_type = db.Column(db.String(50))  # login_failed, suspicious_activity, etc.
    
    # User info (if applicable)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    username = db.Column(db.String(80))
    
    # Request info
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    endpoint = db.Column(db.String(200))
    
    # Event details
    description = db.Column(db.Text)
    severity = db.Column(db.String(20))  # low, medium, high, critical
    
    def __repr__(self):
        return f'<SecurityLog {self.event_type} at {self.timestamp}>'

class Conversation(db.Model):
    """Model for tracking ongoing conversations with the LLM."""
    __tablename__ = 'conversation'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), unique=True, nullable=False)  # For frontend tracking
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # User tracking
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    
    # Conversation settings
    model_config = db.Column(db.String(100), nullable=False)
    rag_k = db.Column(db.Integer, default=3)
    
    # Current state
    status = db.Column(db.String(20), default='active')  # active, generating, completed, archived
    system_description = db.Column(db.Text)  # Latest understood system description
    ready_to_generate = db.Column(db.Boolean, default=False)  # User's readiness indicator
    
    # Domain bounds for certificate validity (extracted from conversation)
    domain_bounds = db.Column(db.Text)  # JSON string: {"x": [-2, 2], "y": [-1, 1]}
    domain_conditions = db.Column(db.Text)  # JSON string: ["x >= -2", "x <= 2", ...]
    
    # Relationship to messages and queries
    messages = db.relationship('ConversationMessage', backref='conversation', lazy=True, cascade='all, delete-orphan', order_by='ConversationMessage.timestamp')
    query_logs = db.relationship('QueryLog', backref='conversation', lazy=True)
    
    def __repr__(self):
        return f'<Conversation {self.id}: {self.status}>'
    
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
    __tablename__ = 'conversation_message'
    
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Message content
    role = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    
    # Metadata
    message_type = db.Column(db.String(30), default='chat')  # 'chat', 'system_clarification', 'generation_request'
    processing_time_seconds = db.Column(db.Float)  # For assistant messages
    context_chunks_used = db.Column(db.Integer, default=0)  # For RAG context
    
    def __repr__(self):
        return f'<ConversationMessage {self.id}: {self.role}>'
    
    def to_dict(self):
        """Convert message to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'message_type': self.message_type,
            'processing_time_seconds': self.processing_time_seconds,
            'context_chunks_used': self.context_chunks_used
        }

class QueryLog(db.Model):
    """Model for storing user queries and their results."""
    __tablename__ = 'query_log'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # User tracking
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    
    system_description = db.Column(db.Text, nullable=False)
    model_config = db.Column(db.String(100), nullable=False)
    rag_k = db.Column(db.Integer, default=3)
    
    # Optional conversation link
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=True)
    
    # Generation results
    llm_output = db.Column(db.Text)
    generated_certificate = db.Column(db.Text)
    context_chunks = db.Column(db.Integer, default=0)
    
    # Domain bounds for barrier certificate validity
    certificate_domain_bounds = db.Column(db.Text)  # JSON string: {"x": [-2, 2], "y": [-1, 1]}
    domain_bounds_conditions = db.Column(db.Text)  # JSON string: ["x >= -2", "x <= 2", "y >= -1", "y <= 1"]
    
    # Processing status
    status = db.Column(db.String(20), default='pending')  # pending, processing, completed, failed
    error_message = db.Column(db.Text)
    
    # Verification summary (JSON string)
    verification_summary = db.Column(db.Text)  # JSON string with summary results
    
    # Processing time
    processing_start = db.Column(db.DateTime)
    processing_end = db.Column(db.DateTime)
    
    # User decision on generated certificate
    user_decision = db.Column(db.String(20))  # 'accepted', 'rejected', 'pending'
    decision_timestamp = db.Column(db.DateTime)
    
    # Relationship to verification results
    verification_results = db.relationship('VerificationResult', backref='query', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<QueryLog {self.id}: {self.status}>'
    
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
    __tablename__ = 'verification_result'
    
    id = db.Column(db.Integer, primary_key=True)
    query_id = db.Column(db.Integer, db.ForeignKey('query_log.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Individual verification checks
    numerical_check_passed = db.Column(db.Boolean, default=False)
    symbolic_check_passed = db.Column(db.Boolean, default=False)
    sos_check_passed = db.Column(db.Boolean, default=False)
    
    # Domain bounds verification
    domain_bounds_check_passed = db.Column(db.Boolean, default=True)  # True if no bounds specified
    domain_bounds_violations = db.Column(db.Integer, default=0)  # Number of violations found
    
    # Overall verification result
    overall_success = db.Column(db.Boolean, default=False)
    
    # Detailed verification information (JSON string)
    verification_details = db.Column(db.Text)  # JSON with detailed results, error messages, etc.
    
    # Verification metadata
    verification_time_seconds = db.Column(db.Float)
    samples_used = db.Column(db.Integer)
    tolerance_used = db.Column(db.Float)
    
    def __repr__(self):
        return f'<VerificationResult {self.id}: overall={self.overall_success}>'

class ModelConfiguration(db.Model):
    """Model for storing different model configurations."""
    __tablename__ = 'model_configuration'
    
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
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<ModelConfiguration {self.name}>'

class SystemBenchmark(db.Model):
    """Model for storing benchmark systems for testing."""
    __tablename__ = 'system_benchmark'
    
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
    certificate_domain_bounds = db.Column(db.Text)  # JSON string: {"x": [-2, 2], "y": [-1, 1]}
    domain_bounds_description = db.Column(db.Text)  # Human-readable description of domain
    
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
        return f'<SystemBenchmark {self.name}>'
    
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
                bounds_desc = ", ".join([f"{var} ∈ [{bounds[0]}, {bounds[1]}]" for var, bounds in bounds_dict.items()])
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