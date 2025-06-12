from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

def init_db(app):
    """Initialize database with Flask app."""
    db.init_app(app)
    return db

class QueryLog(db.Model):
    """Model for storing user queries and their results."""
    __tablename__ = 'query_log'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    system_description = db.Column(db.Text, nullable=False)
    model_config = db.Column(db.String(100), nullable=False)
    rag_k = db.Column(db.Integer, default=3)
    
    # Generation results
    llm_output = db.Column(db.Text)
    generated_certificate = db.Column(db.Text)
    context_chunks = db.Column(db.Integer, default=0)
    
    # Processing status
    status = db.Column(db.String(20), default='pending')  # pending, processing, completed, failed
    error_message = db.Column(db.Text)
    
    # Verification summary (JSON string)
    verification_summary = db.Column(db.Text)  # JSON string with summary results
    
    # Processing time
    processing_start = db.Column(db.DateTime)
    processing_end = db.Column(db.DateTime)
    
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
        
        return description 