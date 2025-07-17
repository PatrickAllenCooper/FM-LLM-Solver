"""
Database models for FM-LLM Solver web interface.

Provides SQLAlchemy models for user management, query logging, and results storage.
"""

import json
from datetime import datetime
from typing import Any, Dict, Optional

from flask import Flask
from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

from fm_llm_solver.core.logging import get_logger

db = SQLAlchemy()
logger = get_logger(__name__)


class User(UserMixin, db.Model):
    """User model for authentication and authorization."""

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(
        db.String(20), default="user", nullable=False
    )  # user, premium, admin
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_login = db.Column(db.DateTime)

    # Request tracking
    requests_today = db.Column(db.Integer, default=0, nullable=False)
    last_request_date = db.Column(db.Date)

    # Relationships
    queries = db.relationship(
        "QueryLog", backref="user", lazy="dynamic", cascade="all, delete-orphan"
    )
    conversations = db.relationship(
        "Conversation", backref="user", lazy="dynamic", cascade="all, delete-orphan"
    )

    def set_password(self, password: str) -> None:
        """Set password hash."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Check password against hash."""
        return check_password_hash(self.password_hash, password)

    def get_daily_limit(self) -> int:
        """Get daily request limit based on user role."""
        limits = {"user": 50, "premium": 200, "admin": 1000}
        return limits.get(self.role, 50)

    def can_make_request(self) -> bool:
        """Check if user can make another request today."""
        today = datetime.now().date()

        # Reset counter if it's a new day
        if self.last_request_date != today:
            self.requests_today = 0
            self.last_request_date = today
            db.session.commit()

        return self.requests_today < self.get_daily_limit()

    def record_request(self) -> None:
        """Record a new request."""
        today = datetime.now().date()

        if self.last_request_date != today:
            self.requests_today = 1
            self.last_request_date = today
        else:
            self.requests_today += 1

        db.session.commit()

    @property
    def is_admin(self) -> bool:
        """Check if user is admin."""
        return self.role == "admin"

    @property
    def is_premium(self) -> bool:
        """Check if user is premium or admin."""
        return self.role in ["premium", "admin"]

    def __repr__(self) -> str:
        return f"<User {self.username}>"


class QueryLog(db.Model):
    """Model for logging certificate generation queries."""

    __tablename__ = "query_logs"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    system_description = db.Column(db.Text, nullable=False)
    model_config = db.Column(db.String(50), nullable=False)
    rag_k = db.Column(db.Integer, default=3, nullable=False)

    # Domain bounds (stored as JSON)
    domain_bounds = db.Column(db.Text)

    # Generation results
    llm_output = db.Column(db.Text)
    generated_certificate = db.Column(db.Text)
    context_chunks = db.Column(db.Text)  # JSON array of context chunks

    # Verification summary (JSON)
    verification_summary = db.Column(db.Text)

    # Status and timing
    status = db.Column(
        db.String(20), default="pending", nullable=False
    )  # pending, processing, completed, failed
    error_message = db.Column(db.Text)
    timestamp = db.Column(
        db.DateTime, default=datetime.utcnow, nullable=False, index=True
    )
    processing_start = db.Column(db.DateTime)
    processing_end = db.Column(db.DateTime)

    # User feedback
    user_decision = db.Column(db.String(20))  # accepted, rejected, pending
    decision_timestamp = db.Column(db.DateTime)

    # Linked conversation
    conversation_id = db.Column(
        db.Integer, db.ForeignKey("conversations.id"), nullable=True
    )

    # Relationships
    verification_results = db.relationship(
        "VerificationResult",
        backref="query",
        lazy="dynamic",
        cascade="all, delete-orphan",
    )

    def set_domain_bounds_dict(self, bounds: Dict[str, Any]) -> None:
        """Set domain bounds from dictionary."""
        self.domain_bounds = json.dumps(bounds) if bounds else None

    def get_domain_bounds_dict(self) -> Optional[Dict[str, Any]]:
        """Get domain bounds as dictionary."""
        if not self.domain_bounds:
            return None
        try:
            return json.loads(self.domain_bounds)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Failed to parse domain bounds for query {self.id}")
            return None

    def get_verification_summary_dict(self) -> Dict[str, bool]:
        """Get verification summary as dictionary."""
        if not self.verification_summary:
            return {
                "numerical": False,
                "symbolic": False,
                "sos": False,
                "overall": False,
            }
        try:
            return json.loads(self.verification_summary)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Failed to parse verification summary for query {self.id}")
            return {
                "numerical": False,
                "symbolic": False,
                "sos": False,
                "overall": False,
            }

    def get_context_chunks_list(self) -> list:
        """Get context chunks as list."""
        if not self.context_chunks:
            return []
        try:
            return json.loads(self.context_chunks)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Failed to parse context chunks for query {self.id}")
            return []

    @property
    def processing_time(self) -> Optional[float]:
        """Get processing time in seconds."""
        if self.processing_start and self.processing_end:
            return (self.processing_end - self.processing_start).total_seconds()
        return None

    def __repr__(self) -> str:
        return f"<QueryLog {self.id}: {self.status}>"


class VerificationResult(db.Model):
    """Model for storing detailed verification results."""

    __tablename__ = "verification_results"

    id = db.Column(db.Integer, primary_key=True)
    query_id = db.Column(db.Integer, db.ForeignKey("query_logs.id"), nullable=False)

    # Verification check results
    numerical_check_passed = db.Column(db.Boolean, default=False, nullable=False)
    symbolic_check_passed = db.Column(db.Boolean, default=False, nullable=False)
    sos_check_passed = db.Column(db.Boolean, default=False, nullable=False)
    overall_success = db.Column(db.Boolean, default=False, nullable=False)

    # Detailed results (JSON)
    verification_details = db.Column(db.Text)

    # Timing
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def get_verification_details_dict(self) -> Dict[str, Any]:
        """Get verification details as dictionary."""
        if not self.verification_details:
            return {}
        try:
            return json.loads(self.verification_details)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Failed to parse verification details for result {self.id}")
            return {}

    def __repr__(self) -> str:
        return f"<VerificationResult {self.id}: overall={self.overall_success}>"


class Conversation(db.Model):
    """Model for storing conversation sessions."""

    __tablename__ = "conversations"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    session_id = db.Column(db.String(64), unique=True, nullable=False, index=True)

    # Configuration
    model_config = db.Column(db.String(50), nullable=False)
    rag_k = db.Column(db.Integer, default=3, nullable=False)

    # Status
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    ready_to_generate = db.Column(db.Boolean, default=False, nullable=False)

    # Extracted system description
    system_description = db.Column(db.Text)
    domain_bounds = db.Column(db.Text)  # JSON

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    messages = db.relationship(
        "ConversationMessage",
        backref="conversation",
        lazy="dynamic",
        cascade="all, delete-orphan",
        order_by="ConversationMessage.timestamp",
    )
    queries = db.relationship("QueryLog", backref="conversation", lazy="dynamic")

    def set_domain_bounds_dict(self, bounds: Dict[str, Any]) -> None:
        """Set domain bounds from dictionary."""
        self.domain_bounds = json.dumps(bounds) if bounds else None

    def get_domain_bounds_dict(self) -> Optional[Dict[str, Any]]:
        """Get domain bounds as dictionary."""
        if not self.domain_bounds:
            return None
        try:
            return json.loads(self.domain_bounds)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Failed to parse domain bounds for conversation {self.id}")
            return None

    def add_message(
        self, role: str, content: str, message_type: str = "chat"
    ) -> "ConversationMessage":
        """Add a message to the conversation."""
        message = ConversationMessage(
            conversation_id=self.id,
            role=role,
            content=content,
            message_type=message_type,
        )
        db.session.add(message)
        self.updated_at = datetime.utcnow()
        return message

    def get_messages_list(self) -> list:
        """Get all messages as a list of dictionaries."""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "message_type": msg.message_type,
            }
            for msg in self.messages.order_by(ConversationMessage.timestamp)
        ]

    def __repr__(self) -> str:
        return f"<Conversation {self.session_id}>"


class ConversationMessage(db.Model):
    """Model for storing individual conversation messages."""

    __tablename__ = "conversation_messages"

    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(
        db.Integer, db.ForeignKey("conversations.id"), nullable=False
    )

    # Message content
    role = db.Column(db.String(20), nullable=False)  # user, assistant, system
    content = db.Column(db.Text, nullable=False)
    message_type = db.Column(
        db.String(20), default="chat", nullable=False
    )  # chat, system_description, domain_bounds

    # Timing
    timestamp = db.Column(
        db.DateTime, default=datetime.utcnow, nullable=False, index=True
    )

    def __repr__(self) -> str:
        return f"<ConversationMessage {self.id}: {self.role}>"


def init_db(app: Flask) -> SQLAlchemy:
    """Initialize database with Flask app."""
    db.init_app(app)

    with app.app_context():
        # Create tables
        db.create_all()

        # Create default admin user if it doesn't exist
        admin = User.query.filter_by(username="admin").first()
        if not admin:
            admin = User(username="admin", email="admin@fmllmsolver.com", role="admin")
            admin.set_password("admin123")  # Change in production!
            db.session.add(admin)
            db.session.commit()
            logger.info(
                "Created default admin user (username: admin, password: admin123)"
            )

    return db
