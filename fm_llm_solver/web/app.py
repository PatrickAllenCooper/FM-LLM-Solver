"""
Flask application factory for FM-LLM Solver web interface.

Creates and configures the Flask application with proper initialization.
"""

import os
from pathlib import Path
from typing import Optional

from flask import Flask, g
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from flask_migrate import Migrate
import redis

from fm_llm_solver.core.config import Config, load_config
from fm_llm_solver.core.logging import configure_logging, get_logger
from fm_llm_solver.services.certificate_generator import CertificateGenerator
from fm_llm_solver.services.verifier import CertificateVerifier
from fm_llm_solver.services.knowledge_base import KnowledgeBase
from fm_llm_solver.services.model_provider import ModelProviderFactory
from fm_llm_solver.services.cache import RedisCache
from fm_llm_solver.services.monitor import MonitoringService
from fm_llm_solver.web.middleware import (
    setup_request_logging,
    setup_error_handlers,
    setup_security_headers
)


# Initialize extensions
db = SQLAlchemy()
login_manager = LoginManager()
limiter = Limiter(key_func=get_remote_address)
migrate = Migrate()


def create_app(config_path: Optional[str] = None, test_config: Optional[dict] = None) -> Flask:
    """
    Create and configure the Flask application.
    
    Args:
        config_path: Path to configuration file
        test_config: Optional test configuration
        
    Returns:
        Configured Flask application
    """
    # Create Flask app
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static")
    )
    
    # Load configuration
    if test_config:
        app.config.update(test_config)
        config = Config(**test_config)
    else:
        config = load_config(config_path)
        app.config.from_object(config_to_flask(config))
    
    # Configure logging
    configure_logging(
        level=config.logging.level,
        log_dir=config.paths.log_dir,
        console=config.logging.console,
        structured=config.logging.structured
    )
    
    logger = get_logger(__name__)
    logger.info("Creating Flask application")
    
    # Initialize extensions
    init_extensions(app, config)
    
    # Initialize services
    init_services(app, config)
    
    # Register routes
    register_routes(app)
    
    # Setup middleware
    setup_middleware(app)
    
    # Register CLI commands
    register_cli_commands(app)
    
    logger.info("Flask application created successfully")
    
    return app


def config_to_flask(config: Config) -> dict:
    """Convert FM-LLM config to Flask config."""
    return {
        # Database
        'SQLALCHEMY_DATABASE_URI': config.deployment.database_url,
        'SQLALCHEMY_TRACK_MODIFICATIONS': False,
        
        # Session
        'SECRET_KEY': config.security.session.secret_key,
        'PERMANENT_SESSION_LIFETIME': config.security.session.permanent_session_lifetime,
        'SESSION_COOKIE_SECURE': True,
        'SESSION_COOKIE_HTTPONLY': True,
        'SESSION_COOKIE_SAMESITE': 'Lax',
        
        # Security
        'WTF_CSRF_ENABLED': True,
        'WTF_CSRF_TIME_LIMIT': None,
        
        # Rate limiting
        'RATELIMIT_STORAGE_URI': config.deployment.redis_url or 'memory://',
        'RATELIMIT_DEFAULT': f"{config.security.rate_limit.requests_per_day}/day",
        
        # File uploads
        'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB
        
        # Custom config
        'FM_CONFIG': config
    }


def init_extensions(app: Flask, config: Config) -> None:
    """Initialize Flask extensions."""
    logger = get_logger(__name__)
    
    # Database
    db.init_app(app)
    migrate.init_app(app, db)
    
    # Authentication
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message_category = 'info'
    
    # Rate limiting
    limiter.init_app(app)
    
    # CORS
    if config.security.cors:
        CORS(
            app,
            origins=config.security.cors.allowed_origins,
            methods=config.security.cors.allowed_methods,
            allow_headers=config.security.cors.allowed_headers
        )
    
    logger.info("Flask extensions initialized")


def init_services(app: Flask, config: Config) -> None:
    """Initialize application services."""
    logger = get_logger(__name__)
    
    with app.app_context():
        # Initialize cache
        cache = None
        if config.deployment.redis_url:
            try:
                cache = RedisCache(redis.from_url(config.deployment.redis_url))
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")
        
        # Initialize model provider
        model_provider = ModelProviderFactory.create(
            provider=config.model.provider,
            config=config.model
        )
        
        # Initialize knowledge base
        knowledge_base = None
        if config.rag.enabled:
            try:
                knowledge_base = KnowledgeBase(config)
                logger.info("Knowledge base initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize knowledge base: {e}")
        
        # Initialize services
        app.certificate_generator = CertificateGenerator(
            config=config,
            model_provider=model_provider,
            knowledge_store=knowledge_base,
            cache=cache
        )
        
        app.verifier = CertificateVerifier(config)
        
        app.monitoring_service = MonitoringService(
            config=config,
            db=db
        )
        
        # Store config
        app.fm_config = config
        
        logger.info("Application services initialized")


def register_routes(app: Flask) -> None:
    """Register application routes."""
    try:
        from fm_llm_solver.web.routes import main_bp
        app.register_blueprint(main_bp)
        get_logger(__name__).info("Main routes registered")
        
        # TODO: Register other blueprints as they are implemented
        # from fm_llm_solver.web.routes import api_bp, auth_bp, monitoring_bp
        # app.register_blueprint(api_bp, url_prefix='/api/v1')
        # app.register_blueprint(auth_bp, url_prefix='/auth')
        # app.register_blueprint(monitoring_bp, url_prefix='/monitoring')
        
    except ImportError as e:
        get_logger(__name__).warning(f"Some route blueprints not available yet: {e}")


def setup_middleware(app: Flask) -> None:
    """Setup application middleware."""
    setup_request_logging(app)
    setup_error_handlers(app)
    setup_security_headers(app)
    
    get_logger(__name__).info("Middleware configured")


def register_cli_commands(app: Flask) -> None:
    """Register CLI commands."""
    
    @app.cli.command()
    def init_db():
        """Initialize the database."""
        db.create_all()
        print("Database initialized.")
    
    @app.cli.command()
    def build_kb():
        """Build the knowledge base."""
        from fm_llm_solver.services.knowledge_base_builder import build_knowledge_base
        
        config = app.fm_config
        build_knowledge_base(config)
        print("Knowledge base built.")
    
    @app.cli.command()
    def test_generation():
        """Test certificate generation."""
        from fm_llm_solver.core.types import SystemDescription
        
        system = SystemDescription(
            dynamics={"x": "-x + y", "y": "x - y"},
            initial_set="x**2 + y**2 <= 0.5",
            unsafe_set="x**2 + y**2 >= 2.0"
        )
        
        result = app.certificate_generator.generate(system)
        
        if result.success:
            print(f"Certificate: {result.certificate}")
            print(f"Confidence: {result.confidence:.2f}")
        else:
            print(f"Generation failed: {result.error}")


# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id: str):
    """Load user by ID."""
    from fm_llm_solver.web.models import User
    return User.query.get(int(user_id)) 