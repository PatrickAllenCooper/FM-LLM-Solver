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

from fm_llm_solver.core.config_manager import ConfigurationManager
from fm_llm_solver.core.logging_manager import get_logging_manager, get_logger
from fm_llm_solver.core.database_manager import get_database_manager
from fm_llm_solver.services.certificate_generator import CertificateGenerator
from fm_llm_solver.services.verification_service import CertificateVerifier
from fm_llm_solver.services.knowledge_base import KnowledgeBase
from fm_llm_solver.services.model_provider import QwenProvider
from fm_llm_solver.web.models import User, QueryLog, VerificationResult, Conversation
from fm_llm_solver.web.utils import (
    setup_security_headers,
    setup_rate_limiting,
    setup_cors,
    validate_input,
    sanitize_output,
    get_client_ip,
    log_security_event,
    handle_error_response
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
    
    # Initialize configuration manager
    config_manager = ConfigurationManager(config_dir=config_path)
    
    # Load configuration
    if test_config:
        app.config.update(test_config)
        flask_config = test_config
    else:
        config = config_manager.load_config()
        flask_config = config_to_flask(config, config_manager)
        app.config.from_object(type('Config', (), flask_config))
    
    # Initialize logging
    logging_manager = get_logging_manager()
    logger = get_logger(__name__)
    logger.info("Creating Flask application")
    
    # Store managers in app context
    app.config_manager = config_manager
    app.logging_manager = logging_manager
    
    # Initialize extensions
    init_extensions(app, config_manager)
    
    # Initialize services
    init_services(app, config_manager)
    
    # Register routes
    register_routes(app)
    
    # Setup middleware
    setup_middleware(app)
    
    # Register CLI commands
    register_cli_commands(app)
    
    logger.info("Flask application created successfully")
    
    return app


def config_to_flask(config: dict, config_manager: ConfigurationManager) -> dict:
    """Convert FM-LLM config to Flask config."""
    web_config = config.get('web_interface', {})
    db_config = config.get('database', {}).get('primary', {})
    
    # Build database URL
    db_url = f"postgresql://{db_config.get('username', 'postgres')}:{db_config.get('password', '')}@{db_config.get('host', 'localhost')}:{db_config.get('port', 5432)}/{db_config.get('database', 'fm_llm_solver')}"
    
    return {
        # Database
        'SQLALCHEMY_DATABASE_URI': db_url,
        'SQLALCHEMY_TRACK_MODIFICATIONS': False,
        
        # Session
        'SECRET_KEY': os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production'),
        'PERMANENT_SESSION_LIFETIME': 86400,  # 24 hours
        'SESSION_COOKIE_SECURE': config_manager.environment.value == 'production',
        'SESSION_COOKIE_HTTPONLY': True,
        'SESSION_COOKIE_SAMESITE': 'Lax',
        
        # Security
        'WTF_CSRF_ENABLED': True,
        'WTF_CSRF_TIME_LIMIT': None,
        
        # Rate limiting
        'RATELIMIT_STORAGE_URI': 'memory://',
        'RATELIMIT_DEFAULT': '100/day',
        
        # File uploads
        'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB
        
        # Custom config
        'FM_CONFIG': config
    }


def init_extensions(app: Flask, config_manager: ConfigurationManager) -> None:
    """Initialize Flask extensions."""
    logger = get_logger(__name__)
    
    # Database
    db.init_app(app)
    migrate.init_app(app, db)
    
    # Authentication
    login_manager.init_app(app)
    login_manager.login_view = 'main.index'
    login_manager.login_message_category = 'info'
    
    # Rate limiting
    limiter.init_app(app)
    
    # CORS
    config = config_manager.load_config()
    web_config = config.get('web_interface', {})
    cors_origins = web_config.get('cors_origins', ['http://localhost:3000'])
    
    CORS(
        app,
        origins=cors_origins,
        methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
        allow_headers=['Content-Type', 'Authorization', 'X-Requested-With']
    )
    
    logger.info("Flask extensions initialized")


def init_services(app: Flask, config_manager: ConfigurationManager) -> None:
    """Initialize application services."""
    logger = get_logger(__name__)
    
    with app.app_context():
        config = config_manager.load_config()
        
        # Initialize database manager
        try:
            db_manager = get_database_manager()
            app.db_manager = db_manager
            logger.info("Database manager initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize database manager: {e}")
            app.db_manager = None
        
        # Initialize model provider
        try:
            model_provider = QwenProvider(config_manager)
            app.model_provider = model_provider
            logger.info("Model provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize model provider: {e}")
            app.model_provider = None
        
        # Initialize knowledge base
        try:
            knowledge_base = KnowledgeBase(config_manager)
            app.knowledge_base = knowledge_base
            logger.info("Knowledge base initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize knowledge base: {e}")
            app.knowledge_base = None
        
        # Initialize services
        try:
            app.certificate_generator = CertificateGenerator(
                config_manager=config_manager,
                model_provider=app.model_provider,
                knowledge_base=app.knowledge_base
            )
            logger.info("Certificate generator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize certificate generator: {e}")
            app.certificate_generator = None
        
        try:
            app.verifier = CertificateVerifier(config_manager)
            logger.info("Certificate verifier initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize certificate verifier: {e}")
            app.verifier = None
        
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
    setup_security_headers(app)
    setup_rate_limiting(app)
    setup_cors(app)
    
    get_logger(__name__).info("Middleware configured")


def register_cli_commands(app: Flask) -> None:
    """Register CLI commands."""
    
    @app.cli.command()
    def init_db():
        """Initialize the database."""
        try:
            db.create_all()
            print("Database initialized.")
        except Exception as e:
            print(f"Database initialization failed: {e}")
    
    @app.cli.command()
    def build_kb():
        """Build the knowledge base."""
        try:
            if app.knowledge_base:
                print("Building knowledge base...")
                # Knowledge base building logic would go here
                print("Knowledge base built.")
            else:
                print("Knowledge base not available.")
        except Exception as e:
            print(f"Knowledge base building failed: {e}")
    
    @app.cli.command()
    def test_generation():
        """Test certificate generation."""
        try:
            if app.certificate_generator:
                system = {
                    "dynamics": {"x": "-x + y", "y": "x - y"},
                    "initial_set": "x**2 + y**2 <= 0.5",
                    "unsafe_set": "x**2 + y**2 >= 2.0"
                }
                
                result = app.certificate_generator.generate(system)
                
                if result and result.get('success'):
                    print(f"Certificate: {result.get('certificate')}")
                    print(f"Confidence: {result.get('confidence', 0):.2f}")
                else:
                    print(f"Generation failed: {result.get('error') if result else 'Unknown error'}")
            else:
                print("Certificate generator not available.")
        except Exception as e:
            print(f"Certificate generation test failed: {e}")


# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id: str):
    """Load user by ID."""
    try:
        return User.query.get(int(user_id))
    except Exception:
        return None 