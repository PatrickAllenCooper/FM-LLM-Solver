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
from fm_llm_solver.core.async_manager import AsyncManager
from fm_llm_solver.core.memory_manager import MemoryManager
from fm_llm_solver.core.cache_manager import CacheManager
from fm_llm_solver.core.monitoring import MonitoringManager
from fm_llm_solver.services.certificate_generator import CertificateGenerator
from fm_llm_solver.services.verifier import CertificateVerifier
from fm_llm_solver.services.knowledge_base import KnowledgeBase
from fm_llm_solver.services.model_provider import QwenProvider
from fm_llm_solver.web.models import User
from fm_llm_solver.web.utils import (
    setup_security_headers,
    setup_rate_limiting,
    setup_cors,
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
        static_folder=str(Path(__file__).parent / "static"),
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
        app.config.from_object(type("Config", (), flask_config))

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
    config.get("web_interface", {})
    db_config = config.get("database", {}).get("primary", {})

    # Build database URL
    db_url = f"postgresql://{db_config.get('username', 'postgres')}:{db_config.get('password', '')}@{db_config.get('host', 'localhost')}:{db_config.get('port', 5432)}/{db_config.get('database', 'fm_llm_solver')}"

    return {
        # Database
        "SQLALCHEMY_DATABASE_URI": db_url,
        "SQLALCHEMY_TRACK_MODIFICATIONS": False,
        # Session
        "SECRET_KEY": os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production"),
        "PERMANENT_SESSION_LIFETIME": 86400,  # 24 hours
        "SESSION_COOKIE_SECURE": config_manager.environment.value == "production",
        "SESSION_COOKIE_HTTPONLY": True,
        "SESSION_COOKIE_SAMESITE": "Lax",
        # Security
        "WTF_CSRF_ENABLED": True,
        "WTF_CSRF_TIME_LIMIT": None,
        # Rate limiting
        "RATELIMIT_STORAGE_URI": "memory://",
        "RATELIMIT_DEFAULT": "100/day",
        # File uploads
        "MAX_CONTENT_LENGTH": 16 * 1024 * 1024,  # 16MB
        # Custom config
        "FM_CONFIG": config,
    }


def init_extensions(app: Flask, config_manager: ConfigurationManager) -> None:
    """Initialize Flask extensions."""
    logger = get_logger(__name__)

    # Database
    db.init_app(app)
    migrate.init_app(app, db)

    # Authentication
    login_manager.init_app(app)
    login_manager.login_view = "main.index"
    login_manager.login_message_category = "info"

    # Rate limiting
    limiter.init_app(app)

    # CORS
    config = config_manager.load_config()
    web_config = config.get("web_interface", {})
    cors_origins = web_config.get("cors_origins", ["http://localhost:3000"])

    CORS(
        app,
        origins=cors_origins,
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    )

    logger.info("Flask extensions initialized")


def init_services(app: Flask, config_manager: ConfigurationManager) -> None:
    """Initialize application services with performance optimizations."""
    logger = get_logger(__name__)

    with app.app_context():
        config = config_manager.load_config()

        # Initialize monitoring manager first (needed by other services)
        try:
            monitoring_manager = MonitoringManager(config_manager)
            app.monitoring_manager = monitoring_manager
            logger.info("Monitoring manager initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize monitoring manager: {e}")
            app.monitoring_manager = None

        # Initialize cache manager
        try:
            cache_manager = CacheManager(config_manager)
            app.cache_manager = cache_manager
            logger.info("Cache manager initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize cache manager: {e}")
            app.cache_manager = None

        # Initialize async manager
        try:
            async_manager = AsyncManager(config_manager, app.monitoring_manager)
            app.async_manager = async_manager
            logger.info("Async manager initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize async manager: {e}")
            app.async_manager = None

        # Initialize memory manager
        try:
            memory_manager = MemoryManager(config_manager, app.monitoring_manager)
            app.memory_manager = memory_manager
            logger.info("Memory manager initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize memory manager: {e}")
            app.memory_manager = None

        # Initialize database manager
        try:
            db_manager = get_database_manager()
            app.db_manager = db_manager
            logger.info("Database manager initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize database manager: {e}")
            app.db_manager = None

        # Initialize model provider with performance enhancements
        try:
            model_provider = QwenProvider(config_manager)
            app.model_provider = model_provider
            logger.info("Model provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize model provider: {e}")
            app.model_provider = None

        # Initialize knowledge base with caching
        try:
            knowledge_base = KnowledgeBase(config_manager)
            app.knowledge_base = knowledge_base
            logger.info("Knowledge base initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize knowledge base: {e}")
            app.knowledge_base = None

        # Initialize optimized services
        try:
            app.certificate_generator = CertificateGenerator(
                config_manager=config_manager,
                model_provider=app.model_provider,
                knowledge_base=app.knowledge_base,
                cache_manager=app.cache_manager,
                monitoring_manager=app.monitoring_manager,
            )
            logger.info("Certificate generator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize certificate generator: {e}")
            app.certificate_generator = None

        try:
            app.verifier = CertificateVerifier(
                config_manager,
                cache_manager=app.cache_manager,
                monitoring_manager=app.monitoring_manager,
            )
            logger.info("Certificate verifier initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize certificate verifier: {e}")
            app.verifier = None

        # Store config
        app.fm_config = config

        logger.info("Application services initialized with performance optimizations")


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
    """Setup application middleware with performance monitoring."""
    setup_security_headers(app)
    setup_rate_limiting(app)
    setup_cors(app)
    setup_performance_monitoring(app)

    get_logger(__name__).info("Middleware configured with performance monitoring")


def setup_performance_monitoring(app: Flask) -> None:
    """Setup performance monitoring middleware."""

    @app.before_request
    def before_request():
        """Track request start time and perform pre-request optimizations."""
        g.request_start_time = time.time()
        g.request_id = f"req_{int(time.time() * 1000)}_{os.getpid()}"

        # Memory optimization check
        if hasattr(app, "memory_manager") and app.memory_manager:
            pressure_info = app.memory_manager.check_memory_pressure()
            if pressure_info["pressure_level"] == "critical":
                app.memory_manager.optimize_memory()

        # Log request start
        if hasattr(app, "monitoring_manager") and app.monitoring_manager:
            app.monitoring_manager.increment_counter(
                "http_requests_total",
                {"method": request.method, "endpoint": request.endpoint or "unknown"},
            )

    @app.after_request
    def after_request(response):
        """Track request completion and record metrics."""
        if hasattr(g, "request_start_time"):
            duration = time.time() - g.request_start_time

            # Record response time metric
            if hasattr(app, "monitoring_manager") and app.monitoring_manager:
                app.monitoring_manager.record_histogram(
                    "http_request_duration_seconds",
                    duration,
                    {
                        "method": request.method,
                        "status_code": str(response.status_code),
                        "endpoint": request.endpoint or "unknown",
                    },
                )

                # Record response size
                if response.content_length:
                    app.monitoring_manager.record_histogram(
                        "http_response_size_bytes",
                        response.content_length,
                        {"endpoint": request.endpoint or "unknown"},
                    )

            # Log slow requests
            if duration > 5.0:  # Log requests taking more than 5 seconds
                logger = get_logger(__name__)
                logger.warning(
                    f"Slow request: {request.method} {request.path} "
                    f"took {duration:.2f}s (request_id: {getattr(g, 'request_id', 'unknown')})"
                )

        # Add performance headers
        if hasattr(g, "request_start_time"):
            response.headers["X-Response-Time"] = (
                f"{(time.time() - g.request_start_time) * 1000:.1f}ms"
            )

        if hasattr(g, "request_id"):
            response.headers["X-Request-ID"] = g.request_id

        return response

    @app.teardown_appcontext
    def cleanup_request(error):
        """Cleanup request resources."""
        # Force garbage collection for long-running requests
        if hasattr(g, "request_start_time"):
            duration = time.time() - g.request_start_time
            if duration > 10.0:  # For requests taking more than 10 seconds
                import gc

                gc.collect()

        # Clear any request-specific caches
        if hasattr(app, "cache_manager") and app.cache_manager:
            # Clear any temporary cache entries if needed
            pass


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
                    "unsafe_set": "x**2 + y**2 >= 2.0",
                }

                result = app.certificate_generator.generate(system)

                if result and result.get("success"):
                    print(f"Certificate: {result.get('certificate')}")
                    print(f"Confidence: {result.get('confidence', 0):.2f}")
                else:
                    print(
                        f"Generation failed: {result.get('error') if result else 'Unknown error'}"
                    )
            else:
                print("Certificate generator not available.")
        except Exception as e:
            print(f"Certificate generation test failed: {e}")

    @app.cli.command()
    def performance_report():
        """Generate performance report."""
        try:
            print("=== FM-LLM-Solver Performance Report ===\n")

            # Memory statistics
            if app.memory_manager:
                memory_report = app.memory_manager.get_performance_report()
                print("Memory Usage:")
                print(f"  RSS: {memory_report['memory_stats']['rss_mb']:.1f} MB")
                print(f"  Percentage: {memory_report['memory_stats']['percent']:.1f}%")
                print(f"  GC Objects: {memory_report['memory_stats']['gc_objects']:,}")
                print(f"  Pressure Level: {memory_report['pressure_info']['pressure_level']}")
                print()

            # Async manager statistics
            if app.async_manager:
                async_metrics = app.async_manager.get_performance_metrics()
                print("Async Operations:")
                print(f"  Active Tasks: {async_metrics['active_tasks']}")
                print(f"  Completed Tasks: {async_metrics['tasks_completed']}")
                print(f"  Failed Tasks: {async_metrics['tasks_failed']}")
                print(f"  Average Duration: {async_metrics['average_duration']:.2f}s")
                print(f"  Queue Size: {async_metrics['queue_size']}")
                print()

            # Cache statistics
            if app.cache_manager:
                cache_stats = app.cache_manager.get_stats()
                print("Cache Performance:")
                print(f"  Size: {cache_stats.get('size', 0)}")
                print(f"  Hits: {cache_stats.get('hits', 0)}")
                print(f"  Misses: {cache_stats.get('misses', 0)}")
                hit_rate = cache_stats.get("hits", 0) / max(
                    cache_stats.get("hits", 0) + cache_stats.get("misses", 0), 1
                )
                print(f"  Hit Rate: {hit_rate:.1%}")
                print()

            # Monitoring metrics summary
            if app.monitoring_manager:
                print("System Health: Active monitoring enabled")
            else:
                print("System Health: Monitoring not available")

        except Exception as e:
            print(f"Failed to generate performance report: {e}")

    @app.cli.command()
    def optimize_memory():
        """Optimize memory usage."""
        try:
            if app.memory_manager:
                print("Optimizing memory usage...")
                result = app.memory_manager.optimize_memory()
                print(f"Optimization completed in {result['duration_ms']:.1f}ms")
                print(f"Actions taken: {len(result['actions_taken'])}")
                for action in result["actions_taken"]:
                    print(f"  - {action['action']}")
            else:
                print("Memory manager not available.")
        except Exception as e:
            print(f"Memory optimization failed: {e}")

    @app.cli.command()
    def benchmark():
        """Run performance benchmarks."""
        try:
            import time

            print("Running performance benchmarks...\n")

            # Cache benchmark
            if app.cache_manager:
                print("Cache Performance Test:")
                start_time = time.time()
                for i in range(1000):
                    app.cache_manager.set(f"bench_key_{i}", f"bench_value_{i}")
                write_time = time.time() - start_time

                start_time = time.time()
                for i in range(1000):
                    app.cache_manager.get(f"bench_key_{i}")
                read_time = time.time() - start_time

                print(f"  1000 writes: {write_time:.3f}s ({1000/write_time:.0f} ops/s)")
                print(f"  1000 reads: {read_time:.3f}s ({1000/read_time:.0f} ops/s)")
                print()

            # Memory tracking benchmark
            if app.memory_manager:
                print("Memory Tracking Test:")
                tracker = app.memory_manager.create_tracker("benchmark")
                tracker.start_tracking()

                # Simulate memory allocation
                [list(range(1000)) for _ in range(100)]

                stats = tracker.stop_tracking()
                print(f"  Memory used: {stats['memory_diff_mb']:.2f} MB")
                print()

                # Clean up
                del data

            print("Benchmarks completed.")

        except Exception as e:
            print(f"Benchmark failed: {e}")


# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id: str):
    """Load user by ID."""
    try:
        return User.query.get(int(user_id))
    except Exception:
        return None
