#!/usr/bin/env python3
"""
Startup script for the FM-LLM Solver Web Interface.

This script launches the web interface for barrier certificate generation
using the existing project configuration and infrastructure.

Usage:
    python run_web_interface.py [options]

Options:
    --config PATH       Path to configuration file (default: config.yaml)
    --host HOST         Host to bind to (default: from config)
    --port PORT         Port to bind to (default: from config)
    --debug             Enable debug mode (default: from config)
    --no-debug          Disable debug mode
    --help              Show this help message

Examples:
    # Run with default settings
    python run_web_interface.py
    
    # Run on different port
    python run_web_interface.py --port 8080
    
    # Run with debug disabled
    python run_web_interface.py --no-debug
    
    # Run with custom config
    python run_web_interface.py --config my_config.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

def setup_logging(debug=False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('web_interface.log')
        ]
    )

def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []
    
    try:
        import flask
    except ImportError:
        missing_deps.append('flask')
    
    try:
        import flask_sqlalchemy
    except ImportError:
        missing_deps.append('flask-sqlalchemy')
    
    try:
        import torch
    except ImportError:
        missing_deps.append('torch')
    
    try:
        import transformers
    except ImportError:
        missing_deps.append('transformers')
    
    try:
        import sentence_transformers
    except ImportError:
        missing_deps.append('sentence-transformers')
    
    try:
        import faiss
    except ImportError:
        missing_deps.append('faiss-cpu or faiss-gpu')
    
    if missing_deps:
        print("Error: Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nPlease install missing dependencies:")
        print("  pip install flask flask-sqlalchemy")
        print("  # Install other dependencies as per project requirements")
        return False
    
    return True

def check_project_setup(config):
    """Check if the project is properly set up."""
    issues = []
    warnings = []
    
    # Check if knowledge base exists (make this a warning, not a blocker)
    try:
        from knowledge_base.kb_utils import list_available_kbs
        available_kbs = list_available_kbs(config)
        if not any(available_kbs.values()):
            warnings.append("No knowledge base found. RAG functionality will be limited. Run knowledge_base/knowledge_base_builder.py to build one.")
    except Exception as e:
        warnings.append(f"Could not check knowledge base status: {e}. RAG functionality may be limited.")
    
    # Check if at least base model is available
    try:
        base_model = config.fine_tuning.base_model_name
        if not base_model:
            issues.append("No base model configured in fine_tuning.base_model_name")
    except Exception as e:
        warnings.append(f"Could not check model configuration: {e}")
    
    # Check and create output directories
    try:
        os.makedirs(config.paths.ft_output_dir, exist_ok=True)
        
        # Create database directory with proper permissions
        db_path = config.get('web_interface', {}).get('database_path', 'web_interface/instance/app.db')
        db_dir = os.path.dirname(db_path)
        os.makedirs(db_dir, exist_ok=True)
        
        # Test write permissions
        test_file = os.path.join(db_dir, 'test_write.tmp')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            issues.append(f"Cannot write to database directory {db_dir}: {e}")
            
    except Exception as e:
        issues.append(f"Could not create required directories: {e}")
    
    # Show warnings (non-blocking)
    if warnings:
        print("Notice: Setup recommendations:")
        for warning in warnings:
            print(f"  - {warning}")
        print()
    
    # Show critical issues (blocking)
    if issues:
        print("Error: Critical setup issues detected:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease fix these issues before starting the web interface.")
        return False
    
    return True

def main():
    """Main function to launch the web interface."""
    parser = argparse.ArgumentParser(
        description="Launch the FM-LLM Solver Web Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Usage:')[1] if 'Usage:' in __doc__ else ""
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--host', 
        type=str,
        help='Host to bind to (overrides config)'
    )
    parser.add_argument(
        '--port', 
        type=int,
        help='Port to bind to (overrides config)'
    )
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode (overrides config)'
    )
    parser.add_argument(
        '--no-debug', 
        action='store_true',
        help='Disable debug mode (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        from utils.config_loader import load_config
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration from {args.config}: {e}")
        sys.exit(1)
    
    # Determine debug mode
    debug = config.get('web_interface', {}).get('debug', True)
    if args.debug:
        debug = True
    elif args.no_debug:
        debug = False
    
    # Setup logging
    setup_logging(debug)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting FM-LLM Solver Web Interface")
    logger.info(f"Configuration file: {args.config}")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check project setup
    if not check_project_setup(config):
        sys.exit(1)
    
    # Set environment variables
    if 'SECRET_KEY' not in os.environ:
        # Generate a random secret key for development
        import secrets
        secret_key = secrets.token_hex(32)
        os.environ['SECRET_KEY'] = secret_key
        logger.warning("Generated temporary SECRET_KEY. Set SECRET_KEY environment variable for production.")
    
    # Import and configure Flask app
    try:
        from web_interface.app import app
        
        # Override config with command line arguments
        web_config = config.get('web_interface', {})
        host = args.host or web_config.get('host', '127.0.0.1')
        port = args.port or web_config.get('port', 5000)
        
        logger.info(f"Starting server on http://{host}:{port}")
        logger.info(f"Debug mode: {debug}")
        
        if debug:
            logger.info("Web interface accessible at:")
            logger.info(f"  Home: http://{host}:{port}/")
            logger.info(f"  History: http://{host}:{port}/history")
            logger.info(f"  About: http://{host}:{port}/about")
        
        # Create database tables
        with app.app_context():
            from web_interface.models import db
            db.create_all()
            logger.info("Database initialized")
        
        # Start the Flask development server
        app.run(
            host=host,
            port=port,
            debug=debug,
            use_reloader=debug,
            threaded=True
        )
        
    except ImportError as e:
        logger.error(f"Failed to import web interface modules: {e}")
        logger.error("Make sure all dependencies are installed and the project structure is correct.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start web interface: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 