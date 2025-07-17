#!/usr/bin/env python3
"""
FM-LLM Solver Web Interface - Web-Only Version
Runs the web interface without AI/ML dependencies for hybrid deployment.
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
        ]
    )

def check_dependencies():
    """Check if required dependencies are available - Web-only version."""
    missing_deps = []
    
    # Only check web dependencies
    web_deps = ['flask', 'flask_sqlalchemy', 'psycopg2', 'redis']
    
    for dep in web_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)
    
    if missing_deps:
        print("Error: Missing required web dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nPlease install missing dependencies:")
        print("  pip install flask flask-sqlalchemy psycopg2-binary redis")
        return False
    
    return True

def main():
    """Main entry point for the web interface."""
    parser = argparse.ArgumentParser(description="FM-LLM Solver Web Interface")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    logging.info("Starting FM-LLM Solver Web Interface (Web-Only Mode)")
    logging.info(f"Configuration file: {args.config}")
    
    # Check web dependencies only
    if not check_dependencies():
        sys.exit(1)
    
    # Import and start web application
    try:
        from web_interface.app import create_app
        app = create_app(config_file=args.config)
        
        logging.info(f"Starting web server on {args.host}:{args.port}")
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
    except Exception as e:
        logging.error(f"Failed to start web interface: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
