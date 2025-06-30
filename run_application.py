#!/usr/bin/env python3
"""
Main entry point for FM-LLM Solver application.

This script starts the web interface and/or inference API based on configuration.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from fm_llm_solver import __version__, configure_logging, load_config
from fm_llm_solver.core.exceptions import ConfigurationError


def run_web_interface(config_path: str, host: str, port: int, debug: bool):
    """Run the Flask web interface."""
    from fm_llm_solver.web import create_app
    
    print(f"Starting FM-LLM Solver Web Interface v{__version__}")
    
    try:
        # Create and run app
        app = create_app(config_path)
        
        # Override debug if specified
        if debug:
            app.debug = True
        
        app.run(host=host, port=port, debug=app.debug)
        
    except Exception as e:
        print(f"Failed to start web interface: {e}")
        sys.exit(1)


def run_inference_api(config_path: str, host: str, port: int):
    """Run the FastAPI inference API."""
    import uvicorn
    
    print(f"Starting FM-LLM Solver Inference API v{__version__}")
    
    try:
        # Set config path in environment for FastAPI app
        os.environ["FM_LLM_CONFIG"] = config_path
        
        # Run with uvicorn
        uvicorn.run(
            "fm_llm_solver.api:app",
            host=host,
            port=port,
            reload=False,
            log_config=None  # Use our logging config
        )
        
    except Exception as e:
        print(f"Failed to start inference API: {e}")
        sys.exit(1)


def run_both(config_path: str, web_host: str, web_port: int, 
             api_host: str, api_port: int, debug: bool):
    """Run both web interface and inference API."""
    import multiprocessing
    
    print(f"Starting FM-LLM Solver v{__version__} (Web + API)")
    
    # Create processes
    web_process = multiprocessing.Process(
        target=run_web_interface,
        args=(config_path, web_host, web_port, debug)
    )
    
    api_process = multiprocessing.Process(
        target=run_inference_api,
        args=(config_path, api_host, api_port)
    )
    
    try:
        # Start processes
        web_process.start()
        api_process.start()
        
        # Wait for processes
        web_process.join()
        api_process.join()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
        web_process.terminate()
        api_process.terminate()
        
        web_process.join()
        api_process.join()
        
        print("Shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=f"FM-LLM Solver v{__version__} - Barrier Certificate Generation"
    )
    
    # Common arguments
    parser.add_argument(
        "-c", "--config",
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Web interface
    web_parser = subparsers.add_parser("web", help="Run web interface")
    web_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    web_parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to (default: 5000)"
    )
    web_parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )
    
    # Inference API
    api_parser = subparsers.add_parser("api", help="Run inference API")
    api_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    api_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    
    # Both
    both_parser = subparsers.add_parser("both", help="Run both web interface and API")
    both_parser.add_argument(
        "--web-host",
        default="127.0.0.1",
        help="Web interface host (default: 127.0.0.1)"
    )
    both_parser.add_argument(
        "--web-port",
        type=int,
        default=5000,
        help="Web interface port (default: 5000)"
    )
    both_parser.add_argument(
        "--api-host",
        default="127.0.0.1",
        help="API host (default: 127.0.0.1)"
    )
    both_parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="API port (default: 8000)"
    )
    both_parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )
    
    # Build knowledge base
    kb_parser = subparsers.add_parser("build-kb", help="Build knowledge base")
    kb_parser.add_argument(
        "--papers-dir",
        help="Directory containing papers to process"
    )
    kb_parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if knowledge base exists"
    )
    
    # Run tests
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage report"
    )
    test_parser.add_argument(
        "tests",
        nargs="*",
        help="Specific test files or directories to run"
    )
    
    args = parser.parse_args()
    
    # Default to web if no command specified
    if not args.command:
        args.command = "web"
    
    # Load and validate configuration
    try:
        config = load_config(args.config)
        
        # Configure logging early
        configure_logging(
            level=config.logging.level,
            log_dir=config.paths.log_dir,
            console=config.logging.console,
            structured=config.logging.structured
        )
        
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Execute command
    if args.command == "web":
        run_web_interface(args.config, args.host, args.port, args.debug)
    
    elif args.command == "api":
        run_inference_api(args.config, args.host, args.port)
    
    elif args.command == "both":
        run_both(
            args.config,
            args.web_host, args.web_port,
            args.api_host, args.api_port,
            args.debug
        )
    
    elif args.command == "build-kb":
        from fm_llm_solver.services.knowledge_base_builder import build_knowledge_base
        
        print("Building knowledge base...")
        
        try:
            build_knowledge_base(
                config,
                papers_dir=args.papers_dir,
                force_rebuild=args.force
            )
            print("Knowledge base built successfully")
            
        except Exception as e:
            print(f"Failed to build knowledge base: {e}")
            sys.exit(1)
    
    elif args.command == "test":
        import subprocess
        
        # Build pytest command
        cmd = ["pytest", "-v"]
        
        if args.coverage:
            cmd.extend(["--cov=fm_llm_solver", "--cov-report=html", "--cov-report=term"])
        
        if args.tests:
            cmd.extend(args.tests)
        else:
            cmd.append("tests/")
        
        # Run tests
        result = subprocess.run(cmd)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main() 