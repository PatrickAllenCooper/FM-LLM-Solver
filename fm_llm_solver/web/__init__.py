"""
Web interface module for FM-LLM Solver.

Provides a Flask-based web application for interacting with the system.
"""

from fm_llm_solver.web.app import create_app
from fm_llm_solver.web.routes import (
    register_api_routes,
    register_auth_routes,
    register_main_routes,
    register_monitoring_routes,
)

__all__ = [
    "create_app",
    "register_main_routes",
    "register_api_routes",
    "register_auth_routes",
    "register_monitoring_routes",
]
