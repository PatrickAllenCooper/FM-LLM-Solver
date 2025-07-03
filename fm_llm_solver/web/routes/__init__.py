"""
Route blueprints for FM-LLM Solver web interface.

Organizes routes into logical modules for better maintainability.
"""

from .main import main_bp
from .api import api_bp
from .auth import auth_bp
from .monitoring import monitoring_bp

__all__ = [
    "main_bp",
    "api_bp", 
    "auth_bp",
    "monitoring_bp"
] 