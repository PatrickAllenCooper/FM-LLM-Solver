"""
Web Interface Package for FM-LLM Solver.

Optimized imports and centralized initialization.
"""

import os
import sys
from typing import Optional

# ============================================================================
# Path Management (Centralized)
# ============================================================================
def setup_project_path():
    """Add project root to Python path if not already present."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root

# Initialize once at package level
PROJECT_ROOT = setup_project_path()

# ============================================================================
# Lazy Import Utilities
# ============================================================================
class LazyImport:
    """Lazy import utility to defer heavy imports until needed."""
    
    def __init__(self, module_name, fallback=None):
        self.module_name = module_name
        self.fallback = fallback
        self._module = None
    
    def __getattr__(self, name):
        if self._module is None:
            try:
                self._module = __import__(self.module_name, fromlist=[name])
            except ImportError:
                if self.fallback:
                    self._module = self.fallback()
                else:
                    raise
        return getattr(self._module, name)

# ============================================================================
# Common Service Imports (Cached)
# ============================================================================
_cached_imports = {}

def get_config_loader():
    """Get config loader with caching."""
    if 'config_loader' not in _cached_imports:
        try:
            from utils.config_loader import load_config
            _cached_imports['config_loader'] = load_config
        except ImportError:
            # Fallback configuration
            def fallback_config():
                return {
                    "web_interface": {
                        "database_path": "instance/test.db",
                        "host": "127.0.0.1",
                        "port": 5000,
                        "debug": True,
                    }
                }
            _cached_imports['config_loader'] = fallback_config
    return _cached_imports['config_loader']

def get_certificate_generator():
    """Get certificate generator with lazy loading."""
    if 'certificate_generator' not in _cached_imports:
        from .certificate_generator import CertificateGenerator
        _cached_imports['certificate_generator'] = CertificateGenerator
    return _cached_imports['certificate_generator']

def get_verification_service():
    """Get verification service with lazy loading."""
    if 'verification_service' not in _cached_imports:
        try:
            from .verification_service import VerificationService
            _cached_imports['verification_service'] = VerificationService
        except ImportError:
            # Placeholder service
            class PlaceholderVerificationService:
                def __init__(self, config):
                    self.config = config
            _cached_imports['verification_service'] = PlaceholderVerificationService
    return _cached_imports['verification_service']

# ============================================================================
# Package Metadata
# ============================================================================
__version__ = "2.0.0"
__title__ = "FM-LLM Solver Web Interface"
__description__ = "Web interface for barrier certificate generation using LLMs"

# Export commonly used items
__all__ = [
    'PROJECT_ROOT',
    'LazyImport', 
    'get_config_loader',
    'get_certificate_generator',
    'get_verification_service',
]
