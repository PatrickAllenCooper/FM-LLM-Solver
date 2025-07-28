"""
FM-LLM-Solver Comprehensive API

This module provides clean, comprehensive API endpoints for authorized external access
to all core functions as specified in the GuidanceDoc.

API Features:
- Authentication and authorization
- Certificate generation with all model types
- RAG dataset configuration
- User management
- Query history and monitoring
- System status and health checks
"""

from .auth import APIAuth
from .endpoints import APIEndpoints
from .middleware import APIMiddleware
from .schemas import APISchemas

__all__ = [
    'APIAuth',
    'APIEndpoints', 
    'APIMiddleware',
    'APISchemas'
] 