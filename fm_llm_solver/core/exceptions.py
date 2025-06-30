"""
Exception hierarchy for FM-LLM Solver.

Provides structured exception handling throughout the system.
"""

from typing import Optional, Dict, Any


class FMLLMSolverError(Exception):
    """Base exception for all FM-LLM Solver errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error context
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }


class ConfigurationError(FMLLMSolverError):
    """Raised when configuration is invalid or missing."""
    pass


class ValidationError(FMLLMSolverError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        if field:
            self.details["field"] = field


class GenerationError(FMLLMSolverError):
    """Raised when certificate generation fails."""
    pass


class ModelError(GenerationError):
    """Raised when model operations fail."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        if model_name:
            self.details["model"] = model_name


class VerificationError(FMLLMSolverError):
    """Raised when certificate verification fails."""
    pass


class KnowledgeBaseError(FMLLMSolverError):
    """Raised when knowledge base operations fail."""
    pass


class APIError(FMLLMSolverError):
    """Raised when API operations fail."""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.details["status_code"] = status_code


class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", **kwargs):
        super().__init__(message, status_code=429, **kwargs)


class AuthenticationError(APIError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication required", **kwargs):
        super().__init__(message, status_code=401, **kwargs)


class AuthorizationError(APIError):
    """Raised when authorization fails."""
    
    def __init__(self, message: str = "Insufficient permissions", **kwargs):
        super().__init__(message, status_code=403, **kwargs)


class TimeoutError(FMLLMSolverError):
    """Raised when an operation times out."""
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        if timeout_seconds:
            self.details["timeout_seconds"] = timeout_seconds


class ResourceNotFoundError(FMLLMSolverError):
    """Raised when a requested resource is not found."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, 
                 resource_id: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        if resource_type:
            self.details["resource_type"] = resource_type
        if resource_id:
            self.details["resource_id"] = resource_id 