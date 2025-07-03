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


class DatabaseError(FMLLMSolverError):
    """Raised when database operations fail."""
    
    def __init__(self, message: str, operation: Optional[str] = None, 
                 table: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        if operation:
            self.details["operation"] = operation
        if table:
            self.details["table"] = table


class CacheError(FMLLMSolverError):
    """Raised when cache operations fail."""
    
    def __init__(self, message: str, cache_key: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        if cache_key:
            self.details["cache_key"] = cache_key


class ServiceUnavailableError(FMLLMSolverError):
    """Raised when a service is unavailable."""
    
    def __init__(self, message: str, service_name: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        if service_name:
            self.details["service"] = service_name


class SecurityError(FMLLMSolverError):
    """Raised when security violations occur."""
    
    def __init__(self, message: str, violation_type: Optional[str] = None, 
                 user_id: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        if violation_type:
            self.details["violation_type"] = violation_type
        if user_id:
            self.details["user_id"] = user_id


class DataIntegrityError(FMLLMSolverError):
    """Raised when data integrity is compromised."""
    
    def __init__(self, message: str, data_type: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        if data_type:
            self.details["data_type"] = data_type


class ExternalServiceError(FMLLMSolverError):
    """Raised when external service calls fail."""
    
    def __init__(self, message: str, service_name: Optional[str] = None, 
                 response_code: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        if service_name:
            self.details["service"] = service_name
        if response_code:
            self.details["response_code"] = response_code


class ParseError(FMLLMSolverError):
    """Raised when parsing fails."""
    
    def __init__(self, message: str, parser_type: Optional[str] = None, 
                 input_type: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        if parser_type:
            self.details["parser_type"] = parser_type
        if input_type:
            self.details["input_type"] = input_type


class RetryableError(FMLLMSolverError):
    """Base class for errors that can be retried."""
    
    def __init__(self, message: str, retry_after: Optional[float] = None, 
                 max_retries: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        if retry_after:
            self.details["retry_after"] = retry_after
        if max_retries:
            self.details["max_retries"] = max_retries


class NonRetryableError(FMLLMSolverError):
    """Base class for errors that should not be retried."""
    pass 