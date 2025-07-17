"""Exception handling for DDoS.AI platform"""
from typing import Any, Dict, Optional, List, Union
from fastapi import HTTPException, status
import logging
import traceback
import sys

# Configure logging
logger = logging.getLogger(__name__)


class DDoSAIException(Exception):
    """Base exception for DDoS.AI platform"""
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary"""
        return {
            "error": self.message,
            "status_code": self.status_code,
            "details": self.details,
            "error_type": self.__class__.__name__
        }


class ValidationError(DDoSAIException):
    """Exception for validation errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, details=details)


class NotFoundError(DDoSAIException):
    """Exception for resource not found"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=status.HTTP_404_NOT_FOUND, details=details)


class DatabaseError(DDoSAIException):
    """Exception for database errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, details=details)


class AIModelError(DDoSAIException):
    """Exception for AI model errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, details=details)


class AIModelNotFoundError(AIModelError):
    """Exception for AI model not found"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=status.HTTP_404_NOT_FOUND, details=details)


class TrafficIngestionError(DDoSAIException):
    """Errors during traffic ingestion and parsing"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=status.HTTP_400_BAD_REQUEST, details=details)


class FeatureExtractionError(DDoSAIException):
    """Errors during feature extraction from packets"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, details=details)


class SimulationError(DDoSAIException):
    """Exception for simulation errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, details=details)


class ConfigurationError(DDoSAIException):
    """Exception for configuration errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, details=details)


class WebSocketError(DDoSAIException):
    """Exception for WebSocket errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, details=details)


class RateLimitError(DDoSAIException):
    """Exception for rate limiting"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=status.HTTP_429_TOO_MANY_REQUESTS, details=details)


class AuthenticationError(DDoSAIException):
    """Exception for authentication errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=status.HTTP_401_UNAUTHORIZED, details=details)


class AuthorizationError(DDoSAIException):
    """Exception for authorization errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=status.HTTP_403_FORBIDDEN, details=details)


class ExternalServiceError(DDoSAIException):
    """Exception for external service errors"""
    def __init__(self, message: str, service_name: str, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["service_name"] = service_name
        super().__init__(message, status_code=status.HTTP_502_BAD_GATEWAY, details=details)


class TimeoutError(DDoSAIException):
    """Exception for timeout errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=status.HTTP_504_GATEWAY_TIMEOUT, details=details)


class CircuitBreakerOpenError(DDoSAIException):
    """Exception for when circuit breaker is open"""
    def __init__(self, message: str, retry_after: int = 30, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["retry_after"] = retry_after
        super().__init__(message, status_code=status.HTTP_503_SERVICE_UNAVAILABLE, details=details)


class ResourceExhaustedError(DDoSAIException):
    """Exception for resource exhaustion (memory, CPU, etc.)"""
    def __init__(self, message: str, resource_type: str, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["resource_type"] = resource_type
        super().__init__(message, status_code=status.HTTP_503_SERVICE_UNAVAILABLE, details=details)


class ModelOverloadedError(DDoSAIException):
    """Exception for when AI models are overloaded"""
    def __init__(self, message: str, model_name: str, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["model_name"] = model_name
        super().__init__(message, status_code=status.HTTP_503_SERVICE_UNAVAILABLE, details=details)


class NetworkError(DDoSAIException):
    """Exception for network-related errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=status.HTTP_502_BAD_GATEWAY, details=details)


class RetryExhaustedError(DDoSAIException):
    """Exception for when all retry attempts have been exhausted"""
    def __init__(self, message: str, operation: str, attempts: int, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["operation"] = operation
        details["attempts"] = attempts
        super().__init__(message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, details=details)


class ErrorHandler:
    """Error handler for DDoS.AI platform"""
    
    @staticmethod
    def log_exception(exc: Exception, include_traceback: bool = True) -> None:
        """Log an exception"""
        if isinstance(exc, DDoSAIException):
            logger.error(
                f"{exc.__class__.__name__}: {exc.message} (Status: {exc.status_code})"
                f" Details: {exc.details}"
            )
        else:
            logger.error(f"Unhandled exception: {str(exc)}")
        
        if include_traceback:
            logger.error(traceback.format_exc())
    
    @staticmethod
    def handle_exception(exc: Exception) -> Dict[str, Any]:
        """Handle an exception and return a standardized response"""
        # Log the exception
        ErrorHandler.log_exception(exc)
        
        # Convert to DDoSAIException if it's not already
        if not isinstance(exc, DDoSAIException):
            if isinstance(exc, HTTPException):
                exc = DDoSAIException(
                    message=str(exc.detail),
                    status_code=exc.status_code,
                    details={"headers": exc.headers}
                )
            else:
                exc = DDoSAIException(
                    message=str(exc),
                    status_code=500,
                    details={"exception_type": exc.__class__.__name__}
                )
        
        # Return standardized error response
        return exc.to_dict()
    
    @staticmethod
    def format_validation_errors(errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format validation errors from Pydantic"""
        formatted_errors = {}
        for error in errors:
            loc = "->".join([str(l) for l in error["loc"]])
            formatted_errors[loc] = error["msg"]
        
        return ValidationError(
            message="Validation error",
            details={"errors": formatted_errors}
        ).to_dict()


def get_exception_handler():
    """Get exception handler for FastAPI"""
    async def exception_handler(request, exc):
        return ErrorHandler.handle_exception(exc)
    
    return exception_handler