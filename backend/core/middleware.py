"""Middleware for DDoS.AI platform"""
import time
import logging
from typing import Callable, Dict, Any
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import json
import uuid

from core.exceptions import DDoSAIException, ErrorHandler, RateLimitError

# Configure logging
logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses"""
    
    def __init__(self, app: ASGIApp, exclude_paths: list = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/health", "/metrics"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip logging for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request
        start_time = time.time()
        method = request.method
        path = request.url.path
        query_params = dict(request.query_params)
        client_host = request.client.host if request.client else "unknown"
        
        logger.info(
            f"Request {request_id}: {method} {path} from {client_host}"
            f" Query params: {query_params}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Log response
            process_time = time.time() - start_time
            status_code = response.status_code
            logger.info(
                f"Response {request_id}: {status_code} processed in {process_time:.4f}s"
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
        except Exception as exc:
            # Log exception
            process_time = time.time() - start_time
            logger.error(
                f"Exception in {request_id}: {exc.__class__.__name__} processed in {process_time:.4f}s"
            )
            ErrorHandler.log_exception(exc)
            
            # Handle exception
            error_response = ErrorHandler.handle_exception(exc)
            status_code = error_response.get("status_code", 500)
            
            # Create JSON response
            response = Response(
                content=json.dumps(error_response),
                status_code=status_code,
                media_type="application/json"
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting"""
    
    def __init__(self, app: ASGIApp, rate_limit: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.rate_limit = rate_limit
        self.window_seconds = window_seconds
        self.requests = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health and metrics endpoints
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)
            
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Check rate limit
        current_time = time.time()
        if client_ip in self.requests:
            # Clean up old requests
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if current_time - req_time < self.window_seconds
            ]
            
            # Check if rate limit exceeded
            if len(self.requests[client_ip]) >= self.rate_limit:
                logger.warning(f"Rate limit exceeded for {client_ip}")
                raise RateLimitError(
                    message="Rate limit exceeded",
                    details={
                        "rate_limit": self.rate_limit,
                        "window_seconds": self.window_seconds,
                        "client_ip": client_ip
                    }
                )
        else:
            self.requests[client_ip] = []
        
        # Add current request
        self.requests[client_ip].append(current_time)
        
        # Process request
        return await call_next(request)


class CircuitBreakerMiddleware(BaseHTTPMiddleware):
    """Middleware implementing the circuit breaker pattern"""
    
    def __init__(
        self, 
        app: ASGIApp, 
        failure_threshold: int = 5,
        reset_timeout: int = 30,
        exclude_paths: list = None
    ):
        super().__init__(app)
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.exclude_paths = exclude_paths or ["/health", "/metrics"]
        self.failure_count = 0
        self.circuit_open = False
        self.last_failure_time = 0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip circuit breaker for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Check if circuit is open
        current_time = time.time()
        if self.circuit_open:
            # Check if reset timeout has elapsed
            if current_time - self.last_failure_time > self.reset_timeout:
                logger.info("Circuit breaker: Attempting to reset circuit")
                self.circuit_open = False
                self.failure_count = 0
            else:
                # Circuit is still open, return error
                logger.warning(f"Circuit breaker: Circuit open, rejecting request to {request.url.path}")
                return Response(
                    content=json.dumps({
                        "error": "Service temporarily unavailable",
                        "status_code": 503,
                        "details": {
                            "retry_after": int(self.reset_timeout - (current_time - self.last_failure_time))
                        },
                        "error_type": "CircuitBreakerOpen"
                    }),
                    status_code=503,
                    media_type="application/json",
                    headers={"Retry-After": str(int(self.reset_timeout - (current_time - self.last_failure_time)))}
                )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Reset failure count on success
            if response.status_code < 500:
                self.failure_count = 0
            else:
                # Increment failure count on server errors
                self.failure_count += 1
                self.last_failure_time = current_time
                
                # Check if failure threshold exceeded
                if self.failure_count >= self.failure_threshold:
                    logger.error(f"Circuit breaker: Failure threshold exceeded ({self.failure_count}), opening circuit")
                    self.circuit_open = True
            
            return response
        except Exception:
            # Increment failure count on exception
            self.failure_count += 1
            self.last_failure_time = current_time
            
            # Check if failure threshold exceeded
            if self.failure_count >= self.failure_threshold:
                logger.error(f"Circuit breaker: Failure threshold exceeded ({self.failure_count}), opening circuit")
                self.circuit_open = True
            
            # Re-raise exception to be handled by error handler
            raise