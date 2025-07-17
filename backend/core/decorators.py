"""Decorators for DDoS.AI platform"""
import functools
import time
import logging
import asyncio
from typing import Callable, Any, Dict, TypeVar, cast
from fastapi import Request, Response

from core.exceptions import DDoSAIException, ErrorHandler
from core.recovery import RetryWithBackoff, ErrorRecovery

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for function return type
T = TypeVar('T')


def handle_errors(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to handle errors in API routes"""
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        start_time = time.time()
        request = None
        
        # Find request object in args or kwargs
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break
        if request is None and 'request' in kwargs:
            request = kwargs['request']
        
        try:
            # Execute the function
            result = await func(*args, **kwargs)
            
            # Record request metrics
            latency = time.time() - start_time
            
            return result
        except Exception as exc:
            # Record error metrics
            latency = time.time() - start_time
            
            # Log the error
            ErrorHandler.log_exception(exc)
            
            # Check if it's a critical error
            is_critical = not isinstance(exc, DDoSAIException) or getattr(exc, 'status_code', 500) >= 500
            if is_critical:
                ErrorRecovery.record_error(exc, is_critical=True)
            
            # Raise the exception to be handled by FastAPI
            raise
    
    return cast(Callable[..., T], wrapper)


def rate_limit(limit: int, window_seconds: int = 60) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to rate limit API routes"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Store request counts per client
        request_counts: Dict[str, Dict[str, Any]] = {}
        
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            # Find request object in args or kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            if request is None and 'request' in kwargs:
                request = kwargs['request']
            
            if request is None:
                logger.warning("Rate limit decorator used on function without request parameter")
                return await func(*args, **kwargs)
            
            # Get client IP
            client_ip = request.client.host if request.client else "unknown"
            current_time = time.time()
            
            # Initialize or clean up client request count
            if client_ip not in request_counts:
                request_counts[client_ip] = {
                    "count": 0,
                    "window_start": current_time
                }
            elif current_time - request_counts[client_ip]["window_start"] > window_seconds:
                # Reset window if it has expired
                request_counts[client_ip] = {
                    "count": 0,
                    "window_start": current_time
                }
            
            # Check rate limit
            if request_counts[client_ip]["count"] >= limit:
                from core.exceptions import RateLimitError
                raise RateLimitError(
                    message="Rate limit exceeded",
                    details={
                        "limit": limit,
                        "window_seconds": window_seconds,
                        "client_ip": client_ip
                    }
                )
            
            # Increment request count
            request_counts[client_ip]["count"] += 1
            
            # Execute the function
            return await func(*args, **kwargs)
        
        return cast(Callable[..., T], wrapper)
    
    return decorator


def retry_operation(
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    exceptions_to_retry: tuple = (Exception,)
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to retry operations with exponential backoff"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            return await RetryWithBackoff.retry_async(
                func,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                exceptions_to_retry=exceptions_to_retry,
                *args, **kwargs
            )
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            return RetryWithBackoff.retry_sync(
                func,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                exceptions_to_retry=exceptions_to_retry,
                *args, **kwargs
            )
        
        if asyncio.iscoroutinefunction(func):
            return cast(Callable[..., T], async_wrapper)
        else:
            return cast(Callable[..., T], sync_wrapper)
    
    return decorator


def log_execution_time(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to log execution time of functions"""
    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> T:
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(
                f"Function {func.__name__} executed in {execution_time:.4f} seconds"
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Function {func.__name__} failed after {execution_time:.4f} seconds: {e}"
            )
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> T:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(
                f"Function {func.__name__} executed in {execution_time:.4f} seconds"
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Function {func.__name__} failed after {execution_time:.4f} seconds: {e}"
            )
            raise
    
    if asyncio.iscoroutinefunction(func):
        return cast(Callable[..., T], async_wrapper)
    else:
        return cast(Callable[..., T], sync_wrapper)