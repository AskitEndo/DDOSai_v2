"""Recovery utilities for DDoS.AI platform"""
import logging
import signal
import sys
import threading
import time
import asyncio
from typing import Callable, List, Dict, Any, Optional
import traceback

# Configure logging
logger = logging.getLogger(__name__)

# Shutdown handlers
SHUTDOWN_HANDLERS: List[Callable] = []

# Recovery state
RECOVERY_STATE = {
    "is_shutting_down": False,
    "shutdown_reason": None,
    "critical_errors": 0,
    "last_error_time": 0,
    "recovery_attempts": 0
}


class GracefulShutdown:
    """Utility for graceful shutdown"""
    
    @staticmethod
    def register_shutdown_handler(handler: Callable) -> None:
        """Register a shutdown handler"""
        SHUTDOWN_HANDLERS.append(handler)
        logger.debug(f"Registered shutdown handler: {handler.__name__}")
    
    @staticmethod
    def handle_shutdown(sig=None, frame=None, reason: str = None) -> None:
        """Handle shutdown signal"""
        if RECOVERY_STATE["is_shutting_down"]:
            logger.warning("Shutdown already in progress, ignoring signal")
            return
        
        RECOVERY_STATE["is_shutting_down"] = True
        RECOVERY_STATE["shutdown_reason"] = reason or (f"Signal {sig}" if sig else "Unknown reason")
        
        logger.info(f"Initiating graceful shutdown: {RECOVERY_STATE['shutdown_reason']}")
        
        # Execute shutdown handlers
        for handler in SHUTDOWN_HANDLERS:
            try:
                logger.debug(f"Executing shutdown handler: {handler.__name__}")
                handler()
            except Exception as e:
                logger.error(f"Error in shutdown handler {handler.__name__}: {e}")
        
        logger.info("Graceful shutdown completed")
        
        # Exit with appropriate code
        if sig == signal.SIGTERM:
            sys.exit(0)
        elif sig == signal.SIGINT:
            sys.exit(130)  # 128 + SIGINT value (2)
        else:
            sys.exit(1)
    
    @staticmethod
    def setup_signal_handlers() -> None:
        """Set up signal handlers for graceful shutdown"""
        signal.signal(signal.SIGTERM, GracefulShutdown.handle_shutdown)
        signal.signal(signal.SIGINT, GracefulShutdown.handle_shutdown)
        logger.debug("Signal handlers set up for graceful shutdown")


class ErrorRecovery:
    """Utility for error recovery"""
    
    @staticmethod
    def record_error(error: Exception, is_critical: bool = False) -> None:
        """Record an error for recovery tracking"""
        current_time = time.time()
        
        # Reset critical error count if last error was more than 5 minutes ago
        if current_time - RECOVERY_STATE["last_error_time"] > 300:  # 5 minutes
            RECOVERY_STATE["critical_errors"] = 0
        
        RECOVERY_STATE["last_error_time"] = current_time
        
        if is_critical:
            RECOVERY_STATE["critical_errors"] += 1
            logger.error(
                f"Critical error recorded ({RECOVERY_STATE['critical_errors']} in current window): {error}"
                f"\n{traceback.format_exc()}"
            )
            
            # Check if we need to initiate recovery
            if RECOVERY_STATE["critical_errors"] >= 5:  # 5 critical errors in 5 minutes
                ErrorRecovery.initiate_recovery()
        else:
            logger.error(f"Error recorded: {error}\n{traceback.format_exc()}")
    
    @staticmethod
    def initiate_recovery() -> None:
        """Initiate recovery process"""
        if RECOVERY_STATE["is_shutting_down"]:
            logger.warning("Shutdown already in progress, skipping recovery")
            return
        
        RECOVERY_STATE["recovery_attempts"] += 1
        logger.warning(
            f"Initiating recovery process (attempt {RECOVERY_STATE['recovery_attempts']})"
            f" due to {RECOVERY_STATE['critical_errors']} critical errors"
        )
        
        # Reset critical error count
        RECOVERY_STATE["critical_errors"] = 0
        
        # Implement recovery logic here
        # For example, restart services, reconnect to databases, etc.
        
        # If recovery attempts exceed threshold, initiate shutdown
        if RECOVERY_STATE["recovery_attempts"] >= 3:  # 3 recovery attempts
            logger.critical(
                f"Recovery attempts exceeded threshold ({RECOVERY_STATE['recovery_attempts']}),"
                f" initiating shutdown"
            )
            GracefulShutdown.handle_shutdown(reason="Recovery attempts exceeded threshold")
    
    @staticmethod
    async def watchdog(check_interval: int = 60) -> None:
        """Watchdog process to monitor system health"""
        logger.info(f"Starting watchdog process with check interval {check_interval} seconds")
        
        while not RECOVERY_STATE["is_shutting_down"]:
            try:
                # Implement health checks here
                # For example, check database connectivity, API health, etc.
                
                # Sleep until next check
                await asyncio.sleep(check_interval)
            except Exception as e:
                logger.error(f"Error in watchdog process: {e}")
                await asyncio.sleep(check_interval)


class CircuitBreaker:
    """Circuit breaker pattern implementation for preventing cascading failures"""
    
    # Circuit breaker states
    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"      # Failing state, requests are rejected immediately
    HALF_OPEN = "half_open"  # Testing state, limited requests pass through
    
    # Circuit breakers registry (keyed by name)
    _breakers = {}
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: int = 30,
        half_open_max_calls: int = 3
    ):
        """Initialize a circuit breaker
        
        Args:
            name: Unique name for this circuit breaker
            failure_threshold: Number of failures before opening the circuit
            reset_timeout: Seconds to wait before attempting to close the circuit
            half_open_max_calls: Maximum number of calls to allow in half-open state
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls
        
        # State tracking
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
        self.success_count = 0
        
        # Register this circuit breaker
        CircuitBreaker._breakers[name] = self
        logger.debug(f"Circuit breaker '{name}' initialized in {self.state} state")
    
    @classmethod
    def get(cls, name: str) -> 'CircuitBreaker':
        """Get a circuit breaker by name, creating it if it doesn't exist"""
        if name not in cls._breakers:
            cls._breakers[name] = CircuitBreaker(name)
        return cls._breakers[name]
    
    def allow_request(self) -> bool:
        """Check if a request should be allowed through the circuit breaker"""
        current_time = time.time()
        
        if self.state == self.CLOSED:
            return True
        
        elif self.state == self.OPEN:
            # Check if reset timeout has elapsed
            if current_time - self.last_failure_time > self.reset_timeout:
                logger.info(f"Circuit breaker '{self.name}': Transitioning from OPEN to HALF_OPEN")
                self.state = self.HALF_OPEN
                self.half_open_calls = 0
                self.success_count = 0
                return True
            return False
        
        elif self.state == self.HALF_OPEN:
            # Allow limited calls in half-open state
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False
        
        return True
    
    def record_success(self) -> None:
        """Record a successful operation"""
        if self.state == self.HALF_OPEN:
            self.success_count += 1
            
            # If we've had enough successes in half-open state, close the circuit
            if self.success_count >= self.half_open_max_calls:
                logger.info(f"Circuit breaker '{self.name}': Transitioning from HALF_OPEN to CLOSED")
                self.state = self.CLOSED
                self.failure_count = 0
                self.half_open_calls = 0
                self.success_count = 0
        
        # In closed state, reset failure count on success
        elif self.state == self.CLOSED:
            self.failure_count = 0
    
    def record_failure(self) -> None:
        """Record a failed operation"""
        current_time = time.time()
        self.failure_count += 1
        self.last_failure_time = current_time
        
        if self.state == self.CLOSED and self.failure_count >= self.failure_threshold:
            logger.warning(
                f"Circuit breaker '{self.name}': Failure threshold exceeded ({self.failure_count}), "
                f"transitioning from CLOSED to OPEN"
            )
            self.state = self.OPEN
        
        elif self.state == self.HALF_OPEN:
            logger.warning(
                f"Circuit breaker '{self.name}': Failure in HALF_OPEN state, "
                f"transitioning back to OPEN"
            )
            self.state = self.OPEN
    
    async def execute_async(self, func: Callable, *args, **kwargs):
        """Execute an async function with circuit breaker protection"""
        if not self.allow_request():
            from core.exceptions import CircuitBreakerOpenError
            retry_after = int(self.reset_timeout - (time.time() - self.last_failure_time))
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is open",
                retry_after=max(1, retry_after),
                details={"circuit_breaker": self.name}
            )
        
        try:
            result = await func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise
    
    def execute_sync(self, func: Callable, *args, **kwargs):
        """Execute a sync function with circuit breaker protection"""
        if not self.allow_request():
            from core.exceptions import CircuitBreakerOpenError
            retry_after = int(self.reset_timeout - (time.time() - self.last_failure_time))
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is open",
                retry_after=max(1, retry_after),
                details={"circuit_breaker": self.name}
            )
        
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise


class RetryWithBackoff:
    """Utility for retrying operations with exponential backoff"""
    
    @staticmethod
    async def retry_async(
        func: Callable,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 10.0,
        backoff_factor: float = 2.0,
        exceptions_to_retry: tuple = (Exception,),
        *args, **kwargs
    ):
        """Retry an async function with exponential backoff"""
        retries = 0
        delay = base_delay
        operation_name = getattr(func, "__name__", "unknown_operation")
        
        while True:
            try:
                return await func(*args, **kwargs)
            except exceptions_to_retry as e:
                retries += 1
                if retries > max_retries:
                    logger.error(f"Max retries ({max_retries}) exceeded for {operation_name}: {e}")
                    from core.exceptions import RetryExhaustedError
                    raise RetryExhaustedError(
                        message=f"Max retries exceeded: {str(e)}",
                        operation=operation_name,
                        attempts=max_retries,
                        details={"original_error": str(e)}
                    ) from e
                
                # Calculate next delay with jitter
                import random
                jitter = random.uniform(0.8, 1.2)
                delay = min(delay * backoff_factor * jitter, max_delay)
                
                logger.warning(
                    f"Retry {retries}/{max_retries} for {operation_name} after {delay:.2f}s: {e}"
                )
                
                # Wait before retrying
                await asyncio.sleep(delay)
    
    @staticmethod
    def retry_sync(
        func: Callable,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 10.0,
        backoff_factor: float = 2.0,
        exceptions_to_retry: tuple = (Exception,),
        *args, **kwargs
    ):
        """Retry a sync function with exponential backoff"""
        retries = 0
        delay = base_delay
        operation_name = getattr(func, "__name__", "unknown_operation")
        
        while True:
            try:
                return func(*args, **kwargs)
            except exceptions_to_retry as e:
                retries += 1
                if retries > max_retries:
                    logger.error(f"Max retries ({max_retries}) exceeded for {operation_name}: {e}")
                    from core.exceptions import RetryExhaustedError
                    raise RetryExhaustedError(
                        message=f"Max retries exceeded: {str(e)}",
                        operation=operation_name,
                        attempts=max_retries,
                        details={"original_error": str(e)}
                    ) from e
                
                # Calculate next delay with jitter
                import random
                jitter = random.uniform(0.8, 1.2)
                delay = min(delay * backoff_factor * jitter, max_delay)
                
                logger.warning(
                    f"Retry {retries}/{max_retries} for {operation_name} after {delay:.2f}s: {e}"
                )
                
                # Wait before retrying
                time.sleep(delay)