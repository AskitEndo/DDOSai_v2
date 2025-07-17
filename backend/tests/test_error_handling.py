"""
Unit tests for error handling mechanisms in DDoS.AI platform
"""
import pytest
import asyncio
import time
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.exceptions import (
    DDoSAIException, ValidationError, NotFoundError, 
    CircuitBreakerOpenError, RetryExhaustedError, 
    ModelOverloadedError, ResourceExhaustedError
)
from core.recovery import CircuitBreaker, RetryWithBackoff, ErrorRecovery
from core.decorators import retry_operation, handle_errors


class TestExceptions:
    """Test exception hierarchy and error handling"""
    
    def test_base_exception(self):
        """Test base DDoSAIException"""
        exc = DDoSAIException("Test error", 500, {"test": "value"})
        assert exc.message == "Test error"
        assert exc.status_code == 500
        assert exc.details == {"test": "value"}
        
        # Test to_dict method
        exc_dict = exc.to_dict()
        assert exc_dict["error"] == "Test error"
        assert exc_dict["status_code"] == 500
        assert exc_dict["details"] == {"test": "value"}
        assert exc_dict["error_type"] == "DDoSAIException"
    
    def test_validation_error(self):
        """Test ValidationError"""
        exc = ValidationError("Invalid input", {"field": "value"})
        assert exc.message == "Invalid input"
        assert exc.status_code == 422
        assert exc.details == {"field": "value"}
    
    def test_not_found_error(self):
        """Test NotFoundError"""
        exc = NotFoundError("Resource not found", {"resource_id": "123"})
        assert exc.message == "Resource not found"
        assert exc.status_code == 404
        assert exc.details == {"resource_id": "123"}
    
    def test_circuit_breaker_open_error(self):
        """Test CircuitBreakerOpenError"""
        exc = CircuitBreakerOpenError("Circuit breaker is open", 30, {"circuit": "test"})
        assert exc.message == "Circuit breaker is open"
        assert exc.status_code == 503
        assert exc.details["retry_after"] == 30
        assert exc.details["circuit"] == "test"
    
    def test_retry_exhausted_error(self):
        """Test RetryExhaustedError"""
        exc = RetryExhaustedError("Max retries exceeded", "test_operation", 3, {"last_error": "timeout"})
        assert exc.message == "Max retries exceeded"
        assert exc.status_code == 500
        assert exc.details["operation"] == "test_operation"
        assert exc.details["attempts"] == 3
        assert exc.details["last_error"] == "timeout"


class TestCircuitBreaker:
    """Test circuit breaker pattern implementation"""
    
    def setup_method(self):
        """Setup for each test"""
        # Clear circuit breakers registry
        CircuitBreaker._breakers = {}
        
        # Create a test circuit breaker
        self.cb = CircuitBreaker(
            name="test_breaker",
            failure_threshold=2,
            reset_timeout=1,
            half_open_max_calls=2
        )
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization"""
        assert self.cb.name == "test_breaker"
        assert self.cb.failure_threshold == 2
        assert self.cb.reset_timeout == 1
        assert self.cb.half_open_max_calls == 2
        assert self.cb.state == CircuitBreaker.CLOSED
        assert self.cb.failure_count == 0
    
    def test_circuit_breaker_get(self):
        """Test getting circuit breaker by name"""
        cb = CircuitBreaker.get("test_breaker")
        assert cb is self.cb
        
        # Test creating new circuit breaker if it doesn't exist
        new_cb = CircuitBreaker.get("new_breaker")
        assert new_cb.name == "new_breaker"
        assert new_cb is CircuitBreaker._breakers["new_breaker"]
    
    def test_circuit_breaker_state_transitions(self):
        """Test circuit breaker state transitions"""
        # Initially closed
        assert self.cb.state == CircuitBreaker.CLOSED
        assert self.cb.allow_request() is True
        
        # Record failures to open the circuit
        self.cb.record_failure()
        assert self.cb.state == CircuitBreaker.CLOSED
        assert self.cb.failure_count == 1
        
        self.cb.record_failure()
        assert self.cb.state == CircuitBreaker.OPEN
        assert self.cb.allow_request() is False
        
        # Wait for reset timeout
        time.sleep(1.1)
        
        # Should transition to half-open
        assert self.cb.allow_request() is True
        assert self.cb.state == CircuitBreaker.HALF_OPEN
        
        # Record success to close the circuit
        self.cb.record_success()
        assert self.cb.state == CircuitBreaker.HALF_OPEN
        assert self.cb.success_count == 1
        
        self.cb.record_success()
        assert self.cb.state == CircuitBreaker.CLOSED
        assert self.cb.failure_count == 0
    
    def test_circuit_breaker_half_open_failure(self):
        """Test failure in half-open state"""
        # Open the circuit
        self.cb.record_failure()
        self.cb.record_failure()
        assert self.cb.state == CircuitBreaker.OPEN
        
        # Wait for reset timeout
        time.sleep(1.1)
        
        # Should transition to half-open
        assert self.cb.allow_request() is True
        assert self.cb.state == CircuitBreaker.HALF_OPEN
        
        # Record failure in half-open state
        self.cb.record_failure()
        assert self.cb.state == CircuitBreaker.OPEN
    
    def test_circuit_breaker_execute_sync(self):
        """Test execute_sync method"""
        # Test successful execution
        result = self.cb.execute_sync(lambda x: x * 2, 5)
        assert result == 10
        
        # Open the circuit
        self.cb.record_failure()
        self.cb.record_failure()
        
        # Test execution with open circuit
        with pytest.raises(CircuitBreakerOpenError):
            self.cb.execute_sync(lambda x: x * 2, 5)


class TestRetryWithBackoff:
    """Test retry with exponential backoff"""
    
    @pytest.mark.asyncio
    async def test_retry_async_success(self):
        """Test successful async retry"""
        mock_func = MagicMock()
        mock_func.return_value = asyncio.Future()
        mock_func.return_value.set_result("success")
        
        result = await RetryWithBackoff.retry_async(
            mock_func,
            max_retries=3,
            base_delay=0.01,
            max_delay=0.1,
            exceptions_to_retry=(ValueError,)
        )
        
        assert result == "success"
        assert mock_func.call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_async_with_retries(self):
        """Test async retry with failures"""
        mock_func = MagicMock()
        
        # Fail twice, then succeed
        side_effects = [
            ValueError("Failure 1"),
            ValueError("Failure 2"),
            asyncio.Future()
        ]
        side_effects[2].set_result("success after retries")
        
        mock_func.side_effect = side_effects
        
        result = await RetryWithBackoff.retry_async(
            mock_func,
            max_retries=3,
            base_delay=0.01,
            max_delay=0.1,
            exceptions_to_retry=(ValueError,)
        )
        
        assert result == "success after retries"
        assert mock_func.call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_async_max_retries_exceeded(self):
        """Test async retry with max retries exceeded"""
        mock_func = MagicMock()
        mock_func.side_effect = ValueError("Persistent failure")
        
        with pytest.raises(RetryExhaustedError) as exc_info:
            await RetryWithBackoff.retry_async(
                mock_func,
                max_retries=2,
                base_delay=0.01,
                max_delay=0.1,
                exceptions_to_retry=(ValueError,)
            )
        
        assert "Max retries exceeded" in str(exc_info.value)
        assert mock_func.call_count == 3  # Initial call + 2 retries
    
    def test_retry_sync_success(self):
        """Test successful sync retry"""
        mock_func = MagicMock(return_value="success")
        
        result = RetryWithBackoff.retry_sync(
            mock_func,
            max_retries=3,
            base_delay=0.01,
            max_delay=0.1,
            exceptions_to_retry=(ValueError,)
        )
        
        assert result == "success"
        assert mock_func.call_count == 1
    
    def test_retry_sync_with_retries(self):
        """Test sync retry with failures"""
        mock_func = MagicMock()
        
        # Fail twice, then succeed
        mock_func.side_effect = [
            ValueError("Failure 1"),
            ValueError("Failure 2"),
            "success after retries"
        ]
        
        result = RetryWithBackoff.retry_sync(
            mock_func,
            max_retries=3,
            base_delay=0.01,
            max_delay=0.1,
            exceptions_to_retry=(ValueError,)
        )
        
        assert result == "success after retries"
        assert mock_func.call_count == 3


class TestDecorators:
    """Test error handling decorators"""
    
    @pytest.mark.asyncio
    async def test_retry_operation_decorator(self):
        """Test retry_operation decorator"""
        call_count = 0
        
        @retry_operation(max_retries=2, base_delay=0.01, exceptions_to_retry=(ValueError,))
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Test error")
            return "success"
        
        result = await test_function()
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_handle_errors_decorator(self):
        """Test handle_errors decorator"""
        @handle_errors
        async def test_function(error=False):
            if error:
                raise ValueError("Test error")
            return "success"
        
        # Test successful execution
        result = await test_function()
        assert result == "success"
        
        # Test error handling
        with pytest.raises(ValueError):
            await test_function(error=True)


if __name__ == "__main__":
    pytest.main(["-v", __file__])