"""
Unit tests for metrics collection and monitoring
"""
import pytest
import time
import json
import os
import tempfile
from unittest.mock import patch, MagicMock

# Add parent directory to path to import modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.metrics import MetricsCollector, PerformanceTimer


class TestMetricsCollector:
    """Test metrics collection functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        # Reset metrics before each test
        MetricsCollector.reset_metrics()
    
    def test_record_packet_processed(self):
        """Test recording processed packets"""
        # Record some packets
        MetricsCollector.record_packet_processed(0.1, False)
        MetricsCollector.record_packet_processed(0.2, True)
        MetricsCollector.record_packet_processed(0.3, False)
        
        # Get metrics
        metrics = MetricsCollector.get_metrics()
        
        # Check packet counts
        assert metrics["packets"]["total"] == 3
        assert metrics["packets"]["malicious"] == 1
        assert metrics["packets"]["benign"] == 2
        
        # Check processing time
        assert metrics["processing"]["avg_time"] == pytest.approx(0.2, 0.01)
        assert metrics["processing"]["avg_time_ms"] == pytest.approx(200, 10)
    
    def test_record_model_inference(self):
        """Test recording model inference times"""
        # Record some model inferences
        MetricsCollector.record_model_inference("autoencoder", 0.05)
        MetricsCollector.record_model_inference("gnn", 0.1)
        MetricsCollector.record_model_inference("autoencoder", 0.07)
        
        # Get metrics
        metrics = MetricsCollector.get_metrics()
        
        # Check model metrics
        assert "autoencoder" in metrics["models"]
        assert "gnn" in metrics["models"]
        assert metrics["models"]["autoencoder"]["avg_time"] == pytest.approx(0.06, 0.01)
        assert metrics["models"]["gnn"]["avg_time"] == pytest.approx(0.1, 0.01)
        assert metrics["models"]["autoencoder"]["count"] == 2
        assert metrics["models"]["gnn"]["count"] == 1
    
    def test_record_error(self):
        """Test recording errors"""
        # Record some errors
        MetricsCollector.record_error()
        MetricsCollector.record_error()
        
        # Record some packets to calculate error rate
        MetricsCollector.record_packet_processed(0.1, False)
        MetricsCollector.record_packet_processed(0.2, True)
        
        # Get metrics
        metrics = MetricsCollector.get_metrics()
        
        # Check error count and rate
        assert metrics["errors"]["count"] == 2
        assert metrics["errors"]["rate"] == pytest.approx(100, 0.1)  # 2 errors / 2 packets * 100
    
    def test_export_metrics(self):
        """Test exporting metrics to a file"""
        # Record some data
        MetricsCollector.record_packet_processed(0.1, False)
        MetricsCollector.record_packet_processed(0.2, True)
        MetricsCollector.record_model_inference("autoencoder", 0.05)
        
        # Export to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            file_path = tmp.name
        
        try:
            # Export metrics
            result = MetricsCollector.export_metrics(file_path)
            assert result is True
            
            # Read the file and check content
            with open(file_path, 'r') as f:
                exported_metrics = json.load(f)
            
            assert exported_metrics["packets"]["total"] == 2
            assert exported_metrics["packets"]["malicious"] == 1
            assert "autoencoder" in exported_metrics["models"]
        finally:
            # Clean up
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    def test_reset_metrics(self):
        """Test resetting metrics"""
        # Record some data
        MetricsCollector.record_packet_processed(0.1, False)
        MetricsCollector.record_packet_processed(0.2, True)
        
        # Reset metrics
        MetricsCollector.reset_metrics()
        
        # Get metrics
        metrics = MetricsCollector.get_metrics()
        
        # Check that metrics were reset
        assert metrics["packets"]["total"] == 0
        assert metrics["packets"]["malicious"] == 0
        assert metrics["packets"]["benign"] == 0


class TestPerformanceTimer:
    """Test performance timer context manager"""
    
    def test_performance_timer_operation(self):
        """Test timing an operation"""
        with patch('core.metrics.MetricsCollector.record_model_inference') as mock_record:
            # Use the timer
            with PerformanceTimer("test_operation", model_name="test_model"):
                # Simulate some work
                time.sleep(0.1)
            
            # Check that record_model_inference was called
            mock_record.assert_called_once()
            assert mock_record.call_args[0][0] == "test_model"
            assert isinstance(mock_record.call_args[0][1], float)
            assert mock_record.call_args[0][1] > 0
    
    def test_performance_timer_error(self):
        """Test timing an operation that raises an error"""
        with patch('core.metrics.MetricsCollector.record_error') as mock_record_error:
            with patch('core.metrics.MetricsCollector.record_model_inference') as mock_record_inference:
                # Use the timer with an operation that raises an exception
                try:
                    with PerformanceTimer("test_operation", model_name="test_model"):
                        # Simulate some work
                        time.sleep(0.1)
                        # Raise an exception
                        raise ValueError("Test error")
                except ValueError:
                    pass
                
                # Check that record_model_inference was called
                mock_record_inference.assert_called_once()
                
                # Check that record_error was called
                mock_record_error.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-v", __file__])