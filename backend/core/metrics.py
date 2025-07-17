"""
Performance metrics collection and reporting for DDoS.AI platform
"""
import time
import psutil
import logging
import threading
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server

# Configure logging
logger = logging.getLogger(__name__)

# Global metrics registry
METRICS = {}

# Performance tracking
PACKET_COUNTER = 0
PACKET_PROCESSING_TIMES = []
MALICIOUS_PACKET_COUNTER = 0
MODEL_INFERENCE_TIMES = {}
ERROR_COUNTER = 0
LAST_METRICS_RESET = time.time()

# Prometheus metrics
PROMETHEUS_METRICS = {
    "counters": {},
    "gauges": {},
    "histograms": {},
    "summaries": {}
}

class MetricsCollector:
    """Metrics collection and reporting service"""
    
    @staticmethod
    def initialize(enable_prometheus: bool = True, prometheus_port: int = 8001) -> None:
        """Initialize metrics collection"""
        logger.info("Initializing metrics collection")
        
        # Reset metrics
        MetricsCollector.reset_metrics()
        
        # Start Prometheus server if enabled
        if enable_prometheus:
            try:
                # Create Prometheus metrics
                # Counters
                PROMETHEUS_METRICS["counters"]["packets_total"] = Counter(
                    "ddosai_packets_total", 
                    "Total number of packets processed"
                )
                PROMETHEUS_METRICS["counters"]["malicious_packets_total"] = Counter(
                    "ddosai_malicious_packets_total", 
                    "Total number of malicious packets detected"
                )
                PROMETHEUS_METRICS["counters"]["errors_total"] = Counter(
                    "ddosai_errors_total", 
                    "Total number of errors encountered"
                )
                
                # Gauges
                PROMETHEUS_METRICS["gauges"]["cpu_usage"] = Gauge(
                    "ddosai_cpu_usage", 
                    "Current CPU usage percentage"
                )
                PROMETHEUS_METRICS["gauges"]["memory_usage"] = Gauge(
                    "ddosai_memory_usage", 
                    "Current memory usage percentage"
                )
                PROMETHEUS_METRICS["gauges"]["threat_level"] = Gauge(
                    "ddosai_threat_level", 
                    "Current threat level (0-5)"
                )
                
                # Histograms
                PROMETHEUS_METRICS["histograms"]["packet_processing_time"] = Histogram(
                    "ddosai_packet_processing_time_seconds", 
                    "Time taken to process a packet",
                    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0)
                )
                
                # Summaries
                PROMETHEUS_METRICS["summaries"]["model_inference_time"] = Summary(
                    "ddosai_model_inference_time_seconds", 
                    "Time taken for model inference",
                    ["model_name"]
                )
                
                # Start Prometheus HTTP server
                start_http_server(prometheus_port)
                logger.info(f"Prometheus metrics server started on port {prometheus_port}")
                
                # Start metrics collection thread
                threading.Thread(
                    target=MetricsCollector._metrics_collection_thread,
                    daemon=True
                ).start()
                
            except Exception as e:
                logger.error(f"Failed to start Prometheus metrics server: {e}")
    
    @staticmethod
    def _metrics_collection_thread() -> None:
        """Background thread for collecting system metrics"""
        while True:
            try:
                # Update system metrics
                cpu_usage = psutil.cpu_percent(interval=1)
                memory_usage = psutil.virtual_memory().percent
                
                # Update Prometheus gauges
                PROMETHEUS_METRICS["gauges"]["cpu_usage"].set(cpu_usage)
                PROMETHEUS_METRICS["gauges"]["memory_usage"].set(memory_usage)
                
                # Calculate threat level (0-5) based on recent malicious packet ratio
                if PACKET_COUNTER > 0:
                    threat_ratio = MALICIOUS_PACKET_COUNTER / PACKET_COUNTER
                    threat_level = min(5, int(threat_ratio * 10))
                    PROMETHEUS_METRICS["gauges"]["threat_level"].set(threat_level)
                
                # Sleep for 5 seconds
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in metrics collection thread: {e}")
                time.sleep(10)  # Longer sleep on error
    
    @staticmethod
    def reset_metrics() -> None:
        """Reset all metrics"""
        global PACKET_COUNTER, PACKET_PROCESSING_TIMES, MALICIOUS_PACKET_COUNTER
        global MODEL_INFERENCE_TIMES, ERROR_COUNTER, LAST_METRICS_RESET
        
        PACKET_COUNTER = 0
        PACKET_PROCESSING_TIMES = []
        MALICIOUS_PACKET_COUNTER = 0
        MODEL_INFERENCE_TIMES = {}
        ERROR_COUNTER = 0
        LAST_METRICS_RESET = time.time()
        
        logger.debug("Metrics reset")
    
    @staticmethod
    def record_packet_processed(processing_time: float, is_malicious: bool = False) -> None:
        """Record a processed packet"""
        global PACKET_COUNTER, PACKET_PROCESSING_TIMES, MALICIOUS_PACKET_COUNTER
        
        PACKET_COUNTER += 1
        PACKET_PROCESSING_TIMES.append(processing_time)
        
        # Keep only the last 1000 processing times to avoid memory issues
        if len(PACKET_PROCESSING_TIMES) > 1000:
            PACKET_PROCESSING_TIMES = PACKET_PROCESSING_TIMES[-1000:]
        
        if is_malicious:
            MALICIOUS_PACKET_COUNTER += 1
        
        # Update Prometheus metrics
        try:
            PROMETHEUS_METRICS["counters"]["packets_total"].inc()
            if is_malicious:
                PROMETHEUS_METRICS["counters"]["malicious_packets_total"].inc()
            PROMETHEUS_METRICS["histograms"]["packet_processing_time"].observe(processing_time)
        except Exception as e:
            logger.error(f"Error updating Prometheus metrics: {e}")
    
    @staticmethod
    def record_model_inference(model_name: str, inference_time: float) -> None:
        """Record model inference time"""
        global MODEL_INFERENCE_TIMES
        
        if model_name not in MODEL_INFERENCE_TIMES:
            MODEL_INFERENCE_TIMES[model_name] = []
        
        MODEL_INFERENCE_TIMES[model_name].append(inference_time)
        
        # Keep only the last 1000 inference times per model
        if len(MODEL_INFERENCE_TIMES[model_name]) > 1000:
            MODEL_INFERENCE_TIMES[model_name] = MODEL_INFERENCE_TIMES[model_name][-1000:]
        
        # Update Prometheus metrics
        try:
            PROMETHEUS_METRICS["summaries"]["model_inference_time"].labels(model_name=model_name).observe(inference_time)
        except Exception as e:
            logger.error(f"Error updating Prometheus metrics: {e}")
    
    @staticmethod
    def record_error() -> None:
        """Record an error"""
        global ERROR_COUNTER
        
        ERROR_COUNTER += 1
        
        # Update Prometheus metrics
        try:
            PROMETHEUS_METRICS["counters"]["errors_total"].inc()
        except Exception as e:
            logger.error(f"Error updating Prometheus metrics: {e}")
    
    @staticmethod
    def get_metrics() -> Dict[str, Any]:
        """Get current metrics"""
        # Calculate average processing time
        avg_processing_time = sum(PACKET_PROCESSING_TIMES) / len(PACKET_PROCESSING_TIMES) if PACKET_PROCESSING_TIMES else 0
        
        # Calculate model inference times
        model_inference_metrics = {}
        for model_name, times in MODEL_INFERENCE_TIMES.items():
            if times:
                model_inference_metrics[model_name] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "count": len(times)
                }
        
        # Calculate error rate
        error_rate = ERROR_COUNTER / PACKET_COUNTER * 100 if PACKET_COUNTER > 0 else 0
        
        # Calculate threat level (0-5) based on recent malicious packet ratio
        if PACKET_COUNTER > 0:
            threat_ratio = MALICIOUS_PACKET_COUNTER / PACKET_COUNTER
            threat_level = min(5, int(threat_ratio * 10))
        else:
            threat_level = 0
        
        # Get system metrics
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - LAST_METRICS_RESET,
            "packets": {
                "total": PACKET_COUNTER,
                "malicious": MALICIOUS_PACKET_COUNTER,
                "benign": PACKET_COUNTER - MALICIOUS_PACKET_COUNTER,
                "malicious_ratio": MALICIOUS_PACKET_COUNTER / PACKET_COUNTER if PACKET_COUNTER > 0 else 0
            },
            "processing": {
                "avg_time": avg_processing_time,
                "avg_time_ms": avg_processing_time * 1000,  # Convert to ms
                "throughput": PACKET_COUNTER / (time.time() - LAST_METRICS_RESET) if time.time() > LAST_METRICS_RESET else 0
            },
            "models": model_inference_metrics,
            "errors": {
                "count": ERROR_COUNTER,
                "rate": error_rate
            },
            "threat_level": threat_level,
            "system": {
                "cpu_usage": cpu_usage,
                "memory_usage": memory.percent,
                "disk_usage": disk.percent
            }
        }
    
    @staticmethod
    def export_metrics(file_path: str) -> bool:
        """Export metrics to a JSON file"""
        try:
            metrics = MetricsCollector.get_metrics()
            with open(file_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Metrics exported to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return False


class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, operation_name: str, model_name: Optional[str] = None):
        self.operation_name = operation_name
        self.model_name = model_name
        self.start_time = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        
        if self.model_name:
            # Record as model inference
            MetricsCollector.record_model_inference(self.model_name, execution_time)
            logger.debug(f"Model {self.model_name} inference took {execution_time:.4f}s")
        else:
            # Record as general operation
            logger.debug(f"Operation {self.operation_name} took {execution_time:.4f}s")
        
        # Record error if exception occurred
        if exc_type is not None:
            MetricsCollector.record_error()
            logger.error(f"Error in {self.operation_name}: {exc_val}")
        
        return False  # Don't suppress exceptions