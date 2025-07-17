"""Monitoring utilities for DDoS.AI platform"""
import time
import psutil
import logging
from typing import Dict, Any, List
import os
import platform
import socket
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

# Start time of the application
START_TIME = time.time()

# Request metrics
REQUEST_COUNT = 0
REQUEST_LATENCIES = []
ERROR_COUNT = 0


class MonitoringService:
    """Service for monitoring system health and metrics"""
    
    # Component health status registry
    _component_status = {}
    
    # Performance thresholds
    THRESHOLDS = {
        "cpu_usage": 90,  # Percentage
        "memory_usage": 90,  # Percentage
        "disk_usage": 90,  # Percentage
        "request_latency": 1000,  # ms
        "error_rate": 5,  # Percentage
    }
    
    @staticmethod
    def register_component(component_name: str, status: str = "unknown", details: Dict[str, Any] = None) -> None:
        """Register a component for health monitoring"""
        MonitoringService._component_status[component_name] = {
            "status": status,
            "details": details or {},
            "last_updated": datetime.now().isoformat()
        }
        logger.debug(f"Registered component for monitoring: {component_name}")
    
    @staticmethod
    def update_component_status(component_name: str, status: str, details: Dict[str, Any] = None) -> None:
        """Update the status of a monitored component"""
        if component_name not in MonitoringService._component_status:
            MonitoringService.register_component(component_name, status, details)
            return
        
        MonitoringService._component_status[component_name] = {
            "status": status,
            "details": details or {},
            "last_updated": datetime.now().isoformat()
        }
        
        # Log status changes
        if status == "unhealthy":
            logger.warning(f"Component {component_name} is unhealthy: {details}")
        elif status == "degraded":
            logger.warning(f"Component {component_name} is degraded: {details}")
    
    @staticmethod
    def get_health_status() -> Dict[str, Any]:
        """Get health status of the application"""
        try:
            # Basic health check
            uptime = time.time() - START_TIME
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_usage = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('/').percent
            
            # Check if system is healthy based on thresholds
            system_healthy = (
                cpu_usage < MonitoringService.THRESHOLDS["cpu_usage"] and
                memory_usage < MonitoringService.THRESHOLDS["memory_usage"] and
                disk_usage < MonitoringService.THRESHOLDS["disk_usage"]
            )
            
            # Check component health
            components_status = "healthy"
            unhealthy_components = []
            
            for component, status in MonitoringService._component_status.items():
                if status["status"] == "unhealthy":
                    components_status = "unhealthy"
                    unhealthy_components.append(component)
                elif status["status"] == "degraded" and components_status == "healthy":
                    components_status = "degraded"
            
            # Determine overall status
            if not system_healthy or components_status == "unhealthy":
                overall_status = "unhealthy"
            elif components_status == "degraded":
                overall_status = "degraded"
            else:
                overall_status = "healthy"
            
            # Calculate error rate
            error_rate = 0
            if REQUEST_COUNT > 0:
                error_rate = (ERROR_COUNT / REQUEST_COUNT) * 100
            
            return {
                "status": overall_status,
                "uptime": uptime,
                "timestamp": datetime.now().isoformat(),
                "version": os.environ.get("APP_VERSION", "1.0.0"),
                "hostname": socket.gethostname(),
                "system": {
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "disk_usage": disk_usage
                },
                "components": MonitoringService._component_status,
                "metrics": {
                    "request_count": REQUEST_COUNT,
                    "error_count": ERROR_COUNT,
                    "error_rate": error_rate,
                    "avg_latency_ms": sum(REQUEST_LATENCIES) / len(REQUEST_LATENCIES) * 1000 if REQUEST_LATENCIES else 0
                },
                "unhealthy_components": unhealthy_components
            }
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get detailed system information"""
        try:
            return {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "disk_total": psutil.disk_usage('/').total,
                "hostname": socket.gethostname(),
                "ip_address": socket.gethostbyname(socket.gethostname()),
                "uptime": time.time() - START_TIME,
                "start_time": datetime.fromtimestamp(START_TIME).isoformat(),
                "current_time": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {
                "error": str(e)
            }
    
    @staticmethod
    def get_metrics() -> Dict[str, Any]:
        """Get application metrics"""
        try:
            # Calculate request metrics
            avg_latency = sum(REQUEST_LATENCIES) / len(REQUEST_LATENCIES) if REQUEST_LATENCIES else 0
            p95_latency = sorted(REQUEST_LATENCIES)[int(len(REQUEST_LATENCIES) * 0.95)] if len(REQUEST_LATENCIES) > 20 else 0
            p99_latency = sorted(REQUEST_LATENCIES)[int(len(REQUEST_LATENCIES) * 0.99)] if len(REQUEST_LATENCIES) > 100 else 0
            
            # Get system metrics
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_usage = psutil.virtual_memory()
            disk_usage = psutil.disk_usage('/')
            network_io = psutil.net_io_counters()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "uptime": time.time() - START_TIME,
                "requests": {
                    "total": REQUEST_COUNT,
                    "errors": ERROR_COUNT,
                    "success_rate": (REQUEST_COUNT - ERROR_COUNT) / REQUEST_COUNT * 100 if REQUEST_COUNT > 0 else 0,
                    "latency": {
                        "avg_ms": avg_latency * 1000,
                        "p95_ms": p95_latency * 1000,
                        "p99_ms": p99_latency * 1000
                    }
                },
                "system": {
                    "cpu": {
                        "usage_percent": cpu_usage,
                        "count": psutil.cpu_count()
                    },
                    "memory": {
                        "total": memory_usage.total,
                        "available": memory_usage.available,
                        "used": memory_usage.used,
                        "percent": memory_usage.percent
                    },
                    "disk": {
                        "total": disk_usage.total,
                        "used": disk_usage.used,
                        "free": disk_usage.free,
                        "percent": disk_usage.percent
                    },
                    "network": {
                        "bytes_sent": network_io.bytes_sent,
                        "bytes_recv": network_io.bytes_recv,
                        "packets_sent": network_io.packets_sent,
                        "packets_recv": network_io.packets_recv,
                        "errin": network_io.errin,
                        "errout": network_io.errout
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    @staticmethod
    def record_request(latency: float, is_error: bool = False) -> None:
        """Record a request for metrics"""
        global REQUEST_COUNT, REQUEST_LATENCIES, ERROR_COUNT
        
        REQUEST_COUNT += 1
        REQUEST_LATENCIES.append(latency)
        
        # Keep only the last 1000 latencies to avoid memory issues
        if len(REQUEST_LATENCIES) > 1000:
            REQUEST_LATENCIES = REQUEST_LATENCIES[-1000:]
        
        if is_error:
            ERROR_COUNT += 1
    
    @staticmethod
    def reset_metrics() -> None:
        """Reset metrics (for testing)"""
        global REQUEST_COUNT, REQUEST_LATENCIES, ERROR_COUNT
        
        REQUEST_COUNT = 0
        REQUEST_LATENCIES = []
        ERROR_COUNT = 0