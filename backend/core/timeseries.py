"""
Time-series data storage using InfluxDB
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from influxdb_client import Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from core.database import get_influxdb_write_api, get_influxdb_query_api
from models.data_models import TrafficPacket, DetectionResult

# Configure logging
logger = logging.getLogger(__name__)

# InfluxDB bucket and organization
INFLUXDB_BUCKET = "ddosai_metrics"
INFLUXDB_ORG = "ddosai"


class TimeSeriesService:
    """Service for time-series data storage and retrieval"""
    
    @staticmethod
    def write_traffic_metrics(packet: TrafficPacket, detection: Optional[DetectionResult] = None) -> bool:
        """Write traffic metrics to InfluxDB"""
        try:
            write_api = get_influxdb_write_api()
            if not write_api:
                logger.warning("InfluxDB write API not initialized")
                return False
            
            # Create point for traffic metrics
            point = Point("traffic_metrics") \
                .tag("src_ip", packet.src_ip) \
                .tag("dst_ip", packet.dst_ip) \
                .tag("protocol", packet.protocol.value) \
                .field("packet_size", packet.packet_size) \
                .field("ttl", packet.ttl) \
                .field("payload_entropy", packet.payload_entropy) \
                .time(packet.timestamp, WritePrecision.NS)
            
            # Add detection fields if available
            if detection:
                point = point \
                    .tag("is_malicious", str(detection.is_malicious)) \
                    .tag("attack_type", detection.attack_type.value) \
                    .field("threat_score", detection.threat_score) \
                    .field("confidence", detection.confidence)
            
            # Write point to InfluxDB
            write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
            return True
        except Exception as e:
            logger.error(f"Error writing traffic metrics to InfluxDB: {e}")
            return False
    
    @staticmethod
    def write_system_metrics(metrics: Dict[str, Any]) -> bool:
        """Write system metrics to InfluxDB"""
        try:
            write_api = get_influxdb_write_api()
            if not write_api:
                logger.warning("InfluxDB write API not initialized")
                return False
            
            # Create point for system metrics
            point = Point("system_metrics") \
                .field("packets_processed", metrics["packets_processed"]) \
                .field("processing_latency_ms", metrics["processing_latency_ms"]) \
                .field("cpu_usage", metrics["cpu_usage"]) \
                .field("memory_usage", metrics["memory_usage"]) \
                .field("active_connections", metrics["active_connections"]) \
                .field("threat_level", metrics["threat_level"]) \
                .field("malicious_packets", metrics["malicious_packets"]) \
                .field("total_detections", metrics["total_detections"]) \
                .time(datetime.fromisoformat(metrics["timestamp"]), WritePrecision.NS)
            
            # Write point to InfluxDB
            write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
            return True
        except Exception as e:
            logger.error(f"Error writing system metrics to InfluxDB: {e}")
            return False
    
    @staticmethod
    def write_simulation_metrics(metrics: Dict[str, Any]) -> bool:
        """Write simulation metrics to InfluxDB"""
        try:
            write_api = get_influxdb_write_api()
            if not write_api:
                logger.warning("InfluxDB write API not initialized")
                return False
            
            # Create point for simulation metrics
            point = Point("simulation_metrics") \
                .tag("simulation_id", metrics["simulation_id"]) \
                .tag("attack_type", metrics["attack_type"]) \
                .tag("status", metrics["status"]) \
                .field("packets_sent", metrics["packets_sent"]) \
                .field("bytes_sent", metrics["bytes_sent"]) \
                .field("packet_rate", metrics.get("current_packet_rate", metrics["packet_rate"])) \
                .field("errors", metrics.get("errors", 0))
            
            # Add timestamp if available
            if "timestamp" in metrics:
                point = point.time(datetime.fromisoformat(metrics["timestamp"]), WritePrecision.NS)
            else:
                point = point.time(datetime.now(), WritePrecision.NS)
            
            # Write point to InfluxDB
            write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
            return True
        except Exception as e:
            logger.error(f"Error writing simulation metrics to InfluxDB: {e}")
            return False
    
    @staticmethod
    def query_traffic_metrics(
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, Any]] = None,
        aggregation: str = "1m"
    ) -> pd.DataFrame:
        """Query traffic metrics from InfluxDB"""
        try:
            query_api = get_influxdb_query_api()
            if not query_api:
                logger.warning("InfluxDB query API not initialized")
                return pd.DataFrame()
            
            # Build query
            query = f"""
                from(bucket: "{INFLUXDB_BUCKET}")
                |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                |> filter(fn: (r) => r._measurement == "traffic_metrics")
            """
            
            # Add filters if provided
            if filters:
                for key, value in filters.items():
                    if isinstance(value, list):
                        # Multiple values for the same key
                        conditions = " or ".join([f'r["{key}"] == "{v}"' for v in value])
                        query += f'\n|> filter(fn: (r) => {conditions})'
                    else:
                        # Single value
                        query += f'\n|> filter(fn: (r) => r["{key}"] == "{value}")'
            
            # Add aggregation
            query += f"""
                |> aggregateWindow(every: {aggregation}, fn: mean, createEmpty: false)
                |> yield(name: "mean")
            """
            
            # Execute query
            tables = query_api.query_data_frame(query, org=INFLUXDB_ORG)
            
            # Process results
            if isinstance(tables, list):
                if not tables:
                    return pd.DataFrame()
                return pd.concat(tables)
            return tables
        except Exception as e:
            logger.error(f"Error querying traffic metrics from InfluxDB: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def query_system_metrics(
        start_time: datetime,
        end_time: datetime,
        aggregation: str = "1m"
    ) -> pd.DataFrame:
        """Query system metrics from InfluxDB"""
        try:
            query_api = get_influxdb_query_api()
            if not query_api:
                logger.warning("InfluxDB query API not initialized")
                return pd.DataFrame()
            
            # Build query
            query = f"""
                from(bucket: "{INFLUXDB_BUCKET}")
                |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                |> filter(fn: (r) => r._measurement == "system_metrics")
                |> aggregateWindow(every: {aggregation}, fn: mean, createEmpty: false)
                |> yield(name: "mean")
            """
            
            # Execute query
            tables = query_api.query_data_frame(query, org=INFLUXDB_ORG)
            
            # Process results
            if isinstance(tables, list):
                if not tables:
                    return pd.DataFrame()
                return pd.concat(tables)
            return tables
        except Exception as e:
            logger.error(f"Error querying system metrics from InfluxDB: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def query_simulation_metrics(
        simulation_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Query simulation metrics from InfluxDB"""
        try:
            query_api = get_influxdb_query_api()
            if not query_api:
                logger.warning("InfluxDB query API not initialized")
                return pd.DataFrame()
            
            # Set default time range if not provided
            if not start_time:
                start_time = datetime.now() - timedelta(hours=1)
            if not end_time:
                end_time = datetime.now()
            
            # Build query
            query = f"""
                from(bucket: "{INFLUXDB_BUCKET}")
                |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                |> filter(fn: (r) => r._measurement == "simulation_metrics")
                |> filter(fn: (r) => r.simulation_id == "{simulation_id}")
                |> yield()
            """
            
            # Execute query
            tables = query_api.query_data_frame(query, org=INFLUXDB_ORG)
            
            # Process results
            if isinstance(tables, list):
                if not tables:
                    return pd.DataFrame()
                return pd.concat(tables)
            return tables
        except Exception as e:
            logger.error(f"Error querying simulation metrics from InfluxDB: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_traffic_statistics(
        start_time: datetime,
        end_time: datetime,
        group_by: str = "1h"
    ) -> Dict[str, Any]:
        """Get traffic statistics from InfluxDB"""
        try:
            query_api = get_influxdb_query_api()
            if not query_api:
                logger.warning("InfluxDB query API not initialized")
                return {}
            
            # Build query for packet count
            packet_query = f"""
                from(bucket: "{INFLUXDB_BUCKET}")
                |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                |> filter(fn: (r) => r._measurement == "traffic_metrics")
                |> count()
                |> group(columns: ["_time"], mode: "by")
                |> aggregateWindow(every: {group_by}, fn: sum, createEmpty: false)
                |> yield(name: "sum")
            """
            
            # Build query for malicious packet count
            malicious_query = f"""
                from(bucket: "{INFLUXDB_BUCKET}")
                |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                |> filter(fn: (r) => r._measurement == "traffic_metrics")
                |> filter(fn: (r) => r.is_malicious == "True")
                |> count()
                |> group(columns: ["_time"], mode: "by")
                |> aggregateWindow(every: {group_by}, fn: sum, createEmpty: false)
                |> yield(name: "sum")
            """
            
            # Execute queries
            packet_tables = query_api.query_data_frame(packet_query, org=INFLUXDB_ORG)
            malicious_tables = query_api.query_data_frame(malicious_query, org=INFLUXDB_ORG)
            
            # Process results
            packet_df = pd.concat(packet_tables) if isinstance(packet_tables, list) and packet_tables else packet_tables
            malicious_df = pd.concat(malicious_tables) if isinstance(malicious_tables, list) and malicious_tables else malicious_tables
            
            # Calculate statistics
            total_packets = packet_df["_value"].sum() if not packet_df.empty else 0
            malicious_packets = malicious_df["_value"].sum() if not malicious_df.empty else 0
            malicious_percentage = (malicious_packets / total_packets * 100) if total_packets > 0 else 0
            
            return {
                "total_packets": int(total_packets),
                "malicious_packets": int(malicious_packets),
                "malicious_percentage": float(malicious_percentage),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "group_by": group_by
            }
        except Exception as e:
            logger.error(f"Error getting traffic statistics from InfluxDB: {e}")
            return {}