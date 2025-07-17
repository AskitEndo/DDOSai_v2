"""
Database service for handling database operations
"""
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, TypeVar, Generic, Type
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

from core.database import get_db
from core.repositories import (
    TrafficPacketRepository, NetworkFlowRepository, DetectionResultRepository,
    NetworkNodeRepository, NetworkEdgeRepository, SystemMetricsRepository,
    SimulationRunRepository, RedisCacheService
)
from core.timeseries import TimeSeriesService
from models.data_models import (
    TrafficPacket, NetworkFlow, DetectionResult,
    NetworkNode, NetworkEdge, AttackType, ProtocolType
)
from models.db_models import (
    DBTrafficPacket, DBNetworkFlow, DBDetectionResult,
    DBNetworkNode, DBNetworkEdge, DBSystemMetrics, DBSimulationRun
)

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar('T')
M = TypeVar('M')

# Initialize repositories
traffic_packet_repo = TrafficPacketRepository()
network_flow_repo = NetworkFlowRepository()
detection_result_repo = DetectionResultRepository()
network_node_repo = NetworkNodeRepository()
network_edge_repo = NetworkEdgeRepository()
system_metrics_repo = SystemMetricsRepository()
simulation_run_repo = SimulationRunRepository()

# Initialize Redis cache service
redis_cache = RedisCacheService()


class DatabaseService:
    """Service for database operations"""
    
    @staticmethod
    async def store_traffic_packet(packet: TrafficPacket) -> Optional[DBTrafficPacket]:
        """Store a traffic packet in the database"""
        try:
            async for db in get_db():
                # Check if packet already exists
                existing = await traffic_packet_repo.get_by_packet_id(db, packet.packet_id)
                if existing:
                    logger.debug(f"Packet {packet.packet_id} already exists in database")
                    return existing
                
                # Create new packet
                db_packet = await traffic_packet_repo.create(db, packet)
                logger.debug(f"Stored packet {packet.packet_id} in database")
                
                # Store in time-series database
                TimeSeriesService.write_traffic_metrics(packet)
                
                return db_packet
        except Exception as e:
            logger.error(f"Error storing traffic packet: {e}")
            return None
    
    @staticmethod
    async def store_detection_result(detection: DetectionResult) -> Optional[DBDetectionResult]:
        """Store a detection result in the database"""
        try:
            async for db in get_db():
                # Check if detection already exists
                existing = await detection_result_repo.get_by_packet_id(db, detection.packet_id)
                if existing:
                    logger.debug(f"Detection for packet {detection.packet_id} already exists in database")
                    return existing
                
                # Create new detection result
                db_detection = await detection_result_repo.create(db, detection)
                logger.debug(f"Stored detection result for packet {detection.packet_id} in database")
                
                # Update network nodes and edges based on detection
                await DatabaseService.update_network_graph(db, detection)
                
                # Cache detection result in Redis
                await redis_cache.set_cache(
                    f"detection:{detection.packet_id}", 
                    detection.to_dict(),
                    expire=3600  # 1 hour
                )
                
                # Add to recent detections list in Redis
                await redis_cache.add_to_list("recent_detections", detection.to_dict())
                await redis_cache.trim_list("recent_detections", 0, 999)  # Keep last 1000
                
                return db_detection
        except Exception as e:
            logger.error(f"Error storing detection result: {e}")
            return None
    
    @staticmethod
    async def update_network_graph(db: AsyncSession, detection: DetectionResult) -> None:
        """Update network graph based on detection result"""
        try:
            # Get or create source node
            src_node = await network_node_repo.get_by_ip(db, detection.src_ip)
            if not src_node:
                src_node = await network_node_repo.create(db, {
                    "node_id": f"node_{uuid.uuid4().hex[:8]}",
                    "ip_address": detection.src_ip,
                    "packet_count": 1,
                    "byte_count": 0,  # Would be set from packet size
                    "connection_count": 1,
                    "threat_score": detection.threat_score if detection.is_malicious else 0,
                    "is_malicious": detection.is_malicious,
                    "first_seen": detection.timestamp,
                    "last_seen": detection.timestamp
                })
            else:
                # Update existing node
                src_node.packet_count += 1
                src_node.connection_count += 1
                src_node.last_seen = detection.timestamp
                if detection.is_malicious and detection.threat_score > src_node.threat_score:
                    src_node.threat_score = detection.threat_score
                    src_node.is_malicious = True
                db.add(src_node)
            
            # Get or create destination node
            dst_node = await network_node_repo.get_by_ip(db, detection.dst_ip)
            if not dst_node:
                dst_node = await network_node_repo.create(db, {
                    "node_id": f"node_{uuid.uuid4().hex[:8]}",
                    "ip_address": detection.dst_ip,
                    "packet_count": 1,
                    "byte_count": 0,  # Would be set from packet size
                    "connection_count": 1,
                    "threat_score": 0,  # Destination is not considered malicious
                    "is_malicious": False,
                    "first_seen": detection.timestamp,
                    "last_seen": detection.timestamp
                })
            else:
                # Update existing node
                dst_node.packet_count += 1
                dst_node.connection_count += 1
                dst_node.last_seen = detection.timestamp
                db.add(dst_node)
            
            # Get or create edge
            edge = await network_edge_repo.get_by_ip_pair(db, detection.src_ip, detection.dst_ip)
            if not edge:
                edge = await network_edge_repo.create(db, {
                    "edge_id": f"edge_{uuid.uuid4().hex[:8]}",
                    "source_ip": detection.src_ip,
                    "target_ip": detection.dst_ip,
                    "flow_count": 1,
                    "total_bytes": 0,  # Would be set from packet size
                    "avg_packet_size": 0,  # Would be calculated
                    "connection_duration": 0,  # Would be calculated
                    "protocols": [detection.protocol.value] if hasattr(detection, 'protocol') else []
                })
            else:
                # Update existing edge
                edge.flow_count += 1
                if hasattr(detection, 'protocol') and detection.protocol.value not in edge.protocols:
                    edge.protocols.append(detection.protocol.value)
                db.add(edge)
            
            await db.commit()
        except Exception as e:
            logger.error(f"Error updating network graph: {e}")
            await db.rollback()
    
    @staticmethod
    async def store_system_metrics(metrics: Dict[str, Any]) -> Optional[DBSystemMetrics]:
        """Store system metrics in the database"""
        try:
            async for db in get_db():
                # Create new metrics record
                db_metrics = await system_metrics_repo.create(db, metrics)
                logger.debug(f"Stored system metrics in database")
                
                # Store in time-series database
                TimeSeriesService.write_system_metrics(metrics)
                
                # Cache latest metrics in Redis
                await redis_cache.set_cache(
                    "latest_metrics", 
                    metrics,
                    expire=300  # 5 minutes
                )
                
                return db_metrics
        except Exception as e:
            logger.error(f"Error storing system metrics: {e}")
            return None
    
    @staticmethod
    async def store_simulation_run(simulation: Dict[str, Any]) -> Optional[DBSimulationRun]:
        """Store a simulation run in the database"""
        try:
            async for db in get_db():
                # Check if simulation already exists
                existing = await simulation_run_repo.get_by_simulation_id(db, simulation["simulation_id"])
                if existing:
                    # Update existing simulation
                    db_simulation = await simulation_run_repo.update(db, db_obj=existing, obj_in=simulation)
                    logger.debug(f"Updated simulation {simulation['simulation_id']} in database")
                else:
                    # Create new simulation
                    db_simulation = await simulation_run_repo.create(db, simulation)
                    logger.debug(f"Stored simulation {simulation['simulation_id']} in database")
                
                # Store in time-series database
                TimeSeriesService.write_simulation_metrics(simulation)
                
                # Cache simulation in Redis
                await redis_cache.set_cache(
                    f"simulation:{simulation['simulation_id']}", 
                    simulation,
                    expire=3600  # 1 hour
                )
                
                return db_simulation
        except Exception as e:
            logger.error(f"Error storing simulation run: {e}")
            return None
    
    @staticmethod
    async def get_recent_detections(limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent detection results"""
        try:
            # Try to get from Redis cache first
            cached = await redis_cache.get_list("recent_detections", 0, limit - 1)
            if cached:
                logger.debug(f"Retrieved {len(cached)} recent detections from Redis cache")
                return cached
            
            # If not in cache, get from database
            async for db in get_db():
                detections = await detection_result_repo.get_recent(db, limit=limit)
                result = [d.to_dict() for d in detections]
                
                # Cache results in Redis
                for detection in result:
                    await redis_cache.add_to_list("recent_detections", detection)
                await redis_cache.trim_list("recent_detections", 0, 999)  # Keep last 1000
                
                logger.debug(f"Retrieved {len(result)} recent detections from database")
                return result
        except Exception as e:
            logger.error(f"Error getting recent detections: {e}")
            return []
    
    @staticmethod
    async def get_network_graph() -> Dict[str, Any]:
        """Get current network graph"""
        try:
            # Try to get from Redis cache first
            cached = await redis_cache.get_cache("network_graph")
            if cached:
                logger.debug("Retrieved network graph from Redis cache")
                return cached
            
            # If not in cache, get from database
            async for db in get_db():
                nodes = await network_node_repo.get_multi(db, limit=1000)
                edges = await network_edge_repo.get_multi(db, limit=1000)
                
                result = {
                    "nodes": [n.to_dict() for n in nodes],
                    "edges": [e.to_dict() for e in edges],
                    "timestamp": datetime.now().isoformat()
                }
                
                # Cache result in Redis
                await redis_cache.set_cache(
                    "network_graph", 
                    result,
                    expire=60  # 1 minute
                )
                
                logger.debug(f"Retrieved network graph from database: {len(nodes)} nodes, {len(edges)} edges")
                return result
        except Exception as e:
            logger.error(f"Error getting network graph: {e}")
            return {"nodes": [], "edges": [], "timestamp": datetime.now().isoformat()}
    
    @staticmethod
    async def get_system_metrics() -> Dict[str, Any]:
        """Get latest system metrics"""
        try:
            # Try to get from Redis cache first
            cached = await redis_cache.get_cache("latest_metrics")
            if cached:
                logger.debug("Retrieved system metrics from Redis cache")
                return cached
            
            # If not in cache, get from database
            async for db in get_db():
                metrics = await system_metrics_repo.get_recent(db, limit=1)
                if metrics:
                    result = metrics[0].to_dict()
                    
                    # Cache result in Redis
                    await redis_cache.set_cache(
                        "latest_metrics", 
                        result,
                        expire=300  # 5 minutes
                    )
                    
                    logger.debug("Retrieved system metrics from database")
                    return result
                else:
                    return {
                        "timestamp": datetime.now().isoformat(),
                        "packets_processed": 0,
                        "processing_latency_ms": 0,
                        "cpu_usage": 0,
                        "memory_usage": 0,
                        "active_connections": 0,
                        "threat_level": 0,
                        "malicious_packets": 0,
                        "total_detections": 0
                    }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "packets_processed": 0,
                "processing_latency_ms": 0,
                "cpu_usage": 0,
                "memory_usage": 0,
                "active_connections": 0,
                "threat_level": 0,
                "malicious_packets": 0,
                "total_detections": 0
            }
    
    @staticmethod
    async def get_simulation_status(simulation_id: str) -> Optional[Dict[str, Any]]:
        """Get simulation status"""
        try:
            # Try to get from Redis cache first
            cached = await redis_cache.get_cache(f"simulation:{simulation_id}")
            if cached:
                logger.debug(f"Retrieved simulation {simulation_id} from Redis cache")
                return cached
            
            # If not in cache, get from database
            async for db in get_db():
                simulation = await simulation_run_repo.get_by_simulation_id(db, simulation_id)
                if simulation:
                    result = simulation.to_dict()
                    
                    # Cache result in Redis
                    await redis_cache.set_cache(
                        f"simulation:{simulation_id}", 
                        result,
                        expire=3600  # 1 hour
                    )
                    
                    logger.debug(f"Retrieved simulation {simulation_id} from database")
                    return result
        except Exception as e:
            logger.error(f"Error getting simulation status: {e}")
            return None
    
    @staticmethod
    async def get_recent_simulations(limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent simulation runs"""
        try:
            # Try to get from Redis cache first
            cached = await redis_cache.get_list("recent_simulations", 0, limit - 1)
            if cached:
                logger.debug(f"Retrieved {len(cached)} recent simulations from Redis cache")
                return cached
            
            # If not in cache, get from database
            async for db in get_db():
                simulations = await simulation_run_repo.get_recent(db, limit=limit)
                result = [s.to_dict() for s in simulations]
                
                # Cache results in Redis
                for simulation in result:
                    await redis_cache.add_to_list("recent_simulations", simulation)
                await redis_cache.trim_list("recent_simulations", 0, 99)  # Keep last 100
                
                logger.debug(f"Retrieved {len(result)} recent simulations from database")
                return result
        except Exception as e:
            logger.error(f"Error getting recent simulations: {e}")
            return []