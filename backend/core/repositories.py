"""
Repository classes for database operations
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, desc, and_, or_
from sqlalchemy.future import select
from datetime import datetime, timedelta
import json
import logging
from typing import List, Dict, Any, Optional, TypeVar, Generic, Type, Union

from models.db_models import (
    DBTrafficPacket, DBNetworkFlow, DBDetectionResult,
    DBNetworkNode, DBNetworkEdge, DBSystemMetrics, DBSimulationRun
)
from models.data_models import (
    TrafficPacket, NetworkFlow, DetectionResult,
    NetworkNode, NetworkEdge, AttackType, ProtocolType
)
from core.database import redis_client

# Configure logging
logger = logging.getLogger(__name__)

# Generic type for database models
T = TypeVar('T')
M = TypeVar('M')  # Memory model type


class BaseRepository(Generic[T, M]):
    """Base repository for database operations"""
    
    def __init__(self, db_model: Type[T], memory_model: Type[M]):
        self.db_model = db_model
        self.memory_model = memory_model
    
    async def create(self, db: AsyncSession, obj_in: Union[Dict[str, Any], M]) -> T:
        """Create a new record"""
        if isinstance(obj_in, dict):
            obj_data = obj_in
        else:
            obj_data = obj_in.to_dict()
        
        db_obj = self.db_model(**obj_data)
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj
    
    async def get(self, db: AsyncSession, id: int) -> Optional[T]:
        """Get a record by ID"""
        result = await db.execute(select(self.db_model).where(self.db_model.id == id))
        return result.scalars().first()
    
    async def get_by_field(self, db: AsyncSession, field: str, value: Any) -> Optional[T]:
        """Get a record by a specific field"""
        result = await db.execute(
            select(self.db_model).where(getattr(self.db_model, field) == value)
        )
        return result.scalars().first()
    
    async def get_multi(
        self, db: AsyncSession, *, skip: int = 0, limit: int = 100
    ) -> List[T]:
        """Get multiple records"""
        result = await db.execute(
            select(self.db_model).offset(skip).limit(limit)
        )
        return result.scalars().all()
    
    async def update(
        self, db: AsyncSession, *, db_obj: T, obj_in: Union[Dict[str, Any], M]
    ) -> T:
        """Update a record"""
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.to_dict()
        
        for field, value in update_data.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj
    
    async def delete(self, db: AsyncSession, *, id: int) -> T:
        """Delete a record"""
        obj = await self.get(db, id)
        await db.delete(obj)
        await db.commit()
        return obj
    
    async def count(self, db: AsyncSession) -> int:
        """Count all records"""
        result = await db.execute(select(func.count()).select_from(self.db_model))
        return result.scalar_one()


class TrafficPacketRepository(BaseRepository[DBTrafficPacket, TrafficPacket]):
    """Repository for traffic packets"""
    
    def __init__(self):
        super().__init__(DBTrafficPacket, TrafficPacket)
    
    async def get_by_packet_id(self, db: AsyncSession, packet_id: str) -> Optional[DBTrafficPacket]:
        """Get a packet by packet_id"""
        return await self.get_by_field(db, "packet_id", packet_id)
    
    async def get_recent(
        self, db: AsyncSession, *, limit: int = 100
    ) -> List[DBTrafficPacket]:
        """Get recent packets"""
        result = await db.execute(
            select(self.db_model)
            .order_by(desc(self.db_model.timestamp))
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_ip(
        self, db: AsyncSession, ip: str, *, limit: int = 100
    ) -> List[DBTrafficPacket]:
        """Get packets by IP address"""
        result = await db.execute(
            select(self.db_model)
            .where(or_(self.db_model.src_ip == ip, self.db_model.dst_ip == ip))
            .order_by(desc(self.db_model.timestamp))
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_time_range(
        self, db: AsyncSession, start_time: datetime, end_time: datetime, *, limit: int = 100
    ) -> List[DBTrafficPacket]:
        """Get packets by time range"""
        result = await db.execute(
            select(self.db_model)
            .where(and_(
                self.db_model.timestamp >= start_time,
                self.db_model.timestamp <= end_time
            ))
            .order_by(desc(self.db_model.timestamp))
            .limit(limit)
        )
        return result.scalars().all()
    
    async def delete_old_packets(
        self, db: AsyncSession, older_than: datetime
    ) -> int:
        """Delete packets older than a specific time"""
        result = await db.execute(
            delete(self.db_model).where(self.db_model.timestamp < older_than)
        )
        await db.commit()
        return result.rowcount


class NetworkFlowRepository(BaseRepository[DBNetworkFlow, NetworkFlow]):
    """Repository for network flows"""
    
    def __init__(self):
        super().__init__(DBNetworkFlow, NetworkFlow)
    
    async def get_by_flow_id(self, db: AsyncSession, flow_id: str) -> Optional[DBNetworkFlow]:
        """Get a flow by flow_id"""
        return await self.get_by_field(db, "flow_id", flow_id)
    
    async def get_by_ip_pair(
        self, db: AsyncSession, src_ip: str, dst_ip: str, *, limit: int = 100
    ) -> List[DBNetworkFlow]:
        """Get flows by IP pair"""
        result = await db.execute(
            select(self.db_model)
            .where(and_(
                self.db_model.src_ip == src_ip,
                self.db_model.dst_ip == dst_ip
            ))
            .order_by(desc(self.db_model.start_time))
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_active_flows(
        self, db: AsyncSession, *, limit: int = 100
    ) -> List[DBNetworkFlow]:
        """Get active flows (not ended yet)"""
        now = datetime.now()
        result = await db.execute(
            select(self.db_model)
            .where(self.db_model.end_time > now)
            .order_by(desc(self.db_model.start_time))
            .limit(limit)
        )
        return result.scalars().all()


class DetectionResultRepository(BaseRepository[DBDetectionResult, DetectionResult]):
    """Repository for detection results"""
    
    def __init__(self):
        super().__init__(DBDetectionResult, DetectionResult)
    
    async def get_by_detection_id(self, db: AsyncSession, detection_id: str) -> Optional[DBDetectionResult]:
        """Get a detection result by detection_id"""
        return await self.get_by_field(db, "detection_id", detection_id)
    
    async def get_by_packet_id(self, db: AsyncSession, packet_id: str) -> Optional[DBDetectionResult]:
        """Get a detection result by packet_id"""
        return await self.get_by_field(db, "packet_id", packet_id)
    
    async def get_malicious(
        self, db: AsyncSession, *, limit: int = 100
    ) -> List[DBDetectionResult]:
        """Get malicious detection results"""
        result = await db.execute(
            select(self.db_model)
            .where(self.db_model.is_malicious == True)
            .order_by(desc(self.db_model.timestamp))
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_attack_type(
        self, db: AsyncSession, attack_type: str, *, limit: int = 100
    ) -> List[DBDetectionResult]:
        """Get detection results by attack type"""
        result = await db.execute(
            select(self.db_model)
            .where(self.db_model.attack_type == attack_type)
            .order_by(desc(self.db_model.timestamp))
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_recent(
        self, db: AsyncSession, *, limit: int = 100
    ) -> List[DBDetectionResult]:
        """Get recent detection results"""
        result = await db.execute(
            select(self.db_model)
            .order_by(desc(self.db_model.timestamp))
            .limit(limit)
        )
        return result.scalars().all()


class NetworkNodeRepository(BaseRepository[DBNetworkNode, NetworkNode]):
    """Repository for network nodes"""
    
    def __init__(self):
        super().__init__(DBNetworkNode, NetworkNode)
    
    async def get_by_node_id(self, db: AsyncSession, node_id: str) -> Optional[DBNetworkNode]:
        """Get a node by node_id"""
        return await self.get_by_field(db, "node_id", node_id)
    
    async def get_by_ip(self, db: AsyncSession, ip_address: str) -> Optional[DBNetworkNode]:
        """Get a node by IP address"""
        return await self.get_by_field(db, "ip_address", ip_address)
    
    async def get_malicious(
        self, db: AsyncSession, *, limit: int = 100
    ) -> List[DBNetworkNode]:
        """Get malicious nodes"""
        result = await db.execute(
            select(self.db_model)
            .where(self.db_model.is_malicious == True)
            .order_by(desc(self.db_model.threat_score))
            .limit(limit)
        )
        return result.scalars().all()


class NetworkEdgeRepository(BaseRepository[DBNetworkEdge, NetworkEdge]):
    """Repository for network edges"""
    
    def __init__(self):
        super().__init__(DBNetworkEdge, NetworkEdge)
    
    async def get_by_edge_id(self, db: AsyncSession, edge_id: str) -> Optional[DBNetworkEdge]:
        """Get an edge by edge_id"""
        return await self.get_by_field(db, "edge_id", edge_id)
    
    async def get_by_ip_pair(
        self, db: AsyncSession, source_ip: str, target_ip: str
    ) -> Optional[DBNetworkEdge]:
        """Get an edge by IP pair"""
        result = await db.execute(
            select(self.db_model)
            .where(and_(
                self.db_model.source_ip == source_ip,
                self.db_model.target_ip == target_ip
            ))
        )
        return result.scalars().first()
    
    async def get_by_ip(
        self, db: AsyncSession, ip: str, *, limit: int = 100
    ) -> List[DBNetworkEdge]:
        """Get edges by IP address"""
        result = await db.execute(
            select(self.db_model)
            .where(or_(
                self.db_model.source_ip == ip,
                self.db_model.target_ip == ip
            ))
            .limit(limit)
        )
        return result.scalars().all()


class SystemMetricsRepository(BaseRepository[DBSystemMetrics, Dict[str, Any]]):
    """Repository for system metrics"""
    
    def __init__(self):
        super().__init__(DBSystemMetrics, dict)
    
    async def get_recent(
        self, db: AsyncSession, *, limit: int = 100
    ) -> List[DBSystemMetrics]:
        """Get recent system metrics"""
        result = await db.execute(
            select(self.db_model)
            .order_by(desc(self.db_model.timestamp))
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_time_range(
        self, db: AsyncSession, start_time: datetime, end_time: datetime, *, limit: int = 100
    ) -> List[DBSystemMetrics]:
        """Get system metrics by time range"""
        result = await db.execute(
            select(self.db_model)
            .where(and_(
                self.db_model.timestamp >= start_time,
                self.db_model.timestamp <= end_time
            ))
            .order_by(desc(self.db_model.timestamp))
            .limit(limit)
        )
        return result.scalars().all()
    
    async def delete_old_metrics(
        self, db: AsyncSession, older_than: datetime
    ) -> int:
        """Delete metrics older than a specific time"""
        result = await db.execute(
            delete(self.db_model).where(self.db_model.timestamp < older_than)
        )
        await db.commit()
        return result.rowcount


class SimulationRunRepository(BaseRepository[DBSimulationRun, Dict[str, Any]]):
    """Repository for simulation runs"""
    
    def __init__(self):
        super().__init__(DBSimulationRun, dict)
    
    async def get_by_simulation_id(self, db: AsyncSession, simulation_id: str) -> Optional[DBSimulationRun]:
        """Get a simulation run by simulation_id"""
        return await self.get_by_field(db, "simulation_id", simulation_id)
    
    async def get_by_status(
        self, db: AsyncSession, status: str, *, limit: int = 100
    ) -> List[DBSimulationRun]:
        """Get simulation runs by status"""
        result = await db.execute(
            select(self.db_model)
            .where(self.db_model.status == status)
            .order_by(desc(self.db_model.start_time))
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_recent(
        self, db: AsyncSession, *, limit: int = 100
    ) -> List[DBSimulationRun]:
        """Get recent simulation runs"""
        result = await db.execute(
            select(self.db_model)
            .order_by(desc(self.db_model.created_at))
            .limit(limit)
        )
        return result.scalars().all()


class RedisCacheService:
    """Service for Redis cache operations"""
    
    @staticmethod
    async def set_cache(key: str, value: Any, expire: int = 3600) -> bool:
        """Set a value in Redis cache"""
        if redis_client is None:
            logger.warning("Redis client not initialized")
            return False
        
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            await redis_client.set(key, value, ex=expire)
            return True
        except Exception as e:
            logger.error(f"Error setting Redis cache: {e}")
            return False
    
    @staticmethod
    async def get_cache(key: str) -> Any:
        """Get a value from Redis cache"""
        if redis_client is None:
            logger.warning("Redis client not initialized")
            return None
        
        try:
            value = await redis_client.get(key)
            if value is None:
                return None
            
            try:
                # Try to parse as JSON
                return json.loads(value)
            except json.JSONDecodeError:
                # Return as is if not JSON
                return value
        except Exception as e:
            logger.error(f"Error getting Redis cache: {e}")
            return None
    
    @staticmethod
    async def delete_cache(key: str) -> bool:
        """Delete a value from Redis cache"""
        if redis_client is None:
            logger.warning("Redis client not initialized")
            return False
        
        try:
            await redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting Redis cache: {e}")
            return False
    
    @staticmethod
    async def set_hash(key: str, field: str, value: Any) -> bool:
        """Set a hash field in Redis cache"""
        if redis_client is None:
            logger.warning("Redis client not initialized")
            return False
        
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            await redis_client.hset(key, field, value)
            return True
        except Exception as e:
            logger.error(f"Error setting Redis hash: {e}")
            return False
    
    @staticmethod
    async def get_hash(key: str, field: str) -> Any:
        """Get a hash field from Redis cache"""
        if redis_client is None:
            logger.warning("Redis client not initialized")
            return None
        
        try:
            value = await redis_client.hget(key, field)
            if value is None:
                return None
            
            try:
                # Try to parse as JSON
                return json.loads(value)
            except json.JSONDecodeError:
                # Return as is if not JSON
                return value
        except Exception as e:
            logger.error(f"Error getting Redis hash: {e}")
            return None
    
    @staticmethod
    async def get_hash_all(key: str) -> Dict[str, Any]:
        """Get all hash fields from Redis cache"""
        if redis_client is None:
            logger.warning("Redis client not initialized")
            return {}
        
        try:
            values = await redis_client.hgetall(key)
            if not values:
                return {}
            
            result = {}
            for field, value in values.items():
                try:
                    # Try to parse as JSON
                    result[field] = json.loads(value)
                except json.JSONDecodeError:
                    # Return as is if not JSON
                    result[field] = value
            
            return result
        except Exception as e:
            logger.error(f"Error getting Redis hash all: {e}")
            return {}
    
    @staticmethod
    async def delete_hash_field(key: str, field: str) -> bool:
        """Delete a hash field from Redis cache"""
        if redis_client is None:
            logger.warning("Redis client not initialized")
            return False
        
        try:
            await redis_client.hdel(key, field)
            return True
        except Exception as e:
            logger.error(f"Error deleting Redis hash field: {e}")
            return False
    
    @staticmethod
    async def add_to_list(key: str, value: Any) -> bool:
        """Add a value to a Redis list"""
        if redis_client is None:
            logger.warning("Redis client not initialized")
            return False
        
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            await redis_client.lpush(key, value)
            return True
        except Exception as e:
            logger.error(f"Error adding to Redis list: {e}")
            return False
    
    @staticmethod
    async def get_list(key: str, start: int = 0, end: int = -1) -> List[Any]:
        """Get values from a Redis list"""
        if redis_client is None:
            logger.warning("Redis client not initialized")
            return []
        
        try:
            values = await redis_client.lrange(key, start, end)
            if not values:
                return []
            
            result = []
            for value in values:
                try:
                    # Try to parse as JSON
                    result.append(json.loads(value))
                except json.JSONDecodeError:
                    # Return as is if not JSON
                    result.append(value)
            
            return result
        except Exception as e:
            logger.error(f"Error getting Redis list: {e}")
            return []
    
    @staticmethod
    async def trim_list(key: str, start: int = 0, end: int = 999) -> bool:
        """Trim a Redis list to a specific range"""
        if redis_client is None:
            logger.warning("Redis client not initialized")
            return False
        
        try:
            await redis_client.ltrim(key, start, end)
            return True
        except Exception as e:
            logger.error(f"Error trimming Redis list: {e}")
            return False