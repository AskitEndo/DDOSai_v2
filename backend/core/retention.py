"""
Data retention policies and cleanup procedures
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from core.repositories import (
    TrafficPacketRepository, NetworkFlowRepository, DetectionResultRepository,
    SystemMetricsRepository, SimulationRunRepository
)

# Configure logging
logger = logging.getLogger(__name__)

# Repositories
traffic_packet_repo = TrafficPacketRepository()
network_flow_repo = NetworkFlowRepository()
detection_result_repo = DetectionResultRepository()
system_metrics_repo = SystemMetricsRepository()
simulation_run_repo = SimulationRunRepository()


class RetentionPolicy:
    """Data retention policy implementation"""
    
    @staticmethod
    async def cleanup_old_data(db: AsyncSession) -> Dict[str, int]:
        """Clean up old data based on retention policies"""
        try:
            result = {}
            
            # Clean up traffic packets older than 7 days
            packets_deleted = await RetentionPolicy.cleanup_traffic_packets(db, days=7)
            result["traffic_packets"] = packets_deleted
            
            # Clean up system metrics older than 30 days
            metrics_deleted = await RetentionPolicy.cleanup_system_metrics(db, days=30)
            result["system_metrics"] = metrics_deleted
            
            # Clean up completed simulations older than 30 days
            simulations_deleted = await RetentionPolicy.cleanup_simulations(db, days=30)
            result["simulations"] = simulations_deleted
            
            return result
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return {"error": str(e)}
    
    @staticmethod
    async def cleanup_traffic_packets(db: AsyncSession, days: int = 7) -> int:
        """Clean up traffic packets older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            deleted = await traffic_packet_repo.delete_old_packets(db, cutoff_date)
            logger.info(f"Deleted {deleted} traffic packets older than {days} days")
            return deleted
        except Exception as e:
            logger.error(f"Error cleaning up traffic packets: {e}")
            return 0
    
    @staticmethod
    async def cleanup_system_metrics(db: AsyncSession, days: int = 30) -> int:
        """Clean up system metrics older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            deleted = await system_metrics_repo.delete_old_metrics(db, cutoff_date)
            logger.info(f"Deleted {deleted} system metrics older than {days} days")
            return deleted
        except Exception as e:
            logger.error(f"Error cleaning up system metrics: {e}")
            return 0
    
    @staticmethod
    async def cleanup_simulations(db: AsyncSession, days: int = 30) -> int:
        """Clean up completed simulations older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Get old simulations
            result = await db.execute(
                """
                DELETE FROM simulation_runs
                WHERE status IN ('completed', 'error')
                AND created_at < :cutoff_date
                RETURNING id
                """,
                {"cutoff_date": cutoff_date}
            )
            
            deleted_ids = result.fetchall()
            deleted_count = len(deleted_ids)
            
            await db.commit()
            
            logger.info(f"Deleted {deleted_count} completed simulations older than {days} days")
            return deleted_count
        except Exception as e:
            logger.error(f"Error cleaning up simulations: {e}")
            return 0
    
    @staticmethod
    async def schedule_cleanup(db: AsyncSession) -> None:
        """Schedule cleanup tasks"""
        try:
            # This would typically be called by a scheduler like APScheduler
            # For now, we'll just run the cleanup directly
            result = await RetentionPolicy.cleanup_old_data(db)
            logger.info(f"Scheduled cleanup completed: {result}")
        except Exception as e:
            logger.error(f"Error in scheduled cleanup: {e}")


# Retention policy configuration
RETENTION_CONFIG = {
    "traffic_packets": {
        "days": 7,
        "description": "Traffic packets are kept for 7 days"
    },
    "network_flows": {
        "days": 14,
        "description": "Network flows are kept for 14 days"
    },
    "detection_results": {
        "days": 30,
        "description": "Detection results are kept for 30 days"
    },
    "system_metrics": {
        "days": 30,
        "description": "System metrics are kept for 30 days"
    },
    "simulations": {
        "days": 30,
        "description": "Completed simulations are kept for 30 days"
    }
}