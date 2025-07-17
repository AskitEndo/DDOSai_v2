"""
Base traffic ingestor interface
"""
from abc import ABC, abstractmethod
from typing import Iterator, List, Dict, Any, Optional
from datetime import datetime
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_models import TrafficPacket, ProtocolType
from core.exceptions import TrafficIngestionError


class BaseTrafficIngestor(ABC):
    """Abstract base class for all traffic ingestion methods"""
    
    def __init__(self, source_name: str):
        self.source_name = source_name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.packets_processed = 0
        self.errors_encountered = 0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    @abstractmethod
    def ingest(self, source: str, **kwargs) -> Iterator[TrafficPacket]:
        """
        Ingest traffic from the specified source
        
        Args:
            source: Path to file, interface name, or connection string
            **kwargs: Additional parameters specific to ingestion method
            
        Yields:
            TrafficPacket: Individual packets with extracted features
            
        Raises:
            TrafficIngestionError: If ingestion fails
        """
        pass
    
    @abstractmethod
    def validate_source(self, source: str) -> bool:
        """
        Validate that the source is accessible and properly formatted
        
        Args:
            source: Source identifier to validate
            
        Returns:
            bool: True if source is valid
            
        Raises:
            TrafficIngestionError: If source is invalid
        """
        pass
    
    def start_ingestion(self):
        """Mark the start of ingestion process"""
        self.start_time = datetime.now()
        self.packets_processed = 0
        self.errors_encountered = 0
        self.logger.info(f"Starting ingestion from {self.source_name}")
    
    def end_ingestion(self):
        """Mark the end of ingestion process"""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds() if self.start_time else 0
        self.logger.info(
            f"Ingestion completed: {self.packets_processed} packets processed, "
            f"{self.errors_encountered} errors, {duration:.2f}s duration"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get ingestion statistics"""
        duration = 0
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "source_name": self.source_name,
            "packets_processed": self.packets_processed,
            "errors_encountered": self.errors_encountered,
            "duration_seconds": duration,
            "packets_per_second": self.packets_processed / duration if duration > 0 else 0,
            "error_rate": self.errors_encountered / self.packets_processed if self.packets_processed > 0 else 0
        }
    
    def apply_filters(self, packet: TrafficPacket, filters: Dict[str, Any]) -> bool:
        """
        Apply filtering criteria to a packet
        
        Args:
            packet: Traffic packet to filter
            filters: Dictionary of filter criteria
            
        Returns:
            bool: True if packet passes all filters
        """
        if not filters:
            return True
        
        # IP address filters
        if "src_ip_whitelist" in filters:
            if packet.src_ip not in filters["src_ip_whitelist"]:
                return False
        
        if "src_ip_blacklist" in filters:
            if packet.src_ip in filters["src_ip_blacklist"]:
                return False
        
        if "dst_ip_whitelist" in filters:
            if packet.dst_ip not in filters["dst_ip_whitelist"]:
                return False
        
        if "dst_ip_blacklist" in filters:
            if packet.dst_ip in filters["dst_ip_blacklist"]:
                return False
        
        # Protocol filters
        if "protocols" in filters:
            if packet.protocol not in filters["protocols"]:
                return False
        
        # Port filters
        if "port_range" in filters:
            port_min, port_max = filters["port_range"]
            if not (port_min <= packet.src_port <= port_max or port_min <= packet.dst_port <= port_max):
                return False
        
        # Packet size filters
        if "min_packet_size" in filters:
            if packet.packet_size < filters["min_packet_size"]:
                return False
        
        if "max_packet_size" in filters:
            if packet.packet_size > filters["max_packet_size"]:
                return False
        
        # Time range filters
        if "start_time" in filters:
            if packet.timestamp < filters["start_time"]:
                return False
        
        if "end_time" in filters:
            if packet.timestamp > filters["end_time"]:
                return False
        
        return True
    
    def preprocess_packet(self, packet: TrafficPacket) -> TrafficPacket:
        """
        Apply preprocessing to a packet (normalization, feature extraction, etc.)
        
        Args:
            packet: Raw traffic packet
            
        Returns:
            TrafficPacket: Preprocessed packet
        """
        # Basic preprocessing - can be overridden by subclasses
        
        # Normalize protocol names
        if packet.protocol == ProtocolType.HTTP and packet.dst_port == 443:
            # HTTP over port 443 is likely HTTPS
            packet.protocol = ProtocolType.HTTPS
        
        # Ensure packet ID is set
        if not packet.packet_id:
            packet.packet_id = f"pkt_{self.packets_processed:06d}_{int(packet.timestamp.timestamp())}"
        
        return packet