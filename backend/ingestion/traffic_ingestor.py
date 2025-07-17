"""
Unified traffic ingestor that handles multiple data sources
"""
from typing import Iterator, Dict, Any, Optional, Union
from pathlib import Path
import logging

from .base_ingestor import BaseTrafficIngestor
from .pcap_ingestor import PcapIngestor
from .csv_ingestor import CsvIngestor
from .live_ingestor import LiveIngestor, SocketIngestor
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_models import TrafficPacket
from core.exceptions import TrafficIngestionError


class TrafficIngestor:
    """Unified interface for ingesting traffic from multiple sources"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._ingestors = {
            'pcap': PcapIngestor(),
            'csv': CsvIngestor(),
            'live': LiveIngestor(),
            'socket': SocketIngestor()
        }
    
    def ingest(self, source: str, source_type: Optional[str] = None, **kwargs) -> Iterator[TrafficPacket]:
        """
        Ingest traffic from any supported source
        
        Args:
            source: Source identifier (file path, interface name, socket address)
            source_type: Type of source ('pcap', 'csv', 'live', 'socket'). 
                        If None, will auto-detect based on source
            **kwargs: Additional parameters passed to specific ingestor
            
        Yields:
            TrafficPacket: Individual packets with extracted features
            
        Raises:
            TrafficIngestionError: If ingestion fails
        """
        # Auto-detect source type if not specified
        if source_type is None:
            source_type = self._detect_source_type(source)
        
        # Validate source type
        if source_type not in self._ingestors:
            raise TrafficIngestionError(
                f"Unsupported source type: {source_type}. "
                f"Supported types: {list(self._ingestors.keys())}"
            )
        
        # Get appropriate ingestor
        ingestor = self._ingestors[source_type]
        
        self.logger.info(f"Starting ingestion from {source_type} source: {source}")
        
        try:
            # Delegate to specific ingestor
            yield from ingestor.ingest(source, **kwargs)
            
        except Exception as e:
            self.logger.error(f"Ingestion failed: {e}")
            raise TrafficIngestionError(f"Failed to ingest from {source_type} source: {e}")
    
    def _detect_source_type(self, source: str) -> str:
        """Auto-detect source type based on source string"""
        source_lower = source.lower()
        
        # Check for file extensions
        if source_lower.endswith(('.pcap', '.pcapng', '.cap')):
            return 'pcap'
        elif source_lower.endswith('.csv'):
            return 'csv'
        
        # Check for socket format (host:port)
        if ':' in source and not Path(source).exists():
            try:
                host, port_str = source.rsplit(':', 1)
                int(port_str)  # Validate port is numeric
                return 'socket'
            except (ValueError, IndexError):
                pass
        
        # Check if it's a file path
        if Path(source).exists():
            if Path(source).is_file():
                # Try to determine file type by content or extension
                try:
                    with open(source, 'rb') as f:
                        header = f.read(16)
                        # PCAP magic numbers
                        if header.startswith(b'\xa1\xb2\xc3\xd4') or header.startswith(b'\xd4\xc3\xb2\xa1'):
                            return 'pcap'
                        # PCAPNG magic number
                        elif header.startswith(b'\x0a\x0d\x0d\x0a'):
                            return 'pcap'
                except Exception:
                    pass
                
                # Default to CSV for text files
                return 'csv'
        
        # Assume it's a network interface for live capture
        return 'live'
    
    def get_supported_sources(self) -> Dict[str, Dict[str, Any]]:
        """Get information about supported source types"""
        return {
            'pcap': {
                'description': 'PCAP/PCAPNG packet capture files',
                'extensions': ['.pcap', '.pcapng', '.cap'],
                'example': '/path/to/capture.pcap'
            },
            'csv': {
                'description': 'CSV files with traffic data',
                'extensions': ['.csv'],
                'example': '/path/to/traffic.csv',
                'required_columns': self._ingestors['csv'].required_columns
            },
            'live': {
                'description': 'Live packet capture from network interfaces',
                'example': 'eth0, wlan0, any'
            },
            'socket': {
                'description': 'TCP socket connections receiving JSON packet data',
                'example': 'localhost:8080'
            }
        }
    
    def validate_source(self, source: str, source_type: Optional[str] = None) -> bool:
        """
        Validate that a source is accessible and properly formatted
        
        Args:
            source: Source identifier
            source_type: Type of source (auto-detected if None)
            
        Returns:
            bool: True if source is valid
            
        Raises:
            TrafficIngestionError: If source is invalid
        """
        if source_type is None:
            source_type = self._detect_source_type(source)
        
        if source_type not in self._ingestors:
            raise TrafficIngestionError(f"Unsupported source type: {source_type}")
        
        return self._ingestors[source_type].validate_source(source)
    
    def get_statistics(self, source_type: str) -> Dict[str, Any]:
        """Get ingestion statistics for a specific source type"""
        if source_type not in self._ingestors:
            raise TrafficIngestionError(f"Unknown source type: {source_type}")
        
        return self._ingestors[source_type].get_statistics()
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get ingestion statistics for all source types"""
        return {
            source_type: ingestor.get_statistics()
            for source_type, ingestor in self._ingestors.items()
        }


# Convenience functions for direct access
def ingest_pcap(file_path: str, **kwargs) -> Iterator[TrafficPacket]:
    """Convenience function to ingest from PCAP file"""
    ingestor = TrafficIngestor()
    return ingestor.ingest(file_path, 'pcap', **kwargs)


def ingest_csv(file_path: str, **kwargs) -> Iterator[TrafficPacket]:
    """Convenience function to ingest from CSV file"""
    ingestor = TrafficIngestor()
    return ingestor.ingest(file_path, 'csv', **kwargs)


def ingest_live(interface: str, **kwargs) -> Iterator[TrafficPacket]:
    """Convenience function to ingest from live interface"""
    ingestor = TrafficIngestor()
    return ingestor.ingest(interface, 'live', **kwargs)


def ingest_socket(address: str, **kwargs) -> Iterator[TrafficPacket]:
    """Convenience function to ingest from socket"""
    ingestor = TrafficIngestor()
    return ingestor.ingest(address, 'socket', **kwargs)