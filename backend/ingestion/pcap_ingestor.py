"""
PCAP file traffic ingestor using scapy
"""
import os
from typing import Iterator, Dict, Any
from datetime import datetime
import math

try:
    from scapy.all import rdpcap, IP, TCP, UDP, ICMP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_ingestor import BaseTrafficIngestor
from models.data_models import TrafficPacket, ProtocolType
from core.exceptions import TrafficIngestionError


class PcapIngestor(BaseTrafficIngestor):
    """Ingest traffic from PCAP files using scapy"""
    
    def __init__(self):
        super().__init__("PCAP File")
        if not SCAPY_AVAILABLE:
            raise TrafficIngestionError("Scapy is not available. Install with: pip install scapy")
    
    def validate_source(self, source: str) -> bool:
        """Validate PCAP file exists and is readable"""
        if not os.path.exists(source):
            raise TrafficIngestionError(f"PCAP file not found: {source}")
        
        if not os.path.isfile(source):
            raise TrafficIngestionError(f"Source is not a file: {source}")
        
        if not source.lower().endswith(('.pcap', '.pcapng', '.cap')):
            self.logger.warning(f"File extension may not be a PCAP file: {source}")
        
        try:
            # Try to read first packet to validate format
            packets = rdpcap(source, count=1)
            if len(packets) == 0:
                raise TrafficIngestionError(f"PCAP file is empty: {source}")
        except Exception as e:
            raise TrafficIngestionError(f"Cannot read PCAP file {source}: {e}")
        
        return True
    
    def ingest(self, source: str, **kwargs) -> Iterator[TrafficPacket]:
        """
        Ingest packets from PCAP file
        
        Args:
            source: Path to PCAP file
            **kwargs: Additional parameters
                - max_packets: Maximum number of packets to read
                - filters: Packet filtering criteria
                - skip_malformed: Skip malformed packets instead of raising error
        """
        self.validate_source(source)
        self.start_ingestion()
        
        max_packets = kwargs.get('max_packets', None)
        filters = kwargs.get('filters', {})
        skip_malformed = kwargs.get('skip_malformed', True)
        
        try:
            # Read packets in chunks to handle large files
            chunk_size = 1000
            packet_count = 0
            
            while max_packets is None or packet_count < max_packets:
                try:
                    # Read chunk of packets
                    remaining = max_packets - packet_count if max_packets else chunk_size
                    chunk_size_actual = min(chunk_size, remaining) if max_packets else chunk_size
                    
                    packets = rdpcap(source, count=chunk_size_actual, skip=packet_count)
                    
                    if len(packets) == 0:
                        break  # End of file
                    
                    for scapy_packet in packets:
                        try:
                            traffic_packet = self._convert_scapy_packet(scapy_packet)
                            
                            if traffic_packet and self.apply_filters(traffic_packet, filters):
                                traffic_packet = self.preprocess_packet(traffic_packet)
                                self.packets_processed += 1
                                yield traffic_packet
                                
                        except Exception as e:
                            self.errors_encountered += 1
                            if skip_malformed:
                                self.logger.warning(f"Skipping malformed packet: {e}")
                                continue
                            else:
                                raise TrafficIngestionError(f"Error processing packet: {e}")
                    
                    packet_count += len(packets)
                    
                    if len(packets) < chunk_size_actual:
                        break  # End of file
                        
                except Exception as e:
                    self.errors_encountered += 1
                    if skip_malformed:
                        self.logger.error(f"Error reading packet chunk: {e}")
                        break
                    else:
                        raise TrafficIngestionError(f"Error reading PCAP file: {e}")
        
        finally:
            self.end_ingestion()
    
    def _convert_scapy_packet(self, scapy_packet) -> TrafficPacket:
        """Convert scapy packet to TrafficPacket"""
        try:
            # Extract timestamp
            timestamp = datetime.fromtimestamp(float(scapy_packet.time))
            
            # Check if packet has IP layer
            if not scapy_packet.haslayer(IP):
                return None  # Skip non-IP packets
            
            ip_layer = scapy_packet[IP]
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            ttl = ip_layer.ttl
            packet_size = len(scapy_packet)
            
            # Determine protocol and extract port information
            protocol = ProtocolType.TCP  # Default
            src_port = 0
            dst_port = 0
            flags = []
            
            if scapy_packet.haslayer(TCP):
                tcp_layer = scapy_packet[TCP]
                protocol = ProtocolType.TCP
                src_port = tcp_layer.sport
                dst_port = tcp_layer.dport
                
                # Extract TCP flags
                if tcp_layer.flags:
                    flag_map = {
                        0x01: 'FIN', 0x02: 'SYN', 0x04: 'RST', 0x08: 'PSH',
                        0x10: 'ACK', 0x20: 'URG', 0x40: 'ECE', 0x80: 'CWR'
                    }
                    for flag_bit, flag_name in flag_map.items():
                        if tcp_layer.flags & flag_bit:
                            flags.append(flag_name)
                
                # Detect HTTP/HTTPS based on common ports
                if dst_port == 80 or src_port == 80:
                    protocol = ProtocolType.HTTP
                elif dst_port == 443 or src_port == 443:
                    protocol = ProtocolType.HTTPS
                    
            elif scapy_packet.haslayer(UDP):
                udp_layer = scapy_packet[UDP]
                protocol = ProtocolType.UDP
                src_port = udp_layer.sport
                dst_port = udp_layer.dport
                
            elif scapy_packet.haslayer(ICMP):
                protocol = ProtocolType.ICMP
                # ICMP doesn't have ports, use type and code
                icmp_layer = scapy_packet[ICMP]
                src_port = icmp_layer.type
                dst_port = icmp_layer.code
            
            # Calculate payload entropy (simplified)
            payload_entropy = self._calculate_entropy(scapy_packet)
            
            return TrafficPacket(
                timestamp=timestamp,
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=protocol,
                packet_size=packet_size,
                ttl=ttl,
                flags=flags,
                payload_entropy=payload_entropy,
                packet_id=None  # Will be set in preprocessing
            )
            
        except Exception as e:
            raise TrafficIngestionError(f"Error converting scapy packet: {e}")
    
    def _calculate_entropy(self, packet) -> float:
        """Calculate Shannon entropy of packet payload"""
        try:
            # Get raw bytes of the packet
            raw_bytes = bytes(packet)
            
            if len(raw_bytes) == 0:
                return 0.0
            
            # Count byte frequencies
            byte_counts = {}
            for byte in raw_bytes:
                byte_counts[byte] = byte_counts.get(byte, 0) + 1
            
            # Calculate entropy
            entropy = 0.0
            total_bytes = len(raw_bytes)
            
            for count in byte_counts.values():
                probability = count / total_bytes
                if probability > 0:
                    entropy -= probability * math.log2(probability)
            
            # Normalize to 0-1 range (max entropy for 8-bit data is 8)
            return min(entropy / 8.0, 1.0)
            
        except Exception:
            # Return default entropy if calculation fails
            return 0.5