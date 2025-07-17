"""
Live traffic ingestor for real-time packet capture
"""
import socket
import threading
import time
from typing import Iterator, Dict, Any, Optional
from datetime import datetime
from queue import Queue, Empty

try:
    from scapy.all import sniff, AsyncSniffer
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_ingestor import BaseTrafficIngestor
from .pcap_ingestor import PcapIngestor
from models.data_models import TrafficPacket
from core.exceptions import TrafficIngestionError


class LiveIngestor(BaseTrafficIngestor):
    """Ingest live traffic from network interfaces"""
    
    def __init__(self):
        super().__init__("Live Interface")
        if not SCAPY_AVAILABLE:
            raise TrafficIngestionError("Scapy is not available. Install with: pip install scapy")
        
        self.pcap_converter = PcapIngestor()
        self.packet_queue = Queue(maxsize=10000)
        self.sniffer: Optional[AsyncSniffer] = None
        self.is_capturing = False
    
    def validate_source(self, source: str) -> bool:
        """Validate network interface exists and is accessible"""
        try:
            # Try to get available interfaces
            from scapy.all import get_if_list
            available_interfaces = get_if_list()
            
            if source not in available_interfaces:
                self.logger.warning(
                    f"Interface '{source}' not found in available interfaces: {available_interfaces}"
                )
                # Don't raise error as interface names can vary by system
            
            return True
            
        except Exception as e:
            raise TrafficIngestionError(f"Cannot validate interface {source}: {e}")
    
    def ingest(self, source: str, **kwargs) -> Iterator[TrafficPacket]:
        """
        Ingest live packets from network interface
        
        Args:
            source: Network interface name (e.g., 'eth0', 'wlan0')
            **kwargs: Additional parameters
                - duration: Maximum capture duration in seconds
                - max_packets: Maximum number of packets to capture
                - filters: Packet filtering criteria
                - bpf_filter: Berkeley Packet Filter string
                - promisc: Enable promiscuous mode
        """
        self.validate_source(source)
        self.start_ingestion()
        
        duration = kwargs.get('duration', None)
        max_packets = kwargs.get('max_packets', None)
        filters = kwargs.get('filters', {})
        bpf_filter = kwargs.get('bpf_filter', None)
        promisc = kwargs.get('promisc', True)
        
        try:
            # Start packet capture in background thread
            self._start_capture(source, bpf_filter, promisc)
            
            start_time = time.time()
            
            while self.is_capturing:
                # Check duration limit
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Check packet count limit
                if max_packets and self.packets_processed >= max_packets:
                    break
                
                try:
                    # Get packet from queue with timeout
                    scapy_packet = self.packet_queue.get(timeout=1.0)
                    
                    try:
                        traffic_packet = self.pcap_converter._convert_scapy_packet(scapy_packet)
                        
                        if traffic_packet and self.apply_filters(traffic_packet, filters):
                            traffic_packet = self.preprocess_packet(traffic_packet)
                            self.packets_processed += 1
                            yield traffic_packet
                            
                    except Exception as e:
                        self.errors_encountered += 1
                        self.logger.warning(f"Error processing live packet: {e}")
                        continue
                
                except Empty:
                    # No packets in queue, continue
                    continue
                except Exception as e:
                    self.errors_encountered += 1
                    self.logger.error(f"Error getting packet from queue: {e}")
                    continue
        
        finally:
            self._stop_capture()
            self.end_ingestion()
    
    def _start_capture(self, interface: str, bpf_filter: Optional[str], promisc: bool):
        """Start background packet capture"""
        try:
            self.is_capturing = True
            
            # Create async sniffer
            self.sniffer = AsyncSniffer(
                iface=interface,
                filter=bpf_filter,
                prn=self._packet_handler,
                store=False,  # Don't store packets in memory
                promisc=promisc
            )
            
            self.sniffer.start()
            self.logger.info(f"Started live capture on interface: {interface}")
            
        except Exception as e:
            self.is_capturing = False
            raise TrafficIngestionError(f"Failed to start packet capture: {e}")
    
    def _stop_capture(self):
        """Stop background packet capture"""
        try:
            self.is_capturing = False
            
            if self.sniffer:
                self.sniffer.stop()
                self.sniffer = None
            
            self.logger.info("Stopped live packet capture")
            
        except Exception as e:
            self.logger.error(f"Error stopping packet capture: {e}")
    
    def _packet_handler(self, packet):
        """Handle captured packets"""
        try:
            if not self.is_capturing:
                return
            
            # Add packet to queue for processing
            if not self.packet_queue.full():
                self.packet_queue.put(packet, block=False)
            else:
                # Queue is full, drop packet
                self.logger.warning("Packet queue full, dropping packet")
                
        except Exception as e:
            self.logger.error(f"Error in packet handler: {e}")


class SocketIngestor(BaseTrafficIngestor):
    """Ingest traffic from socket connections"""
    
    def __init__(self):
        super().__init__("Socket Stream")
        self.socket: Optional[socket.socket] = None
    
    def validate_source(self, source: str) -> bool:
        """Validate socket connection string format"""
        try:
            if ':' not in source:
                raise TrafficIngestionError("Socket source must be in format 'host:port'")
            
            host, port_str = source.rsplit(':', 1)
            port = int(port_str)
            
            if not (1 <= port <= 65535):
                raise TrafficIngestionError(f"Invalid port number: {port}")
            
            return True
            
        except ValueError as e:
            raise TrafficIngestionError(f"Invalid socket source format: {e}")
    
    def ingest(self, source: str, **kwargs) -> Iterator[TrafficPacket]:
        """
        Ingest packets from socket connection
        
        Args:
            source: Socket connection string in format 'host:port'
            **kwargs: Additional parameters
                - timeout: Socket timeout in seconds
                - buffer_size: Socket buffer size
                - max_packets: Maximum number of packets to receive
        """
        self.validate_source(source)
        self.start_ingestion()
        
        host, port_str = source.rsplit(':', 1)
        port = int(port_str)
        timeout = kwargs.get('timeout', 30.0)
        buffer_size = kwargs.get('buffer_size', 4096)
        max_packets = kwargs.get('max_packets', None)
        
        try:
            # Create and connect socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(timeout)
            self.socket.connect((host, port))
            
            self.logger.info(f"Connected to socket: {host}:{port}")
            
            while True:
                if max_packets and self.packets_processed >= max_packets:
                    break
                
                try:
                    # Receive data from socket
                    data = self.socket.recv(buffer_size)
                    
                    if not data:
                        break  # Connection closed
                    
                    # Parse received data as JSON packet information
                    try:
                        import json
                        packet_data = json.loads(data.decode('utf-8'))
                        traffic_packet = self._convert_socket_data(packet_data)
                        
                        if traffic_packet:
                            traffic_packet = self.preprocess_packet(traffic_packet)
                            self.packets_processed += 1
                            yield traffic_packet
                            
                    except json.JSONDecodeError:
                        self.logger.warning("Received non-JSON data from socket")
                        continue
                    except Exception as e:
                        self.errors_encountered += 1
                        self.logger.warning(f"Error processing socket data: {e}")
                        continue
                
                except socket.timeout:
                    self.logger.warning("Socket timeout, continuing...")
                    continue
                except Exception as e:
                    self.errors_encountered += 1
                    self.logger.error(f"Socket error: {e}")
                    break
        
        finally:
            if self.socket:
                self.socket.close()
                self.socket = None
            self.end_ingestion()
    
    def _convert_socket_data(self, data: Dict[str, Any]) -> Optional[TrafficPacket]:
        """Convert socket data to TrafficPacket"""
        try:
            # Expected JSON format with packet information
            required_fields = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']
            
            if not all(field in data for field in required_fields):
                self.logger.warning(f"Socket data missing required fields: {required_fields}")
                return None
            
            from models.data_models import ProtocolType
            
            return TrafficPacket(
                timestamp=datetime.now(),  # Use current time for live data
                src_ip=str(data['src_ip']),
                dst_ip=str(data['dst_ip']),
                src_port=int(data['src_port']),
                dst_port=int(data['dst_port']),
                protocol=ProtocolType(data['protocol'].upper()),
                packet_size=int(data.get('packet_size', 1500)),
                ttl=int(data.get('ttl', 64)),
                flags=data.get('flags', []),
                payload_entropy=float(data.get('payload_entropy', 0.5)),
                packet_id=data.get('packet_id')
            )
            
        except Exception as e:
            raise TrafficIngestionError(f"Error converting socket data: {e}")