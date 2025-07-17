"""
Feature extraction engine for network traffic analysis
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import math
import ipaddress
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_models import TrafficPacket, ProtocolType
from core.exceptions import FeatureExtractionError


class FeatureExtractor:
    """Extract features from network traffic for AI model input"""
    
    def __init__(self, window_size: int = 60):
        """
        Initialize feature extractor
        
        Args:
            window_size: Time window in seconds for flow-level features
        """
        self.window_size = window_size
        self.logger = logging.getLogger(__name__)
        
        # Flow tracking for temporal features
        self.flow_cache = {}
        self.ip_stats = defaultdict(lambda: {
            'packet_count': 0,
            'byte_count': 0,
            'connection_count': 0,
            'protocols': set(),
            'ports': set(),
            'first_seen': None,
            'last_seen': None
        })
        
        # Feature normalization parameters
        self.feature_stats = {
            'packet_size': {'min': 64, 'max': 1500, 'mean': 500, 'std': 300},
            'ttl': {'min': 1, 'max': 255, 'mean': 64, 'std': 32},
            'entropy': {'min': 0.0, 'max': 1.0, 'mean': 0.5, 'std': 0.2},
            'port': {'min': 0, 'max': 65535, 'mean': 32768, 'std': 18000}
        }
    
    def extract_packet_features(self, packet: TrafficPacket) -> np.ndarray:
        """
        Extract packet-level features
        
        Args:
            packet: Traffic packet to extract features from
            
        Returns:
            Feature vector as numpy array
        """
        try:
            features = []
            
            # Basic packet features
            features.extend([
                self._normalize_feature(packet.packet_size, 'packet_size'),
                self._normalize_feature(packet.ttl, 'ttl'),
                packet.payload_entropy,
                self._normalize_feature(packet.src_port, 'port'),
                self._normalize_feature(packet.dst_port, 'port')
            ])
            
            # Protocol encoding (one-hot)
            protocol_features = self._encode_protocol(packet.protocol)
            features.extend(protocol_features)
            
            # TCP flags encoding
            flag_features = self._encode_tcp_flags(packet.flags)
            features.extend(flag_features)
            
            # IP address features
            ip_features = self._extract_ip_features(packet.src_ip, packet.dst_ip)
            features.extend(ip_features)
            
            # Port category features
            port_features = self._extract_port_features(packet.src_port, packet.dst_port)
            features.extend(port_features)
            
            # Temporal features (hour of day, day of week)
            temporal_features = self._extract_temporal_features(packet.timestamp)
            features.extend(temporal_features)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            raise FeatureExtractionError(f"Error extracting packet features: {e}")
    
    def extract_flow_features(self, packets: List[TrafficPacket]) -> np.ndarray:
        """
        Extract flow-level features from a sequence of packets
        
        Args:
            packets: List of packets in the same flow
            
        Returns:
            Flow feature vector as numpy array
        """
        try:
            if not packets:
                raise FeatureExtractionError("Empty packet list provided")
            
            features = []
            
            # Basic flow statistics
            packet_count = len(packets)
            total_bytes = sum(p.packet_size for p in packets)
            duration = (packets[-1].timestamp - packets[0].timestamp).total_seconds()
            
            features.extend([
                math.log1p(packet_count),  # Log-scaled packet count
                math.log1p(total_bytes),   # Log-scaled byte count
                duration,
                total_bytes / packet_count if packet_count > 0 else 0,  # Average packet size
                packet_count / max(duration, 1),  # Packet rate
                total_bytes / max(duration, 1)    # Byte rate
            ])
            
            # Packet size statistics
            sizes = [p.packet_size for p in packets]
            features.extend([
                np.mean(sizes),
                np.std(sizes),
                np.min(sizes),
                np.max(sizes),
                np.median(sizes)
            ])
            
            # Inter-arrival time statistics
            if len(packets) > 1:
                inter_arrivals = [
                    (packets[i].timestamp - packets[i-1].timestamp).total_seconds()
                    for i in range(1, len(packets))
                ]
                features.extend([
                    np.mean(inter_arrivals),
                    np.std(inter_arrivals),
                    np.min(inter_arrivals),
                    np.max(inter_arrivals)
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            # Protocol distribution
            protocol_counts = Counter(p.protocol.value for p in packets)
            protocol_entropy = self._calculate_entropy(list(protocol_counts.values()))
            features.append(protocol_entropy)
            
            # Port diversity
            src_ports = set(p.src_port for p in packets)
            dst_ports = set(p.dst_port for p in packets)
            features.extend([
                len(src_ports),
                len(dst_ports),
                len(src_ports) / packet_count,  # Source port diversity ratio
                len(dst_ports) / packet_count   # Destination port diversity ratio
            ])
            
            # Flag statistics (for TCP flows)
            tcp_packets = [p for p in packets if p.protocol == ProtocolType.TCP]
            if tcp_packets:
                all_flags = [flag for p in tcp_packets for flag in p.flags]
                flag_counts = Counter(all_flags)
                features.extend([
                    flag_counts.get('SYN', 0) / len(tcp_packets),
                    flag_counts.get('ACK', 0) / len(tcp_packets),
                    flag_counts.get('FIN', 0) / len(tcp_packets),
                    flag_counts.get('RST', 0) / len(tcp_packets)
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            raise FeatureExtractionError(f"Error extracting flow features: {e}")
    
    def extract_network_features(self, packets: List[TrafficPacket], 
                               time_window: Optional[timedelta] = None) -> np.ndarray:
        """
        Extract network-level features from traffic within a time window
        
        Args:
            packets: List of packets to analyze
            time_window: Time window for analysis (default: self.window_size)
            
        Returns:
            Network feature vector as numpy array
        """
        try:
            if not packets:
                return np.zeros(20, dtype=np.float32)  # Return zero vector
            
            if time_window is None:
                time_window = timedelta(seconds=self.window_size)
            
            features = []
            
            # Update IP statistics
            self._update_ip_stats(packets)
            
            # Unique IP counts
            src_ips = set(p.src_ip for p in packets)
            dst_ips = set(p.dst_ip for p in packets)
            all_ips = src_ips.union(dst_ips)
            
            features.extend([
                len(src_ips),
                len(dst_ips),
                len(all_ips),
                len(src_ips.intersection(dst_ips))  # Bidirectional IPs
            ])
            
            # Connection patterns
            connections = set((p.src_ip, p.dst_ip) for p in packets)
            features.extend([
                len(connections),
                len(connections) / len(all_ips) if all_ips else 0  # Connection density
            ])
            
            # Protocol distribution
            protocol_counts = Counter(p.protocol.value for p in packets)
            protocol_entropy = self._calculate_entropy(list(protocol_counts.values()))
            features.append(protocol_entropy)
            
            # Port scanning indicators
            port_scan_features = self._detect_port_scanning(packets)
            features.extend(port_scan_features)
            
            # Traffic volume features
            total_packets = len(packets)
            total_bytes = sum(p.packet_size for p in packets)
            duration = time_window.total_seconds()
            
            features.extend([
                total_packets / duration,  # Packet rate
                total_bytes / duration,    # Byte rate
                math.log1p(total_packets),
                math.log1p(total_bytes)
            ])
            
            # Entropy features
            src_ip_entropy = self._calculate_entropy([
                sum(1 for p in packets if p.src_ip == ip) for ip in src_ips
            ])
            dst_ip_entropy = self._calculate_entropy([
                sum(1 for p in packets if p.dst_ip == ip) for ip in dst_ips
            ])
            
            features.extend([src_ip_entropy, dst_ip_entropy])
            
            # Pad or truncate to fixed size
            target_size = 20
            if len(features) < target_size:
                features.extend([0] * (target_size - len(features)))
            else:
                features = features[:target_size]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            raise FeatureExtractionError(f"Error extracting network features: {e}")
    
    def _normalize_feature(self, value: float, feature_type: str) -> float:
        """Normalize feature using z-score normalization"""
        stats = self.feature_stats.get(feature_type, {'mean': 0, 'std': 1})
        return (value - stats['mean']) / stats['std']
    
    def _encode_protocol(self, protocol: ProtocolType) -> List[float]:
        """One-hot encode protocol type"""
        protocols = [ProtocolType.TCP, ProtocolType.UDP, ProtocolType.ICMP, 
                    ProtocolType.HTTP, ProtocolType.HTTPS]
        encoding = [1.0 if protocol == p else 0.0 for p in protocols]
        return encoding
    
    def _encode_tcp_flags(self, flags: List[str]) -> List[float]:
        """Encode TCP flags as binary features"""
        flag_names = ['SYN', 'ACK', 'FIN', 'RST', 'PSH', 'URG']
        encoding = [1.0 if flag in flags else 0.0 for flag in flag_names]
        return encoding
    
    def _extract_ip_features(self, src_ip: str, dst_ip: str) -> List[float]:
        """Extract IP address-based features"""
        features = []
        
        try:
            src_addr = ipaddress.ip_address(src_ip)
            dst_addr = ipaddress.ip_address(dst_ip)
            
            # Private/public classification
            features.extend([
                1.0 if src_addr.is_private else 0.0,
                1.0 if dst_addr.is_private else 0.0,
                1.0 if src_addr.is_loopback else 0.0,
                1.0 if dst_addr.is_loopback else 0.0
            ])
            
            # Same subnet check (for IPv4)
            if isinstance(src_addr, ipaddress.IPv4Address) and isinstance(dst_addr, ipaddress.IPv4Address):
                same_subnet = (int(src_addr) >> 8) == (int(dst_addr) >> 8)  # /24 subnet
                features.append(1.0 if same_subnet else 0.0)
            else:
                features.append(0.0)
                
        except Exception:
            # Default values if IP parsing fails
            features = [0.0] * 5
        
        return features
    
    def _extract_port_features(self, src_port: int, dst_port: int) -> List[float]:
        """Extract port-based features"""
        features = []
        
        # Well-known port indicators
        well_known_ports = {80, 443, 22, 21, 25, 53, 110, 143, 993, 995}
        features.extend([
            1.0 if src_port in well_known_ports else 0.0,
            1.0 if dst_port in well_known_ports else 0.0
        ])
        
        # Port range categories
        features.extend([
            1.0 if src_port < 1024 else 0.0,  # System ports
            1.0 if dst_port < 1024 else 0.0,
            1.0 if 1024 <= src_port < 49152 else 0.0,  # User ports
            1.0 if 1024 <= dst_port < 49152 else 0.0,
            1.0 if src_port >= 49152 else 0.0,  # Dynamic ports
            1.0 if dst_port >= 49152 else 0.0
        ])
        
        return features
    
    def _extract_temporal_features(self, timestamp: datetime) -> List[float]:
        """Extract time-based features"""
        features = []
        
        # Hour of day (normalized)
        hour_norm = timestamp.hour / 23.0
        features.append(hour_norm)
        
        # Day of week (one-hot encoded, simplified to weekday/weekend)
        is_weekend = 1.0 if timestamp.weekday() >= 5 else 0.0
        features.append(is_weekend)
        
        return features
    
    def _detect_port_scanning(self, packets: List[TrafficPacket]) -> List[float]:
        """Detect port scanning patterns"""
        features = []
        
        # Group by source IP
        src_ip_ports = defaultdict(set)
        for packet in packets:
            src_ip_ports[packet.src_ip].add(packet.dst_port)
        
        # Port scan indicators
        max_ports_per_ip = max(len(ports) for ports in src_ip_ports.values()) if src_ip_ports else 0
        avg_ports_per_ip = np.mean([len(ports) for ports in src_ip_ports.values()]) if src_ip_ports else 0
        
        features.extend([
            max_ports_per_ip,
            avg_ports_per_ip,
            1.0 if max_ports_per_ip > 10 else 0.0  # Potential port scan flag
        ])
        
        return features
    
    def _update_ip_stats(self, packets: List[TrafficPacket]):
        """Update IP statistics for network-level features"""
        for packet in packets:
            for ip in [packet.src_ip, packet.dst_ip]:
                stats = self.ip_stats[ip]
                stats['packet_count'] += 1
                stats['byte_count'] += packet.packet_size
                stats['protocols'].add(packet.protocol.value)
                stats['ports'].add(packet.src_port if ip == packet.src_ip else packet.dst_port)
                
                if stats['first_seen'] is None:
                    stats['first_seen'] = packet.timestamp
                stats['last_seen'] = packet.timestamp
    
    def _calculate_entropy(self, values: List[int]) -> float:
        """Calculate Shannon entropy of a list of values"""
        if not values or sum(values) == 0:
            return 0.0
        
        total = sum(values)
        probabilities = [v / total for v in values if v > 0]
        
        entropy = -sum(p * math.log2(p) for p in probabilities)
        return entropy
    
    def get_feature_names(self) -> List[str]:
        """Get names of all extracted features"""
        names = [
            # Packet-level features
            'packet_size_norm', 'ttl_norm', 'payload_entropy', 'src_port_norm', 'dst_port_norm',
            
            # Protocol features (one-hot)
            'proto_tcp', 'proto_udp', 'proto_icmp', 'proto_http', 'proto_https',
            
            # TCP flags
            'flag_syn', 'flag_ack', 'flag_fin', 'flag_rst', 'flag_psh', 'flag_urg',
            
            # IP features
            'src_private', 'dst_private', 'src_loopback', 'dst_loopback', 'same_subnet',
            
            # Port features
            'src_well_known', 'dst_well_known', 'src_system', 'dst_system',
            'src_user', 'dst_user', 'src_dynamic', 'dst_dynamic',
            
            # Temporal features
            'hour_norm', 'is_weekend'
        ]
        
        return names
    
    def reset_cache(self):
        """Reset internal caches and statistics"""
        self.flow_cache.clear()
        self.ip_stats.clear()
        self.logger.info("Feature extractor cache reset")