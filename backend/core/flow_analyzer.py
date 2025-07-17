"""
Network Flow Analyzer for DDoS.AI platform

This module analyzes network flows to detect patterns across multiple packets
and identify potential DDoS attacks based on flow characteristics.
"""
import logging
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from datetime import datetime, timedelta
import ipaddress
import uuid

from models.data_models import TrafficPacket, NetworkFlow, ProtocolType
from core.exceptions import FlowAnalysisError


class FlowKey:
    """Key for identifying unique network flows"""
    
    def __init__(self, src_ip: str, dst_ip: str, src_port: int, dst_port: int, protocol: ProtocolType):
        """Initialize flow key"""
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol
    
    def __eq__(self, other):
        """Check equality"""
        if not isinstance(other, FlowKey):
            return False
        return (self.src_ip == other.src_ip and
                self.dst_ip == other.dst_ip and
                self.src_port == other.src_port and
                self.dst_port == other.dst_port and
                self.protocol == other.protocol)
    
    def __hash__(self):
        """Generate hash for dictionary keys"""
        return hash((self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol))
    
    def reversed(self) -> 'FlowKey':
        """Get reversed flow key (for bidirectional flows)"""
        return FlowKey(
            src_ip=self.dst_ip,
            dst_ip=self.src_ip,
            src_port=self.dst_port,
            dst_port=self.src_port,
            protocol=self.protocol
        )
    
    def __str__(self):
        """String representation"""
        return f"{self.src_ip}:{self.src_port} -> {self.dst_ip}:{self.dst_port} ({self.protocol.value})"


class FlowAnalyzer:
    """Analyzes network flows to detect patterns across multiple packets"""
    
    def __init__(self, flow_timeout: int = 60, max_flows: int = 10000):
        """
        Initialize flow analyzer
        
        Args:
            flow_timeout: Flow timeout in seconds
            max_flows: Maximum number of flows to track
        """
        self.logger = logging.getLogger(__name__)
        self.flow_timeout = flow_timeout
        self.max_flows = max_flows
        
        # Active flows
        self.active_flows: Dict[FlowKey, NetworkFlow] = {}
        
        # Completed flows
        self.completed_flows: List[NetworkFlow] = []
        self.max_completed_flows = 1000
        
        # Flow statistics
        self.flow_stats = {
            "total_flows_processed": 0,
            "active_flows": 0,
            "completed_flows": 0,
            "expired_flows": 0,
            "packets_processed": 0
        }
        
        # IP statistics
        self.ip_stats: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info(f"Flow analyzer initialized with timeout={flow_timeout}s, max_flows={max_flows}")
    
    def process_packet(self, packet: TrafficPacket) -> Optional[NetworkFlow]:
        """
        Process a packet and update flow information
        
        Args:
            packet: Traffic packet to process
            
        Returns:
            NetworkFlow if the flow was completed by this packet, None otherwise
        """
        try:
            # Update statistics
            self.flow_stats["packets_processed"] += 1
            
            # Create flow key
            flow_key = FlowKey(
                src_ip=packet.src_ip,
                dst_ip=packet.dst_ip,
                src_port=packet.src_port,
                dst_port=packet.dst_port,
                protocol=packet.protocol
            )
            
            # Check for reverse flow (for bidirectional tracking)
            reverse_key = flow_key.reversed()
            if reverse_key in self.active_flows:
                flow_key = reverse_key
            
            # Update IP statistics
            self._update_ip_stats(packet)
            
            # Check if flow exists
            if flow_key in self.active_flows:
                # Update existing flow
                flow = self.active_flows[flow_key]
                flow.packet_count += 1
                flow.byte_count += packet.packet_size
                flow.end_time = packet.timestamp
                flow.flow_duration = (flow.end_time - flow.start_time).total_seconds()
                flow.avg_packet_size = flow.byte_count / flow.packet_count
                
                # Check for flow completion (e.g., FIN flag in TCP)
                if packet.protocol == ProtocolType.TCP and "FIN" in packet.flags:
                    # Flow completed
                    completed_flow = self.active_flows.pop(flow_key)
                    self._add_to_completed_flows(completed_flow)
                    self.flow_stats["completed_flows"] += 1
                    self.flow_stats["active_flows"] = len(self.active_flows)
                    return completed_flow
                
                return flow
            else:
                # Create new flow
                flow_id = f"flow_{uuid.uuid4().hex[:8]}"
                new_flow = NetworkFlow(
                    flow_id=flow_id,
                    src_ip=packet.src_ip,
                    dst_ip=packet.dst_ip,
                    src_port=packet.src_port,
                    dst_port=packet.dst_port,
                    protocol=packet.protocol,
                    start_time=packet.timestamp,
                    end_time=packet.timestamp,
                    packet_count=1,
                    byte_count=packet.packet_size,
                    avg_packet_size=float(packet.packet_size),
                    flow_duration=0.0
                )
                
                # Add to active flows
                self.active_flows[flow_key] = new_flow
                self.flow_stats["total_flows_processed"] += 1
                self.flow_stats["active_flows"] = len(self.active_flows)
                
                # Check if we need to expire old flows
                if len(self.active_flows) > self.max_flows:
                    self._expire_oldest_flows()
                
                return new_flow
        except Exception as e:
            self.logger.error(f"Error processing packet for flow analysis: {e}")
            return None
    
    def expire_flows(self, current_time: Optional[datetime] = None) -> List[NetworkFlow]:
        """
        Expire flows that have been inactive for too long
        
        Args:
            current_time: Current time (defaults to now)
            
        Returns:
            List of expired flows
        """
        if current_time is None:
            current_time = datetime.now()
        
        expired_flows = []
        keys_to_remove = []
        
        for key, flow in self.active_flows.items():
            if (current_time - flow.end_time).total_seconds() > self.flow_timeout:
                expired_flows.append(flow)
                keys_to_remove.append(key)
        
        # Remove expired flows
        for key in keys_to_remove:
            del self.active_flows[key]
        
        # Update statistics
        self.flow_stats["expired_flows"] += len(expired_flows)
        self.flow_stats["active_flows"] = len(self.active_flows)
        
        # Add to completed flows
        for flow in expired_flows:
            self._add_to_completed_flows(flow)
        
        return expired_flows
    
    def get_active_flows(self) -> List[NetworkFlow]:
        """Get list of active flows"""
        return list(self.active_flows.values())
    
    def get_completed_flows(self, limit: int = 100) -> List[NetworkFlow]:
        """Get list of completed flows"""
        return self.completed_flows[-limit:]
    
    def get_flow_stats(self) -> Dict[str, Any]:
        """Get flow statistics"""
        return self.flow_stats.copy()
    
    def get_ip_stats(self, ip_address: str) -> Dict[str, Any]:
        """Get statistics for a specific IP address"""
        if ip_address in self.ip_stats:
            return self.ip_stats[ip_address].copy()
        return {
            "packet_count": 0,
            "byte_count": 0,
            "flow_count": 0,
            "first_seen": None,
            "last_seen": None,
            "protocols": {}
        }
    
    def get_top_talkers(self, limit: int = 10) -> List[Tuple[str, Dict[str, Any]]]:
        """Get top talkers (IPs with most traffic)"""
        # Sort IPs by byte count
        sorted_ips = sorted(
            self.ip_stats.items(),
            key=lambda x: x[1]["byte_count"],
            reverse=True
        )
        return sorted_ips[:limit]
    
    def get_flow_by_id(self, flow_id: str) -> Optional[NetworkFlow]:
        """Get flow by ID"""
        # Check active flows
        for flow in self.active_flows.values():
            if flow.flow_id == flow_id:
                return flow
        
        # Check completed flows
        for flow in self.completed_flows:
            if flow.flow_id == flow_id:
                return flow
        
        return None
    
    def get_flows_by_ip(self, ip_address: str) -> List[NetworkFlow]:
        """Get all flows involving a specific IP address"""
        flows = []
        
        # Check active flows
        for flow in self.active_flows.values():
            if flow.src_ip == ip_address or flow.dst_ip == ip_address:
                flows.append(flow)
        
        # Check completed flows
        for flow in self.completed_flows:
            if flow.src_ip == ip_address or flow.dst_ip == ip_address:
                flows.append(flow)
        
        return flows
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """
        Detect flow anomalies that might indicate DDoS attacks
        
        Returns:
            List of anomaly dictionaries
        """
        anomalies = []
        
        # Check for high packet rate flows
        for flow in self.active_flows.values():
            if flow.flow_duration > 0:
                packet_rate = flow.packet_count / flow.flow_duration
                if packet_rate > 100:  # More than 100 packets per second
                    anomalies.append({
                        "type": "high_packet_rate",
                        "flow_id": flow.flow_id,
                        "src_ip": flow.src_ip,
                        "dst_ip": flow.dst_ip,
                        "packet_rate": packet_rate,
                        "severity": "high" if packet_rate > 1000 else "medium"
                    })
        
        # Check for IP fan-out (one source to many destinations)
        src_ip_counts = {}
        for flow_key in self.active_flows:
            src_ip = flow_key.src_ip
            if src_ip not in src_ip_counts:
                src_ip_counts[src_ip] = set()
            src_ip_counts[src_ip].add(flow_key.dst_ip)
        
        for src_ip, dst_ips in src_ip_counts.items():
            if len(dst_ips) > 50:  # One source connecting to many destinations
                anomalies.append({
                    "type": "ip_fan_out",
                    "src_ip": src_ip,
                    "dst_ip_count": len(dst_ips),
                    "severity": "high" if len(dst_ips) > 100 else "medium"
                })
        
        # Check for IP fan-in (many sources to one destination)
        dst_ip_counts = {}
        for flow_key in self.active_flows:
            dst_ip = flow_key.dst_ip
            if dst_ip not in dst_ip_counts:
                dst_ip_counts[dst_ip] = set()
            dst_ip_counts[dst_ip].add(flow_key.src_ip)
        
        for dst_ip, src_ips in dst_ip_counts.items():
            if len(src_ips) > 50:  # Many sources connecting to one destination
                anomalies.append({
                    "type": "ip_fan_in",
                    "dst_ip": dst_ip,
                    "src_ip_count": len(src_ips),
                    "severity": "high" if len(src_ips) > 100 else "medium"
                })
        
        # Check for SYN flood patterns
        syn_counts = {}
        for flow_key, flow in self.active_flows.items():
            if flow.protocol == ProtocolType.TCP:
                dst_ip = flow_key.dst_ip
                if dst_ip not in syn_counts:
                    syn_counts[dst_ip] = 0
                # Estimate SYN count based on flow characteristics
                # In a real implementation, we would track TCP flags
                if flow.packet_count < 3 and flow.flow_duration > 1.0:
                    syn_counts[dst_ip] += 1
        
        for dst_ip, syn_count in syn_counts.items():
            if syn_count > 30:  # High number of potential SYN packets
                anomalies.append({
                    "type": "syn_flood",
                    "dst_ip": dst_ip,
                    "syn_count": syn_count,
                    "severity": "high" if syn_count > 100 else "medium"
                })
        
        return anomalies
    
    def clear(self):
        """Clear all flow data"""
        self.active_flows.clear()
        self.completed_flows.clear()
        self.ip_stats.clear()
        self.flow_stats = {
            "total_flows_processed": 0,
            "active_flows": 0,
            "completed_flows": 0,
            "expired_flows": 0,
            "packets_processed": 0
        }
    
    def _expire_oldest_flows(self, count: int = 100):
        """Expire oldest flows when reaching capacity"""
        if not self.active_flows:
            return
        
        # Sort flows by end time
        sorted_flows = sorted(
            self.active_flows.items(),
            key=lambda x: x[1].end_time
        )
        
        # Expire oldest flows
        to_expire = min(count, len(sorted_flows))
        expired_flows = []
        
        for i in range(to_expire):
            key, flow = sorted_flows[i]
            expired_flows.append(flow)
            del self.active_flows[key]
        
        # Update statistics
        self.flow_stats["expired_flows"] += len(expired_flows)
        self.flow_stats["active_flows"] = len(self.active_flows)
        
        # Add to completed flows
        for flow in expired_flows:
            self._add_to_completed_flows(flow)
    
    def _add_to_completed_flows(self, flow: NetworkFlow):
        """Add flow to completed flows list"""
        self.completed_flows.append(flow)
        if len(self.completed_flows) > self.max_completed_flows:
            self.completed_flows.pop(0)
    
    def _update_ip_stats(self, packet: TrafficPacket):
        """Update IP statistics"""
        # Update source IP stats
        self._update_single_ip_stats(packet.src_ip, packet, is_source=True)
        
        # Update destination IP stats
        self._update_single_ip_stats(packet.dst_ip, packet, is_source=False)
    
    def _update_single_ip_stats(self, ip: str, packet: TrafficPacket, is_source: bool):
        """Update statistics for a single IP address"""
        if ip not in self.ip_stats:
            self.ip_stats[ip] = {
                "packet_count": 0,
                "byte_count": 0,
                "flow_count": 0,
                "first_seen": packet.timestamp,
                "last_seen": packet.timestamp,
                "protocols": {},
                "src_packets": 0,
                "dst_packets": 0,
                "src_bytes": 0,
                "dst_bytes": 0
            }
        
        stats = self.ip_stats[ip]
        stats["packet_count"] += 1
        stats["byte_count"] += packet.packet_size
        stats["last_seen"] = packet.timestamp
        
        # Update protocol stats
        protocol = packet.protocol.value
        if protocol not in stats["protocols"]:
            stats["protocols"][protocol] = 0
        stats["protocols"][protocol] += 1
        
        # Update source/destination specific stats
        if is_source:
            stats["src_packets"] += 1
            stats["src_bytes"] += packet.packet_size
        else:
            stats["dst_packets"] += 1
            stats["dst_bytes"] += packet.packet_size