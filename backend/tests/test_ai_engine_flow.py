"""
Unit tests for AI Engine with Flow Analysis integration
"""
import unittest
import asyncio
from datetime import datetime, timedelta
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.ai_engine import AIEngine
from core.feature_extractor import FeatureExtractor
from core.flow_analyzer import FlowAnalyzer
from models.data_models import TrafficPacket, ProtocolType, AttackType


class MockFeatureExtractor(FeatureExtractor):
    """Mock feature extractor for testing"""
    
    def extract_packet_features(self, packet):
        """Return mock features"""
        return np.array([0.1, 0.2, 0.3, 0.4, 0.5])


class TestAIEngineFlowIntegration(unittest.TestCase):
    """Test cases for AI Engine with Flow Analysis integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.feature_extractor = MockFeatureExtractor()
        self.flow_analyzer = FlowAnalyzer(flow_timeout=10, max_flows=100)
        self.ai_engine = AIEngine(
            feature_extractor=self.feature_extractor,
            flow_analyzer=self.flow_analyzer
        )
        self.base_time = datetime.now()
    
    def test_flow_integration(self):
        """Test flow analyzer integration with AI Engine"""
        # Create a packet
        packet = self._create_packet(
            src_ip="192.168.1.100",
            dst_ip="10.0.0.1",
            src_port=12345,
            dst_port=80,
            protocol=ProtocolType.TCP,
            flags=["SYN"]
        )
        
        # Process packet through AI Engine
        result = asyncio.run(self.ai_engine.analyze_packet(packet))
        
        # Check that flow analysis was performed
        self.assertIsNotNone(result.flow_id)
        self.assertIn("flow_analysis", result.explanation)
        self.assertIsNotNone(result.explanation["flow_analysis"])
        
        # Check flow analysis details
        flow_analysis = result.explanation["flow_analysis"]
        self.assertEqual(flow_analysis["src_ip"], "192.168.1.100")
        self.assertEqual(flow_analysis["dst_ip"], "10.0.0.1")
        self.assertEqual(flow_analysis["src_port"], 12345)
        self.assertEqual(flow_analysis["dst_port"], 80)
        self.assertEqual(flow_analysis["packet_count"], 1)
    
    def test_multiple_packets_same_flow(self):
        """Test processing multiple packets in the same flow"""
        # Create first packet
        packet1 = self._create_packet(
            src_ip="192.168.1.100",
            dst_ip="10.0.0.1",
            src_port=12345,
            dst_port=80,
            protocol=ProtocolType.TCP,
            flags=["SYN"]
        )
        
        # Process first packet
        result1 = asyncio.run(self.ai_engine.analyze_packet(packet1))
        flow_id = result1.flow_id
        
        # Create second packet (same flow)
        packet2 = self._create_packet(
            src_ip="192.168.1.100",
            dst_ip="10.0.0.1",
            src_port=12345,
            dst_port=80,
            protocol=ProtocolType.TCP,
            flags=["ACK"],
            timestamp=self.base_time + timedelta(seconds=1)
        )
        
        # Process second packet
        result2 = asyncio.run(self.ai_engine.analyze_packet(packet2))
        
        # Check that both packets are in the same flow
        self.assertEqual(result2.flow_id, flow_id)
        
        # Check flow analysis details
        flow_analysis = result2.explanation["flow_analysis"]
        self.assertEqual(flow_analysis["packet_count"], 2)
        self.assertAlmostEqual(flow_analysis["flow_duration"], 1.0, delta=0.1)
    
    def test_get_active_flows(self):
        """Test getting active flows from AI Engine"""
        # Create and process packets for different flows
        packet1 = self._create_packet(
            src_ip="192.168.1.100",
            dst_ip="10.0.0.1",
            src_port=12345,
            dst_port=80,
            protocol=ProtocolType.TCP
        )
        asyncio.run(self.ai_engine.analyze_packet(packet1))
        
        packet2 = self._create_packet(
            src_ip="192.168.1.101",
            dst_ip="10.0.0.2",
            src_port=12346,
            dst_port=443,
            protocol=ProtocolType.TCP
        )
        asyncio.run(self.ai_engine.analyze_packet(packet2))
        
        # Get active flows
        active_flows = self.ai_engine.get_active_flows()
        
        # Check active flows
        self.assertEqual(len(active_flows), 2)
        
        # Check that flows have correct IPs
        ips = set()
        for flow in active_flows:
            ips.add(flow["src_ip"])
        
        self.assertIn("192.168.1.100", ips)
        self.assertIn("192.168.1.101", ips)
    
    def test_get_top_talkers(self):
        """Test getting top talkers from AI Engine"""
        # Create and process packets with different sizes
        packet1 = self._create_packet(
            src_ip="192.168.1.100",
            dst_ip="10.0.0.1",
            src_port=12345,
            dst_port=80,
            protocol=ProtocolType.TCP,
            packet_size=1000
        )
        asyncio.run(self.ai_engine.analyze_packet(packet1))
        
        packet2 = self._create_packet(
            src_ip="192.168.1.101",
            dst_ip="10.0.0.2",
            src_port=12346,
            dst_port=443,
            protocol=ProtocolType.TCP,
            packet_size=500
        )
        asyncio.run(self.ai_engine.analyze_packet(packet2))
        
        packet3 = self._create_packet(
            src_ip="192.168.1.100",
            dst_ip="10.0.0.3",
            src_port=12347,
            dst_port=8080,
            protocol=ProtocolType.TCP,
            packet_size=1500
        )
        asyncio.run(self.ai_engine.analyze_packet(packet3))
        
        # Get top talkers
        top_talkers = self.ai_engine.get_top_talkers(limit=2)
        
        # Check top talkers
        self.assertEqual(len(top_talkers), 2)
        self.assertEqual(top_talkers[0]["ip"], "192.168.1.100")  # IP with most traffic
        self.assertEqual(top_talkers[0]["stats"]["byte_count"], 2500)  # 1000 + 1500
    
    def test_flow_anomalies(self):
        """Test detecting flow anomalies"""
        # Create a high packet rate flow
        for i in range(101):  # 101 packets
            packet = self._create_packet(
                src_ip="192.168.1.100",
                dst_ip="10.0.0.1",
                src_port=12345,
                dst_port=80,
                protocol=ProtocolType.TCP,
                timestamp=self.base_time + timedelta(milliseconds=i*10)  # 10ms between packets
            )
            asyncio.run(self.ai_engine.analyze_packet(packet))
        
        # Get flow anomalies
        anomalies = self.ai_engine.get_flow_anomalies()
        
        # Check anomalies
        self.assertGreaterEqual(len(anomalies), 1)
        
        # Check for high packet rate anomaly
        high_rate_anomalies = [a for a in anomalies if a["type"] == "high_packet_rate"]
        self.assertGreaterEqual(len(high_rate_anomalies), 1)
        self.assertEqual(high_rate_anomalies[0]["src_ip"], "192.168.1.100")
        self.assertEqual(high_rate_anomalies[0]["dst_ip"], "10.0.0.1")
    
    def test_batch_processing(self):
        """Test batch processing of packets"""
        # Create multiple packets
        packets = [
            self._create_packet(
                src_ip=f"192.168.1.{100+i}",
                dst_ip=f"10.0.0.{1+i}",
                src_port=12345 + i,
                dst_port=80,
                protocol=ProtocolType.TCP
            )
            for i in range(5)
        ]
        
        # Process packets in batch
        results = asyncio.run(self.ai_engine.analyze_packets_batch(packets))
        
        # Check results
        self.assertEqual(len(results), 5)
        
        # Check that each packet has a flow ID
        for result in results:
            self.assertIsNotNone(result.flow_id)
            self.assertIn("flow_analysis", result.explanation)
            self.assertIsNotNone(result.explanation["flow_analysis"])
    
    def _create_packet(self, src_ip, dst_ip, src_port, dst_port, protocol,
                      flags=None, packet_size=64, timestamp=None):
        """Helper method to create a packet"""
        if timestamp is None:
            timestamp = self.base_time
        if flags is None:
            flags = []
        
        return TrafficPacket(
            timestamp=timestamp,
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=protocol,
            packet_size=packet_size,
            ttl=64,
            flags=flags,
            payload_entropy=0.5,
            packet_id=f"pkt_{src_ip}_{dst_ip}_{src_port}_{dst_port}"
        )


if __name__ == "__main__":
    unittest.main()