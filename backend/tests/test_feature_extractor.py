"""
Unit tests for feature extraction engine
"""
import unittest
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.feature_extractor import FeatureExtractor
from models.data_models import TrafficPacket, ProtocolType
from core.exceptions import FeatureExtractionError


class TestFeatureExtractor(unittest.TestCase):
    """Test cases for FeatureExtractor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.extractor = FeatureExtractor(window_size=60)
        
        # Create sample packets for testing
        self.sample_packets = [
            TrafficPacket(
                timestamp=datetime.now(),
                src_ip="192.168.1.100",
                dst_ip="10.0.0.1",
                src_port=12345,
                dst_port=80,
                protocol=ProtocolType.TCP,
                packet_size=1024,
                ttl=64,
                flags=["SYN"],
                payload_entropy=0.7,
                packet_id="pkt_001"
            ),
            TrafficPacket(
                timestamp=datetime.now() + timedelta(seconds=1),
                src_ip="192.168.1.100",
                dst_ip="10.0.0.1",
                src_port=12345,
                dst_port=80,
                protocol=ProtocolType.TCP,
                packet_size=512,
                ttl=64,
                flags=["ACK"],
                payload_entropy=0.5,
                packet_id="pkt_002"
            ),
            TrafficPacket(
                timestamp=datetime.now() + timedelta(seconds=2),
                src_ip="203.0.113.45",
                dst_ip="192.168.1.100",
                src_port=53,
                dst_port=54321,
                protocol=ProtocolType.UDP,
                packet_size=256,
                ttl=32,
                flags=[],
                payload_entropy=0.3,
                packet_id="pkt_003"
            )
        ]
    
    def test_extract_packet_features(self):
        """Test packet-level feature extraction"""
        packet = self.sample_packets[0]
        features = self.extractor.extract_packet_features(packet)
        
        # Check feature vector properties
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.dtype, np.float32)
        self.assertGreater(len(features), 0)
        
        # Check that features are numeric and finite
        self.assertTrue(np.all(np.isfinite(features)))
        
        # Check specific feature values
        self.assertEqual(features[2], 0.7)  # payload_entropy should be preserved
    
    def test_extract_flow_features(self):
        """Test flow-level feature extraction"""
        # Use first two packets (same flow)
        flow_packets = self.sample_packets[:2]
        features = self.extractor.extract_flow_features(flow_packets)
        
        # Check feature vector properties
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.dtype, np.float32)
        self.assertGreater(len(features), 0)
        
        # Check that features are numeric and finite
        self.assertTrue(np.all(np.isfinite(features)))
        
        # Check packet count feature (log-scaled)
        expected_packet_count = np.log1p(2)  # log(1 + 2)
        self.assertAlmostEqual(features[0], expected_packet_count, places=5)
    
    def test_extract_network_features(self):
        """Test network-level feature extraction"""
        features = self.extractor.extract_network_features(self.sample_packets)
        
        # Check feature vector properties
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(len(features), 20)  # Fixed size
        
        # Check that features are numeric and finite
        self.assertTrue(np.all(np.isfinite(features)))
        
        # Check unique IP counts
        self.assertEqual(features[0], 2)  # 2 unique source IPs
        self.assertEqual(features[1], 2)  # 2 unique destination IPs
    
    def test_protocol_encoding(self):
        """Test protocol one-hot encoding"""
        tcp_encoding = self.extractor._encode_protocol(ProtocolType.TCP)
        udp_encoding = self.extractor._encode_protocol(ProtocolType.UDP)
        
        # Check encoding length
        self.assertEqual(len(tcp_encoding), 5)
        self.assertEqual(len(udp_encoding), 5)
        
        # Check one-hot property
        self.assertEqual(sum(tcp_encoding), 1.0)
        self.assertEqual(sum(udp_encoding), 1.0)
        
        # Check specific encodings
        self.assertEqual(tcp_encoding[0], 1.0)  # TCP is first
        self.assertEqual(udp_encoding[1], 1.0)  # UDP is second
    
    def test_tcp_flags_encoding(self):
        """Test TCP flags encoding"""
        flags = ["SYN", "ACK"]
        encoding = self.extractor._encode_tcp_flags(flags)
        
        # Check encoding length
        self.assertEqual(len(encoding), 6)
        
        # Check specific flags
        self.assertEqual(encoding[0], 1.0)  # SYN
        self.assertEqual(encoding[1], 1.0)  # ACK
        self.assertEqual(encoding[2], 0.0)  # FIN (not present)
    
    def test_ip_features_extraction(self):
        """Test IP address feature extraction"""
        # Test private IP addresses
        features = self.extractor._extract_ip_features("192.168.1.1", "10.0.0.1")
        
        self.assertEqual(len(features), 5)
        self.assertEqual(features[0], 1.0)  # Source is private
        self.assertEqual(features[1], 1.0)  # Destination is private
        
        # Test public IP addresses
        features = self.extractor._extract_ip_features("8.8.8.8", "1.1.1.1")
        self.assertEqual(features[0], 0.0)  # Source is public
        self.assertEqual(features[1], 0.0)  # Destination is public
    
    def test_port_features_extraction(self):
        """Test port-based feature extraction"""
        # Test well-known ports
        features = self.extractor._extract_port_features(12345, 80)
        
        self.assertEqual(len(features), 8)
        self.assertEqual(features[0], 0.0)  # Source not well-known
        self.assertEqual(features[1], 1.0)  # Destination is well-known (80)
        self.assertEqual(features[3], 1.0)  # Destination is system port (<1024)
    
    def test_temporal_features_extraction(self):
        """Test temporal feature extraction"""
        # Test specific time
        test_time = datetime(2024, 1, 15, 14, 30, 0)  # Monday, 2:30 PM
        features = self.extractor._extract_temporal_features(test_time)
        
        self.assertEqual(len(features), 2)
        self.assertAlmostEqual(features[0], 14/23, places=5)  # Hour normalized
        self.assertEqual(features[1], 0.0)  # Monday is not weekend
        
        # Test weekend
        weekend_time = datetime(2024, 1, 13, 10, 0, 0)  # Saturday
        features = self.extractor._extract_temporal_features(weekend_time)
        self.assertEqual(features[1], 1.0)  # Saturday is weekend
    
    def test_entropy_calculation(self):
        """Test Shannon entropy calculation"""
        # Test uniform distribution
        uniform_values = [1, 1, 1, 1]
        entropy = self.extractor._calculate_entropy(uniform_values)
        expected_entropy = 2.0  # log2(4)
        self.assertAlmostEqual(entropy, expected_entropy, places=5)
        
        # Test single value (no entropy)
        single_value = [4, 0, 0, 0]
        entropy = self.extractor._calculate_entropy(single_value)
        self.assertEqual(entropy, 0.0)
        
        # Test empty list
        empty_values = []
        entropy = self.extractor._calculate_entropy(empty_values)
        self.assertEqual(entropy, 0.0)
    
    def test_port_scanning_detection(self):
        """Test port scanning pattern detection"""
        # Create packets simulating port scan
        scan_packets = []
        base_time = datetime.now()
        
        for i, port in enumerate([80, 443, 22, 21, 25, 53, 110, 143, 993, 995, 8080]):
            packet = TrafficPacket(
                timestamp=base_time + timedelta(milliseconds=i*100),
                src_ip="192.168.1.100",
                dst_ip="10.0.0.1",
                src_port=12345 + i,
                dst_port=port,
                protocol=ProtocolType.TCP,
                packet_size=64,
                ttl=64,
                flags=["SYN"],
                payload_entropy=0.1,
                packet_id=f"scan_{i:03d}"
            )
            scan_packets.append(packet)
        
        features = self.extractor._detect_port_scanning(scan_packets)
        
        self.assertEqual(len(features), 3)
        self.assertEqual(features[0], 11)  # Max ports per IP
        self.assertEqual(features[1], 11)  # Average ports per IP
        self.assertEqual(features[2], 1.0)  # Port scan flag (>10 ports)
    
    def test_feature_normalization(self):
        """Test feature normalization"""
        # Test packet size normalization
        normalized = self.extractor._normalize_feature(1000, 'packet_size')
        
        # Should be z-score normalized
        stats = self.extractor.feature_stats['packet_size']
        expected = (1000 - stats['mean']) / stats['std']
        self.assertAlmostEqual(normalized, expected, places=5)
    
    def test_empty_packet_list_handling(self):
        """Test handling of empty packet lists"""
        # Flow features with empty list should raise error
        with self.assertRaises(FeatureExtractionError):
            self.extractor.extract_flow_features([])
        
        # Network features with empty list should return zero vector
        features = self.extractor.extract_network_features([])
        self.assertTrue(np.all(features == 0))
        self.assertEqual(len(features), 20)
    
    def test_malformed_packet_handling(self):
        """Test handling of edge case packets"""
        # Create packet with unusual but valid values
        edge_case_packet = TrafficPacket(
            timestamp=datetime.now(),
            src_ip="127.0.0.1",  # Valid loopback IP
            dst_ip="10.0.0.1",
            src_port=0,  # Edge case: port 0
            dst_port=65535,  # Edge case: max port
            protocol=ProtocolType.ICMP,
            packet_size=1,  # Minimum packet size
            ttl=1,  # Minimum TTL
            flags=[],  # No flags
            payload_entropy=0.0,  # Minimum entropy
            packet_id="edge_case"
        )
        
        # Should handle gracefully and return features
        features = self.extractor.extract_packet_features(edge_case_packet)
        self.assertIsInstance(features, np.ndarray)
        self.assertTrue(np.all(np.isfinite(features)))
    
    def test_feature_names(self):
        """Test feature name retrieval"""
        names = self.extractor.get_feature_names()
        
        self.assertIsInstance(names, list)
        self.assertGreater(len(names), 0)
        self.assertTrue(all(isinstance(name, str) for name in names))
        
        # Check for expected feature names
        expected_names = ['packet_size_norm', 'ttl_norm', 'payload_entropy', 
                         'proto_tcp', 'flag_syn', 'src_private']
        for name in expected_names:
            self.assertIn(name, names)
    
    def test_cache_reset(self):
        """Test cache reset functionality"""
        # Add some data to caches
        self.extractor.extract_network_features(self.sample_packets)
        
        # Verify caches have data
        self.assertGreater(len(self.extractor.ip_stats), 0)
        
        # Reset and verify caches are empty
        self.extractor.reset_cache()
        self.assertEqual(len(self.extractor.flow_cache), 0)
        self.assertEqual(len(self.extractor.ip_stats), 0)
    
    def test_feature_consistency(self):
        """Test that feature extraction is consistent"""
        packet = self.sample_packets[0]
        
        # Extract features multiple times
        features1 = self.extractor.extract_packet_features(packet)
        features2 = self.extractor.extract_packet_features(packet)
        
        # Should be identical
        np.testing.assert_array_equal(features1, features2)
    
    def test_feature_vector_sizes(self):
        """Test that feature vectors have expected sizes"""
        packet = self.sample_packets[0]
        
        # Packet features
        packet_features = self.extractor.extract_packet_features(packet)
        expected_packet_size = (
            5 +  # Basic features
            5 +  # Protocol encoding
            6 +  # TCP flags
            5 +  # IP features
            8 +  # Port features
            2    # Temporal features
        )
        self.assertEqual(len(packet_features), expected_packet_size)
        
        # Flow features
        flow_features = self.extractor.extract_flow_features([packet])
        self.assertGreater(len(flow_features), 0)
        
        # Network features (fixed size)
        network_features = self.extractor.extract_network_features([packet])
        self.assertEqual(len(network_features), 20)


if __name__ == '__main__':
    unittest.main()