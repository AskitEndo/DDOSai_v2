"""
Unit tests for AI Engine Orchestrator
"""
import unittest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.ai_engine import AIEngine
from models.data_models import TrafficPacket, DetectionResult, ProtocolType, AttackType
from core.feature_extractor import FeatureExtractor
from ai.autoencoder_detector import AutoencoderDetector
from ai.gnn_analyzer import GNNAnalyzer
from ai.rl_threat_scorer import RLThreatScorer
from ai.xai_explainer import XAIExplainer


class TestAIEngine(unittest.TestCase):
    """Test cases for AIEngine class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock components
        self.feature_extractor = MagicMock(spec=FeatureExtractor)
        self.autoencoder_detector = MagicMock(spec=AutoencoderDetector)
        self.gnn_analyzer = MagicMock(spec=GNNAnalyzer)
        self.rl_threat_scorer = MagicMock(spec=RLThreatScorer)
        self.xai_explainer = MagicMock(spec=XAIExplainer)
        
        # Configure mock behavior
        self.feature_extractor.extract_packet_features.return_value = np.random.random(31)
        self.autoencoder_detector.predict.return_value = (False, 0.8, {"reconstruction_error": 0.05})
        self.gnn_analyzer.predict.return_value = (False, 0.7, {"malicious_probability": 0.3})
        self.rl_threat_scorer.predict.return_value = (False, 0.9, {"threat_score": 30})
        
        # Initialize AI Engine
        self.ai_engine = AIEngine(
            feature_extractor=self.feature_extractor,
            autoencoder_detector=self.autoencoder_detector,
            gnn_analyzer=self.gnn_analyzer,
            rl_threat_scorer=self.rl_threat_scorer,
            xai_explainer=self.xai_explainer
        )
        
        # Create sample packet
        self.sample_packet = TrafficPacket(
            timestamp=datetime.now(),
            src_ip="192.168.1.100",
            dst_ip="10.0.0.1",
            src_port=12345,
            dst_port=80,
            protocol=ProtocolType.TCP,
            packet_size=1024,
            ttl=64,
            flags=["ACK"],
            payload_entropy=0.7,
            packet_id="test_packet_001"
        )
    
    def test_initialization(self):
        """Test AI Engine initialization"""
        self.assertEqual(self.ai_engine.feature_extractor, self.feature_extractor)
        self.assertEqual(self.ai_engine.autoencoder_detector, self.autoencoder_detector)
        self.assertEqual(self.ai_engine.gnn_analyzer, self.gnn_analyzer)
        self.assertEqual(self.ai_engine.rl_threat_scorer, self.rl_threat_scorer)
        self.assertEqual(self.ai_engine.xai_explainer, self.xai_explainer)
        self.assertEqual(len(self.ai_engine.detection_history), 0)
        self.assertEqual(len(self.ai_engine.packet_buffer), 0)
    
    def test_analyze_packet(self):
        """Test packet analysis"""
        # Run analysis
        result = asyncio.run(self.ai_engine.analyze_packet(self.sample_packet))
        
        # Check result
        self.assertIsInstance(result, DetectionResult)
        self.assertEqual(result.packet_id, "test_packet_001")
        self.assertFalse(result.is_malicious)  # All mocks return benign
        self.assertEqual(result.attack_type, AttackType.BENIGN)
        
        # Check that components were called
        self.feature_extractor.extract_packet_features.assert_called_once_with(self.sample_packet)
        self.autoencoder_detector.predict.assert_called_once()
        self.gnn_analyzer.predict.assert_called_once()
        self.rl_threat_scorer.predict.assert_called_once()
        
        # Check that result was added to history
        self.assertEqual(len(self.ai_engine.detection_history), 1)
        self.assertEqual(self.ai_engine.detection_history[0], result)
        
        # Check that packet was added to buffer
        self.assertEqual(len(self.ai_engine.packet_buffer), 1)
        self.assertEqual(self.ai_engine.packet_buffer[0], self.sample_packet)
    
    def test_analyze_malicious_packet(self):
        """Test analysis of malicious packet"""
        # Configure mocks to return malicious results
        self.autoencoder_detector.predict.return_value = (True, 0.9, {"reconstruction_error": 0.15})
        self.gnn_analyzer.predict.return_value = (True, 0.8, {"malicious_probability": 0.8})
        self.rl_threat_scorer.predict.return_value = (True, 0.95, {"threat_score": 85})
        
        # Create malicious packet
        malicious_packet = TrafficPacket(
            timestamp=datetime.now(),
            src_ip="203.0.113.45",
            dst_ip="192.168.1.1",
            src_port=12345,
            dst_port=80,
            protocol=ProtocolType.TCP,
            packet_size=64,
            ttl=32,
            flags=["SYN"],
            payload_entropy=0.1,
            packet_id="malicious_packet_001"
        )
        
        # Run analysis
        result = asyncio.run(self.ai_engine.analyze_packet(malicious_packet))
        
        # Check result
        self.assertIsInstance(result, DetectionResult)
        self.assertEqual(result.packet_id, "malicious_packet_001")
        self.assertTrue(result.is_malicious)
        self.assertEqual(result.attack_type, AttackType.SYN_FLOOD)
        self.assertEqual(result.threat_score, 85)  # From RL scorer
        
        # Check explanation
        self.assertIn("model_results", result.explanation)
        self.assertEqual(result.explanation["consensus_votes"], 3)
        self.assertEqual(result.explanation["total_models"], 3)
        self.assertEqual(result.explanation["total_models"], 3)
    
    def test_analyze_packets_batch(self):
        """Test batch packet analysis"""
        # Create batch of packets
        packets = [
            self.sample_packet,
            TrafficPacket(
                timestamp=datetime.now(),
                src_ip="192.168.1.101",
                dst_ip="10.0.0.2",
                src_port=12346,
                dst_port=443,
                protocol=ProtocolType.TCP,
                packet_size=1024,
                ttl=64,
                flags=["ACK"],
                payload_entropy=0.7,
                packet_id="test_packet_002"
            )
        ]
        
        # Run batch analysis
        results = asyncio.run(self.ai_engine.analyze_packets_batch(packets))
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], DetectionResult)
        self.assertIsInstance(results[1], DetectionResult)
        
        # Check that both packets were added to history and buffer
        self.assertEqual(len(self.ai_engine.detection_history), 2)
        self.assertEqual(len(self.ai_engine.packet_buffer), 2)
    
    def test_get_explanation(self):
        """Test getting explanation for a prediction"""
        # First analyze a packet to cache the prediction
        result = asyncio.run(self.ai_engine.analyze_packet(self.sample_packet))
        
        # Configure XAI explainer mock
        self.xai_explainer.explain_prediction.return_value = {
            "top_features": [
                {"feature_name": "packet_size", "importance_score": 0.8},
                {"feature_name": "payload_entropy", "importance_score": 0.6}
            ]
        }
        
        # Get explanation
        explanation = self.ai_engine.get_explanation(result.packet_id)
        
        # Check explanation
        self.assertEqual(explanation["prediction_id"], result.packet_id)
        self.assertIn("explanation", explanation)
        
        # Check that XAI explainer was called
        self.xai_explainer.explain_prediction.assert_called_once()
    
    def test_get_recent_detections(self):
        """Test getting recent detections"""
        # Add some detections to history
        for i in range(5):
            packet = TrafficPacket(
                timestamp=datetime.now(),
                src_ip=f"192.168.1.{100 + i}",
                dst_ip="10.0.0.1",
                src_port=12345,
                dst_port=80,
                protocol=ProtocolType.TCP,
                packet_size=1024,
                ttl=64,
                flags=["ACK"],
                payload_entropy=0.7,
                packet_id=f"test_packet_{i:03d}"
            )
            asyncio.run(self.ai_engine.analyze_packet(packet))
        
        # Get recent detections
        detections = self.ai_engine.get_recent_detections(limit=3)
        
        # Check detections
        self.assertEqual(len(detections), 3)
        self.assertEqual(detections[0].packet_id, "test_packet_002")
        self.assertEqual(detections[2].packet_id, "test_packet_004")
    
    def test_get_performance_metrics(self):
        """Test getting performance metrics"""
        # Add some detections to history
        for i in range(5):
            packet = TrafficPacket(
                timestamp=datetime.now(),
                src_ip=f"192.168.1.{100 + i}",
                dst_ip="10.0.0.1",
                src_port=12345,
                dst_port=80,
                protocol=ProtocolType.TCP,
                packet_size=1024,
                ttl=64,
                flags=["ACK"],
                payload_entropy=0.7,
                packet_id=f"test_packet_{i:03d}"
            )
            asyncio.run(self.ai_engine.analyze_packet(packet))
        
        # Get metrics
        metrics = self.ai_engine.get_performance_metrics()
        
        # Check metrics
        self.assertEqual(metrics["packet_count"], 5)
        self.assertIn("avg_processing_time", metrics)
        self.assertIn("max_processing_time", metrics)
        self.assertIn("min_processing_time", metrics)
        self.assertEqual(metrics["malicious_count"], 0)  # All packets are benign
        self.assertEqual(metrics["benign_count"], 5)
    
    def test_error_handling(self):
        """Test error handling during packet analysis"""
        # Configure feature extractor to raise exception
        self.feature_extractor.extract_packet_features.side_effect = Exception("Test error")
        
        # Run analysis
        result = asyncio.run(self.ai_engine.analyze_packet(self.sample_packet))
        
        # Check result
        self.assertIsInstance(result, DetectionResult)
        self.assertFalse(result.is_malicious)
        self.assertEqual(result.attack_type, AttackType.BENIGN)
        self.assertEqual(result.detection_method, "error_fallback")
        self.assertIn("error", result.explanation)
        self.assertEqual(result.explanation["error"], "Test error")
    
    def test_attack_type_detection(self):
        """Test attack type detection"""
        # Configure mocks to return malicious results
        self.autoencoder_detector.predict.return_value = (True, 0.9, {"reconstruction_error": 0.15})
        self.gnn_analyzer.predict.return_value = (True, 0.8, {"malicious_probability": 0.8})
        self.rl_threat_scorer.predict.return_value = (True, 0.95, {"threat_score": 85})
        
        # Test SYN flood detection
        syn_packet = TrafficPacket(
            timestamp=datetime.now(),
            src_ip="203.0.113.45",
            dst_ip="192.168.1.1",
            src_port=12345,
            dst_port=80,
            protocol=ProtocolType.TCP,
            packet_size=64,
            ttl=32,
            flags=["SYN"],
            payload_entropy=0.1,
            packet_id="syn_flood_packet"
        )
        result = asyncio.run(self.ai_engine.analyze_packet(syn_packet))
        self.assertEqual(result.attack_type, AttackType.SYN_FLOOD)
        
        # Test UDP flood detection
        udp_packet = TrafficPacket(
            timestamp=datetime.now(),
            src_ip="203.0.113.45",
            dst_ip="192.168.1.1",
            src_port=12345,
            dst_port=53,
            protocol=ProtocolType.UDP,
            packet_size=64,
            ttl=32,
            flags=[],
            payload_entropy=0.1,
            packet_id="udp_flood_packet"
        )
        result = asyncio.run(self.ai_engine.analyze_packet(udp_packet))
        self.assertEqual(result.attack_type, AttackType.UDP_FLOOD)
        
        # Test HTTP flood detection
        http_packet = TrafficPacket(
            timestamp=datetime.now(),
            src_ip="203.0.113.45",
            dst_ip="192.168.1.1",
            src_port=12345,
            dst_port=80,
            protocol=ProtocolType.HTTP,
            packet_size=1024,
            ttl=32,
            flags=["ACK"],
            payload_entropy=0.1,
            packet_id="http_flood_packet"
        )
        result = asyncio.run(self.ai_engine.analyze_packet(http_packet))
        self.assertEqual(result.attack_type, AttackType.HTTP_FLOOD)
    
    def test_clear_history(self):
        """Test clearing history"""
        # Add some detections to history
        for i in range(5):
            packet = TrafficPacket(
                timestamp=datetime.now(),
                src_ip=f"192.168.1.{100 + i}",
                dst_ip="10.0.0.1",
                src_port=12345,
                dst_port=80,
                protocol=ProtocolType.TCP,
                packet_size=1024,
                ttl=64,
                flags=["ACK"],
                payload_entropy=0.7,
                packet_id=f"test_packet_{i:03d}"
            )
            asyncio.run(self.ai_engine.analyze_packet(packet))
        
        # Clear history
        self.ai_engine.clear_history()
        
        # Check that history is cleared
        self.assertEqual(len(self.ai_engine.detection_history), 0)
        self.assertEqual(len(self.ai_engine.packet_buffer), 0)
        self.assertEqual(len(self.ai_engine.processing_times), 0)
        self.assertEqual(len(self.ai_engine.prediction_cache), 0)
    
    def test_missing_components(self):
        """Test behavior with missing components"""
        # Create AI Engine with missing components
        ai_engine = AIEngine(feature_extractor=self.feature_extractor)
        
        # Run analysis
        result = asyncio.run(ai_engine.analyze_packet(self.sample_packet))
        
        # Check result
        self.assertIsInstance(result, DetectionResult)
        self.assertFalse(result.is_malicious)
        self.assertEqual(result.attack_type, AttackType.BENIGN)
        
        # Check that only feature extractor was called
        self.feature_extractor.extract_packet_features.assert_called_once()
        self.autoencoder_detector.predict.assert_not_called()
        self.gnn_analyzer.predict.assert_not_called()
        self.rl_threat_scorer.predict.assert_not_called()
    
    def test_missing_feature_extractor(self):
        """Test behavior with missing feature extractor"""
        # Create AI Engine with missing feature extractor
        ai_engine = AIEngine()
        
        # Run analysis
        result = asyncio.run(ai_engine.analyze_packet(self.sample_packet))
        
        # Check result
        self.assertIsInstance(result, DetectionResult)
        self.assertFalse(result.is_malicious)
        self.assertEqual(result.attack_type, AttackType.BENIGN)
        self.assertEqual(result.detection_method, "error_fallback")
        self.assertIn("error", result.explanation)
        self.assertIn("Feature extractor not initialized", result.explanation["error"])


if __name__ == "__main__":
    unittest.main()