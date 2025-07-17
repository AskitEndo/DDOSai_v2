"""
Integration tests for AI pipeline components
"""
import pytest
import os
import sys
import numpy as np
import json
from datetime import datetime
import uuid
from typing import Dict, Any, List

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import AI components
from models.data_models import TrafficPacket, DetectionResult, ProtocolType
from core.feature_extractor import FeatureExtractor
from ai.autoencoder_detector import AutoencoderDetector
from ai.gnn_analyzer import GNNAnalyzer
from ai.rl_threat_scorer import RLThreatScorer
from ai.xai_explainer import XAIExplainer
from ai.ai_engine import AIEngine

# Test data
def create_test_packet(is_malicious=False):
    """Create a test packet"""
    if is_malicious:
        # Create a packet that looks like a SYN flood attack
        return TrafficPacket(
            timestamp=datetime.now(),
            src_ip=f"192.168.1.{np.random.randint(1, 255)}",  # Random source IP
            dst_ip="10.0.0.1",  # Same target
            src_port=np.random.randint(1024, 65535),  # Random source port
            dst_port=80,  # Web server port
            protocol=ProtocolType.TCP,
            packet_size=64,  # Small packet size
            ttl=64,
            flags=["SYN"],  # SYN flag only
            payload_entropy=0.1,  # Low entropy
            packet_id=f"test_malicious_{uuid.uuid4().hex[:8]}"
        )
    else:
        # Create a normal-looking packet
        return TrafficPacket(
            timestamp=datetime.now(),
            src_ip=f"10.1.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
            dst_ip=f"10.0.0.{np.random.randint(1, 10)}",
            src_port=np.random.randint(1024, 65535),
            dst_port=np.random.randint(1, 1024),
            protocol=ProtocolType.TCP,
            packet_size=np.random.randint(64, 1500),
            ttl=64,
            flags=["ACK"] if np.random.random() > 0.5 else ["ACK", "PSH"],
            payload_entropy=np.random.uniform(0.6, 0.9),
            packet_id=f"test_benign_{uuid.uuid4().hex[:8]}"
        )

@pytest.fixture
def ai_components():
    """Initialize AI components for testing"""
    # Initialize feature extractor
    feature_extractor = FeatureExtractor()
    
    # Initialize autoencoder detector
    autoencoder_detector = AutoencoderDetector(input_dim=31, threshold_percentile=95.0)
    
    # Initialize GNN analyzer
    gnn_analyzer = GNNAnalyzer(node_feature_dim=31, hidden_dim=64)
    
    # Initialize RL threat scorer
    rl_threat_scorer = RLThreatScorer(state_dim=31, num_threat_levels=11)
    
    # Initialize XAI explainer
    feature_names = feature_extractor.get_feature_names()
    xai_explainer = XAIExplainer(feature_names=feature_names)
    
    # Initialize AI Engine orchestrator
    ai_engine = AIEngine(
        feature_extractor=feature_extractor,
        autoencoder_detector=autoencoder_detector,
        gnn_analyzer=gnn_analyzer,
        rl_threat_scorer=rl_threat_scorer,
        xai_explainer=xai_explainer
    )
    
    return {
        "feature_extractor": feature_extractor,
        "autoencoder_detector": autoencoder_detector,
        "gnn_analyzer": gnn_analyzer,
        "rl_threat_scorer": rl_threat_scorer,
        "xai_explainer": xai_explainer,
        "ai_engine": ai_engine
    }

class TestAIPipeline:
    """Test AI pipeline components integration"""
    
    def test_feature_extraction(self, ai_components):
        """Test feature extraction from packets"""
        feature_extractor = ai_components["feature_extractor"]
        
        # Create test packets
        benign_packet = create_test_packet(is_malicious=False)
        malicious_packet = create_test_packet(is_malicious=True)
        
        # Extract features
        benign_features = feature_extractor.extract_packet_features(benign_packet)
        malicious_features = feature_extractor.extract_packet_features(malicious_packet)
        
        # Check feature dimensions
        assert len(benign_features) == 31, "Feature vector should have 31 dimensions"
        assert len(malicious_features) == 31, "Feature vector should have 31 dimensions"
        
        # Check feature types
        assert isinstance(benign_features, np.ndarray), "Features should be a numpy array"
        assert benign_features.dtype == np.float32, "Features should be float32"
    
    def test_autoencoder_detection(self, ai_components):
        """Test autoencoder anomaly detection"""
        feature_extractor = ai_components["feature_extractor"]
        autoencoder_detector = ai_components["autoencoder_detector"]
        
        # Create test packets
        benign_packet = create_test_packet(is_malicious=False)
        malicious_packet = create_test_packet(is_malicious=True)
        
        # Extract features
        benign_features = feature_extractor.extract_packet_features(benign_packet)
        malicious_features = feature_extractor.extract_packet_features(malicious_packet)
        
        # Run detection
        benign_result = autoencoder_detector.predict(benign_features)
        malicious_result = autoencoder_detector.predict(malicious_features)
        
        # Check result format
        assert len(benign_result) == 3, "Result should be a tuple of (is_anomaly, confidence, explanation)"
        assert isinstance(benign_result[0], bool), "First element should be a boolean"
        assert isinstance(benign_result[1], float), "Second element should be a float"
        assert isinstance(benign_result[2], dict), "Third element should be a dictionary"
        
        # Check explanation
        assert "reconstruction_error" in benign_result[2], "Explanation should include reconstruction error"
    
    def test_gnn_analysis(self, ai_components):
        """Test GNN network analysis"""
        feature_extractor = ai_components["feature_extractor"]
        gnn_analyzer = ai_components["gnn_analyzer"]
        
        # Create test packets
        benign_packet = create_test_packet(is_malicious=False)
        malicious_packet = create_test_packet(is_malicious=True)
        
        # Extract features
        benign_features = feature_extractor.extract_packet_features(benign_packet)
        malicious_features = feature_extractor.extract_packet_features(malicious_packet)
        
        # Run analysis
        benign_result = gnn_analyzer.predict(benign_features)
        malicious_result = gnn_analyzer.predict(malicious_features)
        
        # Check result format
        assert len(benign_result) == 3, "Result should be a tuple of (is_malicious, confidence, explanation)"
        assert isinstance(benign_result[0], bool), "First element should be a boolean"
        assert isinstance(benign_result[1], float), "Second element should be a float"
        assert isinstance(benign_result[2], dict), "Third element should be a dictionary"
    
    def test_rl_threat_scoring(self, ai_components):
        """Test RL threat scoring"""
        feature_extractor = ai_components["feature_extractor"]
        rl_threat_scorer = ai_components["rl_threat_scorer"]
        
        # Create test packets
        benign_packet = create_test_packet(is_malicious=False)
        malicious_packet = create_test_packet(is_malicious=True)
        
        # Extract features
        benign_features = feature_extractor.extract_packet_features(benign_packet)
        malicious_features = feature_extractor.extract_packet_features(malicious_packet)
        
        # Run threat scoring
        benign_result = rl_threat_scorer.predict(benign_features, {})
        malicious_result = rl_threat_scorer.predict(malicious_features, {})
        
        # Check result format
        assert len(benign_result) == 3, "Result should be a tuple of (is_malicious, confidence, explanation)"
        assert isinstance(benign_result[0], bool), "First element should be a boolean"
        assert isinstance(benign_result[1], float), "Second element should be a float"
        assert isinstance(benign_result[2], dict), "Third element should be a dictionary"
        
        # Check explanation
        assert "threat_score" in benign_result[2], "Explanation should include threat score"
        assert 0 <= benign_result[2]["threat_score"] <= 100, "Threat score should be between 0 and 100"
    
    def test_xai_explanation(self, ai_components):
        """Test XAI explanation generation"""
        feature_extractor = ai_components["feature_extractor"]
        xai_explainer = ai_components["xai_explainer"]
        
        # Create test packet
        packet = create_test_packet(is_malicious=True)
        
        # Extract features
        features = feature_extractor.extract_packet_features(packet)
        
        # Create a simple prediction function for the explainer
        def predict_fn(features_array):
            # Return probabilities for binary classification (benign, malicious)
            return np.array([[0.2, 0.8]])
        
        # Create dummy prediction result
        prediction = (
            True,  # is_malicious
            0.8,   # confidence
            {"threat_score": 75}  # explanation
        )
        
        # Initialize explainers with dummy data
        dummy_data = np.random.random((10, len(features)))
        xai_explainer.initialize_explainers(dummy_data, predict_fn)
        
        # Generate explanation
        explanation = xai_explainer.explain_prediction(
            features,
            predict_fn,
            prediction,
            method="lime"
        )
        
        # Check explanation format
        assert "feature_importance" in explanation, "Explanation should include feature importance"
        assert "top_features" in explanation, "Explanation should include top features"
        assert len(explanation["top_features"]) > 0, "Top features should not be empty"
    
    def test_ai_engine_integration(self, ai_components):
        """Test AI Engine orchestration of all components"""
        ai_engine = ai_components["ai_engine"]
        
        # Create test packets
        benign_packet = create_test_packet(is_malicious=False)
        malicious_packet = create_test_packet(is_malicious=True)
        
        # Analyze packets
        benign_result = ai_engine.analyze_packet(benign_packet)
        malicious_result = ai_engine.analyze_packet(malicious_packet)
        
        # Check result types
        assert isinstance(benign_result, DetectionResult), "Result should be a DetectionResult"
        assert isinstance(malicious_result, DetectionResult), "Result should be a DetectionResult"
        
        # Check result fields
        assert hasattr(benign_result, "is_malicious"), "Result should have is_malicious field"
        assert hasattr(benign_result, "threat_score"), "Result should have threat_score field"
        assert hasattr(benign_result, "confidence"), "Result should have confidence field"
        assert hasattr(benign_result, "explanation"), "Result should have explanation field"
        
        # Check that malicious packet has higher threat score
        assert malicious_result.threat_score >= benign_result.threat_score, \
            "Malicious packet should have higher threat score"
    
    def test_batch_processing(self, ai_components):
        """Test batch processing of multiple packets"""
        ai_engine = ai_components["ai_engine"]
        
        # Create batch of test packets
        packets = [create_test_packet(is_malicious=i % 2 == 0) for i in range(10)]
        
        # Process batch
        results = []
        for packet in packets:
            result = ai_engine.analyze_packet(packet)
            results.append(result)
        
        # Check results
        assert len(results) == 10, "Should have 10 results"
        for result in results:
            assert isinstance(result, DetectionResult), "Result should be a DetectionResult"
    
    def test_explanation_retrieval(self, ai_components):
        """Test retrieving explanation for a prediction"""
        ai_engine = ai_components["ai_engine"]
        
        # Create and analyze a packet
        packet = create_test_packet(is_malicious=True)
        result = ai_engine.analyze_packet(packet)
        
        # Get explanation
        explanation = ai_engine.get_explanation(packet.packet_id)
        
        # Check explanation
        assert explanation is not None, "Explanation should not be None"
        assert "prediction_id" in explanation, "Explanation should include prediction ID"
        assert explanation["prediction_id"] == packet.packet_id, "Prediction ID should match"
    
    def test_performance_metrics(self, ai_components):
        """Test performance metrics collection"""
        ai_engine = ai_components["ai_engine"]
        
        # Process some packets
        for _ in range(5):
            packet = create_test_packet(is_malicious=False)
            ai_engine.analyze_packet(packet)
        
        # Get performance metrics
        metrics = ai_engine.get_performance_metrics()
        
        # Check metrics
        assert "packet_count" in metrics, "Metrics should include packet count"
        assert metrics["packet_count"] >= 5, "Packet count should be at least 5"
        assert "avg_processing_time" in metrics, "Metrics should include average processing time"
        assert metrics["avg_processing_time"] > 0, "Average processing time should be positive"

if __name__ == "__main__":
    pytest.main(["-v", __file__])