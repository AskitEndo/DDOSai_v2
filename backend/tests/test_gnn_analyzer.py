"""
Unit tests for Graph Neural Network analyzer
"""
import unittest
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import List
import tempfile
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.gnn_analyzer import GNNAnalyzer, GNNNetwork
from models.data_models import TrafficPacket, ProtocolType
from core.exceptions import ModelInferenceError


class TestGNNNetwork(unittest.TestCase):
    """Test cases for GNNNetwork class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 11  # Node feature dimension
        self.hidden_dim = 32
        self.network = GNNNetwork(self.input_dim, self.hidden_dim)
    
    def test_network_initialization(self):
        """Test network initialization"""
        self.assertEqual(self.network.input_dim, self.input_dim)
        self.assertEqual(self.network.hidden_dim, self.hidden_dim)
        self.assertEqual(self.network.output_dim, 1)
        self.assertEqual(self.network.num_layers, 2)
    
    def test_forward_pass(self):
        """Test forward pass through network"""
        num_nodes = 5
        num_edges = 8
        
        # Create sample graph data
        x = torch.randn(num_nodes, self.input_dim)
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4],
            [1, 0, 2, 1, 3, 2, 4, 3]
        ], dtype=torch.long)
        
        output = self.network(x, edge_index)
        
        # Check output shape and range
        self.assertEqual(output.shape, (1, 1))  # Single graph output
        self.assertTrue(0 <= output.item() <= 1)  # Sigmoid output
    
    def test_batched_forward_pass(self):
        """Test forward pass with batched graphs"""
        # Create two small graphs
        num_nodes_1, num_nodes_2 = 3, 4
        total_nodes = num_nodes_1 + num_nodes_2
        
        x = torch.randn(total_nodes, self.input_dim)
        
        # Edge indices for two graphs
        edge_index = torch.tensor([
            [0, 1, 1, 2, 3, 4, 4, 5, 5, 6],  # Graph 1: nodes 0-2, Graph 2: nodes 3-6
            [1, 0, 2, 1, 4, 3, 5, 4, 6, 5]
        ], dtype=torch.long)
        
        # Batch assignment
        batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])  # First 3 nodes to graph 0, rest to graph 1
        
        output = self.network(x, edge_index, batch)
        
        # Check output shape for batched graphs
        self.assertEqual(output.shape, (2, 1))  # Two graph outputs
        self.assertTrue(torch.all((output >= 0) & (output <= 1)))


class TestGNNAnalyzer(unittest.TestCase):
    """Test cases for GNNAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.node_feature_dim = 11
        self.analyzer = GNNAnalyzer(self.node_feature_dim, hidden_dim=32, window_size=60)
        
        # Create sample traffic data
        self.normal_packets = self._create_normal_traffic()
        self.attack_packets = self._create_attack_traffic()
    
    def _create_normal_traffic(self) -> List[TrafficPacket]:
        """Create normal traffic patterns"""
        packets = []
        base_time = datetime.now()
        
        # Normal web browsing pattern
        for i in range(20):
            packet = TrafficPacket(
                timestamp=base_time + timedelta(seconds=i),
                src_ip=f"192.168.1.{100 + (i % 5)}",  # 5 client IPs
                dst_ip="93.184.216.34",  # Web server
                src_port=12000 + i,
                dst_port=80 if i % 2 == 0 else 443,
                protocol=ProtocolType.HTTP if i % 2 == 0 else ProtocolType.HTTPS,
                packet_size=800 + (i % 200),
                ttl=64,
                flags=["SYN", "ACK"] if i % 3 != 0 else ["SYN"],
                payload_entropy=0.6 + (i % 10) * 0.02,
                packet_id=f"normal_{i:03d}"
            )
            packets.append(packet)
        
        return packets
    
    def _create_attack_traffic(self) -> List[TrafficPacket]:
        """Create attack traffic patterns (SYN flood)"""
        packets = []
        base_time = datetime.now()
        
        # SYN flood from multiple sources to single target
        for i in range(50):
            packet = TrafficPacket(
                timestamp=base_time + timedelta(milliseconds=i * 20),
                src_ip=f"203.0.113.{i % 20}",  # 20 attacking IPs
                dst_ip="192.168.1.1",  # Target server
                src_port=12000 + i,
                dst_port=80,
                protocol=ProtocolType.TCP,
                packet_size=64,  # Small SYN packets
                ttl=32,
                flags=["SYN"],  # Only SYN flags
                payload_entropy=0.1,  # Low entropy
                packet_id=f"attack_{i:03d}"
            )
            packets.append(packet)
        
        return packets
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        self.assertEqual(self.analyzer.node_feature_dim, self.node_feature_dim)
        self.assertEqual(self.analyzer.model_name, "GNNAnalyzer")
        self.assertEqual(self.analyzer.version, "1.0.0")
        self.assertFalse(self.analyzer.is_trained)
    
    def test_graph_construction_normal(self):
        """Test graph construction from normal traffic"""
        graph = self.analyzer.construct_graph_from_packets(self.normal_packets)
        
        self.assertIsNotNone(graph)
        self.assertGreaterEqual(graph.num_nodes, self.analyzer.min_nodes)
        self.assertGreater(graph.edge_index.size(1), 0)
        self.assertEqual(graph.x.shape[1], self.node_feature_dim)
    
    def test_graph_construction_attack(self):
        """Test graph construction from attack traffic"""
        graph = self.analyzer.construct_graph_from_packets(self.attack_packets)
        
        self.assertIsNotNone(graph)
        self.assertGreaterEqual(graph.num_nodes, self.analyzer.min_nodes)
        self.assertGreater(graph.edge_index.size(1), 0)
    
    def test_graph_construction_insufficient_data(self):
        """Test graph construction with insufficient data"""
        # Too few packets
        few_packets = self.normal_packets[:2]
        graph = self.analyzer.construct_graph_from_packets(few_packets)
        self.assertIsNone(graph)
        
        # Empty packet list
        graph = self.analyzer.construct_graph_from_packets([])
        self.assertIsNone(graph)
    
    def test_node_feature_extraction(self):
        """Test node feature extraction"""
        # Create IP mapping
        ip_set = set()
        for packet in self.normal_packets:
            ip_set.add(packet.src_ip)
            ip_set.add(packet.dst_ip)
        
        ip_to_idx = {ip: idx for idx, ip in enumerate(sorted(ip_set))}
        
        features = self.analyzer._extract_node_features(self.normal_packets, ip_to_idx)
        
        # Check feature matrix shape
        self.assertEqual(features.shape[0], len(ip_to_idx))
        self.assertEqual(features.shape[1], self.node_feature_dim)
        
        # Check that features are numeric and finite
        self.assertTrue(np.all(np.isfinite(features)))
        
        # Check that some features are non-zero (there should be activity)
        self.assertTrue(np.any(features > 0))
    
    def test_training_data_preparation(self):
        """Test training data preparation"""
        # Prepare training data
        training_data = [
            (self.normal_packets, False),  # Normal traffic
            (self.attack_packets, True)    # Attack traffic
        ]
        
        # Test that we can construct graphs from training data
        valid_graphs = 0
        for packets, label in training_data:
            graph = self.analyzer.construct_graph_from_packets(packets)
            if graph is not None:
                valid_graphs += 1
        
        self.assertGreater(valid_graphs, 0)
    
    def test_training(self):
        """Test model training"""
        # Prepare training data
        training_data = []
        
        # Add multiple normal traffic samples
        for i in range(5):
            normal_sample = self._create_normal_traffic()
            training_data.append((normal_sample, False))
        
        # Add multiple attack samples
        for i in range(5):
            attack_sample = self._create_attack_traffic()
            training_data.append((attack_sample, True))
        
        # Train with reduced epochs for faster testing
        self.analyzer.num_epochs = 10
        
        training_stats = self.analyzer.train(training_data)
        
        # Check training completed successfully
        self.assertTrue(self.analyzer.is_trained)
        self.assertIn('epochs_trained', training_stats)
        self.assertIn('final_loss', training_stats)
        self.assertIn('training_graphs', training_stats)
        self.assertGreater(training_stats['training_graphs'], 0)
    
    def test_training_validation(self):
        """Test training input validation"""
        # Empty training data
        with self.assertRaises(ModelInferenceError):
            self.analyzer.train([])
        
        # Training data with no valid graphs
        invalid_data = [([], False), ([], True)]
        with self.assertRaises(ModelInferenceError):
            self.analyzer.train(invalid_data)
    
    def test_prediction_before_training(self):
        """Test prediction before training raises error"""
        with self.assertRaises(ModelInferenceError):
            self.analyzer.predict(self.normal_packets)
    
    def test_prediction_after_training(self):
        """Test prediction after training"""
        # Train model first
        training_data = [
            (self.normal_packets, False),
            (self.attack_packets, True)
        ]
        
        self.analyzer.num_epochs = 5  # Quick training
        self.analyzer.train(training_data)
        
        # Test prediction on normal traffic
        is_malicious, confidence, explanation = self.analyzer.predict(self.normal_packets)
        
        self.assertIsInstance(is_malicious, bool)
        self.assertIsInstance(confidence, float)
        self.assertIsInstance(explanation, dict)
        self.assertIn('malicious_probability', explanation)
        self.assertIn('num_nodes', explanation)
        self.assertIn('num_edges', explanation)
        
        # Test prediction on attack traffic
        is_malicious_attack, confidence_attack, explanation_attack = self.analyzer.predict(self.attack_packets)
        
        self.assertIsInstance(is_malicious_attack, bool)
        self.assertIsInstance(confidence_attack, float)
        self.assertIsInstance(explanation_attack, dict)
    
    def test_prediction_insufficient_data(self):
        """Test prediction with insufficient data for graph construction"""
        # Train model first
        training_data = [(self.normal_packets, False), (self.attack_packets, True)]
        self.analyzer.num_epochs = 5
        self.analyzer.train(training_data)
        
        # Test with insufficient data
        few_packets = self.normal_packets[:2]
        is_malicious, confidence, explanation = self.analyzer.predict(few_packets)
        
        # Should return benign with low confidence
        self.assertFalse(is_malicious)
        self.assertLess(confidence, 0.5)
        self.assertIn('error', explanation)
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        # Train model first
        training_data = [(self.normal_packets, False), (self.attack_packets, True)]
        self.analyzer.num_epochs = 5
        self.analyzer.train(training_data)
        
        # Test prediction before save
        original_prediction = self.analyzer.predict(self.normal_packets)
        
        # Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_gnn.pth")
            
            save_success = self.analyzer.save_model(model_path)
            self.assertTrue(save_success)
            self.assertTrue(os.path.exists(model_path))
            
            # Create new analyzer and load model
            new_analyzer = GNNAnalyzer(self.node_feature_dim, hidden_dim=32)
            self.assertFalse(new_analyzer.is_trained)
            
            load_success = new_analyzer.load_model(model_path)
            self.assertTrue(load_success)
            self.assertTrue(new_analyzer.is_trained)
            
            # Test that loaded model gives similar prediction
            loaded_prediction = new_analyzer.predict(self.normal_packets)
            
            # Predictions should be the same classification
            self.assertEqual(loaded_prediction[0], original_prediction[0])
    
    def test_save_untrained_model(self):
        """Test saving untrained model returns False"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "untrained.pth")
            result = self.analyzer.save_model(model_path)
            self.assertFalse(result)
    
    def test_load_nonexistent_model(self):
        """Test loading non-existent model returns False"""
        result = self.analyzer.load_model("nonexistent_model.pth")
        self.assertFalse(result)
    
    def test_network_structure_analysis(self):
        """Test network structure analysis"""
        analysis = self.analyzer.analyze_network_structure(self.normal_packets)
        
        # Check that analysis contains expected metrics
        expected_keys = ['num_nodes', 'num_edges', 'density', 'avg_clustering']
        for key in expected_keys:
            self.assertIn(key, analysis)
        
        # Check that values are reasonable
        self.assertGreater(analysis['num_nodes'], 0)
        self.assertGreater(analysis['num_edges'], 0)
        self.assertGreaterEqual(analysis['density'], 0)
        self.assertLessEqual(analysis['density'], 1)
    
    def test_network_structure_analysis_insufficient_data(self):
        """Test network structure analysis with insufficient data"""
        analysis = self.analyzer.analyze_network_structure([])
        self.assertIn('error', analysis)
    
    def test_model_info(self):
        """Test model information retrieval"""
        info = self.analyzer.get_model_info()
        
        self.assertEqual(info['model_name'], 'GNNAnalyzer')
        self.assertEqual(info['version'], '1.0.0')
        self.assertFalse(info['is_trained'])
        
        # After training
        training_data = [(self.normal_packets, False), (self.attack_packets, True)]
        self.analyzer.num_epochs = 5
        self.analyzer.train(training_data)
        
        info = self.analyzer.get_model_info()
        self.assertTrue(info['is_trained'])
    
    def test_device_handling(self):
        """Test device handling (CPU/CUDA)"""
        # Test CPU device
        cpu_analyzer = GNNAnalyzer(self.node_feature_dim, device='cpu')
        self.assertEqual(cpu_analyzer.device, 'cpu')
        
        # Test CUDA device (if available)
        if torch.cuda.is_available():
            cuda_analyzer = GNNAnalyzer(self.node_feature_dim, device='cuda')
            self.assertEqual(cuda_analyzer.device, 'cuda')
    
    def test_large_graph_handling(self):
        """Test handling of large graphs"""
        # Create traffic with many IPs (should be limited)
        large_packets = []
        base_time = datetime.now()
        
        for i in range(200):  # Many packets with different IPs
            packet = TrafficPacket(
                timestamp=base_time + timedelta(seconds=i),
                src_ip=f"10.0.{i//10}.{i%10}",  # Many different IPs
                dst_ip="192.168.1.1",
                src_port=12000 + i,
                dst_port=80,
                protocol=ProtocolType.TCP,
                packet_size=100,
                ttl=64,
                flags=["SYN"],
                payload_entropy=0.5,
                packet_id=f"large_{i:03d}"
            )
            large_packets.append(packet)
        
        graph = self.analyzer.construct_graph_from_packets(large_packets)
        
        # Should limit graph size
        if graph is not None:
            self.assertLessEqual(graph.num_nodes, self.analyzer.max_nodes)


if __name__ == '__main__':
    # Set up logging to see training progress
    import logging
    logging.basicConfig(level=logging.INFO)
    
    unittest.main()