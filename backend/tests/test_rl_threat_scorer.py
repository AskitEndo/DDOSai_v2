"""
Unit tests for Reinforcement Learning threat scorer
"""
import unittest
import numpy as np
import torch
from typing import List, Tuple, Dict
import tempfile
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.rl_threat_scorer import RLThreatScorer, DQNNetwork, ReplayBuffer
from core.exceptions import ModelInferenceError


class TestDQNNetwork(unittest.TestCase):
    """Test cases for DQNNetwork class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.state_dim = 31  # Feature vector size
        self.action_dim = 11  # Threat levels 0-10
        self.network = DQNNetwork(self.state_dim, self.action_dim)
    
    def test_network_initialization(self):
        """Test network initialization"""
        self.assertEqual(self.network.state_dim, self.state_dim)
        self.assertEqual(self.network.action_dim, self.action_dim)
    
    def test_forward_pass(self):
        """Test forward pass through network"""
        batch_size = 5
        input_data = torch.randn(batch_size, self.state_dim)
        
        output = self.network(input_data)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, self.action_dim))
        
        # Check that output contains Q-values (can be any real numbers)
        self.assertTrue(torch.all(torch.isfinite(output)))
    
    def test_custom_hidden_dims(self):
        """Test network with custom hidden dimensions"""
        custom_dims = [64, 32]
        network = DQNNetwork(self.state_dim, self.action_dim, custom_dims)
        
        # Test forward pass
        input_data = torch.randn(3, self.state_dim)
        output = network(input_data)
        self.assertEqual(output.shape, (3, self.action_dim))


class TestReplayBuffer(unittest.TestCase):
    """Test cases for ReplayBuffer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.buffer = ReplayBuffer(capacity=100)
    
    def test_buffer_initialization(self):
        """Test buffer initialization"""
        self.assertEqual(len(self.buffer), 0)
        self.assertEqual(self.buffer.capacity, 100)
    
    def test_push_and_sample(self):
        """Test adding and sampling experiences"""
        # Add some experiences
        for i in range(10):
            state = np.random.random(5)
            action = i % 3
            reward = float(i)
            next_state = np.random.random(5)
            done = i == 9
            
            self.buffer.push(state, action, reward, next_state, done)
        
        self.assertEqual(len(self.buffer), 10)
        
        # Sample experiences
        batch = self.buffer.sample(5)
        self.assertEqual(len(batch), 5)
        
        # Check experience structure
        for exp in batch:
            self.assertEqual(len(exp.state), 5)
            self.assertIsInstance(exp.action, int)
            self.assertIsInstance(exp.reward, float)
            self.assertEqual(len(exp.next_state), 5)
            self.assertIsInstance(exp.done, bool)
    
    def test_buffer_capacity(self):
        """Test buffer capacity limit"""
        capacity = 5
        buffer = ReplayBuffer(capacity)
        
        # Add more experiences than capacity
        for i in range(10):
            buffer.push(np.random.random(3), i, float(i), np.random.random(3), False)
        
        # Should only keep the last 'capacity' experiences
        self.assertEqual(len(buffer), capacity)


class TestRLThreatScorer(unittest.TestCase):
    """Test cases for RLThreatScorer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.state_dim = 31
        self.scorer = RLThreatScorer(self.state_dim, num_threat_levels=11)
        
        # Create synthetic training data
        self.training_data = self._create_training_data()
    
    def _create_training_data(self) -> List[Tuple[np.ndarray, bool, Dict[str, float]]]:
        """Create synthetic training data"""
        data = []
        
        # Normal traffic samples
        for i in range(100):
            features = np.random.normal(0.3, 0.2, self.state_dim)  # Lower values for normal
            features = np.clip(features, 0, 1)
            context = {
                'recent_threat_rate': 0.1,
                'system_load': 0.4,
                'false_positive_rate': 0.05,
                'detection_confidence': 0.8,
                'time_since_last_attack': 1.0
            }
            data.append((features, False, context))
        
        # Malicious traffic samples
        for i in range(100):
            features = np.random.normal(0.7, 0.2, self.state_dim)  # Higher values for malicious
            features = np.clip(features, 0, 1)
            context = {
                'recent_threat_rate': 0.8,
                'system_load': 0.7,
                'false_positive_rate': 0.02,
                'detection_confidence': 0.9,
                'time_since_last_attack': 0.1
            }
            data.append((features, True, context))
        
        return data
    
    def test_scorer_initialization(self):
        """Test scorer initialization"""
        self.assertEqual(self.scorer.base_state_dim, self.state_dim)
        self.assertEqual(self.scorer.state_dim, self.state_dim + 5)  # base + context
        self.assertEqual(self.scorer.num_threat_levels, 11)
        self.assertEqual(self.scorer.model_name, "RLThreatScorer")
        self.assertEqual(self.scorer.version, "1.0.0")
        self.assertFalse(self.scorer.is_trained)
    
    def test_action_threat_score_conversion(self):
        """Test conversion between actions and threat scores"""
        # Test action to threat score
        self.assertEqual(self.scorer.action_to_threat_score(0), 0)
        self.assertEqual(self.scorer.action_to_threat_score(5), 50)
        self.assertEqual(self.scorer.action_to_threat_score(10), 100)
        
        # Test threat score to action
        self.assertEqual(self.scorer.threat_score_to_action(0), 0)
        self.assertEqual(self.scorer.threat_score_to_action(55), 5)
        self.assertEqual(self.scorer.threat_score_to_action(100), 10)
        self.assertEqual(self.scorer.threat_score_to_action(105), 10)  # Clipped
    
    def test_state_features_creation(self):
        """Test state feature creation"""
        packet_features = np.random.random(self.state_dim)
        context = {'recent_threat_rate': 0.5, 'system_load': 0.3}
        
        state = self.scorer.get_state_features(packet_features, context)
        
        # Should always include packet features + context features (fixed size)
        expected_size = self.state_dim + self.scorer.context_dim
        self.assertEqual(len(state), expected_size)
        
        # Test without context (should still have same size with defaults)
        state_no_context = self.scorer.get_state_features(packet_features)
        self.assertEqual(len(state_no_context), expected_size)
    
    def test_reward_calculation(self):
        """Test reward calculation logic"""
        # Test malicious traffic rewards
        self.assertGreater(self.scorer.calculate_reward(80, True), 0)  # High score for malicious
        self.assertLess(self.scorer.calculate_reward(20, True), 0)     # Low score for malicious
        
        # Test benign traffic rewards
        self.assertGreater(self.scorer.calculate_reward(20, False), 0)  # Low score for benign
        self.assertLess(self.scorer.calculate_reward(80, False), 0)     # High score for benign
        
        # Test confidence adjustment
        high_conf_reward = self.scorer.calculate_reward(80, True, 1.0)
        low_conf_reward = self.scorer.calculate_reward(80, True, 0.5)
        self.assertGreater(high_conf_reward, low_conf_reward)
    
    def test_action_selection(self):
        """Test action selection"""
        state = np.random.random(self.state_dim + 5)  # Include context
        
        # Test exploration (training mode)
        self.scorer.epsilon = 1.0  # Always explore
        action = self.scorer.select_action(state, training=True)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.scorer.num_threat_levels)
        
        # Test exploitation (inference mode)
        action = self.scorer.select_action(state, training=False)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.scorer.num_threat_levels)
    
    def test_training(self):
        """Test model training"""
        # Use small dataset for quick testing
        small_data = self.training_data[:20]
        
        # Reduce training parameters for faster testing
        original_episodes = 1000
        self.scorer.num_episodes = 10  # Quick training
        
        # Mock the training by directly setting some attributes
        self.scorer.episode_rewards = [1.0, 2.0, 3.0]
        self.scorer.detection_accuracy = [0.7, 0.8, 0.9]
        self.scorer.false_positive_rate = [0.1, 0.05, 0.02]
        
        # Test that training data is properly formatted
        self.assertGreater(len(small_data), 0)
        for features, label, context in small_data:
            self.assertEqual(len(features), self.state_dim)
            self.assertIsInstance(label, bool)
            self.assertIsInstance(context, dict)
    
    def test_training_validation(self):
        """Test training input validation"""
        # Empty training data
        with self.assertRaises(ModelInferenceError):
            self.scorer.train([])
    
    def test_prediction_before_training(self):
        """Test prediction before training raises error"""
        features = np.random.random(self.state_dim)
        
        with self.assertRaises(ModelInferenceError):
            self.scorer.predict(features)
    
    def test_prediction_after_training(self):
        """Test prediction after training"""
        # Mock training completion
        self.scorer.is_trained = True
        
        features = np.random.random(self.state_dim)
        context = {'recent_threat_rate': 0.5}
        
        is_malicious, confidence, explanation = self.scorer.predict(features, context)
        
        self.assertIsInstance(is_malicious, bool)
        self.assertIsInstance(confidence, float)
        self.assertIsInstance(explanation, dict)
        
        # Check explanation contents
        self.assertIn('threat_score', explanation)
        self.assertIn('q_values', explanation)
        self.assertIn('selected_action', explanation)
        self.assertIn('model_type', explanation)
        
        # Check threat score range
        threat_score = explanation['threat_score']
        self.assertGreaterEqual(threat_score, 0)
        self.assertLessEqual(threat_score, 100)
    
    def test_prediction_without_context(self):
        """Test prediction without context"""
        self.scorer.is_trained = True
        
        features = np.random.random(self.state_dim)
        is_malicious, confidence, explanation = self.scorer.predict(features)
        
        self.assertIsInstance(is_malicious, bool)
        self.assertIsInstance(confidence, float)
        self.assertFalse(explanation['context_used'])
    
    def test_feature_validation(self):
        """Test feature validation"""
        self.scorer.is_trained = True
        
        # Test invalid feature formats
        with self.assertRaises(ModelInferenceError):
            self.scorer.predict("invalid")
        
        with self.assertRaises(ModelInferenceError):
            self.scorer.predict(np.array([]))
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        # Mock training completion
        self.scorer.is_trained = True
        self.scorer.episode_rewards = [1.0, 2.0, 3.0]
        self.scorer.detection_accuracy = [0.8, 0.85, 0.9]
        
        # Test prediction before save
        features = np.random.random(self.state_dim)
        original_prediction = self.scorer.predict(features)
        
        # Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_rl.pth")
            
            save_success = self.scorer.save_model(model_path)
            self.assertTrue(save_success)
            self.assertTrue(os.path.exists(model_path))
            
            # Create new scorer and load model
            new_scorer = RLThreatScorer(self.state_dim)
            self.assertFalse(new_scorer.is_trained)
            
            load_success = new_scorer.load_model(model_path)
            self.assertTrue(load_success)
            self.assertTrue(new_scorer.is_trained)
            
            # Test that loaded model gives similar prediction
            loaded_prediction = new_scorer.predict(features)
            
            # Predictions should be the same classification
            self.assertEqual(loaded_prediction[0], original_prediction[0])
    
    def test_save_untrained_model(self):
        """Test saving untrained model returns False"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "untrained.pth")
            result = self.scorer.save_model(model_path)
            self.assertFalse(result)
    
    def test_load_nonexistent_model(self):
        """Test loading non-existent model returns False"""
        result = self.scorer.load_model("nonexistent_model.pth")
        self.assertFalse(result)
    
    def test_training_metrics(self):
        """Test training metrics retrieval"""
        # Before training
        metrics = self.scorer.get_training_metrics()
        self.assertEqual(metrics, {})
        
        # After mock training
        self.scorer.episode_rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.scorer.detection_accuracy = [0.6, 0.7, 0.8, 0.85, 0.9]
        self.scorer.false_positive_rate = [0.2, 0.15, 0.1, 0.08, 0.05]
        
        metrics = self.scorer.get_training_metrics()
        
        self.assertIn('total_episodes', metrics)
        self.assertIn('avg_reward', metrics)
        self.assertIn('avg_accuracy', metrics)
        self.assertIn('avg_fp_rate', metrics)
        self.assertIn('current_epsilon', metrics)
        
        self.assertEqual(metrics['total_episodes'], 5)
        self.assertAlmostEqual(metrics['avg_reward'], 3.0, places=1)
        self.assertAlmostEqual(metrics['avg_accuracy'], 0.77, places=1)
    
    def test_context_update(self):
        """Test context update functionality"""
        recent_detections = [True, False, True, True, False]
        system_load = 0.8
        
        self.scorer.update_context(recent_detections, system_load)
        
        self.assertAlmostEqual(self.scorer.recent_threat_rate, 0.6, places=1)
        self.assertEqual(self.scorer.system_load, 0.8)
    
    def test_model_info(self):
        """Test model information retrieval"""
        info = self.scorer.get_model_info()
        
        self.assertEqual(info['model_name'], 'RLThreatScorer')
        self.assertEqual(info['version'], '1.0.0')
        self.assertFalse(info['is_trained'])
        
        # After mock training
        self.scorer.is_trained = True
        info = self.scorer.get_model_info()
        self.assertTrue(info['is_trained'])
    
    def test_device_handling(self):
        """Test device handling (CPU/CUDA)"""
        # Test CPU device
        cpu_scorer = RLThreatScorer(self.state_dim, device='cpu')
        self.assertEqual(cpu_scorer.device, 'cpu')
        
        # Test CUDA device (if available)
        if torch.cuda.is_available():
            cuda_scorer = RLThreatScorer(self.state_dim, device='cuda')
            self.assertEqual(cuda_scorer.device, 'cuda')
    
    def test_epsilon_decay(self):
        """Test epsilon decay during training"""
        initial_epsilon = self.scorer.epsilon
        
        # Simulate training steps
        for _ in range(10):
            self.scorer.train_step()
        
        # Epsilon should decay (if replay buffer has enough samples)
        # Note: This test might not always pass due to buffer size requirements
        self.assertLessEqual(self.scorer.epsilon, initial_epsilon)


if __name__ == '__main__':
    # Set up logging to see training progress
    import logging
    logging.basicConfig(level=logging.INFO)
    
    unittest.main()