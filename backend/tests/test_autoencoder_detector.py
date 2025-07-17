"""
Unit tests for autoencoder anomaly detector
"""
import unittest
import numpy as np
import torch
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.autoencoder_detector import AutoencoderDetector, AutoencoderNetwork
from core.exceptions import ModelInferenceError


class TestAutoencoderNetwork(unittest.TestCase):
    """Test cases for AutoencoderNetwork class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 31  # Same as feature extractor output
        self.network = AutoencoderNetwork(self.input_dim)
    
    def test_network_initialization(self):
        """Test network initialization"""
        self.assertEqual(self.network.input_dim, self.input_dim)
        self.assertEqual(self.network.hidden_dims, [64, 32, 16, 32, 64])
        
        # Check that network has encoder and decoder
        self.assertTrue(hasattr(self.network, 'encoder'))
        self.assertTrue(hasattr(self.network, 'decoder'))
    
    def test_forward_pass(self):
        """Test forward pass through network"""
        batch_size = 10
        input_data = torch.randn(batch_size, self.input_dim)
        
        output = self.network(input_data)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, self.input_dim))
        
        # Check output is in valid range (sigmoid output)
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))
    
    def test_encode_function(self):
        """Test encoding function"""
        batch_size = 5
        input_data = torch.randn(batch_size, self.input_dim)
        
        encoded = self.network.encode(input_data)
        
        # Encoded dimension should be the bottleneck dimension (16)
        self.assertEqual(encoded.shape[0], batch_size)
        self.assertLess(encoded.shape[1], self.input_dim)  # Compressed representation
    
    def test_custom_hidden_dims(self):
        """Test network with custom hidden dimensions"""
        custom_dims = [32, 16, 8, 16, 32]
        network = AutoencoderNetwork(self.input_dim, custom_dims)
        
        self.assertEqual(network.hidden_dims, custom_dims)
        
        # Test forward pass
        input_data = torch.randn(3, self.input_dim)
        output = network(input_data)
        self.assertEqual(output.shape, (3, self.input_dim))


class TestAutoencoderDetector(unittest.TestCase):
    """Test cases for AutoencoderDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 31
        self.detector = AutoencoderDetector(self.input_dim)
        
        # Create synthetic training data (normal traffic)
        np.random.seed(42)  # For reproducible tests
        self.normal_data = self._generate_normal_data(1000)
        self.anomalous_data = self._generate_anomalous_data(100)
    
    def _generate_normal_data(self, n_samples: int) -> np.ndarray:
        """Generate synthetic normal traffic data"""
        # Normal traffic: moderate values, some correlation
        data = np.random.normal(0.5, 0.2, (n_samples, self.input_dim))
        
        # Add some structure (correlations)
        data[:, 1] = data[:, 0] * 0.8 + np.random.normal(0, 0.1, n_samples)  # Correlated features
        data[:, 2] = np.random.uniform(0.3, 0.7, n_samples)  # Entropy in normal range
        
        # Clip to valid range
        data = np.clip(data, 0, 1)
        return data
    
    def _generate_anomalous_data(self, n_samples: int) -> np.ndarray:
        """Generate synthetic anomalous traffic data"""
        # Anomalous traffic: extreme values, different patterns
        data = np.random.normal(0.2, 0.4, (n_samples, self.input_dim))
        
        # Add anomalous patterns - make them very different from normal
        data[:, 0] = np.random.uniform(0.95, 1.0, n_samples)  # Very high values
        data[:, 1] = np.random.uniform(0.0, 0.05, n_samples)  # Very low values (break correlation)
        data[:, 2] = np.random.uniform(0.0, 0.1, n_samples)  # Very low entropy
        data[:, 3] = np.random.uniform(0.9, 1.0, n_samples)  # Another high feature
        data[:, 4] = np.random.uniform(0.0, 0.1, n_samples)  # Another low feature
        
        # Make more features anomalous
        for i in range(5, min(10, self.input_dim)):
            if i % 2 == 0:
                data[:, i] = np.random.uniform(0.9, 1.0, n_samples)
            else:
                data[:, i] = np.random.uniform(0.0, 0.1, n_samples)
        
        # Clip to valid range
        data = np.clip(data, 0, 1)
        return data
    
    def test_detector_initialization(self):
        """Test detector initialization"""
        self.assertEqual(self.detector.input_dim, self.input_dim)
        self.assertEqual(self.detector.model_name, "AutoencoderDetector")
        self.assertEqual(self.detector.version, "1.0.0")
        self.assertFalse(self.detector.is_trained)
        self.assertIsNone(self.detector.anomaly_threshold)
    
    def test_training(self):
        """Test model training"""
        # Train with reduced epochs for faster testing
        self.detector.num_epochs = 10
        
        training_stats = self.detector.train(self.normal_data)
        
        # Check training completed successfully
        self.assertTrue(self.detector.is_trained)
        self.assertIsNotNone(self.detector.anomaly_threshold)
        self.assertGreater(self.detector.anomaly_threshold, 0)
        
        # Check training statistics
        self.assertIn('epochs_trained', training_stats)
        self.assertIn('final_loss', training_stats)
        self.assertIn('anomaly_threshold', training_stats)
        self.assertIn('training_samples', training_stats)
        self.assertEqual(training_stats['training_samples'], len(self.normal_data))
    
    def test_training_validation(self):
        """Test training input validation"""
        # Empty training data
        with self.assertRaises(ModelInferenceError):
            self.detector.train(np.array([]))
        
        # Wrong input dimension
        wrong_dim_data = np.random.random((100, self.input_dim + 5))
        with self.assertRaises(ModelInferenceError):
            self.detector.train(wrong_dim_data)
    
    def test_prediction_before_training(self):
        """Test prediction before training raises error"""
        test_sample = np.random.random(self.input_dim)
        
        with self.assertRaises(ModelInferenceError):
            self.detector.predict(test_sample)
    
    def test_single_prediction(self):
        """Test single sample prediction"""
        # Train model first
        self.detector.num_epochs = 5  # Quick training
        self.detector.train(self.normal_data)
        
        # Test normal sample
        normal_sample = self.normal_data[0]
        is_malicious, confidence, explanation = self.detector.predict(normal_sample)
        
        self.assertIsInstance(is_malicious, bool)
        self.assertIsInstance(confidence, float)
        self.assertIsInstance(explanation, dict)
        self.assertIn('reconstruction_error', explanation)
        self.assertIn('anomaly_threshold', explanation)
        
        # Test anomalous sample
        anomalous_sample = self.anomalous_data[0]
        is_malicious_anom, confidence_anom, explanation_anom = self.detector.predict(anomalous_sample)
        
        # Just check that we can make predictions on both types
        # The actual discrimination will be tested in the accuracy test
        self.assertIsInstance(is_malicious_anom, bool)
        self.assertIsInstance(confidence_anom, float)
        self.assertIsInstance(explanation_anom, dict)
    
    def test_batch_prediction(self):
        """Test batch prediction"""
        # Train model first
        self.detector.num_epochs = 5
        self.detector.train(self.normal_data)
        
        # Test batch of mixed samples
        test_batch = np.vstack([self.normal_data[:5], self.anomalous_data[:5]])
        results = self.detector.predict_batch(test_batch)
        
        self.assertEqual(len(results), 10)
        
        # Check result format
        for is_malicious, confidence, explanation in results:
            self.assertIsInstance(is_malicious, bool)
            self.assertIsInstance(confidence, float)
            self.assertIsInstance(explanation, dict)
    
    def test_feature_validation(self):
        """Test feature validation"""
        # Train model first
        self.detector.num_epochs = 5
        self.detector.train(self.normal_data)
        
        # Test invalid feature formats
        with self.assertRaises(ModelInferenceError):
            self.detector.predict("invalid")
        
        with self.assertRaises(ModelInferenceError):
            self.detector.predict(np.array([]))
        
        # Wrong dimension
        wrong_dim = np.random.random(self.input_dim + 1)
        with self.assertRaises(ModelInferenceError):
            self.detector.predict(wrong_dim)
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        # Train model first
        self.detector.num_epochs = 5
        training_stats = self.detector.train(self.normal_data)
        original_threshold = self.detector.anomaly_threshold
        
        # Test prediction before save
        test_sample = self.normal_data[0]
        original_prediction = self.detector.predict(test_sample)
        
        # Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_autoencoder.pth")
            
            save_success = self.detector.save_model(model_path)
            self.assertTrue(save_success)
            self.assertTrue(os.path.exists(model_path))
            
            # Create new detector and load model
            new_detector = AutoencoderDetector(self.input_dim)
            self.assertFalse(new_detector.is_trained)
            
            load_success = new_detector.load_model(model_path)
            self.assertTrue(load_success)
            self.assertTrue(new_detector.is_trained)
            self.assertEqual(new_detector.anomaly_threshold, original_threshold)
            
            # Test that loaded model gives same prediction
            loaded_prediction = new_detector.predict(test_sample)
            
            # Predictions should be very similar (allowing for small numerical differences)
            self.assertEqual(loaded_prediction[0], original_prediction[0])  # Same classification
            self.assertAlmostEqual(loaded_prediction[1], original_prediction[1], places=4)  # Similar confidence
    
    def test_save_untrained_model(self):
        """Test saving untrained model returns False"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "untrained.pth")
            
            # Should return False instead of raising exception
            result = self.detector.save_model(model_path)
            self.assertFalse(result)
    
    def test_load_nonexistent_model(self):
        """Test loading non-existent model returns False"""
        result = self.detector.load_model("nonexistent_model.pth")
        self.assertFalse(result)
    
    def test_reconstruction_statistics(self):
        """Test reconstruction statistics"""
        # Before training
        stats = self.detector.get_reconstruction_statistics()
        self.assertEqual(stats, {})
        
        # After training
        self.detector.num_epochs = 5
        self.detector.train(self.normal_data)
        
        stats = self.detector.get_reconstruction_statistics()
        
        self.assertIn('count', stats)
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('percentiles', stats)
        self.assertGreater(stats['count'], 0)
    
    def test_threshold_update(self):
        """Test threshold update functionality"""
        # Train model first
        self.detector.num_epochs = 5
        self.detector.train(self.normal_data)
        
        original_threshold = self.detector.anomaly_threshold
        
        # Update threshold to different percentile
        self.detector.update_threshold(90.0)
        new_threshold = self.detector.anomaly_threshold
        
        self.assertNotEqual(original_threshold, new_threshold)
        self.assertEqual(self.detector.threshold_percentile, 90.0)
    
    def test_threshold_update_without_training(self):
        """Test threshold update without training data raises error"""
        with self.assertRaises(ModelInferenceError):
            self.detector.update_threshold(90.0)
    
    def test_model_info(self):
        """Test model information retrieval"""
        info = self.detector.get_model_info()
        
        self.assertEqual(info['model_name'], 'AutoencoderDetector')
        self.assertEqual(info['version'], '1.0.0')
        self.assertFalse(info['is_trained'])
        
        # After training
        self.detector.num_epochs = 5
        self.detector.train(self.normal_data)
        
        info = self.detector.get_model_info()
        self.assertTrue(info['is_trained'])
    
    def test_anomaly_detection_accuracy(self):
        """Test that model can distinguish normal from anomalous traffic"""
        # Train model
        self.detector.num_epochs = 20  # More epochs for better accuracy
        self.detector.train(self.normal_data)
        
        # Test on normal data
        normal_predictions = self.detector.predict_batch(self.normal_data[:50])
        normal_anomaly_rate = sum(pred[0] for pred in normal_predictions) / len(normal_predictions)
        
        # Test on anomalous data
        anomalous_predictions = self.detector.predict_batch(self.anomalous_data[:50])
        anomalous_detection_rate = sum(pred[0] for pred in anomalous_predictions) / len(anomalous_predictions)
        
        # Normal data should have reasonable anomaly rate (autoencoders can have higher false positives)
        self.assertLess(normal_anomaly_rate, 0.6)  # Less than 60% false positives
        
        # Anomalous data should have higher detection rate than normal data
        self.assertGreater(anomalous_detection_rate, normal_anomaly_rate)  # Better than random
        
        print(f"Normal data anomaly rate: {normal_anomaly_rate:.2%}")
        print(f"Anomalous data detection rate: {anomalous_detection_rate:.2%}")
    
    def test_device_handling(self):
        """Test device handling (CPU/CUDA)"""
        # Test CPU device
        cpu_detector = AutoencoderDetector(self.input_dim, device='cpu')
        self.assertEqual(cpu_detector.device, 'cpu')
        
        # Test CUDA device (if available)
        if torch.cuda.is_available():
            cuda_detector = AutoencoderDetector(self.input_dim, device='cuda')
            self.assertEqual(cuda_detector.device, 'cuda')
    
    def test_custom_architecture(self):
        """Test custom network architecture"""
        custom_dims = [32, 16, 8, 16, 32]
        detector = AutoencoderDetector(self.input_dim, hidden_dims=custom_dims)
        
        self.assertEqual(detector.hidden_dims, custom_dims)
        
        # Test training with custom architecture
        detector.num_epochs = 5
        training_stats = detector.train(self.normal_data)
        
        self.assertTrue(detector.is_trained)
        self.assertIn('final_loss', training_stats)


if __name__ == '__main__':
    # Set up logging to see training progress
    import logging
    logging.basicConfig(level=logging.INFO)
    
    unittest.main()