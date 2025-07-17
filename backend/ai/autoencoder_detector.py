"""
Autoencoder-based anomaly detection for DDoS traffic
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, List, Tuple, Optional
import pickle
import logging
from pathlib import Path
import json
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_detector import BaseDetector
from core.exceptions import ModelInferenceError


class AutoencoderNetwork(nn.Module):
    """Neural network architecture for autoencoder"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = None):
        """
        Initialize autoencoder network
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions (default: [64, 32, 16, 32, 64])
        """
        super(AutoencoderNetwork, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32, 16, 32, 64]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim
        
        # Encoder (compress)
        for i, hidden_dim in enumerate(hidden_dims[:len(hidden_dims)//2 + 1]):
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers[:-1])  # Remove last dropout
        
        # Build decoder layers
        decoder_layers = []
        
        # Decoder (reconstruct)
        for i, hidden_dim in enumerate(hidden_dims[len(hidden_dims)//2 + 1:]):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Final reconstruction layer
        decoder_layers.extend([
            nn.Linear(prev_dim, input_dim),
            nn.Sigmoid()  # Normalize output to [0, 1]
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        """Forward pass through autoencoder"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Get encoded representation"""
        return self.encoder(x)


class AutoencoderDetector(BaseDetector):
    """Autoencoder-based anomaly detector for network traffic"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = None, 
                 threshold_percentile: float = 95.0, device: str = None):
        """
        Initialize autoencoder detector
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: Hidden layer dimensions
            threshold_percentile: Percentile for anomaly threshold
            device: Device to run model on ('cpu' or 'cuda')
        """
        super().__init__("AutoencoderDetector", "1.0.0")
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [64, 32, 16, 32, 64]
        self.threshold_percentile = threshold_percentile
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize network
        self.network = AutoencoderNetwork(input_dim, self.hidden_dims)
        self.network.to(self.device)
        
        # Training parameters
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 100
        self.early_stopping_patience = 10
        
        # Anomaly detection threshold
        self.anomaly_threshold = None
        self.reconstruction_errors = []
        
        # Model state
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        self.logger = logging.getLogger(__name__)
    
    def train(self, training_data: np.ndarray, labels: np.ndarray = None) -> Dict[str, Any]:
        """
        Train the autoencoder on normal traffic patterns
        
        Args:
            training_data: Feature vectors for training (normal traffic only)
            labels: Not used for unsupervised learning (optional)
            
        Returns:
            Training metrics and statistics
        """
        try:
            self.logger.info(f"Starting autoencoder training with {len(training_data)} samples")
            
            # Validate input
            if len(training_data) == 0:
                raise ModelInferenceError("Empty training data provided")
            
            if training_data.shape[1] != self.input_dim:
                raise ModelInferenceError(
                    f"Input dimension mismatch: expected {self.input_dim}, got {training_data.shape[1]}"
                )
            
            # Normalize training data
            training_data = self._normalize_data(training_data)
            
            # Convert to PyTorch tensors
            train_tensor = torch.FloatTensor(training_data).to(self.device)
            train_dataset = TensorDataset(train_tensor, train_tensor)  # Autoencoder: input = target
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            # Training loop
            training_losses = []
            best_loss = float('inf')
            patience_counter = 0
            
            self.network.train()
            
            for epoch in range(self.num_epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                for batch_data, _ in train_loader:
                    # Forward pass
                    self.optimizer.zero_grad()
                    reconstructed = self.network(batch_data)
                    loss = self.criterion(reconstructed, batch_data)
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches
                training_losses.append(avg_loss)
                
                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.6f}")
            
            # Calculate reconstruction errors for threshold setting
            self.network.eval()
            reconstruction_errors = []
            
            with torch.no_grad():
                for batch_data, _ in train_loader:
                    reconstructed = self.network(batch_data)
                    errors = torch.mean((batch_data - reconstructed) ** 2, dim=1)
                    reconstruction_errors.extend(errors.cpu().numpy())
            
            self.reconstruction_errors = reconstruction_errors
            
            # Set anomaly threshold based on percentile
            self.anomaly_threshold = np.percentile(reconstruction_errors, self.threshold_percentile)
            
            self.is_trained = True
            
            # Training statistics
            training_stats = {
                "epochs_trained": len(training_losses),
                "final_loss": training_losses[-1],
                "best_loss": best_loss,
                "anomaly_threshold": self.anomaly_threshold,
                "training_samples": len(training_data),
                "reconstruction_error_stats": {
                    "mean": np.mean(reconstruction_errors),
                    "std": np.std(reconstruction_errors),
                    "min": np.min(reconstruction_errors),
                    "max": np.max(reconstruction_errors),
                    "percentiles": {
                        "50": np.percentile(reconstruction_errors, 50),
                        "90": np.percentile(reconstruction_errors, 90),
                        "95": np.percentile(reconstruction_errors, 95),
                        "99": np.percentile(reconstruction_errors, 99)
                    }
                }
            }
            
            self.logger.info(f"Training completed. Threshold: {self.anomaly_threshold:.6f}")
            return training_stats
            
        except Exception as e:
            raise ModelInferenceError(f"Training failed: {e}")
    
    def predict(self, features: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Make prediction on input features
        
        Args:
            features: Feature vector for prediction
            
        Returns:
            Tuple of (is_malicious, confidence_score, explanation)
        """
        try:
            if not self.is_trained:
                raise ModelInferenceError("Model must be trained before making predictions")
            
            # Validate input
            if not self.validate_features(features):
                raise ModelInferenceError("Invalid feature format")
            
            # Handle single sample vs batch
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            if features.shape[1] != self.input_dim:
                raise ModelInferenceError(
                    f"Input dimension mismatch: expected {self.input_dim}, got {features.shape[1]}"
                )
            
            # Normalize features
            features_normalized = self._normalize_data(features)
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(features_normalized).to(self.device)
            
            # Make prediction
            self.network.eval()
            with torch.no_grad():
                reconstructed = self.network(input_tensor)
                reconstruction_error = torch.mean((input_tensor - reconstructed) ** 2, dim=1)
                reconstruction_error = reconstruction_error.cpu().numpy()
            
            # Determine if anomalous
            is_malicious = bool(reconstruction_error[0] > self.anomaly_threshold)
            
            # Calculate confidence score (normalized reconstruction error)
            confidence = float(min(reconstruction_error[0] / (self.anomaly_threshold * 2), 1.0))
            
            # Create explanation
            explanation = {
                "reconstruction_error": float(reconstruction_error[0]),
                "anomaly_threshold": float(self.anomaly_threshold),
                "error_ratio": float(reconstruction_error[0] / self.anomaly_threshold),
                "model_type": "autoencoder",
                "feature_importance": self._calculate_feature_importance(
                    features_normalized[0], reconstructed[0].cpu().numpy()
                )
            }
            
            return is_malicious, confidence, explanation
            
        except Exception as e:
            raise ModelInferenceError(f"Prediction failed: {e}")
    
    def predict_batch(self, features_batch: np.ndarray) -> List[Tuple[bool, float, Dict[str, Any]]]:
        """
        Make predictions on a batch of features for efficiency
        
        Args:
            features_batch: Batch of feature vectors
            
        Returns:
            List of prediction tuples
        """
        try:
            if not self.is_trained:
                raise ModelInferenceError("Model must be trained before making predictions")
            
            if len(features_batch) == 0:
                return []
            
            # Normalize features
            features_normalized = self._normalize_data(features_batch)
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(features_normalized).to(self.device)
            
            # Make predictions
            self.network.eval()
            with torch.no_grad():
                reconstructed = self.network(input_tensor)
                reconstruction_errors = torch.mean((input_tensor - reconstructed) ** 2, dim=1)
                reconstruction_errors = reconstruction_errors.cpu().numpy()
            
            # Process results
            results = []
            for i, error in enumerate(reconstruction_errors):
                is_malicious = bool(error > self.anomaly_threshold)
                confidence = float(min(error / (self.anomaly_threshold * 2), 1.0))
                
                explanation = {
                    "reconstruction_error": float(error),
                    "anomaly_threshold": float(self.anomaly_threshold),
                    "error_ratio": float(error / self.anomaly_threshold),
                    "model_type": "autoencoder"
                }
                
                results.append((is_malicious, confidence, explanation))
            
            return results
            
        except Exception as e:
            raise ModelInferenceError(f"Batch prediction failed: {e}")
    
    def save_model(self, filepath: str) -> bool:
        """Save trained model to file"""
        try:
            if not self.is_trained:
                raise ModelInferenceError("Cannot save untrained model")
            
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model state
            model_state = {
                'network_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_params': {
                    'input_dim': self.input_dim,
                    'hidden_dims': self.hidden_dims,
                    'threshold_percentile': self.threshold_percentile,
                    'learning_rate': self.learning_rate,
                    'batch_size': self.batch_size,
                    'num_epochs': self.num_epochs
                },
                'anomaly_threshold': self.anomaly_threshold,
                'reconstruction_errors': self.reconstruction_errors,
                'is_trained': self.is_trained,
                'version': self.version,
                'saved_at': datetime.now().isoformat()
            }
            
            torch.save(model_state, filepath)
            self.logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model from file"""
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise ModelInferenceError(f"Model file not found: {filepath}")
            
            # Load model state
            model_state = torch.load(filepath, map_location=self.device, weights_only=False)
            
            # Restore model parameters
            self.input_dim = model_state['model_params']['input_dim']
            self.hidden_dims = model_state['model_params']['hidden_dims']
            self.threshold_percentile = model_state['model_params']['threshold_percentile']
            self.learning_rate = model_state['model_params']['learning_rate']
            self.batch_size = model_state['model_params']['batch_size']
            self.num_epochs = model_state['model_params']['num_epochs']
            
            # Recreate network with loaded parameters
            self.network = AutoencoderNetwork(self.input_dim, self.hidden_dims)
            self.network.to(self.device)
            self.network.load_state_dict(model_state['network_state_dict'])
            
            # Restore optimizer
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
            self.optimizer.load_state_dict(model_state['optimizer_state_dict'])
            
            # Restore other state
            self.anomaly_threshold = model_state['anomaly_threshold']
            self.reconstruction_errors = model_state['reconstruction_errors']
            self.is_trained = model_state['is_trained']
            
            self.logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to [0, 1] range"""
        # Simple min-max normalization
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        
        # Avoid division by zero
        data_range = data_max - data_min
        data_range[data_range == 0] = 1
        
        normalized = (data - data_min) / data_range
        return normalized
    
    def _calculate_feature_importance(self, original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance based on reconstruction errors"""
        feature_errors = np.abs(original - reconstructed)
        total_error = np.sum(feature_errors)
        
        if total_error == 0:
            return {}
        
        # Get top contributing features
        feature_importance = feature_errors / total_error
        top_indices = np.argsort(feature_importance)[-5:]  # Top 5 features
        
        importance_dict = {}
        for i, idx in enumerate(top_indices):
            importance_dict[f"feature_{idx}"] = float(feature_importance[idx])
        
        return importance_dict
    
    def get_reconstruction_statistics(self) -> Dict[str, Any]:
        """Get statistics about reconstruction errors"""
        if not self.reconstruction_errors:
            return {}
        
        errors = np.array(self.reconstruction_errors)
        return {
            "count": len(errors),
            "mean": float(np.mean(errors)),
            "std": float(np.std(errors)),
            "min": float(np.min(errors)),
            "max": float(np.max(errors)),
            "median": float(np.median(errors)),
            "percentiles": {
                "25": float(np.percentile(errors, 25)),
                "50": float(np.percentile(errors, 50)),
                "75": float(np.percentile(errors, 75)),
                "90": float(np.percentile(errors, 90)),
                "95": float(np.percentile(errors, 95)),
                "99": float(np.percentile(errors, 99))
            }
        }
    
    def update_threshold(self, new_percentile: float):
        """Update anomaly detection threshold"""
        if not self.reconstruction_errors:
            raise ModelInferenceError("No reconstruction errors available for threshold update")
        
        self.threshold_percentile = new_percentile
        self.anomaly_threshold = np.percentile(self.reconstruction_errors, new_percentile)
        self.logger.info(f"Threshold updated to {self.anomaly_threshold:.6f} ({new_percentile}th percentile)")