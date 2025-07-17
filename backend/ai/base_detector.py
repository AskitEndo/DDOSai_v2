"""
Base interface for all AI detection models
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_models import TrafficPacket, DetectionResult


class BaseDetector(ABC):
    """Abstract base class for all AI detection models"""
    
    def __init__(self, model_name: str, version: str = "1.0.0"):
        self.model_name = model_name
        self.version = version
        self.is_trained = False
        self.model_params = {}
    
    @abstractmethod
    def train(self, training_data: np.ndarray, labels: np.ndarray = None) -> Dict[str, Any]:
        """
        Train the detection model
        
        Args:
            training_data: Feature vectors for training
            labels: Ground truth labels (optional for unsupervised models)
            
        Returns:
            Training metrics and statistics
        """
        pass
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Make prediction on input features
        
        Args:
            features: Feature vector for prediction
            
        Returns:
            Tuple of (is_malicious, confidence_score, explanation)
        """
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> bool:
        """Save trained model to file"""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> bool:
        """Load trained model from file"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata"""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "is_trained": self.is_trained,
            "parameters": self.model_params
        }
    
    def validate_features(self, features: np.ndarray) -> bool:
        """Validate input feature format and dimensions"""
        if not isinstance(features, np.ndarray):
            return False
        if len(features.shape) != 1 and len(features.shape) != 2:
            return False
        return True


class ModelEnsemble:
    """Ensemble class for combining multiple detection models"""
    
    def __init__(self, models: List[BaseDetector], weights: List[float] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
    
    def predict_consensus(self, features: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Make consensus prediction using all models
        
        Args:
            features: Feature vector for prediction
            
        Returns:
            Tuple of (is_malicious, weighted_confidence, combined_explanation)
        """
        predictions = []
        confidences = []
        explanations = {}
        
        for i, model in enumerate(self.models):
            is_mal, conf, exp = model.predict(features)
            predictions.append(is_mal)
            confidences.append(conf)
            explanations[model.model_name] = exp
        
        # Weighted voting
        weighted_score = sum(
            pred * conf * weight 
            for pred, conf, weight in zip(predictions, confidences, self.weights)
        )
        
        final_confidence = sum(
            conf * weight 
            for conf, weight in zip(confidences, self.weights)
        )
        
        is_malicious = weighted_score > 0.5
        
        combined_explanation = {
            "individual_predictions": explanations,
            "weighted_score": weighted_score,
            "consensus_confidence": final_confidence,
            "model_weights": dict(zip([m.model_name for m in self.models], self.weights))
        }
        
        return is_malicious, final_confidence, combined_explanation