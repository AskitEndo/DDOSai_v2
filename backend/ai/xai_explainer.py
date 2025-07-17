"""
Explainable AI module for model interpretability and transparency
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Callable
import logging
from datetime import datetime
import json

# XAI libraries
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_models import TrafficPacket, DetectionResult
from core.exceptions import ModelInferenceError


class XAIExplainer:
    """Explainable AI module for model interpretability"""
    
    def __init__(self, feature_names: List[str] = None, model_type: str = "ensemble"):
        """
        Initialize XAI explainer
        
        Args:
            feature_names: Names of input features
            model_type: Type of model being explained
        """
        self.feature_names = feature_names or []
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)
        
        # SHAP explainer (will be initialized when needed)
        self.shap_explainer = None
        
        # LIME explainer (will be initialized when needed)
        self.lime_explainer = None
        
        # Training data for background (needed for SHAP)
        self.background_data = None
        
        # Feature statistics for analysis
        self.feature_stats = {}
    
    def initialize_explainers(self, training_data: np.ndarray, 
                            model_predict_fn: Callable[[np.ndarray], np.ndarray]):
        """
        Initialize SHAP and LIME explainers with training data
        
        Args:
            training_data: Training data for background distribution
            model_predict_fn: Model prediction function
        """
        try:
            self.logger.info("Initializing XAI explainers...")
            
            # Store background data
            self.background_data = training_data
            
            # Calculate feature statistics
            self._calculate_feature_stats(training_data)
            
            # Initialize SHAP explainer
            # Use a subset of training data as background for efficiency
            background_size = min(100, len(training_data))
            background_sample = training_data[np.random.choice(
                len(training_data), background_size, replace=False
            )]
            
            self.shap_explainer = shap.Explainer(
                model_predict_fn, 
                background_sample,
                feature_names=self.feature_names
            )
            
            # Initialize LIME explainer
            self.lime_explainer = LimeTabularExplainer(
                training_data,
                feature_names=self.feature_names,
                class_names=['Benign', 'Malicious'],
                mode='classification',
                discretize_continuous=True
            )
            
            self.logger.info("XAI explainers initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize explainers: {e}")
            raise ModelInferenceError(f"XAI initialization failed: {e}")
    
    def explain_prediction(self, features: np.ndarray, 
                         model_predict_fn: Callable[[np.ndarray], np.ndarray],
                         prediction_result: Tuple[bool, float, Dict[str, Any]],
                         method: str = "both") -> Dict[str, Any]:
        """
        Generate explanation for a model prediction
        
        Args:
            features: Input features for the prediction
            model_predict_fn: Model prediction function
            prediction_result: Original prediction result (is_malicious, confidence, explanation)
            method: Explanation method ("shap", "lime", or "both")
            
        Returns:
            Comprehensive explanation dictionary
        """
        try:
            if self.shap_explainer is None or self.lime_explainer is None:
                raise ModelInferenceError("Explainers not initialized. Call initialize_explainers first.")
            
            # Handle single sample
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            is_malicious, confidence, original_explanation = prediction_result
            
            explanation = {
                "prediction": {
                    "is_malicious": is_malicious,
                    "confidence": confidence,
                    "threat_score": original_explanation.get("threat_score", 0),
                    "model_type": self.model_type
                },
                "timestamp": datetime.now().isoformat(),
                "feature_importance": {},
                "explanations": {}
            }
            
            # Generate SHAP explanation
            if method in ["shap", "both"]:
                shap_explanation = self._generate_shap_explanation(features[0], model_predict_fn)
                explanation["explanations"]["shap"] = shap_explanation
                explanation["feature_importance"].update(shap_explanation["feature_importance"])
            
            # Generate LIME explanation
            if method in ["lime", "both"]:
                lime_explanation = self._generate_lime_explanation(features[0], model_predict_fn)
                explanation["explanations"]["lime"] = lime_explanation
                
                # Combine feature importance (average SHAP and LIME if both available)
                if method == "both":
                    explanation["feature_importance"] = self._combine_feature_importance(
                        explanation["explanations"]["shap"]["feature_importance"],
                        lime_explanation["feature_importance"]
                    )
                else:
                    explanation["feature_importance"] = lime_explanation["feature_importance"]
            
            # Add top influential features
            explanation["top_features"] = self._get_top_features(
                explanation["feature_importance"], top_k=5
            )
            
            # Generate counterfactuals
            explanation["counterfactuals"] = self._generate_counterfactuals(
                features[0], model_predict_fn, is_malicious
            )
            
            # Add decision path
            explanation["decision_path"] = self._generate_decision_path(
                features[0], explanation["top_features"], is_malicious
            )
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Failed to generate explanation: {e}")
            raise ModelInferenceError(f"Explanation generation failed: {e}")
    
    def _generate_shap_explanation(self, features: np.ndarray, 
                                 model_predict_fn: Callable) -> Dict[str, Any]:
        """Generate SHAP-based explanation"""
        try:
            # Get SHAP values
            shap_values = self.shap_explainer(features.reshape(1, -1))
            
            # Extract values for the positive class (malicious)
            if hasattr(shap_values, 'values') and len(shap_values.values.shape) > 2:
                # Multi-class output
                values = shap_values.values[0, :, 1]  # Malicious class
            else:
                # Binary output
                values = shap_values.values[0] if hasattr(shap_values, 'values') else shap_values[0]
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, importance in enumerate(values):
                feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
                feature_importance[feature_name] = float(importance)
            
            return {
                "method": "SHAP",
                "feature_importance": feature_importance,
                "base_value": float(shap_values.base_values[0]) if hasattr(shap_values, 'base_values') else 0.0,
                "explanation": "SHAP values represent the contribution of each feature to the prediction"
            }
            
        except Exception as e:
            self.logger.warning(f"SHAP explanation failed: {e}")
            return {
                "method": "SHAP",
                "feature_importance": {},
                "error": str(e)
            }
    
    def _generate_lime_explanation(self, features: np.ndarray, 
                                 model_predict_fn: Callable) -> Dict[str, Any]:
        """Generate LIME-based explanation"""
        try:
            # Generate LIME explanation
            explanation = self.lime_explainer.explain_instance(
                features,
                model_predict_fn,
                num_features=len(self.feature_names),
                top_labels=2
            )
            
            # Extract feature importance for malicious class (class 1)
            feature_importance = {}
            for feature_idx, importance in explanation.as_list(label=1):
                if isinstance(feature_idx, str):
                    feature_name = feature_idx
                else:
                    feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f"feature_{feature_idx}"
                feature_importance[feature_name] = float(importance)
            
            return {
                "method": "LIME",
                "feature_importance": feature_importance,
                "local_prediction": explanation.local_pred[1] if len(explanation.local_pred) > 1 else explanation.local_pred[0],
                "explanation": "LIME values show local feature importance for this specific prediction"
            }
            
        except Exception as e:
            self.logger.warning(f"LIME explanation failed: {e}")
            return {
                "method": "LIME",
                "feature_importance": {},
                "error": str(e)
            }
    
    def _combine_feature_importance(self, shap_importance: Dict[str, float], 
                                  lime_importance: Dict[str, float]) -> Dict[str, float]:
        """Combine SHAP and LIME feature importance scores"""
        combined = {}
        all_features = set(shap_importance.keys()) | set(lime_importance.keys())
        
        for feature in all_features:
            shap_val = shap_importance.get(feature, 0.0)
            lime_val = lime_importance.get(feature, 0.0)
            
            # Average the two importance scores
            combined[feature] = (shap_val + lime_val) / 2.0
        
        return combined
    
    def _get_top_features(self, feature_importance: Dict[str, float], 
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """Get top-k most important features"""
        # Sort features by absolute importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        top_features = []
        for i, (feature_name, importance) in enumerate(sorted_features[:top_k]):
            feature_info = {
                "rank": i + 1,
                "feature_name": feature_name,
                "importance_score": importance,
                "direction": "increases" if importance > 0 else "decreases",
                "description": self._get_feature_description(feature_name)
            }
            top_features.append(feature_info)
        
        return top_features
    
    def _get_feature_description(self, feature_name: str) -> str:
        """Get human-readable description of a feature"""
        descriptions = {
            "packet_size_norm": "Normalized packet size",
            "ttl_norm": "Normalized time-to-live value",
            "payload_entropy": "Randomness of packet payload",
            "proto_tcp": "TCP protocol indicator",
            "proto_udp": "UDP protocol indicator",
            "proto_http": "HTTP protocol indicator",
            "flag_syn": "SYN flag presence",
            "flag_ack": "ACK flag presence",
            "src_private": "Source IP is private",
            "dst_private": "Destination IP is private",
            "src_well_known": "Source port is well-known",
            "dst_well_known": "Destination port is well-known",
            "hour_norm": "Normalized hour of day",
            "is_weekend": "Weekend time indicator",
            "recent_threat_rate": "Recent threat detection rate",
            "system_load": "Current system load",
            "false_positive_rate": "Recent false positive rate"
        }
        
        return descriptions.get(feature_name, f"Feature: {feature_name}")
    
    def _generate_counterfactuals(self, features: np.ndarray, 
                                model_predict_fn: Callable,
                                current_prediction: bool) -> List[Dict[str, Any]]:
        """Generate counterfactual explanations"""
        try:
            counterfactuals = []
            
            # Get current prediction probability
            current_prob = model_predict_fn(features.reshape(1, -1))[0]
            if hasattr(current_prob, '__len__') and len(current_prob) > 1:
                current_prob = current_prob[1]  # Malicious class probability
            
            # Try modifying top features to see what would change the prediction
            if hasattr(self, 'feature_stats') and self.feature_stats:
                for i, feature_name in enumerate(self.feature_names[:5]):  # Top 5 features
                    if feature_name in self.feature_stats:
                        stats = self.feature_stats[feature_name]
                        
                        # Try different values for this feature
                        for multiplier in [0.5, 1.5, 2.0]:
                            modified_features = features.copy()
                            
                            # Modify the feature value
                            if stats['std'] > 0:
                                new_value = stats['mean'] + multiplier * stats['std']
                                modified_features[i] = new_value
                                
                                # Get prediction for modified features
                                new_prob = model_predict_fn(modified_features.reshape(1, -1))[0]
                                if hasattr(new_prob, '__len__') and len(new_prob) > 1:
                                    new_prob = new_prob[1]
                                
                                new_prediction = new_prob > 0.5
                                
                                # If prediction changed, add as counterfactual
                                if new_prediction != current_prediction:
                                    counterfactuals.append({
                                        "feature": feature_name,
                                        "original_value": float(features[i]),
                                        "counterfactual_value": float(new_value),
                                        "original_prediction": current_prediction,
                                        "counterfactual_prediction": new_prediction,
                                        "probability_change": float(new_prob - current_prob)
                                    })
                                    
                                    if len(counterfactuals) >= 3:  # Limit to 3 counterfactuals
                                        break
                    
                    if len(counterfactuals) >= 3:
                        break
            
            return counterfactuals
            
        except Exception as e:
            self.logger.warning(f"Counterfactual generation failed: {e}")
            return []
    
    def _generate_decision_path(self, features: np.ndarray, 
                              top_features: List[Dict[str, Any]],
                              prediction: bool) -> List[str]:
        """Generate human-readable decision path"""
        try:
            path = []
            
            # Start with overall assessment
            if prediction:
                path.append("Traffic classified as MALICIOUS based on the following factors:")
            else:
                path.append("Traffic classified as BENIGN based on the following factors:")
            
            # Add top contributing features
            for feature_info in top_features[:3]:  # Top 3 features
                feature_name = feature_info["feature_name"]
                importance = feature_info["importance_score"]
                direction = feature_info["direction"]
                description = feature_info["description"]
                
                if abs(importance) > 0.01:  # Only include significant features
                    if importance > 0:
                        path.append(f"• {description} strongly indicates malicious activity")
                    else:
                        path.append(f"• {description} suggests benign behavior")
            
            # Add confidence assessment
            if len(top_features) > 0:
                max_importance = max(abs(f["importance_score"]) for f in top_features)
                if max_importance > 0.5:
                    path.append("High confidence in prediction due to strong feature signals")
                elif max_importance > 0.2:
                    path.append("Moderate confidence in prediction")
                else:
                    path.append("Low confidence - features show weak signals")
            
            return path
            
        except Exception as e:
            self.logger.warning(f"Decision path generation failed: {e}")
            return ["Unable to generate decision path"]
    
    def _calculate_feature_stats(self, training_data: np.ndarray):
        """Calculate feature statistics from training data"""
        try:
            self.feature_stats = {}
            
            for i in range(training_data.shape[1]):
                feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
                
                feature_values = training_data[:, i]
                self.feature_stats[feature_name] = {
                    'mean': float(np.mean(feature_values)),
                    'std': float(np.std(feature_values)),
                    'min': float(np.min(feature_values)),
                    'max': float(np.max(feature_values)),
                    'median': float(np.median(feature_values))
                }
                
        except Exception as e:
            self.logger.warning(f"Feature statistics calculation failed: {e}")
    
    def generate_global_explanation(self, model_predict_fn: Callable,
                                  sample_data: np.ndarray = None,
                                  num_samples: int = 100) -> Dict[str, Any]:
        """
        Generate global model explanation using feature importance
        
        Args:
            model_predict_fn: Model prediction function
            sample_data: Sample data for analysis (uses background if None)
            num_samples: Number of samples to analyze
            
        Returns:
            Global explanation dictionary
        """
        try:
            if sample_data is None:
                if self.background_data is None:
                    raise ModelInferenceError("No sample data available for global explanation")
                sample_data = self.background_data
            
            # Sample subset for efficiency
            if len(sample_data) > num_samples:
                indices = np.random.choice(len(sample_data), num_samples, replace=False)
                sample_data = sample_data[indices]
            
            # Calculate SHAP values for all samples
            if self.shap_explainer is None:
                raise ModelInferenceError("SHAP explainer not initialized")
            
            shap_values = self.shap_explainer(sample_data)
            
            # Extract values for analysis
            if hasattr(shap_values, 'values') and len(shap_values.values.shape) > 2:
                values = shap_values.values[:, :, 1]  # Malicious class
            else:
                values = shap_values.values if hasattr(shap_values, 'values') else shap_values
            
            # Calculate global feature importance
            global_importance = np.mean(np.abs(values), axis=0)
            
            # Create global explanation
            global_explanation = {
                "model_type": self.model_type,
                "analysis_timestamp": datetime.now().isoformat(),
                "samples_analyzed": len(sample_data),
                "global_feature_importance": {},
                "feature_statistics": self.feature_stats,
                "model_behavior": {}
            }
            
            # Add global feature importance
            for i, importance in enumerate(global_importance):
                feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
                global_explanation["global_feature_importance"][feature_name] = float(importance)
            
            # Add model behavior analysis
            predictions = model_predict_fn(sample_data)
            if hasattr(predictions[0], '__len__') and len(predictions[0]) > 1:
                malicious_probs = [pred[1] for pred in predictions]
            else:
                malicious_probs = predictions
            
            global_explanation["model_behavior"] = {
                "average_malicious_probability": float(np.mean(malicious_probs)),
                "prediction_distribution": {
                    "benign_predictions": int(np.sum(np.array(malicious_probs) < 0.5)),
                    "malicious_predictions": int(np.sum(np.array(malicious_probs) >= 0.5))
                },
                "confidence_statistics": {
                    "mean_confidence": float(np.mean(np.abs(np.array(malicious_probs) - 0.5) * 2)),
                    "high_confidence_predictions": int(np.sum(np.abs(np.array(malicious_probs) - 0.5) > 0.4))
                }
            }
            
            return global_explanation
            
        except Exception as e:
            self.logger.error(f"Global explanation generation failed: {e}")
            raise ModelInferenceError(f"Global explanation failed: {e}")
    
    def export_explanation(self, explanation: Dict[str, Any], 
                         format: str = "json", filepath: str = None) -> str:
        """
        Export explanation to file
        
        Args:
            explanation: Explanation dictionary to export
            format: Export format ("json" or "html")
            filepath: Output file path (optional)
            
        Returns:
            Exported content as string
        """
        try:
            if format == "json":
                content = json.dumps(explanation, indent=2, default=str)
                if filepath:
                    with open(filepath, 'w') as f:
                        f.write(content)
                return content
            
            elif format == "html":
                content = self._generate_html_explanation(explanation)
                if filepath:
                    with open(filepath, 'w') as f:
                        f.write(content)
                return content
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            self.logger.error(f"Explanation export failed: {e}")
            raise ModelInferenceError(f"Export failed: {e}")
    
    def _generate_html_explanation(self, explanation: Dict[str, Any]) -> str:
        """Generate HTML representation of explanation"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DDoS.AI Prediction Explanation</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                .prediction {{ font-size: 18px; font-weight: bold; }}
                .malicious {{ color: #d32f2f; }}
                .benign {{ color: #388e3c; }}
                .feature {{ margin: 10px 0; padding: 10px; border-left: 3px solid #2196f3; }}
                .importance {{ font-weight: bold; }}
                .counterfactual {{ background-color: #fff3e0; padding: 10px; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>DDoS.AI Prediction Explanation</h1>
                <p>Generated: {explanation.get('timestamp', 'Unknown')}</p>
            </div>
            
            <div class="prediction {'malicious' if explanation['prediction']['is_malicious'] else 'benign'}">
                Prediction: {'MALICIOUS' if explanation['prediction']['is_malicious'] else 'BENIGN'}
                (Confidence: {explanation['prediction']['confidence']:.2f})
            </div>
            
            <h2>Top Contributing Features</h2>
        """
        
        for feature in explanation.get('top_features', []):
            html += f"""
            <div class="feature">
                <div class="importance">#{feature['rank']} {feature['feature_name']}</div>
                <div>Importance: {feature['importance_score']:.3f}</div>
                <div>{feature['description']}</div>
            </div>
            """
        
        html += "<h2>Decision Path</h2><ul>"
        for step in explanation.get('decision_path', []):
            html += f"<li>{step}</li>"
        html += "</ul>"
        
        if explanation.get('counterfactuals'):
            html += "<h2>What Would Change the Prediction?</h2>"
            for cf in explanation['counterfactuals']:
                html += f"""
                <div class="counterfactual">
                    If {cf['feature']} changed from {cf['original_value']:.3f} to {cf['counterfactual_value']:.3f},
                    prediction would be {'MALICIOUS' if cf['counterfactual_prediction'] else 'BENIGN'}
                </div>
                """
        
        html += "</body></html>"
        return html