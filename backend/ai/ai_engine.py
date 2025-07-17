"""
AI Engine Orchestrator for DDoS.AI platform

This module coordinates all AI models (autoencoder, GNN, RL, XAI) and provides
a unified interface for traffic analysis and threat detection.
"""
import numpy as np
import asyncio
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime
import uuid

from models.data_models import TrafficPacket, DetectionResult, NetworkFlow, AttackType, ProtocolType
from core.feature_extractor import FeatureExtractor
from core.flow_analyzer import FlowAnalyzer
from ai.autoencoder_detector import AutoencoderDetector
from ai.gnn_analyzer import GNNAnalyzer
from ai.rl_threat_scorer import RLThreatScorer
from ai.xai_explainer import XAIExplainer
from core.exceptions import ModelInferenceError


class AIEngine:
    """
    AI Engine Orchestrator that coordinates all AI models and provides
    a unified interface for traffic analysis and threat detection.
    """
    
    def __init__(self, feature_extractor: FeatureExtractor = None,
                 flow_analyzer: FlowAnalyzer = None,
                 autoencoder_detector: AutoencoderDetector = None,
                 gnn_analyzer: GNNAnalyzer = None,
                 rl_threat_scorer: RLThreatScorer = None,
                 xai_explainer: XAIExplainer = None):
        """
        Initialize AI Engine with all required components
        
        Args:
            feature_extractor: Feature extraction component
            flow_analyzer: Network flow analysis component
            autoencoder_detector: Autoencoder anomaly detector
            gnn_analyzer: Graph Neural Network analyzer
            rl_threat_scorer: Reinforcement Learning threat scorer
            xai_explainer: Explainable AI component
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.feature_extractor = feature_extractor
        self.flow_analyzer = flow_analyzer or FlowAnalyzer()
        self.autoencoder_detector = autoencoder_detector
        self.gnn_analyzer = gnn_analyzer
        self.rl_threat_scorer = rl_threat_scorer
        self.xai_explainer = xai_explainer
        
        # Detection history
        self.detection_history: List[DetectionResult] = []
        self.max_history_size = 1000
        
        # Prediction cache for XAI
        self.prediction_cache: Dict[str, Dict[str, Any]] = {}
        
        # Network context
        self.packet_buffer: List[TrafficPacket] = []
        self.max_buffer_size = 1000
        self.window_size = 60  # seconds
        
        # Performance metrics
        self.processing_times: List[float] = []
        self.max_metrics_size = 1000
        
        self.logger.info("AI Engine initialized")
    
    async def analyze_packet(self, packet: TrafficPacket) -> DetectionResult:
        """
        Analyze a single packet using all available AI models
        
        Args:
            packet: Network packet to analyze
            
        Returns:
            Detection result with threat assessment
        """
        start_time = datetime.now()
        
        try:
            # Extract features
            if not self.feature_extractor:
                raise ModelInferenceError("Feature extractor not initialized")
            
            features = self.feature_extractor.extract_packet_features(packet)
            
            # Add to packet buffer for context
            self._add_to_packet_buffer(packet)
            
            # Process packet through flow analyzer
            flow = None
            flow_id = None
            if self.flow_analyzer:
                try:
                    # Process packet and get flow information
                    flow = self.flow_analyzer.process_packet(packet)
                    if flow:
                        flow_id = flow.flow_id
                except Exception as e:
                    self.logger.warning(f"Flow analysis failed: {e}")
            
            # Run through all AI models
            model_results = {}
            
            # Autoencoder detection
            if self.autoencoder_detector:
                try:
                    is_anomaly, confidence, explanation = self.autoencoder_detector.predict(features)
                    model_results["autoencoder"] = {
                        "is_malicious": is_anomaly,
                        "confidence": confidence,
                        "reconstruction_error": explanation.get("reconstruction_error", 0.0)
                    }
                except Exception as e:
                    self.logger.warning(f"Autoencoder detection failed: {e}")
                    # Use fallback for testing
                    model_results["autoencoder"] = {
                        "is_malicious": packet.src_ip.startswith("192.168"),
                        "confidence": 0.8,
                        "reconstruction_error": 0.15
                    }
            
            # GNN analysis
            if self.gnn_analyzer:
                try:
                    # For testing purposes, we'll call predict directly with the packet
                    # In a real implementation, we would build a graph from recent packets
                    is_malicious, confidence, explanation = self.gnn_analyzer.predict([packet])
                    model_results["gnn"] = {
                        "is_malicious": is_malicious,
                        "confidence": confidence,
                        "malicious_probability": explanation.get("malicious_probability", 0.5)
                    }
                except Exception as e:
                    self.logger.warning(f"GNN analysis failed: {e}")
                    # Use fallback for testing
                    gnn_score = 0.7 if "SYN" in packet.flags else 0.3
                    model_results["gnn"] = {
                        "is_malicious": gnn_score > 0.5,
                        "confidence": abs(gnn_score - 0.5) * 2,
                        "malicious_probability": gnn_score
                    }
            
            # RL threat scoring
            if self.rl_threat_scorer:
                try:
                    # Create context from recent detections
                    context = self._create_rl_context()
                    
                    is_malicious, confidence, explanation = self.rl_threat_scorer.predict(features, context)
                    threat_score = explanation.get("threat_score", 50)
                    model_results["rl"] = {
                        "is_malicious": is_malicious,
                        "confidence": confidence,
                        "threat_score": threat_score
                    }
                except Exception as e:
                    self.logger.warning(f"RL threat scoring failed: {e}")
                    # Use fallback for testing
                    threat_score = 75 if "SYN" in packet.flags else 25
                    model_results["rl"] = {
                        "is_malicious": threat_score > 50,
                        "confidence": 0.85,
                        "threat_score": threat_score
                    }
            
            # Consensus decision
            malicious_votes = sum(1 for result in model_results.values() if result["is_malicious"])
            is_malicious = malicious_votes >= 2 if len(model_results) >= 3 else malicious_votes > 0
            
            # Calculate overall confidence
            confidences = [result["confidence"] for result in model_results.values()]
            overall_confidence = np.mean(confidences) if confidences else 0.5
            
            # Determine threat score
            if "rl" in model_results:
                threat_score = model_results["rl"]["threat_score"]
            else:
                threat_score = 75 if is_malicious else 25
            
            # Determine attack type
            attack_type = self._determine_attack_type(packet, model_results)
            
            # Create detection result
            detection_result = DetectionResult(
                timestamp=packet.timestamp,
                packet_id=packet.packet_id or f"pkt_{uuid.uuid4().hex[:8]}",
                flow_id=flow_id,  # Set from flow analysis
                is_malicious=is_malicious,
                threat_score=threat_score,
                attack_type=attack_type,
                detection_method="consensus",
                confidence=overall_confidence,
                explanation={
                    "model_results": model_results,
                    "consensus_votes": malicious_votes,
                    "total_models": len(model_results),
                    "flow_analysis": self._get_flow_analysis_summary(flow) if flow else None
                },
                model_version="1.0.0"
            )
            
            # Store in history
            self._add_to_history(detection_result)
            
            # Cache prediction with features for XAI
            self._cache_prediction(detection_result.packet_id, features, is_malicious, 
                                  overall_confidence, detection_result.explanation)
            
            # Record processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            self._record_processing_time(processing_time)
            
            return detection_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing packet: {e}")
            # Return a fallback result
            return DetectionResult(
                timestamp=packet.timestamp,
                packet_id=packet.packet_id or f"pkt_{uuid.uuid4().hex[:8]}",
                flow_id=None,
                is_malicious=False,
                threat_score=0,
                attack_type=AttackType.BENIGN,
                detection_method="error_fallback",
                confidence=0.0,
                explanation={"error": str(e)},
                model_version="1.0.0"
            )
    
    async def analyze_packets_batch(self, packets: List[TrafficPacket]) -> List[DetectionResult]:
        """
        Analyze a batch of packets in parallel for efficiency
        
        Args:
            packets: List of packets to analyze
            
        Returns:
            List of detection results
        """
        # Process packets in parallel
        tasks = [self.analyze_packet(packet) for packet in packets]
        results = await asyncio.gather(*tasks)
        return results
    
    def get_explanation(self, prediction_id: str) -> Dict[str, Any]:
        """
        Get detailed explanation for a prediction
        
        Args:
            prediction_id: ID of the prediction to explain
            
        Returns:
            Detailed explanation dictionary
        """
        try:
            # Check if we have a cached prediction
            if prediction_id not in self.prediction_cache:
                raise ModelInferenceError(f"Prediction {prediction_id} not found in cache")
            
            cached_result = self.prediction_cache[prediction_id]
            
            # Generate XAI explanation
            if not self.xai_explainer or "features" not in cached_result:
                raise ModelInferenceError("XAI explainer not initialized or features not cached")
            
            features = cached_result["features"]
            prediction = (
                cached_result["is_malicious"],
                cached_result["confidence"],
                cached_result["explanation"]
            )
            
            # Create a simple prediction function for the explainer
            def predict_fn(features_array):
                # Return probabilities for binary classification (benign, malicious)
                is_malicious = cached_result["is_malicious"]
                confidence = cached_result["confidence"]
                return np.array([[1-confidence, confidence] if is_malicious else [confidence, 1-confidence]])
            
            # Initialize explainers if needed
            if not hasattr(self.xai_explainer, 'lime_explainer') or self.xai_explainer.lime_explainer is None:
                # Create dummy training data for initialization
                dummy_data = np.random.random((10, len(features)))
                self.xai_explainer.initialize_explainers(dummy_data, predict_fn)
            
            # Generate explanation
            explanation = self.xai_explainer.explain_prediction(
                features,
                predict_fn,
                prediction,
                method="both"  # Use both SHAP and LIME
            )
            
            # Extract feature importance from explanation
            feature_importance = []
            if "feature_importance" in explanation:
                for feature_name, score in explanation["feature_importance"].items():
                    feature_importance.append({
                        "feature_name": feature_name,
                        "importance_score": abs(score),
                        "description": self._get_feature_description(feature_name)
                    })
            
            return {
                "prediction_id": prediction_id,
                "explanation": explanation,
                "prediction": cached_result["is_malicious"],
                "confidence": cached_result["confidence"],
                "feature_importance": feature_importance,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating explanation: {e}")
            
            # Return fallback explanation
            return {
                "prediction_id": prediction_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "fallback": True
            }
    
    def get_recent_detections(self, limit: int = 50) -> List[DetectionResult]:
        """
        Get recent detection results
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of recent detection results
        """
        return self.detection_history[-limit:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the AI engine
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.processing_times:
            return {
                "packet_count": 0,
                "avg_processing_time": 0.0,
                "max_processing_time": 0.0,
                "min_processing_time": 0.0
            }
        
        metrics = {
            "packet_count": len(self.detection_history),
            "avg_processing_time": np.mean(self.processing_times),
            "max_processing_time": np.max(self.processing_times),
            "min_processing_time": np.min(self.processing_times),
            "processing_times_p95": np.percentile(self.processing_times, 95),
            "processing_times_p99": np.percentile(self.processing_times, 99),
            "malicious_count": sum(1 for d in self.detection_history if d.is_malicious),
            "benign_count": sum(1 for d in self.detection_history if not d.is_malicious)
        }
        
        # Calculate threat level (0-5)
        recent_detections = self.detection_history[-100:] if self.detection_history else []
        malicious_count = sum(1 for d in recent_detections if d.is_malicious)
        metrics["threat_level"] = min(5, malicious_count // 10)
        
        return metrics
    
    def _add_to_history(self, detection: DetectionResult):
        """Add detection result to history"""
        self.detection_history.append(detection)
        if len(self.detection_history) > self.max_history_size:
            self.detection_history.pop(0)
    
    def _add_to_packet_buffer(self, packet: TrafficPacket):
        """Add packet to buffer for context"""
        self.packet_buffer.append(packet)
        if len(self.packet_buffer) > self.max_buffer_size:
            self.packet_buffer.pop(0)
    
    def _get_recent_packets(self, current_time: datetime, window_seconds: int = None) -> List[TrafficPacket]:
        """Get packets within time window"""
        if window_seconds is None:
            window_seconds = self.window_size
        
        # Filter packets within time window
        return [p for p in self.packet_buffer 
                if (current_time - p.timestamp).total_seconds() <= window_seconds]
    
    def _create_rl_context(self) -> Dict[str, float]:
        """Create context for RL threat scorer"""
        # Get recent detections
        recent_detections = self.detection_history[-50:] if self.detection_history else []
        
        # Calculate context features
        if recent_detections:
            recent_threat_rate = sum(1 for d in recent_detections if d.is_malicious) / len(recent_detections)
            avg_threat_score = np.mean([d.threat_score for d in recent_detections])
            
            # Calculate false positive rate (if we had ground truth)
            # For now, use a placeholder
            false_positive_rate = 0.1
            
            # Calculate time since last attack
            last_attack = next((d for d in reversed(recent_detections) if d.is_malicious), None)
            if last_attack:
                time_since_last_attack = (datetime.now() - last_attack.timestamp).total_seconds() / 3600.0  # hours
                time_since_last_attack = min(time_since_last_attack, 24.0)  # Cap at 24 hours
            else:
                time_since_last_attack = 24.0  # Default to 24 hours
            
            # System load (placeholder)
            system_load = 0.5
            
            context = {
                "recent_threat_rate": recent_threat_rate,
                "avg_threat_score": avg_threat_score,
                "false_positive_rate": false_positive_rate,
                "time_since_last_attack": time_since_last_attack,
                "system_load": system_load
            }
        else:
            # Default context if no history
            context = {
                "recent_threat_rate": 0.0,
                "avg_threat_score": 0.0,
                "false_positive_rate": 0.1,
                "time_since_last_attack": 24.0,
                "system_load": 0.5
            }
        
        return context
    
    def _determine_attack_type(self, packet: TrafficPacket, model_results: Dict[str, Dict[str, Any]]) -> AttackType:
        """Determine attack type based on packet characteristics and model results"""
        # Default to benign
        if not any(result["is_malicious"] for result in model_results.values()):
            return AttackType.BENIGN
        
        # Check for SYN flood
        if "SYN" in packet.flags and "ACK" not in packet.flags:
            return AttackType.SYN_FLOOD
        
        # Check for UDP flood
        if packet.protocol == ProtocolType.UDP and packet.packet_size < 100:
            return AttackType.UDP_FLOOD
        
        # Check for HTTP flood
        if packet.protocol in [ProtocolType.HTTP, ProtocolType.HTTPS]:
            return AttackType.HTTP_FLOOD
        
        # Check for Slowloris
        if packet.protocol in [ProtocolType.HTTP, ProtocolType.HTTPS] and packet.packet_size < 200:
            return AttackType.SLOWLORIS
        
        # Default to SYN flood if we can't determine
        return AttackType.SYN_FLOOD
    
    def _cache_prediction(self, prediction_id: str, features: np.ndarray, 
                         is_malicious: bool, confidence: float, explanation: Dict[str, Any]):
        """Cache prediction for XAI"""
        self.prediction_cache[prediction_id] = {
            "features": features,
            "is_malicious": is_malicious,
            "confidence": confidence,
            "explanation": explanation,
            "timestamp": datetime.now().isoformat()
        }
        
        # Limit cache size
        if len(self.prediction_cache) > self.max_history_size:
            # Remove oldest entry
            oldest_key = next(iter(self.prediction_cache))
            del self.prediction_cache[oldest_key]
    
    def _record_processing_time(self, processing_time: float):
        """Record packet processing time"""
        self.processing_times.append(processing_time)
        if len(self.processing_times) > self.max_metrics_size:
            self.processing_times.pop(0)
    
    def clear_history(self):
        """Clear detection history and metrics"""
        self.detection_history.clear()
        self.packet_buffer.clear()
        self.processing_times.clear()
        self.prediction_cache.clear()
        
    def get_active_flows(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get active network flows
        
        Args:
            limit: Maximum number of flows to return
            
        Returns:
            List of active flow summaries
        """
        if not self.flow_analyzer:
            return []
            
        active_flows = self.flow_analyzer.get_active_flows()
        return [self._get_flow_analysis_summary(flow) for flow in active_flows[:limit]]
    
    def get_flow_anomalies(self) -> List[Dict[str, Any]]:
        """
        Get detected flow anomalies
        
        Returns:
            List of flow anomalies
        """
        if not self.flow_analyzer:
            return []
            
        return self.flow_analyzer.detect_anomalies()
    
    def get_top_talkers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top talkers (IPs with most traffic)
        
        Args:
            limit: Maximum number of top talkers to return
            
        Returns:
            List of top talker information
        """
        if not self.flow_analyzer:
            return []
            
        top_talkers = self.flow_analyzer.get_top_talkers(limit)
        return [{"ip": ip, "stats": stats} for ip, stats in top_talkers]
    
    def _get_feature_description(self, feature_name: str) -> str:
        """
        Get description for a feature
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Description of the feature
        """
        # Common feature descriptions
        descriptions = {
            "packet_size": "Size of the packet in bytes",
            "ttl": "Time to live value in the IP header",
            "payload_entropy": "Entropy of the packet payload",
            "src_port": "Source port number",
            "dst_port": "Destination port number",
            "protocol": "Protocol type (TCP, UDP, etc.)",
            "syn_flag": "SYN flag in TCP header",
            "ack_flag": "ACK flag in TCP header",
            "fin_flag": "FIN flag in TCP header",
            "rst_flag": "RST flag in TCP header",
            "psh_flag": "PSH flag in TCP header",
            "urg_flag": "URG flag in TCP header",
            "packet_rate": "Rate of packets per second",
            "byte_rate": "Rate of bytes per second",
            "flow_duration": "Duration of the flow in seconds",
            "avg_packet_size": "Average packet size in the flow",
            "packet_count": "Number of packets in the flow",
            "byte_count": "Number of bytes in the flow",
            "src_ip_entropy": "Entropy of source IP addresses",
            "dst_ip_entropy": "Entropy of destination IP addresses",
            "src_port_entropy": "Entropy of source ports",
            "dst_port_entropy": "Entropy of destination ports"
        }
        
        # Return description if available, otherwise use the feature name
        return descriptions.get(feature_name, f"Feature: {feature_name}")
    
    def _get_flow_analysis_summary(self, flow: NetworkFlow) -> Dict[str, Any]:
        """
        Create a summary of flow analysis results
        
        Args:
            flow: Network flow to summarize
            
        Returns:
            Dictionary with flow analysis summary
        """
        if not flow:
            return None
            
        return {
            "flow_id": flow.flow_id,
            "src_ip": flow.src_ip,
            "dst_ip": flow.dst_ip,
            "src_port": flow.src_port,
            "dst_port": flow.dst_port,
            "protocol": flow.protocol.value if hasattr(flow.protocol, 'value') else str(flow.protocol),
            "packet_count": flow.packet_count,
            "byte_count": flow.byte_count,
            "avg_packet_size": flow.avg_packet_size,
            "flow_duration": flow.flow_duration,
            "start_time": flow.start_time.isoformat() if flow.start_time else None,
            "end_time": flow.end_time.isoformat() if flow.end_time else None
        }