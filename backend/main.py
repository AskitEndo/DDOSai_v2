"""
Main FastAPI application for DDoS.AI backend
"""
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocketState
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn
import asyncio
import logging
import traceback
from datetime import datetime
import json
import uuid
import os
from typing import List, Dict, Any, Optional
import numpy as np

from models.data_models import (
    TrafficPacket, NetworkFlow, DetectionResult, 
    NetworkNode, NetworkEdge, ProtocolType, AttackType
)
from core.config import config
from core.exceptions import DDoSAIException, ErrorHandler, get_exception_handler
from core.middleware import RequestLoggingMiddleware, RateLimitingMiddleware, CircuitBreakerMiddleware
from core.logging_config import configure_logging, StructuredLogger
from core.decorators import handle_errors, retry_operation, log_execution_time
from core.monitoring import MonitoringService
from core.feature_extractor import FeatureExtractor
from ai.autoencoder_detector import AutoencoderDetector
from ai.gnn_analyzer import GNNAnalyzer
from ai.rl_threat_scorer import RLThreatScorer
from ai.xai_explainer import XAIExplainer
from api_docs import configure_api_docs, tags_metadata

# Configure logging
configure_logging(
    log_level=os.environ.get("LOG_LEVEL", "INFO"),
    json_logs=os.environ.get("JSON_LOGS", "false").lower() == "true",
    log_file=os.environ.get("LOG_FILE")
)
logger = StructuredLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DDoS.AI API",
    description="AI-Powered DDoS Detection Platform API",
    version="1.0.0"
)

# Configure API documentation
app = configure_api_docs(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(
    CircuitBreakerMiddleware,
    failure_threshold=5,
    reset_timeout=30
)
app.add_middleware(
    RequestLoggingMiddleware,
    exclude_paths=["/health", "/metrics"]
)
app.add_middleware(
    RateLimitingMiddleware,
    rate_limit=100,
    window_seconds=60
)

# Add exception handlers
@app.exception_handler(DDoSAIException)
async def ddosai_exception_handler(request: Request, exc: DDoSAIException):
    """Handle DDoSAI exceptions"""
    ErrorHandler.log_exception(exc)
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "details": {"headers": exc.headers},
            "error_type": "HTTPException",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    error_response = ErrorHandler.format_validation_errors(exc.errors())
    return JSONResponse(
        status_code=422,
        content=error_response
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    error_response = ErrorHandler.handle_exception(exc)
    return JSONResponse(
        status_code=error_response.get("status_code", 500),
        content=error_response
    )

# Global AI components
feature_extractor: Optional[FeatureExtractor] = None
autoencoder_detector: Optional[AutoencoderDetector] = None
gnn_analyzer: Optional[GNNAnalyzer] = None
rl_threat_scorer: Optional[RLThreatScorer] = None
xai_explainer: Optional[XAIExplainer] = None

# AI Engine orchestrator
from ai.ai_engine import AIEngine
ai_engine: Optional[AIEngine] = None

# WebSocket connections for real-time updates
active_connections: List[WebSocket] = []

# Storage for network state and fallback detection history
network_nodes: Dict[str, NetworkNode] = {}
network_edges: Dict[str, NetworkEdge] = {}
detection_history: List[DetectionResult] = []
prediction_cache: Dict[str, Dict[str, Any]] = {}

async def initialize_ai_components():
    """Initialize AI components on startup"""
    global feature_extractor, autoencoder_detector, gnn_analyzer, rl_threat_scorer, xai_explainer, ai_engine
    
    try:
        logger.info("Initializing AI components...")
        
        # Initialize feature extractor
        feature_extractor = FeatureExtractor()
        logger.info("Feature extractor initialized")
        
        # Initialize autoencoder detector
        autoencoder_detector = AutoencoderDetector(input_dim=31, threshold_percentile=95.0)
        logger.info("Autoencoder detector initialized")
        
        # Initialize GNN analyzer
        gnn_analyzer = GNNAnalyzer(node_feature_dim=31, hidden_dim=64)
        logger.info("GNN analyzer initialized")
        
        # Initialize RL threat scorer
        rl_threat_scorer = RLThreatScorer(state_dim=31, num_threat_levels=11)
        logger.info("RL threat scorer initialized")
        
        # Initialize XAI explainer
        feature_names = feature_extractor.get_feature_names()
        xai_explainer = XAIExplainer(feature_names=feature_names)
        logger.info("XAI explainer initialized")
        
        # Initialize AI Engine orchestrator
        ai_engine = AIEngine(
            feature_extractor=feature_extractor,
            autoencoder_detector=autoencoder_detector,
            gnn_analyzer=gnn_analyzer,
            rl_threat_scorer=rl_threat_scorer,
            xai_explainer=xai_explainer
        )
        logger.info("AI Engine orchestrator initialized")
        
        logger.info("All AI components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize AI components: {e}")
        raise

async def broadcast_to_websockets(message: Dict[str, Any]):
    """Broadcast message to all connected WebSocket clients"""
    if not active_connections:
        return
    
    message_str = json.dumps(message)
    disconnected = []
    
    for connection in active_connections:
        try:
            if connection.client_state == WebSocketState.CONNECTED:
                await connection.send_text(message_str)
            else:
                disconnected.append(connection)
        except Exception as e:
            logger.warning(f"Failed to send message to WebSocket client: {e}")
            disconnected.append(connection)
    
    # Remove disconnected clients
    for connection in disconnected:
        if connection in active_connections:
            active_connections.remove(connection)

async def analyze_packet_with_ai(packet: TrafficPacket) -> DetectionResult:
    """Analyze a packet using all AI models"""
    from core.recovery import CircuitBreaker
    from core.metrics import MetricsCollector, PerformanceTimer
    
    # Start timing the packet processing
    start_time = time.time()
    
    # Initialize circuit breakers for each model
    autoencoder_cb = CircuitBreaker.get("autoencoder_detector")
    gnn_cb = CircuitBreaker.get("gnn_analyzer")
    rl_cb = CircuitBreaker.get("rl_threat_scorer")
    
    try:
        # Extract features
        features = feature_extractor.extract_packet_features(packet)
        
        # Run through all AI models
        model_results = {}
        
        # Autoencoder detection with circuit breaker
        if autoencoder_detector:
            try:
                # Check if circuit breaker allows the request
                if autoencoder_cb.allow_request():
                    try:
                        # Time the model inference
                        with PerformanceTimer("autoencoder_inference", model_name="autoencoder"):
                            is_anomaly, confidence, explanation = autoencoder_detector.predict(features)
                        
                        model_results["autoencoder"] = {
                            "is_malicious": is_anomaly,
                            "confidence": confidence,
                            "reconstruction_error": explanation.get("reconstruction_error", 0.0)
                        }
                        # Record success
                        autoencoder_cb.record_success()
                    except Exception as e:
                        # Record failure
                        autoencoder_cb.record_failure()
                        logger.warning(f"Autoencoder detection failed: {e}")
                        # Use fallback rule-based detection
                        model_results["autoencoder"] = {
                            "is_malicious": packet.src_ip.startswith("192.168"),  # Simple rule for testing
                            "confidence": 0.6,  # Lower confidence for fallback
                            "reconstruction_error": 0.15,
                            "fallback": True
                        }
                else:
                    # Circuit is open, use fallback
                    logger.warning("Autoencoder circuit breaker open, using fallback")
                    model_results["autoencoder"] = {
                        "is_malicious": packet.src_ip.startswith("192.168"),
                        "confidence": 0.5,  # Even lower confidence for circuit breaker fallback
                        "reconstruction_error": 0.2,
                        "fallback": True,
                        "circuit_breaker": "open"
                    }
            except Exception as e:
                logger.warning(f"Autoencoder detection and fallback failed: {e}")
                # Use very basic fallback
                model_results["autoencoder"] = {
                    "is_malicious": False,
                    "confidence": 0.3,
                    "reconstruction_error": 0.0,
                    "fallback": True,
                    "error": str(e)
                }
                # Record error
                MetricsCollector.record_error()
        
        # GNN analysis with circuit breaker
        if gnn_analyzer:
            try:
                # Check if circuit breaker allows the request
                if gnn_cb.allow_request():
                    try:
                        # Time the model inference
                        with PerformanceTimer("gnn_inference", model_name="gnn"):
                            # For single packet analysis, we'll use a simplified approach
                            # In a real implementation, this would use network graph context
                            gnn_score = 0.7 if "SYN" in packet.flags else 0.3
                        
                        model_results["gnn"] = {
                            "is_malicious": gnn_score > 0.5,
                            "confidence": abs(gnn_score - 0.5) * 2,
                            "malicious_probability": gnn_score
                        }
                        # Record success
                        gnn_cb.record_success()
                    except Exception as e:
                        # Record failure
                        gnn_cb.record_failure()
                        logger.warning(f"GNN analysis failed: {e}")
                        # Use fallback rule-based detection
                        gnn_score = 0.6 if "SYN" in packet.flags else 0.4
                        model_results["gnn"] = {
                            "is_malicious": gnn_score > 0.5,
                            "confidence": abs(gnn_score - 0.5) * 1.5,  # Lower confidence for fallback
                            "malicious_probability": gnn_score,
                            "fallback": True
                        }
                else:
                    # Circuit is open, use fallback
                    logger.warning("GNN circuit breaker open, using fallback")
                    gnn_score = 0.55 if "SYN" in packet.flags else 0.45
                    model_results["gnn"] = {
                        "is_malicious": gnn_score > 0.5,
                        "confidence": abs(gnn_score - 0.5),  # Even lower confidence for circuit breaker fallback
                        "malicious_probability": gnn_score,
                        "fallback": True,
                        "circuit_breaker": "open"
                    }
            except Exception as e:
                logger.warning(f"GNN analysis and fallback failed: {e}")
                # Use very basic fallback
                model_results["gnn"] = {
                    "is_malicious": False,
                    "confidence": 0.3,
                    "malicious_probability": 0.3,
                    "fallback": True,
                    "error": str(e)
                }
                # Record error
                MetricsCollector.record_error()
        
        # RL threat scoring with circuit breaker
        if rl_threat_scorer:
            try:
                # Check if circuit breaker allows the request
                if rl_cb.allow_request():
                    try:
                        # Time the model inference
                        with PerformanceTimer("rl_inference", model_name="rl_threat_scorer"):
                            is_malicious, confidence, explanation = rl_threat_scorer.predict(features, {})
                        
                        threat_score = explanation.get("threat_score", 50)
                        model_results["rl"] = {
                            "is_malicious": is_malicious,
                            "confidence": confidence,
                            "threat_score": threat_score
                        }
                        # Record success
                        rl_cb.record_success()
                    except Exception as e:
                        # Record failure
                        rl_cb.record_failure()
                        logger.warning(f"RL threat scoring failed: {e}")
                        # Use fallback rule-based scoring
                        threat_score = 65 if "SYN" in packet.flags else 35
                        model_results["rl"] = {
                            "is_malicious": threat_score > 50,
                            "confidence": 0.7,  # Lower confidence for fallback
                            "threat_score": threat_score,
                            "fallback": True
                        }
                else:
                    # Circuit is open, use fallback
                    logger.warning("RL circuit breaker open, using fallback")
                    threat_score = 60 if "SYN" in packet.flags else 40
                    model_results["rl"] = {
                        "is_malicious": threat_score > 50,
                        "confidence": 0.5,  # Even lower confidence for circuit breaker fallback
                        "threat_score": threat_score,
                        "fallback": True,
                        "circuit_breaker": "open"
                    }
            except Exception as e:
                logger.warning(f"RL threat scoring and fallback failed: {e}")
                # Use very basic fallback
                model_results["rl"] = {
                    "is_malicious": False,
                    "confidence": 0.3,
                    "threat_score": 30,
                    "fallback": True,
                    "error": str(e)
                }
                # Record error
                MetricsCollector.record_error()
        
        # Consensus decision with weighted voting based on confidence
        total_weight = 0
        weighted_malicious_score = 0
        
        for model_name, result in model_results.items():
            weight = result["confidence"]
            total_weight += weight
            if result["is_malicious"]:
                weighted_malicious_score += weight
        
        # Determine if malicious based on weighted voting
        is_malicious = False
        if total_weight > 0:
            malicious_ratio = weighted_malicious_score / total_weight
            is_malicious = malicious_ratio > 0.5
        
        # Calculate overall confidence
        confidences = [result["confidence"] for result in model_results.values()]
        overall_confidence = np.mean(confidences) if confidences else 0.5
        
        # Determine threat score
        if "rl" in model_results:
            threat_score = model_results["rl"]["threat_score"]
        else:
            threat_score = 75 if is_malicious else 25
        
        # Determine attack type with fallback logic
        if is_malicious:
            # Simple heuristic for attack type determination
            if "SYN" in packet.flags:
                attack_type = AttackType.SYN_FLOOD
            elif packet.protocol == "UDP":
                attack_type = AttackType.UDP_FLOOD
            elif packet.protocol == "HTTP":
                attack_type = AttackType.HTTP_FLOOD
            else:
                attack_type = AttackType.UNKNOWN
        else:
            attack_type = AttackType.BENIGN
        
        # Create detection result
        detection_result = DetectionResult(
            timestamp=packet.timestamp,
            packet_id=packet.packet_id,
            flow_id=None,  # Would be determined by flow analysis
            is_malicious=is_malicious,
            threat_score=threat_score,
            attack_type=attack_type,
            detection_method="weighted_consensus",
            confidence=overall_confidence,
            explanation={
                "model_results": model_results,
                "weighted_malicious_score": weighted_malicious_score,
                "total_weight": total_weight,
                "malicious_ratio": weighted_malicious_score / total_weight if total_weight > 0 else 0
            },
            model_version="1.0.0"
        )
        
        # Store in history
        detection_history.append(detection_result)
        if len(detection_history) > 1000:  # Limit history size
            detection_history.pop(0)
        
        # Cache prediction with features for XAI
        prediction_cache[packet.packet_id] = {
            "features": features,
            "is_malicious": is_malicious,
            "confidence": overall_confidence,
            "explanation": detection_result.explanation
        }
        
        # Broadcast to WebSocket clients
        await broadcast_to_websockets({
            "type": "detection",
            "data": detection_result.to_dict()
        })
        
        # Update monitoring metrics
        from core.monitoring import MonitoringService
        MonitoringService.update_component_status(
            "ai_detection", 
            "healthy" if all(not result.get("fallback", False) for result in model_results.values()) else "degraded",
            {
                "models_used": list(model_results.keys()),
                "fallbacks_used": [model for model, result in model_results.items() if result.get("fallback", False)]
            }
        )
        
        # Record packet processing time and result
        processing_time = time.time() - start_time
        MetricsCollector.record_packet_processed(processing_time, is_malicious)
        
        return detection_result
        
    except Exception as e:
        logger.error(f"Error analyzing packet: {e}")
        # Update monitoring status
        from core.monitoring import MonitoringService
        MonitoringService.update_component_status(
            "ai_detection", 
            "unhealthy",
            {"error": str(e)}
        )
        
        # Record error
        MetricsCollector.record_error()
        
        # Record packet processing time (failed)
        processing_time = time.time() - start_time
        MetricsCollector.record_packet_processed(processing_time, False)
        
        # Return a fallback result
        return DetectionResult(
            timestamp=packet.timestamp,
            packet_id=packet.packet_id,
            flow_id=None,
            is_malicious=False,
            threat_score=0,
            attack_type=AttackType.BENIGN,
            detection_method="error_fallback",
            confidence=0.0,
            explanation={"error": str(e)},
            model_version="1.0.0"
        )

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "DDoS.AI API is running", "version": "1.0.0"}

@app.get("/health")
@handle_errors
async def health_check(detailed: bool = False):
    """Health check endpoint
    
    Args:
        detailed: If True, return detailed health information
    """
    if detailed:
        # Get detailed health status from monitoring service
        health_status = MonitoringService.get_health_status()
        return health_status
    else:
        # Simple health check for load balancers
        return {
            "status": "healthy", 
            "timestamp": datetime.now().isoformat(),
            "version": os.environ.get("APP_VERSION", "1.0.0")
        }

@app.post("/api/analyze")
@handle_errors
@log_execution_time
async def analyze_traffic(packet_data: Dict[str, Any], request: Request):
    """Analyze traffic packet for threats"""
    # Convert packet data to TrafficPacket object
    packet = TrafficPacket(
        timestamp=datetime.fromisoformat(packet_data.get("timestamp", datetime.now().isoformat())),
        src_ip=packet_data.get("src_ip", "0.0.0.0"),
        dst_ip=packet_data.get("dst_ip", "0.0.0.0"),
        src_port=packet_data.get("src_port", 0),
        dst_port=packet_data.get("dst_port", 0),
        protocol=ProtocolType(packet_data.get("protocol", "TCP")),
        packet_size=packet_data.get("packet_size", 64),
        ttl=packet_data.get("ttl", 64),
        flags=packet_data.get("flags", []),
        payload_entropy=packet_data.get("payload_entropy", 0.5),
        packet_id=packet_data.get("packet_id", f"pkt_{uuid.uuid4().hex[:8]}")
    )
    
    # Analyze packet with AI Engine
    if ai_engine:
        result = await ai_engine.analyze_packet(packet)
        
        # Broadcast to WebSocket clients
        await broadcast_to_websockets({
            "type": "detection",
            "data": result.to_dict()
        })
        
        return result.to_dict()
    else:
        # Fallback to direct implementation if AI Engine is not initialized
        from core.exceptions import AIModelError
        raise AIModelError("AI Engine not initialized", details={"component": "ai_engine"})

@app.get("/api/detections")
@handle_errors
@log_execution_time
async def get_detections(request: Request, limit: int = 50):
    """Get recent detection results"""
    if ai_engine:
        # Use AI Engine to get recent detections
        detections = ai_engine.get_recent_detections(limit)
        return [detection.to_dict() for detection in detections]
    else:
        # Fallback to direct implementation
        return [detection.to_dict() for detection in detection_history[-limit:]]

@app.get("/api/graph/current")
@handle_errors
@log_execution_time
async def get_network_graph(request: Request):
    """Get current network graph state"""
    return {
        "nodes": [node.to_dict() for node in network_nodes.values()],
        "edges": [edge.to_dict() for edge in network_edges.values()],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/metrics")
@handle_errors
@log_execution_time
async def get_system_metrics(request: Request, detailed: bool = False):
    """Get system performance metrics
    
    Args:
        detailed: If True, return detailed metrics information
    """
    # Use the metrics collector for comprehensive metrics
    from core.metrics import MetricsCollector
    
    if detailed:
        # Return detailed metrics from the metrics collector
        return MetricsCollector.get_metrics()
    
    # Get basic system metrics
    import psutil
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    
    if ai_engine:
        # Get metrics from AI Engine
        ai_metrics = ai_engine.get_performance_metrics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "packets_processed": ai_metrics["packet_count"],
            "processing_latency_ms": int(ai_metrics["avg_processing_time"] * 1000),  # Convert to ms and cast to int
            "cpu_usage": cpu_percent,
            "memory_usage": memory.percent,
            "active_connections": len(active_connections),
            "threat_level": ai_metrics["threat_level"],
            "malicious_packets": ai_metrics["malicious_count"],
            "total_detections": ai_metrics["packet_count"]
        }
    else:
        # Fallback to direct implementation
        # Calculate threat level based on recent detections
        recent_detections = detection_history[-100:] if detection_history else []
        malicious_count = sum(1 for d in recent_detections if d.is_malicious)
        threat_level = min(5, malicious_count // 10)  # Scale to 0-5
        
        return {
            "timestamp": datetime.now().isoformat(),
            "packets_processed": len(detection_history),
            "processing_latency_ms": 150,  # Would be calculated from actual processing times
            "cpu_usage": cpu_percent,
            "memory_usage": memory.percent,
            "active_connections": len(active_connections),
            "threat_level": threat_level,
            "malicious_packets": malicious_count,
            "total_detections": len(recent_detections)
        }

@app.post("/api/simulate/start")
@handle_errors
@log_execution_time
async def start_simulation(config: Dict[str, Any], request: Request):
    """Start attack simulation"""
    # Validate simulation configuration
    if not config.get('attack_type'):
        from core.exceptions import ValidationError
        raise ValidationError("Attack type is required", details={"field": "attack_type"})
    
    # Generate simulation ID
    import random
    simulation_id = f"sim_{random.randint(1000, 9999)}"
    
    # Start simulation in background task (would be implemented in a real system)
    # Here we're just returning a response
    return {
        "status": "started",
        "simulation_id": simulation_id,
        "message": f"Started {config.get('attack_type')} simulation",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/simulate/stop")
@handle_errors
@log_execution_time
async def stop_simulation(simulation_id: str, request: Request = None):
    """Stop attack simulation"""
    return {
        "status": "stopped",
        "simulation_id": simulation_id,
        "message": "Simulation stopped",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/explain/{prediction_id}")
@handle_errors
@log_execution_time
async def get_explanation(prediction_id: str, request: Request):
    """Get XAI explanation for a prediction"""
    # Use AI Engine for explanation if available
    if ai_engine:
        explanation = ai_engine.get_explanation(prediction_id)
        return explanation
    
    # Fallback to direct implementation
    elif prediction_id in prediction_cache:
        cached_result = prediction_cache[prediction_id]
        
        # Generate XAI explanation using the actual explainer
        if xai_explainer and "features" in cached_result:
            features = cached_result["features"]
            prediction = (
                cached_result["is_malicious"],
                cached_result["confidence"],
                cached_result["explanation"]
            )
            
            try:
                # Create a simple prediction function for the explainer
                def predict_fn(features_array):
                    # Return probabilities for binary classification (benign, malicious)
                    is_malicious = cached_result["is_malicious"]
                    confidence = cached_result["confidence"]
                    return np.array([[1-confidence, confidence] if is_malicious else [confidence, 1-confidence]])
                
                # Initialize explainers if needed
                if not hasattr(xai_explainer, 'lime_explainer') or xai_explainer.lime_explainer is None:
                    # Create dummy training data for initialization
                    dummy_data = np.random.random((10, len(features)))
                    xai_explainer.initialize_explainers(dummy_data, predict_fn)
                
                explanation = xai_explainer.explain_prediction(
                    features,
                    predict_fn,
                    prediction,
                    method="lime"
                )
                
                return {
                    "prediction_id": prediction_id,
                    "explanation": explanation,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.warning(f"XAI explanation failed: {e}, using fallback")
                from core.exceptions import AIModelError
                raise AIModelError("Failed to generate XAI explanation", details={"error": str(e)})
    
    # If prediction not found, raise NotFoundError
    from core.exceptions import NotFoundError
    raise NotFoundError(f"Prediction with ID {prediction_id} not found", 
                        details={"prediction_id": prediction_id})

# WebSocket endpoints
@app.websocket("/ws/live-feed")
async def websocket_live_feed(websocket: WebSocket):
    """WebSocket endpoint for live traffic feed"""
    connection_id = str(uuid.uuid4())
    logger.info(f"WebSocket connection attempt: {connection_id}", connection_id=connection_id)
    
    try:
        await websocket.accept()
        active_connections.append(websocket)
        logger.info(f"WebSocket connection accepted: {connection_id}", connection_id=connection_id)
        
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection_established",
            "connection_id": connection_id,
            "timestamp": datetime.now().isoformat()
        })
        
        while True:
            # Keep connection alive and handle incoming messages
            try:
                data = await websocket.receive_text()
                logger.debug(f"WebSocket message received: {connection_id}", connection_id=connection_id, data=data[:100])
                
                # Process the message
                try:
                    message = json.loads(data)
                    # Handle different message types
                    if message.get("type") == "ping":
                        await websocket.send_json({
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        # Echo back or handle client messages if needed
                        await websocket.send_json({
                            "type": "echo",
                            "data": message,
                            "timestamp": datetime.now().isoformat()
                        })
                except json.JSONDecodeError:
                    # Handle non-JSON messages
                    await websocket.send_json({
                        "type": "error",
                        "error": "Invalid JSON format",
                        "timestamp": datetime.now().isoformat()
                    })
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {connection_id}", connection_id=connection_id)
                break
            except Exception as e:
                logger.error(f"WebSocket error: {connection_id} - {e}", connection_id=connection_id, error=str(e))
                # Try to send error message to client
                try:
                    await websocket.send_json({
                        "type": "error",
                        "error": "Internal server error",
                        "timestamp": datetime.now().isoformat()
                    })
                except:
                    # If we can't send the error, just break the connection
                    break
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected during handshake: {connection_id}", connection_id=connection_id)
    except Exception as e:
        logger.error(f"WebSocket connection error: {connection_id} - {e}", connection_id=connection_id, error=str(e))
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
        logger.info(f"WebSocket connection closed: {connection_id}", connection_id=connection_id)

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for real-time threat alerts"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            try:
                data = await websocket.receive_text()
                # Handle alert subscriptions or filters
                await websocket.send_text(f"Alert subscription: {data}")
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket alerts error: {e}")
                break
                
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    try:
        # Set up graceful shutdown handlers
        from core.recovery import GracefulShutdown
        GracefulShutdown.setup_signal_handlers()
        
        # Initialize AI components
        await initialize_ai_components()
        
        # Initialize database connections
        from core.database import init_redis, init_influxdb
        await init_redis()
        init_influxdb()
        
        # Initialize metrics collection
        try:
            from core.metrics import MetricsCollector
            enable_prometheus = os.environ.get("ENABLE_PROMETHEUS", "false").lower() == "true"  # Default to false
            prometheus_port = int(os.environ.get("PROMETHEUS_PORT", "8001"))
            MetricsCollector.initialize(enable_prometheus=enable_prometheus, prometheus_port=prometheus_port)
            logger.info(f"Metrics collection initialized (Prometheus: {enable_prometheus}, Port: {prometheus_port})")
        except ImportError:
            logger.warning("Prometheus client not installed. Metrics collection disabled.")
            # Set environment variable to disable Prometheus
            os.environ["ENABLE_PROMETHEUS"] = "false"
        
        # Start watchdog process
        from core.recovery import ErrorRecovery
        asyncio.create_task(ErrorRecovery.watchdog())
        
        logger.info("All components initialized successfully", service="api", status="started")
    except Exception as e:
        logger.critical(f"Failed to initialize components: {e}", service="api", status="failed")
        # Record critical error and attempt recovery
        from core.recovery import ErrorRecovery
        ErrorRecovery.record_error(e, is_critical=True)
        raise

# Don't initialize AI components immediately - let the startup event handle it
# This avoids running the event loop twice and causing RuntimeError
import asyncio

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down DDoS.AI API...", service="api", status="stopping")
    
    try:
        # Close all WebSocket connections
        for connection in active_connections:
            try:
                await connection.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket connection: {e}")
        active_connections.clear()
        
        # Close database connections
        from core.database import close_db_connections
        await close_db_connections()
        
        logger.info("Shutdown completed successfully", service="api", status="stopped")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", service="api", status="error")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug
    )