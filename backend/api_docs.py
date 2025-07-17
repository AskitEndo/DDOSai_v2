"""
API Documentation and OpenAPI schema configuration for DDoS.AI backend
"""
from fastapi import FastAPI
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

# Request/Response models for API documentation

class ProtocolTypeEnum(str, Enum):
    """Protocol type enumeration"""
    TCP = "TCP"
    UDP = "UDP"
    HTTP = "HTTP"
    HTTPS = "HTTPS"
    ICMP = "ICMP"

class AttackTypeEnum(str, Enum):
    """Attack type enumeration"""
    BENIGN = "BENIGN"
    SYN_FLOOD = "SYN_FLOOD"
    UDP_FLOOD = "UDP_FLOOD"
    HTTP_FLOOD = "HTTP_FLOOD"
    SLOWLORIS = "SLOWLORIS"
    PING_FLOOD = "PING_FLOOD"

class PacketAnalysisRequest(BaseModel):
    """Request model for packet analysis"""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Packet timestamp in ISO format")
    src_ip: str = Field(..., description="Source IP address", example="192.168.1.100")
    dst_ip: str = Field(..., description="Destination IP address", example="10.0.0.1")
    src_port: int = Field(default=0, description="Source port number", example=12345)
    dst_port: int = Field(default=0, description="Destination port number", example=80)
    protocol: ProtocolTypeEnum = Field(default=ProtocolTypeEnum.TCP, description="Network protocol")
    packet_size: int = Field(default=64, description="Packet size in bytes", example=1024)
    ttl: int = Field(default=64, description="Time to live value", example=64)
    flags: List[str] = Field(default_factory=list, description="TCP flags", example=["SYN", "ACK"])
    payload_entropy: float = Field(default=0.5, description="Payload entropy value", example=0.7)
    packet_id: str = Field(default="", description="Unique packet identifier", example="pkt_001")

class DetectionResponse(BaseModel):
    """Response model for detection results"""
    timestamp: str = Field(..., description="Detection timestamp")
    packet_id: str = Field(..., description="Packet identifier")
    flow_id: str = Field(None, description="Flow identifier")
    is_malicious: bool = Field(..., description="Whether packet is malicious")
    threat_score: int = Field(..., description="Threat score (0-100)")
    attack_type: AttackTypeEnum = Field(..., description="Type of attack detected")
    detection_method: str = Field(..., description="Detection method used")
    confidence: float = Field(..., description="Confidence score (0-1)")
    explanation: Dict[str, Any] = Field(..., description="Explanation of detection")
    model_version: str = Field(..., description="Model version used")

class NetworkNode(BaseModel):
    """Network node model"""
    node_id: str = Field(..., description="Unique node identifier")
    ip_address: str = Field(..., description="IP address of the node")
    packet_count: int = Field(..., description="Number of packets processed")
    byte_count: int = Field(..., description="Total bytes processed")
    connection_count: int = Field(..., description="Number of connections")
    threat_score: int = Field(..., description="Node threat score")
    is_malicious: bool = Field(..., description="Whether node is malicious")
    first_seen: str = Field(..., description="First seen timestamp")
    last_seen: str = Field(..., description="Last seen timestamp")

class NetworkEdge(BaseModel):
    """Network edge model"""
    edge_id: str = Field(..., description="Unique edge identifier")
    source_ip: str = Field(..., description="Source IP address")
    target_ip: str = Field(..., description="Target IP address")
    flow_count: int = Field(..., description="Number of flows")
    total_bytes: int = Field(..., description="Total bytes transferred")
    avg_packet_size: float = Field(..., description="Average packet size")
    connection_duration: float = Field(..., description="Connection duration in seconds")
    protocols: List[str] = Field(..., description="Protocols used")

class NetworkGraphResponse(BaseModel):
    """Network graph response model"""
    nodes: List[NetworkNode] = Field(..., description="Network nodes")
    edges: List[NetworkEdge] = Field(..., description="Network edges")
    timestamp: str = Field(..., description="Graph timestamp")

class SystemMetricsResponse(BaseModel):
    """System metrics response model"""
    timestamp: str = Field(..., description="Metrics timestamp")
    packets_processed: int = Field(..., description="Total packets processed")
    processing_latency_ms: int = Field(..., description="Average processing latency in milliseconds")
    cpu_usage: float = Field(..., description="CPU usage percentage")
    memory_usage: float = Field(..., description="Memory usage percentage")
    active_connections: int = Field(..., description="Number of active WebSocket connections")
    threat_level: int = Field(..., description="Current threat level (0-5)")
    malicious_packets: int = Field(default=0, description="Number of malicious packets detected")
    total_detections: int = Field(default=0, description="Total number of detections")

class SimulationConfig(BaseModel):
    """Simulation configuration model"""
    attack_type: str = Field(..., description="Type of attack to simulate", example="syn_flood")
    target_ip: str = Field(default="192.168.1.1", description="Target IP address")
    duration: int = Field(default=60, description="Simulation duration in seconds")
    packet_rate: int = Field(default=1000, description="Packets per second")
    packet_size: int = Field(default=64, description="Packet size in bytes")

class SimulationResponse(BaseModel):
    """Simulation response model"""
    status: str = Field(..., description="Simulation status")
    simulation_id: str = Field(..., description="Unique simulation identifier")
    message: str = Field(..., description="Status message")
    timestamp: str = Field(..., description="Response timestamp")

class FeatureImportance(BaseModel):
    """Feature importance model"""
    feature_name: str = Field(..., description="Name of the feature")
    importance_score: float = Field(..., description="Importance score")
    description: str = Field(..., description="Feature description")

class ExplanationResponse(BaseModel):
    """XAI explanation response model"""
    prediction_id: str = Field(..., description="Prediction identifier")
    model_name: str = Field(..., description="Model name")
    prediction: bool = Field(..., description="Prediction result")
    confidence: float = Field(..., description="Prediction confidence")
    feature_importance: List[FeatureImportance] = Field(..., description="Feature importance scores")
    decision_path: List[str] = Field(..., description="Decision path")
    counterfactuals: List[Dict[str, Any]] = Field(..., description="Counterfactual explanations")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: str = Field(None, description="Request identifier")

# API Tags for organization
tags_metadata = [
    {
        "name": "health",
        "description": "Health check and system status endpoints",
    },
    {
        "name": "analysis",
        "description": "Traffic analysis and threat detection endpoints",
    },
    {
        "name": "network",
        "description": "Network graph and topology endpoints",
    },
    {
        "name": "metrics",
        "description": "System performance and monitoring endpoints",
    },
    {
        "name": "simulation",
        "description": "Attack simulation and testing endpoints",
    },
    {
        "name": "explanation",
        "description": "Explainable AI and model interpretation endpoints",
    },
    {
        "name": "websocket",
        "description": "Real-time WebSocket endpoints",
    },
]

# OpenAPI configuration
openapi_config = {
    "title": "DDoS.AI API",
    "description": """
    ## DDoS.AI Platform API
    
    A comprehensive AI-powered DDoS detection and analysis platform that provides:
    
    * **Real-time Traffic Analysis**: Analyze network packets using multiple AI models
    * **Threat Detection**: Detect various types of DDoS attacks with high accuracy
    * **Network Visualization**: Visualize network topology and attack patterns
    * **Explainable AI**: Understand why the AI made specific decisions
    * **Attack Simulation**: Simulate various DDoS attacks for testing
    * **Real-time Monitoring**: WebSocket-based real-time updates
    
    ### AI Models
    
    The platform uses multiple AI models for comprehensive threat detection:
    
    * **Autoencoder**: Anomaly detection based on reconstruction error
    * **Graph Neural Network (GNN)**: Network topology-based malicious node detection
    * **Reinforcement Learning**: Adaptive threat scoring based on historical data
    * **Explainable AI (XAI)**: SHAP and LIME-based model explanations
    
    ### Performance Requirements
    
    * Packet processing latency: < 200ms
    * Explanation generation: < 500ms
    * Throughput: 100k packets/second
    * False positive rate: < 2%
    
    ### Authentication
    
    Currently, the API does not require authentication for development purposes.
    In production, implement proper authentication and authorization.
    """,
    "version": "1.0.0",
    "contact": {
        "name": "DDoS.AI Team",
        "email": "support@ddos-ai.com",
    },
    "license_info": {
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    "tags": tags_metadata,
}

def configure_api_docs(app: FastAPI):
    """Configure API documentation"""
    # Update OpenAPI schema
    app.title = openapi_config["title"]
    app.description = openapi_config["description"]
    app.version = openapi_config["version"]
    app.contact = openapi_config["contact"]
    app.license_info = openapi_config["license_info"]
    app.openapi_tags = openapi_config["tags"]
    
    return app

# Example responses for documentation
example_responses = {
    "packet_analysis": {
        "200": {
            "description": "Successful analysis",
            "content": {
                "application/json": {
                    "example": {
                        "timestamp": "2024-01-15T10:30:00.000Z",
                        "packet_id": "pkt_001",
                        "flow_id": "flow_001",
                        "is_malicious": True,
                        "threat_score": 85,
                        "attack_type": "SYN_FLOOD",
                        "detection_method": "consensus",
                        "confidence": 0.92,
                        "explanation": {
                            "model_results": {
                                "autoencoder": {"is_malicious": True, "confidence": 0.9},
                                "gnn": {"is_malicious": True, "confidence": 0.88},
                                "rl": {"is_malicious": True, "confidence": 0.95}
                            },
                            "consensus_votes": 3,
                            "total_models": 3
                        },
                        "model_version": "1.0.0"
                    }
                }
            }
        }
    },
    "network_graph": {
        "200": {
            "description": "Current network graph",
            "content": {
                "application/json": {
                    "example": {
                        "nodes": [
                            {
                                "node_id": "node_192_168_1_100",
                                "ip_address": "192.168.1.100",
                                "packet_count": 150,
                                "byte_count": 75000,
                                "connection_count": 5,
                                "threat_score": 25,
                                "is_malicious": False,
                                "first_seen": "2024-01-15T10:00:00.000Z",
                                "last_seen": "2024-01-15T10:30:00.000Z"
                            }
                        ],
                        "edges": [
                            {
                                "edge_id": "edge_1_2",
                                "source_ip": "192.168.1.100",
                                "target_ip": "10.0.0.1",
                                "flow_count": 3,
                                "total_bytes": 15000,
                                "avg_packet_size": 500,
                                "connection_duration": 30.5,
                                "protocols": ["TCP", "HTTP"]
                            }
                        ],
                        "timestamp": "2024-01-15T10:30:00.000Z"
                    }
                }
            }
        }
    },
    "explanation": {
        "200": {
            "description": "XAI explanation",
            "content": {
                "application/json": {
                    "example": {
                        "prediction_id": "pred_001",
                        "model_name": "consensus",
                        "prediction": True,
                        "confidence": 0.92,
                        "feature_importance": [
                            {
                                "feature_name": "packet_rate",
                                "importance_score": 0.85,
                                "description": "Rate of incoming packets"
                            },
                            {
                                "feature_name": "syn_flags",
                                "importance_score": 0.72,
                                "description": "Proportion of SYN flags"
                            }
                        ],
                        "decision_path": [
                            "High packet rate detected",
                            "SYN flag anomaly",
                            "Low source diversity"
                        ],
                        "counterfactuals": [
                            {
                                "feature": "packet_rate",
                                "current": 5000,
                                "threshold": 1000
                            }
                        ]
                    }
                }
            }
        }
    }
}