"""
Configuration management for DDoS.AI platform
"""
import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for AI models"""
    autoencoder_threshold: float = 0.1
    autoencoder_hidden_dims: list = None
    gnn_hidden_dim: int = 64
    gnn_num_layers: int = 2
    rl_state_dim: int = 32
    rl_action_dim: int = 101
    batch_size: int = 32
    learning_rate: float = 0.001
    
    def __post_init__(self):
        if self.autoencoder_hidden_dims is None:
            self.autoencoder_hidden_dims = [64, 32, 16]


@dataclass
class APIConfig:
    """Configuration for API server"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: list = None
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["http://localhost:3000"]


@dataclass
class RedisConfig:
    """Configuration for Redis cache"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = None
    max_connections: int = 10


@dataclass
class ProcessingConfig:
    """Configuration for traffic processing"""
    max_packet_buffer: int = 10000
    processing_timeout: int = 30
    feature_vector_size: int = 32
    sliding_window_size: int = 60  # seconds
    max_concurrent_flows: int = 1000


class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.api = APIConfig()
        self.redis = RedisConfig()
        self.processing = ProcessingConfig()
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # API Configuration
        self.api.host = os.getenv("API_HOST", self.api.host)
        self.api.port = int(os.getenv("API_PORT", self.api.port))
        self.api.debug = os.getenv("API_DEBUG", "false").lower() == "true"
        
        # Redis Configuration
        self.redis.host = os.getenv("REDIS_HOST", self.redis.host)
        self.redis.port = int(os.getenv("REDIS_PORT", self.redis.port))
        self.redis.password = os.getenv("REDIS_PASSWORD", self.redis.password)
        
        # Model Configuration
        self.model.autoencoder_threshold = float(
            os.getenv("AUTOENCODER_THRESHOLD", self.model.autoencoder_threshold)
        )
        self.model.batch_size = int(
            os.getenv("MODEL_BATCH_SIZE", self.model.batch_size)
        )
        
        # Processing Configuration
        self.processing.max_packet_buffer = int(
            os.getenv("MAX_PACKET_BUFFER", self.processing.max_packet_buffer)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "model": self.model.__dict__,
            "api": self.api.__dict__,
            "redis": self.redis.__dict__,
            "processing": self.processing.__dict__
        }


# Global configuration instance
config = Config()