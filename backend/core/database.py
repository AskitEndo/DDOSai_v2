"""
Database configuration and session management for DDoS.AI platform
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
import redis.asyncio as aioredis
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
import logging
from typing import AsyncGenerator, Optional
import os

from core.config import config

# Configure logging
logger = logging.getLogger(__name__)

# SQLAlchemy Base for ORM models
Base = declarative_base()

# PostgreSQL connection
SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    f"postgresql+asyncpg://{os.getenv('DB_USER', 'postgres')}:{os.getenv('DB_PASSWORD', 'postgres')}@"
    f"{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'ddosai')}"
)

# Create async engine
engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    echo=config.api.debug,
    future=True,
    poolclass=NullPool  # Use NullPool for better async behavior
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False,
    autoflush=False
)

# Redis client
redis_client: Optional[aioredis.Redis] = None

# InfluxDB client for time-series data
influxdb_client: Optional[InfluxDBClient] = None
influxdb_write_api = None
influxdb_query_api = None

async def init_redis():
    """Initialize Redis connection"""
    global redis_client
    try:
        redis_client = await aioredis.from_url(
            f"redis://{config.redis.host}:{config.redis.port}/{config.redis.db}",
            password=config.redis.password,
            encoding="utf-8",
            decode_responses=True,
            max_connections=config.redis.max_connections
        )
        logger.info(f"Redis connection established to {config.redis.host}:{config.redis.port}")
        return redis_client
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        return None

def init_influxdb():
    """Initialize InfluxDB connection"""
    global influxdb_client, influxdb_write_api, influxdb_query_api
    try:
        influxdb_url = os.getenv("INFLUXDB_URL", "http://localhost:8086")
        influxdb_token = os.getenv("INFLUXDB_TOKEN", "")
        influxdb_org = os.getenv("INFLUXDB_ORG", "ddosai")
        influxdb_bucket = os.getenv("INFLUXDB_BUCKET", "ddosai_metrics")
        
        influxdb_client = InfluxDBClient(
            url=influxdb_url,
            token=influxdb_token,
            org=influxdb_org
        )
        
        influxdb_write_api = influxdb_client.write_api(write_options=SYNCHRONOUS)
        influxdb_query_api = influxdb_client.query_api()
        
        logger.info(f"InfluxDB connection established to {influxdb_url}")
        return influxdb_client
    except Exception as e:
        logger.error(f"Failed to connect to InfluxDB: {e}")
        return None

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def get_redis() -> aioredis.Redis:
    """Get Redis client"""
    if redis_client is None:
        await init_redis()
    return redis_client

def get_influxdb_write_api():
    """Get InfluxDB write API"""
    if influxdb_write_api is None:
        init_influxdb()
    return influxdb_write_api

def get_influxdb_query_api():
    """Get InfluxDB query API"""
    if influxdb_query_api is None:
        init_influxdb()
    return influxdb_query_api

async def close_db_connections():
    """Close all database connections"""
    global redis_client, influxdb_client
    
    # Close Redis connection
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")
    
    # Close InfluxDB connection
    if influxdb_client:
        influxdb_client.close()
        logger.info("InfluxDB connection closed")