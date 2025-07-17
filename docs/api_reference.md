# DDoS.AI API Reference

This document provides a comprehensive reference for the DDoS.AI platform API endpoints.

## Base URL

All API endpoints are relative to the base URL:

```
http://localhost:8000
```

For production deployments, replace with your actual domain.

## Authentication

Currently, the API uses simple API key authentication. Include your API key in the request header:

```
X-API-Key: your_api_key_here
```

## Common Response Formats

All API responses follow a standard format:

### Success Response

```json
{
  "status": "success",
  "data": { ... },
  "timestamp": "2023-07-17T10:15:30.123Z"
}
```

### Error Response

```json
{
  "status": "error",
  "error": "Error message",
  "error_code": "ERROR_CODE",
  "details": { ... },
  "timestamp": "2023-07-17T10:15:30.123Z"
}
```

## API Endpoints

### Health Check

#### GET /health

Check if the API is running.

**Parameters:**

- `detailed` (optional): Boolean. If true, returns detailed health information.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2023-07-17T10:15:30.123Z",
  "version": "1.0.0"
}
```

With `detailed=true`:

```json
{
  "status": "healthy",
  "timestamp": "2023-07-17T10:15:30.123Z",
  "version": "1.0.0",
  "components": {
    "ai_detection": "healthy",
    "database": "healthy",
    "cache": "healthy"
  },
  "system": {
    "cpu_usage": 25.5,
    "memory_usage": 42.3,
    "disk_usage": 68.7
  }
}
```

### Traffic Analysis

#### POST /api/analyze

Analyze a network packet for threats.

**Request Body:**

```json
{
  "src_ip": "192.168.1.100",
  "dst_ip": "10.0.0.1",
  "src_port": 12345,
  "dst_port": 80,
  "protocol": "TCP",
  "flags": ["SYN"],
  "packet_size": 64,
  "ttl": 64,
  "payload_entropy": 0.5,
  "timestamp": "2023-07-17T10:15:30.123Z",
  "packet_id": "pkt_a1b2c3d4"
}
```

**Response:**

```json
{
  "packet_id": "pkt_a1b2c3d4",
  "timestamp": "2023-07-17T10:15:30.123Z",
  "is_malicious": true,
  "confidence": 0.95,
  "threat_score": 85,
  "attack_type": "SYN_FLOOD",
  "detection_method": "weighted_consensus",
  "model_version": "1.0.0",
  "explanation": {
    "model_results": {
      "autoencoder": {
        "is_malicious": true,
        "confidence": 0.92,
        "reconstruction_error": 0.15
      },
      "gnn": {
        "is_malicious": true,
        "confidence": 0.88,
        "malicious_probability": 0.94
      },
      "rl": {
        "is_malicious": true,
        "confidence": 0.85,
        "threat_score": 85
      }
    }
  }
}
```

### Explanations

#### GET /api/explain/{prediction_id}

Get a detailed explanation for a prediction.

**Parameters:**

- `prediction_id`: String. The ID of the prediction to explain.

**Response:**

```json
{
  "prediction_id": "pkt_a1b2c3d4",
  "explanation": {
    "feature_importance": [
      {"feature": "src_port", "importance": 0.35},
      {"feature": "flags", "importance": 0.25},
      {"feature": "packet_size", "importance": 0.15},
      {"feature": "payload_entropy", "importance": 0.10},
      {"feature": "protocol", "importance": 0.08},
      {"feature": "ttl", "importance": 0.07}
    ],
    "top_features": [
      {"feature": "src_port", "value": 12345, "contribution": "high"},
      {"feature": "flags", "value": ["SYN"], "contribution": "high"},
      {"feature": "packet_size", "value": 64, "contribution": "medium"}
    ],
    "counterfactuals": [
      {
        "feature": "flags",
        "original_value": ["SYN"],
        "counterfactual_value": ["SYN", "ACK"],
        "new_prediction": "benign",
        "confidence": 0.75
      }
    ],
    "visualization_data": {
      "heatmap": [...],
      "decision_boundary": [...]
    }
  },
  "timestamp": "2023-07-17T10:15:35.456Z"
}
```

### Detection History

#### GET /api/detections

Get recent detection results.

**Parameters:**

- `limit` (optional): Integer. Maximum number of detections to return. Default: 50.
- `offset` (optional): Integer. Number of detections to skip. Default: 0.
- `malicious_only` (optional): Boolean. If true, returns only malicious detections. Default: false.

**Response:**

```json
[
  {
    "packet_id": "pkt_a1b2c3d4",
    "timestamp": "2023-07-17T10:15:30.123Z",
    "is_malicious": true,
    "confidence": 0.95,
    "threat_score": 85,
    "attack_type": "SYN_FLOOD"
  },
  {
    "packet_id": "pkt_e5f6g7h8",
    "timestamp": "2023-07-17T10:15:29.456Z",
    "is_malicious": false,
    "confidence": 0.88,
    "threat_score": 15,
    "attack_type": "BENIGN"
  }
]
```

### Network Graph

#### GET /api/graph/current

Get the current network graph state.

**Response:**

```json
{
  "nodes": [
    {
      "id": "192.168.1.100",
      "type": "source",
      "threat_level": 0.85,
      "connections": 15,
      "packets_sent": 150,
      "packets_received": 120
    },
    {
      "id": "10.0.0.1",
      "type": "target",
      "threat_level": 0.1,
      "connections": 5,
      "packets_sent": 120,
      "packets_received": 150
    }
  ],
  "edges": [
    {
      "source": "192.168.1.100",
      "target": "10.0.0.1",
      "packets": 150,
      "bytes": 9600,
      "protocols": ["TCP"],
      "is_malicious": true
    }
  ],
  "timestamp": "2023-07-17T10:15:30.123Z"
}
```

### System Metrics

#### GET /api/metrics

Get system performance metrics.

**Parameters:**

- `detailed` (optional): Boolean. If true, returns detailed metrics. Default: false.

**Response:**

```json
{
  "timestamp": "2023-07-17T10:15:30.123Z",
  "packets_processed": 15000,
  "processing_latency_ms": 25,
  "cpu_usage": 35.5,
  "memory_usage": 42.3,
  "active_connections": 5,
  "threat_level": 2,
  "malicious_packets": 150,
  "total_detections": 15000
}
```

With `detailed=true`:

```json
{
  "timestamp": "2023-07-17T10:15:30.123Z",
  "uptime": 3600,
  "packets": {
    "total": 15000,
    "malicious": 150,
    "benign": 14850,
    "malicious_ratio": 0.01
  },
  "processing": {
    "avg_time": 0.025,
    "avg_time_ms": 25,
    "throughput": 250
  },
  "models": {
    "autoencoder": {
      "avg_time": 0.015,
      "min_time": 0.01,
      "max_time": 0.05,
      "count": 15000
    },
    "gnn": {
      "avg_time": 0.02,
      "min_time": 0.015,
      "max_time": 0.06,
      "count": 15000
    },
    "rl_threat_scorer": {
      "avg_time": 0.018,
      "min_time": 0.012,
      "max_time": 0.055,
      "count": 15000
    }
  },
  "errors": {
    "count": 25,
    "rate": 0.17
  },
  "threat_level": 2,
  "system": {
    "cpu_usage": 35.5,
    "memory_usage": 42.3,
    "disk_usage": 68.7
  }
}
```

### Attack Simulation

#### POST /api/simulate/start

Start an attack simulation.

**Request Body:**

```json
{
  "attack_type": "SYN_FLOOD",
  "target_ip": "10.0.0.1",
  "target_port": 80,
  "duration": 60,
  "packet_rate": 1000,
  "packet_size": 64
}
```

**Response:**

```json
{
  "status": "started",
  "simulation_id": "sim_1234",
  "message": "Started SYN_FLOOD simulation",
  "timestamp": "2023-07-17T10:15:30.123Z"
}
```

#### POST /api/simulate/stop

Stop an attack simulation.

**Request Body:**

```json
{
  "simulation_id": "sim_1234"
}
```

**Response:**

```json
{
  "status": "stopped",
  "simulation_id": "sim_1234",
  "message": "Simulation stopped",
  "timestamp": "2023-07-17T10:15:35.456Z"
}
```

## WebSocket API

The DDoS.AI platform also provides real-time updates via WebSocket connections.

### Connection

Connect to the WebSocket endpoint:

```
ws://localhost:8000/ws
```

### Message Types

#### Detection Updates

```json
{
  "type": "detection",
  "data": {
    "packet_id": "pkt_a1b2c3d4",
    "timestamp": "2023-07-17T10:15:30.123Z",
    "is_malicious": true,
    "confidence": 0.95,
    "threat_score": 85,
    "attack_type": "SYN_FLOOD"
  }
}
```

#### Graph Updates

```json
{
  "type": "graph_update",
  "data": {
    "nodes": [...],
    "edges": [...],
    "timestamp": "2023-07-17T10:15:30.123Z"
  }
}
```

#### System Alerts

```json
{
  "type": "alert",
  "data": {
    "level": "warning",
    "message": "High CPU usage detected",
    "details": {
      "cpu_usage": 85.5,
      "threshold": 80.0
    },
    "timestamp": "2023-07-17T10:15:30.123Z"
  }
}
```

## Error Codes

| Code                   | Description                        |
| ---------------------- | ---------------------------------- |
| `VALIDATION_ERROR`     | Invalid request parameters or body |
| `AUTHENTICATION_ERROR` | Invalid or missing API key         |
| `AUTHORIZATION_ERROR`  | Insufficient permissions           |
| `RESOURCE_NOT_FOUND`   | Requested resource not found       |
| `INTERNAL_ERROR`       | Internal server error              |
| `MODEL_ERROR`          | AI model inference error           |
| `RATE_LIMIT_EXCEEDED`  | Too many requests                  |
| `SIMULATION_ERROR`     | Error in attack simulation         |

## Rate Limiting

The API enforces rate limits to prevent abuse. Current limits:

- 100 requests per minute for `/api/analyze` endpoint
- 300 requests per minute for other endpoints

When rate limit is exceeded, the API returns a 429 status code with a `Retry-After` header indicating when to retry.
