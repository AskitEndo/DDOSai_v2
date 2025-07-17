# DDoS.AI Architecture Documentation

This document provides a comprehensive overview of the DDoS.AI platform architecture, including system components, data flow, and design decisions.

## System Architecture Overview

DDoS.AI is built using a microservices architecture with the following main components:

1. **Backend API Service**: FastAPI-based REST and WebSocket API
2. **AI Engine**: Core detection and analysis components
3. **Frontend Application**: React-based user interface
4. **Monitoring System**: Prometheus and Grafana for metrics
5. **Simulation Engine**: Traffic and attack simulation

### Architecture Diagram

```mermaid
graph TD
    subgraph "Frontend"
        UI[React UI]
        State[State Management]
        Components[UI Components]
    end

    subgraph "Backend"
        API[FastAPI Service]
        AIEngine[AI Engine]
        Ingestion[Traffic Ingestion]
        Storage[Data Storage]
    end

    subgraph "AI Models"
        Autoencoder[Autoencoder Detector]
        GNN[Graph Neural Network]
        RL[Reinforcement Learning]
        XAI[Explainable AI]
    end

    subgraph "Monitoring"
        Prometheus[Prometheus]
        Grafana[Grafana]
        Metrics[Metrics Collector]
    end

    subgraph "Simulation"
        SimEngine[Simulation Engine]
        AttackGen[Attack Generator]
    end

    %% Frontend connections
    UI --> State
    State --> Components
    UI -- WebSocket --> API
    UI -- REST API --> API

    %% Backend connections
    API --> AIEngine
    API --> Storage
    API --> Ingestion
    API --> Metrics

    %% AI Engine connections
    AIEngine --> Autoencoder
    AIEngine --> GNN
    AIEngine --> RL
    AIEngine --> XAI

    %% Monitoring connections
    API --> Prometheus
    Prometheus --> Grafana
    Metrics --> Prometheus

    %% Simulation connections
    SimEngine --> AttackGen
    AttackGen --> Ingestion
```

## Component Details

### Backend API Service

The backend API service is built using FastAPI, a modern, high-performance web framework for building APIs with Python. It provides both REST API endpoints and WebSocket connections for real-time updates.

#### Key Features

- Asynchronous request handling for high throughput
- WebSocket support for real-time updates
- Comprehensive error handling and validation
- Middleware for logging, rate limiting, and circuit breaking
- Health check and metrics endpoints

#### API Endpoints

The API provides the following main endpoints:

- `/api/analyze`: Analyze network packets for threats
- `/api/explain/{prediction_id}`: Get explanations for predictions
- `/api/detections`: Get recent detection results
- `/api/graph/current`: Get current network graph state
- `/api/metrics`: Get system performance metrics
- `/api/simulate/start`: Start attack simulation
- `/api/simulate/stop`: Stop attack simulation

### AI Engine

The AI Engine is the core of the DDoS.AI platform, responsible for analyzing network traffic and detecting threats. It orchestrates multiple AI models to achieve high accuracy and explainability.

#### Architecture

```mermaid
graph TD
    subgraph "AI Engine"
        Orchestrator[AI Engine Orchestrator]
        FeatureExtractor[Feature Extractor]
        ModelManager[Model Manager]
        ResultAggregator[Result Aggregator]
    end

    subgraph "Models"
        Autoencoder[Autoencoder Detector]
        GNN[Graph Neural Network]
        RL[RL Threat Scorer]
        XAI[XAI Explainer]
    end

    %% Data flow
    Input[Traffic Data] --> FeatureExtractor
    FeatureExtractor --> Orchestrator
    Orchestrator --> ModelManager
    ModelManager --> Autoencoder
    ModelManager --> GNN
    ModelManager --> RL
    Autoencoder --> ResultAggregator
    GNN --> ResultAggregator
    RL --> ResultAggregator
    ResultAggregator --> XAI
    ResultAggregator --> Output[Detection Result]
```

#### Model Details

1. **Autoencoder Detector**

   - Architecture: 64→32→16→32→64 neurons
   - Training: Unsupervised learning on normal traffic
   - Detection: Reconstruction error threshold
   - Purpose: Anomaly detection based on packet features

2. **Graph Neural Network (GNN) Analyzer**

   - Architecture: 2-layer Graph Convolutional Network
   - Training: Semi-supervised learning on labeled graphs
   - Detection: Node classification for malicious probability
   - Purpose: Network topology analysis and pattern detection

3. **Reinforcement Learning (RL) Threat Scorer**

   - Architecture: Deep Q-Network
   - Training: Reinforcement learning with rewards for accurate detection
   - Detection: Threat score assignment (0-100)
   - Purpose: Adaptive threat scoring based on context

4. **Explainable AI (XAI) Module**
   - Methods: SHAP and LIME integration
   - Purpose: Provide explanations for model decisions
   - Features: Feature importance, counterfactuals, decision boundaries

### Frontend Application

The frontend application is built using React with TypeScript, providing a responsive and interactive user interface for monitoring and analyzing network traffic.

#### Architecture

```mermaid
graph TD
    subgraph "Frontend Architecture"
        Router[React Router]
        Store[State Management]
        API[API Client]
        WebSocket[WebSocket Client]
    end

    subgraph "Pages"
        Dashboard[Dashboard]
        Simulation[Simulation]
        Analytics[Analytics]
        Settings[Settings]
    end

    subgraph "Components"
        NetworkGraph[Network Graph]
        ThreatPanel[Threat Score Panel]
        XAIPanel[XAI Panel]
        MetricsPanel[Metrics Panel]
        PerformanceMonitor[Performance Monitor]
    end

    %% Connections
    Router --> Pages
    Router --> Dashboard
    Router --> Simulation
    Router --> Analytics
    Router --> Settings

    Dashboard --> NetworkGraph
    Dashboard --> ThreatPanel
    Dashboard --> XAIPanel
    Dashboard --> MetricsPanel
    Dashboard --> PerformanceMonitor

    Components --> Store
    Store --> API
    Store --> WebSocket
```

#### Key Components

1. **Dashboard**: Main monitoring view with real-time updates
2. **Network Graph**: D3.js-based visualization of network traffic
3. **Threat Score Panel**: Display of threat levels and recent detections
4. **XAI Panel**: Visualization of model explanations
5. **Metrics Panel**: Display of system performance metrics
6. **Performance Monitor**: Detailed system and model performance monitoring

### Monitoring System

The monitoring system is built using Prometheus and Grafana, providing comprehensive metrics collection and visualization.

#### Architecture

```mermaid
graph TD
    subgraph "Metrics Collection"
        MetricsCollector[Metrics Collector]
        PrometheusClient[Prometheus Client]
    end

    subgraph "Prometheus"
        PrometheusServer[Prometheus Server]
        TSDB[Time Series Database]
        AlertManager[Alert Manager]
    end

    subgraph "Grafana"
        Dashboards[Dashboards]
        DataSources[Data Sources]
        Alerts[Alerts]
    end

    %% Connections
    MetricsCollector --> PrometheusClient
    PrometheusClient --> PrometheusServer
    PrometheusServer --> TSDB
    PrometheusServer --> AlertManager
    TSDB --> DataSources
    DataSources --> Dashboards
    AlertManager --> Alerts
```

#### Metrics Collected

1. **System Metrics**

   - CPU usage
   - Memory usage
   - Disk usage
   - Network I/O

2. **Application Metrics**

   - Request count and rate
   - Response time
   - Error rate
   - Active connections

3. **AI Model Metrics**

   - Inference time
   - Accuracy
   - Confidence
   - Feature importance

4. **Traffic Metrics**
   - Packet count
   - Packet size distribution
   - Protocol distribution
   - Threat level

### Simulation Engine

The simulation engine allows for generating synthetic traffic and attacks for testing and training purposes.

#### Architecture

```mermaid
graph TD
    subgraph "Simulation Engine"
        Controller[Simulation Controller]
        TrafficGen[Traffic Generator]
        AttackGen[Attack Generator]
        ConfigManager[Configuration Manager]
    end

    subgraph "Attack Types"
        SYNFlood[SYN Flood]
        UDPFlood[UDP Flood]
        HTTPFlood[HTTP Flood]
        DNSAmp[DNS Amplification]
    end

    %% Connections
    Controller --> TrafficGen
    Controller --> AttackGen
    Controller --> ConfigManager
    AttackGen --> SYNFlood
    AttackGen --> UDPFlood
    AttackGen --> HTTPFlood
    AttackGen --> DNSAmp
    TrafficGen --> Output[Traffic Output]
    AttackGen --> Output
```

#### Simulation Capabilities

1. **Normal Traffic Generation**

   - Web browsing patterns
   - DNS queries
   - SSH sessions
   - Email traffic

2. **Attack Simulation**

   - SYN flood
   - UDP flood
   - HTTP flood
   - DNS amplification
   - NTP amplification

3. **Configuration Options**
   - Traffic rate
   - Attack duration
   - Target selection
   - Packet characteristics

## Data Flow

### Traffic Analysis Flow

```mermaid
sequenceDiagram
    participant Traffic Source
    participant Ingestion
    participant Feature Extractor
    participant AI Engine
    participant Storage
    participant WebSocket
    participant UI

    Traffic Source->>Ingestion: Network Packets
    Ingestion->>Feature Extractor: Raw Packet Data
    Feature Extractor->>AI Engine: Feature Vectors

    AI Engine->>AI Engine: Autoencoder Detection
    AI Engine->>AI Engine: GNN Analysis
    AI Engine->>AI Engine: RL Threat Scoring
    AI Engine->>AI Engine: Consensus Decision

    AI Engine->>Storage: Store Detection Result
    AI Engine->>WebSocket: Broadcast Detection
    WebSocket->>UI: Real-time Update

    UI->>UI: Update Network Graph
    UI->>UI: Update Threat Panel
```

### Explanation Request Flow

```mermaid
sequenceDiagram
    participant UI
    participant API
    participant XAI Module
    participant Model Cache
    participant Feature Cache

    UI->>API: GET /api/explain/{prediction_id}
    API->>Model Cache: Get Prediction
    Model Cache->>API: Prediction Data
    API->>Feature Cache: Get Features
    Feature Cache->>API: Feature Data
    API->>XAI Module: Generate Explanation
    XAI Module->>XAI Module: SHAP Analysis
    XAI Module->>XAI Module: LIME Analysis
    XAI Module->>API: Explanation Data
    API->>UI: Explanation Response
    UI->>UI: Render Explanation
```

## Deployment Architecture

DDoS.AI can be deployed in various configurations depending on the requirements. Here are the main deployment options:

### Docker Compose Deployment

```mermaid
graph TD
    subgraph "Docker Compose"
        Backend[Backend Container]
        Frontend[Frontend Container]
        Prometheus[Prometheus Container]
        Grafana[Grafana Container]
        Redis[Redis Container]
    end

    Frontend -- HTTP --> Backend
    Frontend -- WebSocket --> Backend
    Backend --> Redis
    Backend --> Prometheus
    Prometheus --> Grafana
```

### Kubernetes Deployment

```mermaid
graph TD
    subgraph "Kubernetes Cluster"
        subgraph "Frontend Namespace"
            FrontendPod[Frontend Pods]
            FrontendService[Frontend Service]
            FrontendIngress[Ingress]
        end

        subgraph "Backend Namespace"
            BackendPod[Backend Pods]
            BackendService[Backend Service]
            BackendHPA[HPA]
        end

        subgraph "Monitoring Namespace"
            PrometheusPod[Prometheus]
            GrafanaPod[Grafana]
            AlertManagerPod[Alert Manager]
        end

        subgraph "Data Namespace"
            RedisPod[Redis]
            PostgresPod[PostgreSQL]
        end
    end

    FrontendIngress --> FrontendService
    FrontendService --> FrontendPod
    FrontendPod --> BackendService
    BackendService --> BackendPod
    BackendPod --> RedisPod
    BackendPod --> PostgresPod
    BackendPod --> PrometheusPod
    PrometheusPod --> GrafanaPod
    PrometheusPod --> AlertManagerPod
    BackendHPA --> BackendPod
```

## Security Architecture

DDoS.AI implements multiple layers of security to protect the system and data:

### Authentication and Authorization

```mermaid
graph TD
    subgraph "Authentication"
        APIKey[API Key]
        JWT[JWT Token]
        OAuth[OAuth 2.0]
    end

    subgraph "Authorization"
        RBAC[Role-Based Access Control]
        Policies[Access Policies]
    end

    subgraph "Security Middleware"
        RateLimit[Rate Limiting]
        InputValidation[Input Validation]
        CORS[CORS]
        TLS[TLS/SSL]
    end

    Request[API Request] --> APIKey
    APIKey --> JWT
    JWT --> RBAC
    RBAC --> Policies
    Policies --> Protected[Protected Resource]

    Request --> RateLimit
    Request --> InputValidation
    Request --> CORS
    Request --> TLS
```

### Data Protection

1. **In Transit**: TLS/SSL encryption for all API communications
2. **At Rest**: Encrypted storage for sensitive data
3. **Input Validation**: Comprehensive validation of all API inputs
4. **Output Sanitization**: Proper encoding of all API responses

## Performance Considerations

### Scalability

The DDoS.AI platform is designed to scale horizontally to handle increasing traffic loads:

1. **Stateless Backend**: The backend API is stateless, allowing for easy scaling
2. **Redis Caching**: Redis is used for caching and session management
3. **Batch Processing**: AI models support batch processing for efficient inference
4. **Async Processing**: Asynchronous processing for non-blocking operations

### Performance Optimization

1. **Model Optimization**: AI models are optimized for inference speed
2. **Circuit Breaker**: Circuit breaker pattern for handling failures
3. **Connection Pooling**: Database and Redis connection pooling
4. **Efficient Data Structures**: Optimized data structures for memory efficiency

## Design Decisions

### Technology Stack Selection

1. **FastAPI**: Chosen for its high performance, async support, and automatic documentation
2. **React**: Chosen for its component-based architecture and efficient rendering
3. **PyTorch**: Chosen for its flexibility and support for custom neural network architectures
4. **Prometheus**: Chosen for its scalable metrics collection and alerting capabilities

### AI Model Selection

1. **Autoencoder**: Chosen for its ability to detect anomalies in high-dimensional data
2. **Graph Neural Network**: Chosen for its ability to analyze network topology
3. **Reinforcement Learning**: Chosen for its adaptive learning capabilities
4. **SHAP/LIME**: Chosen for their ability to explain complex model decisions

## Future Enhancements

1. **Federated Learning**: Implement federated learning for collaborative model training
2. **Transfer Learning**: Implement transfer learning for faster model adaptation
3. **Automated Model Retraining**: Implement automated retraining based on performance metrics
4. **Advanced Visualization**: Enhance visualization capabilities with 3D network graphs
5. **Integration with SIEM**: Integrate with Security Information and Event Management systems
