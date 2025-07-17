# DDoS.AI User Guide

This guide provides step-by-step instructions for setting up and using the DDoS.AI platform.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Dashboard Overview](#dashboard-overview)
5. [Traffic Analysis](#traffic-analysis)
6. [Attack Detection](#attack-detection)
7. [Explainable AI Features](#explainable-ai-features)
8. [Network Visualization](#network-visualization)
9. [Attack Simulation](#attack-simulation)
10. [Performance Monitoring](#performance-monitoring)
11. [Troubleshooting](#troubleshooting)
12. [FAQ](#faq)

## Introduction

DDoS.AI is an advanced platform for detecting and analyzing Distributed Denial of Service (DDoS) attacks using artificial intelligence. The platform combines multiple AI models including autoencoders, graph neural networks, and reinforcement learning to provide accurate detection with explainable results.

### Key Features

- Real-time traffic analysis and anomaly detection
- Multiple AI models working in consensus for high accuracy
- Explainable AI that provides insights into detection decisions
- Interactive network graph visualization
- Attack simulation capabilities for testing and training
- Comprehensive performance monitoring

## Installation

### Prerequisites

- Docker and Docker Compose
- 8GB RAM minimum (16GB recommended)
- 4 CPU cores minimum
- 20GB free disk space

### Quick Start with Docker

1. Clone the repository:

```bash
git clone https://github.com/yourusername/ddosai-platform.git
cd ddosai-platform
```

2. Start the platform using Docker Compose:

```bash
docker-compose up -d
```

3. Access the dashboard at `http://localhost:3000`

### Manual Installation

#### Backend Setup

1. Create a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install backend dependencies:

```bash
cd backend
pip install -r requirements.txt
```

3. Start the backend server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Frontend Setup

1. Install Node.js dependencies:

```bash
cd frontend
npm install
```

2. Start the development server:

```bash
npm run dev
```

3. Access the dashboard at `http://localhost:5173`

## Configuration

### Environment Variables

The platform can be configured using environment variables or a `.env` file:

#### Backend Configuration

Create a `.env` file in the `backend` directory:

```
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false
LOG_LEVEL=INFO
JSON_LOGS=true

# Security
API_KEY_HEADER=X-API-Key
API_KEY=your_secret_api_key_here

# AI Model Configuration
MODEL_THRESHOLD=0.75
AUTOENCODER_THRESHOLD_PERCENTILE=95.0
GNN_CONFIDENCE_THRESHOLD=0.6
THREAT_SCORE_THRESHOLD=50

# Performance
MAX_WORKERS=4
BATCH_SIZE=32
```

#### Frontend Configuration

Create a `.env` file in the `frontend` directory:

```
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
VITE_REFRESH_INTERVAL=5000
```

### Advanced Configuration

For advanced configuration options, refer to the `config.yaml` files in the respective directories.

## Dashboard Overview

The DDoS.AI dashboard provides a comprehensive view of your network traffic and security status.

### Main Components

1. **Header Bar**: Contains the platform name, navigation menu, and user settings.
2. **Network Graph**: Interactive visualization of your network traffic.
3. **Threat Score Panel**: Shows the current threat level and recent detections.
4. **XAI Panel**: Displays explanations for detected threats.
5. **Metrics Panel**: Shows system performance metrics.
6. **Logs Panel**: Displays recent system and detection logs.

### Navigation

- **Dashboard**: Main monitoring view
- **Simulation**: Attack simulation controls
- **Analytics**: Historical data and trends
- **Settings**: Platform configuration

## Traffic Analysis

DDoS.AI continuously analyzes network traffic to detect anomalies and potential attacks.

### Traffic Sources

The platform can ingest traffic from multiple sources:

1. **Live Network Interface**: Capture traffic directly from a network interface.
2. **PCAP Files**: Analyze pre-recorded packet capture files.
3. **CSV Data**: Import traffic data from CSV files.
4. **Simulation**: Generate synthetic traffic for testing.

### Analysis Process

1. **Feature Extraction**: Extract relevant features from network packets.
2. **Anomaly Detection**: Identify anomalous traffic patterns using autoencoder.
3. **Graph Analysis**: Analyze network topology using graph neural networks.
4. **Threat Scoring**: Assign threat scores using reinforcement learning.
5. **Consensus Decision**: Combine results from multiple models for final decision.

## Attack Detection

DDoS.AI can detect various types of DDoS attacks:

### Supported Attack Types

- **SYN Flood**: TCP SYN packet flooding
- **UDP Flood**: UDP packet flooding
- **HTTP Flood**: HTTP request flooding
- **ICMP Flood**: ICMP packet flooding
- **DNS Amplification**: DNS reflection and amplification attacks
- **NTP Amplification**: NTP reflection and amplification attacks

### Detection Methods

The platform uses multiple detection methods:

1. **Statistical Analysis**: Detect anomalies based on statistical properties.
2. **Machine Learning**: Use trained models to identify attack patterns.
3. **Behavioral Analysis**: Monitor changes in network behavior over time.
4. **Signature-Based**: Match known attack signatures.

### Alert Configuration

Configure alert thresholds in the Settings page:

1. Navigate to **Settings > Alerts**.
2. Set the **Threat Score Threshold** (0-100).
3. Configure **Notification Methods** (Email, Slack, Webhook).
4. Set **Alert Frequency** to control how often alerts are sent.

## Explainable AI Features

DDoS.AI provides explanations for its detection decisions, making it easier to understand and trust the system.

### Explanation Types

1. **Feature Importance**: Shows which features contributed most to the decision.
2. **Counterfactuals**: Shows how changing certain features would affect the decision.
3. **Decision Boundary**: Visualizes the decision boundary between normal and malicious traffic.
4. **Similar Cases**: Shows similar historical cases for comparison.

### Using Explanations

1. When a threat is detected, click on it in the Threat Score Panel.
2. The XAI Panel will show the explanation for that detection.
3. Use the tabs to switch between different explanation types.
4. Hover over features to see more details.

## Network Visualization

The Network Graph provides a visual representation of your network traffic.

### Graph Elements

- **Nodes**: Represent IP addresses (sources and targets).
- **Edges**: Represent connections between nodes.
- **Colors**: Indicate threat levels (green to red).
- **Size**: Represents traffic volume.

### Interaction

- **Zoom**: Use mouse wheel or pinch gesture.
- **Pan**: Click and drag the background.
- **Select Node**: Click on a node to see details.
- **Filter**: Use the filter controls to show specific traffic.
- **Time Window**: Adjust the time window using the slider.

## Attack Simulation

DDoS.AI includes a simulation module for testing detection capabilities.

### Running a Simulation

1. Navigate to the **Simulation** page.
2. Select an **Attack Type** from the dropdown.
3. Configure simulation parameters:
   - **Target IP**: The target of the attack.
   - **Target Port**: The port to attack.
   - **Duration**: How long the attack should run.
   - **Packet Rate**: Packets per second.
   - **Packet Size**: Size of each packet.
4. Click **Start Simulation** to begin.
5. Monitor the results in the Dashboard.
6. Click **Stop Simulation** to end early if needed.

### Simulation Scenarios

The platform includes several pre-configured scenarios:

- **Basic SYN Flood**: Simple SYN flood attack.
- **Distributed UDP Flood**: UDP flood from multiple sources.
- **Low-and-Slow HTTP**: Slow HTTP attack that's harder to detect.
- **Mixed Attack**: Combination of different attack types.

## Performance Monitoring

DDoS.AI includes comprehensive performance monitoring to ensure the platform is operating efficiently.

### Metrics Available

- **CPU Usage**: Current and historical CPU utilization.
- **Memory Usage**: Current and historical memory utilization.
- **Packet Processing Rate**: Packets processed per second.
- **Processing Latency**: Time taken to analyze each packet.
- **Model Inference Times**: Time taken by each AI model.
- **Error Rates**: Percentage of errors during processing.

### Viewing Metrics

1. The basic metrics are shown in the Metrics Panel on the Dashboard.
2. For detailed metrics, click **Detailed View** in the Metrics Panel.
3. For historical metrics, navigate to **Analytics > Performance**.

### Performance Optimization

If you notice performance issues:

1. Increase the `BATCH_SIZE` in the configuration.
2. Adjust the `MAX_WORKERS` based on your CPU cores.
3. Consider enabling GPU acceleration if available.
4. Reduce the sampling rate for high-volume traffic.

## Troubleshooting

### Common Issues

#### Backend Won't Start

- Check if the required ports are available (8000 for API).
- Ensure Python dependencies are installed correctly.
- Check the logs for specific error messages.

#### Frontend Connection Issues

- Verify the backend is running and accessible.
- Check the `VITE_API_URL` in the frontend `.env` file.
- Check browser console for CORS or network errors.

#### High CPU Usage

- Reduce the packet processing rate.
- Increase the batch size for more efficient processing.
- Consider scaling horizontally with multiple instances.

#### False Positives/Negatives

- Adjust the model thresholds in the configuration.
- Provide feedback on detections to improve the models.
- Consider retraining the models with your specific traffic patterns.

### Logs

Logs are available in several locations:

- **Docker**: `docker-compose logs -f`
- **Backend**: `backend/logs/ddosai.log`
- **Frontend**: Browser console
- **Dashboard**: Logs panel in the UI

## FAQ

### General Questions

**Q: How accurate is the detection?**  
A: The platform achieves over 95% accuracy on most attack types, with lower false positive rates compared to traditional systems due to the consensus approach of multiple AI models.

**Q: Can DDoS.AI prevent attacks or only detect them?**  
A: The primary function is detection and analysis. For prevention, you can integrate with mitigation systems using the API.

**Q: How much traffic can the platform handle?**  
A: The standard configuration can handle up to 10,000 packets per second. For higher volumes, consider the distributed deployment option.

### Technical Questions

**Q: Does DDoS.AI support IPv6?**  
A: Yes, the platform fully supports both IPv4 and IPv6 traffic.

**Q: Can I add my own detection models?**  
A: Yes, the platform supports custom model integration. See the developer documentation for details.

**Q: Is GPU support required?**  
A: No, the platform runs on CPU by default, but GPU acceleration is supported for higher performance.

**Q: Can I export the detection results?**  
A: Yes, results can be exported in JSON, CSV, or PCAP formats from the Analytics page.
