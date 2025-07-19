# DDoS.AI Platform: Complete Guide

## 🎯 Latest Updates - Cross-Device Attack Monitoring

### New Features Added in Version 2.0

**🌐 Real Cross-Device Attack Testing**

- **Target Server Setup**: Use `start_target_server.bat` to create attack targets on separate machines
- **Live Network Monitoring**: Real-time packet capture and analysis using `psutil`
- **Cross-Device Detection**: Attacks from PC A instantly visible on PC B's dashboard
- **WebSocket Broadcasting**: Real-time attack notifications across all connected clients

**📡 Enhanced Network Monitoring Panel**

- **Live Statistics**: Real bytes sent/received, packet counts, error rates
- **Attack Visualization**: Source/destination IPs, protocols, severity levels
- **Auto-Refresh**: 3-second intervals during active monitoring
- **Attack History**: Chronological list of detected attacks with full details

**🔄 Unified Data Architecture**

- **Backend-First Loading**: Single source of truth eliminating duplicate dummy data
- **Offline Fallback**: Works without backend connection using local dummy data
- **Consistent Data Types**: All sources mapped to unified TypeScript interfaces
- **Real-Time Sync**: WebSocket updates across all dashboard components

This guide provides comprehensive information on how to run, train, monitor, and use the DDoS.AI platform with the new cross-device attack monitoring capabilities.

## Table of Contents

- [Cross-Device Attack Testing](#cross-device-attack-testing)
- [Running the Platform](#running-the-platform)
  - [Full Platform Setup](#full-platform-setup)
  - [Target Server Setup](#target-server-setup)
  - [Development Mode](#development-mode)
- [Understanding the Platform](#understanding-the-platform)
  - [Core Components](#core-components)
  - [Cross-Device Architecture](#cross-device-architecture)
  - [Real-Time Data Flow](#real-time-data-flow)
- [Network Monitoring](#network-monitoring)
  - [Live Traffic Analysis](#live-traffic-analysis)
  - [Attack Detection](#attack-detection)
  - [Cross-Device Alerts](#cross-device-alerts)
- [Training the AI Models](#training-the-ai-models)
  - [Using Sample Data](#using-sample-data)
  - [Adding Your Own Data](#adding-your-own-data)
  - [Custom Model Integration](#custom-model-integration)
- [Running Real Attack Simulations](#running-real-attack-simulations)
  - [Cross-Device Setup](#cross-device-setup)
  - [Attack Configuration](#attack-configuration)
  - [Monitoring Impact](#monitoring-impact)
- [Dashboard Usage](#dashboard-usage)
  - [Network Monitoring Panel](#network-monitoring-panel)
  - [Attack Simulation Interface](#attack-simulation-interface)
  - [Real-Time Visualization](#real-time-visualization)
- [Monitoring & Analytics](#monitoring--analytics)
  - [Prometheus Metrics](#prometheus-metrics)
  - [Grafana Dashboards](#grafana-dashboards)
  - [InfluxDB Time Series](#influxdb-time-series)
- [Troubleshooting](#troubleshooting)
  - [Cross-Device Issues](#cross-device-issues)
  - [Network Monitoring Problems](#network-monitoring-problems)
  - [Performance Tuning](#performance-tuning)

## Cross-Device Attack Testing

### Overview

The DDoS.AI platform now supports real cross-device attack testing where you can launch actual network attacks from one PC (PC A) and monitor their impact on another PC (PC B) in real-time.

### Setup Requirements

**Hardware:**

- Two computers on the same network (LAN or WiFi)
- Minimum 4GB RAM per machine (8GB recommended)
- Network connectivity between machines
- Administrative privileges for network monitoring

**Network Configuration:**

- Both PCs must be on same subnet
- Firewall rules allowing traffic on ports 3000, 8000, 8080
- No network isolation between the test machines

### Step-by-Step Cross-Device Setup

#### 1. Prepare PC A (Attack Source)

```bash
# Clone and setup the main platform
git clone https://github.com/AskitEndo/DDOSai_v2.git
cd DDOSai_v2

# Start the full platform
.\run_dev.bat  # Windows
./run_dev.sh   # Linux/macOS

# Verify platform is running
curl http://localhost:8000/health
# Should return: {"status": "healthy"}

# Access dashboard at: http://localhost:3000
```

#### 2. Prepare PC B (Target Machine)

```bash
# Clone repository on target machine
git clone https://github.com/AskitEndo/DDOSai_v2.git
cd DDOSai_v2/backend

# Start target server
.\start_target_server.bat  # Windows
./start_target_server.sh   # Linux/macOS

# Note the target IP address displayed
# Target server runs on: http://[PC_B_IP]:8080
```

#### 3. Configure Cross-Device Monitoring

**On PC A Dashboard:**

1. Open browser: `http://localhost:3000`
2. Click "Load from Backend" to load real data
3. Navigate to "Network Monitoring" panel
4. Click "Start" to begin network monitoring
5. Navigate to "Simulation" tab

**On PC B (Optional - Monitor Impact):**

1. Open browser: `http://[PC_A_IP]:3000`
2. Access PC A's dashboard to see attack source
3. Monitor target server logs in terminal
4. Watch system resources in Task Manager

#### 4. Execute Cross-Device Attack

**Configure Attack on PC A:**

1. In Simulation panel, enter PC B's IP address
2. Select attack type: HTTP Flood, SYN Flood, UDP Flood
3. Configure parameters:
   - Duration: 30-60 seconds
   - Packet Rate: 1000-5000 packets/second
   - Target Port: 8080 (for HTTP attacks)
4. Click "Start Simulation"

**Monitor Real Impact:**

- **PC A Dashboard**: Network Monitoring shows outgoing attack traffic
- **PC B Terminal**: Target server displays incoming request counts
- **Task Manager**: Both PCs show actual network activity spikes
- **Target Response**: PC B's target server becomes slow/unresponsive

### Verification of Real Attacks

**Network Activity Indicators:**

```bash
# On PC A - Check outgoing connections
netstat -an | findstr 8080

# On PC B - Check incoming connections
netstat -an | findstr :8080

# Monitor network interface statistics
# PC A should show increased bytes_sent
# PC B should show increased bytes_recv
```

**Dashboard Indicators:**

- Real-time attack entries in Network Monitoring panel
- Source IP matches PC A, Destination IP matches PC B
- Attack type corresponds to simulation configuration
- Confidence levels and packet counts update live
- WebSocket notifications appear across all connected dashboards

## Running the Platform

### Target Server Setup (New Feature)

The `start_target_server.bat` script creates a dedicated HTTP server that acts as a realistic target for DDoS attacks. This enables visible attack impact testing across different machines.

**Target Server Features:**

- **HTTP Server**: Responds to incoming requests on port 8080
- **Live Statistics**: Real-time request counting and performance metrics
- **Attack Visualization**: Shows incoming attack traffic with source IP tracking
- **Resource Monitoring**: Displays CPU and memory usage during attacks
- **Response Degradation**: Server becomes slow/unresponsive under attack

**Starting Target Server:**

```bash
# On target machine (PC B)
cd backend

# Windows
.\start_target_server.bat

# Linux/macOS
chmod +x start_target_server.sh
./start_target_server.sh

# Output shows:
# 🎯 Target Server Started
# 📡 Listening on: http://[YOUR_IP]:8080
# 📊 Stats endpoint: http://[YOUR_IP]:8080/stats
# 🔍 Use this IP in attack simulations
```

**Target Server Endpoints:**

- **Main Page**: `http://[TARGET_IP]:8080/` - Shows server status and statistics
- **Stats API**: `http://[TARGET_IP]:8080/stats` - JSON endpoint for programmatic access
- **Health Check**: `http://[TARGET_IP]:8080/health` - Simple health verification

**Monitoring Target Server:**

```bash
# Check real-time statistics
curl http://[TARGET_IP]:8080/stats

# Response example:
{
  "total_requests": 1247,
  "requests_per_second": 85.3,
  "avg_response_time": 0.023,
  "cpu_usage": 45.2,
  "memory_usage": 38.7,
  "active_connections": 12,
  "server_uptime": "00:05:23"
}
```

### Enhanced Platform Setup (Docker Recommended)

The enhanced setup includes all components plus new cross-device monitoring capabilities:

```bash
# Clone the repository
git clone https://github.com/AskitEndo/DDOSai_v2.git
cd DDOSai_v2

# Start development environment (includes all services)
.\run_dev.bat  # Windows
./run_dev.sh   # Linux/macOS

# Check all services are running
docker-compose ps

# Should show:
# - ddosai_backend (FastAPI + AI Engine)
# - ddosai_frontend (React Dashboard)
# - ddosai_db (PostgreSQL)
# - ddosai_redis (Redis Cache)
# - ddosai_prometheus (Metrics)
# - ddosai_grafana (Dashboards)
# - ddosai_influxdb (Time Series)
```

**Access Points:**

- **Main Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Interactive API**: http://localhost:8000/redoc
- **WebSocket Status**: ws://localhost:8000/ws
- **Grafana Monitoring**: http://localhost:3001 (admin/admin)
- **Prometheus Metrics**: http://localhost:9090
- **InfluxDB Interface**: http://localhost:8086

### Full Platform Setup

The full setup includes all components: backend, frontend, database, Redis, Prometheus, and Grafana.

```bash
# Clone the repository
git clone https://github.com/AskitEndo/DDOSai_v2.git
cd ddosai-platform

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# Access the platform
# Dashboard: http://localhost:3000
# API docs: http://localhost:8000/docs
# Grafana: http://localhost:3001 (admin/admin)
```

### Lightweight Setup

If you don't need monitoring and just want the core functionality:

```bash
# Start only essential services
docker-compose -f docker-compose.light.yml up -d

# Or manually start components:

# Terminal 1: Start the backend
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start the frontend
cd frontend
npm install
npm run dev
```

### Development Mode

For development with hot-reloading:

```bash
# Start in development mode
docker-compose -f docker-compose.dev.yml up -d

# Or for individual components:

# Backend with hot-reload
cd backend
uvicorn main:app --reload

# Frontend with hot-reload
cd frontend
npm run dev
```

### Demo Mode

To quickly see the platform in action:

```bash
# Windows
run_demo.bat

# Linux/macOS
chmod +x run_demo.sh
./run_demo.sh
```

## Understanding the Platform

### Core Components

1. **Backend (FastAPI)**

   - AI models for detection
   - API endpoints
   - Traffic processing

2. **Frontend (React)**

   - Dashboard
   - Visualization
   - User interface

3. **AI Models**

   - Autoencoder: Anomaly detection
   - GNN: Network graph analysis
   - RL: Threat scoring

4. **Monitoring**

   - Prometheus: Metrics collection
   - Grafana: Visualization

5. **Data Storage**
   - PostgreSQL: Persistent storage
   - Redis: Caching and real-time data

### How It Works

1. **Traffic Analysis**: The system captures or simulates network packets
2. **Feature Extraction**: Important features are extracted from packets
3. **AI Processing**: Multiple models analyze the traffic
4. **Consensus Decision**: Results are combined for final detection
5. **Visualization**: Results are displayed on the dashboard
6. **Alerting**: Alerts are generated for detected threats

### Data Flow

```
Traffic Source → Ingestion → Feature Extraction → AI Models → Detection Results
                                                              ↓
                                                      Database Storage
                                                              ↓
                                           Dashboard ← API ← Query Results
                                              ↓
                                         User Interface
```

## Training the AI Models

### Using Sample Data

```bash
# Access the backend container
docker-compose exec backend bash

# Train all models with sample data
python -m tools.train_models --model all --dataset data/samples/mixed_syn_flood.json

# Train specific models
python -m tools.train_models --model autoencoder --dataset data/samples/features_packet.csv
python -m tools.train_models --model gnn --dataset data/samples/features_flow.csv
python -m tools.train_models --model rl --dataset data/samples/mixed_syn_flood.json
```

### Adding Your Own Data

1. **Prepare your data**:

   - PCAP files: Network packet captures
   - CSV files: Pre-extracted features
   - JSON files: Structured packet data

2. **Convert PCAP files**:

   ```bash
   docker-compose exec backend python -m tools.pcap_converter \
     --input /path/to/your_traffic.pcap \
     --output /app/data/your_traffic.json
   ```

3. **Import data**:

   ```bash
   docker-compose exec backend python -m tools.load_data \
     --file /app/data/your_traffic.json
   ```

4. **Train with your data**:
   ```bash
   docker-compose exec backend python -m tools.train_models \
     --model all \
     --dataset /app/data/your_traffic.json
   ```

### Custom Model Integration

1. **Create model file**:

   ```bash
   docker-compose exec backend bash
   cd /app/ai
   touch custom_model.py
   ```

2. **Implement your model**:

   ```python
   import torch
   import numpy as np

   class CustomModel:
       def __init__(self, input_dim=31):
           self.input_dim = input_dim
           # Initialize your model

       def train(self, data):
           # Training logic
           pass

       def predict(self, features):
           # Return (is_malicious, confidence, explanation)
           return False, 0.5, {"reason": "Example"}

       def save(self, path):
           # Save model
           pass

       def load(self, path):
           # Load model
           pass
   ```

3. **Integrate with AI Engine**:
   - Edit `/app/ai/ai_engine.py`
   - Add your model to the orchestrator
   - Include in consensus mechanism

## Running Simulations

### Web Interface

1. **Access simulation page**:

   - Go to http://localhost:3000/simulation

2. **Configure simulation**:

   - Attack type: SYN Flood, UDP Flood, HTTP Flood
   - Target IP: e.g., 10.0.0.1
   - Target port: e.g., 80
   - Duration: e.g., 60 seconds
   - Packet rate: e.g., 1000 packets/second

3. **Start and monitor**:
   - Click "Start Simulation"
   - Watch dashboard for detection events

### Command Line

1. **Run predefined attack**:

   ```bash
   docker-compose exec backend python -m simulation.run_attack \
     --type syn_flood \
     --duration 60 \
     --rate 1000 \
     --target 10.0.0.1 \
     --port 80
   ```

2. **Generate custom attack**:

   ```bash
   docker-compose exec backend python -m tools.generate_attack \
     --type udp_flood \
     --output /app/data/custom_attack.json
   ```

3. **Replay attack traffic**:
   ```bash
   docker-compose exec backend python -m tools.replay_traffic \
     --input /app/data/custom_attack.json \
     --rate 500
   ```

### API

```bash
# Start simulation via API
curl -X POST http://localhost:8000/api/simulate/start \
  -H "Content-Type: application/json" \
  -d '{
    "attack_type": "SYN_FLOOD",
    "target_ip": "10.0.0.1",
    "target_port": 80,
    "duration": 60,
    "packet_rate": 1000
  }'

# Stop simulation
curl -X POST http://localhost:8000/api/simulate/stop \
  -H "Content-Type: application/json" \
  -d '{
    "simulation_id": "sim_1234"
  }'
```

## Dashboard Usage

### Main Dashboard Interface

The enhanced DDoS.AI dashboard provides comprehensive real-time monitoring and attack simulation capabilities.

**Dashboard URL**: http://localhost:3000

**Key Components:**

- **Control Buttons**: Load Data, Clear Data, Check Backend, Load from Backend
- **Metrics Cards**: Real-time statistics for packets, threats, CPU, memory
- **Network Monitoring Panel**: Live cross-device attack detection (NEW)
- **Attack Simulation Interface**: Real attack configuration and execution
- **Network Graph**: Interactive visualization of network topology
- **Detection History**: Chronological list of detected threats

### Network Monitoring Panel (New Feature)

The Network Monitoring Panel enables real-time cross-device attack detection and monitoring.

**Panel Features:**

```
📡 Network Monitoring Panel
├── Status Indicator (Active/Inactive with pulsing green light)
├── Control Buttons (Start/Stop/Refresh)
├── Live Network Statistics
│   ├── Bytes Received/Sent (formatted: KB, MB, GB)
│   ├── Packets Received/Sent (with comma separators)
│   └── Network Errors (dropped packets, errors)
├── Detected Attacks List
│   ├── Attack Type (SYN Flood, HTTP Flood, etc.)
│   ├── Source & Destination IPs
│   ├── Severity Level (Critical/High/Medium/Low)
│   ├── Confidence Score (0-100%)
│   ├── Protocol Information (TCP/UDP/HTTP)
│   └── Timestamp (real-time updates)
└── Auto-Refresh (3-second intervals when active)
```

**Using Network Monitoring:**

1. **Start Monitoring**:

   ```
   Dashboard → Network Monitoring Panel → Click "Start"
   Status changes to: "Active" with green pulsing indicator
   ```

2. **Monitor Cross-Device Attacks**:

   ```
   When PC A attacks PC B:
   - PC A: Shows outgoing attack traffic in monitoring panel
   - PC B: Shows incoming attack traffic with source IP = PC A
   - Real-time WebSocket updates across all connected dashboards
   ```

3. **Attack Information Display**:
   ```
   📊 SYN Flood Attack                     🕐 14:23:45
   📍 From: 192.168.1.100 → To: 192.168.1.101
   🔴 Severity: Critical | Confidence: 95.2%
   🌐 Protocol: TCP | Packet Size: 64 bytes
   🏷️ Flags: [SYN, PSH, URG]
   ```

### Attack Simulation Interface

Enhanced simulation interface for real cross-device attack testing.

**Simulation Configuration:**

```
🎯 Attack Simulation Panel
├── Target Configuration
│   ├── IP Address (auto-detected + manual entry)
│   ├── Port Selection (80, 443, 8080, custom)
│   └── IP Detection Button (refresh current IP)
├── Attack Parameters
│   ├── Attack Type (SYN Flood, UDP Flood, HTTP Flood, Slowloris)
│   ├── Duration (10-300 seconds)
│   ├── Packet Rate (100-10,000 packets/second)
│   └── Thread Count (1-50 threads)
├── Simulation Controls
│   ├── Start Simulation Button
│   ├── Stop Simulation Button
│   └── Real-time Status Display
└── Attack History
    ├── Previous Simulations
    ├── Success/Failure Status
    └── Performance Metrics
```

**Real Attack Workflow:**

1. **Configure Target**: Enter target PC's IP address
2. **Select Attack Type**: Choose from available attack vectors
3. **Set Parameters**: Configure intensity and duration
4. **Start Attack**: Click "Start Simulation" for real network traffic
5. **Monitor Impact**: Watch Network Monitoring Panel for real-time detection

### Data Loading Controls

Enhanced data management with unified backend integration.

**Control Buttons:**

```
📊 Data Controls Panel
├── Load Dummy Data
│   ├── Generates 50 sample detections
│   ├── Creates system metrics
│   └── Builds network graph
├── Load from Backend
│   ├── Fetches real API data
│   ├── Includes network monitoring data
│   ├── Merges all data sources
│   └── Shows "Sample Data" indicator when using dummy data
├── Clear Data
│   ├── Removes all loaded data
│   ├── Resets dashboard to empty state
│   └── Maintains connection status
└── Check Backend
    ├── Tests API connectivity
    ├── Shows connection status (Connected/Disconnected)
    └── Updates backend health indicator
```

**Data Loading Strategy:**

- **Empty Start**: Dashboard starts empty, user controls data loading
- **Backend Priority**: "Load from Backend" provides real data when available
- **Offline Fallback**: Dummy data available when backend disconnected
- **Unified Types**: All data sources use consistent TypeScript interfaces

### Real-Time Visualization

**WebSocket Integration:**

- **Live Updates**: Real-time data streaming via WebSocket connections
- **Cross-Device Sync**: Attack notifications broadcast to all connected clients
- **Status Indicators**: Connection status, monitoring status, backend health
- **Auto-Refresh**: Configurable refresh intervals for different components

**Interactive Elements:**

- **Network Graph**: Click nodes to see detection details
- **Attack List**: Click attacks to view full analysis
- **Metrics Cards**: Hover for detailed information
- **Control Panels**: Immediate feedback on user actions

### Navigation & User Experience

**Enhanced Navigation:**

- **Clean Startup**: Empty dashboard until user loads data
- **Persistent State**: Simulation context maintained across page navigation
- **Global Context**: Simulation history and IP detection shared across components
- **Responsive Design**: Works on desktop, tablet, and mobile devices

**User Feedback:**

- **Loading States**: Clear indicators during data operations
- **Error Handling**: Informative error messages with suggested actions
- **Success Notifications**: Confirmation of successful operations
- **Status Indicators**: Real-time status for all system components

## Monitoring

### Dashboard

The main dashboard provides real-time monitoring:

- **URL**: http://localhost:3000
- **Features**:
  - Network graph visualization
  - Threat score panel
  - Traffic statistics
  - Alert notifications

### Grafana

Detailed metrics and dashboards:

- **URL**: http://localhost:3001
- **Default credentials**: admin/admin
- **Dashboards**:
  - DDoS.AI Platform Dashboard
  - System Resources
  - Model Performance
  - Traffic Analysis

### Prometheus

Raw metrics and queries:

- **URL**: http://localhost:9090
- **Key metrics**:
  - `ddosai_packets_total`: Total packets processed
  - `ddosai_malicious_packets_total`: Malicious packets detected
  - `ddosai_packet_processing_time_seconds`: Processing latency
  - `ddosai_model_inference_time_seconds`: Model inference time

### Continuous Monitoring

1. **Set up packet capture**:

   ```bash
   docker-compose exec backend python -m tools.continuous_capture \
     --interface eth0 \
     --output /app/data/captured_traffic
   ```

2. **Configure alerts in Grafana**:

   - Go to http://localhost:3001
   - Navigate to Alerting → Alert Rules
   - Create alerts for high threat levels

3. **Set up notification channels**:

   - Email, Slack, or webhook notifications
   - Get alerted when attacks are detected

4. **Schedule regular model retraining**:
   ```bash
   # Add to crontab
   0 0 * * 0 docker-compose exec -T backend python -m tools.train_models \
     --model all \
     --dataset /app/data/collected_traffic.json
   ```

## Recovery and Backup

### Backup Procedures

1. **Manual backup**:

   ```bash
   ./scripts/backup_restore.sh backup
   ```

2. **Scheduled backups**:

   ```bash
   # Add to crontab
   0 2 * * * /path/to/ddosai-platform/scripts/backup_restore.sh backup
   ```

3. **What gets backed up**:
   - PostgreSQL database
   - Redis data
   - Trained models
   - Configuration files

### Restore Procedures

1. **List available backups**:

   ```bash
   ./scripts/backup_restore.sh list
   ```

2. **Restore from backup**:

   ```bash
   ./scripts/backup_restore.sh restore backup/ddosai_backup_20230717_101530.tar.gz
   ```

3. **Verify restoration**:
   ```bash
   docker-compose ps
   curl http://localhost:8000/health
   ```

### Disaster Recovery

1. **Complete system recovery**:

   ```bash
   # Stop all services
   docker-compose down

   # Restore from latest backup
   ./scripts/backup_restore.sh restore $(ls -t backup/ddosai_backup_*.tar.gz | head -1)

   # Start services
   docker-compose up -d
   ```

2. **Individual component recovery**:

   ```bash
   # Restore only database
   ./scripts/backup_restore.sh restore-db backup/ddosai_backup_20230717_101530.tar.gz

   # Restore only models
   ./scripts/backup_restore.sh restore-models backup/ddosai_backup_20230717_101530.tar.gz
   ```

## Troubleshooting

### Common Issues

1. **Models not detecting attacks**:

   - Check if models are properly trained
   - Verify feature extraction is working
   - Adjust detection thresholds in `/app/core/config.py`

2. **High resource usage**:

   - Adjust batch size in configuration
   - Reduce packet processing rate
   - Scale horizontally with multiple instances

3. **Missing metrics in Grafana**:
   - Check Prometheus connection
   - Verify metrics are being collected
   - Check Grafana data source configuration

### Cross-Device Issues

#### Target Server Connection Problems

**Issue**: Cannot connect to target server on PC B

```bash
# Error: Connection refused to http://[TARGET_IP]:8080
```

**Solutions**:

1. **Check Target Server Status**:

   ```bash
   # On PC B, verify server is running
   curl http://localhost:8080/health
   # Should return: {"status": "healthy"}
   ```

2. **Verify Network Connectivity**:

   ```bash
   # From PC A, test connection to PC B
   ping [PC_B_IP]
   telnet [PC_B_IP] 8080
   ```

3. **Firewall Configuration**:

   ```bash
   # Windows - Allow port 8080
   netsh advfirewall firewall add rule name="DDoS Target Server" dir=in action=allow protocol=TCP localport=8080

   # Linux - Allow port 8080
   sudo ufw allow 8080
   ```

4. **Network Discovery**:
   ```bash
   # Find other machines on network
   nmap -sn 192.168.1.0/24  # Adjust subnet as needed
   arp -a  # Show local ARP table
   ```

#### Network Monitoring Not Detecting Attacks

**Issue**: Attacks are running but not showing in Network Monitoring Panel

**Solutions**:

1. **Start Network Monitoring**:

   ```
   Dashboard → Network Monitoring Panel → Click "Start"
   Verify "Active" status with green pulsing indicator
   ```

2. **Check WebSocket Connection**:

   ```javascript
   // Browser console check
   // Should show: WebSocket connection established
   ```

3. **Verify Attack Configuration**:

   ```
   - Target IP matches PC B's actual IP
   - Attack type is supported (SYN/UDP/HTTP Flood)
   - Duration and packet rate are realistic
   - Target server is running and responding
   ```

4. **Check Backend Logs**:
   ```bash
   docker-compose logs -f backend | grep -i "network\|attack\|monitoring"
   ```

#### Real-Time Updates Not Working

**Issue**: Dashboard not showing live attack updates

**Solutions**:

1. **WebSocket Status Check**:

   ```bash
   # Check WebSocket endpoint
   curl -I http://localhost:8000/ws
   # Should return: 101 Switching Protocols
   ```

2. **Browser Console Debug**:

   ```javascript
   // Check for WebSocket errors
   // Look for connection drops or message failures
   ```

3. **Refresh Data Connections**:
   ```
   - Click "Check Backend" to verify API connectivity
   - Click "Load from Backend" to refresh all data
   - Restart network monitoring if needed
   ```

#### IP Address Detection Problems

**Issue**: Auto IP detection showing wrong address

**Solutions**:

1. **Manual IP Entry**:

   ```
   - Use ipconfig (Windows) or ifconfig (Linux) to find correct IP
   - Enter IP manually in simulation interface
   - Click IP detection refresh button
   ```

2. **Network Interface Selection**:

   ```bash
   # Windows - Show all network interfaces
   ipconfig /all

   # Linux - Show all network interfaces
   ip addr show
   ```

3. **VPN/Virtual Network Issues**:
   ```
   - Disable VPN during testing
   - Use physical network interface IP
   - Avoid virtual machine bridged networks
   ```

#### Performance and Resource Issues

**Issue**: High CPU/Memory usage during cross-device testing

**Solutions**:

1. **Reduce Attack Intensity**:

   ```
   - Lower packet rate (500-1000 instead of 5000+)
   - Shorter duration (30-60 seconds)
   - Fewer threads (5-10 instead of 50)
   ```

2. **Monitor System Resources**:

   ```bash
   # Check system performance
   docker stats  # Docker container resources
   htop         # Overall system performance
   ```

3. **Network Bandwidth Limits**:
   ```
   - Test on gigabit network for best results
   - Monitor network utilization during attacks
   - Consider QoS limits on network equipment
   ```

#### Docker and Service Issues

**Issue**: Services not starting properly

**Solutions**:

1. **Check Docker Status**:

   ```bash
   docker info
   docker-compose ps
   docker-compose logs --tail=50
   ```

2. **Restart Services**:

   ```bash
   docker-compose down
   docker-compose up -d --build
   ```

3. **Port Conflicts**:

   ```bash
   # Check if ports are already in use
   netstat -an | findstr ":3000 :8000 :8080"
   ```

4. **Resource Allocation**:
   ```bash
   # Increase Docker memory limits if needed
   # Docker Desktop → Settings → Resources → Memory: 8GB+
   ```

### Network Monitoring Problems

### Logs

1. **View all logs**:

   ```bash
   docker-compose logs
   ```

2. **View specific service logs**:

   ```bash
   docker-compose logs backend
   docker-compose logs frontend
   docker-compose logs prometheus
   ```

3. **Follow logs in real-time**:
   ```bash
   docker-compose logs -f backend
   ```

### Performance Tuning

1. **Adjust worker count**:

   - Edit `docker-compose.yml`
   - Modify `MAX_WORKERS` environment variable

2. **Optimize batch processing**:

   - Edit `/app/core/config.py`
   - Adjust `BATCH_SIZE` parameter

3. **Scale services**:
   ```bash
   docker-compose up -d --scale backend=3
   ```

## End-to-End Examples

### Complete Workflow

Here's a complete workflow to set up, train, and test the platform:

```bash
# 1. Start the platform
docker-compose up -d

# 2. Generate training data
docker-compose exec backend python -m tools.generate_data \
  --type mixed \
  --duration 10 \
  --attack-type syn_flood \
  --output /app/data/training_data.json

# 3. Train the models
docker-compose exec backend python -m tools.train_models \
  --model all \
  --dataset /app/data/training_data.json

# 4. Generate test traffic
docker-compose exec backend python -m tools.generate_data \
  --type mixed \
  --duration 5 \
  --attack-type http_flood \
  --output /app/data/test_data.json

# 5. Run the test traffic through the system
docker-compose exec backend python -m tools.replay_traffic \
  --input /app/data/test_data.json \
  --rate 100

# 6. Monitor the results
# Open http://localhost:3000 to see the dashboard
# Check detection results and performance metrics
# View detailed metrics in Grafana at http://localhost:3001

# 7. Export detection results
docker-compose exec backend python -m tools.export_results \
  --output /app/data/detection_results.json
```

### Production Deployment

For production environments:

```bash
# 1. Create production environment file
cp .env.example .env.production
# Edit .env.production with your settings

# 2. Deploy with production configuration
docker-compose -f docker-compose.prod.yml up -d

# 3. Apply security hardening
sudo ./scripts/security_hardening.sh

# 4. Set up regular backups
./scripts/backup_restore.sh backup

# 5. Configure monitoring alerts
# Access Grafana at https://your-domain.com/grafana
# Set up alert rules and notification channels

# 6. Set up SSL certificates
# Either use the included script or configure with Let's Encrypt:
./scripts/setup_ssl.sh your-domain.com

# 7. Monitor system health
curl https://your-domain.com/api/health?detailed=true
```

## How the Platform Works

### Core Functionality

The DDoS.AI platform detects and analyzes DDoS attacks using multiple AI models:

1. **Autoencoder**: Detects anomalies in packet features
2. **Graph Neural Network**: Analyzes network topology
3. **Reinforcement Learning**: Scores threat levels

These models work together to provide high accuracy and explainability.

### Future DDoS Prevention Implementation

The platform roadmap includes advanced DDoS prevention capabilities through the following components:

#### 1. Mitigation Orchestrator

This is the central component that coordinates the entire mitigation process:

**Attack Detection Integration**:

- Receives real-time notifications from the existing AI detection engine
- When an attack is detected with high confidence (>90%), it automatically triggers mitigation
- Maintains a state machine for each active mitigation

**Implementation Approach**:

- We'll create a Python class `MitigationOrchestrator` that runs as a service
- It will subscribe to a message queue where the detection engine publishes attack events
- For each attack, it creates a mitigation session with unique ID and tracking

```python
# Example implementation snippet
class MitigationOrchestrator:
    def __init__(self, config):
        self.config = config
        self.active_mitigations = {}
        self.strategy_selector = StrategySelector(config)
        self.network_adapter = NetworkAdapterLayer(config)

    def handle_attack_detection(self, attack_data):
        # Check confidence threshold
        if attack_data['confidence'] < self.config.threshold:
            return

        # Create mitigation session
        mitigation_id = f"mit_{uuid.uuid4()}"

        # Select appropriate strategy
        strategy = self.strategy_selector.select_strategy(
            attack_data['type'],
            attack_data['characteristics']
        )

        # Apply mitigation
        self.network_adapter.apply_mitigation(strategy)

        # Start monitoring effectiveness
        self.start_effectiveness_monitoring(mitigation_id)
```

#### 2. Strategy Selector

This component determines the best mitigation approach based on attack characteristics:

**Smart Strategy Selection**:

- Analyzes attack patterns (SYN flood, UDP flood, HTTP flood, etc.)
- Selects appropriate countermeasures from a library of strategies
- Handles multiple simultaneous attack types
- Adapts strategies based on effectiveness feedback

**Implementation Approach**:

- Create a rule-based system with configurable strategies
- Use machine learning to improve strategy selection over time
- Store strategies in a database with version control

```json
# Example strategy configuration
{
  "id": "syn_flood_mitigation",
  "name": "SYN Flood Protection",
  "attack_type": "SYN_FLOOD",
  "actions": [
    {
      "type": "rate_limit",
      "parameters": {
        "max_syn_per_second": 100,
        "syn_cookie": true
      },
      "target_devices": ["edge_firewall", "load_balancer"],
      "priority": 1
    },
    {
      "type": "block_sources",
      "parameters": {
        "duration": 300,
        "whitelist": ["trusted_ips"]
      },
      "target_devices": ["edge_firewall"],
      "priority": 2
    }
  ]
}
```

#### 3. Network Adapter Layer

This component translates mitigation strategies into actual network device configurations:

**Multi-Device Integration**:

- Connects to firewalls, load balancers, routers, and CDNs
- Translates generic mitigation actions into device-specific commands
- Verifies that configurations are applied successfully
- Implements fallback mechanisms for device failures

**Implementation Approach**:

- Create adapter classes for different device types (Cisco, F5, CloudFlare, etc.)
- Use standard protocols (NETCONF, REST APIs) for communication
- Implement fallback mechanisms for device failures

```python
# Example adapter implementation
class FirewallAdapter(BaseAdapter):
    def __init__(self, config):
        super().__init__(config)

    def connect(self, device_info):
        # Connect to firewall using appropriate protocol
        if device_info['protocol'] == 'ssh':
            # Connect via SSH
            pass
        elif device_info['protocol'] == 'api':
            # Connect via REST API
            pass

    def apply_rate_limit(self, parameters):
        # Translate generic rate limit to device-specific commands
        commands = self._generate_rate_limit_commands(parameters)
        return self._execute_commands(commands)

    def block_sources(self, sources, parameters):
        # Generate ACLs or firewall rules
        rules = self._generate_block_rules(sources, parameters)
        return self._execute_commands(rules)
```

#### 4. Simulation Engine

This component allows testing mitigation strategies before deployment:

**Safe Testing Environment**:

- Generates realistic attack traffic in a controlled environment
- Applies mitigation strategies without affecting production
- Measures effectiveness metrics
- Compares different strategies side-by-side

**Implementation Approach**:

- Create a sandboxed network environment using containers
- Implement traffic generators for different attack types
- Develop metrics collection for detailed analysis

```python
# Example simulation API
@app.post("/api/simulation/start")
async def start_simulation(simulation_config: SimulationConfig):
    # Create simulation environment
    sim_id = simulation_service.create_simulation(
        attack_type=simulation_config.attack_type,
        mitigation_strategy=simulation_config.strategy,
        duration=simulation_config.duration
    )

    # Start simulation in background task
    background_tasks.add_task(
        simulation_service.run_simulation,
        sim_id
    )

    return {"simulation_id": sim_id, "status": "started"}
```

#### 5. Effectiveness Analyzer

This component evaluates how well mitigation strategies are working:

**Real-time Analysis**:

- Monitors traffic patterns during mitigation
- Calculates effectiveness metrics (traffic reduction, false positives)
- Suggests strategy adjustments when needed
- Generates comprehensive after-action reports

**Implementation Approach**:

- Create a metrics collection system using Prometheus
- Implement analysis algorithms to evaluate effectiveness
- Develop a recommendation engine for strategy improvements

```json
# Example effectiveness metrics
{
  "mitigation_id": "mit_12345",
  "attack_type": "SYN_FLOOD",
  "start_time": "2025-07-17T10:30:00Z",
  "current_metrics": {
    "traffic_reduction": 94.5,
    "false_positives": 0.2,
    "resource_utilization": 45.3,
    "latency_impact": 5.2
  },
  "recommendations": [
    "Increase SYN cookie threshold to reduce false positives",
    "Add additional rate limiting at application layer"
  ]
}
```

#### Integration with Network Infrastructure

The system will integrate with various network devices to implement mitigation:

##### 1. Firewalls

**Implementation**:

- Create firewall rules to block malicious traffic
- Configure rate limiting for specific protocols
- Set up connection tracking and state validation

**Supported Devices**:

- Cisco ASA/Firepower
- Palo Alto Networks
- Fortinet FortiGate
- iptables/nftables

##### 2. Load Balancers

**Implementation**:

- Configure SYN cookies and TCP validation
- Implement request rate limiting
- Set up traffic distribution rules

**Supported Devices**:

- F5 BIG-IP
- NGINX Plus
- HAProxy
- AWS ELB/ALB

##### 3. Routers/Switches

**Implementation**:

- Configure ACLs to filter traffic
- Set up QoS policies for traffic prioritization
- Implement BGP flowspec for upstream filtering

**Supported Devices**:

- Cisco IOS/IOS-XE
- Juniper Junos
- Arista EOS

##### 4. CDN/Cloud Protection

**Implementation**:

- Activate DDoS protection services
- Configure WAF rules
- Set up traffic scrubbing

**Supported Services**:

- Cloudflare
- Akamai
- AWS Shield
- Google Cloud Armor

#### Mitigation Strategies

The system will support various mitigation strategies for different attack types:

##### 1. SYN Flood Mitigation

- SYN cookies
- Connection rate limiting
- TCP validation
- Source IP reputation filtering

##### 2. UDP Flood Mitigation

- Rate limiting by packet size
- DNS/NTP amplification protection
- Source validation
- Traffic classification and prioritization

##### 3. HTTP/Layer 7 Flood Mitigation

- Request rate limiting
- CAPTCHA challenges
- Bot detection
- Application-specific rules

##### 4. Volumetric Attack Mitigation

- Traffic scrubbing
- Anycast distribution
- BGP flowspec announcements
- Upstream provider coordination

#### User Interface and Control

The system will provide both automatic and manual control options:

##### 1. Dashboard Integration

- Real-time mitigation status display
- Traffic visualization before/during/after mitigation
- Effectiveness metrics with historical comparison
- Configuration interface for mitigation rules

##### 2. API Endpoints

- Mitigation control endpoints (start/stop/update)
- Configuration management
- Reporting and metrics access
- Simulation control

##### 3. Manual Override

- Emergency stop button for any active mitigation
- Manual strategy selection option
- Granular control of mitigation parameters
- Audit logging of all manual actions

#### Implementation Timeline

Based on our implementation plan, we'll develop this feature in the following phases:

1. **Phase 1**: Core components setup and Mitigation Orchestrator
2. **Phase 2**: Strategy Selector and Network Adapter Layer
3. **Phase 3**: Simulation Engine and Effectiveness Analyzer
4. **Phase 4**: Integration with existing platform
5. **Phase 5**: Testing, documentation, and deployment

### Why It Uses Certain Components

1. **PyTorch (large download)**:

   - Required for the deep learning models
   - Includes CUDA support for GPU acceleration
   - Powers all three AI models

2. **Prometheus**:

   - Collects and stores metrics
   - Tracks system performance
   - Enables historical analysis

3. **Grafana**:
   - Visualizes metrics from Prometheus
   - Creates dashboards for monitoring
   - Provides alerting capabilities

### Data Flow in Detail

1. **Traffic Source** → Packets enter the system
2. **Feature Extraction** → Extract 31 features per packet
3. **Autoencoder** → Detect anomalies based on reconstruction error
4. **GNN** → Analyze network graph for suspicious patterns
5. **RL** → Assign threat scores based on learned patterns
6. **Consensus** → Combine model outputs for final decision
7. **Storage** → Save results to database
8. **Dashboard** → Display results and alerts

This comprehensive guide should help you understand, set up, and use the DDoS.AI platform effectively.
