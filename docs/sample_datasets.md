# DDoS.AI Sample Datasets

This document provides information about the sample datasets included with the DDoS.AI platform for training, testing, and demonstration purposes.

## Overview

The DDoS.AI platform includes several sample datasets that can be used to:

1. Train the AI models
2. Test detection capabilities
3. Demonstrate platform features
4. Benchmark performance

These datasets are located in the `backend/data/samples` directory and are automatically loaded when the platform is started in demo mode.

## Dataset Types

### 1. Normal Traffic Datasets

These datasets contain normal network traffic patterns without any attacks.

#### 1.1 Web Browsing Traffic

- **Filename**: `normal_web_browsing.json`
- **Size**: 10,000 packets
- **Description**: Simulated web browsing traffic with HTTP and HTTPS requests
- **Features**:
  - Multiple source IPs
  - Destination ports 80 and 443
  - Varied packet sizes
  - Normal TCP flags distribution

#### 1.2 Mixed Normal Traffic

- **Filename**: `normal_mixed.json`
- **Size**: 50,000 packets
- **Description**: Mixed normal traffic including web, DNS, SSH, and email
- **Features**:
  - Multiple protocols (TCP, UDP, ICMP)
  - Various application patterns
  - Realistic timing distribution
  - Diverse network topology

### 2. Attack Datasets

These datasets contain various types of DDoS attacks for training and testing.

#### 2.1 SYN Flood Attack

- **Filename**: `attack_syn_flood.json`
- **Size**: 20,000 packets
- **Description**: SYN flood attack targeting a web server
- **Features**:
  - High rate of SYN packets
  - Multiple source IPs (spoofed)
  - Single destination IP and port
  - Small packet size

#### 2.2 UDP Flood Attack

- **Filename**: `attack_udp_flood.json`
- **Size**: 30,000 packets
- **Description**: UDP flood attack targeting DNS server
- **Features**:
  - High volume UDP traffic
  - Random source ports
  - Destination port 53
  - Large packet sizes

#### 2.3 HTTP Flood Attack

- **Filename**: `attack_http_flood.json`
- **Size**: 15,000 packets
- **Description**: HTTP flood attack targeting web application
- **Features**:
  - Legitimate-looking HTTP GET requests
  - Multiple source IPs
  - Destination port 80
  - High request rate

#### 2.4 DNS Amplification Attack

- **Filename**: `attack_dns_amplification.json`
- **Size**: 10,000 packets
- **Description**: DNS amplification attack
- **Features**:
  - Spoofed source IP (victim)
  - Large DNS response packets
  - High amplification factor
  - UDP protocol

### 3. Mixed Datasets

These datasets contain a mix of normal traffic and attacks for realistic testing.

#### 3.1 Mixed Traffic with SYN Flood

- **Filename**: `mixed_syn_flood.json`
- **Size**: 100,000 packets
- **Description**: Normal traffic with a SYN flood attack
- **Features**:
  - 80% normal traffic, 20% attack traffic
  - Attack starts after 5 minutes
  - Attack duration of 3 minutes
  - Gradual attack ramp-up

#### 3.2 Mixed Traffic with Multiple Attacks

- **Filename**: `mixed_multiple_attacks.json`
- **Size**: 200,000 packets
- **Description**: Normal traffic with multiple attack types
- **Features**:
  - 70% normal traffic, 30% attack traffic
  - SYN flood, UDP flood, and HTTP flood attacks
  - Different attack timings and durations
  - Varied attack intensities

### 4. Feature Datasets

These datasets contain pre-extracted features for direct model training.

#### 4.1 Packet Features

- **Filename**: `features_packet.csv`
- **Size**: 50,000 samples
- **Description**: Packet-level features for training the autoencoder
- **Features**:
  - 31 features per packet
  - Normalized values
  - Labeled (normal/attack)
  - Balanced classes

#### 4.2 Flow Features

- **Filename**: `features_flow.csv`
- **Size**: 10,000 samples
- **Description**: Flow-level features for training the GNN
- **Features**:
  - 25 features per flow
  - Source and destination information
  - Flow statistics
  - Labeled (normal/attack)

## Using the Datasets

### Loading Sample Data

To load the sample datasets, use the `--load-sample-data` flag when starting the backend:

```bash
python -m backend.main --load-sample-data
```

Or set the environment variable:

```bash
export LOAD_SAMPLE_DATA=true
python -m backend.main
```

### Training Models with Sample Data

To train the AI models using the sample datasets:

```bash
python -m backend.tools.train_models --dataset=features_packet.csv --model=autoencoder
python -m backend.tools.train_models --dataset=features_flow.csv --model=gnn
```

### Generating Custom Datasets

You can generate custom datasets using the data generation tool:

```bash
python -m backend.tools.generate_data --type=mixed --duration=10 --attack-type=syn_flood --output=custom_dataset.json
```

## Dataset Format

### JSON Format

The JSON datasets have the following structure:

```json
[
  {
    "timestamp": "2023-07-17T10:15:30.123Z",
    "src_ip": "192.168.1.100",
    "dst_ip": "10.0.0.1",
    "src_port": 12345,
    "dst_port": 80,
    "protocol": "TCP",
    "flags": ["SYN"],
    "packet_size": 64,
    "ttl": 64,
    "payload_entropy": 0.5,
    "packet_id": "pkt_a1b2c3d4"
  }
  // More packets...
]
```

### CSV Format

The CSV feature datasets have the following structure:

```
packet_size,ttl,payload_entropy,is_tcp,is_udp,is_icmp,has_syn,has_ack,has_psh,has_rst,has_fin,src_port,dst_port,src_ip_class_a,src_ip_class_b,src_ip_class_c,dst_ip_class_a,dst_ip_class_b,dst_ip_class_c,is_src_private,is_dst_private,is_common_port,is_ephemeral_port,packet_rate,byte_rate,src_entropy,dst_entropy,port_entropy,protocol_entropy,flow_duration,packets_per_flow,label
64,64,0.1,1,0,0,1,0,0,0,0,12345,80,0,0,1,0,0,1,1,1,1,1,100,6400,0.2,0.9,0.2,0,0.5,100,1
```

## Dataset Statistics

| Dataset                       | Packets | Flows  | Normal % | Attack % | Duration (s) | Avg Packet Size | Protocols      |
| ----------------------------- | ------- | ------ | -------- | -------- | ------------ | --------------- | -------------- |
| normal_web_browsing.json      | 10,000  | 500    | 100%     | 0%       | 300          | 512             | TCP            |
| normal_mixed.json             | 50,000  | 2,500  | 100%     | 0%       | 600          | 384             | TCP, UDP, ICMP |
| attack_syn_flood.json         | 20,000  | 10,000 | 0%       | 100%     | 60           | 64              | TCP            |
| attack_udp_flood.json         | 30,000  | 15,000 | 0%       | 100%     | 60           | 512             | UDP            |
| attack_http_flood.json        | 15,000  | 1,000  | 0%       | 100%     | 60           | 256             | TCP            |
| attack_dns_amplification.json | 10,000  | 5,000  | 0%       | 100%     | 60           | 1024            | UDP            |
| mixed_syn_flood.json          | 100,000 | 20,000 | 80%      | 20%      | 600          | 256             | TCP, UDP       |
| mixed_multiple_attacks.json   | 200,000 | 40,000 | 70%      | 30%      | 1200         | 384             | TCP, UDP       |

## Download Links

The sample datasets can be downloaded individually from the following links:

- [Normal Traffic Datasets (ZIP, 15MB)](https://example.com/ddosai/samples/normal.zip)
- [Attack Datasets (ZIP, 25MB)](https://example.com/ddosai/samples/attacks.zip)
- [Mixed Datasets (ZIP, 50MB)](https://example.com/ddosai/samples/mixed.zip)
- [Feature Datasets (ZIP, 10MB)](https://example.com/ddosai/samples/features.zip)
- [Complete Dataset Collection (ZIP, 100MB)](https://example.com/ddosai/samples/all.zip)

## References

These datasets were created based on patterns observed in the following public datasets:

1. CAIDA DDoS Attack 2007 Dataset
2. CIC DoS Dataset 2017
3. UNSW-NB15 Dataset
4. DARPA 1999 Dataset

The data has been anonymized and simplified for educational and demonstration purposes.
