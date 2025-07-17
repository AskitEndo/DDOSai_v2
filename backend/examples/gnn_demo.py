"""
Graph Neural Network analyzer demonstration script
"""
import sys
import os
import numpy as np
from datetime import datetime, timedelta
from typing import List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.gnn_analyzer import GNNAnalyzer
from models.data_models import TrafficPacket, ProtocolType


def generate_normal_network_traffic(n_samples: int = 100) -> List[TrafficPacket]:
    """Generate normal network traffic patterns"""
    packets = []
    base_time = datetime.now()
    
    # Simulate normal web browsing from multiple clients to web servers
    client_ips = [f"192.168.1.{100 + i}" for i in range(10)]
    server_ips = ["93.184.216.34", "1.1.1.1", "8.8.8.8", "172.217.164.110"]
    
    for i in range(n_samples):
        client_ip = np.random.choice(client_ips)
        server_ip = np.random.choice(server_ips)
        
        # Normal HTTP/HTTPS traffic
        if i % 3 == 0:  # HTTP
            packet = TrafficPacket(
                timestamp=base_time + timedelta(seconds=i * 0.5),
                src_ip=client_ip,
                dst_ip=server_ip,
                src_port=12000 + (i % 1000),
                dst_port=80,
                protocol=ProtocolType.HTTP,
                packet_size=int(np.random.normal(800, 200)),
                ttl=64,
                flags=["SYN", "ACK"] if i % 4 != 0 else ["SYN"],
                payload_entropy=max(0.0, min(1.0, np.random.normal(0.6, 0.1))),
                packet_id=f"normal_{i:05d}"
            )
        elif i % 3 == 1:  # HTTPS
            packet = TrafficPacket(
                timestamp=base_time + timedelta(seconds=i * 0.5),
                src_ip=client_ip,
                dst_ip=server_ip,
                src_port=12000 + (i % 1000),
                dst_port=443,
                protocol=ProtocolType.HTTPS,
                packet_size=int(np.random.normal(600, 150)),
                ttl=64,
                flags=["SYN", "ACK"],
                payload_entropy=max(0.0, min(1.0, np.random.normal(0.7, 0.1))),
                packet_id=f"normal_{i:05d}"
            )
        else:  # DNS
            packet = TrafficPacket(
                timestamp=base_time + timedelta(seconds=i * 0.5),
                src_ip=client_ip,
                dst_ip=server_ip,
                src_port=12000 + (i % 1000),
                dst_port=53,
                protocol=ProtocolType.UDP,
                packet_size=int(np.random.normal(100, 30)),
                ttl=64,
                flags=[],
                payload_entropy=max(0.0, min(1.0, np.random.normal(0.4, 0.1))),
                packet_id=f"normal_{i:05d}"
            )
        
        # Ensure valid ranges
        packet.packet_size = max(64, min(1500, packet.packet_size))
        packets.append(packet)
    
    return packets


def generate_ddos_attack_traffic(attack_type: str = "syn_flood", n_samples: int = 150) -> List[TrafficPacket]:
    """Generate DDoS attack traffic patterns"""
    packets = []
    base_time = datetime.now()
    
    if attack_type == "syn_flood":
        # SYN flood: Many sources targeting single server
        attacker_ips = [f"203.0.113.{i}" for i in range(50)]
        target_ip = "192.168.1.1"
        
        for i in range(n_samples):
            attacker_ip = np.random.choice(attacker_ips)
            
            packet = TrafficPacket(
                timestamp=base_time + timedelta(milliseconds=i * 20),
                src_ip=attacker_ip,
                dst_ip=target_ip,
                src_port=12000 + i,
                dst_port=80,
                protocol=ProtocolType.TCP,
                packet_size=64,  # Small SYN packets
                ttl=32,  # Lower TTL
                flags=["SYN"],  # Only SYN flags
                payload_entropy=0.1,  # Low entropy
                packet_id=f"syn_flood_{i:05d}"
            )
            packets.append(packet)
    
    elif attack_type == "udp_flood":
        # UDP flood: Distributed sources flooding target
        attacker_ips = [f"198.51.100.{i}" for i in range(30)]
        target_ip = "192.168.1.1"
        
        for i in range(n_samples):
            attacker_ip = np.random.choice(attacker_ips)
            
            packet = TrafficPacket(
                timestamp=base_time + timedelta(milliseconds=i * 10),
                src_ip=attacker_ip,
                dst_ip=target_ip,
                src_port=54321,
                dst_port=53 + (i % 10),  # Multiple target ports
                protocol=ProtocolType.UDP,
                packet_size=1024,  # Large packets
                ttl=128,
                flags=[],
                payload_entropy=0.9,  # High entropy (random data)
                packet_id=f"udp_flood_{i:05d}"
            )
            packets.append(packet)
    
    elif attack_type == "distributed_scan":
        # Distributed port scanning
        scanner_ips = [f"172.16.{i//10}.{i%10}" for i in range(20)]
        target_ips = [f"192.168.1.{i}" for i in range(1, 11)]
        target_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 8080]
        
        for i in range(n_samples):
            scanner_ip = np.random.choice(scanner_ips)
            target_ip = np.random.choice(target_ips)
            target_port = np.random.choice(target_ports)
            
            packet = TrafficPacket(
                timestamp=base_time + timedelta(milliseconds=i * 50),
                src_ip=scanner_ip,
                dst_ip=target_ip,
                src_port=12345 + (i % 100),
                dst_port=target_port,
                protocol=ProtocolType.TCP,
                packet_size=64,
                ttl=64,
                flags=["SYN"],
                payload_entropy=0.2,
                packet_id=f"scan_{i:05d}"
            )
            packets.append(packet)
    
    return packets


def create_training_data():
    """Create training data with normal and attack patterns"""
    print("Creating training data...")
    
    training_data = []
    
    # Generate normal traffic samples
    for i in range(10):
        normal_traffic = generate_normal_network_traffic(80)
        training_data.append((normal_traffic, False))  # False = benign
    
    # Generate attack traffic samples
    attack_types = ["syn_flood", "udp_flood", "distributed_scan"]
    
    for attack_type in attack_types:
        for i in range(5):
            attack_traffic = generate_ddos_attack_traffic(attack_type, 100)
            training_data.append((attack_traffic, True))  # True = malicious
    
    print(f"Created {len(training_data)} training samples:")
    print(f"  - Normal samples: {sum(1 for _, label in training_data if not label)}")
    print(f"  - Attack samples: {sum(1 for _, label in training_data if label)}")
    
    return training_data


def train_gnn_analyzer(training_data):
    """Train the GNN analyzer"""
    print("\nTraining GNN analyzer...")
    
    # Initialize analyzer
    node_feature_dim = 11  # Number of node features
    analyzer = GNNAnalyzer(
        node_feature_dim=node_feature_dim,
        hidden_dim=64,
        window_size=60
    )
    
    # Configure training parameters
    analyzer.num_epochs = 30
    analyzer.batch_size = 8
    analyzer.learning_rate = 0.001
    
    # Train the model
    training_stats = analyzer.train(training_data)
    
    print(f"Training completed:")
    print(f"  - Epochs: {training_stats['epochs_trained']}")
    print(f"  - Final loss: {training_stats['final_loss']:.6f}")
    print(f"  - Training graphs: {training_stats['training_graphs']}")
    print(f"  - Valid graph ratio: {training_stats['valid_graph_ratio']:.2%}")
    
    return analyzer


def evaluate_attack_scenarios(analyzer):
    """Evaluate analyzer on different attack scenarios"""
    print("\n" + "="*60)
    print("EVALUATING ATTACK SCENARIOS")
    print("="*60)
    
    # Test scenarios
    scenarios = {
        "Normal Traffic": generate_normal_network_traffic(60),
        "SYN Flood Attack": generate_ddos_attack_traffic("syn_flood", 80),
        "UDP Flood Attack": generate_ddos_attack_traffic("udp_flood", 70),
        "Distributed Scan": generate_ddos_attack_traffic("distributed_scan", 90)
    }
    
    for scenario_name, packets in scenarios.items():
        print(f"\n{scenario_name.upper()}:")
        print("-" * 40)
        
        try:
            # Make prediction
            is_malicious, confidence, explanation = analyzer.predict(packets)
            
            print(f"  Packets analyzed: {len(packets)}")
            print(f"  Classification: {'MALICIOUS' if is_malicious else 'BENIGN'}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Malicious probability: {explanation.get('malicious_probability', 'N/A'):.3f}")
            print(f"  Graph nodes: {explanation.get('num_nodes', 'N/A')}")
            print(f"  Graph edges: {explanation.get('num_edges', 'N/A')}")
            print(f"  Graph density: {explanation.get('graph_density', 'N/A'):.4f}")
            
        except Exception as e:
            print(f"  Error: {e}")


def analyze_network_structures(analyzer):
    """Analyze network structures of different traffic patterns"""
    print("\n" + "="*60)
    print("NETWORK STRUCTURE ANALYSIS")
    print("="*60)
    
    scenarios = {
        "Normal Web Traffic": generate_normal_network_traffic(50),
        "SYN Flood": generate_ddos_attack_traffic("syn_flood", 60),
        "Distributed Scan": generate_ddos_attack_traffic("distributed_scan", 70)
    }
    
    for scenario_name, packets in scenarios.items():
        print(f"\n{scenario_name}:")
        print("-" * 30)
        
        analysis = analyzer.analyze_network_structure(packets)
        
        if 'error' in analysis:
            print(f"  Error: {analysis['error']}")
            continue
        
        print(f"  Nodes: {analysis['num_nodes']}")
        print(f"  Edges: {analysis['num_edges']}")
        print(f"  Density: {analysis['density']:.4f}")
        print(f"  Avg Clustering: {analysis['avg_clustering']:.4f}")
        print(f"  Connected Components: {analysis['num_connected_components']}")
        
        if analysis['diameter'] != -1:
            print(f"  Diameter: {analysis['diameter']}")
            print(f"  Avg Shortest Path: {analysis['avg_shortest_path']:.2f}")
        else:
            print(f"  Graph is not connected")
        
        if 'avg_degree' in analysis:
            print(f"  Avg Degree: {analysis['avg_degree']:.2f}")
            print(f"  Max Degree: {analysis['max_degree']}")
            print(f"  Degree Std: {analysis['degree_std']:.2f}")


def demonstrate_model_persistence(analyzer):
    """Demonstrate model saving and loading"""
    print("\n" + "="*60)
    print("MODEL PERSISTENCE DEMONSTRATION")
    print("="*60)
    
    # Test data
    test_packets = generate_normal_network_traffic(40)
    
    # Make prediction with original model
    original_prediction = analyzer.predict(test_packets)
    print(f"Original model prediction: {original_prediction[0]} (confidence: {original_prediction[1]:.3f})")
    
    # Save model
    model_path = "gnn_model.pth"
    save_success = analyzer.save_model(model_path)
    print(f"Model saved: {save_success}")
    
    if save_success:
        # Create new analyzer and load model
        new_analyzer = GNNAnalyzer(analyzer.node_feature_dim)
        load_success = new_analyzer.load_model(model_path)
        print(f"Model loaded: {load_success}")
        
        if load_success:
            # Make prediction with loaded model
            loaded_prediction = new_analyzer.predict(test_packets)
            print(f"Loaded model prediction: {loaded_prediction[0]} (confidence: {loaded_prediction[1]:.3f})")
            
            # Verify predictions match
            predictions_match = (
                original_prediction[0] == loaded_prediction[0] and
                abs(original_prediction[1] - loaded_prediction[1]) < 0.01
            )
            print(f"Predictions match: {predictions_match}")
        
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)
            print("Model file cleaned up")


def main():
    """Main demonstration function"""
    print("DDoS.AI Graph Neural Network Analyzer Demonstration")
    print("=" * 60)
    
    try:
        # Create training data
        training_data = create_training_data()
        
        # Train GNN analyzer
        analyzer = train_gnn_analyzer(training_data)
        
        # Evaluate on different attack scenarios
        evaluate_attack_scenarios(analyzer)
        
        # Analyze network structures
        analyze_network_structures(analyzer)
        
        # Demonstrate model persistence
        demonstrate_model_persistence(analyzer)
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Insights:")
        print("1. GNN captures network-level patterns and relationships")
        print("2. Graph structure reveals attack characteristics")
        print("3. Different attacks show distinct network topologies")
        print("4. Node features encode IP-level behavior patterns")
        print("5. Model can distinguish coordinated vs. normal traffic")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Add missing import
    from typing import List
    main()