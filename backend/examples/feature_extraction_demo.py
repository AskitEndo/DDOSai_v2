"""
Feature extraction demonstration script
"""
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.feature_extractor import FeatureExtractor
from models.data_models import TrafficPacket, ProtocolType
from ingestion.traffic_ingestor import TrafficIngestor


def create_sample_packets():
    """Create sample traffic packets for demonstration"""
    packets = []
    base_time = datetime.now()
    
    # Normal HTTP traffic
    for i in range(5):
        packet = TrafficPacket(
            timestamp=base_time + timedelta(seconds=i),
            src_ip="192.168.1.100",
            dst_ip="93.184.216.34",  # example.com
            src_port=12345 + i,
            dst_port=80,
            protocol=ProtocolType.HTTP,
            packet_size=512 + i * 100,
            ttl=64,
            flags=["SYN", "ACK"] if i > 0 else ["SYN"],
            payload_entropy=0.6 + i * 0.05,
            packet_id=f"http_{i:03d}"
        )
        packets.append(packet)
    
    # Suspicious SYN flood pattern
    for i in range(10):
        packet = TrafficPacket(
            timestamp=base_time + timedelta(seconds=10 + i * 0.1),
            src_ip="203.0.113.45",  # Suspicious IP
            dst_ip="192.168.1.1",   # Target server
            src_port=12000 + i,
            dst_port=80,
            protocol=ProtocolType.TCP,
            packet_size=64,  # Small SYN packets
            ttl=32,
            flags=["SYN"],  # Only SYN flags
            payload_entropy=0.1,  # Low entropy
            packet_id=f"syn_{i:03d}"
        )
        packets.append(packet)
    
    # UDP flood pattern
    for i in range(8):
        packet = TrafficPacket(
            timestamp=base_time + timedelta(seconds=15 + i * 0.05),
            src_ip="198.51.100.23",
            dst_ip="192.168.1.1",
            src_port=54321,
            dst_port=53 + i,  # Different destination ports
            protocol=ProtocolType.UDP,
            packet_size=1024,
            ttl=128,
            flags=[],
            payload_entropy=0.9,  # High entropy (random data)
            packet_id=f"udp_{i:03d}"
        )
        packets.append(packet)
    
    return packets


def demonstrate_packet_features(extractor, packets):
    """Demonstrate packet-level feature extraction"""
    print("=== Packet-Level Feature Extraction ===")
    
    for i, packet in enumerate(packets[:3]):  # Show first 3 packets
        features = extractor.extract_packet_features(packet)
        print(f"\nPacket {i+1} ({packet.packet_id}):")
        print(f"  Source: {packet.src_ip}:{packet.src_port}")
        print(f"  Destination: {packet.dst_ip}:{packet.dst_port}")
        print(f"  Protocol: {packet.protocol.value}")
        print(f"  Size: {packet.packet_size} bytes")
        print(f"  Flags: {packet.flags}")
        print(f"  Feature vector shape: {features.shape}")
        print(f"  Sample features: {features[:5]}")  # Show first 5 features


def demonstrate_flow_features(extractor, packets):
    """Demonstrate flow-level feature extraction"""
    print("\n=== Flow-Level Feature Extraction ===")
    
    # Group packets by flow (src_ip, dst_ip, protocol)
    flows = {}
    for packet in packets:
        flow_key = (packet.src_ip, packet.dst_ip, packet.protocol.value)
        if flow_key not in flows:
            flows[flow_key] = []
        flows[flow_key].append(packet)
    
    for i, (flow_key, flow_packets) in enumerate(flows.items()):
        if i >= 3:  # Show first 3 flows
            break
            
        features = extractor.extract_flow_features(flow_packets)
        src_ip, dst_ip, protocol = flow_key
        
        print(f"\nFlow {i+1}: {src_ip} -> {dst_ip} ({protocol})")
        print(f"  Packets in flow: {len(flow_packets)}")
        print(f"  Duration: {(flow_packets[-1].timestamp - flow_packets[0].timestamp).total_seconds():.2f}s")
        print(f"  Feature vector shape: {features.shape}")
        print(f"  Packet count (log): {features[0]:.3f}")
        print(f"  Total bytes (log): {features[1]:.3f}")
        print(f"  Average packet size: {features[3]:.1f}")


def demonstrate_network_features(extractor, packets):
    """Demonstrate network-level feature extraction"""
    print("\n=== Network-Level Feature Extraction ===")
    
    # Analyze different time windows
    time_windows = [
        ("Normal traffic", packets[:5]),
        ("SYN flood period", packets[5:15]),
        ("UDP flood period", packets[15:]),
        ("All traffic", packets)
    ]
    
    for window_name, window_packets in time_windows:
        features = extractor.extract_network_features(window_packets)
        
        print(f"\n{window_name}:")
        print(f"  Packets analyzed: {len(window_packets)}")
        print(f"  Unique source IPs: {int(features[0])}")
        print(f"  Unique destination IPs: {int(features[1])}")
        print(f"  Total unique IPs: {int(features[2])}")
        print(f"  Connections: {int(features[4])}")
        print(f"  Protocol entropy: {features[6]:.3f}")
        print(f"  Packet rate: {features[8]:.1f} pps")
        print(f"  Byte rate: {features[9]:.1f} Bps")


def demonstrate_feature_names(extractor):
    """Show available feature names"""
    print("\n=== Available Feature Names ===")
    
    names = extractor.get_feature_names()
    print(f"Total features: {len(names)}")
    print("\nFeature categories:")
    
    categories = {
        "Basic": [n for n in names if any(x in n for x in ['size', 'ttl', 'entropy', 'port'])],
        "Protocol": [n for n in names if 'proto_' in n],
        "Flags": [n for n in names if 'flag_' in n],
        "IP": [n for n in names if any(x in n for x in ['private', 'loopback', 'subnet'])],
        "Port": [n for n in names if any(x in n for x in ['well_known', 'system', 'user', 'dynamic'])],
        "Temporal": [n for n in names if any(x in n for x in ['hour', 'weekend'])]
    }
    
    for category, feature_names in categories.items():
        print(f"  {category}: {len(feature_names)} features")
        if feature_names:
            print(f"    Examples: {', '.join(feature_names[:3])}")


def analyze_attack_patterns(extractor, packets):
    """Analyze and compare different attack patterns"""
    print("\n=== Attack Pattern Analysis ===")
    
    # Separate packets by pattern
    normal_packets = [p for p in packets if 'http_' in p.packet_id]
    syn_flood_packets = [p for p in packets if 'syn_' in p.packet_id]
    udp_flood_packets = [p for p in packets if 'udp_' in p.packet_id]
    
    patterns = [
        ("Normal HTTP", normal_packets),
        ("SYN Flood", syn_flood_packets),
        ("UDP Flood", udp_flood_packets)
    ]
    
    for pattern_name, pattern_packets in patterns:
        if not pattern_packets:
            continue
            
        # Extract network features for this pattern
        network_features = extractor.extract_network_features(pattern_packets)
        
        # Extract packet features for statistical analysis
        packet_features = [extractor.extract_packet_features(p) for p in pattern_packets]
        
        print(f"\n{pattern_name} Pattern:")
        print(f"  Packets: {len(pattern_packets)}")
        print(f"  Unique source IPs: {int(network_features[0])}")
        print(f"  Connection density: {network_features[5]:.3f}")
        print(f"  Protocol entropy: {network_features[6]:.3f}")
        
        # Analyze packet-level patterns
        if packet_features:
            import numpy as np
            packet_matrix = np.array(packet_features)
            
            # Look at specific features that might indicate attacks
            entropy_idx = 2  # payload_entropy is at index 2
            syn_flag_idx = 11  # flag_syn is at index 11 (after protocol encoding)
            
            avg_entropy = np.mean(packet_matrix[:, entropy_idx])
            syn_ratio = np.mean(packet_matrix[:, syn_flag_idx])
            
            print(f"  Average payload entropy: {avg_entropy:.3f}")
            print(f"  SYN flag ratio: {syn_ratio:.3f}")


def main():
    """Main demonstration function"""
    print("DDoS.AI Feature Extraction Demonstration")
    print("=" * 50)
    
    # Initialize feature extractor
    extractor = FeatureExtractor(window_size=60)
    
    # Create sample traffic data
    print("Creating sample traffic data...")
    packets = create_sample_packets()
    print(f"Generated {len(packets)} sample packets")
    
    # Demonstrate different types of feature extraction
    demonstrate_packet_features(extractor, packets)
    demonstrate_flow_features(extractor, packets)
    demonstrate_network_features(extractor, packets)
    demonstrate_feature_names(extractor)
    analyze_attack_patterns(extractor, packets)
    
    print("\n" + "=" * 50)
    print("Feature extraction demonstration completed!")
    print("\nNext steps:")
    print("1. Use these features to train AI models")
    print("2. Implement real-time feature extraction pipeline")
    print("3. Add feature selection and dimensionality reduction")
    print("4. Integrate with traffic ingestion system")


if __name__ == "__main__":
    main()