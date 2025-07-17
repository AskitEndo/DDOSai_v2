"""
Autoencoder anomaly detection demonstration script
"""
import sys
import os
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.autoencoder_detector import AutoencoderDetector
from core.feature_extractor import FeatureExtractor
from models.data_models import TrafficPacket, ProtocolType


def generate_training_data(n_samples: int = 2000) -> np.ndarray:
    """Generate synthetic normal traffic data for training"""
    print(f"Generating {n_samples} normal traffic samples for training...")
    
    # Create feature extractor
    extractor = FeatureExtractor()
    
    # Generate normal traffic packets
    packets = []
    base_time = datetime.now()
    
    for i in range(n_samples):
        # Generate valid values first
        if i % 3 == 0:  # HTTP traffic
            packet_size = max(64, min(1500, int(np.random.normal(800, 200))))
            payload_entropy = max(0.0, min(1.0, np.random.normal(0.6, 0.1)))
            
            packet = TrafficPacket(
                timestamp=base_time + timedelta(seconds=i * 0.1),
                src_ip=f"192.168.1.{100 + (i % 50)}",
                dst_ip="93.184.216.34",  # example.com
                src_port=12000 + (i % 1000),
                dst_port=80,
                protocol=ProtocolType.HTTP,
                packet_size=packet_size,
                ttl=64,
                flags=["SYN", "ACK"] if i % 4 != 0 else ["SYN"],
                payload_entropy=payload_entropy,
                packet_id=f"normal_http_{i:05d}"
            )
        elif i % 3 == 1:  # HTTPS traffic
            packet_size = max(64, min(1500, int(np.random.normal(600, 150))))
            payload_entropy = max(0.0, min(1.0, np.random.normal(0.7, 0.1)))
            
            packet = TrafficPacket(
                timestamp=base_time + timedelta(seconds=i * 0.1),
                src_ip=f"192.168.1.{100 + (i % 50)}",
                dst_ip="1.1.1.1",  # Cloudflare DNS
                src_port=12000 + (i % 1000),
                dst_port=443,
                protocol=ProtocolType.HTTPS,
                packet_size=packet_size,
                ttl=64,
                flags=["SYN", "ACK"],
                payload_entropy=payload_entropy,
                packet_id=f"normal_https_{i:05d}"
            )
        else:  # DNS traffic
            packet_size = max(64, min(1500, int(np.random.normal(100, 30))))
            payload_entropy = max(0.0, min(1.0, np.random.normal(0.4, 0.1)))
            
            packet = TrafficPacket(
                timestamp=base_time + timedelta(seconds=i * 0.1),
                src_ip=f"192.168.1.{100 + (i % 50)}",
                dst_ip="8.8.8.8",  # Google DNS
                src_port=12000 + (i % 1000),
                dst_port=53,
                protocol=ProtocolType.UDP,
                packet_size=packet_size,
                ttl=64,
                flags=[],
                payload_entropy=payload_entropy,
                packet_id=f"normal_dns_{i:05d}"
            )
        
        packets.append(packet)
    
    # Extract features from packets
    features = []
    for packet in packets:
        feature_vector = extractor.extract_packet_features(packet)
        features.append(feature_vector)
    
    return np.array(features)


def generate_attack_scenarios() -> dict:
    """Generate different attack scenarios for testing"""
    print("Generating attack scenarios...")
    
    extractor = FeatureExtractor()
    scenarios = {}
    base_time = datetime.now()
    
    # 1. SYN Flood Attack
    syn_flood_packets = []
    for i in range(100):
        packet = TrafficPacket(
            timestamp=base_time + timedelta(milliseconds=i * 10),
            src_ip=f"203.0.113.{i % 50}",  # Distributed sources
            dst_ip="192.168.1.1",  # Target server
            src_port=12000 + i,
            dst_port=80,
            protocol=ProtocolType.TCP,
            packet_size=64,  # Small SYN packets
            ttl=32,  # Lower TTL
            flags=["SYN"],  # Only SYN flags
            payload_entropy=0.1,  # Low entropy
            packet_id=f"syn_flood_{i:03d}"
        )
        syn_flood_packets.append(packet)
    
    scenarios['syn_flood'] = np.array([
        extractor.extract_packet_features(p) for p in syn_flood_packets
    ])
    
    # 2. UDP Flood Attack
    udp_flood_packets = []
    for i in range(80):
        packet = TrafficPacket(
            timestamp=base_time + timedelta(milliseconds=i * 5),
            src_ip=f"198.51.100.{i % 30}",
            dst_ip="192.168.1.1",
            src_port=54321,
            dst_port=53 + (i % 10),  # Multiple target ports
            protocol=ProtocolType.UDP,
            packet_size=1024,  # Large packets
            ttl=128,
            flags=[],
            payload_entropy=0.9,  # High entropy (random data)
            packet_id=f"udp_flood_{i:03d}"
        )
        udp_flood_packets.append(packet)
    
    scenarios['udp_flood'] = np.array([
        extractor.extract_packet_features(p) for p in udp_flood_packets
    ])
    
    # 3. Port Scanning
    port_scan_packets = []
    target_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 8080, 8443]
    for i, port in enumerate(target_ports):
        packet = TrafficPacket(
            timestamp=base_time + timedelta(milliseconds=i * 100),
            src_ip="172.16.0.100",  # Single scanner
            dst_ip="192.168.1.1",
            src_port=12345 + i,
            dst_port=port,
            protocol=ProtocolType.TCP,
            packet_size=64,
            ttl=64,
            flags=["SYN"],
            payload_entropy=0.2,
            packet_id=f"port_scan_{i:03d}"
        )
        port_scan_packets.append(packet)
    
    scenarios['port_scan'] = np.array([
        extractor.extract_packet_features(p) for p in port_scan_packets
    ])
    
    # 4. Normal traffic (for comparison)
    normal_test_packets = []
    for i in range(50):
        packet = TrafficPacket(
            timestamp=base_time + timedelta(seconds=i),
            src_ip=f"192.168.1.{150 + (i % 20)}",
            dst_ip="93.184.216.34",
            src_port=13000 + i,
            dst_port=443,
            protocol=ProtocolType.HTTPS,
            packet_size=int(np.random.normal(700, 100)),
            ttl=64,
            flags=["SYN", "ACK"],
            payload_entropy=np.random.normal(0.65, 0.05),
            packet_id=f"normal_test_{i:03d}"
        )
        packet.packet_size = max(64, min(1500, packet.packet_size))
        packet.payload_entropy = max(0.0, min(1.0, packet.payload_entropy))
        normal_test_packets.append(packet)
    
    scenarios['normal'] = np.array([
        extractor.extract_packet_features(p) for p in normal_test_packets
    ])
    
    return scenarios


def train_autoencoder(training_data: np.ndarray) -> AutoencoderDetector:
    """Train the autoencoder detector"""
    print("\nTraining autoencoder detector...")
    
    # Initialize detector
    input_dim = training_data.shape[1]
    detector = AutoencoderDetector(
        input_dim=input_dim,
        hidden_dims=[64, 32, 16, 32, 64],
        threshold_percentile=95.0
    )
    
    # Configure training parameters
    detector.num_epochs = 50
    detector.batch_size = 32
    detector.learning_rate = 0.001
    
    # Train the model
    training_stats = detector.train(training_data)
    
    print(f"Training completed:")
    print(f"  - Epochs: {training_stats['epochs_trained']}")
    print(f"  - Final loss: {training_stats['final_loss']:.6f}")
    print(f"  - Anomaly threshold: {training_stats['anomaly_threshold']:.6f}")
    print(f"  - Training samples: {training_stats['training_samples']}")
    
    return detector


def evaluate_scenarios(detector: AutoencoderDetector, scenarios: dict):
    """Evaluate detector on different attack scenarios"""
    print("\n" + "="*60)
    print("EVALUATING ATTACK SCENARIOS")
    print("="*60)
    
    for scenario_name, scenario_data in scenarios.items():
        print(f"\n{scenario_name.upper().replace('_', ' ')} SCENARIO:")
        print("-" * 40)
        
        # Make predictions
        predictions = detector.predict_batch(scenario_data)
        
        # Calculate statistics
        total_samples = len(predictions)
        malicious_count = sum(1 for pred in predictions if pred[0])
        detection_rate = malicious_count / total_samples
        avg_confidence = np.mean([pred[1] for pred in predictions])
        avg_reconstruction_error = np.mean([
            pred[2]['reconstruction_error'] for pred in predictions
        ])
        
        print(f"  Samples analyzed: {total_samples}")
        print(f"  Detected as malicious: {malicious_count} ({detection_rate:.1%})")
        print(f"  Average confidence: {avg_confidence:.3f}")
        print(f"  Average reconstruction error: {avg_reconstruction_error:.6f}")
        print(f"  Anomaly threshold: {detector.anomaly_threshold:.6f}")
        
        # Show some individual predictions
        print(f"  Sample predictions:")
        for i, (is_mal, conf, exp) in enumerate(predictions[:3]):
            status = "MALICIOUS" if is_mal else "NORMAL"
            print(f"    Sample {i+1}: {status} (confidence: {conf:.3f}, error: {exp['reconstruction_error']:.6f})")


def demonstrate_model_persistence(detector: AutoencoderDetector, test_data: np.ndarray):
    """Demonstrate model saving and loading"""
    print("\n" + "="*60)
    print("MODEL PERSISTENCE DEMONSTRATION")
    print("="*60)
    
    # Make prediction with original model
    original_prediction = detector.predict(test_data[0])
    print(f"Original model prediction: {original_prediction[0]} (confidence: {original_prediction[1]:.3f})")
    
    # Save model
    model_path = "autoencoder_model.pth"
    save_success = detector.save_model(model_path)
    print(f"Model saved: {save_success}")
    
    # Create new detector and load model
    new_detector = AutoencoderDetector(detector.input_dim)
    load_success = new_detector.load_model(model_path)
    print(f"Model loaded: {load_success}")
    
    # Make prediction with loaded model
    loaded_prediction = new_detector.predict(test_data[0])
    print(f"Loaded model prediction: {loaded_prediction[0]} (confidence: {loaded_prediction[1]:.3f})")
    
    # Verify predictions match
    predictions_match = (
        original_prediction[0] == loaded_prediction[0] and
        abs(original_prediction[1] - loaded_prediction[1]) < 0.001
    )
    print(f"Predictions match: {predictions_match}")
    
    # Clean up
    if os.path.exists(model_path):
        os.remove(model_path)
        print("Model file cleaned up")


def analyze_reconstruction_statistics(detector: AutoencoderDetector):
    """Analyze reconstruction error statistics"""
    print("\n" + "="*60)
    print("RECONSTRUCTION ERROR ANALYSIS")
    print("="*60)
    
    stats = detector.get_reconstruction_statistics()
    
    print(f"Training reconstruction errors:")
    print(f"  Count: {stats['count']}")
    print(f"  Mean: {stats['mean']:.6f}")
    print(f"  Std: {stats['std']:.6f}")
    print(f"  Min: {stats['min']:.6f}")
    print(f"  Max: {stats['max']:.6f}")
    print(f"  Median: {stats['median']:.6f}")
    
    print(f"\nPercentiles:")
    for percentile, value in stats['percentiles'].items():
        print(f"  {percentile}th: {value:.6f}")
    
    print(f"\nCurrent anomaly threshold: {detector.anomaly_threshold:.6f}")
    print(f"Threshold percentile: {detector.threshold_percentile}%")


def main():
    """Main demonstration function"""
    print("DDoS.AI Autoencoder Anomaly Detection Demonstration")
    print("=" * 60)
    
    try:
        # Generate training data
        training_data = generate_training_data(1500)
        print(f"Training data shape: {training_data.shape}")
        
        # Train autoencoder
        detector = train_autoencoder(training_data)
        
        # Generate attack scenarios
        scenarios = generate_attack_scenarios()
        
        # Evaluate on different scenarios
        evaluate_scenarios(detector, scenarios)
        
        # Demonstrate model persistence
        demonstrate_model_persistence(detector, scenarios['normal'])
        
        # Analyze reconstruction statistics
        analyze_reconstruction_statistics(detector)
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Insights:")
        print("1. Autoencoder learns normal traffic patterns during training")
        print("2. Anomalous traffic produces higher reconstruction errors")
        print("3. Threshold can be adjusted based on desired sensitivity")
        print("4. Model can be saved and loaded for production use")
        print("5. Different attack types show different error patterns")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()