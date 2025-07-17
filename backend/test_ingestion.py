#!/usr/bin/env python3
"""
Test script for traffic ingestion system
"""
import sys
import os

# Add backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion.traffic_ingestor import TrafficIngestor
from models.data_models import ProtocolType

def test_csv_ingestion():
    """Test CSV traffic ingestion"""
    print("Testing CSV traffic ingestion...")
    
    try:
        ingestor = TrafficIngestor()
        
        # Test with sample CSV data
        csv_file = 'data/sample_traffic.csv'
        if not os.path.exists(csv_file):
            print(f"Sample CSV file not found: {csv_file}")
            return
        
        print(f"Ingesting from: {csv_file}")
        packets = list(ingestor.ingest(csv_file, max_packets=5))
        
        print(f"Successfully ingested {len(packets)} packets")
        
        if packets:
            for i, packet in enumerate(packets[:3]):
                print(f"  Packet {i+1}: {packet.src_ip}:{packet.src_port} -> {packet.dst_ip}:{packet.dst_port} ({packet.protocol.value})")
                print(f"    Size: {packet.packet_size} bytes, TTL: {packet.ttl}, Entropy: {packet.payload_entropy:.2f}")
        
        # Get statistics
        stats = ingestor.get_statistics('csv')
        print(f"\nIngestion Statistics:")
        print(f"  Packets processed: {stats['packets_processed']}")
        print(f"  Errors encountered: {stats['errors_encountered']}")
        print(f"  Duration: {stats['duration_seconds']:.2f} seconds")
        print(f"  Rate: {stats['packets_per_second']:.2f} packets/sec")
        
    except Exception as e:
        print(f"Error during CSV ingestion: {e}")
        import traceback
        traceback.print_exc()

def test_source_detection():
    """Test automatic source type detection"""
    print("\nTesting source type detection...")
    
    ingestor = TrafficIngestor()
    
    test_sources = [
        ('data.csv', 'csv'),
        ('capture.pcap', 'pcap'),
        ('traffic.pcapng', 'pcap'),
        ('localhost:8080', 'socket'),
        ('eth0', 'live'),
        ('wlan0', 'live')
    ]
    
    for source, expected in test_sources:
        detected = ingestor._detect_source_type(source)
        status = "✓" if detected == expected else "✗"
        print(f"  {status} {source} -> {detected} (expected: {expected})")

def test_supported_sources():
    """Test getting supported source information"""
    print("\nSupported traffic sources:")
    
    ingestor = TrafficIngestor()
    sources = ingestor.get_supported_sources()
    
    for source_type, info in sources.items():
        print(f"  {source_type.upper()}: {info['description']}")
        print(f"    Example: {info['example']}")
        if 'extensions' in info:
            print(f"    Extensions: {', '.join(info['extensions'])}")
        if 'required_columns' in info:
            print(f"    Required columns: {', '.join(info['required_columns'])}")
        print()

if __name__ == "__main__":
    print("DDoS.AI Traffic Ingestion System Test")
    print("=" * 40)
    
    test_source_detection()
    test_supported_sources()
    test_csv_ingestion()
    
    print("\nTest completed!")