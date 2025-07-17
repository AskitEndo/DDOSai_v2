"""
Unit tests for data models and validation
"""
import pytest
from datetime import datetime, timedelta
from backend.models.data_models import (
    TrafficPacket, NetworkFlow, DetectionResult, NetworkNode, NetworkEdge,
    ProtocolType, AttackType, serialize_models, validate_json_data
)
from backend.core.exceptions import ValidationError


class TestTrafficPacket:
    """Test cases for TrafficPacket data model"""
    
    def test_valid_packet_creation(self):
        """Test creating a valid traffic packet"""
        packet = TrafficPacket(
            timestamp=datetime.now(),
            src_ip="192.168.1.100",
            dst_ip="10.0.0.1",
            src_port=12345,
            dst_port=80,
            protocol=ProtocolType.TCP,
            packet_size=1500,
            ttl=64,
            flags=["SYN"],
            payload_entropy=0.75,
            packet_id="pkt_001"
        )
        assert packet.src_ip == "192.168.1.100"
        assert packet.protocol == ProtocolType.TCP
        assert packet.validate() is True
    
    def test_invalid_ip_address(self):
        """Test packet creation with invalid IP address"""
        with pytest.raises(ValidationError, match="Invalid IP address"):
            TrafficPacket(
                timestamp=datetime.now(),
                src_ip="invalid_ip",
                dst_ip="10.0.0.1",
                src_port=12345,
                dst_port=80,
                protocol=ProtocolType.TCP,
                packet_size=1500,
                ttl=64,
                flags=["SYN"],
                payload_entropy=0.75
            )
    
    def test_invalid_port_range(self):
        """Test packet creation with invalid port numbers"""
        with pytest.raises(ValidationError, match="Invalid source port"):
            TrafficPacket(
                timestamp=datetime.now(),
                src_ip="192.168.1.100",
                dst_ip="10.0.0.1",
                src_port=70000,  # Invalid port
                dst_port=80,
                protocol=ProtocolType.TCP,
                packet_size=1500,
                ttl=64,
                flags=["SYN"],
                payload_entropy=0.75
            )
    
    def test_invalid_packet_size(self):
        """Test packet creation with invalid packet size"""
        with pytest.raises(ValidationError, match="Invalid packet size"):
            TrafficPacket(
                timestamp=datetime.now(),
                src_ip="192.168.1.100",
                dst_ip="10.0.0.1",
                src_port=12345,
                dst_port=80,
                protocol=ProtocolType.TCP,
                packet_size=0,  # Invalid size
                ttl=64,
                flags=["SYN"],
                payload_entropy=0.75
            )
    
    def test_invalid_entropy_range(self):
        """Test packet creation with invalid entropy value"""
        with pytest.raises(ValidationError, match="Invalid payload entropy"):
            TrafficPacket(
                timestamp=datetime.now(),
                src_ip="192.168.1.100",
                dst_ip="10.0.0.1",
                src_port=12345,
                dst_port=80,
                protocol=ProtocolType.TCP,
                packet_size=1500,
                ttl=64,
                flags=["SYN"],
                payload_entropy=1.5  # Invalid entropy > 1.0
            )
    
    def test_invalid_tcp_flags(self):
        """Test packet creation with invalid TCP flags"""
        with pytest.raises(ValidationError, match="Invalid TCP flag"):
            TrafficPacket(
                timestamp=datetime.now(),
                src_ip="192.168.1.100",
                dst_ip="10.0.0.1",
                src_port=12345,
                dst_port=80,
                protocol=ProtocolType.TCP,
                packet_size=1500,
                ttl=64,
                flags=["INVALID_FLAG"],
                payload_entropy=0.75
            )
    
    def test_packet_serialization(self):
        """Test packet to_dict and from_dict methods"""
        original_packet = TrafficPacket(
            timestamp=datetime.now(),
            src_ip="192.168.1.100",
            dst_ip="10.0.0.1",
            src_port=12345,
            dst_port=80,
            protocol=ProtocolType.TCP,
            packet_size=1500,
            ttl=64,
            flags=["SYN", "ACK"],
            payload_entropy=0.75,
            packet_id="pkt_001"
        )
        
        # Test serialization
        packet_dict = original_packet.to_dict()
        assert packet_dict["src_ip"] == "192.168.1.100"
        assert packet_dict["protocol"] == "TCP"
        
        # Test deserialization
        reconstructed_packet = TrafficPacket.from_dict(packet_dict)
        assert reconstructed_packet.src_ip == original_packet.src_ip
        assert reconstructed_packet.protocol == original_packet.protocol


class TestNetworkFlow:
    """Test cases for NetworkFlow data model"""
    
    def test_valid_flow_creation(self):
        """Test creating a valid network flow"""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=30)
        
        flow = NetworkFlow(
            flow_id="flow_001",
            src_ip="192.168.1.100",
            dst_ip="10.0.0.1",
            start_time=start_time,
            end_time=end_time,
            packet_count=100,
            byte_count=150000,
            avg_packet_size=1500,
            flow_duration=30.0,
            protocol=ProtocolType.TCP,
            src_port_range=[12345, 12350],
            dst_port_range=[80]
        )
        assert flow.flow_id == "flow_001"
        assert flow.validate() is True
    
    def test_invalid_time_ordering(self):
        """Test flow creation with invalid time ordering"""
        start_time = datetime.now()
        end_time = start_time - timedelta(seconds=30)  # End before start
        
        with pytest.raises(ValidationError, match="start time must be before end time"):
            NetworkFlow(
                flow_id="flow_001",
                src_ip="192.168.1.100",
                dst_ip="10.0.0.1",
                start_time=start_time,
                end_time=end_time,
                packet_count=100,
                byte_count=150000,
                avg_packet_size=1500,
                flow_duration=30.0,
                protocol=ProtocolType.TCP,
                src_port_range=[12345],
                dst_port_range=[80]
            )
    
    def test_negative_counts(self):
        """Test flow creation with negative counts"""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=30)
        
        with pytest.raises(ValidationError, match="Invalid packet count"):
            NetworkFlow(
                flow_id="flow_001",
                src_ip="192.168.1.100",
                dst_ip="10.0.0.1",
                start_time=start_time,
                end_time=end_time,
                packet_count=-1,  # Invalid negative count
                byte_count=150000,
                avg_packet_size=1500,
                flow_duration=30.0,
                protocol=ProtocolType.TCP,
                src_port_range=[12345],
                dst_port_range=[80]
            )


class TestDetectionResult:
    """Test cases for DetectionResult data model"""
    
    def test_valid_detection_result(self):
        """Test creating a valid detection result"""
        result = DetectionResult(
            timestamp=datetime.now(),
            packet_id="pkt_001",
            flow_id="flow_001",
            is_malicious=True,
            threat_score=85,
            attack_type=AttackType.SYN_FLOOD,
            detection_method="autoencoder",
            confidence=0.92,
            explanation={"top_features": ["packet_rate", "syn_flags"]},
            model_version="1.0.0"
        )
        assert result.threat_score == 85
        assert result.validate() is True
    
    def test_invalid_threat_score(self):
        """Test detection result with invalid threat score"""
        with pytest.raises(ValidationError, match="Invalid threat score"):
            DetectionResult(
                timestamp=datetime.now(),
                packet_id="pkt_001",
                flow_id="flow_001",
                is_malicious=True,
                threat_score=150,  # Invalid score > 100
                attack_type=AttackType.SYN_FLOOD,
                detection_method="autoencoder",
                confidence=0.92,
                explanation={"top_features": ["packet_rate"]},
                model_version="1.0.0"
            )
    
    def test_invalid_confidence(self):
        """Test detection result with invalid confidence"""
        with pytest.raises(ValidationError, match="Invalid confidence"):
            DetectionResult(
                timestamp=datetime.now(),
                packet_id="pkt_001",
                flow_id="flow_001",
                is_malicious=True,
                threat_score=85,
                attack_type=AttackType.SYN_FLOOD,
                detection_method="autoencoder",
                confidence=1.5,  # Invalid confidence > 1.0
                explanation={"top_features": ["packet_rate"]},
                model_version="1.0.0"
            )
    
    def test_invalid_detection_method(self):
        """Test detection result with invalid detection method"""
        with pytest.raises(ValidationError, match="Invalid detection method"):
            DetectionResult(
                timestamp=datetime.now(),
                packet_id="pkt_001",
                flow_id="flow_001",
                is_malicious=True,
                threat_score=85,
                attack_type=AttackType.SYN_FLOOD,
                detection_method="invalid_method",
                confidence=0.92,
                explanation={"top_features": ["packet_rate"]},
                model_version="1.0.0"
            )


class TestNetworkNode:
    """Test cases for NetworkNode data model"""
    
    def test_valid_node_creation(self):
        """Test creating a valid network node"""
        now = datetime.now()
        node = NetworkNode(
            node_id="node_001",
            ip_address="192.168.1.100",
            packet_count=1000,
            byte_count=1500000,
            connection_count=50,
            threat_score=25,
            is_malicious=False,
            first_seen=now,
            last_seen=now + timedelta(minutes=30)
        )
        assert node.ip_address == "192.168.1.100"
        assert node.validate() is True
    
    def test_invalid_time_ordering_node(self):
        """Test node creation with invalid time ordering"""
        now = datetime.now()
        with pytest.raises(ValidationError, match="First seen time must be before"):
            NetworkNode(
                node_id="node_001",
                ip_address="192.168.1.100",
                packet_count=1000,
                byte_count=1500000,
                connection_count=50,
                threat_score=25,
                is_malicious=False,
                first_seen=now + timedelta(minutes=30),
                last_seen=now  # Last seen before first seen
            )


class TestNetworkEdge:
    """Test cases for NetworkEdge data model"""
    
    def test_valid_edge_creation(self):
        """Test creating a valid network edge"""
        edge = NetworkEdge(
            edge_id="edge_001",
            source_ip="192.168.1.100",
            target_ip="10.0.0.1",
            flow_count=10,
            total_bytes=150000,
            avg_packet_size=1500,
            connection_duration=60.0,
            protocols=[ProtocolType.TCP, ProtocolType.UDP]
        )
        assert edge.source_ip == "192.168.1.100"
        assert edge.validate() is True
    
    def test_empty_protocols_list(self):
        """Test edge creation with empty protocols list"""
        with pytest.raises(ValidationError, match="At least one protocol must be specified"):
            NetworkEdge(
                edge_id="edge_001",
                source_ip="192.168.1.100",
                target_ip="10.0.0.1",
                flow_count=10,
                total_bytes=150000,
                avg_packet_size=1500,
                connection_duration=60.0,
                protocols=[]  # Empty protocols list
            )


class TestUtilityFunctions:
    """Test cases for utility functions"""
    
    def test_serialize_models(self):
        """Test serializing a list of models"""
        packet1 = TrafficPacket(
            timestamp=datetime.now(),
            src_ip="192.168.1.100",
            dst_ip="10.0.0.1",
            src_port=12345,
            dst_port=80,
            protocol=ProtocolType.TCP,
            packet_size=1500,
            ttl=64,
            flags=["SYN"],
            payload_entropy=0.75
        )
        
        packet2 = TrafficPacket(
            timestamp=datetime.now(),
            src_ip="192.168.1.101",
            dst_ip="10.0.0.2",
            src_port=12346,
            dst_port=443,
            protocol=ProtocolType.TCP,
            packet_size=1200,
            ttl=64,
            flags=["ACK"],
            payload_entropy=0.65
        )
        
        serialized = serialize_models([packet1, packet2])
        assert len(serialized) == 2
        assert serialized[0]["src_ip"] == "192.168.1.100"
        assert serialized[1]["src_ip"] == "192.168.1.101"
    
    def test_validate_json_data(self):
        """Test JSON data validation"""
        data = {"field1": "value1", "field2": "value2"}
        required_fields = ["field1", "field2"]
        
        # Valid data
        assert validate_json_data(data, required_fields) is True
        
        # Missing field
        with pytest.raises(ValidationError, match="Missing required fields"):
            validate_json_data(data, ["field1", "field2", "field3"])


if __name__ == "__main__":
    pytest.main([__file__])