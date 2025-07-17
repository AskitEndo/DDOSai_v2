"""
Unit tests for traffic ingestion system
"""
import pytest
import tempfile
import os
import csv
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from backend.ingestion.traffic_ingestor import TrafficIngestor
from backend.ingestion.csv_ingestor import CsvIngestor
from backend.ingestion.pcap_ingestor import PcapIngestor
from backend.models.data_models import TrafficPacket, ProtocolType
from backend.core.exceptions import TrafficIngestionError


class TestCsvIngestor:
    """Test cases for CSV traffic ingestion"""
    
    def create_test_csv(self, filename: str, data: list):
        """Create a test CSV file with given data"""
        with open(filename, 'w', newline='') as csvfile:
            if data:
                writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
    
    def test_valid_csv_ingestion(self):
        """Test ingesting valid CSV data"""
        test_data = [
            {
                'timestamp': '2024-01-15T10:00:01',
                'src_ip': '192.168.1.100',
                'dst_ip': '10.0.0.1',
                'src_port': 12345,
                'dst_port': 80,
                'protocol': 'TCP',
                'packet_size': 1500,
                'ttl': 64,
                'payload_entropy': 0.75,
                'flags': 'SYN',
                'packet_id': 'pkt_001'
            },
            {
                'timestamp': '2024-01-15T10:00:02',
                'src_ip': '192.168.1.101',
                'dst_ip': '10.0.0.2',
                'src_port': 12346,
                'dst_port': 443,
                'protocol': 'TCP',
                'packet_size': 1200,
                'ttl': 64,
                'payload_entropy': 0.65,
                'flags': 'ACK',
                'packet_id': 'pkt_002'
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.create_test_csv(f.name, test_data)
            
            try:
                ingestor = CsvIngestor()
                packets = list(ingestor.ingest(f.name))
                
                assert len(packets) == 2
                assert packets[0].src_ip == '192.168.1.100'
                assert packets[0].protocol == ProtocolType.TCP
                assert packets[1].src_ip == '192.168.1.101'
                
                # Check statistics
                stats = ingestor.get_statistics()
                assert stats['packets_processed'] == 2
                assert stats['errors_encountered'] == 0
                
            finally:
                os.unlink(f.name)
    
    def test_csv_missing_columns(self):
        """Test CSV with missing required columns"""
        test_data = [
            {
                'timestamp': '2024-01-15T10:00:01',
                'src_ip': '192.168.1.100',
                # Missing required columns
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.create_test_csv(f.name, test_data)
            
            try:
                ingestor = CsvIngestor()
                with pytest.raises(TrafficIngestionError, match="missing required columns"):
                    ingestor.validate_source(f.name)
                    
            finally:
                os.unlink(f.name)
    
    def test_csv_normalization(self):
        """Test CSV feature normalization"""
        test_data = [
            {
                'timestamp': '2024-01-15T10:00:01',
                'src_ip': '192.168.1.100',
                'dst_ip': '10.0.0.1',
                'src_port': 12345,
                'dst_port': 80,
                'protocol': 'TCP',
                'packet_size': 70000,  # Exceeds normal range
                'ttl': 300,  # Exceeds normal range
                'payload_entropy': 1.5,  # Exceeds normal range
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.create_test_csv(f.name, test_data)
            
            try:
                ingestor = CsvIngestor()
                packets = list(ingestor.ingest(f.name, normalize_features=True))
                
                assert len(packets) == 1
                packet = packets[0]
                
                # Check normalization applied
                assert packet.packet_size <= 65535
                assert packet.ttl <= 255
                assert packet.payload_entropy <= 1.0
                
            finally:
                os.unlink(f.name)
    
    def test_csv_column_mapping(self):
        """Test automatic column mapping"""
        ingestor = CsvIngestor()
        
        # Test column mapping
        mapping = ingestor.get_column_mapping()
        assert 'timestamp' in mapping
        assert 'time' in mapping['timestamp']
        assert 'source_ip' in mapping['src_ip']


class TestTrafficIngestor:
    """Test cases for unified traffic ingestor"""
    
    def test_source_type_detection(self):
        """Test automatic source type detection"""
        ingestor = TrafficIngestor()
        
        # Test file extension detection
        assert ingestor._detect_source_type('test.pcap') == 'pcap'
        assert ingestor._detect_source_type('data.csv') == 'csv'
        assert ingestor._detect_source_type('capture.pcapng') == 'pcap'
        
        # Test socket format detection
        assert ingestor._detect_source_type('localhost:8080') == 'socket'
        assert ingestor._detect_source_type('192.168.1.1:9999') == 'socket'
        
        # Test interface detection (fallback)
        assert ingestor._detect_source_type('eth0') == 'live'
        assert ingestor._detect_source_type('wlan0') == 'live'
    
    def test_supported_sources(self):
        """Test getting supported source information"""
        ingestor = TrafficIngestor()
        sources = ingestor.get_supported_sources()
        
        assert 'pcap' in sources
        assert 'csv' in sources
        assert 'live' in sources
        assert 'socket' in sources
        
        # Check CSV source has required columns info
        assert 'required_columns' in sources['csv']
        assert isinstance(sources['csv']['required_columns'], list)
    
    def test_invalid_source_type(self):
        """Test handling of invalid source types"""
        ingestor = TrafficIngestor()
        
        with pytest.raises(TrafficIngestionError, match="Unsupported source type"):
            list(ingestor.ingest('test.txt', 'invalid_type'))
    
    @patch('backend.ingestion.csv_ingestor.CsvIngestor.ingest')
    def test_csv_ingestion_delegation(self, mock_ingest):
        """Test that CSV ingestion is properly delegated"""
        mock_packet = Mock(spec=TrafficPacket)
        mock_ingest.return_value = iter([mock_packet])
        
        ingestor = TrafficIngestor()
        packets = list(ingestor.ingest('test.csv', 'csv'))
        
        assert len(packets) == 1
        mock_ingest.assert_called_once()


class TestPcapIngestor:
    """Test cases for PCAP ingestion (mocked since scapy may not be available)"""
    
    @patch('backend.ingestion.pcap_ingestor.SCAPY_AVAILABLE', True)
    @patch('backend.ingestion.pcap_ingestor.rdpcap')
    def test_pcap_validation(self, mock_rdpcap):
        """Test PCAP file validation"""
        # Mock successful PCAP reading
        mock_packet = Mock()
        mock_rdpcap.return_value = [mock_packet]
        
        with tempfile.NamedTemporaryFile(suffix='.pcap', delete=False) as f:
            try:
                ingestor = PcapIngestor()
                assert ingestor.validate_source(f.name) is True
                mock_rdpcap.assert_called()
                
            finally:
                os.unlink(f.name)
    
    @patch('backend.ingestion.pcap_ingestor.SCAPY_AVAILABLE', False)
    def test_pcap_scapy_unavailable(self):
        """Test PCAP ingestor when scapy is not available"""
        with pytest.raises(TrafficIngestionError, match="Scapy is not available"):
            PcapIngestor()
    
    def test_pcap_file_not_found(self):
        """Test PCAP validation with non-existent file"""
        if not hasattr(PcapIngestor, '__init__'):
            pytest.skip("Scapy not available")
        
        try:
            ingestor = PcapIngestor()
            with pytest.raises(TrafficIngestionError, match="PCAP file not found"):
                ingestor.validate_source('nonexistent.pcap')
        except TrafficIngestionError as e:
            if "Scapy is not available" in str(e):
                pytest.skip("Scapy not available")
            raise


class TestIngestionFiltering:
    """Test cases for packet filtering during ingestion"""
    
    def create_test_packet(self, **kwargs) -> TrafficPacket:
        """Create a test traffic packet"""
        defaults = {
            'timestamp': datetime.now(),
            'src_ip': '192.168.1.100',
            'dst_ip': '10.0.0.1',
            'src_port': 12345,
            'dst_port': 80,
            'protocol': ProtocolType.TCP,
            'packet_size': 1500,
            'ttl': 64,
            'flags': ['SYN'],
            'payload_entropy': 0.75
        }
        defaults.update(kwargs)
        return TrafficPacket(**defaults)
    
    def test_ip_filtering(self):
        """Test IP address filtering"""
        from backend.ingestion.base_ingestor import BaseTrafficIngestor
        
        ingestor = BaseTrafficIngestor("test")
        packet = self.create_test_packet()
        
        # Test whitelist
        filters = {'src_ip_whitelist': ['192.168.1.100']}
        assert ingestor.apply_filters(packet, filters) is True
        
        filters = {'src_ip_whitelist': ['192.168.1.101']}
        assert ingestor.apply_filters(packet, filters) is False
        
        # Test blacklist
        filters = {'src_ip_blacklist': ['192.168.1.100']}
        assert ingestor.apply_filters(packet, filters) is False
        
        filters = {'src_ip_blacklist': ['192.168.1.101']}
        assert ingestor.apply_filters(packet, filters) is True
    
    def test_protocol_filtering(self):
        """Test protocol filtering"""
        from backend.ingestion.base_ingestor import BaseTrafficIngestor
        
        ingestor = BaseTrafficIngestor("test")
        packet = self.create_test_packet()
        
        # Test protocol filter
        filters = {'protocols': [ProtocolType.TCP]}
        assert ingestor.apply_filters(packet, filters) is True
        
        filters = {'protocols': [ProtocolType.UDP]}
        assert ingestor.apply_filters(packet, filters) is False
    
    def test_packet_size_filtering(self):
        """Test packet size filtering"""
        from backend.ingestion.base_ingestor import BaseTrafficIngestor
        
        ingestor = BaseTrafficIngestor("test")
        packet = self.create_test_packet(packet_size=1000)
        
        # Test size filters
        filters = {'min_packet_size': 500}
        assert ingestor.apply_filters(packet, filters) is True
        
        filters = {'min_packet_size': 1500}
        assert ingestor.apply_filters(packet, filters) is False
        
        filters = {'max_packet_size': 1500}
        assert ingestor.apply_filters(packet, filters) is True
        
        filters = {'max_packet_size': 500}
        assert ingestor.apply_filters(packet, filters) is False


if __name__ == "__main__":
    pytest.main([__file__])