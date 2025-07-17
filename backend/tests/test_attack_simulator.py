"""
Unit tests for Attack Simulator
"""
import unittest
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import from the simulation package
from simulation.attack_simulator import AttackSimulator, SimulationStatus
from models.data_models import AttackType, ProtocolType
from core.exceptions import SimulationError


class TestAttackSimulator(unittest.TestCase):
    """Test cases for AttackSimulator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.simulator = AttackSimulator()
        
        # Mock scapy for testing
        self.scapy_patcher = patch('simulation.attack_simulator.scapy')
        self.mock_scapy = self.scapy_patcher.start()
        
        # Mock socket for testing
        self.socket_patcher = patch('simulation.attack_simulator.socket')
        self.mock_socket = self.socket_patcher.start()
        
        # Set up mock socket
        self.mock_sock = MagicMock()
        self.mock_socket.socket.return_value = self.mock_sock
    
    def tearDown(self):
        """Tear down test fixtures"""
        # Stop any running simulations
        if self.simulator.status == SimulationStatus.RUNNING:
            try:
                self.simulator.stop_simulation()
            except:
                pass
        
        # Stop patchers
        self.scapy_patcher.stop()
        self.socket_patcher.stop()
    
    def test_initialization(self):
        """Test simulator initialization"""
        self.assertEqual(self.simulator.status, SimulationStatus.IDLE)
        self.assertIsNone(self.simulator.current_simulation)
        self.assertIsNone(self.simulator.simulation_thread)
        self.assertFalse(self.simulator.stop_event.is_set())
        
        # Check default safety limits
        self.assertEqual(self.simulator.max_packet_rate, 10000)
        self.assertEqual(self.simulator.max_duration, 300)
        self.assertEqual(self.simulator.max_packet_size, 1500)
    
    def test_helper_methods(self):
        """Test helper methods"""
        # Test random IP generation
        ip_range = self.simulator._generate_random_ip_range(10)
        self.assertEqual(len(ip_range), 10)
        for ip in ip_range:
            # Check IP is valid
            self.assertRegex(ip, r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
        
        # Test random URL generation
        urls = self.simulator._generate_random_urls(5)
        self.assertEqual(len(urls), 5)
        for url in urls:
            # Check URL starts with /
            self.assertTrue(url.startswith('/'))


if __name__ == "__main__":
    unittest.main()