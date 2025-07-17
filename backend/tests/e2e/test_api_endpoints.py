"""
End-to-end tests for API endpoints
"""
import pytest
import requests
import json
import time
import uuid
from datetime import datetime

# Base URL for API tests
BASE_URL = "http://localhost:8000"

# Test data
TEST_PACKET = {
    "src_ip": "192.168.1.100",
    "dst_ip": "10.0.0.1",
    "src_port": 12345,
    "dst_port": 80,
    "protocol": "TCP",
    "flags": "SYN",
    "packet_size": 64,
    "ttl": 64,
    "payload_entropy": 0.5,
    "timestamp": datetime.now().isoformat(),
    "packet_id": f"test_{uuid.uuid4().hex[:8]}"
}

@pytest.fixture(scope="module")
def api_health_check():
    """Check if API is available before running tests"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            pytest.skip("API is not available")
    except requests.RequestException:
        pytest.skip("API is not available")

class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_health_endpoint(self, api_health_check):
        """Test health endpoint"""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_detailed_health_endpoint(self, api_health_check):
        """Test detailed health endpoint"""
        response = requests.get(f"{BASE_URL}/health?detailed=true")
        assert response.status_code == 200
        data = response.json()
        assert "components" in data
        assert "system" in data
        assert "timestamp" in data
    
    def test_metrics_endpoint(self, api_health_check):
        """Test metrics endpoint"""
        response = requests.get(f"{BASE_URL}/api/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "packets_processed" in data
        assert "cpu_usage" in data
        assert "memory_usage" in data
    
    def test_detailed_metrics_endpoint(self, api_health_check):
        """Test detailed metrics endpoint"""
        response = requests.get(f"{BASE_URL}/api/metrics?detailed=true")
        assert response.status_code == 200
        data = response.json()
        assert "packets" in data
        assert "processing" in data
        assert "system" in data
        assert "errors" in data
    
    def test_analyze_endpoint(self, api_health_check):
        """Test analyze endpoint"""
        response = requests.post(f"{BASE_URL}/api/analyze", json=TEST_PACKET)
        assert response.status_code == 200
        data = response.json()
        assert "packet_id" in data
        assert "is_malicious" in data
        assert "threat_score" in data
        assert "confidence" in data
        assert "attack_type" in data
        
        # Store prediction ID for explanation test
        prediction_id = data["packet_id"]
        return prediction_id
    
    def test_explanation_endpoint(self, api_health_check):
        """Test explanation endpoint"""
        # First analyze a packet to get a prediction ID
        prediction_id = self.test_analyze_endpoint(api_health_check)
        
        # Wait a moment for the explanation to be generated
        time.sleep(1)
        
        # Get explanation
        response = requests.get(f"{BASE_URL}/api/explain/{prediction_id}")
        assert response.status_code == 200
        data = response.json()
        assert "prediction_id" in data
        assert "explanation" in data
    
    def test_detections_endpoint(self, api_health_check):
        """Test detections endpoint"""
        # First analyze a packet to ensure there's at least one detection
        self.test_analyze_endpoint(api_health_check)
        
        # Get detections
        response = requests.get(f"{BASE_URL}/api/detections")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert "packet_id" in data[0]
        assert "is_malicious" in data[0]
    
    def test_network_graph_endpoint(self, api_health_check):
        """Test network graph endpoint"""
        response = requests.get(f"{BASE_URL}/api/graph/current")
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "edges" in data
        assert "timestamp" in data
    
    def test_simulation_start_endpoint(self, api_health_check):
        """Test simulation start endpoint"""
        simulation_config = {
            "attack_type": "SYN_FLOOD",
            "target_ip": "10.0.0.1",
            "target_port": 80,
            "duration": 5,
            "packet_rate": 10
        }
        response = requests.post(f"{BASE_URL}/api/simulate/start", json=simulation_config)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "started"
        assert "simulation_id" in data
        
        # Return simulation ID for stop test
        return data["simulation_id"]
    
    def test_simulation_stop_endpoint(self, api_health_check):
        """Test simulation stop endpoint"""
        # First start a simulation to get a simulation ID
        simulation_id = self.test_simulation_start_endpoint(api_health_check)
        
        # Stop simulation
        response = requests.post(f"{BASE_URL}/api/simulate/stop", json={"simulation_id": simulation_id})
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "stopped"
        assert "simulation_id" in data
        assert data["simulation_id"] == simulation_id

if __name__ == "__main__":
    pytest.main(["-v", __file__])