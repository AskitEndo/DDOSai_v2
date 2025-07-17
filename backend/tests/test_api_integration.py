"""
Integration tests for FastAPI backend service
"""
import pytest
import asyncio
import json
from datetime import datetime
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from models.data_models import TrafficPacket, ProtocolType, AttackType

class TestAPIIntegration:
    """Integration tests for the FastAPI backend"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "DDoS.AI API is running"
        assert data["version"] == "1.0.0"
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_analyze_traffic_endpoint(self, client):
        """Test traffic analysis endpoint"""
        # Create sample packet data
        packet_data = {
            "timestamp": datetime.now().isoformat(),
            "src_ip": "192.168.1.100",
            "dst_ip": "10.0.0.1",
            "src_port": 12345,
            "dst_port": 80,
            "protocol": "TCP",
            "packet_size": 1024,
            "ttl": 64,
            "flags": ["SYN"],
            "payload_entropy": 0.7,
            "packet_id": "test_packet_001"
        }
        
        response = client.post("/api/analyze", json=packet_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "packet_id" in data
        assert "is_malicious" in data
        assert "threat_score" in data
        assert "detection_method" in data
        assert "confidence" in data
        assert "explanation" in data
        assert data["packet_id"] == "test_packet_001"
    
    def test_analyze_traffic_with_minimal_data(self, client):
        """Test traffic analysis with minimal packet data"""
        packet_data = {
            "src_ip": "203.0.113.45",
            "dst_ip": "192.168.1.1"
        }
        
        response = client.post("/api/analyze", json=packet_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "is_malicious" in data
        assert "threat_score" in data
        assert isinstance(data["threat_score"], int)
        assert 0 <= data["threat_score"] <= 100
    
    def test_get_detections_endpoint(self, client):
        """Test get detections endpoint"""
        # First analyze some packets to populate detection history
        for i in range(5):
            packet_data = {
                "src_ip": f"192.168.1.{100 + i}",
                "dst_ip": "10.0.0.1",
                "packet_id": f"test_packet_{i:03d}"
            }
            client.post("/api/analyze", json=packet_data)
        
        # Get detections
        response = client.get("/api/detections")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 5
        
        # Check detection structure
        if data:
            detection = data[0]
            assert "timestamp" in detection
            assert "packet_id" in detection
            assert "is_malicious" in detection
            assert "threat_score" in detection
    
    def test_get_detections_with_limit(self, client):
        """Test get detections with limit parameter"""
        response = client.get("/api/detections?limit=10")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 10
    
    def test_get_network_graph(self, client):
        """Test network graph endpoint"""
        response = client.get("/api/graph/current")
        assert response.status_code == 200
        
        data = response.json()
        assert "nodes" in data
        assert "edges" in data
        assert "timestamp" in data
        assert isinstance(data["nodes"], list)
        assert isinstance(data["edges"], list)
    
    def test_get_system_metrics(self, client):
        """Test system metrics endpoint"""
        response = client.get("/api/metrics")
        assert response.status_code == 200
        
        data = response.json()
        required_fields = [
            "timestamp", "packets_processed", "processing_latency_ms",
            "cpu_usage", "memory_usage", "active_connections", "threat_level"
        ]
        
        for field in required_fields:
            assert field in data
        
        # Check data types and ranges
        assert isinstance(data["packets_processed"], int)
        assert isinstance(data["processing_latency_ms"], int)
        assert isinstance(data["cpu_usage"], (int, float))
        assert isinstance(data["memory_usage"], (int, float))
        assert isinstance(data["threat_level"], int)
        assert 0 <= data["threat_level"] <= 5
    
    def test_start_simulation(self, client):
        """Test start simulation endpoint"""
        config = {
            "attack_type": "syn_flood",
            "target_ip": "192.168.1.1",
            "duration": 60,
            "packet_rate": 1000
        }
        
        response = client.post("/api/simulate/start", json=config)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "started"
        assert "simulation_id" in data
        assert "message" in data
        assert "timestamp" in data
        assert "syn_flood" in data["message"]
    
    def test_stop_simulation(self, client):
        """Test stop simulation endpoint"""
        simulation_id = "sim_1234"
        
        response = client.post(f"/api/simulate/stop?simulation_id={simulation_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "stopped"
        assert data["simulation_id"] == simulation_id
        assert "message" in data
        assert "timestamp" in data
    
    def test_get_explanation(self, client):
        """Test XAI explanation endpoint"""
        # First analyze a packet to create a prediction in the cache
        packet_data = {
            "src_ip": "192.168.1.100",
            "dst_ip": "10.0.0.1",
            "packet_id": "test_explanation_001"
        }
        client.post("/api/analyze", json=packet_data)
        
        # Now get the explanation for this prediction
        prediction_id = "test_explanation_001"
        response = client.get(f"/api/explain/{prediction_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["prediction_id"] == prediction_id
        assert "prediction" in data
        assert "confidence" in data
        assert "feature_importance" in data
        
        # Check feature importance structure
        feature_importance = data["feature_importance"]
        assert isinstance(feature_importance, list)
        if feature_importance:
            feature = feature_importance[0]
            assert "feature_name" in feature
            assert "importance_score" in feature
            assert "description" in feature
    
    def test_websocket_live_feed(self, client):
        """Test WebSocket live feed endpoint"""
        with client.websocket_connect("/ws/live-feed") as websocket:
            # Send test message
            websocket.send_text("test message")
            
            # Receive echo
            data = websocket.receive_text()
            assert "Echo: test message" in data
    
    def test_websocket_alerts(self, client):
        """Test WebSocket alerts endpoint"""
        with client.websocket_connect("/ws/alerts") as websocket:
            # Send subscription message
            websocket.send_text("subscribe:high_threat")
            
            # Receive confirmation
            data = websocket.receive_text()
            assert "Alert subscription" in data
    
    def test_error_handling_invalid_packet_data(self, client):
        """Test error handling with invalid packet data"""
        invalid_data = {
            "src_ip": "invalid_ip",
            "dst_ip": "also_invalid",
            "src_port": "not_a_number"
        }
        
        response = client.post("/api/analyze", json=invalid_data)
        # Should handle gracefully and return some result or error
        assert response.status_code in [200, 422, 500]
    
    def test_concurrent_analysis_requests(self, client):
        """Test handling of concurrent analysis requests"""
        import concurrent.futures
        import threading
        
        def analyze_packet(packet_id):
            packet_data = {
                "src_ip": "192.168.1.100",
                "dst_ip": "10.0.0.1",
                "packet_id": f"concurrent_test_{packet_id}"
            }
            response = client.post("/api/analyze", json=packet_data)
            return response.status_code == 200
        
        # Send 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(analyze_packet, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        assert all(results)
    
    def test_detection_history_persistence(self, client):
        """Test that detection history persists across requests"""
        # Analyze a packet with unique ID
        unique_id = f"history_test_{datetime.now().timestamp()}"
        packet_data = {
            "src_ip": "192.168.1.200",
            "dst_ip": "10.0.0.2",
            "packet_id": unique_id
        }
        
        # Analyze packet
        response = client.post("/api/analyze", json=packet_data)
        assert response.status_code == 200
        
        # Check if it appears in detection history
        response = client.get("/api/detections")
        assert response.status_code == 200
        
        detections = response.json()
        packet_ids = [d["packet_id"] for d in detections]
        assert unique_id in packet_ids
    
    def test_metrics_update_after_analysis(self, client):
        """Test that metrics update after packet analysis"""
        # Get initial metrics
        response = client.get("/api/metrics")
        initial_metrics = response.json()
        initial_count = initial_metrics["packets_processed"]
        
        # Analyze a packet
        packet_data = {
            "src_ip": "192.168.1.150",
            "dst_ip": "10.0.0.3",
            "packet_id": "metrics_test_001"
        }
        client.post("/api/analyze", json=packet_data)
        
        # Get updated metrics
        response = client.get("/api/metrics")
        updated_metrics = response.json()
        updated_count = updated_metrics["packets_processed"]
        
        # Count should have increased
        assert updated_count > initial_count
    
    def test_api_response_times(self, client):
        """Test API response times meet requirements"""
        import time
        
        # Test analyze endpoint response time
        packet_data = {
            "src_ip": "192.168.1.100",
            "dst_ip": "10.0.0.1",
            "packet_id": "performance_test"
        }
        
        start_time = time.time()
        response = client.post("/api/analyze", json=packet_data)
        end_time = time.time()
        
        assert response.status_code == 200
        response_time_ms = (end_time - start_time) * 1000
        
        # Should process within 200ms as per requirements
        # Allow some tolerance for test environment
        assert response_time_ms < 500  # 500ms tolerance for testing
    
    def test_explanation_endpoint_response_time(self, client):
        """Test explanation endpoint response time"""
        import time
        
        # First analyze a packet to create a prediction in the cache
        packet_data = {
            "src_ip": "192.168.1.100",
            "dst_ip": "10.0.0.1",
            "packet_id": "performance_test"
        }
        client.post("/api/analyze", json=packet_data)
        
        # Now test the explanation endpoint response time
        start_time = time.time()
        response = client.get("/api/explain/performance_test")
        end_time = time.time()
        
        assert response.status_code == 200
        response_time_ms = (end_time - start_time) * 1000
        
        # Should respond within 500ms as per requirements
        # Allow more tolerance for test environment
        assert response_time_ms < 2000  # 2000ms tolerance for testing

class TestAPIErrorHandling:
    """Test error handling scenarios"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_missing_required_fields(self, client):
        """Test handling of missing required fields"""
        # Empty request
        response = client.post("/api/analyze", json={})
        # Should handle gracefully
        assert response.status_code in [200, 422]
    
    def test_invalid_json_data(self, client):
        """Test handling of invalid JSON data"""
        response = client.post(
            "/api/analyze",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_nonexistent_endpoints(self, client):
        """Test handling of nonexistent endpoints"""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404
        
        response = client.post("/api/nonexistent")
        assert response.status_code == 404
    
    def test_invalid_http_methods(self, client):
        """Test invalid HTTP methods on endpoints"""
        # GET on POST endpoint
        response = client.get("/api/analyze")
        assert response.status_code == 405
        
        # POST on GET endpoint
        response = client.post("/api/detections")
        assert response.status_code == 405

if __name__ == "__main__":
    pytest.main([__file__, "-v"])