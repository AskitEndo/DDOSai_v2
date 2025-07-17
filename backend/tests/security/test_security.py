"""
Security tests for DDoS.AI platform
"""
import pytest
import requests
import json
import time
import uuid
import random
import string
from datetime import datetime

# Base URL for API tests
BASE_URL = "http://localhost:8000"

@pytest.fixture(scope="module")
def api_health_check():
    """Check if API is available before running tests"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            pytest.skip("API is not available")
    except requests.RequestException:
        pytest.skip("API is not available")

def generate_random_string(length=10):
    """Generate a random string of fixed length"""
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

class TestSecurityVulnerabilities:
    """Test for common security vulnerabilities"""
    
    def test_sql_injection_attempts(self, api_health_check):
        """Test SQL injection prevention"""
        # List of common SQL injection payloads
        sql_payloads = [
            "' OR 1=1 --",
            "'; DROP TABLE users; --",
            "' UNION SELECT * FROM users --",
            "1' OR '1'='1",
            "1; DROP TABLE users",
            "' OR ''='",
            "' OR 1 --",
            "admin'--",
            "1' OR '1'='1'#",
            "' OR '1'='1"
        ]
        
        # Test each payload in different parameters
        for payload in sql_payloads:
            # Test in packet_id
            packet = {
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
                "packet_id": payload
            }
            
            response = requests.post(f"{BASE_URL}/api/analyze", json=packet)
            # Should not return 500 (server error)
            assert response.status_code != 500, f"SQL injection vulnerability with payload: {payload}"
            
            # Test in explanation endpoint
            response = requests.get(f"{BASE_URL}/api/explain/{payload}")
            # Should not return 500 (server error)
            assert response.status_code != 500, f"SQL injection vulnerability with payload: {payload}"
    
    def test_xss_attempts(self, api_health_check):
        """Test XSS prevention"""
        # List of common XSS payloads
        xss_payloads = [
            "<script>alert(1)</script>",
            "<img src=x onerror=alert(1)>",
            "<svg onload=alert(1)>",
            "javascript:alert(1)",
            "\"><script>alert(1)</script>",
            "<body onload=alert(1)>",
            "<img src=\"javascript:alert(1)\">",
            "<iframe src=\"javascript:alert(1)\"></iframe>",
            "<a href=\"javascript:alert(1)\">click me</a>",
            "';alert(1);//"
        ]
        
        # Test each payload in different parameters
        for payload in xss_payloads:
            # Test in packet_id
            packet = {
                "src_ip": "192.168.1.100",
                "dst_ip": "10.0.0.1",
                "src_port": 12345,
                "dst_port": 80,
                "protocol": "TCP",
                "flags": payload,  # XSS in flags
                "packet_size": 64,
                "ttl": 64,
                "payload_entropy": 0.5,
                "timestamp": datetime.now().isoformat(),
                "packet_id": f"test_{uuid.uuid4().hex[:8]}"
            }
            
            response = requests.post(f"{BASE_URL}/api/analyze", json=packet)
            # Should not return 500 (server error)
            assert response.status_code != 500, f"Possible XSS vulnerability with payload: {payload}"
    
    def test_path_traversal_attempts(self, api_health_check):
        """Test path traversal prevention"""
        # List of common path traversal payloads
        traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system.ini",
            "/etc/passwd",
            "C:\\Windows\\system.ini",
            "file:///etc/passwd",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd",
            "..%c0%af..%c0%af..%c0%afetc%c0%afpasswd",
            "/dev/null; cat /etc/passwd"
        ]
        
        # Test each payload in explanation endpoint
        for payload in traversal_payloads:
            response = requests.get(f"{BASE_URL}/api/explain/{payload}")
            # Should not return 200 with sensitive data
            if response.status_code == 200:
                # Check if response contains sensitive data
                assert "root:" not in response.text, f"Path traversal vulnerability with payload: {payload}"
                assert "Windows" not in response.text, f"Path traversal vulnerability with payload: {payload}"
    
    def test_command_injection_attempts(self, api_health_check):
        """Test command injection prevention"""
        # List of common command injection payloads
        cmd_payloads = [
            "& cat /etc/passwd",
            "; cat /etc/passwd",
            "| cat /etc/passwd",
            "`cat /etc/passwd`",
            "$(cat /etc/passwd)",
            "&& cat /etc/passwd",
            "|| cat /etc/passwd",
            "; ping -c 3 127.0.0.1;",
            "& ping -n 3 127.0.0.1 &",
            "| id",
            "; id",
            "& id",
            "&&id",
            "& id &",
            "| id |",
            "; id;",
            "%0Aid",
            "& sleep 5 &"
        ]
        
        # Test each payload in different parameters
        for payload in cmd_payloads:
            # Test in src_ip
            packet = {
                "src_ip": payload,
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
            
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/api/analyze", json=packet)
            response_time = time.time() - start_time
            
            # Should not return 500 (server error)
            assert response.status_code != 500, f"Possible command injection vulnerability with payload: {payload}"
            
            # For sleep payloads, check if response time is suspiciously long
            if "sleep" in payload:
                assert response_time < 3, f"Possible command injection with sleep payload: {payload}"
    
    def test_rate_limiting(self, api_health_check):
        """Test rate limiting functionality"""
        # Send many requests in quick succession
        responses = []
        for _ in range(100):
            packet = {
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
            
            response = requests.post(f"{BASE_URL}/api/analyze", json=packet)
            responses.append(response.status_code)
        
        # Check if any rate limiting responses (429) were received
        assert 429 in responses, "Rate limiting not implemented or not working"
    
    def test_invalid_input_handling(self, api_health_check):
        """Test handling of invalid inputs"""
        # Test with invalid JSON
        response = requests.post(
            f"{BASE_URL}/api/analyze",
            data="This is not JSON",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [400, 422], "Invalid JSON not properly handled"
        
        # Test with missing required fields
        packet = {
            "src_port": 12345,
            "dst_port": 80,
            # Missing src_ip and dst_ip
        }
        response = requests.post(f"{BASE_URL}/api/analyze", json=packet)
        assert response.status_code in [400, 422], "Missing required fields not properly handled"
        
        # Test with invalid field types
        packet = {
            "src_ip": "192.168.1.100",
            "dst_ip": "10.0.0.1",
            "src_port": "not_a_number",  # Should be an integer
            "dst_port": 80,
            "protocol": "TCP",
            "flags": "SYN",
            "packet_size": 64,
            "ttl": 64,
            "payload_entropy": 0.5,
            "timestamp": datetime.now().isoformat(),
            "packet_id": f"test_{uuid.uuid4().hex[:8]}"
        }
        response = requests.post(f"{BASE_URL}/api/analyze", json=packet)
        assert response.status_code in [400, 422], "Invalid field types not properly handled"
    
    def test_cors_headers(self, api_health_check):
        """Test CORS headers"""
        # Send OPTIONS request to check CORS headers
        response = requests.options(f"{BASE_URL}/api/metrics")
        
        # Check if CORS headers are present
        assert "Access-Control-Allow-Origin" in response.headers, "CORS headers not set"
        assert "Access-Control-Allow-Methods" in response.headers, "CORS headers not set"
        assert "Access-Control-Allow-Headers" in response.headers, "CORS headers not set"
    
    def test_http_methods(self, api_health_check):
        """Test HTTP method restrictions"""
        # Test endpoints with incorrect HTTP methods
        response = requests.delete(f"{BASE_URL}/api/analyze")
        assert response.status_code in [404, 405], "DELETE method should not be allowed on /api/analyze"
        
        response = requests.put(f"{BASE_URL}/api/analyze")
        assert response.status_code in [404, 405], "PUT method should not be allowed on /api/analyze"
        
        response = requests.post(f"{BASE_URL}/api/metrics")
        assert response.status_code in [404, 405], "POST method should not be allowed on /api/metrics"

if __name__ == "__main__":
    pytest.main(["-v", __file__])