"""
Simple HTTP Target Server for DDoS Simulation Testing

This server creates a lightweight HTTP service that can be used as a target
for DDoS simulation testing. It includes basic logging and statistics.
"""

import http.server
import socketserver
import threading
import time
import logging
import json
from datetime import datetime
from typing import Dict, Any
import argparse

class DDoSTestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler that tracks requests for DDoS testing"""
    
    request_count = 0
    start_time = datetime.now()
    request_stats = {
        "total_requests": 0,
        "get_requests": 0,
        "post_requests": 0,
        "bytes_received": 0,
        "start_time": None,
        "last_request": None
    }
    
    def do_GET(self):
        """Handle GET requests"""
        DDoSTestHandler.request_count += 1
        DDoSTestHandler.request_stats["total_requests"] += 1
        DDoSTestHandler.request_stats["get_requests"] += 1
        DDoSTestHandler.request_stats["last_request"] = datetime.now().isoformat()
        
        if DDoSTestHandler.request_stats["start_time"] is None:
            DDoSTestHandler.request_stats["start_time"] = datetime.now().isoformat()
        
        # Log request
        client_ip = self.client_address[0]
        logging.info(f"GET request #{DDoSTestHandler.request_count} from {client_ip} - {self.path}")
        
        # Send response
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        # Simple response
        response = f"""
        <html>
        <head><title>DDoS Test Target</title></head>
        <body>
        <h1>DDoS Simulation Target Server</h1>
        <p>Request #{DDoSTestHandler.request_count}</p>
        <p>Time: {datetime.now()}</p>
        <p>Client IP: {client_ip}</p>
        <p>Path: {self.path}</p>
        <hr>
        <h2>Statistics</h2>
        <pre>{json.dumps(DDoSTestHandler.request_stats, indent=2)}</pre>
        </body>
        </html>
        """
        self.wfile.write(response.encode())
    
    def do_POST(self):
        """Handle POST requests"""
        DDoSTestHandler.request_count += 1
        DDoSTestHandler.request_stats["total_requests"] += 1
        DDoSTestHandler.request_stats["post_requests"] += 1
        DDoSTestHandler.request_stats["last_request"] = datetime.now().isoformat()
        
        if DDoSTestHandler.request_stats["start_time"] is None:
            DDoSTestHandler.request_stats["start_time"] = datetime.now().isoformat()
        
        # Read POST data
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > 0:
            post_data = self.rfile.read(content_length)
            DDoSTestHandler.request_stats["bytes_received"] += content_length
        
        # Log request
        client_ip = self.client_address[0]
        logging.info(f"POST request #{DDoSTestHandler.request_count} from {client_ip} - {self.path}")
        
        # Send response
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        response = {
            "status": "success",
            "request_id": DDoSTestHandler.request_count,
            "timestamp": datetime.now().isoformat(),
            "client_ip": client_ip
        }
        self.wfile.write(json.dumps(response).encode())
    
    def log_message(self, format, *args):
        """Override to reduce noise"""
        pass

class DDoSTestServer:
    """DDoS Test Target Server"""
    
    def __init__(self, port=8080):
        self.port = port
        self.server = None
        self.server_thread = None
        self.stats_thread = None
        self.running = False
    
    def start(self):
        """Start the test server"""
        try:
            self.server = socketserver.TCPServer(("", self.port), DDoSTestHandler)
            self.server.allow_reuse_address = True
            
            print(f"🎯 DDoS Test Target Server starting on port {self.port}")
            print(f"📊 Access stats at: http://localhost:{self.port}")
            print(f"🔗 Target URL: http://your-ip:{self.port}")
            print("=" * 50)
            
            # Start server in a separate thread
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            
            # Start statistics display thread
            self.stats_thread = threading.Thread(target=self._display_stats, daemon=True)
            self.stats_thread.start()
            
            self.running = True
            print("✅ Server is running! Press Ctrl+C to stop.")
            
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False
        
        return True
    
    def stop(self):
        """Stop the test server"""
        if self.server:
            print("\n🛑 Stopping server...")
            self.running = False
            self.server.shutdown()
            self.server.server_close()
            print("✅ Server stopped.")
    
    def _display_stats(self):
        """Display live statistics"""
        while self.running:
            try:
                time.sleep(5)  # Update every 5 seconds
                if DDoSTestHandler.request_count > 0:
                    elapsed = (datetime.now() - DDoSTestHandler.start_time).total_seconds()
                    rps = DDoSTestHandler.request_count / elapsed if elapsed > 0 else 0
                    
                    print(f"\n📊 LIVE STATS:")
                    print(f"   Total Requests: {DDoSTestHandler.request_count}")
                    print(f"   Requests/Second: {rps:.2f}")
                    print(f"   Running Time: {elapsed:.1f}s")
                    print(f"   Last Request: {DDoSTestHandler.request_stats.get('last_request', 'None')}")
                    print("-" * 40)
                    
            except Exception as e:
                pass

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='DDoS Simulation Target Server')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on (default: 8080)')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ddos_target_server.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create and start server
    server = DDoSTestServer(args.port)
    
    try:
        if server.start():
            # Keep server running
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        server.stop()

if __name__ == "__main__":
    main()
