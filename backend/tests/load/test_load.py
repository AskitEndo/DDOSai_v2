"""
Load testing for DDoS.AI platform
"""
import pytest
import asyncio
import aiohttp
import time
import uuid
import json
import statistics
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any

# Base URL for API tests
BASE_URL = "http://localhost:8000"

# Test configuration
DEFAULT_DURATION = 30  # seconds
DEFAULT_RATE = 10  # requests per second
DEFAULT_WORKERS = 5
DEFAULT_RAMP_UP = 5  # seconds

# Test packet template
def generate_test_packet():
    """Generate a test packet with unique ID"""
    return {
        "src_ip": f"192.168.1.{np.random.randint(1, 255)}",
        "dst_ip": f"10.0.0.{np.random.randint(1, 10)}",
        "src_port": np.random.randint(1024, 65535),
        "dst_port": 80,
        "protocol": "TCP",
        "flags": "SYN",
        "packet_size": 64,
        "ttl": 64,
        "payload_entropy": 0.5,
        "timestamp": datetime.now().isoformat(),
        "packet_id": f"load_{uuid.uuid4().hex[:8]}"
    }

class LoadTestResults:
    """Class to store and analyze load test results"""
    
    def __init__(self):
        self.requests = 0
        self.success = 0
        self.errors = 0
        self.latencies = []
        self.start_time = None
        self.end_time = None
        self.error_types = {}
    
    def start(self):
        """Start the test timer"""
        self.start_time = time.time()
    
    def stop(self):
        """Stop the test timer"""
        self.end_time = time.time()
    
    def record_success(self, latency):
        """Record a successful request"""
        self.requests += 1
        self.success += 1
        self.latencies.append(latency)
    
    def record_error(self, error_type):
        """Record a failed request"""
        self.requests += 1
        self.errors += 1
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
    
    def get_summary(self):
        """Get test summary statistics"""
        if not self.start_time or not self.end_time:
            return {"error": "Test not completed"}
        
        duration = self.end_time - self.start_time
        
        if not self.latencies:
            return {
                "duration": duration,
                "requests": self.requests,
                "success": self.success,
                "errors": self.errors,
                "error_rate": 100.0 if self.requests > 0 else 0.0,
                "requests_per_second": 0.0,
                "error_types": self.error_types
            }
        
        latencies_ms = [l * 1000 for l in self.latencies]
        
        return {
            "duration": duration,
            "requests": self.requests,
            "success": self.success,
            "errors": self.errors,
            "error_rate": (self.errors / self.requests * 100) if self.requests > 0 else 0.0,
            "requests_per_second": self.requests / duration if duration > 0 else 0,
            "latency": {
                "min_ms": min(latencies_ms),
                "max_ms": max(latencies_ms),
                "mean_ms": statistics.mean(latencies_ms),
                "median_ms": statistics.median(latencies_ms),
                "p95_ms": np.percentile(latencies_ms, 95),
                "p99_ms": np.percentile(latencies_ms, 99),
                "stddev_ms": statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0
            },
            "error_types": self.error_types
        }
    
    def plot_results(self, output_file=None):
        """Generate plots of test results"""
        if not self.latencies:
            print("No data to plot")
            return
        
        latencies_ms = [l * 1000 for l in self.latencies]
        
        plt.figure(figsize=(12, 10))
        
        # Latency histogram
        plt.subplot(2, 2, 1)
        plt.hist(latencies_ms, bins=30, alpha=0.7, color='blue')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Count')
        plt.title('Response Latency Distribution')
        plt.grid(True, alpha=0.3)
        
        # Latency over time
        plt.subplot(2, 2, 2)
        plt.plot(range(len(latencies_ms)), latencies_ms, 'b-', alpha=0.5)
        plt.xlabel('Request Number')
        plt.ylabel('Latency (ms)')
        plt.title('Latency Over Time')
        plt.grid(True, alpha=0.3)
        
        # Error pie chart
        plt.subplot(2, 2, 3)
        if self.errors > 0:
            labels = list(self.error_types.keys())
            sizes = list(self.error_types.values())
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Error Types')
        else:
            plt.text(0.5, 0.5, 'No Errors', horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            plt.title('Error Types (None)')
        
        # Summary statistics
        plt.subplot(2, 2, 4)
        plt.axis('off')
        summary = self.get_summary()
        summary_text = (
            f"Load Test Summary\n\n"
            f"Duration: {summary['duration']:.2f} seconds\n"
            f"Total Requests: {summary['requests']}\n"
            f"Successful: {summary['success']}\n"
            f"Failed: {summary['errors']}\n"
            f"Error Rate: {summary['error_rate']:.2f}%\n"
            f"Requests/sec: {summary['requests_per_second']:.2f}\n\n"
            f"Latency (ms):\n"
            f"  Min: {summary['latency']['min_ms']:.2f}\n"
            f"  Max: {summary['latency']['max_ms']:.2f}\n"
            f"  Mean: {summary['latency']['mean_ms']:.2f}\n"
            f"  Median: {summary['latency']['median_ms']:.2f}\n"
            f"  P95: {summary['latency']['p95_ms']:.2f}\n"
            f"  P99: {summary['latency']['p99_ms']:.2f}\n"
        )
        plt.text(0.1, 0.5, summary_text, fontsize=10, va='center')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            print(f"Results saved to {output_file}")
        else:
            plt.show()

async def worker(session, results, rate_limiter, endpoint="/api/analyze"):
    """Worker to send requests at the specified rate"""
    while True:
        try:
            # Wait for rate limiter to allow a request
            await rate_limiter.get()
            
            # Generate test data
            if endpoint == "/api/analyze":
                data = generate_test_packet()
            elif endpoint == "/api/metrics":
                data = None
            else:
                data = {}
            
            start_time = time.time()
            
            try:
                if data is None:
                    # GET request
                    async with session.get(f"{BASE_URL}{endpoint}", timeout=5) as response:
                        await response.json()
                        if response.status == 200:
                            results.record_success(time.time() - start_time)
                        else:
                            results.record_error(f"HTTP {response.status}")
                else:
                    # POST request
                    async with session.post(f"{BASE_URL}{endpoint}", json=data, timeout=5) as response:
                        await response.json()
                        if response.status == 200:
                            results.record_success(time.time() - start_time)
                        else:
                            results.record_error(f"HTTP {response.status}")
            except asyncio.TimeoutError:
                results.record_error("Timeout")
            except aiohttp.ClientError as e:
                results.record_error(f"ClientError: {type(e).__name__}")
            except Exception as e:
                results.record_error(f"Error: {type(e).__name__}")
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Worker error: {e}")

class RateLimiter:
    """Rate limiter for controlling request rate"""
    
    def __init__(self, rate, ramp_up_time=0):
        self.rate = rate
        self.interval = 1.0 / rate if rate > 0 else float('inf')
        self.last_check = time.time()
        self.queue = asyncio.Queue()
        self.ramp_up_time = ramp_up_time
        self.start_time = time.time()
    
    async def _add_to_queue(self):
        """Add items to the queue at the specified rate"""
        current_rate = self.rate
        
        while True:
            now = time.time()
            
            # Calculate current rate during ramp-up
            if self.ramp_up_time > 0:
                elapsed = now - self.start_time
                if elapsed < self.ramp_up_time:
                    # Linear ramp-up from 10% to 100% of target rate
                    current_rate = self.rate * (0.1 + 0.9 * elapsed / self.ramp_up_time)
                else:
                    current_rate = self.rate
            
            # Calculate wait time to maintain rate
            current_interval = 1.0 / current_rate if current_rate > 0 else float('inf')
            wait_time = self.last_check + current_interval - now
            
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            # Add item to queue
            await self.queue.put(True)
            self.last_check = time.time()
    
    async def start(self):
        """Start the rate limiter"""
        self.start_time = time.time()
        self.task = asyncio.create_task(self._add_to_queue())
    
    async def stop(self):
        """Stop the rate limiter"""
        if hasattr(self, 'task'):
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
    
    async def get(self):
        """Get an item from the queue (blocks until rate limit allows)"""
        return await self.queue.get()

async def run_load_test(
    duration=DEFAULT_DURATION,
    rate=DEFAULT_RATE,
    workers=DEFAULT_WORKERS,
    ramp_up=DEFAULT_RAMP_UP,
    endpoint="/api/analyze"
):
    """Run a load test against the API"""
    results = LoadTestResults()
    
    # Create rate limiter
    rate_limiter = RateLimiter(rate, ramp_up)
    await rate_limiter.start()
    
    # Create HTTP session
    async with aiohttp.ClientSession() as session:
        # Start worker tasks
        worker_tasks = []
        for _ in range(workers):
            task = asyncio.create_task(worker(session, results, rate_limiter, endpoint))
            worker_tasks.append(task)
        
        # Start recording results
        results.start()
        
        # Run for specified duration
        await asyncio.sleep(duration)
        
        # Stop recording results
        results.stop()
        
        # Stop rate limiter
        await rate_limiter.stop()
        
        # Cancel worker tasks
        for task in worker_tasks:
            task.cancel()
        
        # Wait for all tasks to complete
        await asyncio.gather(*worker_tasks, return_exceptions=True)
    
    return results

@pytest.mark.asyncio
async def test_analyze_endpoint_load():
    """Load test for analyze endpoint"""
    print("\nRunning load test for /api/analyze endpoint...")
    results = await run_load_test(
        duration=10,  # Short duration for CI/CD
        rate=20,
        workers=5,
        endpoint="/api/analyze"
    )
    
    summary = results.get_summary()
    print(f"Load test completed: {summary['requests']} requests, "
          f"{summary['error_rate']:.2f}% errors, "
          f"{summary['requests_per_second']:.2f} req/s, "
          f"avg latency: {summary['latency']['mean_ms']:.2f}ms")
    
    # Basic assertions for CI/CD
    assert summary['error_rate'] < 10.0, "Error rate too high"
    assert summary['requests_per_second'] > 5.0, "Throughput too low"
    assert summary['latency']['p95_ms'] < 500.0, "P95 latency too high"
    
    # Generate plot for manual review
    results.plot_results("load_test_analyze.png")

@pytest.mark.asyncio
async def test_metrics_endpoint_load():
    """Load test for metrics endpoint"""
    print("\nRunning load test for /api/metrics endpoint...")
    results = await run_load_test(
        duration=10,  # Short duration for CI/CD
        rate=10,
        workers=3,
        endpoint="/api/metrics"
    )
    
    summary = results.get_summary()
    print(f"Load test completed: {summary['requests']} requests, "
          f"{summary['error_rate']:.2f}% errors, "
          f"{summary['requests_per_second']:.2f} req/s, "
          f"avg latency: {summary['latency']['mean_ms']:.2f}ms")
    
    # Basic assertions for CI/CD
    assert summary['error_rate'] < 5.0, "Error rate too high"
    assert summary['latency']['p95_ms'] < 200.0, "P95 latency too high"
    
    # Generate plot for manual review
    results.plot_results("load_test_metrics.png")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        # Run full load test (not for CI/CD)
        asyncio.run(run_load_test(
            duration=60,
            rate=50,
            workers=10,
            ramp_up=10,
            endpoint="/api/analyze"
        ))
    else:
        # Run with pytest
        pytest.main(["-v", __file__])