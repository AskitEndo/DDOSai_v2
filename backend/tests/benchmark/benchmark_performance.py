#!/usr/bin/env python3
"""
Performance benchmarking for DDoS.AI platform
"""
import argparse
import asyncio
import aiohttp
import time
import json
import uuid
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import test data generator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generate_test_data import generate_test_packet, ATTACK_TYPES, NORMAL_TRAFFIC

# Base URL for API tests
BASE_URL = "http://localhost:8000"

class BenchmarkResults:
    """Class to store and analyze benchmark results"""
    
    def __init__(self, name):
        self.name = name
        self.requests = 0
        self.success = 0
        self.errors = 0
        self.latencies = []
        self.start_time = None
        self.end_time = None
        self.error_types = {}
        self.throughput_samples = []  # (timestamp, throughput) tuples
        self.latency_samples = []     # (timestamp, latency) tuples
        self.cpu_samples = []         # (timestamp, cpu_usage) tuples
        self.memory_samples = []      # (timestamp, memory_usage) tuples
    
    def start(self):
        """Start the benchmark timer"""
        self.start_time = time.time()
    
    def stop(self):
        """Stop the benchmark timer"""
        self.end_time = time.time()
    
    def record_success(self, latency):
        """Record a successful request"""
        self.requests += 1
        self.success += 1
        self.latencies.append(latency)
        self.latency_samples.append((time.time(), latency))
    
    def record_error(self, error_type):
        """Record a failed request"""
        self.requests += 1
        self.errors += 1
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
    
    def record_throughput(self, timestamp, throughput):
        """Record throughput sample"""
        self.throughput_samples.append((timestamp, throughput))
    
    def record_system_metrics(self, timestamp, cpu_usage, memory_usage):
        """Record system metrics"""
        self.cpu_samples.append((timestamp, cpu_usage))
        self.memory_samples.append((timestamp, memory_usage))
    
    def get_summary(self):
        """Get benchmark summary statistics"""
        if not self.start_time or not self.end_time:
            return {"error": "Benchmark not completed"}
        
        duration = self.end_time - self.start_time
        
        if not self.latencies:
            return {
                "name": self.name,
                "duration": duration,
                "requests": self.requests,
                "success": self.success,
                "errors": self.errors,
                "error_rate": 100.0 if self.requests > 0 else 0.0,
                "requests_per_second": 0.0,
                "error_types": self.error_types
            }
        
        latencies_ms = [l * 1000 for l in self.latencies]
        
        # Calculate average CPU and memory usage
        avg_cpu = np.mean([cpu for _, cpu in self.cpu_samples]) if self.cpu_samples else 0
        avg_memory = np.mean([mem for _, mem in self.memory_samples]) if self.memory_samples else 0
        
        return {
            "name": self.name,
            "duration": duration,
            "requests": self.requests,
            "success": self.success,
            "errors": self.errors,
            "error_rate": (self.errors / self.requests * 100) if self.requests > 0 else 0.0,
            "requests_per_second": self.requests / duration if duration > 0 else 0,
            "latency": {
                "min_ms": min(latencies_ms),
                "max_ms": max(latencies_ms),
                "mean_ms": np.mean(latencies_ms),
                "median_ms": np.median(latencies_ms),
                "p95_ms": np.percentile(latencies_ms, 95),
                "p99_ms": np.percentile(latencies_ms, 99),
                "stddev_ms": np.std(latencies_ms)
            },
            "system": {
                "avg_cpu_usage": avg_cpu,
                "avg_memory_usage": avg_memory
            },
            "error_types": self.error_types
        }
    
    def plot_results(self, output_dir=None):
        """Generate plots of benchmark results"""
        if not self.latencies:
            print("No data to plot")
            return
        
        latencies_ms = [l * 1000 for l in self.latencies]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        
        # Latency histogram
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.hist(latencies_ms, bins=30, alpha=0.7, color='blue')
        ax1.set_xlabel('Latency (ms)')
        ax1.set_ylabel('Count')
        ax1.set_title('Response Latency Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Latency over time
        ax2 = fig.add_subplot(2, 2, 2)
        if self.latency_samples:
            times = [(t - self.start_time) for t, _ in self.latency_samples]
            latencies = [l * 1000 for _, l in self.latency_samples]
            ax2.plot(times, latencies, 'b-', alpha=0.5)
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Latency (ms)')
            ax2.set_title('Latency Over Time')
            ax2.grid(True, alpha=0.3)
        
        # Throughput over time
        ax3 = fig.add_subplot(2, 2, 3)
        if self.throughput_samples:
            times = [(t - self.start_time) for t, _ in self.throughput_samples]
            throughputs = [tp for _, tp in self.throughput_samples]
            ax3.plot(times, throughputs, 'g-')
            ax3.set_xlabel('Time (seconds)')
            ax3.set_ylabel('Throughput (req/s)')
            ax3.set_title('Throughput Over Time')
            ax3.grid(True, alpha=0.3)
        
        # CPU and Memory usage
        ax4 = fig.add_subplot(2, 2, 4)
        if self.cpu_samples and self.memory_samples:
            cpu_times = [(t - self.start_time) for t, _ in self.cpu_samples]
            cpu_values = [cpu for _, cpu in self.cpu_samples]
            memory_times = [(t - self.start_time) for t, _ in self.memory_samples]
            memory_values = [mem for _, mem in self.memory_samples]
            
            ax4.plot(cpu_times, cpu_values, 'r-', label='CPU Usage (%)')
            ax4.plot(memory_times, memory_values, 'b-', label='Memory Usage (%)')
            ax4.set_xlabel('Time (seconds)')
            ax4.set_ylabel('Usage (%)')
            ax4.set_title('System Resource Usage')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Add summary text
        summary = self.get_summary()
        summary_text = (
            f"Benchmark: {self.name}\n\n"
            f"Duration: {summary['duration']:.2f} seconds\n"
            f"Total Requests: {summary['requests']}\n"
            f"Successful: {summary['success']}\n"
            f"Failed: {summary['errors']}\n"
            f"Error Rate: {summary['error_rate']:.2f}%\n"
            f"Throughput: {summary['requests_per_second']:.2f} req/s\n\n"
            f"Latency (ms):\n"
            f"  Min: {summary['latency']['min_ms']:.2f}\n"
            f"  Max: {summary['latency']['max_ms']:.2f}\n"
            f"  Mean: {summary['latency']['mean_ms']:.2f}\n"
            f"  Median: {summary['latency']['median_ms']:.2f}\n"
            f"  P95: {summary['latency']['p95_ms']:.2f}\n"
            f"  P99: {summary['latency']['p99_ms']:.2f}\n\n"
            f"System:\n"
            f"  Avg CPU: {summary['system']['avg_cpu_usage']:.2f}%\n"
            f"  Avg Memory: {summary['system']['avg_memory_usage']:.2f}%\n"
        )
        
        # Add text to figure
        plt.figtext(0.5, 0.01, summary_text, ha='center', fontsize=10, 
                   bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.suptitle(f"DDoS.AI Performance Benchmark: {self.name}", fontsize=16)
        
        # Save to file if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"benchmark_{self.name.replace(' ', '_').lower()}.png")
            plt.savefig(output_file)
            print(f"Results saved to {output_file}")
            
            # Also save raw data
            data_file = os.path.join(output_dir, f"benchmark_{self.name.replace(' ', '_').lower()}.json")
            with open(data_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Raw data saved to {data_file}")
        else:
            plt.show()

async def worker(session, results, queue, endpoint="/api/analyze"):
    """Worker to send requests"""
    while True:
        try:
            # Get next packet from queue
            packet = await queue.get()
            
            if packet is None:
                # End signal
                queue.task_done()
                break
            
            start_time = time.time()
            
            try:
                # Send request
                async with session.post(f"{BASE_URL}{endpoint}", json=packet, timeout=5) as response:
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
            
            # Mark task as done
            queue.task_done()
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Worker error: {e}")
            if queue.qsize() > 0:
                queue.task_done()

async def metrics_collector(results, interval=1.0):
    """Collect system metrics periodically"""
    import psutil
    
    while True:
        try:
            # Get system metrics
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_usage = psutil.virtual_memory().percent
            
            # Record metrics
            timestamp = time.time()
            results.record_system_metrics(timestamp, cpu_usage, memory_usage)
            
            # Also get metrics from API
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{BASE_URL}/api/metrics", timeout=2) as response:
                        if response.status == 200:
                            metrics = await response.json()
                            # Record throughput if available
                            if "packets_processed" in metrics:
                                results.record_throughput(timestamp, metrics.get("processing_latency_ms", 0))
            except Exception:
                pass
            
            # Wait for next interval
            await asyncio.sleep(interval)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Metrics collector error: {e}")
            await asyncio.sleep(interval)

async def packet_generator(queue, packet_rate, duration, attack_start=None, attack_duration=None, attack_rate=None, attack_type="syn_flood"):
    """Generate packets at the specified rate"""
    start_time = time.time()
    end_time = start_time + duration
    
    # Calculate attack times if specified
    if attack_start is not None and attack_duration is not None:
        attack_start_time = start_time + attack_start
        attack_end_time = attack_start_time + attack_duration
    else:
        attack_start_time = None
        attack_end_time = None
    
    # Generate packets
    packet_count = 0
    attack_count = 0
    
    while time.time() < end_time:
        current_time = time.time()
        
        # Determine if we're in attack phase
        in_attack_phase = (attack_start_time is not None and 
                          attack_start_time <= current_time < attack_end_time)
        
        # Calculate current rate
        current_rate = attack_rate if in_attack_phase else packet_rate
        
        # Calculate sleep time to maintain rate
        sleep_time = 1.0 / current_rate
        
        # Generate packet
        if in_attack_phase:
            # Generate attack packet
            packet = generate_test_packet(
                timestamp=datetime.now(),
                is_attack=True,
                attack_type=attack_type
            )
            attack_count += 1
        else:
            # Generate normal packet
            packet = generate_test_packet(
                timestamp=datetime.now(),
                is_attack=False
            )
        
        # Add to queue
        await queue.put(packet)
        packet_count += 1
        
        # Sleep to maintain rate
        await asyncio.sleep(sleep_time)
    
    print(f"Generated {packet_count} packets ({attack_count} attack packets)")

async def run_benchmark(
    name,
    duration=60,
    packet_rate=10,
    workers=5,
    attack_start=None,
    attack_duration=None,
    attack_rate=None,
    attack_type="syn_flood"
):
    """Run a benchmark"""
    results = BenchmarkResults(name)
    
    # Create packet queue
    queue = asyncio.Queue()
    
    # Start metrics collector
    metrics_task = asyncio.create_task(metrics_collector(results))
    
    # Start packet generator
    generator_task = asyncio.create_task(
        packet_generator(
            queue,
            packet_rate,
            duration,
            attack_start,
            attack_duration,
            attack_rate,
            attack_type
        )
    )
    
    # Create HTTP session
    async with aiohttp.ClientSession() as session:
        # Start worker tasks
        worker_tasks = []
        for _ in range(workers):
            task = asyncio.create_task(worker(session, results, queue))
            worker_tasks.append(task)
        
        # Start recording results
        results.start()
        
        # Wait for generator to complete
        await generator_task
        
        # Wait for all packets to be processed
        await queue.join()
        
        # Stop recording results
        results.stop()
        
        # Cancel metrics collector
        metrics_task.cancel()
        try:
            await metrics_task
        except asyncio.CancelledError:
            pass
        
        # Send end signal to workers
        for _ in range(workers):
            await queue.put(None)
        
        # Wait for workers to complete
        await asyncio.gather(*worker_tasks)
    
    return results

async def run_benchmarks(output_dir=None):
    """Run a series of benchmarks"""
    benchmarks = [
        # Baseline benchmark
        {
            "name": "Baseline (10 req/s)",
            "duration": 30,
            "packet_rate": 10,
            "workers": 5
        },
        # Medium load benchmark
        {
            "name": "Medium Load (50 req/s)",
            "duration": 30,
            "packet_rate": 50,
            "workers": 10
        },
        # High load benchmark
        {
            "name": "High Load (100 req/s)",
            "duration": 30,
            "packet_rate": 100,
            "workers": 20
        },
        # Attack simulation benchmark
        {
            "name": "SYN Flood Attack",
            "duration": 60,
            "packet_rate": 10,
            "workers": 10,
            "attack_start": 20,
            "attack_duration": 20,
            "attack_rate": 100,
            "attack_type": "syn_flood"
        },
        # HTTP Flood Attack benchmark
        {
            "name": "HTTP Flood Attack",
            "duration": 60,
            "packet_rate": 10,
            "workers": 10,
            "attack_start": 20,
            "attack_duration": 20,
            "attack_rate": 100,
            "attack_type": "http_flood"
        }
    ]
    
    results = []
    
    for benchmark in benchmarks:
        print(f"\nRunning benchmark: {benchmark['name']}")
        benchmark_result = await run_benchmark(**benchmark)
        results.append(benchmark_result)
        
        # Plot results
        benchmark_result.plot_results(output_dir)
        
        # Print summary
        summary = benchmark_result.get_summary()
        print(f"Benchmark completed: {summary['requests']} requests, "
              f"{summary['error_rate']:.2f}% errors, "
              f"{summary['requests_per_second']:.2f} req/s, "
              f"avg latency: {summary['latency']['mean_ms']:.2f}ms")
    
    # Generate comparison report
    if output_dir:
        generate_comparison_report(results, output_dir)
    
    return results

def generate_comparison_report(results, output_dir):
    """Generate a comparison report of all benchmarks"""
    # Extract data for comparison
    names = [r.name for r in results]
    throughputs = [r.get_summary()["requests_per_second"] for r in results]
    latencies = [r.get_summary()["latency"]["mean_ms"] for r in results]
    p95_latencies = [r.get_summary()["latency"]["p95_ms"] for r in results]
    error_rates = [r.get_summary()["error_rate"] for r in results]
    cpu_usages = [r.get_summary()["system"]["avg_cpu_usage"] for r in results]
    memory_usages = [r.get_summary()["system"]["avg_memory_usage"] for r in results]
    
    # Create figure with subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    
    # Throughput comparison
    axs[0, 0].bar(names, throughputs, color='blue')
    axs[0, 0].set_title('Throughput Comparison')
    axs[0, 0].set_ylabel('Requests/second')
    axs[0, 0].set_xticklabels(names, rotation=45, ha='right')
    
    # Latency comparison
    axs[0, 1].bar(names, latencies, color='green', alpha=0.7, label='Mean')
    axs[0, 1].bar(names, p95_latencies, color='red', alpha=0.5, label='P95')
    axs[0, 1].set_title('Latency Comparison')
    axs[0, 1].set_ylabel('Latency (ms)')
    axs[0, 1].set_xticklabels(names, rotation=45, ha='right')
    axs[0, 1].legend()
    
    # Error rate comparison
    axs[1, 0].bar(names, error_rates, color='orange')
    axs[1, 0].set_title('Error Rate Comparison')
    axs[1, 0].set_ylabel('Error Rate (%)')
    axs[1, 0].set_xticklabels(names, rotation=45, ha='right')
    
    # CPU usage comparison
    axs[1, 1].bar(names, cpu_usages, color='purple')
    axs[1, 1].set_title('CPU Usage Comparison')
    axs[1, 1].set_ylabel('CPU Usage (%)')
    axs[1, 1].set_xticklabels(names, rotation=45, ha='right')
    
    # Memory usage comparison
    axs[2, 0].bar(names, memory_usages, color='brown')
    axs[2, 0].set_title('Memory Usage Comparison')
    axs[2, 0].set_ylabel('Memory Usage (%)')
    axs[2, 0].set_xticklabels(names, rotation=45, ha='right')
    
    # Summary table
    axs[2, 1].axis('tight')
    axs[2, 1].axis('off')
    table_data = [
        ['Benchmark', 'Throughput', 'Mean Latency', 'P95 Latency', 'Error Rate', 'CPU', 'Memory'],
    ]
    for i, name in enumerate(names):
        table_data.append([
            name,
            f"{throughputs[i]:.2f} req/s",
            f"{latencies[i]:.2f} ms",
            f"{p95_latencies[i]:.2f} ms",
            f"{error_rates[i]:.2f}%",
            f"{cpu_usages[i]:.2f}%",
            f"{memory_usages[i]:.2f}%"
        ])
    axs[2, 1].table(cellText=table_data, loc='center', cellLoc='center')
    
    plt.tight_layout()
    plt.suptitle('DDoS.AI Performance Benchmark Comparison', fontsize=16)
    plt.subplots_adjust(top=0.95)
    
    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "benchmark_comparison.png")
    plt.savefig(output_file)
    print(f"Comparison report saved to {output_file}")
    
    # Also save raw data
    data = {
        "benchmarks": [r.get_summary() for r in results]
    }
    data_file = os.path.join(output_dir, "benchmark_comparison.json")
    with open(data_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Raw comparison data saved to {data_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Performance benchmarking for DDoS.AI platform")
    parser.add_argument("--output-dir", default="benchmark_results",
                        help="Directory for benchmark results")
    parser.add_argument("--single", action="store_true",
                        help="Run only a single benchmark instead of the full suite")
    parser.add_argument("--duration", type=int, default=30,
                        help="Benchmark duration in seconds (for single benchmark)")
    parser.add_argument("--rate", type=int, default=50,
                        help="Request rate in requests per second (for single benchmark)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of worker tasks (for single benchmark)")
    parser.add_argument("--attack", action="store_true",
                        help="Include attack simulation (for single benchmark)")
    
    args = parser.parse_args()
    
    if args.single:
        # Run a single benchmark
        benchmark_params = {
            "name": "Custom Benchmark",
            "duration": args.duration,
            "packet_rate": args.rate,
            "workers": args.workers
        }
        
        if args.attack:
            benchmark_params.update({
                "name": "Custom Attack Benchmark",
                "attack_start": args.duration // 3,
                "attack_duration": args.duration // 3,
                "attack_rate": args.rate * 2,
                "attack_type": "syn_flood"
            })
        
        result = asyncio.run(run_benchmark(**benchmark_params))
        result.plot_results(args.output_dir)
        
        # Print summary
        summary = result.get_summary()
        print(f"Benchmark completed: {summary['requests']} requests, "
              f"{summary['error_rate']:.2f}% errors, "
              f"{summary['requests_per_second']:.2f} req/s, "
              f"avg latency: {summary['latency']['mean_ms']:.2f}ms")
    else:
        # Run full benchmark suite
        asyncio.run(run_benchmarks(args.output_dir))

if __name__ == "__main__":
    main()