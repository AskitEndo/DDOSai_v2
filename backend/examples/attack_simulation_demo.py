"""
Demo script for Attack Simulator

This script demonstrates how to use the AttackSimulator class to simulate
different types of DDoS attacks.
"""
import sys
import os
import time
import logging
import argparse
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.simulation.attack_simulator import AttackSimulator, SimulationStatus


def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def run_syn_flood_demo(target_ip, target_port, duration, packet_rate):
    """Run SYN flood demo"""
    print(f"\n=== Running SYN Flood Demo ===")
    print(f"Target: {target_ip}:{target_port}")
    print(f"Duration: {duration} seconds")
    print(f"Packet Rate: {packet_rate} packets/second")
    
    # Create simulator
    simulator = AttackSimulator()
    
    # Start simulation
    simulation_id = simulator.simulate_syn_flood(
        target_ip=target_ip,
        target_port=target_port,
        duration=duration,
        packet_rate=packet_rate
    )
    
    print(f"Started SYN flood simulation: {simulation_id}")
    
    # Monitor simulation
    start_time = datetime.now()
    try:
        while simulator.status == SimulationStatus.RUNNING:
            stats = simulator.get_statistics()
            elapsed = (datetime.now() - start_time).total_seconds()
            
            # Print progress
            print(f"\rElapsed: {elapsed:.1f}s | "
                  f"Packets: {stats['packets_sent']} | "
                  f"Rate: {stats.get('current_packet_rate', 0):.1f} pkt/s", end="")
            
            time.sleep(0.5)
            
            # Check if duration exceeded
            if elapsed > duration + 5:  # Add 5 seconds buffer
                print("\nSimulation taking too long, stopping...")
                simulator.stop_simulation(simulation_id)
                break
    
    except KeyboardInterrupt:
        print("\nStopping simulation...")
        simulator.stop_simulation(simulation_id)
    
    # Get final statistics
    stats = simulator.get_statistics()
    
    print("\n\n=== Simulation Complete ===")
    print(f"Status: {stats['status']}")
    print(f"Packets Sent: {stats['packets_sent']}")
    print(f"Bytes Sent: {stats['bytes_sent']}")
    print(f"Duration: {stats['duration']:.2f} seconds")
    print(f"Average Packet Rate: {stats['packet_rate']:.2f} packets/second")
    print(f"Errors: {stats['errors']}")


def run_udp_flood_demo(target_ip, target_port, duration, packet_rate, packet_size):
    """Run UDP flood demo"""
    print(f"\n=== Running UDP Flood Demo ===")
    print(f"Target: {target_ip}:{target_port}")
    print(f"Duration: {duration} seconds")
    print(f"Packet Rate: {packet_rate} packets/second")
    print(f"Packet Size: {packet_size} bytes")
    
    # Create simulator
    simulator = AttackSimulator()
    
    # Start simulation
    simulation_id = simulator.simulate_udp_flood(
        target_ip=target_ip,
        target_port=target_port,
        duration=duration,
        packet_rate=packet_rate,
        packet_size=packet_size
    )
    
    print(f"Started UDP flood simulation: {simulation_id}")
    
    # Monitor simulation
    start_time = datetime.now()
    try:
        while simulator.status == SimulationStatus.RUNNING:
            stats = simulator.get_statistics()
            elapsed = (datetime.now() - start_time).total_seconds()
            
            # Print progress
            print(f"\rElapsed: {elapsed:.1f}s | "
                  f"Packets: {stats['packets_sent']} | "
                  f"Rate: {stats.get('current_packet_rate', 0):.1f} pkt/s", end="")
            
            time.sleep(0.5)
            
            # Check if duration exceeded
            if elapsed > duration + 5:  # Add 5 seconds buffer
                print("\nSimulation taking too long, stopping...")
                simulator.stop_simulation(simulation_id)
                break
    
    except KeyboardInterrupt:
        print("\nStopping simulation...")
        simulator.stop_simulation(simulation_id)
    
    # Get final statistics
    stats = simulator.get_statistics()
    
    print("\n\n=== Simulation Complete ===")
    print(f"Status: {stats['status']}")
    print(f"Packets Sent: {stats['packets_sent']}")
    print(f"Bytes Sent: {stats['bytes_sent']}")
    print(f"Duration: {stats['duration']:.2f} seconds")
    print(f"Average Packet Rate: {stats['packet_rate']:.2f} packets/second")
    print(f"Errors: {stats['errors']}")


def run_http_flood_demo(target_ip, target_port, duration, request_rate, num_threads, use_https):
    """Run HTTP flood demo"""
    print(f"\n=== Running HTTP Flood Demo ===")
    print(f"Target: {target_ip}:{target_port}")
    print(f"Duration: {duration} seconds")
    print(f"Request Rate: {request_rate} requests/second")
    print(f"Threads: {num_threads}")
    print(f"Protocol: {'HTTPS' if use_https else 'HTTP'}")
    
    # Create simulator
    simulator = AttackSimulator()
    
    # Start simulation
    simulation_id = simulator.simulate_http_flood(
        target_ip=target_ip,
        target_port=target_port,
        duration=duration,
        request_rate=request_rate,
        num_threads=num_threads,
        use_https=use_https
    )
    
    print(f"Started HTTP flood simulation: {simulation_id}")
    
    # Monitor simulation
    start_time = datetime.now()
    try:
        while simulator.status == SimulationStatus.RUNNING:
            stats = simulator.get_statistics()
            elapsed = (datetime.now() - start_time).total_seconds()
            
            # Print progress
            print(f"\rElapsed: {elapsed:.1f}s | "
                  f"Requests: {stats['packets_sent']} | "
                  f"Rate: {stats.get('current_packet_rate', 0):.1f} req/s", end="")
            
            time.sleep(0.5)
            
            # Check if duration exceeded
            if elapsed > duration + 5:  # Add 5 seconds buffer
                print("\nSimulation taking too long, stopping...")
                simulator.stop_simulation(simulation_id)
                break
    
    except KeyboardInterrupt:
        print("\nStopping simulation...")
        simulator.stop_simulation(simulation_id)
    
    # Get final statistics
    stats = simulator.get_statistics()
    
    print("\n\n=== Simulation Complete ===")
    print(f"Status: {stats['status']}")
    print(f"Requests Sent: {stats['packets_sent']}")
    print(f"Bytes Sent: {stats['bytes_sent']}")
    print(f"Duration: {stats['duration']:.2f} seconds")
    print(f"Average Request Rate: {stats['packet_rate']:.2f} requests/second")
    print(f"Errors: {stats['errors']}")


def run_slowloris_demo(target_ip, target_port, duration, num_connections, connection_rate, use_https):
    """Run Slowloris demo"""
    print(f"\n=== Running Slowloris Demo ===")
    print(f"Target: {target_ip}:{target_port}")
    print(f"Duration: {duration} seconds")
    print(f"Connections: {num_connections}")
    print(f"Connection Rate: {connection_rate} connections/second")
    print(f"Protocol: {'HTTPS' if use_https else 'HTTP'}")
    
    # Create simulator
    simulator = AttackSimulator()
    
    # Start simulation
    simulation_id = simulator.simulate_slowloris(
        target_ip=target_ip,
        target_port=target_port,
        duration=duration,
        num_connections=num_connections,
        connection_rate=connection_rate,
        use_https=use_https
    )
    
    print(f"Started Slowloris simulation: {simulation_id}")
    
    # Monitor simulation
    start_time = datetime.now()
    try:
        while simulator.status == SimulationStatus.RUNNING:
            stats = simulator.get_statistics()
            elapsed = (datetime.now() - start_time).total_seconds()
            
            # Print progress
            print(f"\rElapsed: {elapsed:.1f}s | "
                  f"Packets: {stats['packets_sent']} | "
                  f"Rate: {stats.get('current_packet_rate', 0):.1f} pkt/s", end="")
            
            time.sleep(0.5)
            
            # Check if duration exceeded
            if elapsed > duration + 5:  # Add 5 seconds buffer
                print("\nSimulation taking too long, stopping...")
                simulator.stop_simulation(simulation_id)
                break
    
    except KeyboardInterrupt:
        print("\nStopping simulation...")
        simulator.stop_simulation(simulation_id)
    
    # Get final statistics
    stats = simulator.get_statistics()
    
    print("\n\n=== Simulation Complete ===")
    print(f"Status: {stats['status']}")
    print(f"Packets Sent: {stats['packets_sent']}")
    print(f"Bytes Sent: {stats['bytes_sent']}")
    print(f"Duration: {stats['duration']:.2f} seconds")
    print(f"Average Packet Rate: {stats['packet_rate']:.2f} packets/second")
    print(f"Errors: {stats['errors']}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="DDoS Attack Simulator Demo")
    parser.add_argument("--attack-type", choices=["syn", "udp", "http", "slowloris"], default="syn",
                        help="Type of attack to simulate")
    parser.add_argument("--target-ip", default="127.0.0.1", help="Target IP address")
    parser.add_argument("--target-port", type=int, default=80, help="Target port")
    parser.add_argument("--duration", type=int, default=10, help="Attack duration in seconds")
    parser.add_argument("--packet-rate", type=int, default=100, help="Packets per second")
    parser.add_argument("--packet-size", type=int, default=512, help="Packet size in bytes (UDP only)")
    parser.add_argument("--num-threads", type=int, default=5, help="Number of threads (HTTP only)")
    parser.add_argument("--num-connections", type=int, default=100, help="Number of connections (Slowloris only)")
    parser.add_argument("--connection-rate", type=int, default=10, help="Connections per second (Slowloris only)")
    parser.add_argument("--use-https", action="store_true", help="Use HTTPS instead of HTTP")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Run appropriate demo
    if args.attack_type == "syn":
        run_syn_flood_demo(args.target_ip, args.target_port, args.duration, args.packet_rate)
    elif args.attack_type == "udp":
        run_udp_flood_demo(args.target_ip, args.target_port, args.duration, args.packet_rate, args.packet_size)
    elif args.attack_type == "http":
        run_http_flood_demo(args.target_ip, args.target_port, args.duration, args.packet_rate, args.num_threads, args.use_https)
    elif args.attack_type == "slowloris":
        run_slowloris_demo(args.target_ip, args.target_port, args.duration, args.num_connections, args.connection_rate, args.use_https)


if __name__ == "__main__":
    main()