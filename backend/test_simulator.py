"""
Simple test script for Attack Simulator
"""
import sys
import os
import time
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulation.attack_simulator import AttackSimulator, SimulationStatus

def main():
    """Main function"""
    print("Testing Attack Simulator...")
    
    # Create simulator
    simulator = AttackSimulator()
    
    # Print status
    print(f"Initial status: {simulator.status}")
    
    # Test helper methods
    ip_range = simulator._generate_random_ip_range(5)
    print(f"Random IP range: {ip_range}")
    
    urls = simulator._generate_random_urls(3)
    print(f"Random URLs: {urls}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main()