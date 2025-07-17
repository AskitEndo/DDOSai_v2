"""
Graph Neural Network analyzer for network-level DDoS detection
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import networkx as nx

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_detector import BaseDetector
from models.data_models import TrafficPacket, ProtocolType
from core.exceptions import ModelInferenceError


class GNNNetwork(nn.Module):
    """Graph Neural Network for network traffic analysis"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1, num_layers: int = 2):
        """
        Initialize GNN network
        
        Args:
            input_dim: Dimension of node features
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (1 for binary classification)
            num_layers: Number of GCN layers
        """
        super(GNNNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Graph Convolutional layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean + max pooling
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through GNN
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment for nodes (for batched graphs)
        
        Returns:
            Graph-level predictions
        """
        # Apply GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        # Graph-level pooling
        if batch is None:
            # Single graph
            graph_mean = torch.mean(x, dim=0, keepdim=True)
            graph_max = torch.max(x, dim=0, keepdim=True)[0]
        else:
            # Batched graphs
            graph_mean = global_mean_pool(x, batch)
            graph_max = global_max_pool(x, batch)
        
        # Combine pooled representations
        graph_repr = torch.cat([graph_mean, graph_max], dim=1)
        
        # Classification
        output = self.classifier(graph_repr)
        
        return output


class GNNAnalyzer(BaseDetector):
    """Graph Neural Network analyzer for network-level threat detection"""
    
    def __init__(self, node_feature_dim: int, hidden_dim: int = 64, 
                 window_size: int = 60, device: str = None):
        """
        Initialize GNN analyzer
        
        Args:
            node_feature_dim: Dimension of node features
            hidden_dim: Hidden layer dimension
            window_size: Time window for graph construction (seconds)
            device: Device to run model on ('cpu' or 'cuda')
        """
        super().__init__("GNNAnalyzer", "1.0.0")
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize network
        self.network = GNNNetwork(node_feature_dim, hidden_dim)
        self.network.to(self.device)
        
        # Training parameters
        self.learning_rate = 0.001
        self.batch_size = 16
        self.num_epochs = 100
        self.early_stopping_patience = 15
        
        # Graph construction parameters
        self.min_nodes = 3  # Minimum nodes for valid graph
        self.max_nodes = 100  # Maximum nodes to prevent memory issues
        
        # Model state
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCELoss()
        
        self.logger = logging.getLogger(__name__)
    
    def construct_graph_from_packets(self, packets: List[TrafficPacket]) -> Optional[Data]:
        """
        Construct a graph from network packets
        
        Args:
            packets: List of traffic packets within time window
            
        Returns:
            PyTorch Geometric Data object or None if invalid
        """
        if len(packets) < self.min_nodes:
            return None
        
        # Extract unique IP addresses as nodes
        ip_set = set()
        for packet in packets:
            ip_set.add(packet.src_ip)
            ip_set.add(packet.dst_ip)
        
        if len(ip_set) < self.min_nodes:
            return None
        
        # Limit graph size
        if len(ip_set) > self.max_nodes:
            # Keep most active IPs
            ip_counts = Counter()
            for packet in packets:
                ip_counts[packet.src_ip] += 1
                ip_counts[packet.dst_ip] += 1
            
            ip_set = set([ip for ip, _ in ip_counts.most_common(self.max_nodes)])
        
        # Create IP to index mapping
        ip_to_idx = {ip: idx for idx, ip in enumerate(sorted(ip_set))}
        num_nodes = len(ip_set)
        
        # Initialize node features
        node_features = self._extract_node_features(packets, ip_to_idx)
        
        # Build edge list and edge features
        edge_list = []
        edge_weights = []
        
        # Count connections between IPs
        connections = defaultdict(lambda: defaultdict(int))
        for packet in packets:
            if packet.src_ip in ip_to_idx and packet.dst_ip in ip_to_idx:
                src_idx = ip_to_idx[packet.src_ip]
                dst_idx = ip_to_idx[packet.dst_ip]
                connections[src_idx][dst_idx] += 1
        
        # Create edges
        for src_idx in connections:
            for dst_idx, weight in connections[src_idx].items():
                edge_list.append([src_idx, dst_idx])
                edge_weights.append(weight)
        
        if len(edge_list) == 0:
            return None
        
        # Convert to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        # Create PyTorch Geometric data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes
        )
        
        return data
    
    def _extract_node_features(self, packets: List[TrafficPacket], 
                              ip_to_idx: Dict[str, int]) -> np.ndarray:
        """
        Extract features for each node (IP address)
        
        Args:
            packets: List of traffic packets
            ip_to_idx: Mapping from IP to node index
            
        Returns:
            Node feature matrix [num_nodes, feature_dim]
        """
        num_nodes = len(ip_to_idx)
        features = np.zeros((num_nodes, self.node_feature_dim))
        
        # Initialize node statistics
        node_stats = {idx: {
            'packet_count': 0,
            'byte_count': 0,
            'in_degree': 0,
            'out_degree': 0,
            'protocols': set(),
            'ports': set(),
            'avg_packet_size': 0,
            'entropy_sum': 0,
            'syn_count': 0,
            'unique_connections': set()
        } for idx in range(num_nodes)}
        
        # Collect statistics from packets
        for packet in packets:
            if packet.src_ip in ip_to_idx and packet.dst_ip in ip_to_idx:
                src_idx = ip_to_idx[packet.src_ip]
                dst_idx = ip_to_idx[packet.dst_ip]
                
                # Source node statistics
                node_stats[src_idx]['packet_count'] += 1
                node_stats[src_idx]['byte_count'] += packet.packet_size
                node_stats[src_idx]['out_degree'] += 1
                node_stats[src_idx]['protocols'].add(packet.protocol.value)
                node_stats[src_idx]['ports'].add(packet.src_port)
                node_stats[src_idx]['entropy_sum'] += packet.payload_entropy
                node_stats[src_idx]['unique_connections'].add(dst_idx)
                
                if 'SYN' in packet.flags:
                    node_stats[src_idx]['syn_count'] += 1
                
                # Destination node statistics
                node_stats[dst_idx]['in_degree'] += 1
                node_stats[dst_idx]['unique_connections'].add(src_idx)
        
        # Convert statistics to feature vectors
        for idx in range(num_nodes):
            stats = node_stats[idx]
            
            # Basic traffic features
            features[idx, 0] = np.log1p(stats['packet_count'])  # Log packet count
            features[idx, 1] = np.log1p(stats['byte_count'])    # Log byte count
            features[idx, 2] = stats['in_degree']               # In-degree
            features[idx, 3] = stats['out_degree']              # Out-degree
            
            # Derived features
            if stats['packet_count'] > 0:
                features[idx, 4] = stats['byte_count'] / stats['packet_count']  # Avg packet size
                features[idx, 5] = stats['entropy_sum'] / stats['packet_count']  # Avg entropy
                features[idx, 6] = stats['syn_count'] / stats['packet_count']    # SYN ratio
            
            # Protocol and port diversity
            features[idx, 7] = len(stats['protocols'])          # Protocol diversity
            features[idx, 8] = len(stats['ports'])              # Port diversity
            features[idx, 9] = len(stats['unique_connections']) # Connection diversity
            
            # Degree ratio (out/in)
            if stats['in_degree'] > 0:
                features[idx, 10] = stats['out_degree'] / stats['in_degree']
            else:
                features[idx, 10] = stats['out_degree']
            
            # Pad remaining features with zeros if needed
            if self.node_feature_dim > 11:
                features[idx, 11:] = 0
        
        return features
    
    def train(self, training_data: List[Tuple[List[TrafficPacket], bool]], 
              labels: np.ndarray = None) -> Dict[str, Any]:
        """
        Train the GNN on network traffic graphs
        
        Args:
            training_data: List of (packet_list, is_malicious) tuples
            labels: Not used (labels are in training_data)
            
        Returns:
            Training metrics and statistics
        """
        try:
            self.logger.info(f"Starting GNN training with {len(training_data)} samples")
            
            if len(training_data) == 0:
                raise ModelInferenceError("Empty training data provided")
            
            # Convert packets to graphs
            graphs = []
            graph_labels = []
            
            for packets, is_malicious in training_data:
                graph = self.construct_graph_from_packets(packets)
                if graph is not None:
                    graphs.append(graph)
                    graph_labels.append(float(is_malicious))
            
            if len(graphs) == 0:
                raise ModelInferenceError("No valid graphs could be constructed from training data")
            
            self.logger.info(f"Constructed {len(graphs)} valid graphs from training data")
            
            # Training loop
            self.network.train()
            training_losses = []
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.num_epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                # Process graphs in batches
                for i in range(0, len(graphs), self.batch_size):
                    batch_graphs = graphs[i:i + self.batch_size]
                    batch_labels = torch.tensor(
                        graph_labels[i:i + self.batch_size], 
                        dtype=torch.float, 
                        device=self.device
                    ).unsqueeze(1)
                    
                    # Create batch
                    batch = Batch.from_data_list(batch_graphs).to(self.device)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.network(batch.x, batch.edge_index, batch.batch)
                    loss = self.criterion(outputs, batch_labels)
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
                training_losses.append(avg_loss)
                
                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.6f}")
            
            self.is_trained = True
            
            # Training statistics
            training_stats = {
                "epochs_trained": len(training_losses),
                "final_loss": training_losses[-1] if training_losses else 0,
                "best_loss": best_loss,
                "training_graphs": len(graphs),
                "valid_graph_ratio": len(graphs) / len(training_data)
            }
            
            self.logger.info(f"Training completed. Final loss: {training_stats['final_loss']:.6f}")
            return training_stats
            
        except Exception as e:
            raise ModelInferenceError(f"Training failed: {e}")
    
    def predict(self, features: List[TrafficPacket]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Make prediction on network traffic
        
        Args:
            features: List of traffic packets to analyze
            
        Returns:
            Tuple of (is_malicious, confidence_score, explanation)
        """
        try:
            if not self.is_trained:
                raise ModelInferenceError("Model must be trained before making predictions")
            
            # Construct graph from packets
            graph = self.construct_graph_from_packets(features)
            
            if graph is None:
                # Cannot construct valid graph, return low confidence benign
                return False, 0.1, {
                    "error": "Cannot construct valid graph",
                    "num_packets": len(features),
                    "model_type": "gnn"
                }
            
            # Make prediction
            self.network.eval()
            with torch.no_grad():
                graph = graph.to(self.device)
                output = self.network(graph.x, graph.edge_index)
                malicious_prob = output.item()
            
            is_malicious = malicious_prob > 0.5
            confidence = float(malicious_prob if is_malicious else 1 - malicious_prob)
            
            # Create explanation
            explanation = {
                "malicious_probability": float(malicious_prob),
                "num_nodes": graph.num_nodes,
                "num_edges": graph.edge_index.size(1),
                "graph_density": float(graph.edge_index.size(1)) / (graph.num_nodes * (graph.num_nodes - 1)) if graph.num_nodes > 1 else 0,
                "model_type": "gnn",
                "window_size": self.window_size
            }
            
            return is_malicious, confidence, explanation
            
        except Exception as e:
            raise ModelInferenceError(f"Prediction failed: {e}")
    
    def save_model(self, filepath: str) -> bool:
        """Save trained model to file"""
        try:
            if not self.is_trained:
                self.logger.error("Cannot save untrained model")
                return False
            
            model_state = {
                'network_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_params': {
                    'node_feature_dim': self.node_feature_dim,
                    'hidden_dim': self.hidden_dim,
                    'window_size': self.window_size,
                    'learning_rate': self.learning_rate,
                    'batch_size': self.batch_size,
                    'num_epochs': self.num_epochs
                },
                'is_trained': self.is_trained,
                'version': self.version,
                'saved_at': datetime.now().isoformat()
            }
            
            torch.save(model_state, filepath)
            self.logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model from file"""
        try:
            if not os.path.exists(filepath):
                self.logger.error(f"Model file not found: {filepath}")
                return False
            
            model_state = torch.load(filepath, map_location=self.device, weights_only=False)
            
            # Restore model parameters
            self.node_feature_dim = model_state['model_params']['node_feature_dim']
            self.hidden_dim = model_state['model_params']['hidden_dim']
            self.window_size = model_state['model_params']['window_size']
            self.learning_rate = model_state['model_params']['learning_rate']
            self.batch_size = model_state['model_params']['batch_size']
            self.num_epochs = model_state['model_params']['num_epochs']
            
            # Recreate network
            self.network = GNNNetwork(self.node_feature_dim, self.hidden_dim)
            self.network.to(self.device)
            self.network.load_state_dict(model_state['network_state_dict'])
            
            # Restore optimizer
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
            self.optimizer.load_state_dict(model_state['optimizer_state_dict'])
            
            self.is_trained = model_state['is_trained']
            
            self.logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def analyze_network_structure(self, packets: List[TrafficPacket]) -> Dict[str, Any]:
        """
        Analyze network structure from packets
        
        Args:
            packets: List of traffic packets
            
        Returns:
            Network structure analysis
        """
        graph = self.construct_graph_from_packets(packets)
        
        if graph is None:
            return {"error": "Cannot construct valid graph"}
        
        # Convert to NetworkX for analysis
        G = nx.Graph()
        
        # Add nodes
        for i in range(graph.num_nodes):
            G.add_node(i)
        
        # Add edges
        edge_list = graph.edge_index.t().numpy()
        for src, dst in edge_list:
            G.add_edge(src, dst)
        
        # Calculate network metrics
        analysis = {
            "num_nodes": graph.num_nodes,
            "num_edges": len(edge_list),
            "density": nx.density(G),
            "avg_clustering": nx.average_clustering(G),
            "num_connected_components": nx.number_connected_components(G),
            "diameter": nx.diameter(G) if nx.is_connected(G) else -1,
            "avg_shortest_path": nx.average_shortest_path_length(G) if nx.is_connected(G) else -1
        }
        
        # Degree statistics
        degrees = [G.degree(n) for n in G.nodes()]
        if degrees:
            analysis.update({
                "avg_degree": np.mean(degrees),
                "max_degree": max(degrees),
                "degree_std": np.std(degrees)
            })
        
        return analysis