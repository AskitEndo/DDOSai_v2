"""
Reinforcement Learning threat scorer using Deep Q-Network (DQN)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime
import pickle

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_detector import BaseDetector
from models.data_models import TrafficPacket
from core.exceptions import ModelInferenceError


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNNetwork(nn.Module):
    """Deep Q-Network for threat scoring"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        """
        Initialize DQN network
        
        Args:
            state_dim: Dimension of state space (feature vector size)
            action_dim: Dimension of action space (threat score levels)
            hidden_dims: Hidden layer dimensions
        """
        super(DQNNetwork, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer (Q-values for each action)
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through network"""
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for DQN training"""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class RLThreatScorer(BaseDetector):
    """Reinforcement Learning threat scorer using DQN"""
    
    def __init__(self, state_dim: int, num_threat_levels: int = 11, 
                 learning_rate: float = 0.001, device: str = None):
        """
        Initialize RL threat scorer
        
        Args:
            state_dim: Dimension of state space (feature vector size)
            num_threat_levels: Number of threat score levels (0-10 -> 0-100 in steps of 10)
            learning_rate: Learning rate for training
            device: Device to run model on ('cpu' or 'cuda')
        """
        super().__init__("RLThreatScorer", "1.0.0")
        
        self.base_state_dim = state_dim
        self.context_dim = 5  # Fixed context dimension
        self.state_dim = state_dim + self.context_dim  # Total state dimension
        self.num_threat_levels = num_threat_levels
        self.learning_rate = learning_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize networks with full state dimension
        self.q_network = DQNNetwork(self.state_dim, num_threat_levels).to(self.device)
        self.target_network = DQNNetwork(self.state_dim, num_threat_levels).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Training parameters
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Hyperparameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Discount factor
        self.batch_size = 32
        self.target_update_freq = 100  # Update target network every N steps
        self.training_step = 0
        
        # Performance tracking
        self.episode_rewards = []
        self.detection_accuracy = []
        self.false_positive_rate = []
        
        self.logger = logging.getLogger(__name__)
    
    def get_state_features(self, packet_features: np.ndarray, 
                          context_features: Dict[str, float] = None) -> np.ndarray:
        """
        Create state representation from packet features and context
        
        Args:
            packet_features: Feature vector from packet analysis
            context_features: Additional context (recent detections, system load, etc.)
            
        Returns:
            State vector for RL agent (always fixed size)
        """
        state = packet_features.copy()
        
        # Always add context features (use defaults if not provided)
        context_array = np.array([
            context_features.get('recent_threat_rate', 0.0) if context_features else 0.0,
            context_features.get('system_load', 0.5) if context_features else 0.5,
            context_features.get('false_positive_rate', 0.1) if context_features else 0.1,
            context_features.get('detection_confidence', 0.5) if context_features else 0.5,
            context_features.get('time_since_last_attack', 1.0) if context_features else 1.0
        ])
        state = np.concatenate([state, context_array])
        
        return state
    
    def action_to_threat_score(self, action: int) -> int:
        """Convert action index to threat score (0-100)"""
        return action * 10  # 0->0, 1->10, ..., 10->100
    
    def threat_score_to_action(self, threat_score: int) -> int:
        """Convert threat score to action index"""
        return min(threat_score // 10, self.num_threat_levels - 1)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.num_threat_levels - 1)
        else:
            # Exploit: best action according to Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def calculate_reward(self, predicted_score: int, true_label: bool, 
                        confidence: float = 1.0) -> float:
        """
        Calculate reward based on prediction accuracy and other factors
        
        Args:
            predicted_score: Predicted threat score (0-100)
            true_label: True label (True=malicious, False=benign)
            confidence: Confidence in the true label
            
        Returns:
            Reward value
        """
        # Base reward based on accuracy
        if true_label:  # Malicious traffic
            if predicted_score >= 70:  # High threat score for malicious
                reward = 1.0
            elif predicted_score >= 40:  # Medium threat score
                reward = 0.5
            else:  # Low threat score for malicious (missed detection)
                reward = -1.0
        else:  # Benign traffic
            if predicted_score <= 30:  # Low threat score for benign
                reward = 1.0
            elif predicted_score <= 60:  # Medium threat score
                reward = 0.0
            else:  # High threat score for benign (false positive)
                reward = -0.8
        
        # Adjust reward based on confidence
        reward *= confidence
        
        # Add small penalty for extreme scores to encourage moderation
        if predicted_score == 0 or predicted_score == 100:
            reward -= 0.1
        
        return reward
    
    def train_step(self):
        """Perform one training step using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        experiences = self.replay_buffer.sample(self.batch_size)
        
        # Prepare batch tensors
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def train(self, training_data: List[Tuple[np.ndarray, bool, Dict[str, float]]], 
              labels: np.ndarray = None) -> Dict[str, Any]:
        """
        Train the RL threat scorer
        
        Args:
            training_data: List of (features, is_malicious, context) tuples
            labels: Not used (labels are in training_data)
            
        Returns:
            Training metrics and statistics
        """
        try:
            self.logger.info(f"Starting RL training with {len(training_data)} samples")
            
            if len(training_data) == 0:
                raise ModelInferenceError("Empty training data provided")
            
            # Training loop
            num_episodes = 1000
            episode_length = min(50, len(training_data))
            total_losses = []
            
            for episode in range(num_episodes):
                episode_reward = 0
                episode_loss = 0
                correct_predictions = 0
                false_positives = 0
                total_benign = 0
                
                # Sample episode data
                episode_data = random.sample(training_data, episode_length)
                
                for i, (features, is_malicious, context) in enumerate(episode_data):
                    # Create state
                    state = self.get_state_features(features, context)
                    
                    # Select action
                    action = self.select_action(state, training=True)
                    predicted_score = self.action_to_threat_score(action)
                    
                    # Calculate reward
                    reward = self.calculate_reward(predicted_score, is_malicious)
                    episode_reward += reward
                    
                    # Track accuracy
                    if is_malicious and predicted_score >= 50:
                        correct_predictions += 1
                    elif not is_malicious and predicted_score < 50:
                        correct_predictions += 1
                    
                    if not is_malicious:
                        total_benign += 1
                        if predicted_score >= 50:
                            false_positives += 1
                    
                    # Create next state (for simplicity, use random next sample)
                    if i < len(episode_data) - 1:
                        next_features, _, next_context = episode_data[i + 1]
                        next_state = self.get_state_features(next_features, next_context)
                        done = False
                    else:
                        next_state = state  # Terminal state
                        done = True
                    
                    # Store experience
                    self.replay_buffer.push(state, action, reward, next_state, done)
                    
                    # Train if enough experiences
                    if len(self.replay_buffer) >= self.batch_size:
                        loss = self.train_step()
                        if loss is not None:
                            episode_loss += loss
                
                # Track episode metrics
                self.episode_rewards.append(episode_reward)
                accuracy = correct_predictions / episode_length
                self.detection_accuracy.append(accuracy)
                
                fp_rate = false_positives / max(total_benign, 1)
                self.false_positive_rate.append(fp_rate)
                
                if episode_loss > 0:
                    total_losses.append(episode_loss)
                
                # Log progress
                if (episode + 1) % 100 == 0:
                    avg_reward = np.mean(self.episode_rewards[-100:])
                    avg_accuracy = np.mean(self.detection_accuracy[-100:])
                    avg_fp_rate = np.mean(self.false_positive_rate[-100:])
                    
                    self.logger.info(
                        f"Episode {episode + 1}/{num_episodes}, "
                        f"Avg Reward: {avg_reward:.3f}, "
                        f"Accuracy: {avg_accuracy:.3f}, "
                        f"FP Rate: {avg_fp_rate:.3f}, "
                        f"Epsilon: {self.epsilon:.3f}"
                    )
            
            self.is_trained = True
            
            # Training statistics
            training_stats = {
                "episodes_trained": num_episodes,
                "final_epsilon": self.epsilon,
                "avg_episode_reward": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
                "final_accuracy": np.mean(self.detection_accuracy[-100:]) if self.detection_accuracy else 0,
                "final_fp_rate": np.mean(self.false_positive_rate[-100:]) if self.false_positive_rate else 0,
                "avg_loss": np.mean(total_losses) if total_losses else 0,
                "training_samples": len(training_data)
            }
            
            self.logger.info(f"Training completed. Final accuracy: {training_stats['final_accuracy']:.3f}")
            return training_stats
            
        except Exception as e:
            raise ModelInferenceError(f"Training failed: {e}")
    
    def predict(self, features: np.ndarray, context: Dict[str, float] = None) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Make threat score prediction
        
        Args:
            features: Feature vector for prediction
            context: Additional context information
            
        Returns:
            Tuple of (is_malicious, confidence_score, explanation)
        """
        try:
            if not self.is_trained:
                raise ModelInferenceError("Model must be trained before making predictions")
            
            # Validate input
            if not self.validate_features(features):
                raise ModelInferenceError("Invalid feature format")
            
            # Handle single sample
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Create state
            state = self.get_state_features(features[0], context or {})
            
            # Make prediction (no exploration)
            action = self.select_action(state, training=False)
            threat_score = self.action_to_threat_score(action)
            
            # Determine if malicious (threshold at 50)
            is_malicious = threat_score >= 50
            
            # Calculate confidence based on Q-values
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                
                # Confidence is based on the difference between top Q-values
                sorted_q = torch.sort(q_values, descending=True)[0]
                if len(sorted_q[0]) > 1:
                    confidence = float((sorted_q[0][0] - sorted_q[0][1]).abs())
                else:
                    confidence = float(sorted_q[0][0].abs())
                
                confidence = min(confidence / 2.0, 1.0)  # Normalize to [0, 1]
            
            # Create explanation
            explanation = {
                "threat_score": threat_score,
                "q_values": q_values[0].cpu().numpy().tolist(),
                "selected_action": action,
                "confidence_raw": confidence,
                "model_type": "reinforcement_learning",
                "epsilon": self.epsilon,
                "context_used": context is not None
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
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_params': {
                    'state_dim': self.state_dim,
                    'num_threat_levels': self.num_threat_levels,
                    'learning_rate': self.learning_rate,
                    'epsilon': self.epsilon,
                    'gamma': self.gamma,
                    'batch_size': self.batch_size
                },
                'training_stats': {
                    'episode_rewards': self.episode_rewards,
                    'detection_accuracy': self.detection_accuracy,
                    'false_positive_rate': self.false_positive_rate,
                    'training_step': self.training_step
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
            self.state_dim = model_state['model_params']['state_dim']
            self.num_threat_levels = model_state['model_params']['num_threat_levels']
            self.learning_rate = model_state['model_params']['learning_rate']
            self.epsilon = model_state['model_params']['epsilon']
            self.gamma = model_state['model_params']['gamma']
            self.batch_size = model_state['model_params']['batch_size']
            
            # Recreate networks
            self.q_network = DQNNetwork(self.state_dim, self.num_threat_levels).to(self.device)
            self.target_network = DQNNetwork(self.state_dim, self.num_threat_levels).to(self.device)
            
            self.q_network.load_state_dict(model_state['q_network_state_dict'])
            self.target_network.load_state_dict(model_state['target_network_state_dict'])
            
            # Restore optimizer
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
            self.optimizer.load_state_dict(model_state['optimizer_state_dict'])
            
            # Restore training stats
            if 'training_stats' in model_state:
                self.episode_rewards = model_state['training_stats']['episode_rewards']
                self.detection_accuracy = model_state['training_stats']['detection_accuracy']
                self.false_positive_rate = model_state['training_stats']['false_positive_rate']
                self.training_step = model_state['training_stats']['training_step']
            
            self.is_trained = model_state['is_trained']
            
            self.logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training performance metrics"""
        if not self.episode_rewards:
            return {}
        
        return {
            "total_episodes": len(self.episode_rewards),
            "avg_reward": np.mean(self.episode_rewards),
            "final_reward": np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
            "avg_accuracy": np.mean(self.detection_accuracy) if self.detection_accuracy else 0,
            "final_accuracy": np.mean(self.detection_accuracy[-100:]) if len(self.detection_accuracy) >= 100 else np.mean(self.detection_accuracy) if self.detection_accuracy else 0,
            "avg_fp_rate": np.mean(self.false_positive_rate) if self.false_positive_rate else 0,
            "final_fp_rate": np.mean(self.false_positive_rate[-100:]) if len(self.false_positive_rate) >= 100 else np.mean(self.false_positive_rate) if self.false_positive_rate else 0,
            "current_epsilon": self.epsilon,
            "training_steps": self.training_step
        }
    
    def update_context(self, recent_detections: List[bool], system_load: float = 0.5):
        """
        Update context information for adaptive scoring
        
        Args:
            recent_detections: List of recent detection results
            system_load: Current system load (0-1)
        """
        if recent_detections:
            threat_rate = sum(recent_detections) / len(recent_detections)
            self.recent_threat_rate = threat_rate
        
        self.system_load = system_load