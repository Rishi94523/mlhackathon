import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import Dict, List, Tuple, Optional
import pickle

# Define the experience tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQN(nn.Module):
    """Deep Q-Network for the RL agent."""
    
    def __init__(self, input_size: int, hidden_size: int = 256, output_size: int = 26):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class HangmanRLAgent:
    """
    Reinforcement Learning agent for playing Hangman.
    Uses Deep Q-Learning with experience replay.
    """
    
    def __init__(self, state_size: int = 100, learning_rate: float = 0.001,
                 gamma: float = 0.95, epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995, memory_size: int = 10000):
        
        self.state_size = state_size
        self.action_size = 26  # 26 letters in alphabet
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        self.batch_size = 32
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(state_size, hidden_size=256, output_size=self.action_size).to(self.device)
        self.target_network = DQN(state_size, hidden_size=256, output_size=self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Update target network
        self.update_target_network()
        
        # Statistics
        self.losses = []
        
    def encode_state(self, masked_word: str, guessed_letters: set, lives_left: int,
                     hmm_probs: Dict[str, float]) -> np.ndarray:
        """
        Encode the game state into a fixed-size vector.
        """
        state_vector = []
        
        # 1. Word length (normalized)
        word_len = len(masked_word)
        state_vector.append(word_len / 20.0)  # Normalize by max expected length
        
        # 2. Progress (percentage of word revealed)
        revealed = sum(1 for c in masked_word if c != '_')
        progress = revealed / word_len if word_len > 0 else 0
        state_vector.append(progress)
        
        # 3. Lives left (normalized)
        state_vector.append(lives_left / 6.0)
        
        # 4. Number of unique letters revealed
        unique_revealed = len(set(c for c in masked_word if c != '_'))
        state_vector.append(unique_revealed / 26.0)
        
        # 5. Letter frequency in revealed portion
        letter_counts = np.zeros(26)
        for c in masked_word:
            if c != '_':
                idx = ord(c) - ord('a')
                letter_counts[idx] += 1
        if revealed > 0:
            letter_counts = letter_counts / revealed
        state_vector.extend(letter_counts.tolist())
        
        # 6. Binary vector for guessed letters
        guessed_vector = [1.0 if self.alphabet[i] in guessed_letters else 0.0 
                         for i in range(26)]
        state_vector.extend(guessed_vector)
        
        # 7. HMM probability distribution (top features)
        hmm_vector = np.zeros(26)
        for letter, prob in hmm_probs.items():
            if letter in self.alphabet:
                idx = self.alphabet.index(letter)
                hmm_vector[idx] = prob
        state_vector.extend(hmm_vector.tolist())
        
        # 8. Pattern features (position of blanks)
        blank_positions = [1.0 if i < word_len and masked_word[i] == '_' else 0.0 
                          for i in range(20)]  # Max 20 positions
        state_vector.extend(blank_positions)
        
        # Ensure fixed size
        state_vector = state_vector[:self.state_size]
        while len(state_vector) < self.state_size:
            state_vector.append(0.0)
        
        return np.array(state_vector, dtype=np.float32)
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, available_actions: List[str],
            training: bool = True) -> str:
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current state vector
            available_actions: List of available letters to guess
            training: Whether in training mode (uses exploration)
            
        Returns:
            Selected letter to guess
        """
        if not available_actions:
            return None
        
        # Epsilon-greedy exploration during training
        if training and random.random() <= self.epsilon:
            return random.choice(available_actions)
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get Q-values from network
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
        
        # Mask unavailable actions
        action_indices = [self.alphabet.index(letter) for letter in available_actions]
        masked_q_values = np.full(self.action_size, -np.inf)
        for idx in action_indices:
            masked_q_values[idx] = q_values[idx]
        
        # Select best available action
        best_action_idx = np.argmax(masked_q_values)
        return self.alphabet[best_action_idx]
    
    def replay(self):
        """Train the network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)
        
        # Current Q-values
        self.q_network.train()
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath: str):
        """Save the agent's model and parameters."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'losses': self.losses
        }, filepath)
        print(f"RL agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a saved agent."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.losses = checkpoint['losses']
        print(f"RL agent loaded from {filepath}")
