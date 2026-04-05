import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from collections import deque
from models.q_network import DuelingDQNNetwork

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim=16, action_dim=4, lr=1e-4, gamma=0.99, buffer_size=100000):
        # Force CUDA if available (which it should be for your RTX 5080!)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.action_dim = action_dim
        
        # Initialize Dueling Networks
        self.policy_net = DuelingDQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is never trained directly
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)

    def preprocess_batch(self, states):
        """Highly optimized, vectorized one-hot encoding for massive GPU batches"""
        # 1. Convert list of states to a single numpy array: shape (Batch, 4, 4)
        states_np = np.array(states, dtype=np.int64)
        states_np = np.clip(states_np, 0, 15) # Cap at 2^15 to prevent index out of bounds
        
        batch_size = states_np.shape[0]
        one_hot = np.zeros((batch_size, 16, 4, 4), dtype=np.float32)
        
        # 2. Advanced NumPy indexing (lightning fast memory mapping)
        b = np.arange(batch_size)[:, None, None]
        r = np.arange(4)[None, :, None]
        c = np.arange(4)[None, None, :]
        one_hot[b, states_np, r, c] = 1.0
        
        # 3. Move the massive block to GPU in exactly ONE operation
        return torch.FloatTensor(one_hot).to(self.device)

    def preprocess(self, state):
        """Fallback for a single state (used during live gameplay)"""
        return self.preprocess_batch([state])

    def act(self, state, valid_actions, epsilon=0.0):
        # Game Over state fallback
        if len(valid_actions) == 0:
            return 0 
            
        # 1. Random Exploration, but ONLY from legal moves
        if random.random() < epsilon:
            return random.choice(valid_actions)
        
        # 2. Exploitation (Ask the GPU)
        with torch.no_grad():
            state_t = self.preprocess(state)
            q_values = self.policy_net(state_t).squeeze(0) # shape: (4,)
            
            # Action Masking: Create a mask of -infinity for all actions
            mask = torch.full((self.action_dim,), float('-inf')).to(self.device)
            
            # Set valid actions to 0, so they keep their original Q-values
            for action in valid_actions:
                mask[action] = 0.0
                
            # Valid Q-values stay the same, invalid become -infinity
            masked_q_values = q_values + mask
            
            # Select the highest VALID Q-value
            return masked_q_values.argmax().item()

    def train_step(self, batch_size):
        if len(self.memory) < batch_size:
            return 0.0
        
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Vectorized GPU transfers
        states_t = self.preprocess_batch(states)
        next_states_t = self.preprocess_batch(next_states)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # 1. Current Q Values (from Policy Net)
        current_q = self.policy_net(states_t).gather(1, actions_t)

        # 2. DOUBLE DQN LOGIC
        with torch.no_grad():
            # Step A: Use Policy Net to PICK the best action for the next state
            next_actions = self.policy_net(next_states_t).argmax(1).unsqueeze(1)
            
            # Step B: Use Target Net to EVALUATE that action
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            
            # 3. Calculate target Q value
            target_q = rewards_t + (1 - dones_t) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())