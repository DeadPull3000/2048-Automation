import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from collections import deque
from models.q_network import DQNNetwork

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
    def __init__(self, state_dim=16, action_dim=4, lr=1e-4, gamma=0.99, buffer_size=10000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.action_dim = action_dim
        
        self.policy_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)

    def preprocess(self, state):
        # Convert (4,4) log2 board to (16, 4, 4) one-hot
        one_hot = np.zeros((16, 4, 4), dtype=np.float32)
        for r in range(4):
            for c in range(4):
                val = int(state[r][c])
                if val < 16: # Cap at 2^15
                    one_hot[val, r, c] = 1.0
        return torch.FloatTensor(one_hot).unsqueeze(0).to(self.device)

    def act(self, state, valid_actions, epsilon=0.0):
        # If no valid actions (game over state), return 0 safely
        if len(valid_actions) == 0:
            return 0 
            
        # 1. Random Exploration (Epsilon-Greedy), but ONLY from valid actions
        if random.random() < epsilon:
            return random.choice(valid_actions)
        
        # 2. Exploitation (Q-Network prediction)
        with torch.no_grad():
            state_t = self.preprocess(state)
            q_values = self.policy_net(state_t).squeeze(0) # shape: (4,)
            
            # Create a mask of -infinity for all actions
            mask = torch.full((self.action_dim,), float('-inf')).to(self.device)
            
            # Set valid actions to 0, so they keep their original Q-values
            for action in valid_actions:
                mask[action] = 0.0
                
            # Add mask: Valid Q-values stay the same, invalid become -infinity
            masked_q_values = q_values + mask
            
            # Select the highest VALID Q-value
            return masked_q_values.argmax().item()
    def train_step(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Preprocess all states in batch
        states_t = torch.cat([self.preprocess(s) for s in states])
        next_states_t = torch.cat([self.preprocess(s) for s in next_states])
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.policy_net(states_t).gather(1, actions_t)

        # Max Next Q values (Target)
        next_q = self.target_net(next_states_t).max(1)[0].detach()
        target_q = rewards_t + (1 - dones_t) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())