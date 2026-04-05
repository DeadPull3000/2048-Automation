import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQNNetwork(nn.Module):
    def __init__(self, input_channels=16, num_actions=4):
        super(DuelingDQNNetwork, self).__init__()
        
        # --- Feature Layer (CNN) ---
        # 16 input channels (one for each tile value up to 2^15)
        # 2x2 kernels are perfect for 2048 because tiles merge in pairs
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=2)
        
        # --- Value Stream ---
        # Calculates V(s): The overall value of the board state
        self.value_fc = nn.Linear(64 * 2 * 2, 256)
        self.value = nn.Linear(256, 1)
        
        # --- Advantage Stream ---
        # Calculates A(s, a): The advantage of each specific action (Up, Down, Left, Right)
        self.adv_fc = nn.Linear(64 * 2 * 2, 256)
        self.advantage = nn.Linear(256, num_actions)

    def forward(self, x):
        # 1. Shared Feature Extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1) # Flatten for Dense layers
        
        # 2. Value Stream Calculation
        val = F.relu(self.value_fc(x))
        val = self.value(val)
        
        # 3. Advantage Stream Calculation
        adv = F.relu(self.adv_fc(x))
        adv = self.advantage(adv)
        
        # 4. Recombine: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        # Subtracting the mean forces the network to actually learn the relative advantage
        q_values = val + adv - adv.mean(dim=1, keepdim=True)
        
        return q_values