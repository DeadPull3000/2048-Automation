import numpy as np
from env.game2048_env import Game2048Env
from agents.dqn_agent import DQNAgent
import torch

# Hyperparameters
EPISODES = 5000
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 2000 # Over how many episodes to decay
TARGET_UPDATE = 10

import numpy as np
from env.game2048_env import Game2048Env
from agents.dqn_agent import DQNAgent
import torch

# --- Upgraded Hyperparameters ---
EPISODES = 15000           # Deeper training run
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01             # Let it become almost purely greedy at the end
EPS_DECAY = 10000          # Decay over 10,000 episodes (longer exploration phase)
TARGET_UPDATE = 20         # Update target net slightly less frequently for stability
BUFFER_SIZE = 50000        # Larger memory to remember rare big merges

env = Game2048Env()
agent = DQNAgent(state_dim=16, action_dim=4, buffer_size=BUFFER_SIZE)

best_max_tile = 0

print("Starting Dueling Double DQN Training...")

for episode in range(1, EPISODES + 1):
    state, _ = env.reset()
    total_reward = 0
    
    # Linear epsilon decay
    epsilon = max(EPS_END, EPS_START - (episode / EPS_DECAY))
    
    while True:
        # 1. Get valid actions
        valid_actions = env.get_valid_actions()
        
        # 2. Pick action
        action = agent.act(state, valid_actions, epsilon)
        
        # 3. Step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 4. Push to memory
        agent.memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        # 5. Train the network
        agent.train_step(BATCH_SIZE)
        
        if done:
            break
            
    # Update Target Network periodically
    if episode % TARGET_UPDATE == 0:
        agent.update_target()
        
    current_max_tile = 2**np.max(state)
    
    # --- Checkpoint Saving ---
    # Save the model whenever it reaches a new high tile (512 or above)
    if current_max_tile >= best_max_tile and current_max_tile >= 512:
        best_max_tile = current_max_tile
        torch.save(agent.policy_net.state_dict(), "best_model_2048.pth")
        
    # Logging
    if episode % 100 == 0:
        print(f"Episode: {episode}, Score: {info['score']}, Max Tile: {current_max_tile}, Epsilon: {epsilon:.3f}")

# Save the final model at the end
torch.save(agent.policy_net.state_dict(), "final_model_2048.pth")
print("Training Complete. Final model saved as final_model_2048.pth")