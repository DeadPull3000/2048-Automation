import numpy as np
from env.game2048_env import Game2048Env
from agents.dqn_agent import DQNAgent
import torch
import time

# --- GPU-Optimized Hyperparameters ---
EPISODES = 15000
BATCH_SIZE = 1024           # Massive batch size for RTX 5080
TRAIN_FREQ = 64             # Collect 64 game steps on CPU before pausing to train
GRADIENT_STEPS = 16         # Do 16 massive gradient updates per training phase
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 10000           # Slow decay to ensure it explores deep game states
TARGET_UPDATE = 50          # Update target network every 50 episodes
BUFFER_SIZE = 100000        # Huge replay buffer
info = {}

env = Game2048Env()
agent = DQNAgent(state_dim=16, action_dim=4, buffer_size=BUFFER_SIZE)

best_max_tile = 0
global_step = 0

print(f"--- Starting Highly Optimized D3QN Training ---")
print(f"Device computing the Neural Net: {agent.device}")

start_time = time.time()

for episode in range(1, EPISODES + 1):
    state, _ = env.reset()
    total_reward = 0
    epsilon = max(EPS_END, EPS_START - (episode / EPS_DECAY))
    
    done = False
    while not done:
        # --- 1. Fast Experience Collection (CPU Bounded) ---
        valid_actions = env.get_valid_actions()
        action = agent.act(state, valid_actions, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        agent.memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        global_step += 1
        
        # --- 2. Decoupled GPU Training Burst (GPU Bounded) ---
        if global_step % TRAIN_FREQ == 0 and len(agent.memory) >= BATCH_SIZE:
            # Blast the RTX 5080 with a burst of large batches
            for _ in range(GRADIENT_STEPS):
                agent.train_step(BATCH_SIZE)
                
    # Update Target Network periodically
    if episode % TARGET_UPDATE == 0:
        agent.update_target()
        
    current_max_tile = 2**np.max(state)
    
    # --- Checkpoint Saving ---
    # Save the model whenever it reaches a new high tile
    if current_max_tile >= best_max_tile and current_max_tile >= 512:
        best_max_tile = current_max_tile
        torch.save(agent.policy_net.state_dict(), "best_model_2048.pth")
        print(f"*** NEW RECORD TILE: {best_max_tile}! Model saved. ***")
        
    # --- Logging ---
    if episode % 100 == 0:
        elapsed = time.time() - start_time
        print(f"Episode: {episode:5d} | Score: {info['score']:5d} | Max Tile: {current_max_tile:4d} | Epsilon: {epsilon:.3f} | Time: {elapsed:.1f}s")
        start_time = time.time() # Reset timer for the next 100 episodes

# Save the final model at the end
torch.save(agent.policy_net.state_dict(), "final_model_2048.pth")
print("Training Complete. Final model saved as final_model_2048.pth")