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

env = Game2048Env()
agent = DQNAgent(state_dim=16, action_dim=4)

for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    epsilon = max(EPS_END, EPS_START - (episode / EPS_DECAY))
    
    while True:
        # Get valid actions BEFORE making a move
        valid_actions = env.get_valid_actions()
        
        # Pass valid actions to the agent
        action = agent.act(state, valid_actions, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        agent.memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        agent.train_step(BATCH_SIZE)
        
        if done:
            break
            
    if episode % TARGET_UPDATE == 0:
        agent.update_target()
        
    if episode % 100 == 0:
        print(f"Episode: {episode}, Score: {info['score']}, Max Tile: {2**np.max(state)}, Epsilon: {epsilon:.2f}")

# Save the model
torch.save(agent.policy_net.state_dict(), "model_2048.pth")