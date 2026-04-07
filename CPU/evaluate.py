import torch
import numpy as np
from CPU.env.game2048_env import Game2048Env
from CPU.agents.dqn_agent import DQNAgent

def evaluate(model_path, num_games=10):
    env = Game2048Env()
    agent = DQNAgent()
    info = {}
    
    # Load the trained model weights securely
    try:
        agent.policy_net.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"No model found at {model_path}. Evaluating untrained agent.")
        
    agent.policy_net.eval()

    scores = []
    max_tiles =[]

    for i in range(num_games):
        state, _ = env.reset()
        done = False
        while not done:
            # 1. Get valid actions
            valid_actions = env.get_valid_actions()
            
            # 2. Pass valid actions to act() with epsilon=0.0 (pure greedy)
            action = agent.act(state, valid_actions, epsilon=0.0)
            
            # 3. Take step
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        scores.append(info['score'])
        max_tiles.append(2**np.max(state))
        print(f"Game {i+1}: Score: {info['score']}, Max Tile: {max_tiles[-1]}")

    print(f"\n--- Evaluation Results ({num_games} Games) ---")
    print(f"Average Score: {np.mean(scores)}")
    print(f"Highest Tile reached: {np.max(max_tiles)}")

if __name__ == "__main__":
    evaluate("model_2048.pth", num_games=10) 