import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()
        # 4x4 grid
        self.observation_space = spaces.Box(low=0, high=16, shape=(4, 4), dtype=np.int32)
        # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        self.action_space = spaces.Discrete(4)
        self.board = np.zeros((4, 4), dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.add_random_tile()
        self.add_random_tile()
        return self.get_obs(), {}

    def get_obs(self):
        # We store log2 values to keep the numbers small for the AI
        # 0->0, 2->1, 4->2, 8->3, etc.
        temp_board = np.zeros((4, 4), dtype=np.int32)
        for r in range(4):
            for c in range(4):
                if self.board[r][c] > 0:
                    temp_board[r][c] = int(np.log2(self.board[r][c]))
        return temp_board

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            r, c = random.choice(empty_cells)
            self.board[r][c] = 2 if random.random() < 0.9 else 4

    def step(self, action):
        old_board = self.board.copy()
        