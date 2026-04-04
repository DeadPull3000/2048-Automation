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
        
        # Execute move
        reward, moved = self.move(action)
        
        if moved:
            self.add_random_tile()
        
        terminated = not self.can_move()
        truncated = False # Not used here
        
        # Reward Shaping:
        # 1. Score from merges (already in 'reward')
        # 2. Penalty for invalid moves (if it didn't move)
        if not moved:
            reward = -10
        else:
            # 3. Bonus for empty cells (encourages keeping board clean)
            reward += np.count_nonzero(self.board == 0) * 2

        return self.get_obs(), float(reward), terminated, truncated, {"score": np.sum(self.board)}

    def move(self, action):
        # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        rotated_board = np.rot90(self.board, k=action)
        reward = 0
        moved = False
        
        new_board = []
        for row in rotated_board:
            # 1. Slide
            non_zero = row[row != 0]
            # 2. Merge
            new_row = []
            skip = False
            for i in range(len(non_zero)):
                if skip:
                    skip = False
                    continue
                if i + 1 < len(non_zero) and non_zero[i] == non_zero[i+1]:
                    new_val = non_zero[i] * 2
                    new_row.append(new_val)
                    reward += new_val
                    skip = True
                    moved = True
                else:
                    new_row.append(non_zero[i])
            # 3. Pad with zeros
            while len(new_row) < 4:
                new_row.append(0)
            
            if not np.array_equal(row, new_row):
                moved = True
            new_board.append(new_row)
            
        self.board = np.rot90(np.array(new_board), k=-action)
        return reward, moved

    def can_move(self):
        # Check if any move is possible
        if np.any(self.board == 0):
            return True
        for r in range(4):
            for c in range(3):
                if self.board[r][c] == self.board[r][c+1] or self.board[c][r] == self.board[c+1][r]:
                    return True
        return False

    def render(self):
        print(self.board)