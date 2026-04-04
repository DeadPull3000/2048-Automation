import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Tuple, Dict

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()
        # 4x4 grid
        self.observation_space = spaces.Box(low=0, high=16, shape=(4, 4), dtype=np.int32)
        # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        self.action_space = spaces.Discrete(4)
        self.board = np.zeros((4, 4), dtype=np.int32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.add_random_tile()
        self.add_random_tile()
        return self.get_obs(), {}

    def get_obs(self):
        # We store log2 values to keep the numbers small for the AI
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

    def get_heuristics(self):
        # Convert to log2 to scale down values and prevent extreme gradients
        board_log2 = np.where(self.board > 0, np.log2(self.board), 0)
        
        # 1. Smoothness (penalty for adjacent differences)
        diff_horiz = np.sum(np.abs(board_log2[:, :-1] - board_log2[:, 1:]))
        diff_vert = np.sum(np.abs(board_log2[:-1, :] - board_log2[1:, :]))
        smoothness = -(diff_horiz + diff_vert)
        
        # 2. Monotonicity (measures if tiles increase strictly in one direction)
        mono_up = np.sum(np.maximum(board_log2[:-1, :] - board_log2[1:, :], 0))
        mono_down = np.sum(np.maximum(board_log2[1:, :] - board_log2[:-1, :], 0))
        mono_left = np.sum(np.maximum(board_log2[:, :-1] - board_log2[:, 1:], 0))
        mono_right = np.sum(np.maximum(board_log2[:, 1:] - board_log2[:, :-1], 0))
        monotonicity = max(mono_up, mono_down) + max(mono_left, mono_right)
        
        # 3. Corner Max (check if the highest tile is in one of the 4 corners)
        max_val = np.max(self.board)
        max_pos = np.argwhere(self.board == max_val)
        is_corner = any((r in (0, 3) and c in (0, 3)) for r, c in max_pos)
        
        return smoothness, monotonicity, is_corner

    def step(self, action):
        old_board = self.board.copy()
        old_max = np.max(old_board) if np.any(old_board) else 0
        
        # Execute move
        reward_merge, moved = self.move(action)
        
        if moved:
            self.add_random_tile()
        
        terminated = not self.can_move()
        truncated = False
        
        if not moved:
            reward = -10.0 # Small penalty for invalid moves
        else:
            smoothness, monotonicity, is_corner = self.get_heuristics()
            empty_cells = np.count_nonzero(self.board == 0)
            
            # --- Advanced Reward Shaping Formula ---
            # Scale down merge reward so heuristics aren't ignored
            reward = reward_merge * 0.1 
            reward += empty_cells * 2.0
            reward += monotonicity * 1.0
            reward += smoothness * 0.5  # smoothness is a negative value, so adding it acts as a penalty
            if is_corner:
                reward += 10.0
            
            # Huge Bonus for creating a new max tile
            new_max = np.max(self.board)
            if new_max > old_max:
                reward += np.log2(new_max) * 5.0

        return self.get_obs(), float(reward), terminated, truncated, {"score": np.sum(self.board)}

    def move(self, action):
        # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        rotated_board = np.rot90(self.board, k=action)
        reward = 0
        moved = False
        
        new_board =[]
        for row in rotated_board:
            # 1. Slide
            non_zero = row[row != 0]
            # 2. Merge
            new_row =[]
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
        if np.any(self.board == 0):
            return True
        for r in range(4):
            for c in range(3):
                if self.board[r][c] == self.board[r][c+1] or self.board[c][r] == self.board[c+1][r]:
                    return True
        return False
    
    def get_valid_actions(self):
        """Returns a list of valid action integers (0, 1, 2, 3)"""
        valid_actions =[]
        original_board = self.board.copy()
        
        for action in range(4):
            _, moved = self.move(action)
            if moved:
                valid_actions.append(action)
            # Restore the board after testing the move
            self.board = original_board.copy()
            
        return valid_actions

    def render(self):
        print(self.board)