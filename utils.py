"""
Utility functions for Meta-RL ABCD task.
Includes shortest path computation and configuration generation.
"""

import numpy as np
from collections import deque
from itertools import permutations
import random
import os

import config


def pos_to_coords(pos):
    """Convert position index to (row, col) coordinates."""
    return pos // config.GRID_SIZE, pos % config.GRID_SIZE


def coords_to_pos(row, col):
    """Convert (row, col) coordinates to position index."""
    return row * config.GRID_SIZE + col


def get_neighbors(pos):
    """Get valid neighboring positions for a given position."""
    row, col = pos_to_coords(pos)
    neighbors = []

    # Up
    if row > 0:
        neighbors.append((config.ACTION_UP, coords_to_pos(row - 1, col)))
    # Down
    if row < config.GRID_SIZE - 1:
        neighbors.append((config.ACTION_DOWN, coords_to_pos(row + 1, col)))
    # Left
    if col > 0:
        neighbors.append((config.ACTION_LEFT, coords_to_pos(row, col - 1)))
    # Right
    if col < config.GRID_SIZE - 1:
        neighbors.append((config.ACTION_RIGHT, coords_to_pos(row, col + 1)))

    return neighbors


def compute_shortest_paths():
    """
    Pre-compute shortest paths between all position pairs.
    Returns a dict mapping (start, end) to list of valid first actions.
    """
    shortest_paths = {}

    for start in range(config.NUM_POSITIONS):
        for end in range(config.NUM_POSITIONS):
            if start == end:
                shortest_paths[(start, end)] = []
                continue

            # BFS to find all shortest paths
            queue = deque([(start, [])])
            visited = {start: 0}  # position -> min distance
            valid_first_actions = set()
            min_dist = float('inf')

            while queue:
                pos, path = queue.popleft()

                if len(path) > min_dist:
                    break

                if pos == end:
                    min_dist = len(path)
                    if path:
                        valid_first_actions.add(path[0])
                    continue

                for action, next_pos in get_neighbors(pos):
                    new_dist = len(path) + 1
                    if next_pos not in visited or visited[next_pos] >= new_dist:
                        visited[next_pos] = new_dist
                        queue.append((next_pos, path + [action]))

            shortest_paths[(start, end)] = list(valid_first_actions)

    return shortest_paths


def compute_manhattan_distance(pos1, pos2):
    """Compute Manhattan distance between two positions."""
    r1, c1 = pos_to_coords(pos1)
    r2, c2 = pos_to_coords(pos2)
    return abs(r1 - r2) + abs(c1 - c2)


def generate_abcd_configs(num_configs, seed):
    """
    Generate ABCD configurations (4 unique positions from 9).
    Each configuration is a tuple of 4 positions (A, B, C, D).
    """
    rng = np.random.RandomState(seed)
    all_positions = list(range(config.NUM_POSITIONS))

    configs = set()
    attempts = 0
    max_attempts = num_configs * 100

    while len(configs) < num_configs and attempts < max_attempts:
        positions = tuple(rng.choice(all_positions, size=4, replace=False))
        configs.add(positions)
        attempts += 1

    if len(configs) < num_configs:
        raise ValueError(f"Could not generate {num_configs} unique configs")

    return list(configs)


def get_training_configs():
    """Get fixed training configurations."""
    return generate_abcd_configs(config.NUM_TRAINING_CONFIGS, config.TRAIN_CONFIG_SEED)


def get_eval_configs():
    """Get held-out evaluation configurations."""
    # Use different seed to ensure no overlap
    eval_configs = generate_abcd_configs(
        config.NUM_EVAL_CONFIGS + config.NUM_TRAINING_CONFIGS,
        config.EVAL_CONFIG_SEED
    )

    train_configs = set(get_training_configs())
    eval_configs = [c for c in eval_configs if c not in train_configs]

    return eval_configs[:config.NUM_EVAL_CONFIGS]


def action_to_delta(action):
    """Convert action to position delta (drow, dcol)."""
    if action == config.ACTION_UP:
        return -1, 0
    elif action == config.ACTION_DOWN:
        return 1, 0
    elif action == config.ACTION_LEFT:
        return 0, -1
    elif action == config.ACTION_RIGHT:
        return 0, 1
    else:
        raise ValueError(f"Invalid action: {action}")


def action_to_string(action):
    """Convert action to human-readable string."""
    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
    return action_names[action]


def one_hot(index, size):
    """Create one-hot encoded vector."""
    vec = np.zeros(size, dtype=np.float32)
    if index >= 0 and index < size:
        vec[index] = 1.0
    return vec


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def visualize_grid(positions=None, current_pos=None, labels=None):
    """
    Create ASCII visualization of the grid.

    Args:
        positions: Optional list of marked positions
        current_pos: Current agent position
        labels: Optional labels for positions (e.g., ['A', 'B', 'C', 'D'])
    """
    grid = [['.' for _ in range(config.GRID_SIZE)] for _ in range(config.GRID_SIZE)]

    if positions and labels:
        for pos, label in zip(positions, labels):
            row, col = pos_to_coords(pos)
            grid[row][col] = label

    if current_pos is not None:
        row, col = pos_to_coords(current_pos)
        if grid[row][col] == '.':
            grid[row][col] = 'X'
        else:
            grid[row][col] = f'[{grid[row][col]}]'

    lines = []
    for row in grid:
        lines.append(' '.join(str(c) for c in row))

    return '\n'.join(lines)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
