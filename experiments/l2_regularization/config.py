"""
Configuration for L2 Regularization Experiment.

This experiment tests whether L2 regularization (weight decay) affects
the learned representations in the GRU meta-RL agent.
"""

import os

# Paths - relative to this experiment folder
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(EXPERIMENT_DIR, 'models')
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, 'results')
FIGURES_DIR = os.path.join(EXPERIMENT_DIR, 'figures')

# Grid environment
GRID_SIZE = 3
NUM_POSITIONS = GRID_SIZE * GRID_SIZE  # 9
NUM_ACTIONS = 4  # up, down, left, right
SEQUENCE_LENGTH = 4  # A, B, C, D

# Actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

# Agent architecture
GRU_HIDDEN_SIZE = 128
INPUT_DIM = NUM_POSITIONS + NUM_ACTIONS + 1  # position + last_action + last_reward = 14

# Training hyperparameters
LEARNING_RATE = 3e-4
GAMMA = 0.99  # Discount factor
ENTROPY_COEF = 0.01  # Entropy bonus coefficient
VALUE_LOSS_COEF = 0.5  # Value loss coefficient
MAX_GRAD_NORM = 0.5  # Gradient clipping

# === L2 REGULARIZATION (KEY DIFFERENCE) ===
WEIGHT_DECAY = 1e-4  # L2 regularization strength
# Try different values: 1e-5 (weak), 1e-4 (moderate), 1e-3 (strong)

# Session parameters
SESSION_LENGTH = 100  # Steps per session
NUM_EPOCHS = 500000  # Total training epochs

# Configuration generation
NUM_TRAINING_CONFIGS = 100
NUM_EVAL_CONFIGS = 40
TRAIN_CONFIG_SEED = 42
EVAL_CONFIG_SEED = 123

# Rewards
REWARD_CORRECT = 1.0

# Logging
LOG_INTERVAL = 100
CHECKPOINT_INTERVAL = 10000
