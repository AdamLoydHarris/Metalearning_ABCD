"""
Configuration and hyperparameters for Meta-RL ABCD task.
"""

# Grid Configuration
GRID_SIZE = 3  # 3x3 grid
NUM_POSITIONS = GRID_SIZE * GRID_SIZE  # 9 positions

# Action Space
NUM_ACTIONS = 4  # up, down, left, right
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

# Input Dimensions
POSITION_DIM = NUM_POSITIONS  # 9 (one-hot)
ACTION_DIM = NUM_ACTIONS  # 4 (one-hot)
REWARD_DIM = 1  # scalar
INPUT_DIM = POSITION_DIM + ACTION_DIM + REWARD_DIM  # 14

# Agent Architecture
GRU_HIDDEN_SIZE = 128
ACTOR_OUTPUT_DIM = NUM_ACTIONS  # 4
CRITIC_OUTPUT_DIM = 1

# ABCD Task
SEQUENCE_LENGTH = 4  # A, B, C, D

# Training Configuration
NUM_TRAINING_CONFIGS = 100  # Number of fixed training configurations
NUM_EVAL_CONFIGS = 40  # Number of held-out evaluation configurations
SESSION_LENGTH = 100  # Steps per session
NUM_EPOCHS = 500000  # Total training epochs
CHECKPOINT_INTERVAL = 10000  # Save every N epochs
LOG_INTERVAL = 1000  # Print progress every N epochs

# A2C Hyperparameters
LEARNING_RATE = 3e-4
GAMMA = 0.99  # Discount factor
ENTROPY_COEF = 0.01  # Entropy bonus coefficient
VALUE_LOSS_COEF = 0.5  # Value loss coefficient
MAX_GRAD_NORM = 0.5  # Gradient clipping

# Reward Structure
REWARD_CORRECT = 1.0  # Reward for reaching correct next state
REWARD_INCORRECT = 0.0  # No negative rewards

# Random Seeds
TRAIN_CONFIG_SEED = 42  # Seed for generating training configs
EVAL_CONFIG_SEED = 123  # Seed for generating eval configs (different from train)

# Paths
MODEL_DIR = "models"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
