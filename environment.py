"""
3x3 Grid Maze Environment with ABCD Task.

The agent must learn length-4 repeating sequences (A→B→C→D→A...)
within a session through meta-learning.
"""

import numpy as np

import config
from utils import (
    pos_to_coords, coords_to_pos, action_to_delta,
    compute_shortest_paths, one_hot, visualize_grid
)


class ABCDEnvironment:
    """
    3x3 Grid Maze with ABCD sequence task.

    Grid layout (position indices):
        0 1 2
        3 4 5
        6 7 8

    Actions: 0=up, 1=down, 2=left, 3=right
    """

    def __init__(self, abcd_config):
        """
        Initialize environment with an ABCD configuration.

        Args:
            abcd_config: Tuple of 4 positions (A, B, C, D)
        """
        self.abcd_positions = abcd_config  # (A, B, C, D) positions
        self.shortest_paths = compute_shortest_paths()

        # State tracking
        self.current_pos = None
        self.current_sequence_idx = None  # 0=A, 1=B, 2=C, 3=D
        self.last_action = None
        self.last_reward = None
        self.step_count = 0

        # Statistics
        self.transition_attempts = [0, 0, 0, 0]  # Attempts per transition
        self.transition_successes = [0, 0, 0, 0]  # Successes per transition
        self.first_occurrence_success = [None, None, None, None]  # First attempt result

    def reset(self, start_pos=None):
        """
        Reset environment for a new session.

        Args:
            start_pos: Optional starting position (random if None)

        Returns:
            Initial observation
        """
        if start_pos is not None:
            self.current_pos = start_pos
        else:
            self.current_pos = np.random.randint(0, config.NUM_POSITIONS)

        # Determine which sequence state we're at based on position
        if self.current_pos in self.abcd_positions:
            self.current_sequence_idx = self.abcd_positions.index(self.current_pos)
        else:
            # Not at any ABCD position, need to get to A first
            self.current_sequence_idx = 3  # Target is A (next after D)

        self.last_action = -1  # No previous action
        self.last_reward = 0.0
        self.step_count = 0

        # Reset statistics
        self.transition_attempts = [0, 0, 0, 0]
        self.transition_successes = [0, 0, 0, 0]
        self.first_occurrence_success = [None, None, None, None]

        return self._get_observation()

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action: Integer action (0=up, 1=down, 2=left, 3=right)

        Returns:
            observation, reward, done, info
        """
        assert 0 <= action < config.NUM_ACTIONS

        # Get current position coords
        row, col = pos_to_coords(self.current_pos)

        # Apply action
        drow, dcol = action_to_delta(action)
        new_row = row + drow
        new_col = col + dcol

        # Check bounds - stay in place if invalid move
        if 0 <= new_row < config.GRID_SIZE and 0 <= new_col < config.GRID_SIZE:
            self.current_pos = coords_to_pos(new_row, new_col)

        # Check if reached target
        target_idx = (self.current_sequence_idx + 1) % config.SEQUENCE_LENGTH
        target_pos = self.abcd_positions[target_idx]

        reward = 0.0
        reached_target = False

        if self.current_pos == target_pos:
            reward = config.REWARD_CORRECT
            reached_target = True

            # Record transition success
            transition_idx = self.current_sequence_idx  # A→B is transition 0, etc.
            self.transition_attempts[transition_idx] += 1
            self.transition_successes[transition_idx] += 1

            # Record first occurrence
            if self.first_occurrence_success[transition_idx] is None:
                self.first_occurrence_success[transition_idx] = True

            # Update sequence state
            self.current_sequence_idx = target_idx

        # Update state
        self.last_action = action
        self.last_reward = reward
        self.step_count += 1

        # Check if session is done
        done = self.step_count >= config.SESSION_LENGTH

        info = {
            'reached_target': reached_target,
            'target_pos': target_pos,
            'transition_idx': self.current_sequence_idx if reached_target else None,
        }

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        """
        Construct observation vector.

        Returns:
            Observation: [position_onehot(9), last_action_onehot(4), last_reward(1)]
        """
        pos_onehot = one_hot(self.current_pos, config.NUM_POSITIONS)
        action_onehot = one_hot(self.last_action, config.NUM_ACTIONS)
        reward_array = np.array([self.last_reward], dtype=np.float32)

        return np.concatenate([pos_onehot, action_onehot, reward_array])

    def is_optimal_action(self, action, start_pos=None, target_pos=None):
        """
        Check if action is one of the optimal (shortest path) actions.

        Args:
            action: Action taken
            start_pos: Starting position (default: current position before action)
            target_pos: Target position (default: current target)

        Returns:
            True if action follows a shortest path
        """
        if start_pos is None:
            start_pos = self.current_pos
        if target_pos is None:
            target_idx = (self.current_sequence_idx + 1) % config.SEQUENCE_LENGTH
            target_pos = self.abcd_positions[target_idx]

        if start_pos == target_pos:
            return True  # Already at target

        valid_actions = self.shortest_paths.get((start_pos, target_pos), [])
        return action in valid_actions

    def get_current_target(self):
        """Get the current target position."""
        target_idx = (self.current_sequence_idx + 1) % config.SEQUENCE_LENGTH
        return self.abcd_positions[target_idx]

    def get_sequence_state_name(self):
        """Get current sequence state name (A, B, C, or D)."""
        names = ['A', 'B', 'C', 'D']
        return names[self.current_sequence_idx]

    def get_target_name(self):
        """Get target state name."""
        names = ['A', 'B', 'C', 'D']
        target_idx = (self.current_sequence_idx + 1) % config.SEQUENCE_LENGTH
        return names[target_idx]

    def render(self):
        """Render current state as ASCII."""
        print(f"\nStep {self.step_count}")
        print(f"At: {self.get_sequence_state_name()} (pos {self.current_pos})")
        print(f"Target: {self.get_target_name()} (pos {self.get_current_target()})")
        print(visualize_grid(
            positions=self.abcd_positions,
            current_pos=self.current_pos,
            labels=['A', 'B', 'C', 'D']
        ))
        print()

    def get_statistics(self):
        """Get session statistics."""
        return {
            'transition_attempts': self.transition_attempts.copy(),
            'transition_successes': self.transition_successes.copy(),
            'first_occurrence_success': self.first_occurrence_success.copy(),
            'total_reward': sum(self.transition_successes),
        }


class ABCDEnvironmentWithTracking(ABCDEnvironment):
    """
    Extended environment with detailed tracking for analysis.
    Records trajectory and hidden state history.
    """

    def __init__(self, abcd_config):
        super().__init__(abcd_config)
        self.trajectory = []
        self.hidden_states = []

    def reset(self, start_pos=None):
        obs = super().reset(start_pos)
        self.trajectory = [{
            'step': 0,
            'position': self.current_pos,
            'sequence_idx': self.current_sequence_idx,
            'action': None,
            'reward': 0.0,
            'target': self.get_current_target(),
        }]
        self.hidden_states = []
        return obs

    def step(self, action):
        start_pos = self.current_pos
        start_seq_idx = self.current_sequence_idx
        target = self.get_current_target()

        obs, reward, done, info = super().step(action)

        self.trajectory.append({
            'step': self.step_count,
            'position': self.current_pos,
            'sequence_idx': self.current_sequence_idx,
            'action': action,
            'reward': reward,
            'target': target,
            'start_pos': start_pos,
            'start_seq_idx': start_seq_idx,
            'reached_target': info['reached_target'],
        })

        return obs, reward, done, info

    def record_hidden_state(self, hidden_state):
        """Record hidden state for analysis."""
        self.hidden_states.append({
            'step': self.step_count,
            'hidden': hidden_state.copy() if hasattr(hidden_state, 'copy') else hidden_state,
            'position': self.current_pos,
            'sequence_idx': self.current_sequence_idx,
        })

    def get_trajectory(self):
        """Get recorded trajectory."""
        return self.trajectory

    def get_hidden_history(self):
        """Get recorded hidden states."""
        return self.hidden_states
