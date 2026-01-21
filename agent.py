"""
GRU-based Actor-Critic Agent for Meta-RL ABCD task.

The agent uses a GRU to capture temporal dependencies within a session,
allowing it to learn the ABCD sequence through its recurrent memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

import config


class GRUActorCritic(nn.Module):
    """
    GRU-based Actor-Critic network.

    Architecture:
        Input: [position_onehot(9), last_action_onehot(4), last_reward(1)] = 14 dims
        GRU: 128 hidden units
        Actor head: Linear → Softmax over 4 actions
        Critic head: Linear → scalar value estimate
    """

    def __init__(self, input_dim=config.INPUT_DIM, hidden_dim=config.GRU_HIDDEN_SIZE,
                 action_dim=config.NUM_ACTIONS):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # GRU layer
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs, hidden):
        """
        Forward pass.

        Args:
            obs: Observation tensor [batch, input_dim] or [batch, seq_len, input_dim]
            hidden: GRU hidden state [1, batch, hidden_dim]

        Returns:
            action_probs: Action probabilities [batch, action_dim]
            value: State value [batch, 1]
            new_hidden: Updated hidden state [1, batch, hidden_dim]
        """
        # Handle single step vs sequence
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # [batch, 1, input_dim]

        # GRU forward
        gru_out, new_hidden = self.gru(obs, hidden)

        # Take last output if sequence
        if gru_out.size(1) > 1:
            gru_out = gru_out[:, -1, :]  # [batch, hidden_dim]
        else:
            gru_out = gru_out.squeeze(1)  # [batch, hidden_dim]

        # Actor and critic heads
        action_logits = self.actor(gru_out)
        action_probs = F.softmax(action_logits, dim=-1)
        value = self.critic(gru_out)

        return action_probs, value, new_hidden

    def get_action(self, obs, hidden, deterministic=False):
        """
        Sample an action from the policy.

        Args:
            obs: Observation tensor [batch, input_dim]
            hidden: GRU hidden state [1, batch, hidden_dim]
            deterministic: If True, return argmax action

        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: State value estimate
            new_hidden: Updated hidden state
        """
        action_probs, value, new_hidden = self.forward(obs, hidden)

        if deterministic:
            action = action_probs.argmax(dim=-1)
            dist = Categorical(action_probs)
            log_prob = dist.log_prob(action)
        else:
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob, value, new_hidden

    def evaluate_actions(self, obs, hidden, actions):
        """
        Evaluate actions for PPO-style updates.

        Args:
            obs: Observation tensor [batch, seq_len, input_dim]
            hidden: Initial hidden state [1, batch, hidden_dim]
            actions: Actions to evaluate [batch, seq_len]

        Returns:
            log_probs: Log probabilities of actions
            values: State values
            entropy: Policy entropy
        """
        batch_size, seq_len, _ = obs.shape

        log_probs = []
        values = []
        entropies = []

        h = hidden
        for t in range(seq_len):
            action_probs, value, h = self.forward(obs[:, t, :], h)
            dist = Categorical(action_probs)

            log_prob = dist.log_prob(actions[:, t])
            entropy = dist.entropy()

            log_probs.append(log_prob)
            values.append(value.squeeze(-1))
            entropies.append(entropy)

        log_probs = torch.stack(log_probs, dim=1)
        values = torch.stack(values, dim=1)
        entropies = torch.stack(entropies, dim=1)

        return log_probs, values, entropies.mean()

    def init_hidden(self, batch_size=1):
        """Initialize hidden state."""
        return torch.zeros(1, batch_size, self.hidden_dim)

    def get_hidden_state(self, hidden):
        """Extract hidden state as numpy array for analysis."""
        return hidden.squeeze(0).detach().cpu().numpy()


class MetaRLAgent:
    """
    Meta-RL Agent wrapper for training and evaluation.
    Handles device management and provides a cleaner interface.
    """

    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.network = GRUActorCritic().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.LEARNING_RATE
        )

        self.hidden = None

    def reset_hidden(self, batch_size=1):
        """Reset hidden state for new session."""
        self.hidden = self.network.init_hidden(batch_size).to(self.device)

    def act(self, obs, deterministic=False):
        """
        Select action given observation.

        Args:
            obs: Numpy observation array
            deterministic: If True, use argmax policy

        Returns:
            action: Integer action
            log_prob: Log probability (tensor)
            value: Value estimate (tensor)
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad() if deterministic else torch.enable_grad():
            action, log_prob, value, new_hidden = self.network.get_action(
                obs_tensor, self.hidden, deterministic=deterministic
            )

        self.hidden = new_hidden

        return action.item(), log_prob, value

    def get_hidden_numpy(self):
        """Get current hidden state as numpy array."""
        return self.network.get_hidden_state(self.hidden)

    def update(self, rollout):
        """
        Update network using A2C algorithm.

        Args:
            rollout: Dictionary containing:
                - observations: [seq_len, obs_dim]
                - actions: [seq_len]
                - rewards: [seq_len]
                - values: [seq_len]
                - log_probs: [seq_len]

        Returns:
            Dictionary of losses
        """
        obs = torch.FloatTensor(rollout['observations']).unsqueeze(0).to(self.device)
        actions = torch.LongTensor(rollout['actions']).unsqueeze(0).to(self.device)
        rewards = torch.FloatTensor(rollout['rewards']).to(self.device)
        old_values = torch.FloatTensor(rollout['values']).to(self.device)
        old_log_probs = torch.stack(rollout['log_probs']).to(self.device)

        # Compute returns and advantages
        returns = self._compute_returns(rewards)
        advantages = returns - old_values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Forward pass
        initial_hidden = self.network.init_hidden(1).to(self.device)
        log_probs, values, entropy = self.network.evaluate_actions(
            obs, initial_hidden, actions
        )

        log_probs = log_probs.squeeze(0)
        values = values.squeeze(0)

        # Policy loss (negative because we want to maximize)
        policy_loss = -(log_probs * advantages.detach()).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Total loss
        loss = (
            policy_loss
            + config.VALUE_LOSS_COEF * value_loss
            - config.ENTROPY_COEF * entropy
        )

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.MAX_GRAD_NORM)
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
        }

    def _compute_returns(self, rewards):
        """Compute discounted returns."""
        returns = torch.zeros_like(rewards)
        running_return = 0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + config.GAMMA * running_return
            returns[t] = running_return

        return returns

    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def eval_mode(self):
        """Set network to evaluation mode."""
        self.network.eval()

    def train_mode(self):
        """Set network to training mode."""
        self.network.train()
