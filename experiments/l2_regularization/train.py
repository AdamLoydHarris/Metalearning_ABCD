"""
Training loop for L2 Regularization Experiment.

Uses Advantage Actor-Critic (A2C) with weight decay.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
from tqdm import tqdm
import argparse
from collections import deque

# Local imports (with L2 config)
from . import config
from .agent import MetaRLAgent

# Parent imports (shared code)
from environment import ABCDEnvironment
from utils import generate_abcd_configs, ensure_dir, set_seed


def get_training_configs():
    """Get fixed training configurations."""
    return generate_abcd_configs(config.NUM_TRAINING_CONFIGS, config.TRAIN_CONFIG_SEED)


def run_session(agent, env):
    """Run a single session collecting experience."""
    obs = env.reset()
    agent.reset_hidden()

    observations = []
    actions = []
    rewards = []
    values = []
    log_probs = []

    for step in range(config.SESSION_LENGTH):
        observations.append(obs)
        action, log_prob, value = agent.act(obs)

        actions.append(action)
        log_probs.append(log_prob.squeeze())
        values.append(value.squeeze().item())

        obs, reward, done, info = env.step(action)
        rewards.append(reward)

        if done:
            break

    rollout = {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'values': np.array(values),
        'log_probs': log_probs,
    }

    stats = env.get_statistics()
    stats['total_steps'] = len(actions)

    return rollout, stats


def train(args):
    """Main training loop."""
    set_seed(args.seed)
    ensure_dir(config.MODEL_DIR)
    ensure_dir(config.RESULTS_DIR)
    ensure_dir(config.FIGURES_DIR)

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Training on {device}")
    print(f"L2 Regularization (weight_decay) = {config.WEIGHT_DECAY}")

    train_configs = get_training_configs()
    print(f"Generated {len(train_configs)} training configurations")

    # Initialize agent with L2 regularization
    agent = MetaRLAgent(device=device, weight_decay=config.WEIGHT_DECAY)
    agent.train_mode()

    # Training metrics
    loss_history = deque(maxlen=1000)
    reward_history = deque(maxlen=1000)

    # For saving training curve
    checkpoint_rewards = []
    checkpoint_epochs = []

    pbar = tqdm(range(args.num_epochs), desc="Training")

    for epoch in pbar:
        config_idx = np.random.randint(len(train_configs))
        abcd_config = train_configs[config_idx]

        env = ABCDEnvironment(abcd_config)
        rollout, stats = run_session(agent, env)
        losses = agent.update(rollout)

        loss_history.append(losses['loss'])
        reward_history.append(stats['total_reward'])

        if (epoch + 1) % config.LOG_INTERVAL == 0:
            avg_loss = np.mean(loss_history) if loss_history else 0
            avg_reward = np.mean(reward_history) if reward_history else 0

            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'reward': f'{avg_reward:.1f}',
                'entropy': f'{losses["entropy"]:.3f}',
            })

        if (epoch + 1) % config.CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(
                config.MODEL_DIR,
                f'checkpoint_epoch_{epoch + 1}.pt'
            )
            agent.save(checkpoint_path)

            # Record for training curve
            checkpoint_epochs.append(epoch + 1)
            checkpoint_rewards.append(np.mean(reward_history))

            if (epoch + 1) % (config.CHECKPOINT_INTERVAL * 10) == 0:
                print(f"\nEpoch {epoch + 1}: Avg Reward = {avg_reward:.2f}")

    # Save final model
    final_path = os.path.join(config.MODEL_DIR, 'final_model.pt')
    agent.save(final_path)
    print(f"\nTraining complete! Final model saved to: {final_path}")

    # Save training curve data
    np.savez(os.path.join(config.RESULTS_DIR, 'training_history.npz'),
             epochs=np.array(checkpoint_epochs),
             rewards=np.array(checkpoint_rewards),
             weight_decay=config.WEIGHT_DECAY)
    print(f"Training history saved to: {config.RESULTS_DIR}/training_history.npz")

    return agent


def main():
    parser = argparse.ArgumentParser(description='Train Meta-RL ABCD agent with L2 regularization')
    parser.add_argument('--num-epochs', type=int, default=config.NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU training')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='Override weight decay from config')

    args = parser.parse_args()

    if args.weight_decay is not None:
        config.WEIGHT_DECAY = args.weight_decay
        print(f"Overriding weight_decay to {args.weight_decay}")

    train(args)


if __name__ == '__main__':
    main()
