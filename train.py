"""
Training loop for Meta-RL ABCD task.

Uses Advantage Actor-Critic (A2C) to train the GRU-based agent
across multiple ABCD configurations.
"""

import os
import numpy as np
import torch
from tqdm import tqdm
import argparse
from collections import deque

import config
from environment import ABCDEnvironment
from agent import MetaRLAgent
from utils import get_training_configs, ensure_dir, set_seed


def run_session(agent, env):
    """
    Run a single session (100 steps) collecting experience.

    Args:
        agent: MetaRLAgent instance
        env: ABCDEnvironment instance

    Returns:
        rollout: Dictionary of collected experience
        stats: Session statistics
    """
    # Initialize session
    obs = env.reset()
    agent.reset_hidden()

    # Storage for rollout
    observations = []
    actions = []
    rewards = []
    values = []
    log_probs = []

    # Run session
    for step in range(config.SESSION_LENGTH):
        observations.append(obs)

        # Get action from agent
        action, log_prob, value = agent.act(obs)

        # Store
        actions.append(action)
        log_probs.append(log_prob.squeeze())
        values.append(value.squeeze().item())

        # Environment step
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
    # Setup
    set_seed(args.seed)
    ensure_dir(config.MODEL_DIR)

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Training on {device}")

    # Get training configurations
    train_configs = get_training_configs()
    print(f"Generated {len(train_configs)} training configurations")

    # Initialize agent
    agent = MetaRLAgent(device=device)
    agent.train_mode()

    # Training metrics
    loss_history = deque(maxlen=1000)
    reward_history = deque(maxlen=1000)
    success_history = {
        'A_B': deque(maxlen=1000),
        'B_C': deque(maxlen=1000),
        'C_D': deque(maxlen=1000),
        'D_A': deque(maxlen=1000),
    }

    # Progress bar
    pbar = tqdm(range(args.num_epochs), desc="Training")

    for epoch in pbar:
        # Sample a random configuration
        config_idx = np.random.randint(len(train_configs))
        abcd_config = train_configs[config_idx]

        # Create environment
        env = ABCDEnvironment(abcd_config)

        # Run session and collect experience
        rollout, stats = run_session(agent, env)

        # Update agent
        losses = agent.update(rollout)

        # Record metrics
        loss_history.append(losses['loss'])
        reward_history.append(stats['total_reward'])

        transition_names = ['A_B', 'B_C', 'C_D', 'D_A']
        for i, name in enumerate(transition_names):
            attempts = stats['transition_attempts'][i]
            successes = stats['transition_successes'][i]
            if attempts > 0:
                success_history[name].append(successes / attempts)

        # Logging
        if (epoch + 1) % config.LOG_INTERVAL == 0:
            avg_loss = np.mean(loss_history) if loss_history else 0
            avg_reward = np.mean(reward_history) if reward_history else 0
            avg_entropy = losses['entropy']

            success_rates = {}
            for name in transition_names:
                if success_history[name]:
                    success_rates[name] = np.mean(success_history[name])
                else:
                    success_rates[name] = 0

            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'reward': f'{avg_reward:.1f}',
                'entropy': f'{avg_entropy:.3f}',
                'D→A': f'{success_rates["D_A"]:.2f}',
            })

            # Detailed logging every 10 log intervals
            if (epoch + 1) % (config.LOG_INTERVAL * 10) == 0:
                print(f"\nEpoch {epoch + 1}:")
                print(f"  Avg Loss: {avg_loss:.4f}")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  Policy Loss: {losses['policy_loss']:.4f}")
                print(f"  Value Loss: {losses['value_loss']:.4f}")
                print(f"  Entropy: {avg_entropy:.4f}")
                print(f"  Success Rates:")
                for name in transition_names:
                    print(f"    {name}: {success_rates[name]:.3f}")
                print()

        # Checkpointing
        if (epoch + 1) % config.CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(
                config.MODEL_DIR,
                f'checkpoint_epoch_{epoch + 1}.pt'
            )
            agent.save(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = os.path.join(config.MODEL_DIR, 'final_model.pt')
    agent.save(final_path)
    print(f"\nTraining complete! Final model saved to: {final_path}")

    return agent


def main():
    parser = argparse.ArgumentParser(description='Train Meta-RL ABCD agent')
    parser.add_argument('--num-epochs', type=int, default=config.NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        agent = MetaRLAgent()
        agent.load(args.resume)
        # Note: Would need to track epoch number in checkpoint for proper resume
        print("Note: Epoch count restarts from 0 when resuming")

    train(args)


if __name__ == '__main__':
    main()
