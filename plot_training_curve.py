"""
Reconstruct training curve by evaluating checkpoints.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

import config
from environment import ABCDEnvironment
from agent import MetaRLAgent
from utils import get_training_configs, get_eval_configs, set_seed


def evaluate_checkpoint(checkpoint_path, configs, num_sessions=20):
    """
    Evaluate a checkpoint on multiple sessions.

    Returns mean reward per session.
    """
    agent = MetaRLAgent(device='cpu')
    agent.load(checkpoint_path)
    agent.eval_mode()

    rewards = []

    for i in range(num_sessions):
        # Random config
        cfg = configs[i % len(configs)]
        env = ABCDEnvironment(cfg)
        obs = env.reset()
        agent.reset_hidden()

        session_reward = 0
        for step in range(config.SESSION_LENGTH):
            action, _, _ = agent.act(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            session_reward += reward
            if done:
                break

        rewards.append(session_reward)

    return np.mean(rewards), np.std(rewards)


def main():
    set_seed(42)

    # Get checkpoints
    checkpoint_files = sorted(glob('models/checkpoint_epoch_*.pt'))

    # Extract epoch numbers
    epochs = []
    for f in checkpoint_files:
        epoch = int(f.split('_')[-1].replace('.pt', ''))
        epochs.append(epoch)

    print(f"Found {len(checkpoint_files)} checkpoints")
    print(f"Epochs: {epochs[0]} to {epochs[-1]}")

    # Get configs for evaluation
    train_configs = get_training_configs()
    eval_configs = get_eval_configs()

    # Evaluate each checkpoint
    train_rewards = []
    train_stds = []
    eval_rewards = []
    eval_stds = []

    print("\nEvaluating checkpoints...")
    for ckpt_path in tqdm(checkpoint_files):
        # Evaluate on training configs
        mean_r, std_r = evaluate_checkpoint(ckpt_path, train_configs, num_sessions=30)
        train_rewards.append(mean_r)
        train_stds.append(std_r)

        # Evaluate on held-out configs
        mean_r, std_r = evaluate_checkpoint(ckpt_path, eval_configs, num_sessions=30)
        eval_rewards.append(mean_r)
        eval_stds.append(std_r)

    # Convert to arrays
    epochs = np.array(epochs)
    train_rewards = np.array(train_rewards)
    train_stds = np.array(train_stds)
    eval_rewards = np.array(eval_rewards)
    eval_stds = np.array(eval_stds)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Raw rewards with confidence bands
    ax1 = axes[0]
    ax1.fill_between(epochs/1000, train_rewards - train_stds, train_rewards + train_stds,
                     alpha=0.3, color='blue')
    ax1.plot(epochs/1000, train_rewards, 'b-', linewidth=2, label='Train configs')

    ax1.fill_between(epochs/1000, eval_rewards - eval_stds, eval_rewards + eval_stds,
                     alpha=0.3, color='orange')
    ax1.plot(epochs/1000, eval_rewards, 'o-', color='orange', linewidth=2, label='Held-out configs')

    ax1.set_xlabel('Training Epoch (thousands)')
    ax1.set_ylabel('Mean Reward per Session')
    ax1.set_title('Learning Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Rolling average
    ax2 = axes[1]

    # Compute rolling average (window of 5 checkpoints = 50k epochs)
    window = 3
    if len(train_rewards) >= window:
        train_rolling = np.convolve(train_rewards, np.ones(window)/window, mode='valid')
        eval_rolling = np.convolve(eval_rewards, np.ones(window)/window, mode='valid')
        epochs_rolling = epochs[window-1:]

        ax2.plot(epochs_rolling/1000, train_rolling, 'b-', linewidth=2.5, label='Train (rolling avg)')
        ax2.plot(epochs_rolling/1000, eval_rolling, '-', color='orange', linewidth=2.5, label='Held-out (rolling avg)')

    # Also plot raw points faded
    ax2.scatter(epochs/1000, train_rewards, c='blue', alpha=0.3, s=30)
    ax2.scatter(epochs/1000, eval_rewards, c='orange', alpha=0.3, s=30)

    ax2.set_xlabel('Training Epoch (thousands)')
    ax2.set_ylabel('Mean Reward per Session')
    ax2.set_title('Learning Curve (Rolling Average)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/training_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: figures/training_curve.png")

    # Also save the data
    np.savez('results/training_curve_data.npz',
             epochs=epochs,
             train_rewards=train_rewards,
             train_stds=train_stds,
             eval_rewards=eval_rewards,
             eval_stds=eval_stds)
    print("Saved: results/training_curve_data.npz")

    # Print summary
    print(f"\nTraining Summary:")
    print(f"  Initial reward (epoch {epochs[0]}): {train_rewards[0]:.2f}")
    print(f"  Final reward (epoch {epochs[-1]}): {train_rewards[-1]:.2f}")
    print(f"  Peak reward: {train_rewards.max():.2f} at epoch {epochs[train_rewards.argmax()]}")
    print(f"  Final eval reward: {eval_rewards[-1]:.2f}")


if __name__ == '__main__':
    main()
