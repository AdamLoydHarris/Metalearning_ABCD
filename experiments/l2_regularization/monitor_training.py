#!/usr/bin/env python
"""
Live monitoring of training progress.

Watches for new checkpoints and updates a plot of reward vs epoch.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from . import config
from .agent import MetaRLAgent
from environment import ABCDEnvironment
from utils import generate_abcd_configs, set_seed


def get_checkpoint_epochs():
    """Get list of checkpoint epochs that exist."""
    checkpoint_files = glob(os.path.join(config.MODEL_DIR, 'checkpoint_epoch_*.pt'))
    epochs = []
    for f in checkpoint_files:
        try:
            epoch = int(os.path.basename(f).split('_')[-1].replace('.pt', ''))
            epochs.append(epoch)
        except:
            pass
    return sorted(epochs)


def evaluate_checkpoint(checkpoint_path, configs, num_sessions=20):
    """Evaluate a checkpoint and return mean reward."""
    agent = MetaRLAgent(device='cpu')
    agent.load(checkpoint_path)
    agent.eval_mode()

    rewards = []
    for i in range(num_sessions):
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


def update_plot(epochs, rewards, stds, save_path):
    """Update and save the training curve plot."""
    plt.figure(figsize=(10, 6))

    plt.fill_between(np.array(epochs)/1000,
                     np.array(rewards) - np.array(stds),
                     np.array(rewards) + np.array(stds),
                     alpha=0.3, color='#1f77b4')
    plt.plot(np.array(epochs)/1000, rewards, 'o-', color='#1f77b4',
             linewidth=2, markersize=6)

    plt.xlabel('Training Epoch (thousands)', fontsize=12)
    plt.ylabel('Mean Reward per Session', fontsize=12)
    plt.title(f'L2 Regularization Training (weight_decay={config.WEIGHT_DECAY})\nLast update: {epochs[-1]:,} epochs', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 510)
    plt.ylim(0, 50)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    set_seed(42)

    # Get evaluation configs
    eval_configs = generate_abcd_configs(40, config.EVAL_CONFIG_SEED)

    # Storage for results
    results_file = os.path.join(config.RESULTS_DIR, 'live_training_curve.npz')
    plot_file = os.path.join(config.FIGURES_DIR, 'training_curve_live.png')

    # Load existing results if any
    if os.path.exists(results_file):
        data = np.load(results_file)
        evaluated_epochs = list(data['epochs'])
        rewards = list(data['rewards'])
        stds = list(data['stds'])
        print(f"Loaded {len(evaluated_epochs)} existing evaluations")
    else:
        evaluated_epochs = []
        rewards = []
        stds = []

    print("Monitoring training progress...")
    print(f"Plot will be saved to: {plot_file}")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            # Get current checkpoints
            current_epochs = get_checkpoint_epochs()

            # Find new checkpoints to evaluate
            new_epochs = [e for e in current_epochs if e not in evaluated_epochs]

            if new_epochs:
                for epoch in sorted(new_epochs):
                    checkpoint_path = os.path.join(config.MODEL_DIR, f'checkpoint_epoch_{epoch}.pt')
                    print(f"Evaluating epoch {epoch:,}...", end=' ', flush=True)

                    try:
                        mean_r, std_r = evaluate_checkpoint(checkpoint_path, eval_configs, num_sessions=20)
                        evaluated_epochs.append(epoch)
                        rewards.append(mean_r)
                        stds.append(std_r)
                        print(f"reward = {mean_r:.1f} ± {std_r:.1f}")

                        # Sort by epoch
                        sort_idx = np.argsort(evaluated_epochs)
                        evaluated_epochs = [evaluated_epochs[i] for i in sort_idx]
                        rewards = [rewards[i] for i in sort_idx]
                        stds = [stds[i] for i in sort_idx]

                        # Save results
                        np.savez(results_file,
                                epochs=np.array(evaluated_epochs),
                                rewards=np.array(rewards),
                                stds=np.array(stds))

                        # Update plot
                        update_plot(evaluated_epochs, rewards, stds, plot_file)
                        print(f"  Plot updated: {plot_file}")

                    except Exception as e:
                        print(f"Error: {e}")

            # Wait before checking again
            time.sleep(30)

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        if evaluated_epochs:
            print(f"Final results saved to: {results_file}")
            print(f"Final plot saved to: {plot_file}")


if __name__ == '__main__':
    main()
