"""
Evaluation script for Meta-RL ABCD agent.

Tests trained agent on held-out configurations to assess:
- Success rate per transition type
- Learning curves within sessions
- Critical D→A inference on first occurrence
"""

import os
import numpy as np
import torch
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt

import config
from environment import ABCDEnvironmentWithTracking
from agent import MetaRLAgent
from utils import get_eval_configs, ensure_dir, set_seed


def evaluate_session(agent, env, record_hidden=False):
    """
    Run evaluation session with detailed tracking.

    Args:
        agent: MetaRLAgent instance
        env: ABCDEnvironmentWithTracking instance
        record_hidden: Whether to record hidden states

    Returns:
        stats: Session statistics
        trajectory: Full trajectory
        hidden_states: Hidden state history (if recorded)
    """
    obs = env.reset()
    agent.reset_hidden()

    for step in range(config.SESSION_LENGTH):
        # Record hidden state before action
        if record_hidden:
            hidden = agent.get_hidden_numpy()
            env.record_hidden_state(hidden)

        # Get action (deterministic for evaluation)
        action, _, _ = agent.act(obs, deterministic=True)

        # Step
        obs, reward, done, info = env.step(action)

        if done:
            break

    stats = env.get_statistics()
    trajectory = env.get_trajectory()
    hidden_states = env.get_hidden_history() if record_hidden else None

    return stats, trajectory, hidden_states


def analyze_learning_curve(trajectories, num_bins=5):
    """
    Analyze learning curve within sessions by binning steps.

    Args:
        trajectories: List of trajectories from multiple sessions
        num_bins: Number of bins to divide session into

    Returns:
        Dictionary of success rates per bin for each transition
    """
    bin_size = config.SESSION_LENGTH // num_bins

    results = {
        'A_B': [[] for _ in range(num_bins)],
        'B_C': [[] for _ in range(num_bins)],
        'C_D': [[] for _ in range(num_bins)],
        'D_A': [[] for _ in range(num_bins)],
    }

    transition_names = ['A_B', 'B_C', 'C_D', 'D_A']

    for traj in trajectories:
        for entry in traj:
            if 'reached_target' not in entry:
                continue

            step = entry['step']
            bin_idx = min(step // bin_size, num_bins - 1)

            # Determine which transition this was
            start_seq = entry.get('start_seq_idx', 0)
            name = transition_names[start_seq]

            results[name][bin_idx].append(1 if entry['reached_target'] else 0)

    # Compute averages
    averages = {}
    for name in transition_names:
        averages[name] = []
        for bin_data in results[name]:
            if bin_data:
                averages[name].append(np.mean(bin_data))
            else:
                averages[name].append(np.nan)

    return averages


def evaluate(args):
    """Main evaluation function."""
    set_seed(args.seed)
    ensure_dir(config.RESULTS_DIR)
    ensure_dir(config.FIGURES_DIR)

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Evaluating on {device}")

    # Load agent
    agent = MetaRLAgent(device=device)
    if args.checkpoint:
        agent.load(args.checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        # Try to load final model
        final_path = os.path.join(config.MODEL_DIR, 'final_model.pt')
        if os.path.exists(final_path):
            agent.load(final_path)
            print(f"Loaded final model: {final_path}")
        else:
            print("Warning: No checkpoint found, using untrained agent")

    agent.eval_mode()

    # Get evaluation configurations
    eval_configs = get_eval_configs()
    print(f"Evaluating on {len(eval_configs)} held-out configurations")

    # Collect results
    all_stats = []
    all_trajectories = []
    first_trial_results = {'A_B': [], 'B_C': [], 'C_D': [], 'D_A': []}

    transition_names = ['A_B', 'B_C', 'C_D', 'D_A']

    for i, abcd_config in enumerate(eval_configs):
        env = ABCDEnvironmentWithTracking(abcd_config)
        stats, trajectory, _ = evaluate_session(agent, env, record_hidden=False)

        all_stats.append(stats)
        all_trajectories.append(trajectory)

        # Record first trial results
        for j, name in enumerate(transition_names):
            result = stats['first_occurrence_success'][j]
            if result is not None:
                first_trial_results[name].append(1 if result else 0)

        if args.verbose and (i + 1) % 10 == 0:
            print(f"Evaluated {i + 1}/{len(eval_configs)} configurations")

    # Aggregate results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    # Overall success rates
    print("\nOverall Success Rates (across all sessions):")
    for name in transition_names:
        total_attempts = sum(s['transition_attempts'][transition_names.index(name)]
                           for s in all_stats)
        total_successes = sum(s['transition_successes'][transition_names.index(name)]
                             for s in all_stats)
        rate = total_successes / total_attempts if total_attempts > 0 else 0
        print(f"  {name}: {rate:.3f} ({total_successes}/{total_attempts})")

    # First trial success rates (critical for meta-learning)
    print("\nFirst Trial Success Rates (meta-learning metric):")
    for name in transition_names:
        if first_trial_results[name]:
            rate = np.mean(first_trial_results[name])
            n = len(first_trial_results[name])
            print(f"  {name}: {rate:.3f} (n={n})")
        else:
            print(f"  {name}: N/A")

    # D→A is the critical metric (inference of unseen transition)
    if first_trial_results['D_A']:
        da_rate = np.mean(first_trial_results['D_A'])
        print(f"\n*** Critical D→A First Trial Rate: {da_rate:.3f} ***")

    # Learning curve analysis
    print("\nLearning Curve (success rate by session phase):")
    learning_curves = analyze_learning_curve(all_trajectories)
    phases = ['Early', 'Early-Mid', 'Middle', 'Mid-Late', 'Late']
    for name in transition_names:
        rates = learning_curves[name]
        print(f"  {name}: ", end='')
        for phase, rate in zip(phases, rates):
            if not np.isnan(rate):
                print(f"{phase}={rate:.2f} ", end='')
        print()

    # Average reward
    avg_reward = np.mean([s['total_reward'] for s in all_stats])
    print(f"\nAverage Session Reward: {avg_reward:.2f}")

    # Save results
    results = {
        'all_stats': all_stats,
        'first_trial_results': first_trial_results,
        'learning_curves': learning_curves,
        'avg_reward': avg_reward,
    }

    results_path = os.path.join(config.RESULTS_DIR, 'evaluation_results.npy')
    np.save(results_path, results, allow_pickle=True)
    print(f"\nResults saved to: {results_path}")

    # Generate figures
    if args.plot:
        plot_results(results, first_trial_results, learning_curves)

    return results


def plot_results(results, first_trial_results, learning_curves):
    """Generate evaluation figures."""
    transition_names = ['A_B', 'B_C', 'C_D', 'D_A']

    # Figure 1: First trial success rates
    fig, ax = plt.subplots(figsize=(8, 6))
    rates = [np.mean(first_trial_results[name]) if first_trial_results[name] else 0
             for name in transition_names]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.bar(transition_names, rates, color=colors)
    ax.set_ylabel('Success Rate')
    ax.set_xlabel('Transition')
    ax.set_title('First Trial Success Rate (Meta-Learning Metric)')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.25, color='gray', linestyle='--', label='Chance')
    ax.legend()

    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rate:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'first_trial_success.png'), dpi=150)
    plt.close()

    # Figure 2: Learning curves
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(5)
    width = 0.2

    for i, name in enumerate(transition_names):
        rates = learning_curves[name]
        offset = (i - 1.5) * width
        ax.bar(x + offset, rates, width, label=name, color=colors[i])

    ax.set_ylabel('Success Rate')
    ax.set_xlabel('Session Phase')
    ax.set_title('Learning Curve Within Session')
    ax.set_xticks(x)
    ax.set_xticklabels(['Early', 'Early-Mid', 'Middle', 'Mid-Late', 'Late'])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'learning_curves.png'), dpi=150)
    plt.close()

    print(f"Figures saved to: {config.FIGURES_DIR}/")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Meta-RL ABCD agent')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--seed', type=int, default=456,
                        help='Random seed for evaluation')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU evaluation')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--plot', action='store_true', default=True,
                        help='Generate figures')

    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
