"""
Neural representation analysis for Meta-RL ABCD task.

Analyzes GRU hidden states to understand:
- How task state is represented
- Learning dynamics during sessions
- Representational similarity across configurations
"""

import os
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import config
from environment import ABCDEnvironmentWithTracking
from agent import MetaRLAgent
from utils import get_eval_configs, ensure_dir, set_seed


def collect_hidden_states(agent, configs, num_sessions_per_config=1):
    """
    Collect hidden states from multiple sessions.

    Args:
        agent: MetaRLAgent instance
        configs: List of ABCD configurations
        num_sessions_per_config: Sessions to run per configuration

    Returns:
        Dictionary containing hidden states and metadata
    """
    all_hidden = []
    all_positions = []
    all_sequence_idx = []
    all_steps = []
    all_config_idx = []
    all_rewards = []

    for cfg_idx, abcd_config in enumerate(configs):
        for session in range(num_sessions_per_config):
            env = ABCDEnvironmentWithTracking(abcd_config)
            obs = env.reset()
            agent.reset_hidden()

            for step in range(config.SESSION_LENGTH):
                # Record hidden state
                hidden = agent.get_hidden_numpy()
                all_hidden.append(hidden.flatten())
                all_positions.append(env.current_pos)
                all_sequence_idx.append(env.current_sequence_idx)
                all_steps.append(step)
                all_config_idx.append(cfg_idx)
                all_rewards.append(env.last_reward)

                # Take action
                action, _, _ = agent.act(obs, deterministic=True)
                obs, reward, done, info = env.step(action)

                if done:
                    break

    return {
        'hidden': np.array(all_hidden),
        'position': np.array(all_positions),
        'sequence_idx': np.array(all_sequence_idx),
        'step': np.array(all_steps),
        'config_idx': np.array(all_config_idx),
        'reward': np.array(all_rewards),
    }


def analyze_pca(data, save_dir):
    """Perform PCA analysis and create visualizations."""
    hidden = data['hidden']
    sequence_idx = data['sequence_idx']
    steps = data['step']

    # Fit PCA
    pca = PCA(n_components=min(10, hidden.shape[1]))
    hidden_pca = pca.fit_transform(hidden)

    # Plot variance explained
    fig, ax = plt.subplots(figsize=(8, 5))
    var_explained = pca.explained_variance_ratio_
    ax.bar(range(len(var_explained)), var_explained)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained')
    ax.set_title('PCA Variance Explained')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pca_variance.png'), dpi=150)
    plt.close()

    # Plot hidden states colored by sequence state (A, B, C, D)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    labels = ['At A', 'At B', 'At C', 'At D']

    for i in range(4):
        mask = sequence_idx == i
        ax.scatter(hidden_pca[mask, 0], hidden_pca[mask, 1],
                  c=colors[i], label=labels[i], alpha=0.5, s=10)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('GRU Hidden States by Sequence State (PCA)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pca_by_sequence.png'), dpi=150)
    plt.close()

    # Plot colored by session time (early vs late)
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(hidden_pca[:, 0], hidden_pca[:, 1],
                        c=steps, cmap='viridis', alpha=0.5, s=10)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Step in Session')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('GRU Hidden States by Time in Session (PCA)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pca_by_time.png'), dpi=150)
    plt.close()

    return pca, hidden_pca


def analyze_tsne(data, save_dir, perplexity=30, n_samples=5000):
    """Perform t-SNE analysis and create visualizations."""
    hidden = data['hidden']
    sequence_idx = data['sequence_idx']

    # Subsample for t-SNE (computationally expensive)
    if len(hidden) > n_samples:
        idx = np.random.choice(len(hidden), n_samples, replace=False)
        hidden_sub = hidden[idx]
        seq_idx_sub = sequence_idx[idx]
    else:
        hidden_sub = hidden
        seq_idx_sub = sequence_idx

    print(f"Running t-SNE on {len(hidden_sub)} samples...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    hidden_tsne = tsne.fit_transform(hidden_sub)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    labels = ['At A', 'At B', 'At C', 'At D']

    for i in range(4):
        mask = seq_idx_sub == i
        ax.scatter(hidden_tsne[mask, 0], hidden_tsne[mask, 1],
                  c=colors[i], label=labels[i], alpha=0.6, s=15)

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('GRU Hidden States by Sequence State (t-SNE)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tsne_by_sequence.png'), dpi=150)
    plt.close()

    return hidden_tsne


def decode_goal(data, save_dir):
    """
    Train decoder to predict current goal from hidden state.

    Tests whether the hidden state encodes the target location.
    """
    hidden = data['hidden']
    sequence_idx = data['sequence_idx']

    # Target is next in sequence
    target_idx = (sequence_idx + 1) % 4

    # Train logistic regression decoder
    print("Training goal decoder...")
    clf = LogisticRegression(max_iter=1000, multi_class='multinomial')

    # Cross-validation
    scores = cross_val_score(clf, hidden, target_idx, cv=5)
    mean_acc = np.mean(scores)
    std_acc = np.std(scores)

    print(f"Goal decoding accuracy: {mean_acc:.3f} (+/- {std_acc:.3f})")
    print(f"Chance level: 0.250")

    # Train final model for analysis
    clf.fit(hidden, target_idx)

    # Plot decoding accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(['Decoder', 'Chance'], [mean_acc, 0.25],
           color=['#2ca02c', 'gray'], yerr=[std_acc, 0])
    ax.set_ylabel('Accuracy')
    ax.set_title('Goal Decoding from Hidden State')
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'goal_decoding.png'), dpi=150)
    plt.close()

    return clf, mean_acc


def analyze_trajectory(agent, abcd_config, save_dir, config_name='example'):
    """
    Analyze trajectory through state space during a single session.
    """
    env = ABCDEnvironmentWithTracking(abcd_config)
    obs = env.reset()
    agent.reset_hidden()

    hidden_trajectory = []
    sequence_trajectory = []
    reward_trajectory = []

    for step in range(config.SESSION_LENGTH):
        hidden = agent.get_hidden_numpy()
        hidden_trajectory.append(hidden.flatten())
        sequence_trajectory.append(env.current_sequence_idx)
        reward_trajectory.append(env.last_reward)

        action, _, _ = agent.act(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if done:
            break

    hidden_trajectory = np.array(hidden_trajectory)

    # PCA on this trajectory
    pca = PCA(n_components=3)
    traj_pca = pca.fit_transform(hidden_trajectory)

    # 2D trajectory plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    # Plot trajectory line
    ax.plot(traj_pca[:, 0], traj_pca[:, 1], 'k-', alpha=0.3, linewidth=0.5)

    # Plot points colored by sequence state
    for i, (x, y, seq) in enumerate(zip(traj_pca[:, 0], traj_pca[:, 1], sequence_trajectory)):
        ax.scatter(x, y, c=colors[seq], s=20, alpha=0.7)

    # Mark start and end
    ax.scatter(traj_pca[0, 0], traj_pca[0, 1], marker='*', s=200,
              c='green', edgecolors='black', label='Start', zorder=5)
    ax.scatter(traj_pca[-1, 0], traj_pca[-1, 1], marker='s', s=100,
              c='red', edgecolors='black', label='End', zorder=5)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'Hidden State Trajectory During Session ({config_name})')

    # Create legend for sequence states
    for i, label in enumerate(['A', 'B', 'C', 'D']):
        ax.scatter([], [], c=colors[i], label=f'At {label}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'trajectory_{config_name}.png'), dpi=150)
    plt.close()

    # 3D trajectory plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(traj_pca[:, 0], traj_pca[:, 1], traj_pca[:, 2], 'k-', alpha=0.3, linewidth=0.5)

    for i, (x, y, z, seq) in enumerate(zip(traj_pca[:, 0], traj_pca[:, 1],
                                           traj_pca[:, 2], sequence_trajectory)):
        ax.scatter(x, y, z, c=colors[seq], s=20, alpha=0.7)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(f'3D Hidden State Trajectory ({config_name})')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'trajectory_3d_{config_name}.png'), dpi=150)
    plt.close()

    return hidden_trajectory, sequence_trajectory


def compute_rsa(data, save_dir, num_configs=10):
    """
    Compute Representational Similarity Analysis across configurations.
    """
    hidden = data['hidden']
    config_idx = data['config_idx']
    sequence_idx = data['sequence_idx']

    unique_configs = np.unique(config_idx)[:num_configs]

    # Compute mean hidden state per (config, sequence_state) pair
    mean_states = {}
    for cfg in unique_configs:
        for seq in range(4):
            mask = (config_idx == cfg) & (sequence_idx == seq)
            if mask.sum() > 0:
                mean_states[(cfg, seq)] = hidden[mask].mean(axis=0)

    # Compute correlation matrix
    keys = sorted(mean_states.keys())
    n = len(keys)
    corr_matrix = np.zeros((n, n))

    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            v1 = mean_states[k1]
            v2 = mean_states[k2]
            corr = np.corrcoef(v1, v2)[0, 1]
            corr_matrix[i, j] = corr

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    cbar = plt.colorbar(im)
    cbar.set_label('Correlation')

    # Create labels
    labels = [f'C{k[0]}-{["A","B","C","D"][k[1]]}' for k in keys]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title('Representational Similarity Across Configurations')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rsa_matrix.png'), dpi=150)
    plt.close()

    return corr_matrix, keys


def analyze(args):
    """Main analysis function."""
    set_seed(args.seed)
    ensure_dir(config.FIGURES_DIR)

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Running analysis on {device}")

    # Load agent
    agent = MetaRLAgent(device=device)
    if args.checkpoint:
        agent.load(args.checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        final_path = os.path.join(config.MODEL_DIR, 'final_model.pt')
        if os.path.exists(final_path):
            agent.load(final_path)
            print(f"Loaded final model: {final_path}")
        else:
            print("Warning: No checkpoint found, using untrained agent")

    agent.eval_mode()

    # Get configurations for analysis
    eval_configs = get_eval_configs()
    analysis_configs = eval_configs[:args.num_configs]
    print(f"Analyzing {len(analysis_configs)} configurations")

    # Collect hidden states
    print("\nCollecting hidden states...")
    data = collect_hidden_states(agent, analysis_configs,
                                 num_sessions_per_config=args.sessions_per_config)
    print(f"Collected {len(data['hidden'])} hidden states")

    # Save raw data
    if args.save_data:
        data_path = os.path.join(config.RESULTS_DIR, 'hidden_states.npy')
        np.save(data_path, data, allow_pickle=True)
        print(f"Saved hidden states to: {data_path}")

    # Run analyses
    print("\nRunning PCA analysis...")
    pca, hidden_pca = analyze_pca(data, config.FIGURES_DIR)

    if args.tsne:
        print("\nRunning t-SNE analysis...")
        hidden_tsne = analyze_tsne(data, config.FIGURES_DIR)

    print("\nTraining goal decoder...")
    decoder, accuracy = decode_goal(data, config.FIGURES_DIR)

    print("\nAnalyzing example trajectories...")
    for i, cfg in enumerate(analysis_configs[:3]):
        analyze_trajectory(agent, cfg, config.FIGURES_DIR, config_name=f'config{i}')

    print("\nComputing RSA...")
    rsa_matrix, rsa_keys = compute_rsa(data, config.FIGURES_DIR)

    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print(f"Figures saved to: {config.FIGURES_DIR}/")
    print(f"\nKey results:")
    print(f"  - Goal decoding accuracy: {accuracy:.3f}")
    print(f"  - PCA variance explained (top 3): {pca.explained_variance_ratio_[:3].sum():.3f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Meta-RL ABCD agent representations')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--seed', type=int, default=789,
                        help='Random seed')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU')
    parser.add_argument('--num-configs', type=int, default=20,
                        help='Number of configurations to analyze')
    parser.add_argument('--sessions-per-config', type=int, default=3,
                        help='Sessions per configuration')
    parser.add_argument('--tsne', action='store_true', default=True,
                        help='Run t-SNE analysis')
    parser.add_argument('--save-data', action='store_true', default=True,
                        help='Save raw hidden state data')

    args = parser.parse_args()
    analyze(args)


if __name__ == '__main__':
    main()
