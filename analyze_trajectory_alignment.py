"""
Trajectory Alignment Analysis for Meta-RL ABCD task.

Analyzes whether the periodic/spiral structure in PCA space
aligns across different ABCD task configurations.

Two approaches:
1. Pooled PCA: Fit PCA on all data, compare trajectories
2. Reference projection: Fit PCA on task 1, project others onto same axes
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

import config
from environment import ABCDEnvironmentWithTracking
from agent import MetaRLAgent
from utils import get_eval_configs, get_training_configs, ensure_dir, set_seed


def collect_session_trajectories(agent, configs, num_sessions_per_config=1):
    """
    Collect hidden state trajectories organized by session.

    Returns list of sessions, each containing:
    - hidden states at each step
    - sequence state (A/B/C/D) at each step
    - position at each step
    """
    sessions = []

    for cfg_idx, abcd_config in enumerate(configs):
        for session_num in range(num_sessions_per_config):
            env = ABCDEnvironmentWithTracking(abcd_config)
            obs = env.reset()
            agent.reset_hidden()

            session_data = {
                'config_idx': cfg_idx,
                'config': abcd_config,
                'hidden': [],
                'sequence_idx': [],
                'position': [],
                'step': [],
                'reward': []
            }

            for step in range(config.SESSION_LENGTH):
                hidden = agent.get_hidden_numpy().flatten()
                session_data['hidden'].append(hidden)
                session_data['sequence_idx'].append(env.current_sequence_idx)
                session_data['position'].append(env.current_pos)
                session_data['step'].append(step)
                session_data['reward'].append(env.last_reward)

                action, _, _ = agent.act(obs, deterministic=True)
                obs, reward, done, info = env.step(action)

                if done:
                    break

            session_data['hidden'] = np.array(session_data['hidden'])
            session_data['sequence_idx'] = np.array(session_data['sequence_idx'])
            session_data['position'] = np.array(session_data['position'])
            sessions.append(session_data)

    return sessions


def compute_mean_sequence_points(session, sequence_states=[0, 1, 2, 3]):
    """
    Compute mean hidden state for each sequence state (A, B, C, D).

    Returns:
        mean_points: (4, hidden_dim) array of mean hidden states
        counts: number of samples per state
    """
    hidden = session['hidden']
    seq_idx = session['sequence_idx']

    mean_points = []
    counts = []

    for state in sequence_states:
        mask = seq_idx == state
        if mask.sum() > 0:
            mean_points.append(hidden[mask].mean(axis=0))
            counts.append(mask.sum())
        else:
            mean_points.append(np.zeros(hidden.shape[1]))
            counts.append(0)

    return np.array(mean_points), counts


def analyze_pooled_pca(sessions, save_dir):
    """
    Approach 1: Pool all data, fit single PCA, compare trajectories.
    """
    print("\n=== APPROACH 1: Pooled PCA ===")

    # Pool all hidden states
    all_hidden = np.vstack([s['hidden'] for s in sessions])

    # Fit PCA
    pca = PCA(n_components=3)
    pca.fit(all_hidden)

    print(f"Variance explained: {pca.explained_variance_ratio_[:3]}")
    print(f"Total: {pca.explained_variance_ratio_[:3].sum():.3f}")

    # Project each session and compute mean A/B/C/D points
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']  # A, B, C, D
    labels = ['A', 'B', 'C', 'D']

    fig = plt.figure(figsize=(16, 12))

    # 3D trajectory plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    all_mean_points_pca = []

    for i, session in enumerate(sessions):
        # Project to PCA space
        hidden_pca = pca.transform(session['hidden'])

        # Plot full trajectory (faded)
        ax1.plot(hidden_pca[:, 0], hidden_pca[:, 1], hidden_pca[:, 2],
                alpha=0.2, linewidth=0.5, color='gray')

        # Compute and plot mean A/B/C/D points
        mean_points, _ = compute_mean_sequence_points(session)
        mean_points_pca = pca.transform(mean_points)
        all_mean_points_pca.append(mean_points_pca)

        for j, (point, color, label) in enumerate(zip(mean_points_pca, colors, labels)):
            ax1.scatter(point[0], point[1], point[2], c=color, s=100,
                       alpha=0.6, edgecolors='black', linewidth=0.5)

    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax1.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    ax1.set_title(f'Pooled PCA: All {len(sessions)} Sessions')

    # Add legend
    for color, label in zip(colors, labels):
        ax1.scatter([], [], [], c=color, s=100, label=label)
    ax1.legend()

    # 2D projection (PC1 vs PC2) - mean points with connections
    ax2 = fig.add_subplot(2, 2, 2)

    all_mean_points_pca = np.array(all_mean_points_pca)  # (n_sessions, 4, 3)

    for i in range(len(sessions)):
        points = all_mean_points_pca[i]
        # Connect A->B->C->D->A
        for j in range(4):
            next_j = (j + 1) % 4
            ax2.plot([points[j, 0], points[next_j, 0]],
                    [points[j, 1], points[next_j, 1]],
                    'k-', alpha=0.3, linewidth=0.5)

        for j, (color, label) in enumerate(zip(colors, labels)):
            ax2.scatter(points[j, 0], points[j, 1], c=color, s=80,
                       alpha=0.6, edgecolors='black', linewidth=0.5)

    # Plot grand mean
    grand_mean = all_mean_points_pca.mean(axis=0)
    for j in range(4):
        next_j = (j + 1) % 4
        ax2.plot([grand_mean[j, 0], grand_mean[next_j, 0]],
                [grand_mean[j, 1], grand_mean[next_j, 1]],
                'k-', linewidth=3, alpha=0.8)
    for j, (color, label) in enumerate(zip(colors, labels)):
        ax2.scatter(grand_mean[j, 0], grand_mean[j, 1], c=color, s=300,
                   edgecolors='black', linewidth=2, marker='*', label=f'{label} (mean)')

    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax2.set_title('Mean A/B/C/D Points Across Sessions')
    ax2.legend()

    # Compute alignment metrics
    # 1. Distance between corresponding states across sessions
    ax3 = fig.add_subplot(2, 2, 3)

    within_state_dists = []
    between_state_dists = []

    for state in range(4):
        state_points = all_mean_points_pca[:, state, :]  # (n_sessions, 3)
        # Within-state: distances between same state across sessions
        dists = cdist(state_points, state_points)
        within_state_dists.extend(dists[np.triu_indices(len(sessions), k=1)])

        # Between-state: distances to other states
        for other_state in range(4):
            if other_state != state:
                other_points = all_mean_points_pca[:, other_state, :]
                between_state_dists.extend(cdist(state_points, other_points).flatten())

    ax3.hist(within_state_dists, bins=20, alpha=0.7, label='Within-state', color='green')
    ax3.hist(between_state_dists, bins=20, alpha=0.7, label='Between-state', color='red')
    ax3.axvline(np.mean(within_state_dists), color='darkgreen', linestyle='--', linewidth=2)
    ax3.axvline(np.mean(between_state_dists), color='darkred', linestyle='--', linewidth=2)
    ax3.set_xlabel('Euclidean Distance in PC Space')
    ax3.set_ylabel('Count')
    ax3.set_title('Cross-Session Alignment')
    ax3.legend()

    # 2. Variance within each state vs between states
    ax4 = fig.add_subplot(2, 2, 4)

    within_var = []
    for state in range(4):
        state_points = all_mean_points_pca[:, state, :]
        within_var.append(state_points.var(axis=0).sum())

    # Total variance
    all_points_flat = all_mean_points_pca.reshape(-1, 3)
    total_var = all_points_flat.var(axis=0).sum()

    bars = ax4.bar(labels + ['Total'], within_var + [total_var],
                   color=colors + ['gray'])
    ax4.set_ylabel('Variance in PC Space')
    ax4.set_title('Within-State Variance (lower = better alignment)')

    # Print alignment ratio
    mean_within_var = np.mean(within_var)
    alignment_ratio = 1 - (mean_within_var / total_var)
    print(f"\nAlignment ratio: {alignment_ratio:.3f} (1.0 = perfect alignment)")
    print(f"Mean within-state distance: {np.mean(within_state_dists):.3f}")
    print(f"Mean between-state distance: {np.mean(between_state_dists):.3f}")
    print(f"Separation ratio: {np.mean(between_state_dists) / np.mean(within_state_dists):.2f}x")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'trajectory_alignment_pooled.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return pca, all_mean_points_pca


def analyze_reference_projection(sessions, save_dir):
    """
    Approach 2: Fit PCA on first task, project all others onto those axes.
    This is the stricter test of generalization.
    """
    print("\n=== APPROACH 2: Reference Projection ===")

    # Use first session as reference
    ref_session = sessions[0]
    ref_hidden = ref_session['hidden']

    # Fit PCA on reference
    pca = PCA(n_components=3)
    pca.fit(ref_hidden)

    print(f"Reference task variance explained: {pca.explained_variance_ratio_[:3]}")

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    labels = ['A', 'B', 'C', 'D']

    fig = plt.figure(figsize=(16, 12))

    # 3D plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    all_mean_points_pca = []

    for i, session in enumerate(sessions):
        # Project onto REFERENCE PCA axes
        hidden_pca = pca.transform(session['hidden'])

        # Plot trajectory
        alpha = 1.0 if i == 0 else 0.3
        lw = 2 if i == 0 else 0.5
        color = 'black' if i == 0 else 'gray'
        ax1.plot(hidden_pca[:, 0], hidden_pca[:, 1], hidden_pca[:, 2],
                alpha=alpha, linewidth=lw, color=color)

        # Mean points
        mean_points, _ = compute_mean_sequence_points(session)
        mean_points_pca = pca.transform(mean_points)
        all_mean_points_pca.append(mean_points_pca)

        marker = '*' if i == 0 else 'o'
        size = 200 if i == 0 else 80
        for j, (point, color_j) in enumerate(zip(mean_points_pca, colors)):
            ax1.scatter(point[0], point[1], point[2], c=color_j, s=size,
                       marker=marker, alpha=0.8, edgecolors='black')

    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    ax1.set_title('Reference Projection (Task 0 = reference)')

    # 2D comparison
    ax2 = fig.add_subplot(2, 2, 2)

    all_mean_points_pca = np.array(all_mean_points_pca)

    # Reference task (thick lines)
    ref_points = all_mean_points_pca[0]
    for j in range(4):
        next_j = (j + 1) % 4
        ax2.plot([ref_points[j, 0], ref_points[next_j, 0]],
                [ref_points[j, 1], ref_points[next_j, 1]],
                'k-', linewidth=3, alpha=1.0)
    for j, (color, label) in enumerate(zip(colors, labels)):
        ax2.scatter(ref_points[j, 0], ref_points[j, 1], c=color, s=300,
                   marker='*', edgecolors='black', linewidth=2, label=f'{label} (ref)')

    # Other tasks (thin lines)
    for i in range(1, len(sessions)):
        points = all_mean_points_pca[i]
        for j in range(4):
            next_j = (j + 1) % 4
            ax2.plot([points[j, 0], points[next_j, 0]],
                    [points[j, 1], points[next_j, 1]],
                    '-', color='gray', alpha=0.3, linewidth=0.5)
        for j, color in enumerate(colors):
            ax2.scatter(points[j, 0], points[j, 1], c=color, s=60,
                       alpha=0.5, edgecolors='black', linewidth=0.5)

    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('Projected onto Reference PC Axes')
    ax2.legend()

    # Procrustes-like alignment measure
    ax3 = fig.add_subplot(2, 2, 3)

    # Compute correlation between reference and each other task
    ref_flat = all_mean_points_pca[0].flatten()
    correlations = []
    for i in range(1, len(sessions)):
        other_flat = all_mean_points_pca[i].flatten()
        r, _ = pearsonr(ref_flat, other_flat)
        correlations.append(r)

    ax3.bar(range(1, len(sessions)), correlations, color='steelblue')
    ax3.axhline(np.mean(correlations), color='red', linestyle='--',
                label=f'Mean r = {np.mean(correlations):.3f}')
    ax3.set_xlabel('Task Index')
    ax3.set_ylabel('Correlation with Reference')
    ax3.set_title('Shape Similarity to Reference Task')
    ax3.set_ylim(-1, 1)
    ax3.legend()

    print(f"\nMean correlation with reference: {np.mean(correlations):.3f} ± {np.std(correlations):.3f}")

    # Per-state alignment
    ax4 = fig.add_subplot(2, 2, 4)

    state_correlations = {l: [] for l in labels}
    for state in range(4):
        ref_state = all_mean_points_pca[0, state, :]
        for i in range(1, len(sessions)):
            other_state = all_mean_points_pca[i, state, :]
            # Cosine similarity
            cos_sim = np.dot(ref_state, other_state) / (np.linalg.norm(ref_state) * np.linalg.norm(other_state))
            state_correlations[labels[state]].append(cos_sim)

    positions = range(4)
    means = [np.mean(state_correlations[l]) for l in labels]
    stds = [np.std(state_correlations[l]) for l in labels]
    ax4.bar(positions, means, yerr=stds, color=colors, capsize=5)
    ax4.set_xticks(positions)
    ax4.set_xticklabels(labels)
    ax4.set_ylabel('Cosine Similarity to Reference')
    ax4.set_title('Per-State Alignment')
    ax4.set_ylim(-1, 1)
    ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)

    for l in labels:
        print(f"  State {l}: cos_sim = {np.mean(state_correlations[l]):.3f} ± {np.std(state_correlations[l]):.3f}")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'trajectory_alignment_reference.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return pca, all_mean_points_pca, correlations


def plot_spiral_structure(sessions, save_dir):
    """
    Visualize the spiral/periodic structure more explicitly.
    Plot PC values vs time within session.
    """
    print("\n=== Spiral Structure Analysis ===")

    # Pool data and fit PCA
    all_hidden = np.vstack([s['hidden'] for s in sessions])
    pca = PCA(n_components=3)
    pca.fit(all_hidden)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    # PC1 vs time
    ax1 = axes[0, 0]
    for session in sessions[:5]:  # First 5 sessions
        hidden_pca = pca.transform(session['hidden'])
        steps = session['step']
        seq_idx = session['sequence_idx']

        ax1.plot(steps, hidden_pca[:, 0], alpha=0.3, color='gray')

        # Mark sequence state transitions
        for state, color in enumerate(colors):
            mask = seq_idx == state
            ax1.scatter(np.array(steps)[mask], hidden_pca[mask, 0],
                       c=color, s=20, alpha=0.5)

    ax1.set_xlabel('Step in Session')
    ax1.set_ylabel('PC1')
    ax1.set_title('PC1 vs Time')

    # PC2 vs time
    ax2 = axes[0, 1]
    for session in sessions[:5]:
        hidden_pca = pca.transform(session['hidden'])
        steps = session['step']
        seq_idx = session['sequence_idx']

        ax2.plot(steps, hidden_pca[:, 1], alpha=0.3, color='gray')

        for state, color in enumerate(colors):
            mask = seq_idx == state
            ax2.scatter(np.array(steps)[mask], hidden_pca[mask, 1],
                       c=color, s=20, alpha=0.5)

    ax2.set_xlabel('Step in Session')
    ax2.set_ylabel('PC2')
    ax2.set_title('PC2 vs Time')

    # PC1 vs PC2 colored by time
    ax3 = axes[1, 0]
    for session in sessions[:5]:
        hidden_pca = pca.transform(session['hidden'])
        steps = np.array(session['step'])

        scatter = ax3.scatter(hidden_pca[:, 0], hidden_pca[:, 1],
                             c=steps, cmap='viridis', s=10, alpha=0.5)

    plt.colorbar(scatter, ax=ax3, label='Step')
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    ax3.set_title('PC1 vs PC2 (colored by time)')

    # Phase plot: unwrap the cycle
    ax4 = axes[1, 1]

    # Convert PC1, PC2 to polar coordinates
    for i, session in enumerate(sessions[:5]):
        hidden_pca = pca.transform(session['hidden'])

        # Center the data
        pc1 = hidden_pca[:, 0] - hidden_pca[:, 0].mean()
        pc2 = hidden_pca[:, 1] - hidden_pca[:, 1].mean()

        # Compute angle (phase)
        phase = np.arctan2(pc2, pc1)
        # Unwrap to see cumulative rotation
        phase_unwrapped = np.unwrap(phase)

        steps = session['step']
        ax4.plot(steps, phase_unwrapped / (2 * np.pi), alpha=0.5, label=f'Task {i}')

    ax4.set_xlabel('Step in Session')
    ax4.set_ylabel('Cumulative Cycles (revolutions)')
    ax4.set_title('Phase Unwrapping: Rotations in PC Space')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spiral_structure.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Compute periodicity metrics
    print("\nPeriodicity analysis:")
    for i, session in enumerate(sessions[:3]):
        hidden_pca = pca.transform(session['hidden'])
        pc1 = hidden_pca[:, 0] - hidden_pca[:, 0].mean()
        pc2 = hidden_pca[:, 1] - hidden_pca[:, 1].mean()
        phase = np.unwrap(np.arctan2(pc2, pc1))
        total_rotations = (phase[-1] - phase[0]) / (2 * np.pi)
        print(f"  Task {i}: {total_rotations:.2f} total rotations over {len(session['step'])} steps")


def main():
    set_seed(42)
    ensure_dir(config.FIGURES_DIR)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running trajectory alignment analysis on {device}")

    # Load agent
    agent = MetaRLAgent(device=device)
    final_path = os.path.join(config.MODEL_DIR, 'final_model.pt')
    if os.path.exists(final_path):
        agent.load(final_path)
        print(f"Loaded final model: {final_path}")
    else:
        print("Warning: No checkpoint found")
        return

    agent.eval_mode()

    # Get configurations - use 5 different ABCD configs
    train_configs = get_training_configs()
    test_configs = train_configs[:5]  # 5 different ABCD configurations

    print(f"\nCollecting trajectories from {len(test_configs)} task configurations...")
    sessions = collect_session_trajectories(agent, test_configs, num_sessions_per_config=1)
    print(f"Collected {len(sessions)} sessions")

    # Print task configs
    print("\nTask configurations:")
    for i, session in enumerate(sessions):
        print(f"  Task {i}: ABCD = {session['config']}")

    # Analysis 1: Pooled PCA
    pca_pooled, mean_points_pooled = analyze_pooled_pca(sessions, config.FIGURES_DIR)

    # Analysis 2: Reference projection
    pca_ref, mean_points_ref, correlations = analyze_reference_projection(sessions, config.FIGURES_DIR)

    # Analysis 3: Spiral structure
    plot_spiral_structure(sessions, config.FIGURES_DIR)

    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print(f"Figures saved to: {config.FIGURES_DIR}/")
    print("  - trajectory_alignment_pooled.png")
    print("  - trajectory_alignment_reference.png")
    print("  - spiral_structure.png")


if __name__ == '__main__':
    main()
