"""
Sequence Structure Analysis for Meta-RL ABCD task.

Three analyses to uncover task structure beyond time:
1. Align by sequence state (A→B→C→D transitions)
2. Remove time trend, analyze residuals
3. Reward-triggered averages around goal arrivals
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.ndimage import gaussian_filter1d

import config
from environment import ABCDEnvironmentWithTracking
from agent import MetaRLAgent
from utils import get_training_configs, ensure_dir, set_seed


def collect_detailed_trajectories(agent, configs, num_sessions_per_config=1):
    """
    Collect trajectories with detailed transition information.
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
                'reward': [],
                'target': [],
                'transition_steps': [],  # Steps where transitions occurred
            }

            prev_seq_idx = env.current_sequence_idx

            for step in range(config.SESSION_LENGTH):
                hidden = agent.get_hidden_numpy().flatten()
                session_data['hidden'].append(hidden)
                session_data['sequence_idx'].append(env.current_sequence_idx)
                session_data['position'].append(env.current_pos)
                session_data['step'].append(step)
                session_data['reward'].append(env.last_reward)
                session_data['target'].append(env.get_current_target())

                # Detect transition
                if env.current_sequence_idx != prev_seq_idx:
                    session_data['transition_steps'].append({
                        'step': step,
                        'from_state': prev_seq_idx,
                        'to_state': env.current_sequence_idx,
                    })
                    prev_seq_idx = env.current_sequence_idx

                action, _, _ = agent.act(obs, deterministic=True)
                obs, reward, done, info = env.step(action)

                if info['reached_target']:
                    # Record the transition that just happened
                    session_data['transition_steps'].append({
                        'step': step + 1,
                        'from_state': session_data['sequence_idx'][-1],
                        'to_state': env.current_sequence_idx,
                        'reward_step': True
                    })

                if done:
                    break

            session_data['hidden'] = np.array(session_data['hidden'])
            session_data['sequence_idx'] = np.array(session_data['sequence_idx'])
            session_data['position'] = np.array(session_data['position'])
            session_data['reward'] = np.array(session_data['reward'])
            sessions.append(session_data)

    return sessions


# =============================================================================
# ANALYSIS 1: Align by sequence state
# =============================================================================

def align_by_sequence_state(sessions, save_dir):
    """
    Align trajectories by sequence state transitions rather than absolute time.

    For each A→B→C→D→A cycle, extract the trajectory segment and align them.
    """
    print("\n" + "="*60)
    print("ANALYSIS 1: Align by Sequence State")
    print("="*60)

    # First, fit PCA on all data
    all_hidden = np.vstack([s['hidden'] for s in sessions])
    pca = PCA(n_components=10)
    pca.fit(all_hidden)

    print(f"PCA variance explained (top 5): {pca.explained_variance_ratio_[:5]}")

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    state_names = ['A', 'B', 'C', 'D']

    # Extract segments between transitions
    # We'll align relative to each goal arrival

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Collect aligned segments for each transition type
    transition_types = ['A→B', 'B→C', 'C→D', 'D→A']
    aligned_segments = {t: [] for t in transition_types}

    window_before = 15  # steps before goal
    window_after = 10   # steps after goal

    for session in sessions:
        hidden_pca = pca.transform(session['hidden'])
        rewards = session['reward']
        seq_idx = session['sequence_idx']

        # Find reward steps (goal arrivals)
        reward_steps = np.where(rewards > 0)[0]

        for r_step in reward_steps:
            if r_step < window_before or r_step + window_after >= len(hidden_pca):
                continue

            # Determine transition type
            state_before = seq_idx[r_step - 1] if r_step > 0 else seq_idx[r_step]
            state_after = seq_idx[min(r_step + 1, len(seq_idx) - 1)]

            # The transition is from state_before to state_after
            # But actually, reaching goal B means we were seeking B (so state was A)
            # After reward, state becomes B
            trans_key = f"{state_names[state_before]}→{state_names[state_after]}"

            if trans_key in aligned_segments:
                segment = hidden_pca[r_step - window_before:r_step + window_after + 1]
                if len(segment) == window_before + window_after + 1:
                    aligned_segments[trans_key].append(segment)

    # Plot aligned trajectories for each transition type
    time_axis = np.arange(-window_before, window_after + 1)

    for idx, (trans_type, segments) in enumerate(aligned_segments.items()):
        if len(segments) == 0:
            continue

        segments = np.array(segments)  # (n_segments, time, pcs)
        mean_seg = segments.mean(axis=0)
        std_seg = segments.std(axis=0)

        ax = axes[0, idx] if idx < 3 else axes[1, idx - 3]

        # Plot PC1 and PC2
        for pc, color, label in [(0, 'blue', 'PC1'), (1, 'orange', 'PC2')]:
            ax.plot(time_axis, mean_seg[:, pc], color=color, linewidth=2, label=label)
            ax.fill_between(time_axis,
                           mean_seg[:, pc] - std_seg[:, pc],
                           mean_seg[:, pc] + std_seg[:, pc],
                           color=color, alpha=0.2)

        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Goal arrival')
        ax.set_xlabel('Steps relative to goal')
        ax.set_ylabel('PC value')
        ax.set_title(f'{trans_type} transition (n={len(segments)})')
        ax.legend(fontsize=8)

    # Plot all transitions overlaid in PC space
    ax_3d = axes[1, 0]
    ax_3d.set_title('Mean trajectories in PC1-PC2 space')

    for idx, (trans_type, segments) in enumerate(aligned_segments.items()):
        if len(segments) == 0:
            continue
        segments = np.array(segments)
        mean_seg = segments.mean(axis=0)

        color = colors[idx]
        ax_3d.plot(mean_seg[:, 0], mean_seg[:, 1], color=color,
                   linewidth=2, label=trans_type)
        # Mark goal arrival
        ax_3d.scatter(mean_seg[window_before, 0], mean_seg[window_before, 1],
                     color=color, s=100, marker='*', edgecolors='black', zorder=5)

    ax_3d.set_xlabel('PC1')
    ax_3d.set_ylabel('PC2')
    ax_3d.legend()

    # Compute alignment metrics across transition types
    ax_metric = axes[1, 1]

    # Compare shapes: correlation between mean trajectories
    mean_trajectories = {}
    for trans_type, segments in aligned_segments.items():
        if len(segments) > 0:
            mean_trajectories[trans_type] = np.array(segments).mean(axis=0)[:, :3].flatten()

    if len(mean_trajectories) >= 2:
        trans_keys = list(mean_trajectories.keys())
        corr_matrix = np.zeros((len(trans_keys), len(trans_keys)))
        for i, k1 in enumerate(trans_keys):
            for j, k2 in enumerate(trans_keys):
                r, _ = stats.pearsonr(mean_trajectories[k1], mean_trajectories[k2])
                corr_matrix[i, j] = r

        im = ax_metric.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax_metric.set_xticks(range(len(trans_keys)))
        ax_metric.set_yticks(range(len(trans_keys)))
        ax_metric.set_xticklabels(trans_keys, rotation=45)
        ax_metric.set_yticklabels(trans_keys)
        ax_metric.set_title('Trajectory shape correlation')
        plt.colorbar(im, ax=ax_metric)

        print("\nTrajectory correlations across transition types:")
        for i, k1 in enumerate(trans_keys):
            for j, k2 in enumerate(trans_keys):
                if i < j:
                    print(f"  {k1} vs {k2}: r = {corr_matrix[i,j]:.3f}")

    # Summary statistics
    ax_summary = axes[1, 2]

    # Variance at goal arrival vs before/after
    variances = {'Before': [], 'At goal': [], 'After': []}
    for trans_type, segments in aligned_segments.items():
        if len(segments) == 0:
            continue
        segments = np.array(segments)
        # Variance across trials at each timepoint
        var_before = segments[:, :window_before, :3].var(axis=0).mean()
        var_at = segments[:, window_before, :3].var(axis=0).mean()
        var_after = segments[:, window_before+1:, :3].var(axis=0).mean()
        variances['Before'].append(var_before)
        variances['At goal'].append(var_at)
        variances['After'].append(var_after)

    x = np.arange(3)
    means = [np.mean(variances[k]) for k in ['Before', 'At goal', 'After']]
    stds = [np.std(variances[k]) for k in ['Before', 'At goal', 'After']]
    ax_summary.bar(x, means, yerr=stds, color=['gray', 'red', 'gray'], capsize=5)
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels(['Before', 'At goal', 'After'])
    ax_summary.set_ylabel('Cross-trial variance')
    ax_summary.set_title('Consistency across trials')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sequence_aligned_trajectories.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return aligned_segments, pca


# =============================================================================
# ANALYSIS 2: Remove time trend, analyze residuals
# =============================================================================

def analyze_residuals(sessions, save_dir):
    """
    Remove the dominant time trend and analyze what structure remains.
    """
    print("\n" + "="*60)
    print("ANALYSIS 2: Remove Time Trend, Analyze Residuals")
    print("="*60)

    # Collect all data with time
    all_hidden = []
    all_time = []
    all_seq_idx = []
    all_position = []
    all_config = []

    for session in sessions:
        all_hidden.append(session['hidden'])
        all_time.append(session['step'])
        all_seq_idx.append(session['sequence_idx'])
        all_position.append(session['position'])
        all_config.extend([session['config_idx']] * len(session['hidden']))

    all_hidden = np.vstack(all_hidden)
    all_time = np.concatenate(all_time)
    all_seq_idx = np.concatenate(all_seq_idx)
    all_position = np.concatenate(all_position)
    all_config = np.array(all_config)

    # Fit PCA first
    pca = PCA(n_components=10)
    hidden_pca = pca.fit_transform(all_hidden)

    print(f"Original PCA variance: {pca.explained_variance_ratio_[:5]}")

    # Method 1: Regress out time from each PC
    residuals_linear = np.zeros_like(hidden_pca)
    time_coefficients = []

    for pc in range(hidden_pca.shape[1]):
        X = all_time.reshape(-1, 1)
        y = hidden_pca[:, pc]

        # Fit polynomial (time trend)
        reg = LinearRegression()
        reg.fit(X, y)
        predicted = reg.predict(X)
        residuals_linear[:, pc] = y - predicted
        time_coefficients.append(reg.coef_[0])

    print(f"Time coefficients for PCs: {time_coefficients[:5]}")

    # Method 2: Regress out time with polynomial
    residuals_poly = np.zeros_like(hidden_pca)

    for pc in range(hidden_pca.shape[1]):
        # Polynomial features
        X_poly = np.column_stack([all_time, all_time**2, all_time**3])
        y = hidden_pca[:, pc]

        reg = LinearRegression()
        reg.fit(X_poly, y)
        predicted = reg.predict(X_poly)
        residuals_poly[:, pc] = y - predicted

    # Re-do PCA on residuals
    pca_residual = PCA(n_components=10)
    residuals_pca = pca_residual.fit_transform(residuals_poly)

    print(f"Residual PCA variance: {pca_residual.explained_variance_ratio_[:5]}")

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    state_names = ['A', 'B', 'C', 'D']

    # Original PC1 vs time
    ax1 = axes[0, 0]
    for state in range(4):
        mask = all_seq_idx == state
        ax1.scatter(all_time[mask], hidden_pca[mask, 0],
                   c=colors[state], alpha=0.3, s=5, label=state_names[state])
    ax1.set_xlabel('Step')
    ax1.set_ylabel('PC1 (original)')
    ax1.set_title('Original: PC1 dominated by time')
    ax1.legend(markerscale=3)

    # Residual PC1 vs time
    ax2 = axes[0, 1]
    for state in range(4):
        mask = all_seq_idx == state
        ax2.scatter(all_time[mask], residuals_pca[mask, 0],
                   c=colors[state], alpha=0.3, s=5, label=state_names[state])
    ax2.set_xlabel('Step')
    ax2.set_ylabel('PC1 (residual)')
    ax2.set_title('Residual: Time trend removed')
    ax2.legend(markerscale=3)

    # Residual PC1 vs PC2 by sequence state
    ax3 = axes[0, 2]
    for state in range(4):
        mask = all_seq_idx == state
        ax3.scatter(residuals_pca[mask, 0], residuals_pca[mask, 1],
                   c=colors[state], alpha=0.3, s=5, label=state_names[state])
    ax3.set_xlabel('Residual PC1')
    ax3.set_ylabel('Residual PC2')
    ax3.set_title('Residual structure by sequence state')
    ax3.legend(markerscale=3)

    # Mean residual per sequence state
    ax4 = axes[1, 0]
    mean_residuals = []
    for state in range(4):
        mask = all_seq_idx == state
        mean_residuals.append(residuals_pca[mask, :3].mean(axis=0))
    mean_residuals = np.array(mean_residuals)

    x = np.arange(4)
    width = 0.25
    for pc in range(3):
        ax4.bar(x + pc*width, mean_residuals[:, pc], width,
               label=f'PC{pc+1}', alpha=0.8)
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(state_names)
    ax4.set_ylabel('Mean residual')
    ax4.set_title('Mean residual by sequence state')
    ax4.legend()
    ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)

    # Residual by position
    ax5 = axes[1, 1]
    mean_by_pos = []
    for pos in range(9):
        mask = all_position == pos
        if mask.sum() > 0:
            mean_by_pos.append(residuals_pca[mask, :2].mean(axis=0))
        else:
            mean_by_pos.append([0, 0])
    mean_by_pos = np.array(mean_by_pos)

    # Color by node degree
    node_degrees = {0: 2, 1: 3, 2: 2, 3: 3, 4: 4, 5: 3, 6: 2, 7: 3, 8: 2}
    degree_colors = {2: '#e41a1c', 3: '#377eb8', 4: '#4daf4a'}

    for pos in range(9):
        ax5.scatter(mean_by_pos[pos, 0], mean_by_pos[pos, 1],
                   c=degree_colors[node_degrees[pos]], s=200,
                   edgecolors='black', linewidth=1.5)
        ax5.annotate(str(pos), (mean_by_pos[pos, 0], mean_by_pos[pos, 1]),
                    ha='center', va='center', fontsize=10, fontweight='bold')

    ax5.set_xlabel('Mean Residual PC1')
    ax5.set_ylabel('Mean Residual PC2')
    ax5.set_title('Residual structure by position')

    # Add legend for degree
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e41a1c', edgecolor='black', label='Corner (deg 2)'),
        Patch(facecolor='#377eb8', edgecolor='black', label='Cardinal (deg 3)'),
        Patch(facecolor='#4daf4a', edgecolor='black', label='Centre (deg 4)')
    ]
    ax5.legend(handles=legend_elements, loc='best')

    # Decoding from residuals: can we decode sequence state?
    ax6 = axes[1, 2]

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    # Decode sequence state from original PCs
    clf_orig = LogisticRegression(max_iter=1000, multi_class='multinomial')
    scores_orig = cross_val_score(clf_orig, hidden_pca[:, :5], all_seq_idx, cv=5)

    # Decode from residual PCs
    clf_resid = LogisticRegression(max_iter=1000, multi_class='multinomial')
    scores_resid = cross_val_score(clf_resid, residuals_pca[:, :5], all_seq_idx, cv=5)

    # Decode position from residuals
    clf_pos = LogisticRegression(max_iter=1000, multi_class='multinomial')
    scores_pos = cross_val_score(clf_pos, residuals_pca[:, :5], all_position, cv=5)

    print(f"\nDecoding accuracy:")
    print(f"  Sequence state from original PCs: {scores_orig.mean():.3f} ± {scores_orig.std():.3f}")
    print(f"  Sequence state from residuals: {scores_resid.mean():.3f} ± {scores_resid.std():.3f}")
    print(f"  Position from residuals: {scores_pos.mean():.3f} ± {scores_pos.std():.3f}")

    bars = ax6.bar(['Seq (original)', 'Seq (residual)', 'Position (residual)'],
                   [scores_orig.mean(), scores_resid.mean(), scores_pos.mean()],
                   yerr=[scores_orig.std(), scores_resid.std(), scores_pos.std()],
                   color=['gray', 'steelblue', 'orange'], capsize=5)
    ax6.axhline(0.25, color='red', linestyle='--', label='Chance (seq)')
    ax6.axhline(1/9, color='orange', linestyle=':', label='Chance (pos)')
    ax6.set_ylabel('Decoding Accuracy')
    ax6.set_title('Decoding from PCs')
    ax6.legend()
    ax6.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'residual_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return residuals_pca, pca_residual


# =============================================================================
# ANALYSIS 3: Reward-triggered averages
# =============================================================================

def reward_triggered_analysis(sessions, save_dir):
    """
    Compute averages locked to reward delivery (goal arrival).
    """
    print("\n" + "="*60)
    print("ANALYSIS 3: Reward-Triggered Averages")
    print("="*60)

    # Fit PCA on all data
    all_hidden = np.vstack([s['hidden'] for s in sessions])
    pca = PCA(n_components=10)
    pca.fit(all_hidden)

    window_before = 20
    window_after = 15

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    state_names = ['A', 'B', 'C', 'D']

    # Collect reward-triggered segments by goal type
    # Goal type = which state we just reached
    goal_triggered = {s: [] for s in state_names}

    for session in sessions:
        hidden_pca = pca.transform(session['hidden'])
        rewards = session['reward']
        seq_idx = session['sequence_idx']

        reward_steps = np.where(rewards > 0)[0]

        for r_step in reward_steps:
            if r_step < window_before or r_step + window_after >= len(hidden_pca):
                continue

            # Which goal did we just reach?
            goal_state = seq_idx[r_step]  # State after reaching goal
            goal_name = state_names[goal_state]

            segment = hidden_pca[r_step - window_before:r_step + window_after + 1]
            if len(segment) == window_before + window_after + 1:
                goal_triggered[goal_name].append(segment)

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    time_axis = np.arange(-window_before, window_after + 1)

    # Plot each goal type separately
    for idx, (goal_name, segments) in enumerate(goal_triggered.items()):
        if len(segments) == 0:
            continue

        segments = np.array(segments)
        mean_seg = segments.mean(axis=0)
        sem_seg = segments.std(axis=0) / np.sqrt(len(segments))

        ax = axes[0, idx] if idx < 3 else axes[1, idx - 3]

        # Plot first 3 PCs
        for pc, ls in [(0, '-'), (1, '--'), (2, ':')]:
            ax.plot(time_axis, mean_seg[:, pc], ls, linewidth=2,
                   label=f'PC{pc+1}', color=colors[idx])
            ax.fill_between(time_axis,
                           mean_seg[:, pc] - sem_seg[:, pc],
                           mean_seg[:, pc] + sem_seg[:, pc],
                           alpha=0.2, color=colors[idx])

        ax.axvline(0, color='black', linestyle='-', linewidth=2)
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Steps relative to reward')
        ax.set_ylabel('PC value')
        ax.set_title(f'Reached goal {goal_name} (n={len(segments)})')
        ax.legend(fontsize=8)

    # Overlay all goal types
    ax_overlay = axes[1, 0]
    for goal_name, segments in goal_triggered.items():
        if len(segments) == 0:
            continue
        segments = np.array(segments)
        mean_seg = segments.mean(axis=0)

        color = colors[state_names.index(goal_name)]
        ax_overlay.plot(time_axis, mean_seg[:, 0], linewidth=2,
                       label=f'Goal {goal_name}', color=color)

    ax_overlay.axvline(0, color='black', linestyle='--', linewidth=2)
    ax_overlay.set_xlabel('Steps relative to reward')
    ax_overlay.set_ylabel('PC1')
    ax_overlay.set_title('PC1: All goal types overlaid')
    ax_overlay.legend()

    # Compute "reset" magnitude
    ax_reset = axes[1, 1]

    reset_magnitudes = []
    goal_labels = []

    for goal_name, segments in goal_triggered.items():
        if len(segments) == 0:
            continue
        segments = np.array(segments)

        # Change in PC1 around reward
        for seg in segments:
            before_mean = seg[window_before-3:window_before, 0].mean()
            after_mean = seg[window_before+1:window_before+4, 0].mean()
            reset_magnitudes.append(after_mean - before_mean)
            goal_labels.append(goal_name)

    reset_magnitudes = np.array(reset_magnitudes)
    goal_labels = np.array(goal_labels)

    positions = []
    means = []
    stds = []
    for i, gn in enumerate(state_names):
        mask = goal_labels == gn
        if mask.sum() > 0:
            positions.append(i)
            means.append(reset_magnitudes[mask].mean())
            stds.append(reset_magnitudes[mask].std())

    ax_reset.bar(positions, means, yerr=stds, color=colors[:len(positions)], capsize=5)
    ax_reset.set_xticks(positions)
    ax_reset.set_xticklabels([state_names[p] for p in positions])
    ax_reset.set_ylabel('PC1 change at reward')
    ax_reset.set_title('Reset magnitude by goal')
    ax_reset.axhline(0, color='black', linestyle='-', linewidth=0.5)

    # 3D trajectory around rewards
    ax_3d = axes[1, 2]

    for goal_name, segments in goal_triggered.items():
        if len(segments) == 0:
            continue
        segments = np.array(segments)
        mean_seg = segments.mean(axis=0)

        color = colors[state_names.index(goal_name)]
        ax_3d.plot(mean_seg[:, 0], mean_seg[:, 1],
                  linewidth=2, label=f'Goal {goal_name}', color=color)
        # Mark reward moment
        ax_3d.scatter(mean_seg[window_before, 0], mean_seg[window_before, 1],
                     s=150, marker='*', color=color, edgecolors='black', zorder=5)

    ax_3d.set_xlabel('PC1')
    ax_3d.set_ylabel('PC2')
    ax_3d.set_title('Trajectory in PC space around reward')
    ax_3d.legend()

    # Statistics
    print("\nReward-triggered statistics:")
    for goal_name in state_names:
        if goal_name in [g for g in goal_triggered if len(goal_triggered[g]) > 0]:
            segments = np.array(goal_triggered[goal_name])
            pc1_before = segments[:, window_before-1, 0].mean()
            pc1_after = segments[:, window_before+1, 0].mean()
            print(f"  Goal {goal_name}: PC1 {pc1_before:.3f} → {pc1_after:.3f} (Δ = {pc1_after-pc1_before:.3f})")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'reward_triggered_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return goal_triggered


def main():
    set_seed(42)
    ensure_dir(config.FIGURES_DIR)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running sequence structure analysis on {device}")

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

    # Get more sessions for better statistics
    train_configs = get_training_configs()
    test_configs = train_configs[:15]  # 15 different tasks

    print(f"\nCollecting trajectories from {len(test_configs)} task configurations...")
    sessions = collect_detailed_trajectories(agent, test_configs, num_sessions_per_config=1)
    print(f"Collected {len(sessions)} sessions")

    # Analysis 1: Align by sequence state
    aligned_segments, pca1 = align_by_sequence_state(sessions, config.FIGURES_DIR)

    # Analysis 2: Remove time trend
    residuals, pca_residual = analyze_residuals(sessions, config.FIGURES_DIR)

    # Analysis 3: Reward-triggered averages
    goal_triggered = reward_triggered_analysis(sessions, config.FIGURES_DIR)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Figures saved to: {config.FIGURES_DIR}/")
    print("  - sequence_aligned_trajectories.png")
    print("  - residual_analysis.png")
    print("  - reward_triggered_analysis.png")


if __name__ == '__main__':
    main()
