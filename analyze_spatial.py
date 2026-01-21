"""
Generalised Spatial Tuning Analysis for Meta-RL ABCD task.

Replicates analyses from Chapter 7 of thesis:
1. Low-dimensional structure (PCA on spatial representations)
2. Representational Similarity Analysis (RSA)
3. Decoder generalisation (SVM leave-one-out)

Grid layout:
    0 1 2     (corners: 0,2,6,8 - degree 2)
    3 4 5     (cardinals: 1,3,5,7 - degree 3)
    6 7 8     (centre: 4 - degree 4)
"""

import os
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy import stats
from scipy.spatial.distance import pdist, squareform

import config
from environment import ABCDEnvironmentWithTracking
from agent import MetaRLAgent
from utils import get_eval_configs, get_training_configs, ensure_dir, set_seed


# Grid topology constants
CORNERS = [0, 2, 6, 8]  # degree 2
CARDINALS = [1, 3, 5, 7]  # degree 3
CENTRE = [4]  # degree 4

# Node degrees
NODE_DEGREES = {
    0: 2, 1: 3, 2: 2,
    3: 3, 4: 4, 5: 3,
    6: 2, 7: 3, 8: 2
}

# Euclidean coordinates for each position
NODE_COORDS = {
    0: (0, 0), 1: (0, 1), 2: (0, 2),
    3: (1, 0), 4: (1, 1), 5: (1, 2),
    6: (2, 0), 7: (2, 1), 8: (2, 2)
}

def get_node_type(pos):
    """Return node type: 'corner', 'cardinal', or 'centre'."""
    if pos in CORNERS:
        return 'corner'
    elif pos in CARDINALS:
        return 'cardinal'
    else:
        return 'centre'


def euclidean_distance(pos1, pos2):
    """Compute Euclidean distance between two positions."""
    c1 = NODE_COORDS[pos1]
    c2 = NODE_COORDS[pos2]
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def collect_hidden_states_extended(agent, configs, num_sessions_per_config=3):
    """
    Collect hidden states with extended metadata for spatial analysis.

    Returns:
        Dictionary with hidden states and metadata including:
        - position history for controlling trajectory effects
        - goal progress (sequence_idx)
        - config index for cross-task analysis
    """
    all_hidden = []
    all_positions = []
    all_sequence_idx = []
    all_steps = []
    all_config_idx = []
    all_rewards = []
    all_prev_positions = []  # For trajectory control

    for cfg_idx, abcd_config in enumerate(configs):
        for session in range(num_sessions_per_config):
            env = ABCDEnvironmentWithTracking(abcd_config)
            obs = env.reset()
            agent.reset_hidden()

            position_history = [-1] * 20  # Track last 20 positions

            for step in range(config.SESSION_LENGTH):
                # Record hidden state
                hidden = agent.get_hidden_numpy()
                all_hidden.append(hidden.flatten())
                all_positions.append(env.current_pos)
                all_sequence_idx.append(env.current_sequence_idx)
                all_steps.append(step)
                all_config_idx.append(cfg_idx)
                all_rewards.append(env.last_reward)
                all_prev_positions.append(position_history.copy())

                # Update position history
                position_history = [env.current_pos] + position_history[:-1]

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
        'prev_positions': np.array(all_prev_positions),
    }


def build_spatial_coefficients(data, goal_progress_bins=4):
    """
    Build spatial coefficients by regressing hidden state onto position indicators.

    For each neuron, fit a model predicting firing from current position
    while controlling for position history (like the thesis Lasso GLM).

    Returns:
        Dictionary with spatial coefficients per goal-progress bin
    """
    hidden = data['hidden']
    positions = data['position']
    sequence_idx = data['sequence_idx']

    n_neurons = hidden.shape[1]
    n_positions = 9

    # Bin goal progress
    gp_bins = np.digitize(sequence_idx, bins=np.linspace(0, 4, goal_progress_bins + 1)[1:-1])

    spatial_coeffs = {}

    for gp in range(goal_progress_bins):
        mask = gp_bins == gp
        if mask.sum() < 100:
            continue

        h_gp = hidden[mask]
        pos_gp = positions[mask]

        # Build position indicator matrix (one-hot)
        X = np.zeros((len(pos_gp), n_positions))
        for i, p in enumerate(pos_gp):
            X[i, p] = 1

        # Fit regression for each neuron
        coeffs = np.zeros((n_neurons, n_positions))

        for n in range(n_neurons):
            y = h_gp[:, n]
            # Simple OLS (can use Lasso for sparsity)
            try:
                model = Lasso(alpha=0.01, max_iter=1000)
                model.fit(X, y)
                coeffs[n, :] = model.coef_
            except:
                # Fallback to mean per position
                for p in range(n_positions):
                    coeffs[n, p] = y[pos_gp == p].mean() if (pos_gp == p).sum() > 0 else 0

        spatial_coeffs[gp] = coeffs

    return spatial_coeffs


def analyze_spatial_subspaces(spatial_coeffs, save_dir):
    """
    Analyze low-dimensional structure of spatial representations.

    Replicates Figure 7.2 from thesis:
    - PCA on spatial coefficients
    - Check if PC1 separates by node degree
    - Check if PC2-3 recover maze topology
    """
    fig, axes = plt.subplots(2, len(spatial_coeffs), figsize=(4*len(spatial_coeffs), 8))
    if len(spatial_coeffs) == 1:
        axes = axes.reshape(-1, 1)

    colors_by_degree = {2: '#e41a1c', 3: '#377eb8', 4: '#4daf4a'}  # corner, cardinal, centre

    for idx, (gp, coeffs) in enumerate(spatial_coeffs.items()):
        # PCA on coefficients
        pca = PCA(n_components=min(3, coeffs.shape[1]))
        coeffs_pca = pca.fit_transform(coeffs.T)  # Transpose: positions x neurons -> positions in PC space

        # Plot PC1 vs PC2 (should show node-degree separation)
        ax1 = axes[0, idx]
        for pos in range(9):
            degree = NODE_DEGREES[pos]
            ax1.scatter(coeffs_pca[pos, 0], coeffs_pca[pos, 1],
                       c=colors_by_degree[degree], s=200, edgecolors='black', linewidth=1.5)
            ax1.annotate(str(pos), (coeffs_pca[pos, 0], coeffs_pca[pos, 1]),
                        ha='center', va='center', fontsize=10, fontweight='bold')

        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax1.set_title(f'Goal Progress Bin {gp}')

        # Plot PC2 vs PC3 (should recover topology)
        if coeffs_pca.shape[1] >= 3:
            ax2 = axes[1, idx]
            for pos in range(9):
                degree = NODE_DEGREES[pos]
                ax2.scatter(coeffs_pca[pos, 1], coeffs_pca[pos, 2],
                           c=colors_by_degree[degree], s=200, edgecolors='black', linewidth=1.5)
                ax2.annotate(str(pos), (coeffs_pca[pos, 1], coeffs_pca[pos, 2]),
                            ha='center', va='center', fontsize=10, fontweight='bold')

            ax2.set_xlabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax2.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e41a1c', edgecolor='black', label='Corner (deg 2)'),
        Patch(facecolor='#377eb8', edgecolor='black', label='Cardinal (deg 3)'),
        Patch(facecolor='#4daf4a', edgecolor='black', label='Centre (deg 4)')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    plt.suptitle('Spatial Subspace Analysis: PCA on Position Coefficients', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spatial_subspace_pca.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return coeffs_pca


def compute_neural_rdm(data, min_speed_filter=False):
    """
    Compute neural Representational Dissimilarity Matrix.

    For each position, compute mean population vector, then
    compute Euclidean distances between all pairs.
    """
    hidden = data['hidden']
    positions = data['position']

    # Compute mean hidden state per position
    mean_vectors = np.zeros((9, hidden.shape[1]))
    for pos in range(9):
        mask = positions == pos
        if mask.sum() > 0:
            mean_vectors[pos] = hidden[mask].mean(axis=0)

    # Compute pairwise Euclidean distances
    rdm = squareform(pdist(mean_vectors, metric='euclidean'))

    return rdm, mean_vectors


def build_model_rdms():
    """
    Build model RDMs for RSA:
    1. Euclidean distance between maze locations
    2. Node degree difference
    3. Categorical place-type (same type = 0, different = 1)
    """
    # Euclidean distance model
    euclidean_rdm = np.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            euclidean_rdm[i, j] = euclidean_distance(i, j)

    # Node degree difference model
    degree_rdm = np.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            degree_rdm[i, j] = abs(NODE_DEGREES[i] - NODE_DEGREES[j])

    # Categorical place-type model (1 if different type, 0 if same)
    categorical_rdm = np.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            categorical_rdm[i, j] = 0 if get_node_type(i) == get_node_type(j) else 1

    return {
        'euclidean': euclidean_rdm,
        'node_degree': degree_rdm,
        'categorical': categorical_rdm
    }


def rsa_regression(neural_rdm, model_rdms, n_permutations=1000):
    """
    Perform RSA: regress model RDMs onto neural RDM.

    Returns regression coefficients and p-values from permutation test.
    """
    # Get upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(9, k=1)
    neural_vec = neural_rdm[triu_idx]

    # Build design matrix from model RDMs
    X = np.column_stack([
        model_rdms['euclidean'][triu_idx],
        model_rdms['node_degree'][triu_idx],
        model_rdms['categorical'][triu_idx]
    ])

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    neural_scaled = (neural_vec - neural_vec.mean()) / neural_vec.std()

    # Fit regression
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_scaled, neural_scaled)
    coeffs = model.coef_

    # Permutation test
    null_coeffs = np.zeros((n_permutations, 3))
    for perm in range(n_permutations):
        perm_idx = np.random.permutation(len(neural_scaled))
        model.fit(X_scaled, neural_scaled[perm_idx])
        null_coeffs[perm] = model.coef_

    # P-values
    p_values = np.array([
        (null_coeffs[:, i] >= coeffs[i]).mean() for i in range(3)
    ])

    return coeffs, p_values, null_coeffs


def analyze_rsa(data, save_dir):
    """
    Perform full RSA analysis.

    Replicates Figure 7.4 from thesis.
    """
    # Compute neural RDM
    neural_rdm, mean_vectors = compute_neural_rdm(data)

    # Build model RDMs
    model_rdms = build_model_rdms()

    # RSA regression
    coeffs, p_values, null_coeffs = rsa_regression(neural_rdm, model_rdms)

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))

    # Row 1: RDM visualizations
    rdm_names = ['Neural RDM', 'Euclidean Model', 'Node-Degree Model', 'Categorical Model']
    rdms = [neural_rdm, model_rdms['euclidean'], model_rdms['node_degree'], model_rdms['categorical']]

    for idx, (name, rdm) in enumerate(zip(rdm_names[:3], rdms[:3])):
        ax = axes[0, idx]
        im = ax.imshow(rdm if idx > 0 else rdm, cmap='viridis')
        ax.set_xticks(range(9))
        ax.set_yticks(range(9))
        ax.set_title(name)
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Row 2: Regression results
    ax_coef = axes[1, 0]
    model_names = ['Euclidean', 'Node-Degree', 'Categorical']
    colors = ['#2ca02c' if p < 0.05 else 'gray' for p in p_values]
    bars = ax_coef.bar(model_names, coeffs, color=colors)
    ax_coef.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax_coef.set_ylabel('Regression Coefficient')
    ax_coef.set_title('RSA Regression Coefficients')

    # Add p-values as text
    for i, (bar, p) in enumerate(zip(bars, p_values)):
        height = bar.get_height()
        ax_coef.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'p={p:.3f}', ha='center', va='bottom', fontsize=9)

    # MDS visualization of neural RDM
    ax_mds = axes[1, 1]
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    pos_2d = mds.fit_transform(neural_rdm)

    colors_by_degree = {2: '#e41a1c', 3: '#377eb8', 4: '#4daf4a'}
    for pos in range(9):
        degree = NODE_DEGREES[pos]
        ax_mds.scatter(pos_2d[pos, 0], pos_2d[pos, 1],
                      c=colors_by_degree[degree], s=200, edgecolors='black', linewidth=1.5)
        ax_mds.annotate(str(pos), (pos_2d[pos, 0], pos_2d[pos, 1]),
                       ha='center', va='center', fontsize=10, fontweight='bold')
    ax_mds.set_title('MDS of Neural RDM')
    ax_mds.set_xlabel('MDS Dim 1')
    ax_mds.set_ylabel('MDS Dim 2')

    # Null distribution comparison
    ax_null = axes[1, 2]
    for i, name in enumerate(model_names):
        ax_null.hist(null_coeffs[:, i], bins=30, alpha=0.5, label=name)
        ax_null.axvline(coeffs[i], color=['green', 'blue', 'orange'][i],
                       linestyle='--', linewidth=2)
    ax_null.set_xlabel('Coefficient')
    ax_null.set_ylabel('Count')
    ax_null.set_title('Null Distributions')
    ax_null.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rsa_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("\n=== RSA Results ===")
    for name, coef, p in zip(model_names, coeffs, p_values):
        sig = "*" if p < 0.05 else ""
        print(f"  {name}: β={coef:.4f}, p={p:.4f} {sig}")

    return coeffs, p_values, neural_rdm


def decode_corner_vs_cardinal(data, save_dir):
    """
    Train SVM to decode corner vs cardinal with leave-one-location-out CV.

    Replicates Figure 7.5c from thesis.
    """
    hidden = data['hidden']
    positions = data['position']

    # Exclude centre (position 4)
    mask = positions != 4
    hidden_filtered = hidden[mask]
    positions_filtered = positions[mask]

    # Labels: 0 = corner, 1 = cardinal
    labels = np.array([0 if p in CORNERS else 1 for p in positions_filtered])

    # Leave-one-location-out CV
    all_positions = CORNERS + CARDINALS  # 8 positions
    predictions = []
    true_labels = []

    for held_out_pos in all_positions:
        # Train on all except held_out_pos
        train_mask = positions_filtered != held_out_pos
        test_mask = positions_filtered == held_out_pos

        if test_mask.sum() == 0:
            continue

        X_train = hidden_filtered[train_mask]
        y_train = labels[train_mask]
        X_test = hidden_filtered[test_mask]
        y_test = labels[test_mask]

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train SVM
        clf = SVC(kernel='rbf', class_weight='balanced', random_state=42)
        clf.fit(X_train_scaled, y_train)

        # Predict
        y_pred = clf.predict(X_test_scaled)
        predictions.extend(y_pred)
        true_labels.extend(y_test)

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Compute balanced accuracy
    from sklearn.metrics import balanced_accuracy_score
    balanced_acc = balanced_accuracy_score(true_labels, predictions)

    # Statistical test against chance (0.5)
    n_correct = (predictions == true_labels).sum()
    n_total = len(predictions)
    # Binomial test
    from scipy.stats import binom_test
    try:
        p_value = binom_test(n_correct, n_total, 0.5, alternative='greater')
    except:
        p_value = stats.binom.sf(n_correct - 1, n_total, 0.5)

    return balanced_acc, p_value


def decode_absolute_position(data, save_dir):
    """
    Train SVM to decode absolute position with leave-one-config-out CV.

    Replicates Figure 7.5e from thesis.
    """
    hidden = data['hidden']
    positions = data['position']
    config_idx = data['config_idx']

    unique_configs = np.unique(config_idx)

    all_predictions = []
    all_true = []

    for held_out_config in unique_configs:
        train_mask = config_idx != held_out_config
        test_mask = config_idx == held_out_config

        if test_mask.sum() == 0:
            continue

        X_train = hidden[train_mask]
        y_train = positions[train_mask]
        X_test = hidden[test_mask]
        y_test = positions[test_mask]

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train 9-way SVM
        clf = SVC(kernel='rbf', class_weight='balanced', random_state=42)
        clf.fit(X_train_scaled, y_train)

        # Predict
        y_pred = clf.predict(X_test_scaled)
        all_predictions.extend(y_pred)
        all_true.extend(y_test)

    all_predictions = np.array(all_predictions)
    all_true = np.array(all_true)

    # Compute accuracy
    accuracy = (all_predictions == all_true).mean()

    # Build confusion matrix
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(all_true, all_predictions, labels=range(9))
    conf_matrix_normalized = conf_matrix.astype(float) / conf_matrix.sum(axis=1, keepdims=True)

    # Analyze errors: proportion of errors within same place-type
    errors_mask = all_predictions != all_true
    if errors_mask.sum() > 0:
        error_true = all_true[errors_mask]
        error_pred = all_predictions[errors_mask]

        same_type_errors = 0
        for t, p in zip(error_true, error_pred):
            if get_node_type(t) == get_node_type(p):
                same_type_errors += 1

        within_type_error_rate = same_type_errors / len(error_true)
    else:
        within_type_error_rate = 0

    return accuracy, conf_matrix_normalized, within_type_error_rate


def analyze_decoders(data, save_dir):
    """
    Run all decoder analyses.

    Replicates Figure 7.5 from thesis.
    """
    print("\n=== Decoder Analyses ===")

    # Corner vs Cardinal
    corner_cardinal_acc, corner_cardinal_p = decode_corner_vs_cardinal(data, save_dir)
    print(f"  Corner vs Cardinal: {corner_cardinal_acc:.3f} (p={corner_cardinal_p:.4f})")

    # Absolute position
    pos_acc, conf_matrix, within_type_errors = decode_absolute_position(data, save_dir)
    print(f"  Absolute Position: {pos_acc:.3f} (chance=0.111)")
    print(f"  Within-type error rate: {within_type_errors:.3f} (chance=0.429)")

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Corner vs Cardinal
    ax1 = axes[0]
    bars = ax1.bar(['Decoder', 'Chance'], [corner_cardinal_acc, 0.5],
                   color=['#2ca02c' if corner_cardinal_p < 0.05 else 'gray', 'lightgray'])
    ax1.axhline(0.5, color='red', linestyle='--', alpha=0.7)
    ax1.set_ylabel('Balanced Accuracy')
    ax1.set_title(f'Corner vs Cardinal\n(p={corner_cardinal_p:.4f})')
    ax1.set_ylim(0, 1)

    # Confusion matrix
    ax2 = axes[1]
    im = ax2.imshow(conf_matrix, cmap='Blues', vmin=0, vmax=1)
    ax2.set_xticks(range(9))
    ax2.set_yticks(range(9))
    ax2.set_xlabel('Predicted Position')
    ax2.set_ylabel('True Position')
    ax2.set_title(f'Position Decoding\n(acc={pos_acc:.3f})')
    plt.colorbar(im, ax=ax2, shrink=0.8)

    # Within-type errors
    ax3 = axes[2]
    chance_within_type = 3/7  # For corners: 3 other corners out of 7 non-self positions
    ax3.bar(['Within-Type\nErrors', 'Chance'], [within_type_errors, chance_within_type],
            color=['#2ca02c', 'lightgray'])
    ax3.axhline(chance_within_type, color='red', linestyle='--', alpha=0.7)
    ax3.set_ylabel('Proportion')
    ax3.set_title('Error Structure')
    ax3.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'decoder_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'corner_cardinal_acc': corner_cardinal_acc,
        'corner_cardinal_p': corner_cardinal_p,
        'position_acc': pos_acc,
        'within_type_errors': within_type_errors
    }


def main():
    parser = argparse.ArgumentParser(description='Generalised Spatial Tuning Analysis')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU')
    parser.add_argument('--num-configs', type=int, default=30,
                        help='Number of configurations to analyze')
    parser.add_argument('--sessions-per-config', type=int, default=5,
                        help='Sessions per configuration')

    args = parser.parse_args()
    set_seed(args.seed)
    ensure_dir(config.FIGURES_DIR)

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Running spatial analysis on {device}")

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

    # Get configurations
    eval_configs = get_eval_configs()
    train_configs = get_training_configs()
    all_configs = train_configs[:args.num_configs]
    print(f"Analyzing {len(all_configs)} configurations")

    # Collect hidden states
    print("\nCollecting hidden states...")
    data = collect_hidden_states_extended(agent, all_configs,
                                          num_sessions_per_config=args.sessions_per_config)
    print(f"Collected {len(data['hidden'])} hidden states")

    # Analysis 1: Low-dimensional structure
    print("\n" + "="*50)
    print("ANALYSIS 1: Spatial Subspace Structure")
    print("="*50)
    spatial_coeffs = build_spatial_coefficients(data)
    analyze_spatial_subspaces(spatial_coeffs, config.FIGURES_DIR)
    print("Saved: spatial_subspace_pca.png")

    # Analysis 2: RSA
    print("\n" + "="*50)
    print("ANALYSIS 2: Representational Similarity Analysis")
    print("="*50)
    rsa_coeffs, rsa_pvals, neural_rdm = analyze_rsa(data, config.FIGURES_DIR)
    print("Saved: rsa_analysis.png")

    # Analysis 3: Decoder generalisation
    print("\n" + "="*50)
    print("ANALYSIS 3: Decoder Generalisation")
    print("="*50)
    decoder_results = analyze_decoders(data, config.FIGURES_DIR)
    print("Saved: decoder_analysis.png")

    # Summary
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print(f"\nKey findings:")
    print(f"  RSA - Euclidean model: β={rsa_coeffs[0]:.3f}, p={rsa_pvals[0]:.4f}")
    print(f"  RSA - Node-degree model: β={rsa_coeffs[1]:.3f}, p={rsa_pvals[1]:.4f}")
    print(f"  Decoder - Corner vs Cardinal: {decoder_results['corner_cardinal_acc']:.3f}")
    print(f"  Decoder - Absolute position: {decoder_results['position_acc']:.3f}")
    print(f"\nFigures saved to: {config.FIGURES_DIR}/")


if __name__ == '__main__':
    main()
