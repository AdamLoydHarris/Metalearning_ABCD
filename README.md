# Meta-RL ABCD Task

A meta-reinforcement learning framework simulating the mouse ABCD task: learning length-4 repeating sequences (A→B→C→D→A...) on a 3x3 grid maze.

## Overview

This project implements a GRU-based actor-critic agent that learns to navigate a 3x3 grid maze, visiting reward locations in a fixed sequence. The key feature is **meta-learning**: the agent learns *how to learn* new sequences within a session using only its recurrent memory, without any weight updates.

![Architecture](figures/architecture_schematic.png)

## Task Description

```
Grid Layout:          Example ABCD Config:
  0 1 2                 A . B
  3 4 5       →         . . .
  6 7 8                 D . C

Sequence: A → B → C → D → A → ...
Reward: +1 for reaching correct next state
```

- **Grid**: 3x3 maze (9 positions)
- **Actions**: 4 cardinal directions (up, down, left, right)
- **Sessions**: 100 steps each, continuous (no resets within session)
- **Configurations**: 4 unique positions randomly assigned as A, B, C, D

## Key Results

### Meta-Learning Demonstrated

The agent learns within-session using only its recurrent memory:

| Metric | Early Trials | Late Trials | Improvement |
|--------|-------------|-------------|-------------|
| P(Shortest Path) | 48% | 72% | +24% |

![Within-Session Learning](figures/within_session_learning.png)

### D→A Inference (Critical Test)

The agent correctly infers the D→A transition **before experiencing it**:
- First D→A occurrence: **47.5%** (chance = 25%)
- Later occurrences: **66.5%**

This demonstrates true sequence learning, not just stimulus-response associations.

### Future Place Cells

We discovered neurons that encode **future positions** rather than current position:

![Future Place Cells](figures/future_cells_260k_gallery.png)

- **~19% of neurons** (24/128) have stronger selectivity for future than current position
- Peak selectivity ranges from t+1 to t+8 steps ahead
- Forms a "ladder" of prospective representations at different time horizons

### Conveyor Belt Structure

Position information at different timepoints lives in **separate neural subspaces**:

![Conveyor Belt](figures/conveyor_belt_subspaces.png)

- Decoder weight correlations between timepoints ≈ 0 (orthogonal)
- The network maintains parallel spatial maps for past, present, and future

### Temporal Horizon

The hidden state encodes a ~20-step window of positions:

![Position Decoding](figures/past_future_position_decoding.png)

| Offset | Decoding Accuracy |
|--------|-------------------|
| t-10 (past) | 84% |
| t-1 (just visited) | 100% |
| t+0 (current) | 98% |
| t+1 (next) | 97% |
| t+10 (future) | 74% |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py
```

Options:
- `--num-epochs`: Training epochs (default: 500,000)
- `--seed`: Random seed (default: 42)
- `--cpu`: Force CPU training

### Evaluation

```bash
python evaluate.py
```

### Neural Analysis

```bash
python analyze.py
```

## Architecture

```
Input (14-dim)                    Output
     │                               │
     ▼                               │
┌─────────────┐                      │
│ Position(9) │                      │
│ Action(4)   │──▶ GRU(128) ──┬──▶ Actor ──▶ π(a|s)
│ Reward(1)   │       ▲       │
└─────────────┘       │       └──▶ Critic ──▶ V(s)
                      │
                   h(t-1)
                 (memory)
```

## Training Details

**Algorithm**: Advantage Actor-Critic (A2C)

**Loss Function**:
```
L = L_policy + 0.5 × L_value - 0.01 × H(π)

where:
  L_policy = -log π(a|s) × A(s,a)    [Policy gradient]
  L_value  = (V(s) - R)²             [Value function MSE]
  H(π)     = -Σ π(a) log π(a)        [Entropy bonus]
  A(s,a)   = R - V(s)                [Advantage]
```

**Hyperparameters**:
| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 |
| Discount (γ) | 0.99 |
| Entropy coefficient | 0.01 |
| Value loss coefficient | 0.5 |
| Gradient clipping | 0.5 |
| GRU hidden size | 128 |

**Meta-RL Setup**:
- 100 training configurations (fixed)
- 40 held-out evaluation configurations
- 100 steps per session
- 500,000 training epochs

## Project Structure

```
claude_abcd/
├── config.py              # Hyperparameters
├── utils.py               # Shortest paths, config generation
├── environment.py         # 3x3 grid + ABCD task
├── agent.py               # GRU actor-critic
├── train.py               # Training loop (A2C)
├── evaluate.py            # Held-out evaluation
├── analyze.py             # Neural analysis
├── train_configs.npy      # 100 training configurations
├── eval_configs.npy       # 40 held-out configurations
├── models/                # Checkpoints
├── results/               # Evaluation data
└── figures/               # Generated plots
```

## Key Figures

| Figure | Description |
|--------|-------------|
| `policy_visualization.png` | Average action probabilities by position |
| `within_session_learning.png` | Learning curves with weights frozen |
| `future_cells_260k_gallery.png` | Future place cell rate maps |
| `past_future_position_decoding.png` | Temporal decoding horizon |
| `conveyor_belt_subspaces.png` | Subspace analysis |
| `architecture_schematic.png` | Network architecture |
| `training_info.png` | Training details |

## References

Based on meta-reinforcement learning principles from:
- Wang et al. (2016) "Learning to reinforcement learn"
- Duan et al. (2016) "RL²: Fast Reinforcement Learning via Slow Reinforcement Learning"
- Wang et al. (2018): Prefrontal corte as a meta-reinforcement learning system
