# L2 Regularization Experiment

This experiment tests whether L2 regularization (weight decay) affects the learned spatial representations in the GRU meta-RL agent.

## Motivation

The base model learns strong concrete position coding but weak generalized spatial coding (node-degree, corner vs cardinal). L2 regularization encourages smaller weights, which might:
- Force the network to learn more distributed/generalizable representations
- Reduce overfitting to specific position encodings
- Potentially encourage more abstract task representations

## Configuration

Key parameter in `config.py`:
```python
WEIGHT_DECAY = 1e-4  # L2 regularization strength
```

Suggested values to try:
- `1e-5`: Weak regularization
- `1e-4`: Moderate regularization (default)
- `1e-3`: Strong regularization

## Usage

### Training
```bash
# From the repository root:
cd experiments/l2_regularization
python run_experiment.py

# Or with custom settings:
python run_experiment.py --weight-decay 1e-3 --num-epochs 200000
```

### Analysis
After training, use the analysis scripts from the main repo:
```bash
# From repo root
python analyze_spatial.py --checkpoint experiments/l2_regularization/models/final_model.pt
```

## Files

- `config.py` - Experiment configuration with L2 settings
- `agent.py` - Modified agent with weight_decay in optimizer
- `train.py` - Training loop
- `run_experiment.py` - Entry point script
- `models/` - Saved checkpoints
- `results/` - Training history and analysis results
- `figures/` - Generated plots

## Comparison

After training, compare the representations with the base model:
1. RSA analysis: Does node-degree explain more variance?
2. Decoder generalization: Can we decode corner vs cardinal better?
3. Sequence alignment: Are A/B/C/D trajectories more consistent?
