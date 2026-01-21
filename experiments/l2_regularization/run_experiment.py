#!/usr/bin/env python
"""
Run the L2 Regularization Experiment.

Usage:
    python run_experiment.py                    # Train with default weight_decay=1e-4
    python run_experiment.py --weight-decay 1e-3  # Train with stronger regularization
    python run_experiment.py --num-epochs 100000  # Shorter training run
"""

import os
import sys

# Ensure we can import from parent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Now run training
from experiments.l2_regularization.train import main

if __name__ == '__main__':
    main()
