"""Deterministic seeding for Saturn fits.

Call `set_all_seeds(seed)` BEFORE model construction, not after — the positional
encoding B matrix is drawn during __init__ and must be under the seeded RNG.

Same-GPU reproducibility requires the two extra knobs (env var + deterministic algos).
Cross-GPU-architecture may still drift at ~1e-6 in intermediates; recovery numbers
(rank-based Spearman) round to the same 3 decimals.
"""
import os
import random
import numpy as np
import torch


def set_all_seeds(seed: int, *, deterministic_cudnn: bool = True,
                    strict_gpu: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if strict_gpu:
        # cuBLAS workspace config needed for deterministic matmul (PyTorch requires this)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # older torch signature
            torch.use_deterministic_algorithms(True)
