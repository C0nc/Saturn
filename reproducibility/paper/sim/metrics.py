"""Isodepth recovery metrics (sign-invariant, monotone-invariant)."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def sign_invariant_spearman(pred: np.ndarray, true: np.ndarray) -> float:
    """|Spearman| between a recovered 1D field and the truth.

    Isodepth is only defined up to monotone reparameterization and sign flip,
    so Spearman magnitude is the natural recovery score.
    """
    r, _ = spearmanr(pred, true)
    if np.isnan(r):
        return 0.0
    return abs(float(r))


def match_recovery(recovered: np.ndarray, rho_true: np.ndarray) -> np.ndarray:
    """Per-modality recovery for an (N, M) recovered stack against (N, M) truth.

    Modality identity is known (deterministic routing / per-modality baselines),
    so we score recovered[:, m] against rho_true[:, m] directly.
    Returns array of M sign-invariant Spearman scores.
    """
    M = rho_true.shape[1]
    return np.array([sign_invariant_spearman(recovered[:, m], rho_true[:, m])
                     for m in range(M)])


def best_factor_recovery(factors: np.ndarray, rho_true_m: np.ndarray) -> float:
    """For methods that output K factors per modality (PCA/NMF/MEFISTO), the
    recovered isodepth is the factor best correlated with the truth."""
    if factors.ndim == 1:
        factors = factors[:, None]
    return max(sign_invariant_spearman(factors[:, k], rho_true_m)
              for k in range(factors.shape[1]))
