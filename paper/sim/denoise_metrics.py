"""Denoising quality metrics: score reconstructed expression against a clean μ_true.

Scale-invariant per-gene Pearson (each gene is z-scored, so a global depth mismatch
between methods does not dominate the score). Per-spot cosine on the log scale
captures overall profile similarity. log-MSE gives a magnitude-aware error.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import pearsonr


def _log_scale(mu):
    return np.log1p(mu)


def per_gene_pearson(mu_hat: np.ndarray, mu_true: np.ndarray) -> np.ndarray:
    """Return per-gene Pearson r between reconstructed and clean expression.
    Both should be spot × gene.
    """
    lo_h, lo_t = _log_scale(mu_hat), _log_scale(mu_true)
    G = lo_h.shape[1]
    r = np.zeros(G)
    for g in range(G):
        if np.std(lo_t[:, g]) < 1e-8 or np.std(lo_h[:, g]) < 1e-8:
            r[g] = np.nan
            continue
        rr, _ = pearsonr(lo_h[:, g], lo_t[:, g])
        r[g] = rr
    return r


def per_spot_cosine(mu_hat: np.ndarray, mu_true: np.ndarray) -> np.ndarray:
    """Per-spot cosine similarity on log-scale profiles."""
    lo_h, lo_t = _log_scale(mu_hat), _log_scale(mu_true)
    # subtract per-spot mean to make comparison scale-neutral
    lo_h = lo_h - lo_h.mean(1, keepdims=True)
    lo_t = lo_t - lo_t.mean(1, keepdims=True)
    num = (lo_h * lo_t).sum(1)
    denom = (np.linalg.norm(lo_h, axis=1) * np.linalg.norm(lo_t, axis=1)).clip(1e-12)
    return num / denom


def log_mse(mu_hat, mu_true):
    return float(np.mean((_log_scale(mu_hat) - _log_scale(mu_true)) ** 2))


def summarize(mu_hat, mu_true) -> dict:
    """Full summary dict for one (recon, truth) pair."""
    r_g = per_gene_pearson(mu_hat, mu_true)
    c_s = per_spot_cosine(mu_hat, mu_true)
    return dict(
        gene_pearson_mean=float(np.nanmean(r_g)),
        gene_pearson_median=float(np.nanmedian(r_g)),
        spot_cosine_mean=float(np.mean(c_s)),
        log_mse=log_mse(mu_hat, mu_true),
    )
