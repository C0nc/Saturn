"""Baseline isodepth recovery: PCA, NMF, and MEFISTO.

Each baseline returns an (N, M) recovered-isodepth stack so it can be scored with
sim.metrics.match_recovery against the ground-truth fields.

PCA / NMF: run per modality, keep top-K factors, pick the factor best correlated
with each truth field at scoring time (handled by metrics.best_factor_recovery,
but here we expose the full factor set and pick the top component as the default
recovered isodepth, with `recover_*_factors` for oracle factor matching).

MEFISTO: a single multi-view smooth factor model (mofapy2) jointly across
modalities with spatial coordinates as the covariate; factor weights per view map
back to a per-modality recovered field via the view-specific factor activity.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler

from metrics import best_factor_recovery


def _log_norm(counts: np.ndarray) -> np.ndarray:
    lib = counts.sum(1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log1p(counts / lib * np.median(lib))


# --------------------------------------------------------------------------- #
def recover_pca(counts: List[np.ndarray], n_factors: int = 10) -> List[np.ndarray]:
    """Return per-modality factor matrices (N, K)."""
    out = []
    for X in counts:
        Xn = StandardScaler().fit_transform(_log_norm(X))
        k = min(n_factors, Xn.shape[1], Xn.shape[0] - 1)
        out.append(PCA(n_components=k).fit_transform(Xn))
    return out


def recover_nmf(counts: List[np.ndarray], n_factors: int = 10) -> List[np.ndarray]:
    out = []
    for X in counts:
        Xn = _log_norm(X)
        Xn = Xn - Xn.min()
        k = min(n_factors, Xn.shape[1])
        out.append(NMF(n_components=k, init="nndsvda", max_iter=400).fit_transform(Xn))
    return out


def recover_mefisto(counts: List[np.ndarray], coords: np.ndarray,
                    n_factors: int = 10, verbose: bool = False,
                    n_iter: int = 30) -> Optional[List[np.ndarray]]:
    """Joint smooth multi-view factor model. Returns per-modality factor
    matrices (the same shared factors, exposed per view for scoring). Returns
    None if mofapy2 is unavailable so the caller can skip gracefully."""
    try:
        from mofapy2.run.entry_point import entry_point
    except Exception:
        return None

    N = coords.shape[0]
    views, view_names = [], []
    for m, X in enumerate(counts):
        views.append(_log_norm(X).astype(np.float32))
        view_names.append(f"mod{m}")

    ent = entry_point()
    # mofapy2 expects data nested as data[view][group]; single group here
    data = [[v] for v in views]
    ent.set_data_options(scale_views=True)
    ent.set_data_matrix(data, views_names=view_names,
                        samples_names=[[str(i) for i in range(N)]],
                        likelihoods=["gaussian"] * len(views))
    ent.set_model_options(factors=n_factors, spikeslab_weights=True, ard_weights=True)
    ent.set_train_options(iter=n_iter, convergence_mode="fast", verbose=verbose, seed=0)
    # MEFISTO: spatial coords as smooth covariate
    ent.set_covariates([coords.astype(np.float32)], covariates_names=["x", "y"])
    # sparse GP (inducing points) keeps the spatial smoothing tractable: exact
    # MEFISTO is O(N^3) per iteration which dominates the whole benchmark.
    try:
        ent.set_smooth_options(n_grid=8, start_opt=10, opt_freq=10,
                               sparseGP=True, frac_inducing=0.4)
    except TypeError:
        ent.set_smooth_options(n_grid=8, start_opt=10, opt_freq=10)
    ent.build()
    ent.run()

    Z = ent.model.nodes["Z"].getExpectation()  # (N, K) shared factors
    # expose the same shared factors for every modality (scoring picks best)
    return [np.asarray(Z) for _ in counts]
