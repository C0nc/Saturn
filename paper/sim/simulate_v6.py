"""v6 simulator -- Saturn's generative model for ANY number of modalities.

Generalises v5 (M=2) to M modalities with a DIRECTED coupling matrix:

    feature j of modality m:  eta_j(x) = base_j + P_{m<-m, j}( d_m(x) )
                                          + sum_{k != m} C[m, k] * mask_{m,k,j} * P_{m<-k, j}( d_k(x) )

  d_m      one GASTON-like isodepth per modality (ramp + curvature); axes are
           separated by the angles in `axis_angles_deg`
  P        piecewise-linear curves along ONE axis, breakpoints shared per axis
  C[m, k]  strength with which axis k shapes modality m (relative to own = 1).
           ASYMMETRIC by design: C[RNA, H3K4me3] need not equal C[H3K4me3, RNA]
  mask     each feature is coupled to axis k with probability coupled_frac

Modality kinds: "rna" -> NB counts -> GLM-PCs;  "peaks" -> Poisson counts -> LSI
(ATAC or CUT&Tag histone marks). Returns the true axes, the four-way parts, and
the TRUE all-feature coupling share matrix to score Saturn's `coupling()` against.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import sys
import numpy as np

from simulate_v5 import _axis_field, _breakpoints, _pw_curves, _nb, _softmax_rows
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/src")
from saturn.preprocess import rna_glmpc, atac_lsi                 # noqa: E402


@dataclass
class SimConfigV6:
    n_side: int = 30
    kinds: List[str] = field(default_factory=lambda: ["rna", "peaks", "peaks"])
    names: List[str] = field(default_factory=lambda: ["RNA", "H3K4me3", "H3K27me3"])
    n_feat: List[int] = field(default_factory=lambda: [500, 800, 800])
    depth: List[float] = field(default_factory=lambda: [2000.0, 1500.0, 1500.0])
    K: List[int] = field(default_factory=lambda: [10, 15, 15])
    axis_angles_deg: List[float] = field(default_factory=lambda: [0.0, 60.0, 120.0])
    L: int = 4                                   # domains per axis
    curvature: float = 0.6
    # C[m][k]: axis k -> modality m. Default: both marks shape RNA (chromatin
    # drives expression), RNA barely shapes the marks, the marks mildly each other.
    C: Optional[List[List[float]]] = None
    coupled_frac: float = 0.4
    jump_frac: float = 0.3
    flat_frac: float = 0.3
    nb_theta: float = 10.0
    seed: int = 0

    def coupling(self) -> np.ndarray:
        M = len(self.kinds)
        if self.C is not None:
            return np.asarray(self.C, dtype=float)
        C = np.full((M, M), 0.15)
        if M >= 2:
            C[0, 1:] = 0.6                        # marks -> RNA
            C[1:, 0] = 0.2                        # RNA -> marks
        np.fill_diagonal(C, 0.0)
        return C


@dataclass
class SimDataV6:
    coords: np.ndarray
    d: np.ndarray                    # (N, M) true axes
    targets: List[np.ndarray]        # per-modality globally scaled embeddings
    counts: List[np.ndarray]
    true_share: np.ndarray           # (M, M) all-feature variance share of axis k in modality m
    C: np.ndarray                    # the coupling matrix used
    config: SimConfigV6 = None


def simulate_v6(cfg: SimConfigV6) -> SimDataV6:
    rng = np.random.default_rng(cfg.seed)
    M = len(cfg.kinds)
    C = cfg.coupling()
    xs = np.linspace(0, 1, cfg.n_side)
    gx, gy = np.meshgrid(xs, xs)
    coords = np.stack([gx.ravel(), gy.ravel()], 1)
    coords = coords + rng.normal(0, 0.3 / cfg.n_side, size=coords.shape)
    N = coords.shape[0]

    phi = rng.uniform(0, 2 * np.pi)
    d = np.stack([_axis_field(coords, phi + np.deg2rad(a), cfg.curvature)
                  for a in cfg.axis_angles_deg[:M]], 1)
    bps = [_breakpoints(d[:, k], cfg.L, rng) for k in range(M)]
    kw = dict(jump_frac=cfg.jump_frac, flat_frac=cfg.flat_frac, rng=rng)

    counts, targets = [], []
    share = np.zeros((M, M))
    for m in range(M):
        G = cfg.n_feat[m]
        parts = []
        for k in range(M):
            amp = 1.0 if k == m else C[m, k]
            P = _pw_curves(d[:, k], bps[k], G, amp, **kw) if amp > 0 else np.zeros((N, G))
            if k != m:
                P = P * (rng.random(G) < cfg.coupled_frac)[None, :]
            parts.append(P)
        v = np.stack([p.var(0) for p in parts])                     # (M, G)
        share[m] = (v / (v.sum(0, keepdims=True) + 1e-12)).mean(1)
        base = rng.normal(0, 1.0 if cfg.kinds[m] == "rna" else 0.5, size=G)
        eta = base[None, :] + sum(parts)
        lib = rng.lognormal(np.log(cfg.depth[m]), 0.3, size=(N, 1))
        mu = lib * _softmax_rows(eta)
        if cfg.kinds[m] == "rna":
            Y = _nb(mu, cfg.nb_theta, rng)
            T = rna_glmpc(Y, cfg.K[m])
        else:
            Y = rng.poisson(mu).astype(np.float32)
            T = atac_lsi(Y, cfg.K[m], drop_first=1)
        counts.append(Y); targets.append(T)

    return SimDataV6(coords=coords.astype(np.float32), d=d.astype(np.float32),
                     targets=targets, counts=counts, true_share=share, C=C, config=cfg)
