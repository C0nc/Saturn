"""Simulate multi-modal spatial data with known per-modality isodepth fields.

Each modality m has a ground-truth scalar isodepth field rho_m(x,y). Inter-modality
alignment is controlled by `corr`: rho_m = corr * z_shared + sqrt(1-corr^2) * z_m,
where z_shared and z_m are independent smooth random fields. Per-modality features
(genes/peaks) are monotone-but-nonlinear functions of rho_m, emitted as counts.

This is the substrate for the 5-modality "pressure test": as we add modalities and
vary their gradient correlation, how well can each method recover every rho_m?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


def _smooth_field(coords: np.ndarray, rng: np.random.Generator,
                  n_basis: int = 6, length_scale: float = 0.35) -> np.ndarray:
    """A smooth scalar field = sum of random 2D Gaussian bumps + a linear ramp."""
    N = coords.shape[0]
    f = np.zeros(N)
    # random gaussian bumps
    centers = rng.uniform(0, 1, size=(n_basis, 2))
    amps = rng.normal(0, 1, size=n_basis)
    for c, a in zip(centers, amps):
        d2 = ((coords - c) ** 2).sum(1)
        f += a * np.exp(-d2 / (2 * length_scale ** 2))
    # plus a random linear ramp so fields have a dominant direction
    theta = rng.uniform(0, 2 * np.pi)
    f += 1.5 * (coords @ np.array([np.cos(theta), np.sin(theta)]))
    f = (f - f.mean()) / (f.std() + 1e-8)
    return f


@dataclass
class SimConfig:
    n_side: int = 40                 # grid is n_side x n_side spots
    M: int = 5                       # number of modalities
    corr: float = 0.5                # inter-modality gradient correlation in [0,1]
    genes_per_mod: int = 80
    libsize: float = 2000.0
    noise: float = 1.0               # multiplicative dispersion knob
    response: str = "nonmonotone"    # gene response vs isodepth:
                                     #   "nonmonotone" -> Gaussian-bump markers
                                     #     (switch on then off along the axis;
                                     #      breaks linear/PCA recoverability)
                                     #   "monotone"    -> sigmoidal tanh (easy)
    bump_frac: float = 0.8           # fraction of genes with non-monotone bumps
    bump_width: float = 0.5          # width of Gaussian bumps (in rho z-units)
    seed: int = 0


@dataclass
class SimData:
    coords: np.ndarray                       # (N, 2)
    rho_true: np.ndarray                     # (N, M)
    counts: List[np.ndarray]                 # M arrays of (N, G_m)
    config: SimConfig = field(default=None)


def simulate(cfg: SimConfig) -> SimData:
    rng = np.random.default_rng(cfg.seed)
    # spot grid in [0,1]^2 with mild jitter (so it isn't a perfect lattice)
    xs = np.linspace(0, 1, cfg.n_side)
    gx, gy = np.meshgrid(xs, xs)
    coords = np.stack([gx.ravel(), gy.ravel()], 1)
    coords = coords + rng.normal(0, 0.3 / cfg.n_side, size=coords.shape)
    N = coords.shape[0]

    # shared + modality-specific smooth fields -> controlled correlation
    z_shared = _smooth_field(coords, rng)
    rho_true = np.zeros((N, cfg.M))
    c = float(np.clip(cfg.corr, 0.0, 1.0))
    for m in range(cfg.M):
        z_m = _smooth_field(coords, rng)
        r = c * z_shared + np.sqrt(max(1 - c * c, 0.0)) * z_m
        rho_true[:, m] = (r - r.mean()) / (r.std() + 1e-8)

    # emit counts: each gene's expression is a NONLINEAR function of rho_m.
    # By default most genes are non-monotone "bump" markers (active in a window
    # of isodepth, like cortical-layer / spatial-domain markers) so that the
    # rank-order of expression does NOT follow rho monotonically -- this is what
    # makes isodepth recovery a genuine test rather than something a linear PCA
    # picks up trivially.
    counts: List[np.ndarray] = []
    for m in range(cfg.M):
        rho = rho_true[:, m]
        lo, hi = rho.min(), rho.max()
        G = cfg.genes_per_mod
        base = rng.uniform(-1.0, 0.5, size=G)
        n_bump = int(round(cfg.bump_frac * G))
        is_bump = np.zeros(G, dtype=bool)
        is_bump[:n_bump] = True
        rng.shuffle(is_bump)

        lograte = np.zeros((N, G))
        # non-monotone Gaussian bumps tiling the isodepth axis
        centers = rng.uniform(lo, hi, size=G)
        widths = cfg.bump_width * rng.uniform(0.5, 1.5, size=G)
        amp = rng.uniform(1.0, 2.5, size=G)
        bump = amp[None, :] * np.exp(
            -((rho[:, None] - centers[None, :]) ** 2) / (2 * widths[None, :] ** 2))
        # monotone sigmoidal genes for the remainder
        slope = rng.normal(0, 1.5, size=G)
        bias = rng.normal(0, 1.0, size=G)
        mono = np.tanh(slope[None, :] * rho[:, None] + bias[None, :])

        lograte = base[None, :] + np.where(is_bump[None, :], bump, mono)
        rate = np.exp(lograte)
        rate = rate / rate.mean(1, keepdims=True)  # normalize per spot
        lam = cfg.libsize * rate / G * cfg.noise
        counts.append(rng.poisson(lam).astype(np.float32))

    return SimData(coords=coords.astype(np.float32), rho_true=rho_true.astype(np.float32),
                   counts=counts, config=cfg)
