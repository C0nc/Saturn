"""Extended simulator for isodepth-recovery + denoising benchmark.

Adds:
  * Four sim REGIMES that shape the ground-truth isodepth field:
      - "smooth"     — original (random smooth field + linear ramp)
      - "domains"    — piecewise-constant zones with sharp boundaries
      - "cascade"    — modality-specific temporal offset (RNA leads, chromatin lags)
      - "multiscale" — fast + slow gradients superposed
  * NoiseSpec knob dictating post-Poisson corruption:
      - libsize    : mean library size (Poisson depth)
      - dropout_p  : Bernoulli dropout applied to counts
      - lane_sigma : DBiT-style per-row multiplicative log-normal noise
      - ambient_alpha : fraction of reads sampled from a shared ambient profile
  * Returns clean μ_true (spot × gene rate BEFORE Poisson/dropout/lane/ambient)
    so downstream can score denoising against a real ground truth.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


# ---------------------------------------------------------------- helpers
def _smooth_field(coords, rng, n_basis=6, length_scale=0.35):
    N = coords.shape[0]
    f = np.zeros(N)
    centers = rng.uniform(0, 1, size=(n_basis, 2))
    amps = rng.normal(0, 1, size=n_basis)
    for c, a in zip(centers, amps):
        d2 = ((coords - c) ** 2).sum(1)
        f += a * np.exp(-d2 / (2 * length_scale ** 2))
    theta = rng.uniform(0, 2 * np.pi)
    f += 1.5 * (coords @ np.array([np.cos(theta), np.sin(theta)]))
    f = (f - f.mean()) / (f.std() + 1e-8)
    return f


def _domain_field(coords, rng, n_zones=4):
    """Piecewise-constant field: k Voronoi zones with random values +
    small smooth perturbation so boundaries aren't perfectly flat."""
    N = coords.shape[0]
    centers = rng.uniform(0.05, 0.95, size=(n_zones, 2))
    vals = rng.normal(0, 1.2, size=n_zones)
    d2 = ((coords[:, None, :] - centers[None, :, :]) ** 2).sum(-1)   # (N, n_zones)
    zone = d2.argmin(1)
    f = vals[zone]
    # smooth perturbation
    f = f + 0.25 * _smooth_field(coords, rng)
    f = (f - f.mean()) / (f.std() + 1e-8)
    return f


def _multiscale_field(coords, rng):
    """Fast (low length-scale) + slow (high length-scale) fields superposed."""
    slow = _smooth_field(coords, rng, n_basis=4, length_scale=0.5)
    fast = _smooth_field(coords, rng, n_basis=15, length_scale=0.08)
    f = 0.7 * slow + 0.5 * fast
    f = (f - f.mean()) / (f.std() + 1e-8)
    return f


# ---------------------------------------------------------------- config
@dataclass
class NoiseSpec:
    libsize: float = 2000.0          # mean per-spot library size (Poisson depth)
    dropout_p: float = 0.0           # Bernoulli dropout probability
    lane_sigma: float = 0.0          # log-normal σ for per-row multiplicative noise
    ambient_alpha: float = 0.0       # fraction of reads from shared ambient profile

    def label(self) -> str:
        p = f"L{int(self.libsize)}"
        if self.dropout_p > 0:    p += f".d{self.dropout_p:.1f}"
        if self.lane_sigma > 0:   p += f".s{self.lane_sigma:.1f}"
        if self.ambient_alpha > 0:p += f".a{self.ambient_alpha:.1f}"
        return p


@dataclass
class SimConfigV2:
    n_side: int = 30
    M: int = 2
    corr: float = 0.4
    genes_per_mod: int = 60
    regime: str = "smooth"           # smooth | domains | cascade | multiscale
    bump_frac: float = 0.8
    bump_width: float = 0.5
    cascade_shift: float = 0.4       # inter-modality ρ shift (units of ρ std)
    # Cross-modality emission strength.
    #
    # `corr` correlates the LATENT fields (rho_m = c*z_shared + sqrt(1-c^2)*z_m)
    # but modality m's OBSERVATION has always been a function of rho_m alone.
    # Saturn's joint decoder is dec_m(Phi) — it assumes modality m's features
    # depend on the WHOLE isodepth vector — so with cross_weight=0 that decoder
    # is over-specified for the data and its spare capacity is what lets a folded
    # rho_m survive (it disambiguates the fold using rho_of_another_modality).
    #
    # cross_weight > 0 makes each gene respond to a mixture dominated by its own
    # modality plus a random contribution from the others, which is the structure
    # the joint decoder is built for.
    #   0.0 -> identical to the previous generator (pure rho_m)
    #   0.3 -> mild cross-talk;  1.0 -> other modalities as strong as own
    cross_weight: float = 0.0
    noise: NoiseSpec = field(default_factory=NoiseSpec)
    seed: int = 0


@dataclass
class SimDataV2:
    coords: np.ndarray               # (N, 2)
    rho_true: np.ndarray             # (N, M)
    counts: List[np.ndarray]         # M arrays of (N, G_m) - noisy observations
    mu_true: List[np.ndarray]        # M arrays of (N, G_m) - clean rates (denoise GT)
    config: SimConfigV2 = None


# ---------------------------------------------------------------- regimes
def _build_rho(cfg: SimConfigV2, coords, rng) -> np.ndarray:
    """Return (N, M) ground-truth isodepth stack per regime."""
    M = cfg.M
    c = float(np.clip(cfg.corr, 0.0, 1.0))
    if cfg.regime == "smooth":
        z_shared = _smooth_field(coords, rng)
        rho = np.zeros((coords.shape[0], M))
        for m in range(M):
            z_m = _smooth_field(coords, rng)
            r = c * z_shared + np.sqrt(max(1 - c * c, 0.0)) * z_m
            rho[:, m] = (r - r.mean()) / (r.std() + 1e-8)
        return rho
    if cfg.regime == "domains":
        z_shared = _domain_field(coords, rng, n_zones=4)
        rho = np.zeros((coords.shape[0], M))
        for m in range(M):
            z_m = _domain_field(coords, rng, n_zones=4 + m)
            r = c * z_shared + np.sqrt(max(1 - c * c, 0.0)) * z_m
            rho[:, m] = (r - r.mean()) / (r.std() + 1e-8)
        return rho
    if cfg.regime == "cascade":
        # base field shared across modalities, per-modality lagged shift along
        # the dominant gradient direction — mimics RNA leading chromatin (or vice versa).
        base = _smooth_field(coords, rng)
        # dominant direction from base's gradient (linear ramp part dominates)
        theta = rng.uniform(0, 2 * np.pi)
        u = np.array([np.cos(theta), np.sin(theta)])
        proj = coords @ u
        rho = np.zeros((coords.shape[0], M))
        for m in range(M):
            shift = (m - (M - 1) / 2.0) * cfg.cascade_shift
            # small per-modality independent perturbation so the fields aren't identical
            perturb = 0.15 * _smooth_field(coords, rng)
            r = base + shift * (proj - proj.mean()) + perturb
            rho[:, m] = (r - r.mean()) / (r.std() + 1e-8)
        return rho
    if cfg.regime == "multiscale":
        z_shared = _multiscale_field(coords, rng)
        rho = np.zeros((coords.shape[0], M))
        for m in range(M):
            z_m = _multiscale_field(coords, rng)
            r = c * z_shared + np.sqrt(max(1 - c * c, 0.0)) * z_m
            rho[:, m] = (r - r.mean()) / (r.std() + 1e-8)
        return rho
    raise ValueError(f"unknown regime {cfg.regime!r}")


# ---------------------------------------------------------------- emission
def _effective_rho(rho_all: np.ndarray, m: int, cfg: SimConfigV2, rng) -> np.ndarray:
    """Per-gene effective isodepth for modality m, shape (N, G).

    Each gene of modality m responds to its own isodepth plus (when
    cross_weight > 0) a random contribution from the other modalities:

        rho_g = rho_m + cross_weight * sum_{k != m} a_gk * rho_k      (z-scored)

    With cross_weight = 0 every gene sees exactly rho_m, reproducing the
    original generator. With cross_weight > 0 the observation of modality m is
    genuinely a function of the whole isodepth vector — the structure Saturn's
    joint decoder dec_m(Phi) is designed to capture.
    """
    N, M = rho_all.shape
    G = cfg.genes_per_mod
    rho_g = np.repeat(rho_all[:, m][:, None], G, axis=1)          # (N, G)
    if cfg.cross_weight > 0 and M > 1:
        for k in range(M):
            if k == m:
                continue
            a = rng.normal(0.0, 1.0, size=G)                       # per-gene loading
            rho_g = rho_g + cfg.cross_weight * a[None, :] * rho_all[:, k][:, None]
        mu = rho_g.mean(0, keepdims=True)
        sd = rho_g.std(0, keepdims=True) + 1e-8
        rho_g = (rho_g - mu) / sd                                  # keep rho z-scored
    return rho_g


def _clean_rate(rho_all: np.ndarray, m: int, cfg: SimConfigV2, rng) -> np.ndarray:
    """Clean rate matrix (N, G) for modality m: unit-mean per spot, pre-noise."""
    G = cfg.genes_per_mod
    rho_g = _effective_rho(rho_all, m, cfg, rng)                   # (N, G)
    base = rng.uniform(-1.0, 0.5, size=G)
    n_bump = int(round(cfg.bump_frac * G))
    is_bump = np.zeros(G, dtype=bool)
    is_bump[:n_bump] = True
    rng.shuffle(is_bump)
    lo, hi = rho_g.min(), rho_g.max()
    centers = rng.uniform(lo, hi, size=G)
    widths = cfg.bump_width * rng.uniform(0.5, 1.5, size=G)
    amp = rng.uniform(1.0, 2.5, size=G)
    bump = amp[None, :] * np.exp(
        -((rho_g - centers[None, :]) ** 2) / (2 * widths[None, :] ** 2))
    slope = rng.normal(0, 1.5, size=G)
    bias = rng.normal(0, 1.0, size=G)
    mono = np.tanh(slope[None, :] * rho_g + bias[None, :])
    lograte = base[None, :] + np.where(is_bump[None, :], bump, mono)
    rate = np.exp(lograte)
    rate = rate / rate.mean(1, keepdims=True)     # unit-mean per spot
    return rate.astype(np.float32)


def _apply_noise(clean_mu: np.ndarray, spec: NoiseSpec, rng) -> np.ndarray:
    """Emit noisy counts given a per-spot per-gene rate.
    Order: ambient mix → lane multiplicative → depth scale → Poisson → dropout.
    """
    mu = clean_mu.copy()

    # 1) ambient contamination: mix in fraction of a shared spatial-flat profile
    if spec.ambient_alpha > 0:
        gene_mean = mu.mean(0, keepdims=True)             # (1, G)
        gene_mean = gene_mean / gene_mean.mean() * mu.mean()  # normalize scale
        mu = (1.0 - spec.ambient_alpha) * mu + spec.ambient_alpha * gene_mean

    # 2) lane stripes: log-normal per-row multiplicative
    if spec.lane_sigma > 0:
        lane = np.exp(rng.normal(0, spec.lane_sigma, size=(mu.shape[0], 1)))
        mu = mu * lane

    # 3) library size scale, then Poisson
    G = mu.shape[1]
    lam = mu * (spec.libsize / G)
    counts = rng.poisson(lam.clip(min=1e-8)).astype(np.float32)

    # 4) dropout on the counts
    if spec.dropout_p > 0:
        mask = (rng.random(counts.shape) >= spec.dropout_p).astype(np.float32)
        counts = counts * mask

    return counts


# ---------------------------------------------------------------- main entry
def simulate_v2(cfg: SimConfigV2) -> SimDataV2:
    rng = np.random.default_rng(cfg.seed)
    xs = np.linspace(0, 1, cfg.n_side)
    gx, gy = np.meshgrid(xs, xs)
    coords = np.stack([gx.ravel(), gy.ravel()], 1)
    coords = coords + rng.normal(0, 0.3 / cfg.n_side, size=coords.shape)
    coords = coords.astype(np.float32)

    rho_true = _build_rho(cfg, coords, rng)
    counts_list, mu_list = [], []
    for m in range(cfg.M):
        clean = _clean_rate(rho_true, m, cfg, rng)     # unit-mean rate
        # rescale clean to per-spot rate at nominal depth so "clean" and "counts" live on same scale
        G = clean.shape[1]
        clean_at_depth = clean * (cfg.noise.libsize / G)   # expected counts, no noise
        # noisy counts
        noisy = _apply_noise(clean, cfg.noise, rng)
        mu_list.append(clean_at_depth.astype(np.float32))
        counts_list.append(noisy)
    return SimDataV2(coords=coords, rho_true=rho_true.astype(np.float32),
                     counts=counts_list, mu_true=mu_list, config=cfg)
