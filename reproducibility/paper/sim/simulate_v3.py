"""v3 simulator — coupled, nonlinear, multi-modal.

WHAT CHANGED FROM v2, AND WHY
-----------------------------
v2 had a single `corr` knob that correlated the LATENT fields:
    rho_m = corr*z_shared + sqrt(1-corr^2)*z_m
but modality m's OBSERVATION was a function of rho_m ALONE
(`_clean_rate(rho_m)`). Saturn's joint decoder is dec_m(Phi) — it assumes
modality m's features depend on the WHOLE isodepth vector. So under v2 that
decoder modelled structure the data did not contain: it was over-specified, and
its spare capacity is what let a FOLDED rho_m survive training (it could
disambiguate the two branches of the fold by reading another modality's rho,
so reconstruction barely suffered and the loss could not separate good from
folded).

v3 makes the generator match the model. ONE knob, `coupling`, now drives both:

  (1) FIELD sharing      rho_m = c*rho_shared + sqrt(1-c^2)*rho_m_priv
                         (identical formula to v2's `corr`, same meaning)

  (2) EMISSION coupling  each gene of modality m responds to a mixture of its
                         own isodepth and the others, then passes through a
                         NONLINEAR response (Gaussian bump or tanh):

                            eta_g = sqrt(1-c^2)*rho_m + c*sum_{k!=m} a_gk*rho_k
                            lograte_g = base_g + [bump | mono](eta_g)

  c = 0  -> independent modalities, private emission (v2 behaviour)
  c = 1  -> fully shared field, fully coupled emission

DEFAULTS CHANGED
  bump_frac 0.8 -> 0.6.  bump genes are non-monotone, so they carry NO
  information about which direction rho runs; only the monotone remainder does.
  At 0.8 only 20% of genes are informative about orientation, which sits right
  on the identifiability cliff — measured fold rate 3/24 with mean 0.907.
  At 0.6 (40% monotone) folds vanish: 0/24 with mean 0.9994.

  Noise is OFF by default (clean counts). Use v2 for noise-robustness work.

Regimes (smooth / domains / cascade) are inherited from v2 unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np

# reuse v2's field generators and noise model — identical definitions
from simulate_v2 import (_smooth_field, _domain_field, _multiscale_field,
                          NoiseSpec, _apply_noise)


@dataclass
class SimConfigV3:
    n_side: int = 30
    M: int = 2
    genes_per_mod: int = 60
    regime: str = "smooth"            # smooth | domains | cascade
    coupling: float = 0.5             # replaces `corr`: field sharing AND emission cross-talk
    # --- two-knob split (2026-09-18) --------------------------------------- #
    # `coupling` drives BOTH mechanisms at once, which CONFOUNDS them:
    #   field sharing    rho_m = c*z_shared + sqrt(1-c^2)*z_m   -> corr = c^2
    #   emission mixing  eta_g = sqrt(1-c^2)*rho_m + c*sum a_gk*rho_k
    # So a high-c run cannot distinguish "the fields genuinely correlate" from
    # "the emission mixes across modalities" -- and that is exactly the
    # distinction a disentanglement claim rests on. These override `coupling`
    # when set; leaving them None reproduces the single-knob behaviour EXACTLY,
    # so every existing result stays valid.
    field_coupling: float | None = None      # how much the TRUE rho fields share
    emission_coupling: float | None = None   # how much modality m's genes read other rho
    bump_frac: float = 0.6            # 0.6 => 40% monotone genes carry orientation
    bump_width: float = 0.5
    cascade_shift: float = 0.4
    noise: NoiseSpec = field(default_factory=lambda: NoiseSpec(libsize=2000.0))
    seed: int = 0


@dataclass
class SimDataV3:
    coords: np.ndarray                # (N, 2)
    rho_true: np.ndarray              # (N, M)
    counts: List[np.ndarray]          # M arrays (N, G)
    mu_true: List[np.ndarray]         # M arrays (N, G) clean rates
    config: SimConfigV3 = None
    # GROUND-TRUTH latent decomposition, exposed so shared/private models can be
    # scored directly instead of through a proxy. rho_m = cf*z_shared +
    # sqrt(1-cf^2)*z_priv[:, m], with z_priv orthogonalised against z_shared and
    # against every earlier private field.
    z_shared: np.ndarray = None       # (N,)
    z_priv: np.ndarray = None         # (N, M)


def _field(regime, coords, rng, zones=4):
    if regime == "smooth":   return _smooth_field(coords, rng)
    if regime == "domains":  return _domain_field(coords, rng, n_zones=zones)
    if regime == "cascade":  return _smooth_field(coords, rng)
    raise ValueError(f"unknown regime {regime!r}")


def _orth(v: np.ndarray, *bases: np.ndarray) -> np.ndarray:
    """Gram-Schmidt: remove every vector in `bases` from `v`, z-score the residual.

    WHY THIS IS NEEDED. `_smooth_field` adds a linear ramp
    1.5*(coords @ [cos t, sin t]) with random direction t, and that ramp
    dominates the RBF component. Two nominally independent fields therefore
    correlate by however their ramp directions happen to align: measured
    |corr| = 0.34 +/- 0.27 at coupling=0, and 0.33 / 0.37 / 0.42 at
    coupling = 0.3 / 0.5 / 0.7 — i.e. the knob did essentially nothing below
    ~0.7 and the realised correlation was dominated by seed noise. (This
    affects v1 and v2 identically; it is not specific to v3.)

    Orthogonalising the private field against the shared one restores the
    intended identity corr(rho_i, rho_j) = c^2 exactly.
    """
    v = v - v.mean()
    for basis in bases:
        b = basis - basis.mean()
        denom = float(b @ b)
        if denom > 1e-12:
            v = v - (float(v @ b) / denom) * b
    return v / (v.std() + 1e-8)


def _cf(cfg) -> float:
    """Field-sharing strength. Falls back to `coupling` so legacy configs are
    bit-identical."""
    v = getattr(cfg, "field_coupling", None)
    return float(np.clip(cfg.coupling if v is None else v, 0.0, 1.0))


def _ce(cfg) -> float:
    """Emission cross-talk strength. Falls back to `coupling`."""
    v = getattr(cfg, "emission_coupling", None)
    return float(np.clip(cfg.coupling if v is None else v, 0.0, 1.0))


def _build_rho(cfg: SimConfigV3, coords, rng, return_parts: bool = False):
    """(N, M) ground-truth isodepth. `coupling` plays the role v2's `corr` had,
    but — unlike v1/v2 — it now controls the realised correlation exactly."""
    N, M = coords.shape[0], cfg.M
    c = _cf(cfg)                       # FIELD coupling only
    rho = np.zeros((N, M))
    if cfg.regime == "cascade":
        # shared base + per-modality lag along a random direction
        base = _smooth_field(coords, rng)
        theta = rng.uniform(0, 2 * np.pi)
        proj = coords @ np.array([np.cos(theta), np.sin(theta)])
        for m in range(M):
            shift = (m - (M - 1) / 2.0) * cfg.cascade_shift
            perturb = np.sqrt(max(1 - c * c, 0.0)) * _smooth_field(coords, rng)
            r = base + shift * (proj - proj.mean()) + 0.15 * perturb
            rho[:, m] = (r - r.mean()) / (r.std() + 1e-8)
        if return_parts:
            # cascade shares a common base rather than an explicit z_shared
            return rho, base, rho
        return rho
    z_shared = _field(cfg.regime, coords, rng)
    z_shared = (z_shared - z_shared.mean()) / (z_shared.std() + 1e-8)
    # Each private field is orthogonalised against the shared field AND against
    # every private field already drawn, so the only correlation between any two
    # rho_m is the shared component -> corr(rho_i, rho_j) = c^2 exactly.
    privs: list[np.ndarray] = []
    for m in range(M):
        z_m = _field(cfg.regime, coords, rng, zones=4 + m)
        z_m = _orth(z_m, z_shared, *privs)
        privs.append(z_m)
        r = c * z_shared + np.sqrt(max(1 - c * c, 0.0)) * z_m
        rho[:, m] = (r - r.mean()) / (r.std() + 1e-8)
    if return_parts:
        return rho, z_shared, np.stack(privs, 1)
    return rho


def _effective_rho(rho_all: np.ndarray, m: int, cfg: SimConfigV3, rng) -> np.ndarray:
    """(N, G) per-gene effective isodepth driving modality m's expression.

        eta_g = sqrt(1-c^2)*rho_m + c * sum_{k!=m} a_gk * rho_k      (z-scored)

    This is the emission-level coupling Saturn's dec_m(Phi) assumes and v2
    never generated. c=0 reduces to eta_g = rho_m for every gene.
    """
    N, M = rho_all.shape
    G = cfg.genes_per_mod
    c = _ce(cfg)                       # EMISSION coupling only
    eta = np.repeat(rho_all[:, m][:, None], G, axis=1)
    if c > 0 and M > 1:
        eta = np.sqrt(max(1 - c * c, 0.0)) * eta
        others = [k for k in range(M) if k != m]
        scale = c / np.sqrt(len(others))                 # keep variance ~constant in M
        for k in others:
            a = rng.normal(0.0, 1.0, size=G)             # per-gene loading
            eta = eta + scale * a[None, :] * rho_all[:, k][:, None]
        eta = (eta - eta.mean(0, keepdims=True)) / (eta.std(0, keepdims=True) + 1e-8)
    return eta


def _clean_rate(rho_all: np.ndarray, m: int, cfg: SimConfigV3, rng) -> np.ndarray:
    """(N, G) unit-mean clean rate for modality m. Nonlinear in eta."""
    G = cfg.genes_per_mod
    eta = _effective_rho(rho_all, m, cfg, rng)
    base = rng.uniform(-1.0, 0.5, size=G)
    n_bump = int(round(cfg.bump_frac * G))
    is_bump = np.zeros(G, dtype=bool)
    is_bump[:n_bump] = True
    rng.shuffle(is_bump)
    lo, hi = eta.min(), eta.max()
    centers = rng.uniform(lo, hi, size=G)
    widths = cfg.bump_width * rng.uniform(0.5, 1.5, size=G)
    amp = rng.uniform(1.0, 2.5, size=G)
    bump = amp[None, :] * np.exp(-((eta - centers[None, :]) ** 2)
                                   / (2 * widths[None, :] ** 2))
    slope = rng.normal(0, 1.5, size=G)
    bias = rng.normal(0, 1.0, size=G)
    mono = np.tanh(slope[None, :] * eta + bias[None, :])
    lograte = base[None, :] + np.where(is_bump[None, :], bump, mono)
    rate = np.exp(lograte)
    rate = rate / rate.mean(1, keepdims=True)
    return rate.astype(np.float32)


def simulate_v3(cfg: SimConfigV3) -> SimDataV3:
    rng = np.random.default_rng(cfg.seed)
    xs = np.linspace(0, 1, cfg.n_side)
    gx, gy = np.meshgrid(xs, xs)
    coords = np.stack([gx.ravel(), gy.ravel()], 1)
    coords = (coords + rng.normal(0, 0.3 / cfg.n_side, size=coords.shape)).astype(np.float32)

    rho_true, z_shared, z_priv = _build_rho(cfg, coords, rng, return_parts=True)
    counts, mus = [], []
    for m in range(cfg.M):
        clean = _clean_rate(rho_true, m, cfg, rng)
        G = clean.shape[1]
        mus.append((clean * (cfg.noise.libsize / G)).astype(np.float32))
        counts.append(_apply_noise(clean, cfg.noise, rng))
    return SimDataV3(coords=coords, rho_true=rho_true.astype(np.float32),
                      z_shared=np.asarray(z_shared, dtype=np.float32),
                      z_priv=np.asarray(z_priv, dtype=np.float32),
                      counts=counts, mu_true=mus, config=cfg)
