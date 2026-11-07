"""v4 simulator — generation FROM (rho_shared, rho_priv_m), matching the model.

WHY v4 EXISTS
    v3 built a MIXED per-modality field and emitted from it:
        rho_m  = c*z_shared + sqrt(1-c^2)*z_m
        eta_g  = sqrt(1-ce^2)*rho_m + ce*sum_{k!=m} a_gk*rho_k
    Nothing in that process ever refers to "the shared field" and "modality m's
    private field" as separate generators of the OBSERVATION, so a shared/private
    model has no signal telling it where to put the split. Measured directly on
    v3 with SaturnAPOLLO: the private heads recovered the MIXED rho at 0.999 but
    the true private field at only 0.78, while leaking 0.44 of the shared field,
    and the shared head reached just 0.227.

    v4 generates the way the model reads:

        eta_g^(m) = sqrt(w_s) * s_g * z_shared  +  sqrt(1 - w_s) * p_g * z_priv_m
        lograte_g = base_g + [bump | mono](eta_g)

    Modality m's observation is a function of the shared field and ITS OWN
    private field -- and of no other modality's private field. That is exactly
    the support pattern a strict dec_m(rho_shared, rho_priv_m) assumes, so the
    architecture is correctly specified rather than approximating.

WHAT THE KNOBS MEAN
    shared_frac  w_s in [0,1]. Per-gene weight on the shared field. 0 => the
        modalities share nothing and the problem factorises; 1 => everything is
        shared and the private axes are unidentifiable BY CONSTRUCTION (there is
        no private signal left to find). The interesting range is the middle.
    anchor_frac  fraction of each modality's genes driven by its PRIVATE field
        ALONE (w_s forced to 0 for those genes). These are the anchor features
        of Moran et al. (Sparse VAE, TMLR 2022): each latent needs an observed
        variable it solely influences. Biologically these are modality-private
        markers, which real spatial multi-omics does have. With anchor_frac > 0
        the private axes are identifiable even at high shared_frac.
    cross_talk   OPTIONAL leak: how much modality m's genes read OTHER
        modalities' private fields. 0 (default) is the clean APOLLO-shaped
        world. Raise it to deliberately violate the model's assumption and
        measure what that costs -- this is the v3 pathology, now isolated as its
        own knob instead of being entangled with field sharing.

GROUND TRUTH RETURNED
    z_shared (N,), z_priv (N, M) -- orthogonal by construction, so shared and
    private recovery can each be scored directly. `rho_true` is kept as the
    per-modality EFFECTIVE axis for continuity with v3-era metrics.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np

from simulate_v2 import (_smooth_field, _domain_field, NoiseSpec, _apply_noise)
from simulate_v3 import _field, _orth


@dataclass
class SimConfigV4:
    n_side: int = 30
    M: int = 2
    genes_per_mod: int = 60
    regime: str = "smooth"              # smooth | domains
    shared_frac: float = 0.5            # per-gene weight on the shared field
    anchor_frac: float = 0.0            # genes driven by PRIVATE alone
    cross_talk: float = 0.0             # leak onto OTHER modalities' privates
    bump_frac: float = 0.6              # non-monotone (orientation-free) genes
    bump_width: float = 0.5
    noise: NoiseSpec = field(default_factory=lambda: NoiseSpec(libsize=2000.0))
    seed: int = 0


@dataclass
class SimDataV4:
    coords: np.ndarray                  # (N, 2)
    z_shared: np.ndarray                # (N,)
    z_priv: np.ndarray                  # (N, M)
    rho_true: np.ndarray                # (N, M) effective per-modality axis
    counts: List[np.ndarray]
    mu_true: List[np.ndarray]
    config: SimConfigV4 = None


def _response(eta: np.ndarray, cfg: SimConfigV4, rng) -> np.ndarray:
    """Per-gene nonlinear response: a bump fraction (non-monotone, carries no
    orientation) plus a monotone remainder (which does)."""
    N, G = eta.shape
    out = np.empty_like(eta)
    n_bump = int(round(cfg.bump_frac * G))
    centers = rng.uniform(-1.0, 1.0, size=G)
    slopes = rng.normal(0.0, 1.0, size=G)
    for g in range(G):
        if g < n_bump:
            out[:, g] = np.exp(-0.5 * ((eta[:, g] - centers[g]) / cfg.bump_width) ** 2)
        else:
            out[:, g] = np.tanh(slopes[g] * eta[:, g])
    return out


def simulate_v4(cfg: SimConfigV4) -> SimDataV4:
    rng = np.random.default_rng(cfg.seed)
    xs = np.linspace(0, 1, cfg.n_side)
    gx, gy = np.meshgrid(xs, xs)
    coords = np.stack([gx.ravel(), gy.ravel()], 1)
    coords = (coords + rng.normal(0, 0.3 / cfg.n_side,
                                   size=coords.shape)).astype(np.float32)
    N, M, G = coords.shape[0], cfg.M, cfg.genes_per_mod

    # ---- latent fields: shared + mutually orthogonal privates ---------------
    z_shared = _field(cfg.regime, coords, rng)
    z_shared = (z_shared - z_shared.mean()) / (z_shared.std() + 1e-8)
    privs: list[np.ndarray] = []
    for m in range(M):
        z_m = _field(cfg.regime, coords, rng, zones=4 + m)
        z_m = _orth(z_m, z_shared, *privs)          # orthogonal to shared + earlier
        privs.append(z_m)
    z_priv = np.stack(privs, 1)                     # (N, M)

    ws = float(np.clip(cfg.shared_frac, 0.0, 1.0))
    ct = float(np.clip(cfg.cross_talk, 0.0, 1.0))
    n_anchor = int(round(cfg.anchor_frac * G))

    counts, mus, rho_eff = [], [], np.zeros((N, M), dtype=np.float64)
    for m in range(M):
        s_g = rng.normal(0.0, 1.0, size=G)          # per-gene shared loading
        p_g = rng.normal(0.0, 1.0, size=G)          # per-gene private loading
        w_shared = np.full(G, np.sqrt(ws))
        w_priv = np.full(G, np.sqrt(max(1.0 - ws, 0.0)))
        if n_anchor > 0:                            # anchors: private only
            w_shared[:n_anchor] = 0.0
            w_priv[:n_anchor] = 1.0

        # modality m reads exactly TWO of the three rho: the shared field and
        # its OWN private field. For M=2 the full latent is
        #     (z_shared, z_priv_0, z_priv_1)
        # and modality 0 sees (z_shared, z_priv_0), modality 1 sees
        # (z_shared, z_priv_1). That support pattern IS the strict decoder.
        eta = ((w_shared * s_g)[None, :] * z_shared[:, None]
               + (w_priv * p_g)[None, :] * z_priv[:, m][:, None])

        if ct > 0 and M > 1:                        # deliberate assumption break
            others = [k for k in range(M) if k != m]
            scale = ct / np.sqrt(len(others))
            for k in others:
                a = rng.normal(0.0, 1.0, size=G)
                eta = eta + scale * a[None, :] * z_priv[:, k][:, None]

        eta = (eta - eta.mean(0, keepdims=True)) / (eta.std(0, keepdims=True) + 1e-8)
        # effective per-modality axis = the coherent part of what drives modality m
        rho_eff[:, m] = (np.sqrt(ws) * z_shared + np.sqrt(max(1 - ws, 0.0)) * z_priv[:, m])

        base = rng.normal(0.0, 0.3, size=G)
        clean = np.exp(base[None, :] + _response(eta, cfg, rng))
        clean = clean / clean.sum(1, keepdims=True).clip(min=1e-8)
        mus.append((clean * (cfg.noise.libsize)).astype(np.float32))
        counts.append(_apply_noise(clean * cfg.noise.libsize, cfg.noise, rng))

    rho_eff = (rho_eff - rho_eff.mean(0)) / (rho_eff.std(0) + 1e-8)
    return SimDataV4(coords=coords,
                      z_shared=z_shared.astype(np.float32),
                      z_priv=z_priv.astype(np.float32),
                      rho_true=rho_eff.astype(np.float32),
                      counts=counts, mu_true=mus, config=cfg)
