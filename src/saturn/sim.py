"""Two-modality simulator matching Saturn's generative model (RNA + ATAC, M=2).

GENERATIVE ASSUMPTION
    Each modality has its own isodepth. Every feature of every modality is
    explained ADDITIVELY by its own axis and by the other modality's axis:

        RNA gene g   : eta_g(x) = a_g + s_g( d_R(x) ) + c_g( d_A(x) )
        ATAC peak p  : eta_p(x) = b_p + t_p( d_A(x) ) + e_p( d_R(x) )

        s = RNA <- RNA (own)      c = RNA <- ATAC (cross)
        t = ATAC <- ATAC (own)    e = ATAC <- RNA (cross)

    Each of s, c, t, e is a GASTON-style PIECEWISE-LINEAR curve of ONE axis:
    straight within a domain, optional jump at a domain boundary. Domains belong
    to the AXIS (e.g. layers along d_A), so every feature that reads axis k
    shares that axis's breakpoints -- GASTON's "domains are shared across genes".

    Counts are drawn from these log-rates, then converted to the model's targets
    exactly as Saturn consumes them:
        RNA  -> GLM-PCs  (Poisson Pearson-residual PCA, GASTON's own approximation)
        ATAC -> LSI      (TF-IDF -> SVD, component 1 dropped: it tracks depth)

KNOBS
    axis_angle_deg  angle between the two isodepth gradients. 90 -> nearly
                    uncorrelated axes; 0 -> the same axis (unidentifiable).
    cross_RA        strength of the ATAC axis inside RNA   (c_g), relative to own
    cross_AR        strength of the RNA axis inside ATAC   (e_p), relative to own
    coupled_frac    fraction of features that carry ANY cross term (sparse:
                    not every gene is chromatin-regulated)
    jump_frac       fraction of (feature, boundary) pairs with a discontinuity
    Identifiability needs each modality to be explained more by its OWN axis than
    by the other, i.e. cross_* < 1 (own_amp is the unit).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class SimConfig:
    n_side: int = 30               # 30 x 30 = 900 spots
    n_genes: int = 500
    n_peaks: int = 1000
    L_R: int = 4                   # domains along d_R
    L_A: int = 4                   # domains along d_A
    axis_angle_deg: float = 60.0
    curvature: float = 0.6         # bends level sets into arcs (layer-like)
    own_amp: float = 1.0           # std of each own curve (log scale)
    cross_RA: float = 0.5          # ATAC axis -> RNA
    cross_AR: float = 0.2          # RNA axis  -> ATAC
    coupled_frac: float = 0.4
    jump_frac: float = 0.3
    flat_frac: float = 0.3         # (feature, domain) slopes set to 0
    rna_libsize: float = 2000.0
    atac_depth: float = 1500.0
    nb_theta: float = 10.0         # RNA overdispersion; ATAC is Poisson
    K_R: int = 10                  # GLM-PCs kept
    K_A: int = 15                  # LSI components kept (after dropping #1)
    seed: int = 0


@dataclass
class SimData:
    coords: np.ndarray             # (N, 2)
    d: np.ndarray                  # (N, 2) true isodepths [d_R, d_A], in [0,1]
    domains: np.ndarray            # (N, 2) domain label along each axis
    breakpoints: Dict[str, np.ndarray]
    counts_R: np.ndarray           # (N, G)
    counts_A: np.ndarray           # (N, P)
    Y_R: np.ndarray                # (N, K_R) GLM-PCs, z-scored  <- model target
    Y_A: np.ndarray                # (N, K_A) LSI, z-scored      <- model target
    parts: Dict[str, np.ndarray]   # the four curves evaluated at every spot
    coupled: Dict[str, np.ndarray] # which features carry a cross term
    cross_share: Dict[str, float]  # true variance share of the cross term
    config: SimConfig = None


# ----------------------------------------------------------------- fields
def _axis_field(coords: np.ndarray, angle: float, curvature: float) -> np.ndarray:
    """GASTON-like isodepth: a ramp along direction u, with level sets bent into
    arcs by a quadratic term along the perpendicular v. Low-complexity by design
    (a raw-coordinate ReLU network can represent it), monotone along u."""
    u = np.array([np.cos(angle), np.sin(angle)])
    v = np.array([-np.sin(angle), np.cos(angle)])
    c = coords - coords.mean(0)
    d = c @ u + curvature * (c @ v) ** 2
    return (d - d.min()) / (d.max() - d.min())


def _breakpoints(d: np.ndarray, L: int, rng) -> np.ndarray:
    """L-1 domain boundaries at jittered quantiles of the axis."""
    q = np.linspace(0, 1, L + 1)[1:-1]
    q = np.clip(q + rng.uniform(-0.05, 0.05, size=q.shape), 0.05, 0.95)
    return np.sort(np.quantile(d, q))


# ----------------------------------------------------------------- curves
def _pw_curves(d: np.ndarray, bps: np.ndarray, n_feat: int, amp: float,
               jump_frac: float, flat_frac: float, rng) -> np.ndarray:
    """(N, n_feat) piecewise-linear curves of ONE axis with SHARED breakpoints.

    Per feature and domain: a slope (zeroed with prob flat_frac); per boundary:
    a jump (non-zero with prob jump_frac). Values are chained across domains so
    the curve is straight inside each domain and steps only at boundaries.
    Each curve is then centred and scaled to std = amp, so own vs cross
    strengths are controlled exactly by the amp passed in.
    """
    L = len(bps) + 1
    edges = np.r_[d.min(), bps, d.max()]
    lab = np.digitize(d, bps)                                      # 0..L-1
    slopes = rng.normal(0, 3.0, size=(n_feat, L))
    slopes *= rng.random((n_feat, L)) > flat_frac
    jumps = rng.normal(0, 1.0, size=(n_feat, L - 1))
    jumps *= rng.random((n_feat, L - 1)) < jump_frac

    left = np.zeros((n_feat, L))                                   # value at domain's left edge
    for l in range(1, L):
        left[:, l] = (left[:, l - 1]
                      + slopes[:, l - 1] * (edges[l] - edges[l - 1])
                      + jumps[:, l - 1])
    F = np.empty((d.shape[0], n_feat))
    for l in range(L):
        m = lab == l
        F[m] = left[:, l][None, :] + (d[m, None] - edges[l]) * slopes[:, l][None, :]
    F -= F.mean(0)
    sd = F.std(0)
    F = np.where(sd > 1e-8, F / np.where(sd > 1e-8, sd, 1.0), 0.0) * amp
    return F


# ----------------------------------------------------------------- noise
def _nb(mu: np.ndarray, theta: float, rng) -> np.ndarray:
    """Negative binomial via gamma-Poisson: mean mu, variance mu + mu^2/theta."""
    return rng.poisson(rng.gamma(theta, mu / theta)).astype(np.float32)


def _softmax_rows(eta: np.ndarray) -> np.ndarray:
    e = np.exp(eta - eta.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


# ----------------------------------------------------------------- targets
from .preprocess import rna_glmpc, atac_lsi


# ----------------------------------------------------------------- main
def simulate(cfg: SimConfig) -> SimData:
    rng = np.random.default_rng(cfg.seed)

    # 1. spots on a jittered grid
    xs = np.linspace(0, 1, cfg.n_side)
    gx, gy = np.meshgrid(xs, xs)
    coords = np.stack([gx.ravel(), gy.ravel()], 1)
    coords = coords + rng.normal(0, 0.3 / cfg.n_side, size=coords.shape)
    N = coords.shape[0]

    # 2. two isodepths, gradients separated by axis_angle_deg
    phi = rng.uniform(0, 2 * np.pi)
    d_R = _axis_field(coords, phi, cfg.curvature)
    d_A = _axis_field(coords, phi + np.deg2rad(cfg.axis_angle_deg), cfg.curvature)

    # 3. domains belong to the axis
    bp_R = _breakpoints(d_R, cfg.L_R, rng)
    bp_A = _breakpoints(d_A, cfg.L_A, rng)

    # 4. the four piecewise-linear processes
    G, P = cfg.n_genes, cfg.n_peaks
    kw = dict(jump_frac=cfg.jump_frac, flat_frac=cfg.flat_frac, rng=rng)
    s = _pw_curves(d_R, bp_R, G, cfg.own_amp, **kw)                  # RNA  <- RNA
    t = _pw_curves(d_A, bp_A, P, cfg.own_amp, **kw)                  # ATAC <- ATAC
    c = _pw_curves(d_A, bp_A, G, cfg.own_amp * cfg.cross_RA, **kw)   # RNA  <- ATAC
    e = _pw_curves(d_R, bp_R, P, cfg.own_amp * cfg.cross_AR, **kw)   # ATAC <- RNA
    coupled_R = rng.random(G) < cfg.coupled_frac
    coupled_A = rng.random(P) < cfg.coupled_frac
    c = c * coupled_R[None, :]
    e = e * coupled_A[None, :]

    # 5. additive log-rates
    a = rng.normal(0, 1.0, size=G)                                   # gene baselines
    b = rng.normal(0, 0.5, size=P)                                   # peak baselines
    eta_R = a[None, :] + s + c
    eta_A = b[None, :] + t + e

    # 6. counts: per-spot depth x relative abundance
    lib_R = rng.lognormal(np.log(cfg.rna_libsize), 0.3, size=(N, 1))
    lib_A = rng.lognormal(np.log(cfg.atac_depth), 0.3, size=(N, 1))
    counts_R = _nb(lib_R * _softmax_rows(eta_R), cfg.nb_theta, rng)
    counts_A = rng.poisson(lib_A * _softmax_rows(eta_A)).astype(np.float32)

    # 7. targets Saturn fits
    Y_R = rna_glmpc(counts_R, cfg.K_R)
    Y_A = atac_lsi(counts_A, cfg.K_A, drop_first=1)

    def share(own, cross, mask):
        vc, vo = cross[:, mask].var(0), own[:, mask].var(0)
        return float(np.mean(vc / (vc + vo + 1e-12))) if mask.any() else 0.0

    return SimData(
        coords=coords.astype(np.float32),
        d=np.stack([d_R, d_A], 1).astype(np.float32),
        domains=np.stack([np.digitize(d_R, bp_R), np.digitize(d_A, bp_A)], 1),
        breakpoints={"R": bp_R, "A": bp_A},
        counts_R=counts_R, counts_A=counts_A, Y_R=Y_R, Y_A=Y_A,
        parts={"s": s.astype(np.float32), "c": c.astype(np.float32),
               "t": t.astype(np.float32), "e": e.astype(np.float32)},
        coupled={"R": coupled_R, "A": coupled_A},
        cross_share={"R_from_A": share(s, c, coupled_R),
                     "A_from_R": share(t, e, coupled_A)},
        config=cfg,
    )
