"""Corrected recovery metrics.

THREE PROBLEMS WITH THE ORIGINAL `metrics.py`
---------------------------------------------
1. ASYMMETRIC APPLICATION (the serious one).
   Saturn was scored with `match_recovery` — the single rho it committed to.
   Every baseline was scored with `best_factor_recovery`, whose own docstring
   says it returns "the factor best correlated with the truth": an oracle pick
   of the best of K=8 factors, using the ground truth. Measured effect is large
   enough to decide the ranking — PCA led Saturn 0.946 vs 0.921 on the v3 grid
   while doing a strictly easier task. Neither folds nor architecture moved the
   ordering as much as this did.

2. METRIC CHOICE CHANGES THE WINNER.
       method    |Spear|  |Pears|  R2_lin  R2_mono
       Saturn      0.960    0.949   0.900    0.963
       PCA         0.971    0.956   0.915    0.944
   Spearman credits ANY monotone relation, so a linear projection of a curved
   manifold scores nearly full marks — generous to linear baselines. Pearson is
   too strict: isodepth is only identifiable up to monotone maps, so penalising
   nonlinear spacing penalises a valid recovery. R2_monotone matches the
   identifiability exactly and is the principled primary.

3. ONE NUMBER CONFLATES TWO FAILURES.
       low score + high poly-R2 -> FOLD    (level sets found, ordering reversed)
       low score + low  poly-R2 -> FAILURE (nothing fit; e.g. NSF's 0.000)
   Saturn's failures are folds and are recoverable; reporting one column makes
   them look like NSF's, which they are not.

WHAT THIS MODULE DOES
   * r2_monotone  — primary. Variance of true rho explained by the best MONOTONE
                    transform of the prediction (isotonic). Fold-aware for free:
                    a fold is non-monotone, so it scores near zero.
   * spearman     — secondary, for comparability with prior work.
   * is_fold      — diagnostic separating fold from failure.
   * select_factor— GROUND-TRUTH-FREE factor choice for multi-factor methods,
                    replacing the oracle. Picks the factor with the highest
                    spatial autocorrelation (Moran's I), which is what an analyst
                    without labels would do.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import NearestNeighbors


# --------------------------------------------------------------------- scores
def spearman(pred, true) -> float:
    r = spearmanr(pred, true).correlation
    return 0.0 if r is None or np.isnan(r) else abs(float(r))


def pearson(pred, true) -> float:
    try:
        r = pearsonr(pred, true)[0]
    except Exception:
        return 0.0
    return 0.0 if np.isnan(r) else abs(float(r))


def r2_monotone(pred, true) -> float:
    """PRIMARY. R^2 of the best monotone transform of `pred` predicting `true`.

    Isodepth is identifiable only up to a monotone reparametrisation, so this
    asks the fair question: how much of rho does a monotone function of your
    output explain? Invariant to any monotone distortion, but a FOLDED
    prediction (non-monotone in rho) cannot be fixed by a monotone map and
    scores near zero — so no separate fold detector is needed.
    """
    pred = np.asarray(pred, float).ravel()
    true = np.asarray(true, float).ravel()
    if pred.std() < 1e-12:
        return 0.0
    s = 1.0 if (spearmanr(pred, true).correlation or 0.0) >= 0 else -1.0
    iso = IsotonicRegression(out_of_bounds="clip").fit(s * pred, true)
    fit = iso.predict(s * pred)
    ss_res = float(((true - fit) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return float(max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12)))


def poly_r2(pred, true, deg: int = 6) -> float:
    """R^2 of a free smooth (polynomial) fit — NOT restricted to monotone.
    High here with low spearman is the signature of a fold."""
    pred = np.asarray(pred, float).ravel()
    true = np.asarray(true, float).ravel()
    if pred.std() < 1e-12:
        return 0.0
    z = (pred - pred.mean()) / (pred.std() + 1e-9)
    fit = np.polyval(np.polyfit(z, true, deg), z)
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return float(1.0 - ((true - fit) ** 2).sum() / max(ss_tot, 1e-12))


def is_fold(pred, true, thresh: float = 0.6, poly_thresh: float = 0.7) -> bool:
    """Fold = ordering lost BUT geometry recovered. Distinguishes a recoverable
    mis-orientation from a method that simply never fit anything."""
    return spearman(pred, true) < thresh and poly_r2(pred, true) > poly_thresh


def score_all(pred, true) -> dict:
    return dict(r2_monotone=r2_monotone(pred, true),
                spearman=spearman(pred, true),
                pearson=pearson(pred, true),
                poly_r2=poly_r2(pred, true),
                folded=is_fold(pred, true))


# ------------------------------------------------------- factor selection
def morans_i(v: np.ndarray, coords: np.ndarray, k: int = 6) -> float:
    """Spatial autocorrelation of `v` over a k-NN graph. Used to pick a factor
    WITHOUT ground truth: an isodepth-like field is spatially smooth, noise is
    not. This is the choice an analyst makes when no labels exist."""
    v = np.asarray(v, float).ravel()
    if v.std() < 1e-12:
        return -1.0
    idx = NearestNeighbors(n_neighbors=k + 1).fit(coords) \
            .kneighbors(coords, return_distance=False)[:, 1:]
    z = v - v.mean()
    num = float((z[:, None] * z[idx]).sum())
    den = float((z ** 2).sum()) * k
    return num / max(den, 1e-12)


def select_factor(F: np.ndarray, coords: np.ndarray, truth=None,
                    mode: str = "moran") -> tuple[np.ndarray, int]:
    """Choose one factor from a (N, K) matrix.

    mode="moran"  : GROUND-TRUTH-FREE (default). Highest spatial autocorrelation.
                    This is the fair setting — the method must commit, exactly
                    as Saturn does.
    mode="first"  : first component (PCA convention).
    mode="oracle" : best correlated with `truth` — the ORIGINAL behaviour.
                    Retained only as a sensitivity check / upper bound; it
                    inflates every multi-factor baseline and must be labelled
                    as an oracle wherever it is reported.
    """
    F = np.asarray(F)
    F = F[:, None] if F.ndim == 1 else F
    if mode == "first":
        return F[:, 0], 0
    if mode == "oracle":
        if truth is None:
            raise ValueError("mode='oracle' requires truth")
        k = int(np.argmax([spearman(F[:, j], truth) for j in range(F.shape[1])]))
        return F[:, k], k
    k = int(np.argmax([morans_i(F[:, j], coords) for j in range(F.shape[1])]))
    return F[:, k], k


# ------------------------------------------------- fair factor selection
def variance_explained(factor: np.ndarray, X: np.ndarray,
                        n_genes: int = 200) -> float:
    """Fraction of modality-X variance explained by a single latent `factor`.

    For each gene, corr(factor, gene)^2 is the R^2 of the univariate regression
    of that gene on the factor; averaging over genes gives the share of that
    modality's variance the factor accounts for.

    THIS IS THE FAIR SELECTOR. The original `best_factor_recovery` picked "the
    factor best correlated with the truth" — an oracle. Variance explained is
    the METHOD'S OWN answer to "which of my latent dimensions describes modality
    m", needs no ground truth, and is what an analyst without labels would use.
    It is also the native criterion for MOFA/MEFISTO, which report variance
    explained per factor per view directly.
    """
    f = np.asarray(factor, float).ravel()
    if f.std() < 1e-12:
        return 0.0
    X = np.asarray(X, float)
    lib = X.sum(1, keepdims=True).clip(min=1)
    Xn = np.log1p(X / lib * 1e4)
    if Xn.shape[1] > n_genes:                      # top-variance genes only
        Xn = Xn[:, np.argsort(-Xn.var(0))[:n_genes]]
    f = (f - f.mean()) / (f.std() + 1e-12)
    Z = (Xn - Xn.mean(0)) / (Xn.std(0) + 1e-12)
    r = (Z * f[:, None]).mean(0)                   # per-gene correlation
    return float(np.mean(r ** 2))


def select_factor_fair(F: np.ndarray, X_modality: np.ndarray):
    """Pick the latent dimension explaining the most variance in this modality.

    Universal across methods that emit a multi-dim latent — PCA, NMF, MOFA,
    MEFISTO, STAMP, NSF, SpatialGlue, MISO, LYNX. GASTON-single needs no
    selection: fitting it per modality already yields one rho per modality.
    """
    F = np.asarray(F)
    F = F[:, None] if F.ndim == 1 else F
    ve = [variance_explained(F[:, k], X_modality) for k in range(F.shape[1])]
    k = int(np.argmax(ve))
    return F[:, k], k, float(ve[k])
