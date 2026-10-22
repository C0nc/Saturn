"""Per-domain piecewise fits for every modality type, behind one interface.

GASTON fits gene expression with a piecewise POISSON model and keeps a domain's slope
only when its likelihood-ratio test beats the zero-slope fit at p < t (default 0.1).
That gate is the right idea for every modality, but the likelihood is not: methylation
residuals and Hi-C compartment scores are continuous and signed, isoform usage and
clone composition are proportions out of a known denominator, and ADT counts are
over-dispersed. This module keeps GASTON's segmentation and its gate, and swaps the
likelihood:

  poisson   counts with a library exposure        -- genes, ATAC peaks/tiles (GASTON)
  gauss     continuous values, no exposure        -- methylation residuals, A/B scores
  binomial  successes out of known trials         -- isoform usage, clone composition
  nb        over-dispersed counts                 -- protein / ADT

Every fitter returns slopes and intercepts shaped (features, domains), with the slope
set to exactly 0 where the gate rejects it, so downstream code counts sloped domains
the same way regardless of modality.
"""
import numpy as np
from scipy import stats


def _gate(ll_full, ll_null, df, t):
    """likelihood-ratio gate, GASTON's rule: keep the slope only when p < t"""
    lr = 2 * (ll_full - ll_null)
    return stats.chi2.sf(np.maximum(lr, 0), df) < t


def fit_gauss(Y, lab, d, t=0.1):
    """continuous features: per-domain OLS, F-test against the intercept-only fit.

    Y (spots, features). Returns slope, inter, each (features, domains). With n <= 2
    spots in a domain the slope is not identifiable and is set to 0.
    """
    Y = np.asarray(Y, float)
    D = int(lab.max()) + 1
    slope = np.zeros((Y.shape[1], D)); inter = np.zeros((Y.shape[1], D))
    for k in range(D):
        m = lab == k
        n = int(m.sum())
        yk = Y[m]
        inter[:, k] = yk.mean(0)
        if n <= 2:
            continue
        x = d[m] - d[m].mean()
        sxx = (x ** 2).sum()
        if sxx <= 0:
            continue
        b = (x @ yk) / sxx                                   # OLS slope per feature
        rss_full = ((yk - yk.mean(0) - np.outer(x, b)) ** 2).sum(0)
        rss_null = ((yk - yk.mean(0)) ** 2).sum(0)
        F = (rss_null - rss_full) / np.maximum(rss_full / (n - 2), 1e-12)
        keep = stats.f.sf(np.maximum(F, 0), 1, n - 2) < t
        slope[:, k] = np.where(keep, b, 0.0)
        inter[:, k] = yk.mean(0) - np.where(keep, b, 0.0) * d[m].mean()
    return slope, inter


def fit_binomial(S, N, lab, d, t=0.1, max_iter=25, tol=1e-6):
    """proportions: logit(p) = a + b * d per domain by IRLS, LRT gate.

    S (spots, features) successes, N (spots, features) or (spots,) trials.
    """
    S = np.asarray(S, float)
    N = np.broadcast_to(np.asarray(N, float).reshape(len(S), -1), S.shape) if np.ndim(N) > 1 \
        else np.repeat(np.asarray(N, float)[:, None], S.shape[1], axis=1)
    D = int(lab.max()) + 1
    slope = np.zeros((S.shape[1], D)); inter = np.zeros((S.shape[1], D))

    def irls(s, n, x, with_slope):
        b = np.zeros(2)
        p0 = np.clip(s.sum() / max(n.sum(), 1e-9), 1e-6, 1 - 1e-6)
        b[0] = np.log(p0 / (1 - p0))
        X = np.c_[np.ones(len(x)), x] if with_slope else np.ones((len(x), 1))
        b = b[:X.shape[1]]
        for _ in range(max_iter):
            eta = X @ b
            p = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
            W = n * p * (1 - p)
            z = eta + (s - n * p) / np.maximum(W, 1e-9)
            A = X.T @ (W[:, None] * X) + 1e-8 * np.eye(X.shape[1])
            bn = np.linalg.solve(A, X.T @ (W * z))
            if np.max(np.abs(bn - b)) < tol:
                b = bn; break
            b = bn
        eta = X @ b
        p = np.clip(1 / (1 + np.exp(-np.clip(eta, -30, 30))), 1e-9, 1 - 1e-9)
        ll = float((s * np.log(p) + (n - s) * np.log(1 - p)).sum())
        return b, ll

    for k in range(D):
        m = lab == k
        if m.sum() <= 2:
            continue
        x = d[m] - d[m].mean()
        for j in range(S.shape[1]):
            s, n = S[m, j], N[m, j]
            if n.sum() <= 0:
                continue
            b1, ll1 = irls(s, n, x, True)
            b0, ll0 = irls(s, n, x, False)
            ok = _gate(ll1, ll0, 1, t)
            slope[j, k] = b1[1] if ok else 0.0
            inter[j, k] = (b1[0] - b1[1] * d[m].mean()) if ok else b0[0]
    return slope, inter


def dispersion(counts, lib):
    """quick over-dispersion check: var/mean of the depth-normalised counts.

    ~1 means Poisson is adequate (ATAC tiles sit at 1.07-1.09); markedly above 1 calls
    for the negative-binomial fitter instead.
    """
    c = np.asarray(counts, float)
    mu = c.mean(0)
    v = c.var(0)
    ok = mu > 0
    return float(np.median(v[ok] / mu[ok]))
