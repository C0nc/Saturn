"""Spatial transition across the main rho_X-steep zone (Fig 7 j): what changes, in which order.

zone       largest spatially connected component (DBSCAN, eps = 1.6 spot spacings) of the top-20% spots by
           steepness log(|grad rho_X| / |grad rho_RNA|) (crossterm.steepness)
transect   direction u = mean local gradient of rho_X in the zone; band = spots whose offset perpendicular to u lies
           within the zone's 10-90% range, extended upstream by one zone length; s = position along u in spot
           spacings (0 = zone centre)
per feature (smoothed; axes, gain, cell types, TF activity, genes):
  sigmoid   f(s) = a + b / (1 + exp(-(s - s0)/w)): s0 = switch point, |w| = sharpness
  bump      a + b exp(-(s - m)^2 / 2v^2): flags transient features (R2_bump - R2_sig > 0.15)
  steepest  location of the steepest change of the binned ([1,2,1]-smoothed, 18 bins) profile, with a 95% CI from
            200 spot bootstrap resamples -- the robust "where does it switch" used in the figure
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

NBOOT, NBIN = 200, 18


def sig(s, a, b, s0, w):
    return a + b / (1 + np.exp(-np.clip((s - s0) / w, -50, 50)))


def bump(s, a, b, m, v):
    return a + b * np.exp(-(s - m) ** 2 / (2 * v ** 2))


def grad_vec(v, xy, k=25):
    """local gradient vector of a scalar field by k-NN linear fits"""
    X = xy - xy.mean(0); nn = cKDTree(X).query(X, k=k)[1]; g = np.empty((len(X), 2))
    for i in range(len(X)):
        A = X[nn[i]] - X[nn[i]].mean(0); g[i] = np.linalg.lstsq(A, v[nn[i]] - v[nn[i]].mean(), rcond=None)[0]
    return g


def transect(xy, rho_x, steep, top_q=0.8, eps=1.6, min_samples=4):
    """steep zone + transect. Returns dict(s, o, band, zone, u, spacing)."""
    from sklearn.cluster import DBSCAN
    xy = np.asarray(xy, float); sp_ = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
    top = steep >= np.quantile(steep, top_q)
    lab = np.full(len(xy), -1); lab[top] = DBSCAN(eps=eps * sp_, min_samples=min_samples).fit_predict(xy[top])
    if not (lab >= 0).any():
        raise ValueError("no spatially connected steep zone: every top-quantile spot is DBSCAN noise "
                         "(lower top_q or raise eps)")
    zone = lab == pd.Series(lab[lab >= 0]).value_counts().index[0]
    u = grad_vec(np.asarray(rho_x, float), xy)[zone].mean(0); u /= np.linalg.norm(u); n = np.array([-u[1], u[0]])
    c = xy[zone].mean(0); s = (xy - c) @ u / sp_; o = (xy - c) @ n / sp_
    lo, hi = np.percentile(o[zone], [10, 90]); band = (o >= lo) & (o <= hi)
    band &= s >= s[zone].min() - (s[zone].max() - s[zone].min())
    return dict(s=s, o=o, band=band, zone=zone, u=u, spacing=sp_)


def fit_sigmoid_bump(s, y):
    """sigmoid and bump fits on standardised y: s0, |w|, R2_sig, up, R2_bump, peak, peak_sign"""
    y = (y - y.mean()) / (y.std() + 1e-12); tss = ((y - y.mean()) ** 2).sum(); lo, hi = s.min(), s.max()
    out = dict(s0=np.nan, w=np.nan, R2_sig=np.nan, R2_bump=np.nan, peak=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best = None
        for s0 in np.linspace(lo, hi, 7)[1:-1]:
            for sgn in (1, -1):
                try:
                    p, _ = curve_fit(sig, s, y, p0=[-sgn, 2 * sgn, s0, sgn * 2], maxfev=4000,
                                     bounds=([-10, -10, lo, -60], [10, 10, hi, 60]))
                    r = ((y - sig(s, *p)) ** 2).sum()
                    if best is None or r < best[0]:
                        best = (r, p)
                except Exception:
                    pass
        if best:
            out.update(s0=float(best[1][2]), w=float(abs(best[1][3])), R2_sig=float(1 - best[0] / tss),
                       up=bool(best[1][1] * np.sign(best[1][3]) > 0))
        bb = None
        for m in np.linspace(lo, hi, 7)[1:-1]:
            for sgn in (1, -1):
                try:
                    p, _ = curve_fit(bump, s, y, p0=[0, sgn, m, (hi - lo) / 6], maxfev=4000,
                                     bounds=([-10, -10, lo, 0.5], [10, 10, hi, hi - lo]))
                    r = ((y - bump(s, *p)) ** 2).sum()
                    if bb is None or r < bb[0]:
                        bb = (r, p)
                except Exception:
                    pass
        if bb:
            out.update(R2_bump=float(1 - bb[0] / tss), peak=float(bb[1][2]), peak_sign=float(np.sign(bb[1][1])))
    return out


def steepest(s, y, edges):
    """bin centre of the steepest change of the binned, [1,2,1]-smoothed profile and its signed slope (sd per spacing)"""
    bi = np.clip(np.digitize(s, edges[1:-1]), 0, len(edges) - 2); m = pd.Series(y).groupby(bi).mean()
    m = m.reindex(range(len(edges) - 1)).interpolate().bfill().ffill().values
    m = np.convolve(np.r_[m[0], m, m[-1]], [0.25, 0.5, 0.25], "valid"); c = (edges[1:] + edges[:-1]) / 2
    d = np.gradient(m, c) / (np.std(y) + 1e-12); j = int(np.argmax(np.abs(d))); return float(c[j]), float(d[j])


def transition_table(F: pd.DataFrame, tr: dict, n_boot: int = NBOOT, n_bin: int = NBIN, seed: int = 0):
    """F: spots x features (already smoothed). Fits every feature along the band of `tr` (from `transect`).
    Returns (fits DataFrame, binned profiles DataFrame)."""
    Fb, sb = F[tr["band"]], tr["s"][tr["band"]]
    rng = np.random.default_rng(seed); edges = np.linspace(sb.min(), sb.max(), n_bin + 1); rows = []
    for f in F.columns:
        y = Fb[f].values; r = fit_sigmoid_bump(sb, y)
        r["steep_at"], r["steep_slope"] = steepest(sb, y, edges)
        bs = [steepest(sb[i], y[i], edges)[0] for i in (rng.integers(0, len(sb), len(sb)) for _ in range(n_boot))]
        r["steep_lo"], r["steep_hi"] = (float(v) for v in np.percentile(bs, [2.5, 97.5]))
        r["delta_sd"] = float((np.percentile(y, 90) - np.percentile(y, 10)) / (F[f].std() + 1e-12))
        r["transient"] = bool(np.nan_to_num(r.get("R2_bump", np.nan) - r.get("R2_sig", np.nan)) > 0.15)
        rows.append(dict(feature=f, **r))
    bi = np.clip(np.digitize(sb, edges[1:-1]), 0, n_bin - 1)
    prof = Fb.groupby(bi).mean(); prof.insert(0, "s", pd.Series(sb).groupby(bi).mean().values)
    prof.insert(1, "n", pd.Series(sb).groupby(bi).size().values)
    return pd.DataFrame(rows), prof.reset_index(drop=True)
