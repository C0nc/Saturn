"""Spatial transition across the main rho_ATAC-steep zone (Fig 7 h follow-up): why rho_ATAC is steep there.

For each RNA+ATAC case (lpc10, melanoma):
  zone       largest spatially connected component (DBSCAN, eps 1.6 spot spacings) of the top-20% spots by
             log(|grad rho_ATAC| / |grad rho_RNA|)
  transect   direction u = mean local gradient of rho_ATAC in the zone; band = spots whose offset perpendicular to u
             lies within the zone's 10-90% range; coordinate s = position along u (spot spacings, 0 = zone centre)
  features   both axes, RNA local gain, cell-type proportions (cell2location / Starfysh), TF activity (chromVAR),
             composition entropy, and genes (the zone's cross-explained genes + lineage markers); 15-NN smoothed
  sigmoid    each feature's band profile f(s) = a + b / (1 + exp(-(s - s0) / w)), fitted on the spots;
             s0 = where it switches, |w| = how sharp (spot spacings); 200 bootstrap resamples of spots -> 95% CI;
             R2 of the sigmoid vs of a bump a + b exp(-(s - m)^2 / 2 v^2) flags features that peak mid-transition
  steepest   location of the steepest change of each feature's binned band profile ([1,2,1]-smoothed, 18 bins),
             95% CI from 200 bootstrap resamples of spots -- the robust "where does it switch" used in the figure
  axes       global R2 of each switching cell type on B(rho_ATAC) vs B(rho_RNA): which axis carries that population
-> results/real_saturn/saturn_cross_term/<key>_transition_{spots.csv, fits.csv, profiles.csv, summary.json}
Usage: python scripts/saturn_steep_transition.py [lpc10 melanoma]
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import saturn_cross_term as SC
import saturn_cross_mechanism as MC

OUT = SC.OUT; NBOOT, NBIN = 200, 18
GENES = dict(lpc10=["Ttr", "Enpp2", "Ntrk3", "Pcsk2", "Satb2", "Cux2", "Rorb", "Mbp", "Plp1", "Trem2"],
             melanoma=["FOSB", "DUSP6", "BCAN", "ANXA1", "PMEL", "MLANA", "MITF", "AXL", "CD8A", "GZMK"])


def sig(s, a, b, s0, w):
    return a + b / (1 + np.exp(-np.clip((s - s0) / w, -50, 50)))


def bump(s, a, b, m, v):
    return a + b * np.exp(-(s - m) ** 2 / (2 * v ** 2))


def fit1(s, y):
    """sigmoid and bump fits on standardised y; returns s0, |w|, R2_sig, R2_bump, bump centre"""
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
                    if best is None or r < best[0]: best = (r, p)
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
                    if bb is None or r < bb[0]: bb = (r, p)
                except Exception:
                    pass
        if bb:
            out.update(R2_bump=float(1 - bb[0] / tss), peak=float(bb[1][2]), peak_sign=float(np.sign(bb[1][1])))
    return out


def steepest(s, y, edges):
    """bin centre of the steepest change of the binned, [1,2,1]-smoothed profile and its signed slope (sd per spot)"""
    bi = np.clip(np.digitize(s, edges[1:-1]), 0, len(edges) - 2); m = pd.Series(y).groupby(bi).mean()
    m = m.reindex(range(len(edges) - 1)).interpolate().bfill().ffill().values
    m = np.convolve(np.r_[m[0], m, m[-1]], [0.25, 0.5, 0.25], "valid"); c = (edges[1:] + edges[:-1]) / 2
    d = np.gradient(m, c) / (np.std(y) + 1e-12); j = int(np.argmax(np.abs(d))); return float(c[j]), float(d[j])


def grad_vec(v, xy, k=25):
    X = (xy - xy.mean(0)); nn = cKDTree(X).query(X, k=k)[1]; g = np.empty((len(X), 2))
    for i in range(len(X)):
        A = X[nn[i]] - X[nn[i]].mean(0); g[i] = np.linalg.lstsq(A, v[nn[i]] - v[nn[i]].mean(), rcond=None)[0]
    return g


def run(key):
    T = pd.read_csv(f"{OUT}/{key}_spots.csv", index_col=0); T.index = T.index.astype(str); bc = list(T.index)
    xy = T[["x", "y"]].values; sp = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
    lg = np.log((T.grad_X / T.grad_X.median()) / (T.grad_RNA / T.grad_RNA.median())).values
    top = lg >= np.quantile(lg, 0.8)
    lab = np.full(len(T), -1); lab[top] = DBSCAN(eps=1.6 * sp, min_samples=4).fit_predict(xy[top])
    zid = pd.Series(lab[lab >= 0]).value_counts().index[0]; zone = lab == zid
    g = grad_vec(T.rho_X.values, xy); u = g[zone].mean(0); u /= np.linalg.norm(u); n = np.array([-u[1], u[0]])
    c = xy[zone].mean(0); s = (xy - c) @ u / sp; o = (xy - c) @ n / sp
    olo, ohi = np.percentile(o[zone], [10, 90]); band = (o >= olo) & (o <= ohi)
    smin = s[zone].min() - (s[zone].max() - s[zone].min()); band &= s >= smin
    print(f"[{key}] zone {zone.sum()} spots, band {band.sum()} spots, s {s[band].min():.1f}..{s[band].max():.1f}", flush=True)

    C, Z = MC.composition(key, bc), MC.tf_activity(key, bc)
    Cs = pd.DataFrame(MC.smooth(C.values, xy), index=bc, columns=[f"type:{v}" for v in C.columns])
    Zs = pd.DataFrame(MC.smooth(Z.values, xy), index=bc, columns=[f"TF:{v}" for v in Z.columns])
    P = C.values.clip(1e-9); ent = -(P / P.sum(1, keepdims=True) * np.log(P / P.sum(1, keepdims=True))).sum(1)
    Xm, genes = SC.rna(key, bc); steep = json.load(open(f"{OUT}/fig7_steep.json"))[key]["genes"][:6]
    want = list(dict.fromkeys(steep + GENES[key])); gcol = np.isin(genes, want) & ~pd.Series(genes).duplicated().values
    Gx = pd.DataFrame(MC.smooth(SC.logcpm(Xm, gcol), xy), index=bc, columns=[f"gene:{v}" for v in genes[gcol]])
    F = pd.concat([T[["rho_X", "rho_RNA", "gain_RNA"]].rename(columns={"rho_X": "axis:rho_ATAC", "rho_RNA": "axis:rho_RNA",
                                                                       "gain_RNA": "axis:RNA gain"}),
                   pd.Series(MC.smooth(ent[:, None], xy)[:, 0], index=bc, name="mix:composition entropy"), Cs, Zs, Gx], axis=1)
    Fb, sb = F[band], s[band]
    rng = np.random.default_rng(0); rows = []
    edges = np.linspace(sb.min(), sb.max(), NBIN + 1)
    for f in F.columns:
        y = Fb[f].values; r = fit1(sb, y)
        r["steep_at"], r["steep_slope"] = steepest(sb, y, edges)
        bsx = []
        for _ in range(NBOOT):
            i = rng.integers(0, len(sb), len(sb)); bsx.append(steepest(sb[i], y[i], edges)[0])
        r["steep_lo"], r["steep_hi"] = (float(v) for v in np.percentile(bsx, [2.5, 97.5]))
        # how much the feature changes across the band, in units of its whole-section sd
        r["delta_sd"] = float((np.percentile(y, 90) - np.percentile(y, 10)) / (F[f].std() + 1e-12))
        bs = []
        for _ in range(NBOOT if f.split(":")[0] in ("axis", "mix") or r["R2_sig"] > 0.5 else 0):
            i = rng.integers(0, len(sb), len(sb)); b = fit1(sb[i], y[i]); bs.append((b["s0"], b["w"]))
        if bs:
            bs = np.array(bs); r.update(s0_lo=float(np.nanpercentile(bs[:, 0], 2.5)), s0_hi=float(np.nanpercentile(bs[:, 0], 97.5)),
                                        w_lo=float(np.nanpercentile(bs[:, 1], 2.5)), w_hi=float(np.nanpercentile(bs[:, 1], 97.5)))
        rows.append(dict(feature=f, **r))
    R = pd.DataFrame(rows)
    # which axis carries each cell type globally
    Bx, Bo = SC.hinge(T.rho_X.values), SC.hinge(T.rho_RNA.values); one = np.ones((len(T), 1))
    R["R2_on_rhoATAC"] = [float(MC.r2(np.c_[one, Bx], F[[f]].values)[0]) for f in F.columns]
    R["R2_on_rhoRNA"] = [float(MC.r2(np.c_[one, Bo], F[[f]].values)[0]) for f in F.columns]
    R.to_csv(f"{OUT}/{key}_transition_fits.csv", index=False)
    bi = np.clip(np.digitize(sb, edges[1:-1]), 0, NBIN - 1)
    prof = Fb.groupby(bi).mean(); prof.insert(0, "s", pd.Series(sb).groupby(bi).mean().values)
    prof.insert(1, "n", pd.Series(sb).groupby(bi).size().values); prof.to_csv(f"{OUT}/{key}_transition_profiles.csv", index=False)
    pd.DataFrame(dict(barcode=bc, s=s, o=o, band=band, zone=zone)).to_csv(f"{OUT}/{key}_transition_spots.csv", index=False)
    S = dict(spot_spacing=sp, n_zone=int(zone.sum()), n_band=int(band.sum()), u=u.tolist(),
             s_zone=[float(s[zone].min()), float(s[zone].max())])
    json.dump(S, open(f"{OUT}/{key}_transition_summary.json", "w"), indent=1)
    show = R[(R.delta_sd > 0.8) | R.feature.str.startswith(("axis", "mix"))].sort_values("s0")
    print(show[["feature", "s0", "s0_lo", "s0_hi", "w", "R2_sig", "R2_bump", "peak", "delta_sd", "R2_on_rhoATAC", "R2_on_rhoRNA"]]
          .round(2).to_string(index=False))


if __name__ == "__main__":
    for k in (sys.argv[1:] or ["lpc10", "melanoma"]):
        run(k)
