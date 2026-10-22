"""Gene vs cis-peak analyses along a Saturn axis (paper Fig 2 h-i, Fig 3).

Every analysis works on k-NN smoothed, z-scored log1p CP10k matrices of genes (G) and peaks (P), and on
(gene column, peak column) pairs from `cis_pairs`. Distance to the TSS is the internal calibration:
near peaks should agree with their gene more than distal peaks of the same gene.

  cis_pairs          peaks within +-window of each gene's TSS (top max_peaks by count) and, optionally,
                     distal peaks (>= distal_min away, same chromosome) as a background
  function_pairs     1 CO-FIT  pairs whose slopes both survive the LRT gate in the SAME domain
                     2 SHAPE   similarity of the two fitted curves on a common rho grid (z-scored Pearson)
                     3 SPEED   |slope_peak| / |slope_gene| in the shared domain (both z-scored)
                     4 LAG     shift that best aligns the two curves, only where shape similarity >= sim_lag.
                               Descriptive: Granger on fitted values is NOT valid (two noiseless piecewise-linear
                               functions of one rho test "causal" 10-18 % of the time at a 5 % nominal rate)
  gradient_cosine    per domain, magnitude-weighted mean cos of the angle between gene and peak SPATIAL gradients
                     (scale-invariant, so a depth difference between modalities cannot create it)
  along_across_corr  per domain, gene-peak correlation of gradient components ALONG rho and ACROSS rho:
                     agreement only along rho = the isodepth captures cis regulation
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import piecewise as pw

WIN, MAXPK, MAXDIST, DISTAL_MIN, MINPEAK, NGRID, SIM_LAG = 1_000_000, 25, 10, 5_000_000, 50, 120, 0.5


def zz(M):
    return (M - M.mean(0)) / np.maximum(M.std(0), 1e-12)


def smoothed_logcpm(X, cols, xy, k: int = 15):
    """z-scored, k-NN smoothed log1p CP10k of the selected columns (library from all columns)"""
    import scipy.sparse as sp
    from .mechanism import smooth
    X = sp.csr_matrix(X); lib = np.maximum(np.asarray(X.sum(1)).ravel(), 1)
    return zz(smooth(np.log1p(np.asarray(X[:, cols].todense(), np.float32) / lib[:, None] * 1e4), xy, k))


def peak_positions(peak_names):
    """'chr:start-end' -> (chrom array, midpoint array)"""
    pc = np.array([p.split(":")[0] for p in peak_names])
    pm = np.array([(lambda a, b: (float(a) + float(b)) / 2)(*p.split(":")[1].split("-")) for p in peak_names])
    return pc, pm


def tss_table(gtf) -> dict:
    """gene -> (chrom, TSS), first GTF record wins"""
    from .annotate import all_tss
    ch, po, nm = all_tss(gtf); t = {}
    for c_, p_, n_ in zip(ch, po, nm):
        t.setdefault(n_, (c_, p_))
    return t


def cis_pairs(gene_idx, gene_names, peak_names, peak_counts, tss, window=WIN, max_peaks=MAXPK,
              min_count=MINPEAK, distal_min=None, max_distal=MAXDIST):
    """{gene column: cis peak columns}, and with distal_min also {gene column: distal peak columns}"""
    pc, pm = peak_positions(peak_names); cis, dist = {}, {}
    for gi in gene_idx:
        g = gene_names[gi]
        if g not in tss:
            continue
        c_, t_ = tss[g]; same = pc == c_; dd = np.abs(pm - t_)
        s1 = np.where(same & (dd <= window) & (peak_counts >= min_count))[0]
        if len(s1) == 0:
            continue
        if len(s1) > max_peaks:
            s1 = s1[np.argsort(-peak_counts[s1])[:max_peaks]]
        cis[gi] = s1
        if distal_min is not None:
            s2 = np.where(same & (dd >= distal_min) & (peak_counts >= min_count))[0]
            if len(s2) > max_distal:
                s2 = s2[np.argsort(-peak_counts[s2])[:max_distal]]
            dist[gi] = s2
    return (cis, dist) if distal_min is not None else cis


def curves(sl, it, d, lab, grid):
    """z-scored fitted per-domain curves of every feature on a rho grid, (len(grid), features)"""
    kk = lab[np.argmin(np.abs(d[None, :] - grid[:, None]), axis=1)]
    return zz(np.array([it[j][kk] + sl[j][kk] * grid for j in range(sl.shape[0])]).T)


def best_lag_many(A, B, grid, maxfrac=.25, step=2):
    """vectorised over pairs: shift of each peak curve (columns of B) that best matches its gene curve
    (columns of A). Returns (lag in rho units, correlation at that lag)."""
    n = len(grid); mx = int(n * maxfrac)
    a = zz(A[mx:n - mx]); L = a.shape[0]
    best = np.full(A.shape[1], -2.0); bl = np.zeros(A.shape[1], int)
    for s in range(-mx, mx + 1, step):
        r = (a * zz(B[mx + s:n - mx + s])).sum(0) / L
        m = r > best
        best[m] = r[m]; bl[m] = s
    return bl * (grid[1] - grid[0]), best


def function_pairs(G, P, d, lab, pairs, t=pw.T_LRT, n_grid=NGRID, sim_lag=SIM_LAG):
    """G (N x genes), P (N x peaks): smoothed z-scored matrices. pairs: list of (gene col, peak col, info dict).
    One row per (pair, domain) where BOTH slopes survive the gate, with slopes, speed ratio, concordance,
    curve similarity and lag."""
    grid = np.linspace(np.percentile(d, 1), np.percentile(d, 99), n_grid)
    sg, ig = pw.fit_gauss(G, lab, d, t=t); sp_, ip = pw.fit_gauss(P, lab, d, t=t)
    CG, CP = curves(sg, ig, d, lab, grid), curves(sp_, ip, d, lab, grid)
    GA, PB, meta = [], [], []
    for a, b, info in pairs:
        shared = np.where((np.abs(sg[a]) > 1e-12) & (np.abs(sp_[b]) > 1e-12))[0]
        for k in shared:
            GA.append(a); PB.append(b); meta.append((info, int(k), float(sg[a][k]), float(sp_[b][k]), len(shared)))
    if not meta:
        return pd.DataFrame()
    A, B = CG[:, GA], CP[:, PB]
    sim = (zz(A) * zz(B)).sum(0) / A.shape[0]
    lag = np.zeros(len(sim)); lagr = np.full(len(sim), np.nan)
    sel = np.where(np.abs(sim) >= sim_lag)[0]
    if len(sel):
        lag[sel], lagr[sel] = best_lag_many(A[:, sel], B[:, sel], grid)
    rows = []
    for i, (info, k, slg, slp, ns) in enumerate(meta):
        rows.append(dict(**info, domain=k, slope_gene=round(slg, 3), slope_peak=round(slp, 3),
                         speed_ratio=round(abs(slp) / max(abs(slg), 1e-9), 3), n_shared=ns,
                         concordant=int(np.sign(slg) == np.sign(slp)), similarity=round(float(sim[i]), 3),
                         lag_rho=round(float(lag[i]), 4), lag_r=round(float(lagr[i]), 3)))
    return pd.DataFrame(rows)


def gradient_cosine(gG, gP, ga, pb, lab, min_spots: int = 30):
    """gG (N, genes, 2), gP (N, peaks, 2) spatial gradients; ga / pb pair columns. Returns
    {domain: (weighted mean cos, mean |grad gene|, mean |grad peak|)} per pair."""
    out = {}
    for k in range(int(lab.max()) + 1):
        m = np.where(lab == k)[0]
        if len(m) < min_spots:
            continue
        A = gG[m][:, ga]; B = gP[m][:, pb]
        na, nb = np.linalg.norm(A, axis=2), np.linalg.norm(B, axis=2); w = na * nb
        cos = (A * B).sum(2) / np.maximum(w, 1e-12)
        out[k] = ((cos * w).sum(0) / np.maximum(w.sum(0), 1e-12), na.mean(0), nb.mean(0))
    return out


def along_across_corr(gG, gP, G_rho, ga, pb, lab, min_spots: int = 30):
    """per domain: correlation over spots of gene vs peak gradient components ALONG and ACROSS rho.
    Returns ({domain: (r_along, r_across)}, median share of gene-gradient energy along rho)."""
    from .gradient import along_across
    Ag, Bg = along_across(gG, G_rho); Ap, Bp = along_across(gP, G_rho)
    frac = float(np.median((Ag ** 2) / np.maximum(Ag ** 2 + Bg ** 2, 1e-18)))
    out = {}
    for k in range(int(lab.max()) + 1):
        m = np.where(lab == k)[0]
        if len(m) < min_spots:
            continue

        def cor(Xg, Xp):
            a = Xg[m][:, ga]; b = Xp[m][:, pb]
            a = (a - a.mean(0)) / np.maximum(a.std(0), 1e-12); b = (b - b.mean(0)) / np.maximum(b.std(0), 1e-12)
            return (a * b).mean(0)
        out[k] = (cor(Ag, Ap), cor(Bg, Bp))
    return out, frac
