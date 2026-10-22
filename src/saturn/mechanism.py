"""Why the other axis explains an RNA gene, and what changes where that axis is steep (Fig 7 i).

absorption   for each cross-explained gene: dR2 of B(rho_X) on top of B(rho_RNA) (crossterm.fit_split),
             then again with a candidate source block K already in the model:
                 absorbed share = 1 - dR2_X|K / dR2_X
             Blocks used in the paper (all k-NN smoothed, equal width = 6 columns):
               cis    the gene's own +-50 kb accessibility (hinge basis, first 6 columns)
               TF     top-6 PCs of per-spot chromVAR TF activity          (ATAC-only, independent of RNA)
               comp   top-6 PCs of cell-type composition                  (cell2location / Starfysh -- derived
                                                                          from the same RNA: partly circular)
               all    cis + TF + comp
               null   each named block spatially rotated (5 rotations each, mean) = what ANY smooth
                      6-column block absorbs. Rotate blocks separately: a 12-column null is unfair.
drivers      per gene: correlation of its cross profile with each cell type / TF (top 3 each)
boundary     per spot |gradient| (k-NN local linear fit, standardised) of every cell type / TF; log2 ratio of its
             mean in the X-steep quintile vs the RNA-steep quintile = what switches where rho_X is steep
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from .crossterm import hinge, fit_split, rotate

NPC, KS, KG, NROT = 6, 15, 25, 5


def smooth(V, xy, k=KS):
    """mean over the k nearest spots (each spot included)"""
    nn = cKDTree(xy).query(xy, k=k)[1]; return np.asarray(V)[nn].mean(1)


def pcs(V, n=NPC):
    V = (V - V.mean(0)) / (V.std(0) + 1e-9); U, S, _ = np.linalg.svd(V, full_matrices=False); return U[:, :n] * S[:n]


def local_grad(V, xy, k=KG):
    """|gradient| of each standardised column by a local linear fit on the k nearest spots"""
    V = (V - V.mean(0)) / (V.std(0) + 1e-9); X = (xy - xy.mean(0)) / xy.std(0)
    nn = cKDTree(X).query(X, k=k)[1]; out = np.empty(V.shape)
    for i in range(len(X)):
        A = X[nn[i]] - X[nn[i]].mean(0); b = V[nn[i]] - V[nn[i]].mean(0)
        out[i] = np.linalg.norm(np.linalg.lstsq(A, b, rcond=None)[0], axis=0)
    return out


def r2(A, Y):
    c = np.linalg.lstsq(A, Y, rcond=None)[0]
    return 1 - ((Y - A @ c) ** 2).sum(0) / (((Y - Y.mean(0)) ** 2).sum(0) + 1e-12)


def dR2(base, Y, Bx):
    return r2(np.c_[base, Bx], Y) - r2(base, Y)


def absorption(Y, genes, rho_rna, rho_x, xy, blocks: dict, cis=None, n_rot=NROT, seed=0):
    """Y (N x G) log1p CP10k of the cross genes. blocks: name -> (N x p) raw per-spot features (smoothed
    and reduced to NPC PCs here). cis: optional (gene names, N x n smoothed-or-raw windows) from
    crossterm.cis_windows. Returns a per-gene DataFrame with dR2_given_<block> and absorbed_<block>."""
    Y = np.asarray(Y, float); N = len(Y)
    Bo, Bx = hinge(rho_rna), hinge(rho_x); base = np.c_[np.ones(N), Bo]
    K = {nm: pcs(smooth(np.asarray(V), xy)) for nm, V in blocks.items()}        # dtype kept as given
    res = dict(gene=list(genes), dR2_X=dR2(base, Y, Bx))
    for nm, Kb in K.items():
        res[f"dR2_given_{nm}"] = dR2(np.c_[base, Kb], Y, Bx)
    rng = np.random.default_rng(seed); nul = []
    for _ in range(n_rot):
        nul += [dR2(np.c_[base, rotate(xy, Kb, rng)], Y, Bx) for Kb in K.values()]
    res["dR2_given_null"] = np.mean(nul, 0)
    if cis is not None:
        cg, W = cis; Ws = smooth(np.asarray(W, float), xy); gi = {g: i for i, g in enumerate(genes)}
        cd, ad = np.full(len(genes), np.nan), np.full(len(genes), np.nan)
        for j, g in enumerate(cg):
            if g not in gi:
                continue
            i = gi[g]; Kw = hinge(Ws[:, j])[:, :NPC]
            cd[i] = dR2(np.c_[base, Kw], Y[:, [i]], Bx)[0]
            ad[i] = dR2(np.hstack([base, Kw, *K.values()]), Y[:, [i]], Bx)[0]
        res["dR2_given_cis"], res["dR2_given_all"] = cd, ad
    D = pd.DataFrame(res)
    for c in [c for c in D if c.startswith("dR2_given_")]:
        D["absorbed_" + c[len("dR2_given_"):]] = 1 - D[c] / D.dR2_X
    return D


def drivers(Y, genes, rho_rna, rho_x, xy, tables: dict, n=3):
    """per gene: the n features of each table (e.g. {'type': composition, 'tf': chromVAR}) whose smoothed
    profile correlates most strongly (|r|) with the gene's cross profile"""
    _, _, _, prof = fit_split(Y, rho_rna, rho_x); G = prof.shape[1]
    rows = [dict(gene=g) for g in genes]
    for nm, T in tables.items():
        V = smooth(T.values.astype(float), xy); r = np.corrcoef(np.c_[prof, V].T)[:G, G:]
        for i in range(G):
            for j, o in enumerate(np.argsort(-np.abs(r[i]))[:n]):
                rows[i][f"{nm}{j + 1}"] = f"{T.columns[o]} ({r[i, o]:+.2f})"
    return pd.DataFrame(rows)


def boundary_switch(tables: dict, xy, steep):
    """what changes where rho_X is steep: per feature, mean local |gradient| in the X-steep quintile
    (steep >= 80th pct) over the RNA-steep quintile (<= 20th pct), log2; plus level ratio and the
    Spearman of its gradient with steepness. tables: prefix -> DataFrame (spots x features)."""
    feats = pd.concat([pd.DataFrame(smooth(T.values, xy), columns=[f"{p}:{c}" for c in T.columns])
                       for p, T in tables.items()], axis=1)
    top, bot = steep >= np.quantile(steep, 0.8), steep <= np.quantile(steep, 0.2)
    Gm = local_grad(feats.values, xy)
    B = pd.DataFrame(dict(feature=feats.columns, grad_Xsteep=Gm[top].mean(0), grad_RNAsteep=Gm[bot].mean(0),
                          grad_all=Gm.mean(0), level_Xsteep=feats.values[top].mean(0) / (feats.values.mean(0) + 1e-9)))
    B["log2_steep_ratio"] = np.log2(B.grad_Xsteep / B.grad_RNAsteep)
    B["rho_grad_vs_steepness"] = [spearmanr(Gm[:, j], steep).statistic for j in range(Gm.shape[1])]
    return B.sort_values("log2_steep_ratio", ascending=False).reset_index(drop=True)
