"""What the Saturn cross pathways explain (paper Fig 7).

Given a fitted model with an RNA axis and one other modality X:

  spot_decomposition  per spot: own term, cross term, full prediction, and
      local gain   ||Y_i - P_own_i||^2 - ||Y_i - P_full_i||^2, over the mean ||Y_i - Ybar||^2, averaged over
                   the k nearest spots = where the other axis explains this modality's variance
                   (P_own = full prediction with the cross term held at its mean)
      gradients    |grad rho| of each axis by autograd through the isodepth nets (k-NN averaged vectors)
      misalign     1 - |cos| between the two gradient directions
      rate_diff    |log| ratio of the two gradient magnitudes, each over its median
  gene_cross_test     every gene detected in >= 5% of spots: log1p CP10k ~ B(rho_RNA) vs ~ B(rho_RNA) + B(rho_X),
                      B = linear + 5 ReLU hinges at the 10-90% quantiles (Saturn's pathway family); partial F, BH.
                      The gene's cross profile = the B(rho_X) part of the full fit.
  cis_concordance     (X = ATAC) corr(gene cross profile, cross profile of the gene's own +-50 kb accessibility)
                      vs the same gene paired with a random other gene's window
  steepness_test      RNA's local gain vs steepness = log((|grad rho_X|/median) / (|grad rho_RNA|/median)),
                      Spearman, p against rotations of the gain field (spatial autocorrelation kept)
  gene_length_test    gene body length vs cross gain
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree
from scipy.stats import f as fdist, spearmanr, mannwhitneyu

KNN, MIN_DET, NDRAW, WIN = 25, 0.05, 200, 50_000


# ------------------------------------------------------------------ helpers
def hinge(d):
    """linear + 5 ReLU hinges at the 10-90% quantiles: Saturn's pathway family for one axis"""
    d = np.asarray(d, float)
    q = np.quantile(d, np.linspace(0.1, 0.9, 5)); z = (d - d.mean()) / d.std()
    return np.column_stack([z] + [np.maximum(d - k, 0) / d.std() for k in q])


def fit_split(Y, d_own, d_x):
    """Y (N x G). R2 own-only, R2 own + X, partial-F p, and the centred B(d_x) part of the full fit (N x G)"""
    Y = np.asarray(Y, float); N = len(Y); Bo, Bx = hinge(d_own), hinge(d_x)
    A0 = np.c_[np.ones(N), Bo]; A1 = np.c_[A0, Bx]
    tss = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-12
    c0 = np.linalg.lstsq(A0, Y, rcond=None)[0]; r0 = ((Y - A0 @ c0) ** 2).sum(0)
    c1 = np.linalg.lstsq(A1, Y, rcond=None)[0]; r1 = ((Y - A1 @ c1) ** 2).sum(0)
    q, df2 = Bx.shape[1], N - A1.shape[1]
    F = ((r0 - r1) / q) / (r1 / df2 + 1e-12); p = fdist.sf(F, q, df2)
    prof = Bx @ c1[A0.shape[1]:]
    return 1 - r0 / tss, 1 - r1 / tss, p, prof - prof.mean(0)


def bh(p):
    """Benjamini-Hochberg q-values"""
    p = np.asarray(p); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.clip(q, 0, 1); return out


def logcpm(X, cols=None):
    """log1p CP10k of the selected columns, library size from all columns"""
    X = sp.csr_matrix(X); lib = np.asarray(X.sum(1)).ravel().clip(1)
    cols = np.ones(X.shape[1], bool) if cols is None else np.asarray(cols)
    idx = np.where(cols)[0] if cols.dtype == bool else cols
    return np.log1p(np.asarray(sp.csc_matrix(X)[:, idx].astype(float).todense()) / lib[:, None] * 1e4)


def corr_cols(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    return (a * b).sum(0) / (np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0)) + 1e-12)


def rotate(xy, v, rng):
    """values v moved by a random rotation of the section about its centre (nearest spot):
    a null that keeps the spatial autocorrelation of v"""
    c = xy.mean(0); th = rng.uniform(0, 2 * np.pi)
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return v[cKDTree((xy - c) @ Rm.T + c).query(xy, k=1)[1]]


def steepness(T):
    """log((|grad rho_X|/median) / (|grad rho_RNA|/median)) per spot: > 0 where X changes faster"""
    return np.log((T.grad_X / T.grad_X.median()) / (T.grad_RNA / T.grad_RNA.median())).values


# ------------------------------------------------------------------ per spot
def spot_decomposition(model, coords_z, coords, targets, i_rna: int, i_x: int, signs=(1, 1), knn: int = KNN):
    """Per-spot table (rho, gradients, misalignment, rate difference, local gain and cross share of both
    modalities) and a summary (coupling, R2 full / own-only). `signs` orients (rho_RNA, rho_X) as in the
    figures (see saturn.fit.orient). `targets` are the Saturn targets, in model order."""
    model = model.cpu().eval(); model.cross_on = True
    xy = torch.tensor(np.asarray(coords_z, np.float32), requires_grad=True)
    d, full, own, cross = model(xy)
    raw = np.asarray(coords, float); sdv = raw.std(0); nn_ = cKDTree(raw).query(raw, k=knn)[1]
    sg = {i_rna: signs[0], i_x: signs[1]}
    D = d.detach().numpy()
    grads = {}
    for i in (i_rna, i_x):
        g = torch.autograd.grad(d[:, i].sum(), xy, retain_graph=True)[0].numpy() / sdv * sg[i]
        grads[i] = g[nn_].mean(1)
    u = {i: grads[i] / (np.linalg.norm(grads[i], axis=1, keepdims=True) + 1e-12) for i in grads}
    cosv = (u[i_rna] * u[i_x]).sum(1)
    gn = {i: np.linalg.norm(grads[i], axis=1) for i in grads}
    rate = np.abs(np.log((gn[i_x] / np.median(gn[i_x]) + 1e-6) / (gn[i_rna] / np.median(gn[i_rna]) + 1e-6)))
    out = dict(x=raw[:, 0], y=raw[:, 1], rho_RNA=D[:, i_rna] * sg[i_rna], rho_X=D[:, i_x] * sg[i_x], cos=cosv,
               misalign=1 - np.abs(cosv), grad_RNA=gn[i_rna], grad_X=gn[i_x], rate_diff=rate)
    S = dict(coupling=np.round(model.coupling(xy.detach()), 3).tolist())
    for i, nm in ((i_rna, "RNA"), (i_x, "X")):
        Y = np.asarray(targets[i], float); F = full[i].detach().numpy(); C = cross[i].detach().numpy()
        O = own[i].detach().numpy()
        Po = F - C + C.mean(0); den = ((Y - Y.mean(0)) ** 2).sum(1).mean()
        gain = (((Y - Po) ** 2).sum(1) - ((Y - F) ** 2).sum(1)) / den
        out[f"gain_{nm}"] = gain[nn_].mean(1)
        vo, vc = ((O - O.mean(0)) ** 2).sum(1), ((C - C.mean(0)) ** 2).sum(1)
        out[f"crossshare_{nm}"] = (vc / (vo + vc + 1e-12))[nn_].mean(1)
        tss = ((Y - Y.mean(0)) ** 2).sum()
        S[f"R2_{nm}"] = dict(full=float(1 - ((Y - F) ** 2).sum() / tss), own_only=float(1 - ((Y - Po) ** 2).sum() / tss))
    T = pd.DataFrame(out)
    for nm in ("RNA", "X"):
        S[f"spearman_misalign_gain_{nm}"] = float(spearmanr(T.misalign, T[f"gain_{nm}"]).statistic)
        S[f"spearman_ratediff_gain_{nm}"] = float(spearmanr(T.rate_diff, T[f"gain_{nm}"]).statistic)
    return T, S


# ------------------------------------------------------------------ per gene
def gene_cross_test(counts, genes, rho_rna, rho_x, min_det: float = MIN_DET):
    """Which RNA genes need the other axis. Returns (table sorted by dR2_cross, cross profiles N x G
    for the tested genes, tested gene names)."""
    X = sp.csr_matrix(counts); genes = np.asarray(genes).astype(str)
    det = np.asarray((X > 0).mean(0)).ravel(); ok = det >= min_det
    L = logcpm(X, ok); g = genes[ok]
    r0, r1, p, prof = fit_split(L, rho_rna, rho_x)
    G = pd.DataFrame(dict(gene=g, detect=det[ok], R2_own=r0, R2_own_plus_X=r1, dR2_cross=r1 - r0, p=p, q=bh(p),
                          cross_sd=prof.std(0)))
    return G.sort_values("dR2_cross", ascending=False).reset_index(drop=True), prof, g


def cis_windows(atac_counts, peak_names, genes, gtf, window: int = WIN, min_counts: int = 50):
    """log1p CP10k of each gene's summed +-window accessibility around its TSS.
    Returns (genes kept, N x n_kept matrix, peak indices per kept gene)."""
    from .annotate import all_tss
    Xa = sp.csc_matrix(atac_counts).astype(float); lib = np.asarray(Xa.sum(1)).ravel().clip(1)
    pk = pd.Series([str(v) for v in peak_names]).str.extract(r"(chr[\w]+)[:\-_](\d+)[\-_](\d+)")
    pch = pk[0].values; pmid = (pk[1].astype(float) + pk[2].astype(float)).values / 2
    ch, tss, nm = all_tss(gtf); first = pd.Series(range(len(nm)), index=nm).groupby(level=0).first()
    win, keep = {}, []
    for g in genes:
        if g not in first.index:
            continue
        j = first[g]; idx = np.where((pch == ch[j]) & (np.abs(pmid - tss[j]) <= window))[0]
        if len(idx) and Xa[:, idx].sum() >= min_counts:
            win[g] = idx; keep.append(g)
    W = np.column_stack([np.log1p(np.asarray(Xa[:, win[g]].sum(1)).ravel() / lib * 1e4) for g in keep]) \
        if keep else np.zeros((Xa.shape[0], 0))
    return keep, W, win


def cis_concordance(prof, prof_genes, W, W_genes, rho_rna, rho_x, ndraw: int = NDRAW, seed: int = 0):
    """corr(gene cross profile, its own window's cross profile) vs random-window pairing.
    Returns (per-gene DataFrame, summary with z and permutation p)."""
    gi = {g: i for i, g in enumerate(prof_genes)}
    Pg = np.column_stack([prof[:, gi[g]] for g in W_genes])
    _, _, _, Pw = fit_split(W, rho_rna, rho_x)
    obs = corr_cols(Pg, Pw); rng = np.random.default_rng(seed)
    ctrl = np.array([corr_cols(Pg, Pw[:, rng.permutation(len(W_genes))]).mean() for _ in range(ndraw)])
    S = dict(n_genes=len(W_genes), mean_obs=float(obs.mean()), ctrl_mean=float(ctrl.mean()), ctrl_sd=float(ctrl.std()),
             z=float((obs.mean() - ctrl.mean()) / ctrl.std()), p=float((1 + (ctrl >= obs.mean()).sum()) / (ndraw + 1)),
             frac_pos=float((obs > 0).mean()))
    return pd.DataFrame(dict(gene=list(W_genes), concordance=obs)), S


# ------------------------------------------------------------------ steepness and length
def steepness_test(T, nrot: int = 200, seed: int = 0):
    """RNA gain vs steepness (X steeper) and X gain vs the reverse: Spearman, two-sided rotation p,
    gain by steepness quintile, and the same correlation with log depth partialled out if T has 'depth'."""
    xy = T[["x", "y"]].values; s = steepness(T); rng = np.random.default_rng(seed); out = {}
    for nm, lead in (("RNA", s), ("X", -s)):
        g = T[f"gain_{nm}"].values; obs = float(spearmanr(lead, g).statistic)
        null = np.array([spearmanr(lead, rotate(xy, g, rng)).statistic for _ in range(nrot)])
        q = pd.qcut(pd.Series(lead).rank(method="first"), 5, labels=False)
        out[f"steeper_{nm}"] = dict(rho=obs, p_rot=float((1 + (np.abs(null) >= abs(obs)).sum()) / (nrot + 1)),
                                    null_sd=float(null.std()),
                                    gain_by_quintile=pd.Series(g).groupby(q.values).mean().round(4).tolist())
        if "depth" in T:
            out[f"steeper_{nm}"]["rho_depth_partial"] = float(spearmanr(_rank_resid(lead, T.depth.values),
                                                                        _rank_resid(g, T.depth.values)).statistic)
    return out


def _rank_resid(a, b):
    """ranks of a with the ranks of b regressed out (depth control for a Spearman correlation)"""
    from scipy.stats import rankdata
    a, b = rankdata(a), rankdata(b); A = np.c_[np.ones_like(b), b]
    return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]


def gene_length_test(G, lengths: dict, ntop: int = 200):
    """gene body length vs cross gain and vs own-axis R2 (Spearman); top-ntop cross genes vs all (Mann-Whitney)"""
    G = G.assign(length=G.gene.map(lengths)).dropna(subset=["length"])
    top = G.nlargest(ntop, "dR2_cross").length; lg = np.log10(G.length)
    return dict(rho_cross=float(spearmanr(lg, G.dR2_cross).statistic), rho_own=float(spearmanr(lg, G.R2_own).statistic),
                median_top_kb=float(top.median() / 1e3), median_all_kb=float(G.length.median() / 1e3),
                p=float(mannwhitneyu(top, G.length).pvalue))
