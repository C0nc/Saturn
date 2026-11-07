"""Target construction for Saturn: each modality -> a low-dimensional, z-scored target.

Saturn fits continuous per-modality embeddings (as GASTON fits GLM-PCs), not raw
counts. These are the two standard constructions.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _zscore(X: np.ndarray) -> np.ndarray:
    return ((X - X.mean(0)) / (X.std(0) + 1e-8)).astype(np.float32)


def scale_target(P: np.ndarray, scale: str = "global") -> np.ndarray:
    """Centre an (N, K) embedding and put it on the loss scale.

    "global" (DEFAULT): ONE scale for the whole modality, chosen so the average
        column variance is 1. Components keep their relative variance, so the
        MSE is (up to a constant) the fraction of the modality's variance left
        unexplained, and modalities with different K are balanced.
    "zscore": each component to unit variance -- GASTON's default. AVOID when a
        modality can carry more than one axis: it gives a 1%-variance component
        the same weight as a 35%-variance one. Measured on v5 (3 sims), fitting
        one isodepth to 15 z-scored RNA GLM-PCs recovered the WRONG axis
        (own |sp| 0.43, other 0.89); with global scaling, own 0.999.
    """
    P = np.asarray(P, dtype=np.float64)
    P = P - P.mean(0)
    if scale == "zscore":
        return _zscore(P)
    if scale != "global":
        raise ValueError(f"scale must be 'global' or 'zscore', got {scale!r}")
    return (P / np.sqrt(P.var(0).mean() + 1e-12)).astype(np.float32)


def rna_glmpc(counts: np.ndarray, K: int = 10, scale: str = "global") -> np.ndarray:
    """RNA -> K GLM-PCs, via GASTON's own approximation: Poisson Pearson
    residuals (expected count = spot depth x gene fraction, so depth is removed
    explicitly), clipped at sqrt(N), then centred PCA."""
    Y = np.asarray(counts, dtype=np.float64)
    n = Y.sum(1, keepdims=True)
    mu = n * (Y.sum(0) / Y.sum())[None, :]
    R = (Y - mu) / np.sqrt(mu + 1e-8)
    c = np.sqrt(Y.shape[0])
    R = np.clip(R, -c, c)
    R = R - R.mean(0)
    U, S, _ = np.linalg.svd(R, full_matrices=False)
    return scale_target(U[:, :K] * S[:K], scale)


def atac_lsi(counts: np.ndarray, K: int = 15, drop_first: int = 1,
             scale: str = "global", return_dropped: bool = False):
    """ATAC -> K LSI components (Signac/ArchR convention): TF-IDF -> log1p ->
    UNCENTRED SVD, then drop the first `drop_first` components.

    Why drop component 1: SVD of an uncentred non-negative matrix puts each
    spot's overall magnitude on component 1, and that magnitude tracks peaks
    detected, which tracks sequencing depth. Measured on the v5 simulator:
    component 1 |corr| with depth 0.63, with the true ATAC axis 0.07, while
    carrying 55% of the variance; component 2 tracks the ATAC axis at 0.97.
    On real data, check each leading component's correlation with depth and
    drop every one that tracks it (occasionally component 2 as well).
    """
    Y = np.asarray(counts, dtype=np.float64)
    tf = Y / Y.sum(1, keepdims=True).clip(min=1)
    idf = np.log(1 + Y.shape[0] / (Y.sum(0) + 1))
    X = np.log1p(tf * idf[None, :] * 1e4)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    k0 = int(drop_first)
    comps = scale_target(U[:, k0:k0 + K] * S[k0:k0 + K], scale)
    if return_dropped:
        return comps, U[:, :k0] * S[:k0]
    return comps


def zscore_coords(coords: np.ndarray) -> np.ndarray:
    """Centre and scale coordinates by ONE common scale (preserves aspect ratio).

    GASTON z-scores x and y separately, which stretches non-square tissue and
    distorts gradient directions; a single scale keeps geometry intact.
    """
    C = np.asarray(coords, dtype=np.float64)
    C = C - C.mean(0)
    return (C / (C.std() + 1e-8)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Real-data versions: sparse-aware, with feature filtering and depth checks.
# The simulator targets above assume small dense matrices; real DBiT data has
# ~60k genes / ~80k peaks per spot, which would be ~1 GB dense per modality.
# --------------------------------------------------------------------------- #
def _as_csr(X):
    import scipy.sparse as sp
    return X.tocsr().astype(np.float64) if sp.issparse(X) else sp.csr_matrix(np.asarray(X, dtype=np.float64))


def rna_glmpc_real(counts, K: int = 10, n_hvg: int = 3000, min_frac: float = 0.01,
                   method: str = "glmpca", scale: str = "global"):
    """RNA counts (N, G), sparse OK -> K GLM-PCs.

    Genes detected in >= min_frac of spots; the n_hvg with the largest Poisson
    Pearson-residual variance are kept (GASTON/scanpy 'pearson_residuals' HVG idea).
    method="glmpca" (default): true Poisson GLM-PCA (Townes et al. 2019, `glmpca`
      package, as in GASTON): log mu = log(depth) + U V^T fitted iteratively,
      depth an explicit offset; penalty 1, <= 300 iterations.
    method="residual": the closed-form approximation -- PCA of Poisson Pearson
      residuals under the depth x gene-frequency null.
    Returns (targets, info) where info records kept genes and depth.
    """
    X = _as_csr(counts)
    N = X.shape[0]
    det = np.asarray((X > 0).sum(0)).ravel() / N
    keep = np.where(det >= min_frac)[0]
    X = X[:, keep]
    n = np.asarray(X.sum(1)).ravel()
    p = np.asarray(X.sum(0)).ravel() / X.sum()
    # residual variance per gene without densifying all genes at once
    rv = np.zeros(X.shape[1])
    c = np.sqrt(N)
    for j0 in range(0, X.shape[1], 2000):
        sl = slice(j0, j0 + 2000)
        mu = n[:, None] * p[None, sl]
        R = np.clip((X[:, sl].toarray() - mu) / np.sqrt(mu + 1e-8), -c, c)
        rv[sl] = R.var(0)
    top = np.argsort(-rv)[:min(n_hvg, len(rv))]
    Xd = X[:, top].toarray()
    if method == "glmpca":
        from glmpca.glmpca import glmpca
        keep_s = Xd.sum(1) > 0
        F, used = None, None
        for pen in (1, 5, 20, 100):          # glmpca diverges on some counts; its own fix is a bigger penalty
            try:
                res = glmpca(Xd[keep_s].T, L=K, fam="poi", penalty=pen, ctl={"maxIter": 300, "eps": 1e-4})
                F, used = np.asarray(res["factors"]), pen
                if np.isfinite(F).all():
                    break
            except Exception as e:
                print(f"      glmpca penalty={pen} failed ({type(e).__name__}); retrying", flush=True)
        if F is not None and np.isfinite(F).all():
            Z = np.zeros((len(Xd), K)); Z[keep_s] = F
            print(f"      glmpca converged with penalty={used}", flush=True)
            return scale_target(Z - Z.mean(0), scale), dict(genes=keep[top], depth=n, method="glmpca", penalty=used)
        print("      glmpca DIVERGED at every penalty -> Pearson-residual PCA for this dataset", flush=True)
    mu = n[:, None] * p[None, top]
    R = np.clip((Xd - mu) / np.sqrt(mu + 1e-8), -c, c)
    R -= R.mean(0)
    U, S, _ = np.linalg.svd(R, full_matrices=False)
    return scale_target(U[:, :K] * S[:K], scale), dict(genes=keep[top], depth=n, method="residual")


def lsi_real(counts, K: int = 15, min_frac: float = 0.005, max_features: int = 50000,
             drop_first: bool = True, depth_corr_drop: float = 0.5, max_drop: int = 3,
             scale: str = "global"):
    """Peak counts (ATAC) -> K LSI components, sparse.

    TF-IDF -> log1p -> randomized truncated SVD (uncentred, Signac/ArchR).
    drop_first (DEFAULT): always drop component 1, the Signac convention (it is the
    depth component on every dataset we measured: |corr(depth)| 0.97 LPC10, 0.43
    melanoma tumour domain). drop_first=False instead drops each of the first
    `max_drop` components whose |corr| with depth exceeds depth_corr_drop.
    Either way the measured depth correlations are reported.
    """
    from sklearn.decomposition import TruncatedSVD
    X = _as_csr(counts)
    N = X.shape[0]
    det = np.asarray((X > 0).sum(0)).ravel() / N
    keep = np.where(det >= min_frac)[0]
    if len(keep) > max_features:
        keep = keep[np.argsort(-det[keep])[:max_features]]
    X = X[:, keep]
    depth = np.asarray(X.sum(1)).ravel()
    tf = X.multiply(1.0 / np.maximum(depth, 1)[:, None]).tocsr()
    idf = np.log(1 + N / (np.asarray(X.sum(0)).ravel() + 1))
    T = tf.multiply(idf[None, :]).tocsr() * 1e4
    T.data = np.log1p(T.data)
    svd = TruncatedSVD(n_components=K + max_drop, algorithm="randomized",
                       n_iter=7, random_state=0)
    Z = svd.fit_transform(T)                                   # = U * S
    dc = np.array([abs(np.corrcoef(Z[:, i], depth)[0, 1]) for i in range(max_drop)])
    drop = [0] if drop_first else [i for i in range(max_drop) if dc[i] > depth_corr_drop]
    kept = [i for i in range(Z.shape[1]) if i not in drop][:K]
    return scale_target(Z[:, kept], scale), dict(peaks=keep, depth=depth,
                                                 depth_corr=dc, dropped=drop)


# --------------------------------------------------------------------------- #
# Further modalities
# --------------------------------------------------------------------------- #
def isoform_usage_pca(iso_counts, iso_gene, K: int = 10, alpha: float = 5.0,
                      min_gene_reads: int = 200, n_top: int = 3000, depth=None,
                      transform: str = "clr", scale: str = "global"):
    """Long-read isoforms (spots x isoforms counts) -> K components of isoform USAGE.

    Usage is what isoform methods test (Spl-IsoFind, SPLISOSM, scisorseqr): the
    fraction of a gene's reads carried by each isoform, independent of the gene's
    expression level. Per spot s, gene g, isoform i:
        u_si = (n_si + alpha * p_gi) / (n_sg + alpha)
    (empirical-Bayes shrinkage toward the gene's global usage p_gi, so spots with
    few reads of g sit at p_gi instead of 0/1). Usage is compositional (sums to 1
    within a gene), so transform="clr" applies Aitchison's centred log-ratio within
    each gene: log u_si - mean_{i' in g} log u_si' (default; "ratio" keeps u - p).
    Genes need >= 2 isoforms and
    >= min_gene_reads reads overall. Centre on p_gi, keep the n_top isoforms with the
    largest usage variance, regress log(depth) (isoform-assigned reads per spot) out
    of every column -- shrinkage makes low-depth spots uniformly closer to p_gi --
    then PCA.
    """
    C = sp.csr_matrix(iso_counts, dtype=np.float64)
    g = np.asarray(iso_gene)
    codes, genes = pd.factorize(g)
    G = sp.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)), shape=(len(codes), len(genes)))
    tot_iso = np.asarray(C.sum(0)).ravel()
    tot_gene = np.asarray((C @ G).sum(0)).ravel()
    n_iso = np.asarray(G.sum(0)).ravel()
    keep_g = (n_iso >= 2) & (tot_gene >= min_gene_reads)
    keep_i = keep_g[codes] & (tot_iso > 0)
    idx = np.where(keep_i)[0]
    p = tot_iso[idx] / tot_gene[codes[idx]]
    Ns = (C @ G).toarray()[:, codes[idx]]                    # spot x kept-isoform gene totals
    Ni = C[:, idx].toarray()
    U = (Ni + alpha * p) / (Ns + alpha)
    if transform == "clr":
        L = np.log(U)
        gk, ginv = np.unique(codes[idx], return_inverse=True)
        Gk = sp.csr_matrix((np.ones(len(ginv)), (np.arange(len(ginv)), ginv)), shape=(len(ginv), len(gk)))
        gmean = (sp.csr_matrix(L) @ Gk).toarray() / np.asarray(Gk.sum(0)).ravel()
        U = L - gmean[:, ginv]
    else:
        U = U - p
    top = np.argsort(-U.var(0))[:min(n_top, U.shape[1])]
    X = U[:, top]
    if depth is not None:
        Cv = np.c_[np.ones(len(X)), np.log(np.asarray(depth, float) + 1)]
        X = X - Cv @ np.linalg.lstsq(Cv, X, rcond=None)[0]
    X = X - X.mean(0)
    Uu, S, _ = np.linalg.svd(X, full_matrices=False)
    return scale_target(Uu[:, :K] * S[:K], scale), dict(isoforms=idx[top], n_genes=int(keep_g.sum()),
                                                        n_isoforms=int(len(idx)))


def compartment_pca(AB, K: int = 10, scale: str = "global"):
    """Spatial Hi-C (spots x genomic bins of A/B compartment scores) -> K
    components: centred PCA over bins. Each component is a genome-wide pattern of
    compartment switching that varies across space."""
    X = np.asarray(AB, dtype=np.float64)
    X = np.nan_to_num(X - np.nanmean(X, 0))
    X = X[:, X.var(0) > 0]
    U, S, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    return scale_target(U[:, :K] * S[:K], scale), dict(n_bins=X.shape[1])


def protein_clr_pca(counts, K: int = 10, scale: str = "global"):
    """Protein / ADT counts (spots x proteins), ALL proteins -> CLR per spot
    (log1p minus the spot's mean log1p, the Seurat/totalVI convention) -> PCA."""
    X = np.asarray(counts.toarray() if hasattr(counts, "toarray") else counts, dtype=np.float64)
    L = np.log1p(X)
    C = L - L.mean(1, keepdims=True)
    C -= C.mean(0)
    K = min(K, C.shape[1] - 1)
    U, S, _ = np.linalg.svd(C, full_matrices=False)
    return scale_target(U[:, :K] * S[:K], scale), dict(depth=X.sum(1))


def atac_spectral(tiles, n_comps: int = 15, n_features: int = 250000, drop_first: int = 1, scale: str = "global"):
    """ATAC tile matrix -> SnapATAC2 spectral embedding, leading component(s) dropped (the published
    ATAC target for LPC10 and melanoma). `tiles` is an AnnData of fragment tile counts, e.g. from
        data = snap.pp.import_fragments(frag, chrom_sizes=snap.genome.mm10, file=tmp, whitelist=barcodes)
        snap.pp.add_tile_matrix(data, bin_size=5000); tiles = data.to_memory()
    restricted to the spots being fitted. Recompute on every new spot selection: a reduction computed
    on a larger section carries that section's variance, not the selected region's.
    Returns (targets, info with depth and |Spearman| of each dropped component with depth)."""
    import snapatac2 as snap
    from scipy.stats import spearmanr
    A = tiles.copy()
    snap.pp.select_features(A, n_features=n_features)
    snap.tl.spectral(A, n_comps=n_comps)
    Z = np.asarray(A.obsm["X_spectral"])
    depth = np.asarray(_as_csr(A.X).sum(1)).ravel()
    dc = [abs(spearmanr(Z[:, i], depth)[0]) for i in range(drop_first)]
    return scale_target(Z[:, drop_first:], scale), dict(depth=depth, depth_corr_dropped=dc)


def check_target(Y, depth=None, coords=None, labels=None, n_check: int = 5, knn: int = 15, n_spatial: int = 6):
    """Score a candidate reduction BEFORE fitting.

    depth_leak  max |Spearman| of the first n_check components with log depth   (lower is better; < ~0.3)
    label_knn   15-NN 5-fold accuracy predicting `labels` from Y, e.g. RNA-only clusters -- an
                independent biology check that never uses this modality's own clusters (higher is better)
    moran       variance-weighted Moran's I of the first n_check components on a 6-NN spatial graph
                (spatial structure that an isodepth can fit; higher is better)
    var_top     share of the variance held by each component, and `elbow` = first component whose share
                falls below a tenth of the previous one (None when the spectrum is flat -- a flat spectrum,
                e.g. Fast-Higashi top-10 = 8 % of variance, is fitted badly by 1-D isodepths)
    """
    from scipy.stats import spearmanr
    Y = np.asarray(Y, float); ok = np.isfinite(Y).all(1); Y = Y[ok]; n = min(n_check, Y.shape[1])
    v = Y.var(0); share = v / v.sum()
    drop = [i for i in range(1, len(share)) if share[i] < 0.1 * share[i - 1]]
    out = dict(var_share=np.round(share, 4).tolist(), elbow=(drop[0] if drop else None))
    if depth is not None:
        dep = np.log1p(np.asarray(depth, float)[ok])
        out["depth_corr"] = [round(float(abs(spearmanr(Y[:, i], dep)[0])), 3) for i in range(n)]
        out["depth_leak"] = max(out["depth_corr"])
    if labels is not None:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        lab = np.asarray(labels).astype(str)[ok]
        out["label_knn"] = round(float(cross_val_score(KNeighborsClassifier(knn), Y / (Y[:, :1].std() + 1e-12), lab,
                                                       cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()), 3)
    if coords is not None:
        from sklearn.neighbors import NearestNeighbors
        xy = np.asarray(coords, float)[ok]
        nn = NearestNeighbors(n_neighbors=n_spatial + 1).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
        mor, w = [], []
        for i in range(n):
            z = Y[:, i] - Y[:, i].mean()
            mor.append(len(z) * (z[:, None] * z[nn]).sum() / (nn.size * (z ** 2).sum())); w.append(z.var())
        out["moran"] = round(float(np.average(mor, weights=w)), 3)
    return out
