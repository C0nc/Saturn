"""Methylation dimensionality-reduction benchmark on E11 spatial-DMT (GSE270498).

Inputs available (the paper's Seurat object keeps no per-region read counts):
  * beta   : methylation fraction per spot x VMR tile (85,427 tiles, 97% observed;
             exact-0 betas are indistinguishable from uncovered in the sparse export)
  * resid  : the paper's scbs shrunken residuals after iterative-PCA imputation
Coverage proxy: nFeature_DNAm (# covered tiles). NOTE nCount_DNAm is the SUM OF
RESIDUALS (= global methylation offset), not coverage.

METHODS (K components each)
  resid_pca         paper / scbs+MethSCAn: PCA of imputed shrunken residuals (top-var tiles)
  resid_pca_dropcov same, dropping leading PCs with |corr(coverage)| > 0.5
  resid_pca_covreg  regress log coverage out of every tile, then PCA
  resid_raw_ipca    UNSHRUNK residuals from beta (tile mean over covered spots) +
                    iterative PCA imputation of uncovered entries (MethSCAn without shrinkage)
  beta_cellnorm     ALLCools-style: beta / spot global mean beta, missing -> 1, PCA
  mvalue_pca        logit(beta) (array-style M-values), missing -> tile mean, PCA
  win100k_cellnorm  Amethyst-style: mean beta per 100 kb window, / spot global mean, PCA
  methylvi          MethylVI (scvi-tools) beta-binomial VAE; pseudo-coverage 10 where observed
                    (run separately in env `multivi` by meth_methylvi_helper.py)

METRICS (none uses the methylation cluster labels)
  cov_leak   max |Spearman| of the top 5 components with log coverage   (lower = better)
  glob_leak  max |Spearman| of the top 5 components with global methylation (reported, not scored:
             global methylation is partly biology here)
  rna_knn    15-NN 5-fold accuracy predicting RNA-only clusters SCT_snn_res.0.8   (higher = better)
  moran      variance-weighted Moran's I of top 5 components on a 6-NN spatial graph
"""
import os, sys, json, time, numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import cross_val_score, StratifiedKFold
import anndata as ad

K, NTOP = 10, 10000
EXP = "/mnt/HDD2B/Users/yancui/GSE270498/e11_dnam_export"
OUT = "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/results/meth_dr_bench"
os.makedirs(OUT, exist_ok=True)


def load():
    m = ad.read_h5ad("/mnt/HDD2B/Users/yancui/compare_vcr/results_meth/e11_meth.h5ad")
    bars = [l.strip() for l in open(f"{EXP}/barcodes.txt")]
    wins = [l.strip() for l in open(f"{EXP}/windows.txt")]
    B = sp.load_npz(f"{EXP}/beta_spots_x_windows.npz").tocsr()
    idx = pd.Index(bars).get_indexer(m.obs_names)
    assert (idx >= 0).all(), "barcode mismatch"
    B = B[idx]
    assert list(m.var_names) == wins or len(wins) == m.n_vars
    R = m.X.toarray() if hasattr(m.X, "toarray") else np.asarray(m.X)
    xy = pd.read_csv("/mnt/HDD2B/Users/yancui/compare_vcr/results_meth/e11_coords.csv").select_dtypes("number").iloc[:, -2:].values
    return m, B, R.astype(np.float32), np.array(wins), xy


def pca_top(X, k=K, ntop=NTOP):
    top = np.argsort(-X.var(0))[:min(ntop, X.shape[1])]
    Z = PCA(k + 3, random_state=0).fit_transform(X[:, top] - X[:, top].mean(0))
    return Z


def dropcov(Z, cov, k=K, thr=0.5, maxd=3):
    drop = [i for i in range(maxd) if abs(np.corrcoef(Z[:, i], cov)[0, 1]) > thr]
    return Z[:, [i for i in range(Z.shape[1]) if i not in drop][:k]], drop


def dense_beta(B):
    """beta dense + observation mask (nnz)."""
    Bd = B.toarray().astype(np.float32)
    obs = np.zeros_like(Bd, dtype=bool)
    Bc = B.tocoo(); obs[Bc.row, Bc.col] = True
    return Bd, obs


def iterative_pca_impute(Rz, obs, k=K, n_iter=10):
    X = np.where(obs, Rz, 0.0)
    for _ in range(n_iter):
        mu = X.mean(0)
        U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
        L = (U[:, :k] * S[:k]) @ Vt[:k] + mu
        X = np.where(obs, Rz, L)
    return X


def windows_100k(Bd, obs, wins):
    parts = np.array([w.split(".") for w in wins])
    key = pd.Series(parts[:, 0]) + ":" + (parts[:, 1].astype(int) // 100_000).astype(str)
    codes, uniq = pd.factorize(key)
    G = sp.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)), shape=(len(codes), len(uniq)))
    num = sp.csr_matrix(np.where(obs, Bd, 0)) @ G; den = sp.csr_matrix(obs.astype(np.float32)) @ G
    num, den = num.toarray(), den.toarray()
    W = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    keep = (den > 0).mean(0) > 0.5
    W = W[:, keep]
    W = np.where(np.isnan(W), np.nanmean(W, 0), W)
    return W


def metrics(Z, cov, glob, lab, xy):
    Z5 = Z[:, :5]
    cl = max(abs(spearmanr(Z5[:, i], cov)[0]) for i in range(5))
    gl = max(abs(spearmanr(Z5[:, i], glob)[0]) for i in range(5))
    Zs = Z / (Z[:, :1].std() + 1e-12)
    acc = cross_val_score(KNeighborsClassifier(15), Zs, lab, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
    nn = NearestNeighbors(n_neighbors=7).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
    mor, w = [], []
    for i in range(5):
        z = Z5[:, i] - Z5[:, i].mean()
        mor.append(len(z) * (z[:, None] * z[nn]).sum() / (nn.size * (z ** 2).sum())); w.append(z.var())
    return dict(cov_leak=round(cl, 3), glob_leak=round(gl, 3), rna_knn=round(float(acc), 3),
                moran=round(float(np.average(mor, weights=w)), 3))


def main():
    t0 = time.time()
    m, B, R, wins, xy = load()
    cov = np.log(m.obs["nFeature_DNAm"].values.astype(float))
    lab = m.obs["SCT_snn_res.0.8"].astype(str).values
    Bd, obs = dense_beta(B)
    glob = np.where(obs, Bd, 0).sum(1) / obs.sum(1)                       # spot global mean beta
    print(f"loaded {Bd.shape}, observed {obs.mean():.3f}, {time.time()-t0:.0f}s", flush=True)

    emb = {}
    Z = pca_top(R); emb["resid_pca"] = Z[:, :K]
    emb["resid_pca_dropcov"], d = dropcov(Z, cov); print("  dropcov dropped", d)
    Xr = R - np.c_[np.ones(len(cov)), cov] @ np.linalg.lstsq(np.c_[np.ones(len(cov)), cov], R, rcond=None)[0]
    emb["resid_pca_covreg"] = pca_top(Xr)[:, :K]; del Xr
    tmean = np.where(obs, Bd, 0).sum(0) / np.maximum(obs.sum(0), 1)
    top = np.argsort(-np.where(obs, (Bd - tmean) ** 2, 0).sum(0) / np.maximum(obs.sum(0), 1))[:NTOP]
    Ri = iterative_pca_impute((Bd - tmean)[:, top], obs[:, top])
    emb["resid_raw_ipca"] = pca_top(Ri)[:, :K]; del Ri
    Bn = np.where(obs, Bd / glob[:, None], 1.0)
    emb["beta_cellnorm"] = pca_top(Bn)[:, :K]; del Bn
    Mv = np.log((Bd * 10 + 0.5) / ((1 - Bd) * 10 + 0.5)); Mv = np.where(obs, Mv, np.nan)
    Mv = np.where(obs, Mv, np.nanmean(Mv, 0)); emb["mvalue_pca"] = pca_top(Mv)[:, :K]; del Mv
    W = windows_100k(Bd, obs, wins); print(f"  100kb windows: {W.shape[1]}")
    emb["win100k_cellnorm"] = pca_top(W / W.mean(1, keepdims=True))[:, :K]
    mv = f"{OUT}/methylvi_latent.npy"
    if os.path.exists(mv):
        emb["methylvi"] = np.load(mv)

    rows = []
    for k_, Z in emb.items():
        r = dict(method=k_, **metrics(Z, cov, glob, lab, xy)); rows.append(r); print(r, flush=True)
        np.save(f"{OUT}/emb_{k_}.npy", Z)
    df = pd.DataFrame(rows).set_index("method")
    df.to_csv(f"{OUT}/meth_dr_metrics.csv"); print(df.to_string())
    # cross-method agreement of PC-space: CCA-free proxy = mean |corr| of top-3 comps with resid_pca
    json.dump(dict(n_spots=int(len(cov)), coverage_vs_global_rho=float(spearmanr(cov, glob)[0])),
              open(f"{OUT}/meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
