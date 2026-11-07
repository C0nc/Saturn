"""Isoform target for RT&T-MERFISH section 5 by BINOMIAL GLM-PCA on raw counts (100 um bins).

Published isoform/splicing reductions either run PCA on shrunken PSI (LeafCutter, BRIE2, scQuint-PCA)
or model the RAW counts with a count likelihood (scQuint's Dirichlet-multinomial VAE, which
"better distinguishes cell types than PCA"; IsoVAE). This is the count-likelihood route with a linear
decoder, the isoform counterpart of the Poisson GLM-PCA used for RNA (Townes et al. 2019):

    n_major[s, g] ~ Binomial( n_gene[s, g], p[s, g] )
    logit p[s, g] = a_g + U_s . V_g                       L = 10 factors
    objective     = binomial deviance + ridge * (|U|^2 + |V|^2)     (GLM-PCA's penalty)

* n_major = reads of the gene's dominant isoform (most reads section-wide); n_gene = all its isoforms.
* Bins with n_gene = 0 contribute nothing: sparsity is handled by the likelihood, no pooling, no pseudocount.
* Depth enters only as the number of trials: NO depth regression and NO plane removal.
* glmpca's 'mult' family uses the bin total as trials for every feature, so it cannot express per-gene
  trials -- hence this small PyTorch implementation (full-batch L-BFGS).
QC: genes with >= 2 isoforms, >= MIN_READS reads, 0.05 < global major share < 0.95; bins with >= MIN_CELLS cells.
Output factors are rotated to orthogonal, variance-ordered components (SVD of U V^T), then scale_target.

--depthcov: add log isoform depth as a BIN COVARIATE inside the model (GLM-PCA's X term), with a per-gene
slope:  logit p[s, g] = a_g + b_g * log depth_s + U_s . V_g . Without it, factor 1 tracked depth at -0.76
(partial r -0.66 given Slc17a7): bin quality shifts the major share itself, which trials alone cannot absorb.

--hv N: keep only the N genes with the largest binomial OVERDISPERSION of major-isoform usage,
  phi_g = sum_s (n_major - n_gene p_g)^2 / (n_gene p_g (1 - p_g)) / #bins with n_gene > 0   (1 = pure sampling noise),
  i.e. genes whose usage varies across bins more than binomial noise predicts (the HVG rule for usage).

Usage: python scripts/rtt_isoform_binomial_glmpca.py [--depthcov] [--hv 1000]
"""
import sys
import numpy as np, pandas as pd, scipy.sparse as sp, torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/saturn_repo/src")
from prep_real2 import H, save, rna_glmpc_real
from saturn.preprocess import scale_target
import rtt_isoform_variants as V

OUT = f"{H}/Hole/gaston_mix_mm/results/real_saturn/rtt_isoform_variants"
L, RIDGE, MIN_READS, MIN_CELLS, BIN, SEED = 10, 1.0, 100, 5, 100.0, 0


def major_counts(Xi, genes):
    Xi = sp.csc_matrix(Xi).astype(float); tot = np.asarray(Xi.sum(0)).ravel()
    S, N, names = [], [], []
    for g, idx in pd.Series(np.arange(len(genes))).groupby(genes):
        idx = idx.values
        if len(idx) < 2 or tot[idx].sum() < MIN_READS: continue
        j = idx[np.argmax(tot[idx])]; pm = tot[j] / tot[idx].sum()
        if not 0.05 < pm < 0.95: continue
        S.append(np.asarray(Xi[:, j].todense()).ravel()); N.append(np.asarray(Xi[:, idx].sum(1)).ravel()); names.append(g)
    return np.column_stack(S), np.column_stack(N), np.array(names)


def binomial_glmpca(S, N, L=L, ridge=RIDGE, seed=SEED, iters=300, cov=None):
    torch.manual_seed(seed)
    s = torch.as_tensor(S, dtype=torch.float64); n = torch.as_tensor(N, dtype=torch.float64)
    p0 = (s.sum(0) / n.sum(0)).clamp(1e-4, 1 - 1e-4)
    a = torch.log(p0 / (1 - p0)).clone().requires_grad_(True)
    U = (0.01 * torch.randn(S.shape[0], L, dtype=torch.float64)).requires_grad_(True)
    Vv = (0.01 * torch.randn(S.shape[1], L, dtype=torch.float64)).requires_grad_(True)
    c = None if cov is None else torch.as_tensor((cov - cov.mean()) / cov.std(), dtype=torch.float64)[:, None]
    bg = torch.zeros(S.shape[1], dtype=torch.float64, requires_grad=cov is not None)
    off = (lambda: c * bg) if cov is not None else (lambda: 0.0)
    opt = torch.optim.LBFGS([a, U, Vv] + ([bg] if cov is not None else []), lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        eta = a + off() + U @ Vv.T
        nll = -(s * torch.nn.functional.logsigmoid(eta) + (n - s) * torch.nn.functional.logsigmoid(-eta)).sum()
        loss = nll + ridge * 0.5 * (U.pow(2).sum() + Vv.pow(2).sum())
        loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        eta = a + off() + U @ Vv.T
        dev = 2 * (torch.where(s > 0, s * torch.log(s / (n * torch.sigmoid(eta)).clamp_min(1e-12)), torch.zeros_like(s))
                   + torch.where(n - s > 0, (n - s) * torch.log((n - s) / (n * torch.sigmoid(-eta)).clamp_min(1e-12)), torch.zeros_like(s))).sum()
        dev0_eta = (a + off()).expand_as(eta)
        dev0 = 2 * (torch.where(s > 0, s * torch.log(s / (n * torch.sigmoid(dev0_eta)).clamp_min(1e-12)), torch.zeros_like(s))
                    + torch.where(n - s > 0, (n - s) * torch.log((n - s) / (n * torch.sigmoid(-dev0_eta)).clamp_min(1e-12)), torch.zeros_like(s))).sum()
    W = (U @ Vv.T).detach().numpy()
    Us, sv, _ = np.linalg.svd(W - W.mean(0), full_matrices=False)
    F = Us[:, :L] * sv[:L]
    return F, float(1 - dev / dev0), sv[:L] ** 2 / (sv ** 2).sum()


def diagnostics(Y, xy, dep, ann):
    nn = cKDTree(xy).query(xy, k=7)[1][:, 1:]; X = np.c_[np.ones(len(xy)), xy]; rows = []
    for k in range(4):
        y = Y[:, k]; b = np.linalg.lstsq(X, y, rcond=None)[0]
        best = max(ann, key=lambda c: abs(spearmanr(y, ann[c])[0]))
        rows.append(dict(factor=k + 1, nbr_r=round(float(np.corrcoef(y, y[nn].mean(1))[0, 1]), 3),
                         plane_R2=round(float(1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 3),
                         depth_spearman=round(float(spearmanr(y, dep)[0]), 3),
                         anatomy=best.replace("ann_frac_", "").replace("mk_", ""), anatomy_r=round(float(spearmanr(y, ann[best])[0]), 3)))
    return pd.DataFrame(rows)


def main():
    a, iso, ig, ok_i = V.load(5)
    Xg, Xi, bxy, codes, G, n, inv, kb = V.binned(a, iso, ok_i, BIN)
    keep = n >= MIN_CELLS
    S, N, names = major_counts(Xi[keep], np.asarray(ig[ok_i], str))
    hv = int(sys.argv[sys.argv.index("--hv") + 1]) if "--hv" in sys.argv else 0
    if hv:
        pg = S.sum(0) / N.sum(0); obs = N > 0
        phi = (((S - N * pg) ** 2) / np.maximum(N * pg * (1 - pg), 1e-12) * obs).sum(0) / np.maximum(obs.sum(0), 1)
        sel = np.argsort(-phi)[:hv]
        print(f"HV selection: top {hv} of {len(names)} genes by overdispersion; phi median kept {np.median(phi[sel]):.2f} vs all {np.median(phi):.2f}", flush=True)
        S, N, names = S[:, sel], N[:, sel], names[sel]
    print(f"{keep.sum()} bins (>= {MIN_CELLS} cells), {len(names)} genes; reads per bin per gene median "
          f"{np.median(N.mean(0)):.2f}; bins with 0 reads of a gene: {np.mean(N == 0):.0%}", flush=True)
    depthcov = "--depthcov" in sys.argv
    dep_k = np.asarray(Xi.sum(1)).ravel()[keep]
    F, dev_expl, ve = binomial_glmpca(S, N, cov=np.log(dep_k) if depthcov else None)
    print(f"binomial GLM-PCA: deviance explained {dev_expl:.3f}; factor variance shares {np.round(ve[:5], 3)}", flush=True)
    xy = bxy[keep]; dep = np.asarray(Xi.sum(1)).ravel()[keep]
    ann_all = V.annotations(a, iso, Xg, G, n, inv, kb)
    ann = {k: v[keep] for k, v in ann_all.items() if k.startswith("ann_frac_") or k.startswith("mk_")}
    T = diagnostics(F, xy, dep, ann); print(T.to_string(index=False), flush=True)
    T.to_csv(f"{OUT}/binomial_glmpca_diagnostics{'_depthcov' if depthcov else ''}{f'_hv{hv}' if hv else ''}.csv", index=False)
    Yr, ir = rna_glmpc_real(Xg[keep], K=10)
    ex = dict(Y_RNA=Yr, Y_Isoform=scale_target(F, "global").astype(np.float32), depth_RNA=ir["depth"], depth_Isoform=dep)
    ex.update({k: (v[keep] if hasattr(v, "__len__") and len(v) == len(codes) else v) for k, v in ann_all.items()})
    save("rtt_merfish_s5_iso100_binom" + ("_depthcov" if depthcov else "") + (f"_hv{hv}" if hv else ""), xy, [str(u) for u in codes[keep]], ["RNA", "Isoform"], ex)


if __name__ == "__main__":
    main()
