"""Stronger isoform targets for the RT&T-MERFISH section 5 Saturn fit (rtt_merfish_s5).

The original fit's rho_Isoform is 77% a plane in (x, y) while its target is not planar (PC1 plane
R2 0.25) and it tracks rho_RNA at r 0.85: a weak isoform signal per 50 um bin lets the isoform head
fall back to the smoothest field coupled to RNA. Two ways to raise the signal, everything else
identical to prep_real2.rtt_merfish (same cells, RNA GLM-PCA K=10, same annotations):

  rtt_merfish_s5_b100    100 um bins (4x the area, ~4x reads per feature); isoform target as before
                         (isoform_usage_pca: shrunken within-gene usage, CLR, log-depth regressed, PCA 10)
  rtt_merfish_s5_major   50 um bins; ONE feature per multi-isoform gene: usage of its dominant isoform
                         (most reads section-wide), empirical-Bayes shrunk to the gene's global usage
                         (alpha 5), logit, log-depth regressed, top-variance genes -> PCA 10.
                         All minor isoforms of a gene are pooled into one denominator.

  rtt_merfish_s5_major100_drop1
                         100 um bins + major-isoform usage as above, PCA with K+1 components and
                         component 1 DROPPED (as LSI comp 1 is dropped for ATAC): PC1 of this target is
                         a section-wide planar trend (diagnostics printed and saved before dropping).

Usage: python scripts/rtt_isoform_variants.py [b100] [major] [major100_drop1]
"""
import sys
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad
from sklearn.decomposition import PCA
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/saturn_repo/src")
from prep_real2 import H, save, lognorm, rna_glmpc_real
from saturn.preprocess import isoform_usage_pca, scale_target

D = f"{H}/external_data/rtt_merfish"
ALPHA, MIN_GENE_READS, N_TOP, K = 5.0, 100, 3000, 10


def load(sample=5):
    a = ad.read_h5ad(f"{D}/cellxgene_ISO_labels.h5ad")
    a = a[a.obs["sample"].astype(str) == str(sample)].copy()
    iso = np.array([v.startswith("ENSMUST") for v in a.var_names])
    t2g = pd.read_csv(f"{D}/ensembl_tx2gene_mm.tsv", sep="\t").drop_duplicates("tx").set_index("tx")["gene"]
    ig = t2g.reindex(a.var_names[iso]).values; ok_i = pd.notna(ig)
    return a, iso, ig, ok_i


def binned(a, iso, ok_i, bin_um):
    xy = a.obs[["global_x", "global_y"]].values.astype(float)
    key = np.floor(xy / bin_um).astype(np.int64); code = key[:, 0] * 1_000_000 + key[:, 1]
    uniq, inv, cnt = np.unique(code, return_inverse=True, return_counts=True)
    kb = np.where(cnt >= 3)[0]
    G = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(uniq), len(inv)))[kb]
    n = np.asarray(G.sum(1)).ravel(); X = sp.csr_matrix(a.X)
    Xg = G @ X[:, np.where(~iso)[0]]; Xi = G @ X[:, np.where(iso)[0][ok_i]]
    return Xg, Xi, np.asarray(G @ xy) / n[:, None], uniq[kb], G, n, inv, kb


def annotations(a, iso, Xg, G, n, inv, kb):
    ex = {"ann_n_cells": n.astype(np.float32)}
    reg = pd.DataFrame({"b": inv, "r": a.obs["annot"].astype(str).values})
    maj = reg.groupby("b")["r"].agg(lambda v: v.value_counts().index[0])
    ex["ann_ccf_region"] = maj.reindex(kb).fillna("NA").astype(str).values
    ct = a.obs["cell_label"].astype(str).values
    for t in pd.Series(ct).value_counts().index[:8]:
        ex[f"ann_frac_{t}"] = (np.asarray(G @ (ct == t).astype(float)).ravel() / n).astype(np.float32)
    ex.update(lognorm(Xg, ["Snap25", "Mbp", "Plp1", "Slc17a7", "Gad1", "Gfap", "Aqp4", "Rorb", "Fezf2"], a.var_names[~iso]))
    return ex


def major_usage_pca(Xi, genes, drop_first=False, diag=None):
    """one logit-usage feature per gene: its dominant isoform's share, shrunk, depth-regressed, PCA"""
    Xi = sp.csc_matrix(Xi).astype(float); tot = np.asarray(Xi.sum(0)).ravel()
    dep = np.asarray(Xi.sum(1)).ravel().clip(1)
    cols, names = [], []
    for g, idx in pd.Series(np.arange(len(genes))).groupby(genes):
        idx = idx.values
        if len(idx) < 2 or tot[idx].sum() < MIN_GENE_READS: continue
        j = idx[np.argmax(tot[idx])]
        nm = np.asarray(Xi[:, j].todense()).ravel(); ng = np.asarray(Xi[:, idx].sum(1)).ravel()
        p = tot[j] / tot[idx].sum()
        u = (nm + ALPHA * p) / (ng + ALPHA)
        cols.append(np.log(u / (1 - u)) - np.log(p / (1 - p))); names.append(g)
    U = np.column_stack(cols)
    L = np.c_[np.ones(len(dep)), np.log(dep)]
    U = U - L @ np.linalg.lstsq(L, U, rcond=None)[0]
    keep = np.argsort(-U.var(0))[:N_TOP]
    P = PCA(K + int(drop_first), random_state=0).fit(U[:, keep])
    Y = P.transform(U[:, keep])
    if diag is not None:
        diag["pcs"] = Y.copy(); diag["evr"] = P.explained_variance_ratio_
    if drop_first:
        Y = Y[:, 1:]
    print(f"   major-isoform usage: {len(names)} genes, {len(keep)} kept; PCA var {P.explained_variance_ratio_[:5].round(3)}", flush=True)
    return scale_target(Y, "global").astype(np.float32), dep


def pc_diagnostics(diag, xy, dep_i, Xg, a, iso, G, n, inv, kb):
    """what each leading PC of the isoform target is, before PC1 is dropped"""
    from scipy.stats import spearmanr
    from scipy.spatial import cKDTree
    Pcs = diag["pcs"]; X1 = np.c_[np.ones(len(xy)), xy]; nn = cKDTree(xy).query(xy, k=7)[1][:, 1:]
    ann = annotations(a, iso, Xg, G, n, inv, kb)
    fr = {k: v for k, v in ann.items() if k.startswith("ann_frac_") or k.startswith("mk_")}
    dep_r = np.asarray(Xg.sum(1)).ravel()
    rows = []
    for j in range(4):
        y = Pcs[:, j]; b = np.linalg.lstsq(X1, y, rcond=None)[0]
        plane = 1 - ((y - X1 @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        best = max(fr, key=lambda k: abs(spearmanr(y, fr[k])[0]))
        rows.append(dict(PC=j + 1, var_ratio=round(float(diag["evr"][j]), 4), plane_R2=round(float(plane), 3),
                         neighbour_r=round(float(np.corrcoef(y, y[nn].mean(1))[0, 1]), 3),
                         depth_iso=round(float(spearmanr(y, dep_i)[0]), 3), depth_rna=round(float(spearmanr(y, dep_r)[0]), 3),
                         best_annotation=best, best_annotation_r=round(float(spearmanr(y, fr[best])[0]), 3)))
    T = pd.DataFrame(rows); print(T.to_string(index=False), flush=True)
    T.to_csv(f"{H}/Hole/gaston_mix_mm/results/real_saturn/rtt_isoform_variants/major100_pc_diagnostics.csv", index=False)


def build(variant):
    a, iso, ig, ok_i = load(5)
    bin_um = 100.0 if variant in ("b100", "major100_drop1") else 50.0
    Xg, Xi, bxy, codes, G, n, inv, kb = binned(a, iso, ok_i, bin_um)
    print(f"{variant}: {len(codes)} bins of {bin_um:.0f} um; isoform molecules/bin median {np.median(np.asarray(Xi.sum(1)).ravel()):.0f}", flush=True)
    Yr, ir = rna_glmpc_real(Xg, K=10)
    if variant == "b100":
        dep_i = np.asarray(Xi.sum(1)).ravel()
        Yi, info = isoform_usage_pca(Xi, ig[ok_i], K=10, depth=dep_i, min_gene_reads=MIN_GENE_READS)
        print(f"   isoform usage: {info['n_genes']} genes / {info['n_isoforms']} isoforms", flush=True)
    elif variant == "major":
        Yi, dep_i = major_usage_pca(Xi, np.asarray(ig[ok_i], str))
    else:
        diag = {}
        Yi, dep_i = major_usage_pca(Xi, np.asarray(ig[ok_i], str), drop_first=True, diag=diag)
        pc_diagnostics(diag, bxy, dep_i, Xg, a, iso, G, n, inv, kb)
    ex = dict(Y_RNA=Yr, Y_Isoform=Yi, depth_RNA=ir["depth"], depth_Isoform=dep_i)
    ex.update(annotations(a, iso, Xg, G, n, inv, kb))
    save(f"rtt_merfish_s5_{variant}", bxy, [str(u) for u in codes], ["RNA", "Isoform"], ex)


if __name__ == "__main__":
    for v in (sys.argv[1:] or ["b100", "major"]):
        build(v)
