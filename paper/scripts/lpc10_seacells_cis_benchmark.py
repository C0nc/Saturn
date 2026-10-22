"""Benchmark: cis gene-peak (CRE) coupling per domain -- Saturn fitted functions vs SEACells metacells.

LPC10 lesion FOV (Fig 2 fit `lpc10_fov`, 3,425 spots, RNA + ATAC peaks).

Saturn (Fig 2h; saturn_cis_function_pairs.py): for each axis, every (gene, peak, domain) where gene
AND peak are both sloped (LRT-gated piecewise fits along rho) in that domain; concordant = same slope
sign. Benchmark statistic = concordance of near pairs (|TSS distance| < 25 kb) minus concordance of
the distal control (> 1 Mb, same chromosome) -- a cis signal must beat a random peak.

SEACells (Persad et al. 2023), the standard metacell route to gene-peak correlation:
  metacells   SEACells on  (a) ATAC LSI (TF-IDF -> SVD 30, drop comp 1)
                           (b) RNA PCA  (2,000 HVG log-norm -> PCA 30)
              N_MC = n_spots / SPOTS_PER_MC metacells; counts summed per metacell -> log CPM
  domains     each metacell takes the majority Saturn domain of its spots on the axis being scored
              (purity reported)
  scoring     the SAME (gene, peak, domain) triples as Saturn, per axis. Pearson r of gene vs peak
              across the metacells of that domain. Concordant = r > 0. Two filters:
                all     every triple
                sig     |r| with p < 0.05 (the counterpart of Saturn's 'both sloped' gate)
  whole       r across ALL metacells, ignoring domains (the usual SEACells use), on the unique pairs
For every method: near-minus-distal concordance with a chi-square p, overall and per domain.

Outputs results/real_saturn/lpc10_fov_seacells/{summary.csv, per_domain.csv, metacells_*.csv}
Usage: python scripts/lpc10_seacells_cis_benchmark.py
"""
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, scanpy as sc
from scipy.stats import chi2_contingency, t as tdist
from sklearn.decomposition import TruncatedSVD

H = "/mnt/HDD2B/Users/yancui"
R = f"{H}/Hole/gaston_mix_mm/results/real_saturn"
OUT = f"{R}/lpc10_fov_seacells"
H5MU = f"{R}/lpc10_fov_mudata/LPC10_lesion_FOV.gz.h5mu"
PAIRS = f"{R}/lpc10_fov_cis_pairs/cis_function_pairs.csv"
SPOTS_PER_MC, NC, NEAR_KB, SEED = 20, 30, 25, 0


def lsi(X):
    B = (sp.csr_matrix(X) > 0).astype(np.float32)
    tf = sp.diags(1 / np.maximum(np.asarray(B.sum(1)).ravel(), 1)) @ B
    idf = np.log1p(B.shape[0] / np.maximum(np.asarray(B.sum(0)).ravel(), 1))
    E = TruncatedSVD(NC, random_state=SEED).fit_transform(tf @ sp.diags(idf))
    return E[:, 1:]


def pca(X, names):
    a = ad.AnnData(sp.csr_matrix(X).astype(np.float32)); a.var_names = names; a.var_names_make_unique()
    sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=2000)
    a = a[:, a.var.highly_variable].copy(); sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sc.pp.scale(a, max_value=10); sc.tl.pca(a, n_comps=NC, random_state=SEED)
    return a.obsm["X_pca"]


def seacells(E, n_mc):
    import SEACells
    a = ad.AnnData(np.zeros((len(E), 1), np.float32)); a.obsm["X_emb"] = np.asarray(E, np.float32)
    m = SEACells.core.SEACells(a, build_kernel_on="X_emb", n_SEACells=n_mc, n_waypoint_eigs=10,
                               convergence_epsilon=1e-5, verbose=False)
    m.construct_kernel_matrix(); m.initialize_archetypes(); m.fit(min_iter=10, max_iter=100)
    return pd.factorize(a.obs["SEACell"].astype(str))[0]


def aggregate(X, mc):
    G = sp.csr_matrix((np.ones(len(mc)), (mc, np.arange(len(mc)))), shape=(mc.max() + 1, len(mc)))
    A = G @ sp.csr_matrix(X)
    lib = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
    return np.log1p(np.asarray((sp.diags(1e6 / lib) @ A).todense()))


def corr_cols(Gm, Pm):
    """Pearson r between matching columns of Gm and Pm (rows = metacells)"""
    g = Gm - Gm.mean(0); p = Pm - Pm.mean(0)
    den = np.sqrt((g ** 2).sum(0) * (p ** 2).sum(0))
    return np.where(den > 0, (g * p).sum(0) / np.where(den > 0, den, 1), np.nan)


def pval(r, n):
    r = np.clip(r, -0.999999, 0.999999); tt = r * np.sqrt((n - 2) / (1 - r ** 2))
    return 2 * tdist.sf(np.abs(tt), n - 2)


def near_vs_distal(df, col="concordant"):
    near = df[(df.kind == "cis") & (df.kb.abs() < NEAR_KB)][col].dropna().astype(bool)
    dist = df[df.kind == "distal"][col].dropna().astype(bool)
    if len(near) < 5 or len(dist) < 5:
        return dict(near=np.nan, distal=np.nan, gap=np.nan, n_near=len(near), n_distal=len(dist), p=np.nan)
    tab = [[near.sum(), len(near) - near.sum()], [dist.sum(), len(dist) - dist.sum()]]
    return dict(near=round(near.mean(), 3), distal=round(dist.mean(), 3), gap=round(near.mean() - dist.mean(), 3),
                n_near=int(len(near)), n_distal=int(len(dist)), p=float(chi2_contingency(tab)[1]))


def run():
    import mudata as md
    os.makedirs(OUT, exist_ok=True)
    M = md.read_h5mu(H5MU); rna, atac = M.mod["rna"], M.mod["atac"]
    gname = np.array([str(v) for v in rna.var_names]); pname = np.array([str(v) for v in atac.var_names])
    lab = {}
    for m in ("RNA", "ATAC"):
        inv = {v: int(k) for k, v in M.uns["domain_labels"][m].items()}
        lab[m] = np.array([inv[v] for v in M.obs[f"domain_{m}"].astype(str)]);
    P = pd.read_csv(PAIRS)
    P = P[P.gene.isin(set(gname)) & P.peak.isin(set(pname))].copy()
    gi = {g: i for i, g in enumerate(gname)}; pi = {p: i for i, p in enumerate(pname)}
    P["gi"] = P.gene.map(gi); P["pi"] = P.peak.map(pi)
    n_mc = int(round(len(M.obs) / SPOTS_PER_MC))
    print(f"FOV {M.n_obs} spots -> {n_mc} metacells; Saturn triples {len(P)}", flush=True)
    emb = {"SEACells-LSI": lsi(atac.X), "SEACells-PCA": pca(rna.X, gname)}
    rows, per_dom = [], []
    # Saturn reference
    for m in ("RNA", "ATAC"):
        s = P[P.axis == m]
        rows.append(dict(method="Saturn fitted functions", axis=m, filter="both sloped", **near_vs_distal(s)))
        for k, g in s.groupby("domain"):
            per_dom.append(dict(method="Saturn fitted functions", axis=m, domain=M.uns["domain_labels"][m][str(k)], **near_vs_distal(g)))
    for meth, E in emb.items():
        mc = seacells(E, n_mc)
        pd.DataFrame({"metacell": mc}, index=M.obs_names).to_csv(f"{OUT}/metacells_{meth}.csv")
        Gm, Pm = aggregate(rna.X, mc), aggregate(atac.X, mc)
        sizes = np.bincount(mc)
        print(f"{meth}: {mc.max() + 1} metacells, size median {np.median(sizes):.0f}", flush=True)
        # whole-tissue correlation (standard SEACells use), unique pairs
        U = P.drop_duplicates(["gene", "peak"])[["gene", "peak", "kb", "kind", "gi", "pi"]].copy()
        r = corr_cols(Gm[:, U.gi.values], Pm[:, U.pi.values]); U["r"] = r
        U["concordant"] = np.where(np.isfinite(r), r > 0, np.nan)
        rows.append(dict(method=meth, axis="(whole tissue)", filter="all", **near_vs_distal(U)))
        U2 = U[pval(U.r.fillna(0).values, mc.max() + 1) < 0.05]
        rows.append(dict(method=meth, axis="(whole tissue)", filter="p<0.05", **near_vs_distal(U2)))
        for m in ("RNA", "ATAC"):
            # majority Saturn domain per metacell on this axis
            dom_mc = np.array([np.bincount(lab[m][mc == q], minlength=lab[m].max() + 1).argmax() for q in range(mc.max() + 1)])
            purity = np.mean([np.mean(lab[m][mc == q] == dom_mc[q]) for q in range(mc.max() + 1)])
            s = P[P.axis == m].copy(); s["r"] = np.nan; s["n_mc"] = 0
            for k in range(lab[m].max() + 1):
                idx = np.where(dom_mc == k)[0]; sel = (s.domain == k).values
                if len(idx) < 4 or not sel.any(): continue
                s.loc[sel, "r"] = corr_cols(Gm[np.ix_(idx, s.gi.values[sel])], Pm[np.ix_(idx, s.pi.values[sel])])
                s.loc[sel, "n_mc"] = len(idx)
            s["concordant"] = np.where(np.isfinite(s.r), s.r > 0, np.nan)
            sig = s[(s.n_mc >= 4) & (pval(s.r.fillna(0).values, np.maximum(s.n_mc.values, 3)) < 0.05)]
            for filt, df in (("all", s), ("p<0.05", sig)):
                rows.append(dict(method=meth, axis=m, filter=filt, metacell_domain_purity=round(purity, 3), **near_vs_distal(df)))
            for k, g in sig.groupby("domain"):
                per_dom.append(dict(method=meth + " (p<0.05)", axis=m, domain=M.uns["domain_labels"][m][str(k)],
                                    n_metacells=int(g.n_mc.iloc[0]), **near_vs_distal(g)))
            print(f"   axis {m}: metacell domain purity {purity:.2f}; metacells per domain "
                  f"{np.bincount(dom_mc, minlength=lab[m].max() + 1).tolist()}", flush=True)
    S = pd.DataFrame(rows); D = pd.DataFrame(per_dom)
    S.to_csv(f"{OUT}/summary.csv", index=False); D.to_csv(f"{OUT}/per_domain.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n", S.to_string(index=False)); print("\n", D.to_string(index=False)); print(f"-> {OUT}")


if __name__ == "__main__":
    run()
