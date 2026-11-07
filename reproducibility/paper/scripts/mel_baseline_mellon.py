"""Baseline: can a standard embedding + Mellon find the gradient transition zones?

Melanoma RNA+ATAC (Fig 3 spots, n=1,997). Standard, Saturn-free pipelines:
  ATAC  binarise peaks -> TF-IDF -> truncated SVD (LSI, 30 comps) -> drop comp 1 (depth) -> 2..30
  RNA   library-normalise -> log1p -> 2,000 HVG (seurat_v3 on counts) -> scale -> PCA 30
Mellon log-density on each; 'transition' = lowest-density TOP fraction of spots (5/10/20%),
the usual low-density reading of Mellon.

Reference = the gradient transition spots of mel_gradient_transition.py (gradual zones along
grad rho), per modality. Per baseline and threshold:
  recall      fraction of reference transition spots flagged low-density
  precision   fraction of low-density spots that are reference transition spots
  zone hit    reference zones with >= 30% of their spots flagged
  null        the low-density set rotated about the tissue centre 200x (re-read by nearest spot)
              -> chance recall; p = fraction of rotations reaching the observed recall
  coherence   fraction of low-density spots with >= 2 low-density among their 6 neighbours
  depth       Spearman(log-density, depth); median depth low-density vs rest

Outputs results/real_saturn/melanoma_tissue_gradient/baseline_{summary.csv,spots.csv,mellon.png}
Usage: python scripts/mel_baseline_mellon.py
"""
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD, PCA
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/fig3")

H = "/mnt/HDD2B/Users/yancui"
G = f"{H}/Hole/gaston_mix_mm/results/real_saturn/melanoma_tissue_gradient"
TOPS, NROT, NC, SEED, ZHIT = (0.05, 0.1, 0.2), 200, 30, 0, 0.3


def lsi(X):
    B = (sp.csr_matrix(X) > 0).astype(np.float32)
    tf = sp.diags(1 / np.maximum(np.asarray(B.sum(1)).ravel(), 1)) @ B
    idf = np.log1p(B.shape[0] / np.maximum(np.asarray(B.sum(0)).ravel(), 1))
    E = TruncatedSVD(NC, random_state=SEED).fit_transform(tf @ sp.diags(idf))
    E = E[:, 1:]; return (E - E.mean(0)) / E.std(0)


def pca_rna(X, names):
    import anndata as ad, scanpy as sc
    a = ad.AnnData(sp.csr_matrix(X).astype(np.float32)); a.var_names = names; a.var_names_make_unique()
    sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=2000)
    a = a[:, a.var.highly_variable].copy()
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a); sc.pp.scale(a, max_value=10)
    return PCA(NC, random_state=SEED).fit_transform(a.X)


def rotate_idx(xy, rng):
    c = xy.mean(0); th = rng.uniform(0, 2 * np.pi)
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return cKDTree((xy - c) @ Rm.T + c).query(xy, k=1)[1]


def run():
    import mellon
    import fig3_melanoma as F3
    rng = np.random.default_rng(SEED)
    D = F3.load(); xy = np.asarray(D["xy"], float); z = D["z"]
    ref = pd.read_csv(f"{G}/spots.csv", index_col=0)
    assert list(ref.index.astype(str)) == D["bc"]
    nn6 = cKDTree(xy).query(xy, k=7)[1][:, 1:]
    rots = [rotate_idx(xy, rng) for _ in range(NROT)]
    emb = {"ATAC": ("LSI", lsi(D["X"]["ATAC"])), "RNA": ("PCA", pca_rna(D["X"]["RNA"], D["names"]["RNA"]))}
    out = pd.DataFrame(index=D["bc"]); rows = []
    for m, (meth, E) in emb.items():
        dens = np.asarray(mellon.DensityEstimator().fit_predict(E), float)
        out[f"logdens_{meth}"] = dens
        dep = np.asarray(z[f"depth_{m}"], float)
        T = ref[f"transition_{m}"].values.astype(bool); zone = ref[f"zone_{m}"].values
        gz = sorted(set(zone[T]))
        rho_dep = float(spearmanr(dens, dep)[0])
        for top in TOPS:
            low = dens <= np.quantile(dens, top)
            out[f"low_{meth}_{int(top*100)}"] = low
            rec = float((low & T).sum() / T.sum()); prec = float((low & T).sum() / low.sum())
            nl = np.array([(low[r] & T).sum() / T.sum() for r in rots])
            hits = [q for q in gz if low[zone == q].mean() >= ZHIT]
            coh = float((low[nn6][low].sum(1) >= 2).mean())
            coh_null = float(np.median([(low[r][nn6][low[r]].sum(1) >= 2).mean() for r in rots[:50]]))
            rows.append(dict(modality=m, baseline=f"{meth} + Mellon", low_fraction=top, n_low=int(low.sum()),
                             n_reference=int(T.sum()), recall=round(rec, 3), chance_recall=round(float(np.median(nl)), 3),
                             p_rot=round(float(((nl >= rec).sum() + 1) / (NROT + 1)), 4), precision=round(prec, 3),
                             zones_hit=f"{len(hits)}/{len(gz)}", coherence=round(coh, 3), coherence_rot=round(coh_null, 3),
                             depth_low=int(np.median(dep[low])), depth_rest=int(np.median(dep[~low])),
                             spearman_density_depth=round(rho_dep, 3)))
            r = rows[-1]
            print(f"{m} {meth}+Mellon low {top:.0%}: recall {r['recall']:.2f} (chance {r['chance_recall']:.2f}, p {r['p_rot']}) "
                  f"precision {r['precision']:.2f} zones {r['zones_hit']} | coherence {r['coherence']:.2f} "
                  f"(rot {r['coherence_rot']:.2f}) | depth low/rest {r['depth_low']}/{r['depth_rest']}, rho(dens,depth) {rho_dep:+.2f}",
                  flush=True)
    pd.DataFrame(rows).to_csv(f"{G}/baseline_summary.csv", index=False); out.to_csv(f"{G}/baseline_spots.csv")
    fig, A = plt.subplots(2, 3, figsize=(15, 9.5))
    for i, (m, (meth, _)) in enumerate(emb.items()):
        T = ref[f"transition_{m}"].values.astype(bool); low = out[f"low_{meth}_10"].values
        a = A[i, 0]; a.scatter(xy[:, 0], xy[:, 1], c="#e6e6e6", s=4, lw=0); a.scatter(xy[T, 0], xy[T, 1], c="#c4453c", s=6, lw=0)
        a.set_title(f"gradient transition zones (rho_{m})")
        a = A[i, 1]; v = out[f"logdens_{meth}"]
        sc_ = a.scatter(xy[:, 0], xy[:, 1], c=v, s=4, lw=0, cmap="coolwarm", vmin=np.quantile(v, .02), vmax=np.quantile(v, .98))
        plt.colorbar(sc_, ax=a, shrink=0.6); a.set_title(f"{meth} + Mellon log-density")
        a = A[i, 2]; a.scatter(xy[:, 0], xy[:, 1], c="#e6e6e6", s=4, lw=0)
        a.scatter(xy[low & ~T, 0], xy[low & ~T, 1], c="#2e5e8c", s=6, lw=0, label="low density only")
        a.scatter(xy[low & T, 0], xy[low & T, 1], c="#e6b85c", s=8, lw=0, label="both")
        a.scatter(xy[~low & T, 0], xy[~low & T, 1], c="#c4453c", s=4, lw=0, alpha=0.5, label="gradient only")
        a.legend(loc="lower left", fontsize=8); a.set_title(f"{meth} lowest 10% density vs gradient zones")
        for a in A[i]: a.set_aspect("equal"); a.axis("off")
    plt.tight_layout(); plt.savefig(f"{G}/baseline_mellon.png", dpi=110); print(f"-> {G}")


if __name__ == "__main__":
    run()
