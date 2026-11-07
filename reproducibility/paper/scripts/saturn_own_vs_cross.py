"""Own-axis vs cross-axis explanatory power, per feature and per domain.

Saturn fits Y_m,j = base + P_{m<-m}(d_m) + P_{m<-k}(d_k) jointly. This takes the two
terms APART and asks, for every gene and every peak separately:

    R2_own    feature fitted piecewise along its OWN modality's isodepth
    R2_cross  the same feature fitted piecewise along the OTHER modality's isodepth
    R2_joint  both fitted predictions combined linearly

and decomposes the variance the way commonality analysis does:

    unique_own   = R2_joint - R2_cross     only the own axis carries it
    unique_cross = R2_joint - R2_own       only the cross axis carries it
    shared       = R2_own + R2_cross - R2_joint

Each feature is also scored inside every domain of its OWN axis, so a gene can be
own-explained in one domain and cross-explained in another -- which is the question
the modality-level R2 in summary.json cannot answer.

Usage: python scripts/saturn_own_vs_cross.py <dataset> [n_genes] [n_peaks]
"""
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/saturn_repo/src")
from lpc_gaston_fig4l import gaston_domains
from domain_k import kdom
from domain_annot import CFG, R
from saturn.cis import smoothed_logcpm
from saturn.crossterm import own_vs_cross

KDOM, T_LRT = 4, 0.1
# spatial kNN mean: per-spot counts are too sparse for a per-feature R2 to mean anything (median R2 ~0.015).
# Both axes are evaluated on the SAME smoothed values, so the own-vs-cross contrast is unaffected.
KNN = 15


def run(ds, n_genes=1500, n_peaks=2000):
    OUT = f"{R}/{ds}_own_vs_cross"; os.makedirs(OUT, exist_ok=True)
    cfg = CFG[ds]
    ax = pd.read_csv(f"{R}/{ds}/axes.csv", index_col=0); bc = [str(b) for b in ax.index]
    z = dict(np.load(f"{R}/targets_{ds}.npz", allow_pickle=True))
    rna = cfg["rna"](); atac = cfg["atac"]()
    ri = {str(b): i for i, b in enumerate(rna.obs_names)}
    ai = {str(b): i for i, b in enumerate(atac.obs_names)}
    Xr = sp.csr_matrix(rna.X)[[ri[b] for b in bc]]
    Xa = sp.csr_matrix(atac.X)[[ai[b] for b in bc]]
    gn = np.array([str(g) for g in rna.var_names]); pn = np.array([str(p) for p in atac.var_names])
    tg = np.asarray(Xr.sum(0)).ravel().astype(float); tp = np.asarray(Xa.sum(0)).ravel().astype(float)
    kg = np.argsort(-tg)[:n_genes]; kp = np.argsort(-tp)[:n_peaks]
    xy = np.c_[ax.x.values, ax.y.values]
    G = smoothed_logcpm(Xr, kg, xy, KNN); P = smoothed_logcpm(Xa, kp, xy, KNN)
    D = {a: ax[f"d_{a}"].values for a in ("RNA", "ATAC")}
    L = {a: gaston_domains(D[a], z[f"Y_{a}"], L_=kdom(ds, a)) for a in ("RNA", "ATAC")}
    rows = []
    for mod, Y, feats, cnt in (("RNA", G, gn[kg], tg[kg]), ("ATAC", P, pn[kp], tp[kp])):
        cross = "ATAC" if mod == "RNA" else "RNA"
        T = own_vs_cross(Y, D[mod], L[mod], D[cross], L[cross], t=T_LRT)
        T.insert(0, "feature", feats); T.insert(0, "modality", mod); T.insert(0, "dataset", ds)
        T.insert(9, "counts", cnt)
        rows.append(T)
        print(f"   {ds} {mod}: {Y.shape[1]} features | median R2 own {T.R2_own.median():.3f} "
              f"cross {T.R2_cross.median():.3f} joint {T.R2_joint.median():.3f}", flush=True)
    A = pd.concat(rows, ignore_index=True)
    A.to_csv(f"{OUT}/own_vs_cross.csv", index=False)
    print(f"-> {OUT}/own_vs_cross.csv")
    return A


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1500,
        int(sys.argv[3]) if len(sys.argv) > 3 else 2000)
