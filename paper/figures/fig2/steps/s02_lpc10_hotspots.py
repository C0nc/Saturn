"""Step 02 — microglia hotspots on the DBiT LPC10 section itself.

A microglia score is built from RNA genes that mirror the CODEX myeloid channels
(CD11c/CD11b/IBA1 -> Itgax, Itgam, Cd68, Trem2, Aif1, Cx3cr1, P2ry12), pooled over each spot's
13 nearest neighbours as a Poisson z against the section-wide rate. Spots with z > 3 outside
the cortex CellCharter clusters (1, 5, 2) are grouped into connected components; the largest
are the lesion (comp0) and the distal white-matter focus (comp1).

Recovered verbatim (logic unchanged) from the interactive session of 2026-09-20 that produced
results/real_saturn/lpc10_hotspots.npz. Only the paths were made absolute and the diagnostic
PNG dropped.
"""
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from prep_real2 import _read_h5ad_safe, LPC_TP, H

OUT = f"{H}/Hole/gaston_mix_mm/results/real_saturn/lpc10_hotspots.npz"
CORTEX = [1, 5, 2]


def main():
    r = _read_h5ad_safe(f"{H}/{LPC_TP['LPC10']['rna']}")
    xy = np.asarray(r.obsm["spatial"], float); X = sp.csr_matrix(r.X); vn = list(map(str, r.var_names))
    lib = np.asarray(X.sum(1)).ravel().clip(1)

    def score(gs):
        v = np.zeros(X.shape[0])
        for g in gs:
            if g in vn:
                v += np.asarray(X[:, vn.index(g)].todense()).ravel()
        return v

    mg = score(["Itgax", "Itgam", "Cd68", "Trem2", "Aif1", "Cx3cr1", "P2ry12"])
    my = score(["Mbp", "Plp1", "Mog"])
    nb = cKDTree(xy).query(xy, k=13)[1]
    S = mg[nb].sum(1); L = lib[nb].sum(1); rate = mg.sum() / lib.sum()
    zmg = (S - rate * L) / np.sqrt(np.maximum(rate * L, 1e-9))
    myz = np.log1p(my / lib * 1e4); myz = (myz - myz.mean()) / myz.std()
    cc = pd.read_csv(f"{H}/{LPC_TP['LPC10']['cc_csv']}", index_col=0)["cc_K10"].astype(int).values
    cc_xy = np.load(f"{H}/{LPC_TP['LPC10']['cc_coords']}"); lab = cc[cKDTree(cc_xy).query(xy, k=1)[1]]
    pos = (zmg > 3) & (~np.isin(lab, CORTEX))
    c = xy[pos]; knn = cKDTree(c).query(c, k=7)[1]; kk = knn.shape[1] - 1
    adj = sp.csr_matrix((np.ones(len(c) * kk, np.int8), (np.repeat(np.arange(len(c)), kk), knn[:, 1:].ravel())),
                        shape=(len(c), len(c)))
    n, cl = connected_components(adj + adj.T, directed=False)
    order = np.argsort(-np.bincount(cl))[:5]
    np.savez(OUT, **{f"comp{i}": np.where(pos)[0][cl == k] for i, k in enumerate(order)},
             zmg=zmg, myz=myz, lab=lab, xy=xy, barcodes=np.asarray(r.obs_names, dtype=object))
    print(f"-> {OUT}  ({pos.sum()} microglia-positive non-cortex spots)")


if __name__ == "__main__":
    main()
