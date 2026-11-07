"""Melanoma spatial Hi-C + RNA Saturn target.

Recipe:

  RNA   GLM-PCA on the FULL 10x matrix (SpHIC117/RNA/raw), HVGs re-selected from
        62,700 genes.  annotated_adata.h5ad carries a pre-made 2,000-HVG matrix
        holding only 6% of the library (89 of 1,402 counts/spot) and missing
        MITF/MLANA/PMEL/AXL/NGFR -- unusable for a melanoma phenotype axis.
  HiC   Fast-Higashi X_fh (256-d) -> top 10 PCs -> global scaling

The A/B compartment matrix is deliberately NOT a fitting target here -- it stays a
POST-HOC layer, read the way granger_ab_gene.ipynb reads it:

    ab = pd.read_csv('SpHIC117/abcompartment_matrix.csv', index_col=0)
    ab = ab.reindex(adata.obs_names).dropna()          # adata order is the root
    bins from SpHIC117/HIC/higashi/tmp/scCompartment.hdf5['compartment_zscore']['bin']

so downstream scripts align bins -> genes on the same spot index as the fit.

Usage: python scripts/mel_hic_prep.py [--drop_rows 50 51]

--drop_rows Y ...  drop whole chip rows before any reduction (RNA GLM-PCA and the X_fh PCA
                   are recomputed on the kept spots) -> dataset mel_hic_fh_dr<rows>.
                   Rows 50-51 are the DBiT channel dip (Hi-C reads ~-0.8 SD vs +0.9 at
                   row 49); everything else is the mel_hic_fh recipe unchanged.
"""
import sys, argparse
import numpy as np, pandas as pd, scipy.sparse as sp, h5py
from anndata.experimental import read_elem
from scipy.stats import spearmanr
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/src")
from prep_real2 import lognorm, save, H
from saturn.preprocess import rna_glmpc_real, scale_target

AD = f"{H}/HIC_repo/Melanoma/annotated_adata.h5ad"
RAW = f"{H}/SpHIC117/RNA/raw"                    # 62,700 genes x 10,000 barcodes
AB_CSV = f"{H}/SpHIC117/abcompartment_matrix.csv"
AB_H5 = f"{H}/SpHIC117/HIC/higashi/tmp/scCompartment.hdf5"
MARKERS = ["MITF", "MLANA", "PMEL", "TYR", "DCT", "SOX10", "PRAME",          # melanocytic
           "AXL", "NGFR", "SOX9", "ZEB1", "VIM", "FN1", "SERPINE1",          # invasive / NC
           "MKI67", "COL1A1", "PTPRC", "CD3E", "CD68", "LYZ", "PECAM1",      # TME
           "B2M", "HLA-A", "STAT1"]                                          # IFN
ANN = ["cluster_cellcharter_rna_best", "cluster_cellcharter_hic_best",
       "cluster_cellcharter_concat_best", "cluster_cellcharter_rna_12",
       "cluster_cellcharter_hic_12", "max_prop_cell_type"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--drop_rows", type=float, nargs="*", default=[])
    drop = ap.parse_args().drop_rows
    with h5py.File(AD, "r") as f:
        X = sp.csr_matrix(read_elem(f["layers"]["counts"]))
        obs = read_elem(f["obs"]); var = read_elem(f["var"])
        coords = np.asarray(read_elem(f["obsm"]["spatial"]), float)
        FH = np.asarray(read_elem(f["obsm"]["X_fh"]), float)
        CP = np.asarray(read_elem(f["obsm"]["cell_prorportion"]), float)
        cts = [s.decode() if isinstance(s, bytes) else str(s) for s in f["uns"]["cell_types"][:]]
    assert (obs["in_tissue"].values.astype(int) == 1).all(), "unexpected out-of-tissue spots"
    if drop:
        keep = ~np.isin(coords[:, 1], drop)
        print(f"drop rows {drop}: {np.sum(~keep)} spots; keep {keep.sum()}", flush=True)
        X, obs, coords, FH, CP = X[keep], obs[keep], coords[keep], FH[keep], CP[keep]

    # ---- RNA from the FULL 10x matrix, reordered onto the adata spot index ---------
    import gzip
    from scipy.io import mmread
    bcs = [l.strip() for l in gzip.open(f"{RAW}/barcodes.tsv.gz", "rt")]
    feat = pd.read_csv(f"{RAW}/features.tsv.gz", sep="\t", header=None)
    genes = pd.Index([str(v) for v in feat[1]])
    Xf = sp.csr_matrix(mmread(f"{RAW}/matrix.mtx.gz").T)             # spots x genes
    pos = pd.Index(bcs).get_indexer(pd.Index(obs.index))
    assert (pos >= 0).all(), f"{(pos < 0).sum()} adata spots absent from the raw matrix"
    Xf = Xf[pos]
    hvg_depth = np.asarray(X.sum(1)).ravel(); full_depth = np.asarray(Xf.sum(1)).ravel()
    print(f"melanoma: {Xf.shape[0]} spots x {Xf.shape[1]} genes (full 10x matrix); "
          f"counts/spot median {np.median(full_depth):.0f} vs {np.median(hvg_depth):.0f} in the "
          f"h5ad's 2,000-HVG matrix ({np.median(hvg_depth / np.maximum(full_depth, 1)):.0%} of the "
          f"library); markers now present: "
          f"{sum(m in set(genes) for m in MARKERS)}/{len(MARKERS)}", flush=True)

    Yr, ir = rna_glmpc_real(Xf, K=10, n_hvg=4000)
    Z = FH - FH.mean(0)
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    P = (U * S)[:, :10]
    print(f"   Fast-Higashi X_fh {FH.shape} -> 10 PCs "
          f"({100 * (S[:10] ** 2).sum() / (S ** 2).sum():.1f}% of its variance); "
          f"|Spearman| PC1-5 vs RNA depth "
          f"{[round(abs(spearmanr(P[:, k], ir['depth'])[0]), 2) for k in range(5)]}", flush=True)

    ex = dict(Y_RNA=Yr, Y_HiC=scale_target(P, "global").astype(np.float32),
              depth_RNA=ir["depth"], depth_HiC=obs["n_counts"].values.astype(np.float32))
    for c in ANN:
        if c in obs:
            ex[f"ann_{c}"] = obs[c].astype(str).values
    ex["ann_confidence"] = obs["confidence"].values.astype(np.float32)
    for j, t in enumerate(cts):                      # cell2location-style deconvolution
        ex["mk_ct_" + t.replace(" ", "_").replace("-", "_")] = CP[:, j].astype(np.float32)
    ex.update(lognorm(Xf, MARKERS, genes))
    save("mel_hic_fh" + ("_dr" + "_".join(f"{v:g}" for v in drop) if drop else ""), coords, obs.index, ["RNA", "HiC"], ex)

    # ---- post-hoc A/B: verify the notebook's alignment holds on this spot set -------
    ab = pd.read_csv(AB_CSV, index_col=0).reindex(pd.Index(obs.index)).dropna()
    with h5py.File(AB_H5, "r") as f:
        g = f["compartment_zscore"]["bin"]
        nb = len(np.array(g["chrom"]))
    print(f"   post-hoc A/B: {ab.shape[0]}/{len(obs)} spots aligned, {ab.shape[1]} bin columns "
          f"(scCompartment bins {nb}); non-bin columns "
          f"{[c for c in ab.columns if not str(c).startswith('bin_')]}", flush=True)


if __name__ == "__main__":
    main()
