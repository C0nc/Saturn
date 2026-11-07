"""iStar on MEASURED melanoma features (not rho), reusing the registered H&E of the rho run.

rho is smooth by construction, so iStar cannot add sub-spot information to it (held-out: iStar
0.996 vs linear interpolation 0.999). Measured features vary within and between neighbouring
spots, so here a held-out test CAN show whether the H&E adds information.

Features
  RNA  log1p CP10k of domain-program markers (melanocytic, epidermis, IFN, stroma, stress, immune)
  ATAC per-spot chromVAR deviation z (GC/accessibility-matched), RAW (the kNN-smoothed version
       would leak neighbour information into the held-out test)

Writes melanoma_tissue_istar_feat/ and melanoma_tissue_istar_feat_holdout10/ (same held-out spots
as the rho test). H&E, histology embeddings and spot geometry are linked/copied from the rho run.

Usage: python scripts/melanoma_istar_features.py
"""
import os, shutil
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

R = "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/results"
RS = f"{R}/real_saturn"
PROGRAMS = {
    "melanocytic": ["TYR", "PMEL", "MITF", "MLANA", "SOX10", "PRAME", "DCT"],
    "epidermis":   ["KRT10", "SPINK5", "FLG", "DSP", "S100A9"],
    "interferon":  ["STAT1", "XAF1", "MX1"],
    "stroma":      ["COL1A1", "COL1A2", "COL3A1"],
    "stress":      ["HSPH1", "HSPB1", "JUN"],
    "immune":      ["CD74", "PTPRC"],
}


def features(bc):
    r = ad.read_h5ad(f"{R}/melanoma/melanoma_rna.h5ad")[pd.Index(bc)]
    X = sp.csr_matrix(r.X).astype(float); lib = np.asarray(X.sum(1)).ravel().clip(1)
    vn = list(map(str, r.var_names)); cols = {}
    for prog, gs in PROGRAMS.items():
        for g in gs:
            cols[f"RNA_{g}"] = np.log1p(np.asarray(X[:, vn.index(g)].todense()).ravel() / lib * 1e4)
    z = np.load(f"{RS}/melanoma_tissue_tf_activity/tf_z_per_spot.npz", allow_pickle=True)
    order = pd.Index([str(b) for b in z["barcodes"]]).get_indexer(pd.Index(bc))
    assert (order >= 0).all()
    Z = np.asarray(z["z"], float)[order]
    for k, t in enumerate(map(str, z["tfs"])):
        cols[f"TF_{t}"] = Z[:, k]
    return pd.DataFrame(cols, index=pd.Index(bc, name="Spot ID"))


def build(src, dst):
    os.makedirs(dst, exist_ok=True)
    for f in ("he.jpg", "he-scaled.jpg", "embeddings-hist.pickle"):
        tgt = os.path.realpath(src + f)
        if os.path.lexists(dst + f): os.remove(dst + f)
        os.symlink(tgt, dst + f)
    for f in ("locs-raw.tsv", "locs.tsv", "locs-all.tsv", "radius-raw.txt", "radius.txt", "pixel-size-raw.txt",
              "pixel-size.txt", "mask-window.png", "geometry.json", "heldout_spots.txt"):
        if os.path.exists(src + f): shutil.copy(src + f, dst + f)
    allL = pd.read_csv(src + "locs-all.tsv", sep="\t", index_col=0)
    F = features(list(allL.index))
    train = pd.read_csv(src + "locs.tsv", sep="\t", index_col=0).index
    F.loc[train].to_csv(dst + "cnts_ab.tsv", sep="\t")
    F.to_csv(dst + "cnts-all.tsv", sep="\t")
    open(dst + "gene-names.txt", "w").write("\n".join(F.columns) + "\n")
    open(dst + "run_impute.sh", "w").write(f"""#!/usr/bin/env bash
set -eo pipefail
source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate spatialmultivelo
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd /mnt/HDD2B/Users/yancui/istar
python impute.py {dst} --epochs=400 --device=cuda --shape=square
echo DONE
""")
    print(f"-> {dst}: {F.shape[1]} features, {len(train)} training spots")


if __name__ == "__main__":
    build(f"{RS}/melanoma_tissue_istar/", f"{RS}/melanoma_tissue_istar_feat/")
    build(f"{RS}/melanoma_tissue_istar_holdout10/", f"{RS}/melanoma_tissue_istar_feat_holdout10/")
