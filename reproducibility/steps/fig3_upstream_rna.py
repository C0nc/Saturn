"""RNA AnnData from the raw count matrix, for Fig 3 (results/melanoma/melanoma_rna.h5ad).

Recovered from paper/figures/fig3/upstream/melanoma_prep.py: build_rna() and melanoma_score(), code unchanged. Left out:
its single-modality GASTON-Mix-MM fit, which is an old baseline and is not read by Fig 3. The published run's inferCNV call
failed ("requires gtfparse", results/melanoma/prep.log), so cnv_score is all NaN in the published object; it is NaN here too.
Fig 3 reads X, var_names, obsm['spatial'] from this object. Its extra columns compartment and cnv_fixed were added later by
other code and are not read by any Fig 3 step.
"""
import os
import anndata as ad, numpy as np, pandas as pd, scanpy as sc, scipy.io, scipy.sparse as sp

ROOT = os.environ["SATURN_ROOT"]
M = f"{ROOT}/Yufan_Melanoma"
OUT = f"{ROOT}/Hole/gaston_mix_mm/results/melanoma"; os.makedirs(OUT, exist_ok=True)
MEL_MARKERS = ["MLANA", "PMEL", "TYR", "TYRP1", "DCT", "MITF", "SOX10", "MCAM", "S100B"]


def build_rna():
    X = scipy.io.mmread(f"{M}/RNA/raw/matrix.mtx.gz").tocsr()        # genes x spots
    feat = pd.read_csv(f"{M}/RNA/raw/features.tsv.gz", sep="\t", header=None)
    bc = pd.read_csv(f"{M}/RNA/raw/barcodes.tsv.gz", header=None)[0].astype(str).values
    a = ad.AnnData(sp.csr_matrix(X.T))                              # spots x genes
    a.obs_names = bc
    a.var_names = feat[1].astype(str).str.upper().values
    a.var_names_make_unique()
    pos = pd.read_csv(f"{M}/spatial/tissue_positions_list.csv", header=None,
                      index_col=0,
                      names=["bc", "in_tissue", "row", "col", "pxl_row", "pxl_col"])
    common = [b for b in a.obs_names if b in pos.index]
    a = a[common].copy()
    a.obsm["spatial"] = pos.loc[common, ["pxl_col", "pxl_row"]].to_numpy().astype(float)
    a.obs[["array_row", "array_col"]] = pos.loc[common, ["row", "col"]].to_numpy()
    # basic spot filter (very mild)
    sc.pp.filter_cells(a, min_counts=200)
    print(f"RNA: {a.shape} spots after min_counts=200", flush=True)
    return a


def melanoma_score(a):
    g = [m for m in MEL_MARKERS if m in a.var_names]
    b = a[:, g].copy(); sc.pp.normalize_total(b, target_sum=1e4); sc.pp.log1p(b)
    z = (b.X.toarray() if sp.issparse(b.X) else np.asarray(b.X))
    z = (z - z.mean(0)) / (z.std(0) + 1e-9)
    return z.mean(1), g


rna = build_rna()
mscore, mg = melanoma_score(rna)
print(f"melanoma markers found: {mg}", flush=True)
rna.obs["mel_score"] = mscore; rna.obs["cnv_score"] = np.nan
rna.write(os.path.join(OUT, "melanoma_rna.h5ad"))
print("->", OUT)
