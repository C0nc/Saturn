"""Melanoma prep: build RNA AnnData + spatial, compute melanoma marker score and CNV
(infercnvpy), and fit GASTON-Mix-MM (RNA isodepth). Saves to results/melanoma/.

Goal downstream: test whether the isodepth gradient-rate RIDGE coincides with the TUMOR
BORDER (defined by CNV burden / melanocyte markers).
"""

from __future__ import annotations

import os
import pickle
import sys

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
import scanpy as sc
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from gastonmix_mm.data import load_multimodal                       # noqa: E402
from gastonmix_mm.model import GastonMixMM                           # noqa: E402
from gastonmix_mm.train import train_model, build_knn, get_isodepth  # noqa: E402
from gastonmix_mm.gradient_analysis import estimate_gradients        # noqa: E402

M = "<workspace>/Yufan_Melanoma"
GTF = "<workspace>/scdori_run/human_genome/annotation.gtf"
OUT = os.path.join(ROOT, "results", "melanoma"); os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
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


def cnv_score(a):
    import infercnvpy as cnv
    b = a.copy(); sc.pp.normalize_total(b, target_sum=1e4); sc.pp.log1p(b)
    cnv.io.genomic_position_from_gtf(GTF, b, gtf_gene_id="gene_name", adata_gene_id=None)
    b = b[:, b.var["chromosome"].notnull()].copy()
    cnv.tl.infercnv(b, window_size=100, step=10)
    score = np.abs(np.asarray(b.obsm["X_cnv"].todense() if sp.issparse(b.obsm["X_cnv"])
                              else b.obsm["X_cnv"])).mean(1)
    return np.asarray(score).ravel()


def main():
    rna = build_rna()
    mscore, mg = melanoma_score(rna)
    print(f"melanoma markers found: {mg}", flush=True)
    try:
        cscore = cnv_score(rna)
        print("CNV score computed", flush=True)
    except Exception as e:
        print("CNV failed:", e, flush=True); cscore = np.full(rna.shape[0], np.nan)
    rna.obs["mel_score"] = mscore; rna.obs["cnv_score"] = cscore
    rna.write(os.path.join(OUT, "melanoma_rna.h5ad"))

    # fit GASTON-Mix-MM on RNA (single modality) with feature QC
    mm = load_multimodal({"RNA": rna}, coord_source="RNA", n_top={"RNA": 2000},
                         feat_mode={"RNA": "hvg"}, min_coherence=0.05, min_detect=0.02)
    coords, targets, libs = mm.to_torch()
    knn = build_knn(mm.coords, k=6)
    model = GastonMixMM([mm.counts[0].shape[1]], likelihood="poisson",
                        iso_hidden=[128, 128], dec_hidden=[128], enc_dim=16, sigma=3.0)
    model, _ = train_model(model, coords, targets, libsize=libs, knn_idx=knn,
                           n_epochs=1500, lr=2e-3, smooth_weight=2.0, coupling_weight=0.0,
                           self_recon_weight=3.0, var_floor_weight=1.0, n_restarts=3,
                           device=DEV, verbose=0)
    rho = get_isodepth(model, coords, device=DEV)
    grads = estimate_gradients(mm.coords, rho, k=12)
    from gastonmix_mm.sign_calibration import calibrate_fit         # DEFAULT prior calibration
    rho, grads, csigns = calibrate_fit(rho, grads, mm.modality_names)
    # align scores to mm spot order
    order = {b: i for i, b in enumerate(rna.obs_names)}
    idx = np.array([order[b] for b in mm.barcodes])
    pickle.dump({"coords": mm.coords, "rho": rho, "grads": grads, "modality_names": ["RNA"],
                 "barcodes": np.asarray(mm.barcodes), "sign_calibrated": True,
                 "calibration_signs": csigns,
                 "mel_score": mscore[idx], "cnv_score": cscore[idx]},
                open(os.path.join(OUT, "gradient_analysis.pkl"), "wb"))
    print(f"fit done: {mm.coords.shape[0]} spots; saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
