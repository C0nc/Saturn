"""Add ATAC to the melanoma slice (snapatac2 tile matrix), refit GASTON-Mix-MM as
RNA+ATAC (2-mod), and run the modality-dominance analysis + redo the tumor-border test
with the true 2-modality trace(g). Updates results/melanoma/."""

from __future__ import annotations

import os
import pickle
import sys

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from gastonmix_mm.data import load_multimodal                       # noqa: E402
from gastonmix_mm.model import GastonMixMM                           # noqa: E402
from gastonmix_mm.train import train_model, build_knn, get_isodepth  # noqa: E402
from gastonmix_mm.gradient_analysis import estimate_gradients        # noqa: E402
from gastonmix_mm import dg_fields as dg                             # noqa: E402

M = "/mnt/HDD2B/Users/yancui/Yufan_Melanoma"
OUT = os.path.join(ROOT, "results", "melanoma")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build_atac(barcodes):
    import snapatac2 as snap
    import anndata
    import scipy.sparse as sp
    out = os.path.join(OUT, "atac_snap.h5ad")
    if not os.path.exists(out):
        data = snap.pp.import_fragments(f"{M}/ATAC/HmelanomaA.fragments.sort.bed.gz",
                                        chrom_sizes=snap.genome.hg38, file=out,
                                        sorted_by_barcode=False, min_num_fragments=100)
        snap.pp.add_tile_matrix(data, bin_size=5000)
        data.close()
    d = snap.read(out)
    a = anndata.AnnData(X=sp.csr_matrix(d.X[:]))             # snapatac backed -> standard
    a.obs_names = list(d.obs_names); a.var_names = [f"tile{i}" for i in range(a.shape[1])]
    a = a[[b for b in a.obs_names if b in set(barcodes)]].copy()
    print(f"ATAC tile matrix: {a.shape}", flush=True)
    return a


def main():
    rna = ad.read_h5ad(os.path.join(OUT, "melanoma_rna.h5ad"))
    atac = build_atac(np.asarray(rna.obs_names))
    atac.obsm["spatial"] = np.asarray(rna[atac.obs_names].obsm["spatial"])
    mm = load_multimodal({"RNA": rna, "ATAC": atac}, coord_source="RNA",
                         n_top={"RNA": 2000, "ATAC": 3000}, feat_mode={"RNA": "hvg", "ATAC": "count"},
                         min_coherence=0.05, min_detect=0.02)
    print(f"melanoma RNA+ATAC: {len(mm.barcodes)} spots, dims {[c.shape[1] for c in mm.counts]}", flush=True)
    coords, targets, libs = mm.to_torch()
    knn = build_knn(mm.coords, k=6)
    model = GastonMixMM([c.shape[1] for c in mm.counts], likelihood="poisson",
                        iso_hidden=[128, 128], dec_hidden=[128], enc_dim=16, sigma=3.0)
    model, _ = train_model(model, coords, targets, libsize=libs, knn_idx=knn,
                           n_epochs=1500, lr=2e-3, smooth_weight=2.0, coupling_weight=0.0,
                           self_recon_weight=3.0, var_floor_weight=1.0, n_restarts=3,
                           device=DEV, verbose=0)
    rho = get_isodepth(model, coords, device=DEV)
    grads = estimate_gradients(mm.coords, rho, k=12)
    from gastonmix_mm.sign_calibration import calibrate_fit         # DEFAULT prior calibration
    rho, grads, csigns = calibrate_fit(rho, grads, mm.modality_names)
    order = {b: i for i, b in enumerate(rna.obs_names)}
    idx = np.array([order[b] for b in mm.barcodes])
    mel = rna.obs["mel_score"].to_numpy()[idx]; cnv = rna.obs["cnv_score"].to_numpy()[idx]
    pickle.dump({"coords": mm.coords, "rho": rho, "grads": grads,
                 "modality_names": list(mm.modality_names), "barcodes": np.asarray(mm.barcodes),
                 "sign_calibrated": True, "calibration_signs": csigns,
                 "mel_score": mel, "cnv_score": cnv},
                open(os.path.join(OUT, "gradient_analysis_2mod.pkl"), "wb"))

    # modality dominance + tumor relation
    F = dg.all_dg_fields(mm.coords, rho, grads)
    dom = F["pullback"]["dominant_modality"]; contrib = F["pullback"]["modality_contribution"]
    tum = mel >= np.percentile(mel, 60)
    c = mm.coords
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a_, (v, t, cm) in zip(ax, [(dom, f"dominant modality ({'/'.join(mm.modality_names)})", "tab10"),
                                   (contrib[:, 0], f"{mm.modality_names[0]} contribution to g", "viridis"),
                                   (mel, "melanoma marker score (tumor)", "magma")]):
        s = a_.scatter(c[:, 0], c[:, 1], c=v, s=14, cmap=cm); a_.set_title(t, fontsize=9)
        a_.set_aspect("equal"); a_.invert_yaxis(); a_.set_xticks([]); a_.set_yticks([]); fig.colorbar(s, ax=a_, shrink=0.7)
    fig.suptitle("Melanoma RNA+ATAC: modality dominance vs tumor", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "melanoma_modality_dominance.png"), dpi=140); plt.close(fig)
    # which modality dominates in tumor vs normal
    rna_dom_tumor = float(np.mean(dom[tum] == 0)); rna_dom_normal = float(np.mean(dom[~tum] == 0))
    summ = {"rna_contribution_mean": float(contrib[:, 0].mean()),
            "frac_RNA_dominant_in_tumor": rna_dom_tumor,
            "frac_RNA_dominant_in_normal": rna_dom_normal,
            "modalities": list(mm.modality_names), "n_spots": int(len(mm.barcodes))}
    import json
    json.dump(summ, open(os.path.join(OUT, "modality_dominance_summary.json"), "w"), indent=2)
    print(summ, flush=True)
    print("done -> melanoma_modality_dominance.png", flush=True)


if __name__ == "__main__":
    main()
