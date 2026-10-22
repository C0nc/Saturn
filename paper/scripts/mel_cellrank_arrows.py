"""Spatial fate arrows for the melanoma CellRank 2 kernels (Fig 3 k).

Rebuilds the four kernels of mel_cellrank_kernels.py exactly (same AnnData, graphs, DPT root, 0.8 Pseudotime + 0.2
Connectivity; fusion = 0.5 k_rhoRNA + 0.5 k_rhoATAC) and projects each transition matrix onto the tissue the way
CellRank's plot_projection does for an embedding: for spot i,
    v_i = sum_j (T_ij - 1/n_i) * (x_j - x_i) / ||x_j - x_i||        (n_i = non-zero transitions of i)
i.e. the expected direction of the next step minus the direction of a uniform step over the same neighbours.
The kernel graphs are expression graphs (RNA GLM-PCA / ATAC spectral), so neighbours can be far apart in space; the
arrows are therefore averaged on a regular grid (Gaussian, sigma = 1.5 spot pitches) before drawing.
-> results/real_saturn/melanoma_tissue_cellrank_kernels/arrows.npz  (V_<kernel> per spot, xy)
Usage: python scripts/mel_cellrank_arrows.py
"""
import os, sys
import numpy as np, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mel_cellrank_kernels as MK

KERNELS = ["DPT", "rho_RNA", "rho_ATAC", "fusion_w0.50"]


def project(T, xy):
    T = sp.csr_matrix(T); V = np.zeros_like(xy, dtype=float)
    for i in range(T.shape[0]):
        j = T.indices[T.indptr[i]:T.indptr[i + 1]]; w = T.data[T.indptr[i]:T.indptr[i + 1]]
        j, w = j[j != i], w[j != i]
        if not len(j): continue
        dx = xy[j] - xy[i]; u = dx / np.maximum(np.linalg.norm(dx, axis=1, keepdims=True), 1e-12)
        V[i] = ((w - 1.0 / len(j))[:, None] * u).sum(0)
    return V


def main():
    a, D = MK.build()
    xy = np.asarray(a.obsm["spatial"], float)
    ks = {"DPT": MK.kernel(a, "dpt_pseudotime", "rna"), "rho_RNA": MK.kernel(a, "rho_RNA", "rna"),
          "rho_ATAC": MK.kernel(a, "rho_ATAC", "atac")}
    ks["fusion_w0.50"] = 0.5 * ks["rho_RNA"] + 0.5 * ks["rho_ATAC"]
    out = dict(xy=xy, barcodes=np.array(a.obs_names, object))
    for k in KERNELS:
        kern = ks[k]
        kern = kern.compute_transition_matrix() if hasattr(kern, "compute_transition_matrix") else kern
        out[f"V_{k}"] = project(kern.transition_matrix, xy)
        print(k, "mean |v|", np.linalg.norm(out[f"V_{k}"], axis=1).mean().round(4), flush=True)
    np.savez_compressed(f"{MK.OUT}/arrows.npz", **out); print("->", f"{MK.OUT}/arrows.npz")


if __name__ == "__main__":
    main()
