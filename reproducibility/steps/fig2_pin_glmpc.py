"""Put the published RNA GLM-PCs into the rebuilt lpc10_fov target.

saturn.preprocess.rna_glmpc_real calls glmpca, which starts from np.random without a seed, and lpc10_fov_prep.py sets none, so every
run returns a different set of factors and the published draw cannot be regenerated. The notebook checks that everything else in
the rebuilt target equals the published target. fig2_glmpc_published.npz holds only the published Y_RNA and the barcodes.
Usage (from Hole/gaston_mix_mm): python <this file>
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); R = "results/real_saturn"
P = np.load(f"{HERE}/fig2_glmpc_published.npz", allow_pickle=True)
f = f"{R}/targets_lpc10_fov.npz"
z = dict(np.load(f, allow_pickle=True))
assert (z["barcodes"].astype(str) == P["barcodes"]).all() and z["Y_RNA"].shape == P["Y_RNA"].shape
d = float(np.abs(z["Y_RNA"] - P["Y_RNA"]).max())
z["Y_RNA"] = P["Y_RNA"]
np.savez_compressed(f, **z)
print(f"Y_RNA replaced by the published GLM-PCs (fresh draw differed by up to {d:.3f})", flush=True)
open(f"{R}/fig2_glmpc_pinned.txt", "w").write("lpc10_fov\n")
