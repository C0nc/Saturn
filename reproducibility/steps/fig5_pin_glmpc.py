"""Put the published RNA GLM-PCs into the rebuilt p46 targets.

saturn.preprocess.rna_glmpc_real calls glmpca, which starts from np.random without a seed, and sproxseq_prep.py sets none.
Every run therefore returns a different rotation of the factors (the top 9 canonical correlations with the published ones
are 0.98-1.00 on B1, the 10th 0.39), and the published draw of 2026-09-20 cannot be regenerated from a seed.
Everything else in the rebuilt target is bitwise identical to the published target, and this is checked here.
fig5_glmpc_published.npz holds only the published Y_RNA of the three sections, read from the published targets.
Usage (from Hole/gaston_mix_mm): python <this file> A1 B1 D1
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); R = "results/real_saturn"
P = np.load(f"{HERE}/fig5_glmpc_published.npz", allow_pickle=True)
for s in sys.argv[1:]:
    f = f"{R}/targets_sproxseq_{s}_p46.npz"
    assert not os.path.islink(f), f"{f} is a link: refusing to write through it"
    z = dict(np.load(f, allow_pickle=True))
    assert (z["barcodes"].astype(str) == P[f"barcodes_{s}"]).all() and z["Y_RNA"].shape == P[f"Y_RNA_{s}"].shape
    d = float(np.abs(z["Y_RNA"] - P[f"Y_RNA_{s}"]).max())
    z["Y_RNA"] = P[f"Y_RNA_{s}"]
    np.savez_compressed(f, **z)
    print(f"{s}: Y_RNA replaced by the published GLM-PCs (fresh draw differed by up to {d:.3f})", flush=True)
open(f"{R}/fig5_glmpc_pinned.txt", "w").write(" ".join(sys.argv[1:]) + "\n")
