"""Put the published RNA GLM-PCs into the rebuilt Fig 4 target.

saturn.preprocess.rna_glmpc_real calls glmpca, which starts from np.random without a seed, so the published draw cannot be
regenerated. Only Y_RNA is replaced; the isoform target and every other array stay as rebuilt from the raw data.
Usage (from Hole/gaston_mix_mm): python <this file> rtt_merfish_s5_iso100_sv_pca_shrunk
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); f = f"results/real_saturn/targets_{sys.argv[1]}.npz"
assert not os.path.islink(f), f"{f} is a link: refusing to write through it"
P = np.load(f"{HERE}/fig4_glmpc_published.npz", allow_pickle=True); z = dict(np.load(f, allow_pickle=True))
assert (z["barcodes"].astype(str) == P["barcodes"].astype(str)).all() and z["Y_RNA"].shape == P["Y_RNA"].shape
print(f"Y_RNA replaced by the published GLM-PCs (fresh draw differed by up to {float(np.abs(z['Y_RNA'] - P['Y_RNA']).max()):.3f})")
z["Y_RNA"] = P["Y_RNA"]; np.savez_compressed(f, **z)
open("results/real_saturn/fig4_glmpc_pinned.txt", "w").write(sys.argv[1] + "\n")
