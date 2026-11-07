#!/usr/bin/env python
"""Put the published RNA GLM-PCs back into a freshly built targets_mel_hic_fh_dr50_51.npz.

saturn.preprocess.rna_glmpc_real calls glmpca, which starts from np.random.randn; neither it nor mel_hic_prep.py sets a seed,
so each run of the prep gives different GLM-PCs and no rerun can recover the published draw. The rest of the target rebuilds
exactly. This keeps every array mel_hic_prep.py wrote except Y_RNA, and saves the fresh draw for comparison.
Usage: python fig6_restore_glmpc.py <results/real_saturn folder> <published draw .npz>
"""
import sys
import numpy as np

R, G = sys.argv[1], sys.argv[2]; T = f"{R}/targets_mel_hic_fh_dr50_51.npz"
z = dict(np.load(T, allow_pickle=True)); g = np.load(G, allow_pickle=True)
assert list(z["barcodes"].astype(str)) == list(g["barcodes"].astype(str))
np.savez_compressed(f"{R}/glmpc_fresh_draw_mel_hic_fh_dr50_51.npz", Y_RNA=z["Y_RNA"], barcodes=z["barcodes"])
z["Y_RNA"] = g["Y_RNA"]; np.savez_compressed(T, **z); print("Y_RNA replaced by the published draw")
