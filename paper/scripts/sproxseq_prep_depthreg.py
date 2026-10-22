"""Sprox-seq p46 targets with each modality's own log-depth regressed out -> targets_sproxseq_<sec>_p46_dr.npz

The p46 RNA target (Poisson GLM-PCA, depth as an offset) still carries depth: GLM-PC1 vs log UMI r 0.63-0.71 in
every section, and rho_RNA follows depth at |r| 0.38-0.63. Same fix as melanoma_tissue_prep.py: regress log-depth
out of the GLM-PCs and rescale. Protein gets its own log PLA depth regressed too -- on the Xenium FOV fit,
regressing RNA alone moved the confound onto the protein axis (0.23 -> 0.62).
The GLM-PCA / CLR-PCA factors are the existing p46 ones (not recomputed); everything else in the npz is copied.
Usage: python scripts/sproxseq_prep_depthreg.py [A1 B1 D1]
"""
import sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/saturn_repo/src")
from saturn.preprocess import scale_target

R = "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/results/real_saturn"


def regress(Y, depth):
    ld = np.log1p(np.asarray(depth, float)); A = np.c_[np.ones_like(ld), ld]
    Y = np.asarray(Y, float); Y = Y - A @ np.linalg.lstsq(A, Y, rcond=None)[0]
    return scale_target(Y, "global").astype(np.float32)


def main(secs):
    for sec in secs:
        z = dict(np.load(f"{R}/targets_sproxseq_{sec}_p46.npz", allow_pickle=True))
        for m in ("RNA", "Protein"):
            d = np.log1p(z[f"depth_{m}"].astype(float))
            before = max(abs(spearmanr(z[f"Y_{m}"][:, k], d)[0]) for k in range(z[f"Y_{m}"].shape[1]))
            z[f"Y_{m}"] = regress(z[f"Y_{m}"], z[f"depth_{m}"])
            after = max(abs(spearmanr(z[f"Y_{m}"][:, k], d)[0]) for k in range(z[f"Y_{m}"].shape[1]))
            print(f"{sec} {m}: max |r| of a component with log depth {before:.2f} -> {after:.2f}", flush=True)
        np.savez_compressed(f"{R}/targets_sproxseq_{sec}_p46_dr.npz", **z)
        print(f"   saved targets_sproxseq_{sec}_p46_dr.npz", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["A1", "B1", "D1"])
