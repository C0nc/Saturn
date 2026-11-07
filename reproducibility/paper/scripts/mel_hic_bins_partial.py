"""Per-bin chromatin-vs-rho tracking with Hi-C read depth partialled out.

Insulation (and to a lesser degree A/B) from sparse single-spot contacts follows read depth
(spot mean insulation vs log reads Spearman 0.96), so an axis that follows coverage will look
like it tracks insulation. Here every series -- chromatin bin, rho, and each rotated rho of the
200-rotation spatial null -- is rank-transformed and residualised on rank(Hi-C reads) before the
correlation. -> <DS>_insulation_granger/bins_vs_rho_partial.csv

Usage: python scripts/mel_hic_bins_partial.py [--ds mel_hic_fh_dr50_51]
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domain_annot import R
from sproxseq_validate2 import rotated_fields

NROT, SEED = 200, 0


def main():
    ds = sys.argv[sys.argv.index("--ds") + 1] if "--ds" in sys.argv else "mel_hic_fh_dr50_51"
    G = f"{R}/{ds}_insulation_granger"
    ax = pd.read_csv(f"{R}/{ds}/axes.csv", index_col=0)
    C = np.load(f"{G}/cache_series.npz", allow_pickle=True)
    assert list(C["bc"].astype(str)) == [str(b) for b in ax.index]
    rc = np.load(f"{G}/spot_insulation_summary.npy")[:, 2]
    Z = np.c_[np.ones(len(rc)), rankdata(rc)]
    P = Z @ np.linalg.pinv(Z)

    def resid_z(M):
        M = np.apply_along_axis(rankdata, 0, np.atleast_2d(M.T).T).astype(float)
        M = M - P @ M; M -= M.mean(0)
        return M / np.linalg.norm(M, axis=0).clip(1e-12)

    xy = np.c_[ax.x.values, ax.y.values]; rng = np.random.default_rng(SEED); out = []
    for a in ("HiC", "RNA"):
        d = ax[f"d_{a}"].values
        Q = resid_z(np.column_stack([d] + list(rotated_fields(xy, d, NROT, rng))))
        for L, key in (("AB", "AB"), ("insulation", "IN")):
            M = resid_z(C[key]); r = M.T @ Q                       # bins x (1 + NROT)
            p = ((np.abs(r[:, 1:]) >= np.abs(r[:, :1])).sum(1) + 1) / (NROT + 1)
            out.append(pd.DataFrame(dict(axis=a, layer=L, bin=C["names"].astype(str), r=r[:, 0], p_rot=p)))
            print(f"{ds} rho_{a} {L}: {100 * (p < 0.05).mean():.1f}% bins p<0.05 (reads partialled)", flush=True)
    pd.concat(out).to_csv(f"{G}/bins_vs_rho_partial.csv", index=False)


if __name__ == "__main__":
    main()
