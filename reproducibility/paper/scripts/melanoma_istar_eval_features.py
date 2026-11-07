"""Per-feature held-out test: does the adjacent H&E add information beyond spatial interpolation?

At the 200 held-out spots, compare each feature's observed spot value with
  iStar     super-resolved feature averaged over the spot's 3x3 superpixel square, linearly
            calibrated on the TRAINING spots
  linear    griddata linear interpolation from training spots
  knn6      mean of the 6 nearest training spots (the strongest interpolation baseline for noisy data)
Delta r = r(iStar) - r(best baseline), with a 2,000x paired bootstrap 95% CI over held-out spots.
A feature "gains from H&E" only if that CI excludes 0.

Usage: python scripts/melanoma_istar_eval_features.py
"""
import json, pickle
import numpy as np, pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

HO = "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/results/real_saturn/melanoma_tissue_istar_feat_holdout10/"
NBOOT = 2000


def spot_mean(y, L, r=1):
    i = (L.y.values / 16).astype(int); j = (L.x.values / 16).astype(int)
    return np.array([y[max(a - r, 0):a + r + 1, max(b - r, 0):b + r + 1].mean() for a, b in zip(i, j)])


def r_(a, b):
    return np.corrcoef(a, b)[0, 1]


def main():
    L = pd.read_csv(HO + "locs-all.tsv", sep="\t", index_col=0)
    C = pd.read_csv(HO + "cnts-all.tsv", sep="\t", index_col=0).loc[L.index]
    held = pd.read_csv(HO + "heldout_spots.txt", header=None)[0].astype(str).values
    te = L.index.isin(held); tr = ~te; xy = L[["x", "y"]].values
    nb = cKDTree(xy[tr]).query(xy[te], k=6)[1]
    rng = np.random.default_rng(0); B = rng.integers(0, te.sum(), (NBOOT, te.sum()))
    rows = []
    for n in C.columns:
        v = C[n].values
        y = pickle.load(open(HO + f"cnts-super/{n}.pickle", "rb"))
        p = spot_mean(y, L)
        a, b = np.linalg.lstsq(np.c_[p[tr], np.ones(tr.sum())], v[tr], rcond=None)[0]
        ist = (a * p + b)[te]; obs = v[te]
        lin = griddata(xy[tr], v[tr], xy[te], method="linear")
        lin = np.where(np.isnan(lin), griddata(xy[tr], v[tr], xy[te], method="nearest"), lin)
        knn = v[tr][nb].mean(1)
        rb = {"linear": r_(lin, obs), "knn6": r_(knn, obs)}
        best = max(rb, key=rb.get); base = lin if best == "linear" else knn
        d = np.array([r_(ist[s], obs[s]) - r_(base[s], obs[s]) for s in B])
        rows.append(dict(feature=n, kind=n.split("_")[0], r_iStar=r_(ist, obs), r_linear=rb["linear"],
                         r_knn6=rb["knn6"], best_baseline=best, delta_r=r_(ist, obs) - rb[best],
                         ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5)))
    T = pd.DataFrame(rows).sort_values("delta_r", ascending=False)
    T["gains_from_HE"] = T.ci_lo > 0
    T.to_csv(HO + "heldout_eval_features.csv", index=False)
    pd.set_option("display.width", 200)
    print(T.round(3).to_string(index=False))
    for k in ("RNA", "TF"):
        s = T[T.kind == k]
        print(f"\n{k}: {int(s.gains_from_HE.sum())}/{len(s)} features gain from H&E (CI excludes 0); "
              f"median r iStar {s.r_iStar.median():.3f} vs best baseline {s[['r_linear','r_knn6']].max(1).median():.3f}")
    return T


if __name__ == "__main__":
    main()
