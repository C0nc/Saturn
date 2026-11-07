"""Does the held-out protein-complex layer really agree with the fitted isodepth?

A raw correlation between a PLA product and rho is not evidence, for three reasons,
and each gets a control here:

  1. BOTH PROTEINS MAY DRIVE IT. A complex can track rho merely because its two
     proteins do. Controlled by working on the excess over independence
     (observed - marginal_i*marginal_j/total) AND by a partial Spearman of the raw
     product against rho with both marginals regressed out.
  2. ANY SMOOTH FIELD CORRELATES WITH ANY OTHER. Both rho and the complex layer are
     spatially autocorrelated, so |rho| is inflated under the naive null. Controlled
     by a SPATIAL null: rho is replaced by a rotated copy of itself (the tissue is
     rotated about its centroid and rho re-interpolated onto the original spots),
     which preserves its autocorrelation while destroying its alignment with the
     complex. p = fraction of rotations reaching the observed |correlation|.
  3. ONE SECTION IS ONE SAMPLE. Controlled by repeating everything on the three
     sections and asking whether the per-product correlations REPLICATE.

Usage: python scripts/sproxseq_validate2.py [n_rotations]
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from scipy.interpolate import griddata
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
from prep_real2 import H
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "svg.fonttype": "none"})

R = f"{H}/Hole/gaston_mix_mm/results/real_saturn"
SECS = ["A1", "B1", "D1"]
MIN_COUNT, NROT = 200, 200
CLAIMS = [("ITGA4", "VCAM1"), ("CD29", "VCAM1"), ("LFA1", "ICAM1"), ("CD21", "CD35")]


def partial_spearman(y, x, Z):
    """Spearman of y vs x after regressing both on the columns of Z (rank space)"""
    ry, rx = rankdata(y), rankdata(x)
    rz = np.c_[np.ones(len(y)), np.apply_along_axis(rankdata, 0, Z)]
    by = ry - rz @ np.linalg.lstsq(rz, ry, rcond=None)[0]
    bx = rx - rz @ np.linalg.lstsq(rz, rx, rcond=None)[0]
    s = np.sqrt((by ** 2).sum() * (bx ** 2).sum())
    return float((by * bx).sum() / s) if s > 0 else 0.0


def rotated_fields(xy, d, n, rng):
    """n copies of d with its spatial autocorrelation preserved but its alignment gone"""
    c = xy.mean(0)
    out = []
    for _ in range(n):
        th = rng.uniform(0, 2 * np.pi)
        Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        src = (xy - c) @ Rm.T + c
        v = griddata(src, d, xy, method="nearest")      # nearest keeps every spot defined
        out.append(v)
    return np.array(out)


def load(sec):
    ax = pd.read_csv(f"{R}/sproxseq_{sec}/axes.csv", index_col=0)
    P = dict(np.load(f"{R}/sproxseq_{sec}_pla.npz", allow_pickle=True))
    bc = [str(b) for b in P["barcodes"]]
    assert [str(b) for b in ax.index] == bc
    Cnt = P["counts"].astype(float)
    Marg = P["protein_marginal"].astype(float)
    prots = list(P["proteins"].astype(str))
    a_, b_ = P["prot_a"].astype(str), P["prot_b"].astype(str)
    tot = Cnt.sum(1)
    ia = np.array([prots.index(x) if x in prots else -1 for x in a_])
    ib = np.array([prots.index(x) if x in prots else -1 for x in b_])
    ok = (ia >= 0) & (ib >= 0) & (Cnt.sum(0) >= MIN_COUNT)
    exc = np.zeros_like(Cnt)
    exc[:, ok] = Cnt[:, ok] - Marg[:, ia[ok]] * Marg[:, ib[ok]] / np.maximum(tot, 1)[:, None]
    return ax, P, Cnt, exc, Marg, ia, ib, ok, tot, np.c_[ax.x.values, ax.y.values]


def run(nrot=NROT):
    rng = np.random.default_rng(0)
    per_sec, tabs = {}, {}
    for sec in SECS:
        if not os.path.exists(f"{R}/sproxseq_{sec}/axes.csv"):
            print(f"{sec}: not fitted yet -- skipped", flush=True); continue
        ax, P, Cnt, exc, Marg, ia, ib, ok, tot, xy = load(sec)
        prod = P["products"].astype(str); a_, b_ = P["prot_a"].astype(str), P["prot_b"].astype(str)
        res = {}
        for mod in ["RNA", "Protein"]:
            d = ax[f"d_{mod}"].values
            null = rotated_fields(xy, d, nrot, rng)
            F = exc[:, ok] / np.maximum(tot, 1)[:, None]
            obs = np.array([spearmanr(d, F[:, j])[0] for j in range(F.shape[1])])
            # null distribution of |rho| per product, from the rotated fields
            nl = np.empty((nrot, F.shape[1]))
            for t in range(nrot):
                nl[t] = [spearmanr(null[t], F[:, j])[0] for j in range(F.shape[1])]
            p = (1 + (np.abs(nl) >= np.abs(obs)).sum(0)) / (nrot + 1)
            res[mod] = (obs, p, np.abs(nl).mean(0))
            print(f"{sec} d_{mod}: median |rho| observed {np.abs(obs).mean():.3f} vs "
                  f"rotated-null mean {np.abs(nl).mean():.3f}; "
                  f"products with p<0.05: {int((p < 0.05).sum())}/{len(p)}", flush=True)
        # partial correlation of the RAW product, marginals regressed out
        j_ok = np.where(ok)[0]
        part = {}
        for mod in ["RNA", "Protein"]:
            d = ax[f"d_{mod}"].values
            part[mod] = np.array([partial_spearman(Cnt[:, j] / np.maximum(tot, 1), d,
                                                   np.c_[Marg[:, ia[j]], Marg[:, ib[j]]])
                                  for j in j_ok])
        T = pd.DataFrame(dict(product=prod[j_ok], p1=a_[j_ok], p2=b_[j_ok],
                              counts=Cnt[:, j_ok].sum(0).astype(int),
                              rho_exc_RNA=res["RNA"][0].round(3), p_rot_RNA=res["RNA"][1].round(3),
                              rho_exc_Prot=res["Protein"][0].round(3), p_rot_Prot=res["Protein"][1].round(3),
                              partial_RNA=np.round(part["RNA"], 3),
                              partial_Prot=np.round(part["Protein"], 3)))
        T["pair"] = [":".join(sorted([x, y])) for x, y in zip(T.p1, T.p2)]
        T.to_csv(f"{R}/sproxseq_{sec}_validation/pla_validated.csv", index=False)
        per_sec[sec] = T
        tabs[sec] = T.groupby("pair")[["rho_exc_RNA", "rho_exc_Prot", "partial_RNA", "partial_Prot"]].mean()

    # ---- replication across sections -----------------------------------------
    if len(tabs) > 1:
        print("\nreplication of the per-pair correlations across sections:")
        ks = list(tabs)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                m = tabs[ks[i]].join(tabs[ks[j]], lsuffix="_1", rsuffix="_2", how="inner")
                for col in ["rho_exc_RNA", "rho_exc_Prot"]:
                    r = spearmanr(m[f"{col}_1"], m[f"{col}_2"])[0]
                    print(f"  {ks[i]} vs {ks[j]} ({len(m)} shared pairs) {col}: Spearman {r:+.2f}")
        print("\nthe paper's claimed pairs, per section (excess rho on d_Protein / rotation p):")
        for p, q in CLAIMS:
            key = ":".join(sorted([p, q]))
            line = []
            for s in ks:
                T = per_sec[s]; sub = T[T.pair == key]
                if len(sub):
                    line.append(f"{s} {sub.rho_exc_Prot.mean():+.2f} (p={sub.p_rot_Prot.min():.3f})")
            print(f"  {key:16s} " + " | ".join(line))
    print("\nwritten pla_validated.csv per section")
    return per_sec


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else NROT)
