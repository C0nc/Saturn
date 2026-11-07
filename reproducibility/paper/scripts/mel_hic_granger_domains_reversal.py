"""Reversal test for the per-domain Granger analysis (mel_hic_granger_domains.py), both Saturn axes of the melanoma fit.

Same domains, layers, subsamples (same seeds) and test. Each repeat is read along rho forward and reversed.
  net pairs = pairs with chromatin->gene P < 0.05 minus pairs with gene->chromatin P < 0.05
A real lead along the axis changes the sign of net when rho is reversed; an asymmetry from the smoothness of the two series
keeps its sign. Flip statistic per repeat: D = net_forward - net_reversed, and the same D with chromatin circularly shifted.
Empirical two-sided P = (1 + repeats with |D_null| >= |D_observed|) / (repeats + 1), BH over the 12 comparisons.
-> results/real_saturn/<DS>_insulation_granger/granger_domains_reversal_counts.csv, granger_domains_reversal_summary.csv
Usage: python scripts/mel_hic_granger_domains_reversal.py [--repeats 200] [--nsub 500]
"""
import mel_hic_granger_domains as M     # sets single-thread BLAS before numpy
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from saturn.domains import gaston_domains, dp_error_curve as curve, kneedle
from saturn_granger_rho import bh


def both(seed, idx, d, Ss, chrom):
    return {"forward": M.one_repeat(seed, idx, d, Ss, chrom), "reversed": M.one_repeat(seed, idx, -d, Ss, chrom)}


def main():
    C = np.load(f"{M.G}/cache_series.npz", allow_pickle=True); bc = list(C["bc"].astype(str))
    ax = pd.read_csv(f"{M.R}/{M.DS}/axes.csv", index_col=0).loc[bc]; xy = np.c_[ax.x.values, ax.y.values]
    z = dict(np.load(f"{M.R}/targets_{M.DS}.npz", allow_pickle=True))
    S = np.asarray(C["S"], float); Ss = S[cKDTree(xy).query(xy, k=M.K)[1]].mean(1)
    chrom = {"A/B": np.asarray(C["AB"], float), "insulation": np.asarray(C["IN"], float)}
    rows, summ = [], []
    for a in ("RNA", "HiC"):
        d = ax[f"d_{a}"].values; Y = z[f"Y_{a}"]; lab = gaston_domains(d, Y, n_domains=kneedle(curve(d, Y)))
        for k in range(lab.max() + 1):
            idx = np.where(lab == k)[0]; nm = M.NAMES[a][len(idx)]
            out = Parallel(n_jobs=40)(delayed(both)(1000 * k + s + (0 if a == "RNA" else 500000), idx, d, Ss, chrom) for s in range(M.REPS))
            for s, r in enumerate(out):
                for rd in r:
                    for (L, kind), v in r[rd].items():
                        rows.append(dict(axis=a, domain=nm, layer=L, repeat=s, reading=rd, kind=kind, c2g=v["c2g"], g2c=v["g2c"], net=v["c2g"] - v["g2c"]))
            for L in chrom:
                get = lambda rd, kind, f: np.array([r[rd][(L, kind)][f] for r in out])
                net = lambda rd, kind: get(rd, kind, "c2g") - get(rd, kind, "g2c")
                Do, Dn = net("forward", "observed") - net("reversed", "observed"), net("forward", "null") - net("reversed", "null")
                p = (1 + (np.abs(Dn) >= np.abs(Do)).sum()) / (M.REPS + 1)
                summ.append(dict(axis=a, domain=nm, layer=L, c2g_fwd=np.median(get("forward", "observed", "c2g")), g2c_fwd=np.median(get("forward", "observed", "g2c")),
                                 c2g_rev=np.median(get("reversed", "observed", "c2g")), g2c_rev=np.median(get("reversed", "observed", "g2c")),
                                 net_fwd=np.median(net("forward", "observed")), net_rev=np.median(net("reversed", "observed")),
                                 null_lo=np.percentile(np.r_[net("forward", "null"), net("reversed", "null")], 2.5),
                                 null_hi=np.percentile(np.r_[net("forward", "null"), net("reversed", "null")], 97.5),
                                 D_median=np.median(Do), Dnull_median=np.median(Dn), p_emp=p))
                print(summ[-1], flush=True)
    T = pd.DataFrame(summ); T["q_bh"] = bh(T.p_emp.values)
    T["flips"] = np.sign(T.net_fwd) * np.sign(T.net_rev) < 0
    pd.DataFrame(rows).to_csv(f"{M.G}/granger_domains_reversal_counts.csv", index=False)
    T.to_csv(f"{M.G}/granger_domains_reversal_summary.csv", index=False)
    pd.set_option("display.width", 240); print(T.round(3).to_string(index=False)); print("->", M.G)


if __name__ == "__main__":
    main()
