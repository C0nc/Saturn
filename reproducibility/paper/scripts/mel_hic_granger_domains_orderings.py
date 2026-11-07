"""Does the isodepth itself capture Granger coupling? Per-domain counts along rho vs four comparisons, melanoma fit (Fig 6).

Same domains, layers, 500-spot subsamples (same seeds) and test as mel_hic_granger_domains.py. In each repeat:
  rho           spots ordered along the domain's own isodepth
  shift         rho order, chromatin circularly shifted 5-25 windows against genes (alignment broken)
  direction     spots ordered along a random spatial direction (x cos t + y sin t, new t per repeat)
  random        spots in random order
  other         spots ordered along the other modality's isodepth
Detrending uses the ordering variable of that comparison. Count = pairs with either direction P < 0.05.
Empirical one-sided P for each comparison = (1 + repeats with comparison count >= rho count) / (repeats + 1), BH over all
12 groups x 4 comparisons.
-> results/real_saturn/<DS>_insulation_granger/granger_domains_orderings_{counts,summary}.csv
--depth: log Hi-C reads and log RNA depth are regressed from every per-spot series first -> *_depth.csv
Usage: python scripts/mel_hic_granger_domains_orderings.py [--repeats 200] [--nsub 500] [--depth]
"""
import sys
DEPTH = "--depth" in sys.argv
import mel_hic_granger_domains as M     # sets single-thread BLAS before numpy
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from saturn.domains import gaston_domains, dp_error_curve as curve, kneedle
from saturn_granger_rho import bh


def count(Gb, X):
    pf, pr = M.batch_pair(Gb, X)
    return int((np.nan_to_num(np.minimum(pf, pr), nan=1.0) < 0.05).sum())


def along(sub, v, Ss, chrom):
    E = M.windows(sub, v); rb = np.array([v[e].mean() for e in E])
    return M.detrend3(np.vstack([Ss[e].mean(0) for e in E]), rb), {L: M.detrend3(np.vstack([C[e].mean(0) for e in E]), rb) for L, C in chrom.items()}


def one(seed, idx, d, other, xy, Ss, chrom):
    rng = np.random.default_rng(seed)
    sub = rng.choice(idx, min(M.NSUB, len(idx)), replace=False)
    res = {L: {} for L in chrom}
    Gb, Cs = along(sub, d, Ss, chrom)
    for L, Cb in Cs.items():
        res[L]["rho"] = count(Gb, Cb)
        s = rng.integers(5, M.NBIN - 5, Cb.shape[1])
        res[L]["shift"] = count(Gb, Cb[(np.arange(M.NBIN)[:, None] - s[None, :]) % M.NBIN, np.arange(Cb.shape[1])])
    t = rng.uniform(0, np.pi)
    orders = {"direction": xy[:, 0] * np.cos(t) + xy[:, 1] * np.sin(t), "random": rng.random(len(d)), "other": other}
    for name, v in orders.items():
        Gb, Cs = along(sub, v, Ss, chrom)
        for L, Cb in Cs.items(): res[L][name] = count(Gb, Cb)
    return res


def main():
    C = np.load(f"{M.G}/cache_series.npz", allow_pickle=True); bc = list(C["bc"].astype(str))
    ax = pd.read_csv(f"{M.R}/{M.DS}/axes.csv", index_col=0).loc[bc]; xy = np.c_[ax.x.values, ax.y.values]
    z = dict(np.load(f"{M.R}/targets_{M.DS}.npz", allow_pickle=True))
    S = np.asarray(C["S"], float)
    chrom = {"A/B": np.asarray(C["AB"], float), "insulation": np.asarray(C["IN"], float)}
    if DEPTH:
        from continuous_marker_panels import mel_rna
        X, _ = mel_rna(bc); A = np.c_[np.ones(len(bc)), np.load(f"{M.G}/spot_insulation_summary.npy")[:, 2], np.log1p(np.asarray(X.sum(1)).ravel().astype(float))]
        res = lambda V: V - A @ np.linalg.lstsq(A, V, rcond=None)[0]
        S = res(S); chrom = {k: res(v) for k, v in chrom.items()}
    Ss = S[cKDTree(xy).query(xy, k=M.K)[1]].mean(1)
    rows, summ = [], []
    for a, b in (("RNA", "HiC"), ("HiC", "RNA")):
        d = ax[f"d_{a}"].values; other = ax[f"d_{b}"].values; Y = z[f"Y_{a}"]; lab = gaston_domains(d, Y, n_domains=kneedle(curve(d, Y)))
        for k in range(lab.max() + 1):
            idx = np.where(lab == k)[0]; nm = M.NAMES[a][len(idx)]
            out = Parallel(n_jobs=64)(delayed(one)(1000 * k + s + (0 if a == "RNA" else 500000), idx, d, other, xy, Ss, chrom) for s in range(M.REPS))
            for L in chrom:
                o = np.array([r[L]["rho"] for r in out])
                for s, r in enumerate(out):
                    rows.append(dict(axis=a, domain=nm, layer=L, repeat=s, **r[L]))
                for ctrl in ("shift", "direction", "random", "other"):
                    n = np.array([r[L][ctrl] for r in out])
                    summ.append(dict(axis=a, domain=nm, layer=L, comparison=ctrl, rho_median=np.median(o), ctrl_median=np.median(n),
                                     share_ctrl_ge_rho=(n >= o).mean(), p_emp=(1 + (n >= o).sum()) / (M.REPS + 1)))
                print(a, nm, L, {c: float(np.median([r[L][c] for r in out])) for c in out[0][L]}, flush=True)
    T = pd.DataFrame(summ); T["q_bh"] = bh(T.p_emp.values)
    suf = "_depth" if DEPTH else ""
    pd.DataFrame(rows).to_csv(f"{M.G}/granger_domains_orderings_counts{suf}.csv", index=False)
    T.to_csv(f"{M.G}/granger_domains_orderings_summary{suf}.csv", index=False)
    pd.set_option("display.width", 200); print(T.round(3).to_string(index=False)); print("->", M.G)


if __name__ == "__main__":
    main()
