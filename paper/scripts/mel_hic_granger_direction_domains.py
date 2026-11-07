"""Direction of Granger coupling per domain, same test as the full-axis direction panel (mel_hic_insulation_granger.py --direction).
Within each domain of each isodepth, all spots, rho forward and reversed: 30 equal-count windows, 25-NN gene score, cubic detrend,
Granger both directions at lags 1-3 with Bonferroni over lags x directions (saturn_granger_rho.pair_p). For each direction: % of the 2,558 bins with
P < 0.05, bins with BH q < 0.05 over bins, and the same percentage averaged over 100 circular shifts of chromatin (5-25 windows).
-> results/real_saturn/<DS>_insulation_granger/granger_direction_domains.csv
"""
import mel_hic_granger_domains as M     # sets single-thread BLAS before numpy
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from saturn.domains import gaston_domains, dp_error_curve as curve, kneedle
from saturn_granger_rho import bh, pair_p
NULLS = 100


def shifted(seed, Gb, Cb):
    rng = np.random.default_rng(seed); s = rng.integers(5, M.NBIN - 5, Cb.shape[1])
    pf, pr = M.batch_pair(Gb, Cb[(np.arange(M.NBIN)[:, None] - s[None, :]) % M.NBIN, np.arange(Cb.shape[1])])
    return float(np.mean(np.nan_to_num(pf, nan=1) < .05)), float(np.mean(np.nan_to_num(pr, nan=1) < .05))


def lag1(B):
    return float(np.nanmedian((B[1:] * B[:-1]).sum(0) / np.maximum((B * B).sum(0), 1e-12)))


def main():
    C = np.load(f"{M.G}/cache_series.npz", allow_pickle=True); bc = list(C["bc"].astype(str))
    ax = pd.read_csv(f"{M.R}/{M.DS}/axes.csv", index_col=0).loc[bc]; xy = np.c_[ax.x.values, ax.y.values]
    z = dict(np.load(f"{M.R}/targets_{M.DS}.npz", allow_pickle=True))
    S = np.asarray(C["S"], float); Ss = S[cKDTree(xy).query(xy, k=M.K)[1]].mean(1)
    chrom = {"A/B": np.asarray(C["AB"], float), "insulation": np.asarray(C["IN"], float)}
    rows = []
    for a in ("RNA", "HiC"):
        d0 = ax[f"d_{a}"].values; Y = z[f"Y_{a}"]; lab = gaston_domains(d0, Y, n_domains=kneedle(curve(d0, Y)))
        for k in range(lab.max() + 1):
            idx = np.where(lab == k)[0]; nm = M.NAMES[a][len(idx)]
            for sgn, rd in ((1, "forward"), (-1, "reversed")):
                d = sgn * d0; E = M.windows(idx, d); rb = np.array([d[e].mean() for e in E])
                Gb = M.detrend3(np.vstack([Ss[e].mean(0) for e in E]), rb)
                for L, Cm in chrom.items():
                    Cb = M.detrend3(np.vstack([Cm[e].mean(0) for e in E]), rb)
                    pf, pr = M.batch_pair(Gb, Cb)
                    N = np.array(Parallel(n_jobs=40)(delayed(shifted)(1000 * k + s + (0 if sgn == 1 else 300), Gb, Cb) for s in range(NULLS)))
                    rows.append(dict(axis=a, domain=nm, n_spots=len(idx), reading=rd, layer=L, c2g=100 * np.mean(np.nan_to_num(pf, nan=1) < .05), g2c=100 * np.mean(np.nan_to_num(pr, nan=1) < .05),
                                     c2g_q=int((np.nan_to_num(bh(pf), nan=1) < .05).sum()), g2c_q=int((np.nan_to_num(bh(pr), nan=1) < .05).sum()),
                                     null_c2g=100 * N[:, 0].mean(), null_g2c=100 * N[:, 1].mean(), lag_chrom=lag1(Cb), lag_gene=lag1(Gb)))
            print(a, nm, flush=True)
    T = pd.DataFrame(rows); T.to_csv(f"{M.G}/granger_direction_domains.csv", index=False)
    pd.set_option("display.width", 200); print(T.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
