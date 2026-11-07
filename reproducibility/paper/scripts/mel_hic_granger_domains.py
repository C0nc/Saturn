"""Per-domain Granger coupling of chromatin and genes along each Saturn axis of the melanoma fit (Fig 6).

For every domain of rho_RNA and rho_HiC, and each chromatin layer (A/B, insulation) against the bin's genes (2,558 500 kb
bins), same per-pair test as mel_hic_granger_examples.py: spots ordered by rho -> 30 equal-count windows -> window means of
chromatin and of the 25-NN-averaged gene score -> cubic detrend on rho -> Granger both directions, lags 1-3, Bonferroni over
lags x directions (saturn_granger_rho.pair_p). A pair is significant if either direction has P < 0.05.

Each repeat draws the same number of spots from the domain (N_SUB, below the smallest domain) so counts are comparable
between domains. In the same repeat the null keeps the same spots and circularly shifts each bin's chromatin series against
its gene series by a random 5-25 windows. Empirical P (one-sided) = (1 + repeats with null >= observed) / (repeats + 1);
repeats are subsamples of one section, so this describes stability within the section, not biological replication.

-> results/real_saturn/<DS>_insulation_granger/granger_domains_counts.csv  (one row per repeat)
   granger_domains_summary.csv (one row per domain x layer)
Usage: python scripts/mel_hic_granger_domains.py [--repeats 200] [--nsub 500]
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from scipy.stats import f as fdist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "<workspace>/saturn_repo/src")
from saturn.domains import gaston_domains, dp_error_curve as curve, kneedle
from saturn_granger_rho import pair_p, bh
from domain_annot import R

DS = "mel_hic_abins_dr50_51_rnadr"; G = f"{R}/{DS}_insulation_granger"
arg = lambda k, d: int(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
REPS, NSUB = arg("--repeats", 200), arg("--nsub", 500)
NBIN, K, LAGS = 30, 25, (1, 2, 3)
NAMES = {"RNA": {653: "epidermis", 1239: "melanocytic tumour", 1960: "plasma / immune", 2256: "deep tumour"},
         "HiC": {2346: "non-tumour", 3762: "tumour"}}


def detrend3(V, r):
    z = (r - r.mean()) / r.std(); A = np.column_stack([z ** k for k in range(4)])
    return V - A @ np.linalg.lstsq(A, V, rcond=None)[0]


def _rss(A, Y):
    """batched least squares: A bins x n x p, Y bins x n"""
    b = np.linalg.pinv(A) @ Y[..., None]
    return ((Y - (A @ b)[..., 0]) ** 2).sum(1)


def batch_granger(y, x, lag):
    """granger_p for every column at once: does lagged x (T x bins) improve y (T x bins)?"""
    T, B = y.shape; n = T - lag; df2 = n - (1 + 2 * lag)
    Y = y[lag:].T
    Ly = np.stack([y[lag - k: T - k].T for k in range(1, lag + 1)], -1)
    Lx = np.stack([x[lag - k: T - k].T for k in range(1, lag + 1)], -1)
    one = np.ones((B, n, 1))
    r0, r1 = _rss(np.concatenate([one, Ly], -1), Y), _rss(np.concatenate([one, Ly, Lx], -1), Y)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = ((r0 - r1) / lag) / (r1 / df2)
    p = np.where(F > 0, fdist.sf(F, lag, df2), 1.0)
    p[(r1 <= 1e-20) | ~np.isfinite(r0 + r1)] = np.nan                     # constant series: no test
    return p


def batch_pair(g, c):
    """same as pair_p(g, c): min over lags, Bonferroni x (2 directions x lags); returns chromatin->gene, gene->chromatin"""
    m = 2 * len(LAGS)
    out = []
    for yy, xx in ((g, c), (c, g)):
        P = np.stack([batch_granger(yy, xx, L) for L in LAGS])
        allnan = np.isnan(P).all(0); Pm = np.where(allnan, 1.0, np.nanmin(np.where(np.isnan(P), np.inf, P), 0))
        Pm = np.minimum(Pm * m, 1.0); Pm[allnan] = np.nan; out.append(Pm)
    return out


def windows(idx, d):
    return np.array_split(idx[np.argsort(d[idx])], NBIN)


def one_repeat(seed, idx, d, Ss, chrom):
    rng = np.random.default_rng(seed)
    sub = rng.choice(idx, min(NSUB, len(idx)), replace=False) if seed >= 0 else idx
    E = windows(sub, d); rb = np.array([d[e].mean() for e in E])
    Gb = detrend3(np.vstack([Ss[e].mean(0) for e in E]), rb)
    res = {}
    for L, C in chrom.items():
        Cb = detrend3(np.vstack([C[e].mean(0) for e in E]), rb)
        shift = rng.integers(5, NBIN - 5, Cb.shape[1])
        Cn = Cb[(np.arange(NBIN)[:, None] - shift[None, :]) % NBIN, np.arange(Cb.shape[1])]
        for kind, X in (("observed", Cb), ("null", Cn)):
            pf, pr = batch_pair(Gb, X)
            sig = np.nan_to_num(np.minimum(pf, pr), nan=1.0) < 0.05
            q = bh(np.r_[pf, pr]); nb = len(pf); qs = (np.nan_to_num(q[:nb], nan=1) < .05) | (np.nan_to_num(q[nb:], nan=1) < .05)
            res[(L, kind)] = dict(n_sig=int(sig.sum()), n_q=int(qs.sum()), c2g=int((np.nan_to_num(pf, nan=1) < .05).sum()),
                                  g2c=int((np.nan_to_num(pr, nan=1) < .05).sum()), tested=int(np.isfinite(pf).sum()))
    return res


def main():
    C = np.load(f"{G}/cache_series.npz", allow_pickle=True); bc = list(C["bc"].astype(str))
    ax = pd.read_csv(f"{R}/{DS}/axes.csv", index_col=0).loc[bc]; xy = np.c_[ax.x.values, ax.y.values]
    z = dict(np.load(f"{R}/targets_{DS}.npz", allow_pickle=True))
    S = np.asarray(C["S"], float); Ss = S[cKDTree(xy).query(xy, k=K)[1]].mean(1)
    chrom = {"A/B": np.asarray(C["AB"], float), "insulation": np.asarray(C["IN"], float)}

    # the fast test must reproduce the published per-pair test
    d = ax["d_RNA"].values; E = np.array_split(np.argsort(d), NBIN); rb = np.array([d[e].mean() for e in E])
    Gb = detrend3(np.vstack([Ss[e].mean(0) for e in E]), rb); Cb = detrend3(np.vstack([chrom["A/B"][e].mean(0) for e in E]), rb)
    pf, pr = batch_pair(Gb, Cb); J = np.arange(0, Gb.shape[1], 13)
    ref = np.array([pair_p(Gb[:, j], Cb[:, j])[:2] for j in J], float)
    err = max(np.nanmax(np.abs(ref[:, 0] - pf[J])), np.nanmax(np.abs(ref[:, 1] - pr[J])))
    print(f"batch vs pair_p on {len(J)} bins: max |dP| = {err:.2e}", flush=True); assert err < 1e-6

    rows, summ = [], []
    for a in ("RNA", "HiC"):
        d = ax[f"d_{a}"].values; Y = z[f"Y_{a}"]; lab = gaston_domains(d, Y, n_domains=kneedle(curve(d, Y)))
        for k in range(lab.max() + 1):
            idx = np.where(lab == k)[0]; nm = NAMES[a][len(idx)]
            out = Parallel(n_jobs=40)(delayed(one_repeat)(1000 * k + s + (0 if a == "RNA" else 500000), idx, d, Ss, chrom) for s in range(REPS))
            for s, r in enumerate(out):
                for (L, kind), v in r.items(): rows.append(dict(axis=a, domain=nm, n_spots=len(idx), layer=L, repeat=s, kind=kind, **v))
            for L in chrom:
                o = np.array([r[(L, "observed")]["n_sig"] for r in out]); n = np.array([r[(L, "null")]["n_sig"] for r in out])
                oq = np.array([r[(L, "observed")]["n_q"] for r in out]); nq = np.array([r[(L, "null")]["n_q"] for r in out])
                p = (1 + (n >= o).sum()) / (REPS + 1)
                summ.append(dict(axis=a, domain=nm, n_spots=len(idx), layer=L, tested=out[0][(L, "observed")]["tested"],
                                 obs_median=np.median(o), null_median=np.median(n), obs_iqr=f"{np.percentile(o, 25):g}-{np.percentile(o, 75):g}",
                                 null_iqr=f"{np.percentile(n, 25):g}-{np.percentile(n, 75):g}", p_emp=p, q_obs_median=np.median(oq), q_null_median=np.median(nq)))
                print(summ[-1], flush=True)
    T = pd.DataFrame(summ); T["q_bh"] = bh(T.p_emp.values)
    pd.DataFrame(rows).to_csv(f"{G}/granger_domains_counts.csv", index=False)
    T.to_csv(f"{G}/granger_domains_summary.csv", index=False)
    pd.set_option("display.width", 220); print(T.to_string(index=False)); print("->", G)


if __name__ == "__main__":
    main()
