"""Per-domain Spearman coupling of chromatin and genes along each Saturn axis of the melanoma fit (replaces the Granger test in Fig 6).

Same domains, layers, 500-spot subsamples (same seeds), 30 equal-count windows and 25-NN gene scores as
mel_hic_granger_domains.py, without detrending: co-variation along the ordering is the signal. For each of the 2,558 bins, the
Spearman correlation of chromatin and gene score over the windows (two-sided t approximation). In each repeat:
  rho        along the domain's own isodepth
  shift      rho order, chromatin circularly shifted 5-25 windows against genes
  direction  along a random spatial direction (new angle per repeat)
  random     random spot order
  other      along the other modality's isodepth
Recorded: pairs with P < 0.05 (any sign), pairs with P < 0.05 in the expected sign (A/B positive, insulation negative), pairs with
BH q < 0.05 over bins, median signed r. Empirical one-sided P of rho vs each comparison = (1 + repeats with comparison >= rho) /
(repeats + 1), BH over 12 groups x 4 comparisons, for the any-sign and expected-sign counts separately.
-> results/real_saturn/<DS>_insulation_granger/spearman_domains_{counts,summary}.csv
--depth: log Hi-C reads and log RNA depth are regressed from every per-spot series first -> *_depth.csv
Usage: python scripts/mel_hic_spearman_domains.py [--repeats 200] [--nsub 500] [--depth]
"""
import sys
DEPTH = "--depth" in sys.argv
import mel_hic_granger_domains as M     # sets single-thread BLAS before numpy
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from scipy.stats import rankdata, t as tdist
from saturn.domains import gaston_domains, dp_error_curve as curve, kneedle
from saturn_granger_rho import bh
SIGN = {"A/B": 1, "insulation": -1}; ORD = ("rho", "shift", "direction", "random", "other")


def spear(G, C):
    """column-wise Spearman of T x bins matrices; constant columns give NaN"""
    Rg, Rc = rankdata(G, axis=0), rankdata(C, axis=0)
    a, b = Rg - Rg.mean(0), Rc - Rc.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))
        n = G.shape[0]; t = r * np.sqrt((n - 2) / np.maximum(1 - r * r, 1e-12))
    return r, 2 * tdist.sf(np.abs(t), n - 2)


def stats(G, C, sign):
    r, p = spear(G, C); ok = np.isfinite(p); sig = ok & (p < .05)
    return dict(n_any=int(sig.sum()), n_exp=int((sig & (sign * np.nan_to_num(r) > 0)).sum()), n_q=int((np.nan_to_num(bh(p), nan=1) < .05).sum()),
                med_r=float(np.nanmedian(sign * r)))


def means(sub, v, Ss, chrom):
    E = M.windows(sub, v)
    return np.vstack([Ss[e].mean(0) for e in E]), {L: np.vstack([C[e].mean(0) for e in E]) for L, C in chrom.items()}


def one(seed, idx, d, other, xy, Ss, chrom):
    rng = np.random.default_rng(seed); sub = rng.choice(idx, min(M.NSUB, len(idx)), replace=False)
    res = {L: {} for L in chrom}; Gb, Cs = means(sub, d, Ss, chrom)
    for L, Cb in Cs.items():
        res[L]["rho"] = stats(Gb, Cb, SIGN[L])
        s = rng.integers(5, M.NBIN - 5, Cb.shape[1])
        res[L]["shift"] = stats(Gb, Cb[(np.arange(M.NBIN)[:, None] - s[None, :]) % M.NBIN, np.arange(Cb.shape[1])], SIGN[L])
    t = rng.uniform(0, np.pi)
    for nm, v in {"direction": xy[:, 0] * np.cos(t) + xy[:, 1] * np.sin(t), "random": rng.random(len(d)), "other": other}.items():
        Gb, Cs = means(sub, v, Ss, chrom)
        for L, Cb in Cs.items(): res[L][nm] = stats(Gb, Cb, SIGN[L])
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
            out = Parallel(n_jobs=25)(delayed(one)(1000 * k + s + (0 if a == "RNA" else 500000), idx, d, other, xy, Ss, chrom) for s in range(M.REPS))
            for L in chrom:
                for s, r in enumerate(out):
                    for o in ORD: rows.append(dict(axis=a, domain=nm, layer=L, repeat=s, order=o, **r[L][o]))
                get = lambda o, f: np.array([r[L][o][f] for r in out])
                for c in ORD[1:]:
                    summ.append(dict(axis=a, domain=nm, layer=L, comparison=c, rho_any=np.median(get("rho", "n_any")), ctrl_any=np.median(get(c, "n_any")),
                                     p_any=(1 + (get(c, "n_any") >= get("rho", "n_any")).sum()) / (M.REPS + 1),
                                     rho_exp=np.median(get("rho", "n_exp")), ctrl_exp=np.median(get(c, "n_exp")),
                                     p_exp=(1 + (get(c, "n_exp") >= get("rho", "n_exp")).sum()) / (M.REPS + 1),
                                     rho_q=np.median(get("rho", "n_q")), ctrl_q=np.median(get(c, "n_q")), rho_medr=np.median(get("rho", "med_r")), ctrl_medr=np.median(get(c, "med_r"))))
                print(a, nm, L, {o: (float(np.median(get(o, "n_any"))), float(np.median(get(o, "n_exp")))) for o in ORD}, flush=True)
    T = pd.DataFrame(summ); T["q_any"] = bh(T.p_any.values); T["q_exp"] = bh(T.p_exp.values)
    suf = "_depth" if DEPTH else ""
    pd.DataFrame(rows).to_csv(f"{M.G}/spearman_domains_counts{suf}.csv", index=False); T.to_csv(f"{M.G}/spearman_domains_summary{suf}.csv", index=False)
    pd.set_option("display.width", 240); print(T.round(3).to_string(index=False)); print("->", M.G)


if __name__ == "__main__":
    main()
