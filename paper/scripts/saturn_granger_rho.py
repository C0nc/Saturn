"""Isodepth-space Granger causality: does a cis peak PRECEDE its gene along rho?

The LPC3 framework ran Granger along rho over SEACells metacells. Here the ordering
unit is a rho BIN of the Saturn fit, so the test is native to the isodepth:

  within each GASTON domain of the chosen axis
    * order spots by rho, cut into equal-count bins -> a series indexed by isodepth
    * mean log-normalised gene expression and peak accessibility per bin
    * DETREND: regress out the linear trend in rho from both series (see below)
    * Granger F-test peak->gene and gene->peak at lags {1,2,3}
    * min p over lags, Bonferroni x (2 directions x n_lags), then BH across pairs

WHY DETREND. Both a gene and its peak typically rise (or fall) monotonically along rho
inside a domain. Two series sharing a trend are Granger-"causal" in both directions for
purely deterministic reasons. Removing the within-domain linear trend in rho makes the
test about the RESIDUAL lead-lag structure, which is the actual question. The script
reports the raw (trended) result too, so the inflation is visible rather than hidden.

A circular-shift null (one series rotated) calibrates the whole procedure.

Usage: python scripts/saturn_granger_rho.py <dataset> <RNA|ATAC> [n_genes]
"""
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
from prep_real2 import H
from lpc_gaston_fig4l import gaston_domains
from domain_annot import CFG, R, all_tss
from scipy.stats import f as fdist

WIN, KDOM, NBIN, LAGS = 500_000, 4, 30, (1, 2, 3)
MAX_PEAKS, MIN_PEAK_COUNT = 60, 50


def granger_p(y, x, lag):
    """F-test: does adding lagged x improve prediction of y over lagged y alone?"""
    n = len(y) - lag
    if n < 3 * lag + 4: return np.nan
    Y = y[lag:]
    Ly = np.column_stack([y[lag - k: -k] for k in range(1, lag + 1)])
    Lx = np.column_stack([x[lag - k: -k] for k in range(1, lag + 1)])
    one = np.ones((n, 1))
    def rss(A):
        b, *_ = np.linalg.lstsq(A, Y, rcond=None)
        return float(((Y - A @ b) ** 2).sum())
    r0, r1 = rss(np.hstack([one, Ly])), rss(np.hstack([one, Ly, Lx]))
    df2 = n - (1 + 2 * lag)
    if df2 <= 0 or r1 <= 0 or not np.isfinite(r0 + r1): return np.nan
    F = ((r0 - r1) / lag) / (r1 / df2)
    return float(fdist.sf(F, lag, df2)) if F > 0 else 1.0


def pair_p(y, x):
    """min p over lags, Bonferroni-corrected by (2 directions x n_lags)"""
    m = 2 * len(LAGS)
    pf = [granger_p(y, x, L) for L in LAGS]          # peak -> gene
    pr = [granger_p(x, y, L) for L in LAGS]          # gene -> peak
    f = np.nanmin(pf) if np.any(np.isfinite(pf)) else np.nan
    r = np.nanmin(pr) if np.any(np.isfinite(pr)) else np.nan
    return min(f * m, 1.0), min(r * m, 1.0), int(np.nanargmin(pf) + 1) if np.any(np.isfinite(pf)) else -1


def bh(p):
    p = np.asarray(p, float); ok = np.isfinite(p); q = np.full(p.shape, np.nan)
    v = p[ok]; o = np.argsort(v); n = len(v)
    qq = np.minimum.accumulate((v[o] * n / (np.arange(n) + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(qq, 1.0); q[ok] = out
    return q


def lognorm(M):
    lib = np.maximum(np.asarray(M.sum(1)).ravel(), 1)
    return np.log1p(sp.diags(np.median(lib) / lib) @ M)


def detrend(V, r):
    A = np.column_stack([np.ones_like(r), r])
    return V - A @ np.linalg.lstsq(A, V, rcond=None)[0]


def run(ds, axis="RNA", n_genes=200, seed=0, nbin=None):
    global NBIN
    NBIN = nbin or NBIN
    OUT = f"{R}/{ds}_granger"; os.makedirs(OUT, exist_ok=True)
    cfg = CFG[ds]
    ax = pd.read_csv(f"{R}/{ds}/axes.csv", index_col=0); bc = [str(b) for b in ax.index]
    z = dict(np.load(f"{R}/targets_{ds}.npz", allow_pickle=True))
    d = ax[f"d_{axis}"].values
    lab = gaston_domains(d, z[f"Y_{axis}"], L_=KDOM)

    rna = cfg["rna"](); atac = cfg["atac"]()
    ri = {str(b): i for i, b in enumerate(rna.obs_names)}
    ai = {str(b): i for i, b in enumerate(atac.obs_names)}
    Xr = lognorm(sp.csr_matrix(rna.X)[[ri[b] for b in bc]])
    Xa = lognorm(sp.csr_matrix(atac.X)[[ai[b] for b in bc]])
    genes = np.array([str(g) for g in rna.var_names]); peaks = np.array([str(p) for p in atac.var_names])

    gs = pd.read_csv(f"{R}/{ds}_marker_functions/gene_screen.csv")
    cand = gs[gs.n_sloped_domains >= 2].sort_values("R2", ascending=False).feature.astype(str).tolist()
    gidx = {g: i for i, g in enumerate(genes)}
    cand = [g for g in cand if g in gidx][:n_genes]

    ch, po, nm = all_tss(cfg["gtf"])
    tss = {}
    for c_, p_, n_ in zip(ch, po, nm):
        tss.setdefault(n_, (c_, p_))
    pc = np.array([p.split(":")[0] for p in peaks])
    pm = np.array([(lambda a, b: (float(a) + float(b)) / 2)(*p.split(":")[1].split("-")) for p in peaks])
    ptot = np.asarray(Xa.sum(0)).ravel().astype(float)

    rows = []
    rng = np.random.default_rng(seed)
    for k in range(KDOM):
        m = np.where(lab == k)[0]
        if len(m) < NBIN * 3: continue
        o = m[np.argsort(d[m])]
        edges = np.array_split(o, NBIN)
        rb = np.array([d[e].mean() for e in edges])
        Gb = np.vstack([np.asarray(Xr[e].mean(0)).ravel() for e in edges])
        Ab = np.vstack([np.asarray(Xa[e].mean(0)).ravel() for e in edges])
        Gd, Ad = detrend(Gb, rb), detrend(Ab, rb)
        for g in cand:
            if g not in tss: continue
            c_, t_ = tss[g]
            sel = np.where((pc == c_) & (np.abs(pm - t_) <= WIN) & (ptot >= MIN_PEAK_COUNT))[0]
            if len(sel) == 0: continue
            if len(sel) > MAX_PEAKS: sel = sel[np.argsort(-ptot[sel])[:MAX_PEAKS]]
            gi = gidx[g]
            for tag, G, A in (("detrended", Gd, Ad), ("raw", Gb, Ab)):
                y = G[:, gi]
                if y.std() < 1e-9: continue
                for s_ in sel:
                    x = A[:, s_]
                    if x.std() < 1e-9: continue
                    pf, pr, bl = pair_p(y, x)
                    xs = np.roll(x, rng.integers(5, NBIN - 5))       # circular-shift null
                    pf0, pr0, _ = pair_p(y, xs)
                    rows.append(dict(dataset=ds, axis=axis, domain=k, series=tag, gene=g,
                                     peak=peaks[s_], kb=(pm[s_] - t_) / 1e3,
                                     p_peak2gene=pf, p_gene2peak=pr, best_lag=bl,
                                     p_null_p2g=pf0, p_null_g2p=pr0))
        print(f"   {ds} d_{axis} domain {k}: {len(m)} spots -> {NBIN} bins, "
              f"{len([r for r in rows if r['domain']==k and r['series']=='detrended'])} pairs", flush=True)
    T = pd.DataFrame(rows)
    for c_ in ["p_peak2gene", "p_gene2peak", "p_null_p2g", "p_null_g2p"]:
        T["q" + c_[1:]] = np.nan
        for key, gsub in T.groupby(["series", "domain"]):
            T.loc[gsub.index, "q" + c_[1:]] = bh(gsub[c_].values)
    tag = f"_nb{NBIN}" if NBIN != 30 else ""
    T.to_csv(f"{OUT}/granger_rho{axis}_win{WIN//1000}kb{tag}.csv", index=False)
    print(f"\n-> {OUT}/granger_rho{axis}_win{WIN//1000}kb{tag}.csv  ({len(T)} rows)")
    return T


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "RNA",
        int(sys.argv[3]) if len(sys.argv) > 3 else 200,
        nbin=int(sys.argv[4]) if len(sys.argv) > 4 else None)
