"""Melanoma spatial Hi-C + RNA (mel_hic_fh): does the isodepth relate to TAD insulation, and
does chromatin -> expression Granger causality grow with the LENGTH of rho it is read over?

Inputs (all pre-existing):
  axes          results/real_saturn/mel_hic_fh/axes.csv (Saturn, seed 42)
  A/B           SpHIC117/abcompartment_matrix.csv (500 kb, read as granger_ab_gene.ipynb does)
  insulation    SpHIC117/HIC_100kb/higashi/tmp/scTAD.hdf5['insulation'] (100 kb, per spot;
                cell_i -> barcode via HIC_100kb/higashi/filelist.txt, as melanoma_is_draw.ipynb)
  Hi-C depth    SpHIC117/HIC/higashi/read_count_all.npy (the fit's depth_HiC column is RNA n_counts)
  genes         full 10x RNA matrix; gene-set score per 500 kb bin as mel_granger_ab.py

(1) INSULATION vs RHO
    per spot: genome-wide mean insulation and TAD contrast (sd of the insulation profile);
      Spearman with rho, partial on Hi-C reads (insulation from sparse contacts is depth-driven)
    per 500 kb bin (mean of its 100 kb insulation values): Spearman with rho, 200-rotation
      spatial null -> fraction of bins whose insulation tracks rho, next to the same for A/B

(2) GRANGER vs RHO LENGTH. For each axis and each window length f (fraction of the spots
    ordered by rho), REPS windows spread along the axis. Every window is SUBSAMPLED to the same
    N_SUB spots, then cut into NBIN equal-count bins -- so every series has the same length AND
    the same averaging depth; only the stretch of rho it spans changes. Per window: detrend both
    series on rho, F-test chromatin -> gene and gene -> chromatin at lags 1-3 (Bonferroni over
    lags x directions), circular-shift null on the same pairs, BH within window.
    Layers: A/B compartment and insulation, same gene sets.

Options (robustness of (2)):
  --natural   no subsampling: every window uses all its spots (longer rho = more data too)
  --deg K     detrend on a degree-K polynomial of rho instead of a line (shared curved trends
              across domains are the classic source of spurious Granger links)
  --granger-only   skip (1)
  --smooth-genes K  gene-set scores averaged over each spot's K spatial neighbours first, so the
              gene series is as smooth as the (Higashi-imputed) chromatin series. Granger favours
              the less noisy series as the "cause", so direction is only read after this.
  --direction  (3) DIRECTION TESTS on the full axis (all spots, cubic detrend): chromatin->gene vs
              gene->chromatin with rho forward and REVERSED, raw and noise-matched genes, plus the
              lag-1 autocorrelation of every binned series -> direction_tests.csv

  --ds NAME    Saturn fit to read (default mel_hic_fh_dr50_51: chip rows 50-51 dropped, which
              removes the y = 50 crease of the original mel_hic_fh fit)

Usage: python scripts/mel_hic_insulation_granger.py [--ds NAME] [--natural] [--deg 3] [--granger-only]
"""
import os, sys, json
import numpy as np, pandas as pd, scipy.sparse as sp, h5py
from scipy.stats import spearmanr, rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_real2 import H
from domain_annot import R
from saturn_granger_rho import pair_p, bh, detrend, lognorm
from continuous_marker_panels import hic_features_mel, mel_rna
from mel_granger_ab import bin_gene_map
from sproxseq_validate2 import rotated_fields

DS = sys.argv[sys.argv.index("--ds") + 1] if "--ds" in sys.argv else "mel_hic_fh_dr50_51"
OUT = f"{R}/{DS}_insulation_granger"
INS_H5 = f"{H}/SpHIC117/HIC_100kb/higashi/tmp/scTAD.hdf5"
INS_FL = f"{H}/SpHIC117/HIC_100kb/higashi/filelist.txt"
READS = (f"{H}/SpHIC117/HIC/higashi/read_count_all.npy", f"{H}/SpHIC117/HIC/higashi/filelist.txt")
FRACS, REPS, NBIN, NROT, SEED = (0.1, 0.2, 0.35, 0.5, 0.75, 1.0), 5, 30, 200, 0
AXES = ("HiC", "RNA")


def bc_of(path):
    return os.path.basename(path.strip()).replace(".contact.tsv", "")


def load_insulation(bc):
    fl = [bc_of(l) for l in open(INS_FL) if l.strip()]
    with h5py.File(INS_H5, "r") as f:
        g = f["insulation"]; b = g["bin"]
        ch = b["chrom"][:].astype(str); st = b["start"][:]
        keys = sorted([k for k in g.keys() if k.startswith("cell_")], key=lambda k: int(k.split("_")[1]))
        M = np.vstack([g[k][:] for k in keys]).astype(np.float32)
    rows = pd.Index([fl[int(k.split("_")[1])] for k in keys]).get_indexer(pd.Index(bc))
    assert (rows >= 0).all(), "fit spots missing from the insulation file"
    return M[rows], ch, st


def to_ab_bins(M, ch, st, names):
    """mean of the 100 kb insulation values inside each 500 kb A/B bin (NaN-aware)"""
    key = {n: j for j, n in enumerate(names)}
    col = np.array([key.get(f"{c}:{s - s % 500000}-{s - s % 500000 + 500000}", -1) for c, s in zip(ch, st)])
    ok = col >= 0
    A = sp.csr_matrix((np.ones(ok.sum()), (np.where(ok)[0], col[ok])), shape=(len(ch), len(names)))
    fin = np.isfinite(M); Mz = np.where(fin, M, 0)
    num = Mz @ A; den = fin.astype(np.float32) @ A
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(num / den)


def spear_mat(Y, x):
    ry = np.apply_along_axis(rankdata, 0, Y); rx = rankdata(x)
    ry = (ry - ry.mean(0)) / ry.std(0).clip(1e-9); rx = (rx - rx.mean()) / rx.std()
    return (ry * rx[:, None]).mean(0)


def partial_spearman(y, x, z):
    ry, rx, rz = rankdata(y), rankdata(x), rankdata(z)
    Z = np.c_[np.ones(len(y)), rz]
    e = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return float(np.corrcoef(e(ry), e(rx))[0, 1])


DEG, NATURAL, SMOOTH_K = 1, False, 0


def detrend_poly(V, r, deg):
    if deg <= 1: return detrend(V, r)
    z = (r - r.mean()) / r.std()
    A = np.column_stack([z ** k for k in range(deg + 1)])
    return V - A @ np.linalg.lstsq(A, V, rcond=None)[0]


def granger_window(idx, d, layers, rng):
    """idx: spots of the window (already subsampled). layers: {name: (C, S)} chromatin, gene score"""
    o = idx[np.argsort(d[idx])]; edges = np.array_split(o, NBIN)
    rb = np.array([d[e].mean() for e in edges])
    out = {}
    for name, (C, S) in layers.items():
        Cb = detrend_poly(np.vstack([C[e].mean(0) for e in edges]), rb, DEG)
        Sb = detrend_poly(np.vstack([S[e].mean(0) for e in edges]), rb, DEG)
        pf, pr, p0 = [], [], []
        for j in range(C.shape[1]):
            x, y = Cb[:, j], Sb[:, j]
            if x.std() < 1e-9 or y.std() < 1e-9:
                pf.append(np.nan); pr.append(np.nan); p0.append(np.nan); continue
            a, b, _ = pair_p(y, x); c, _, _ = pair_p(y, np.roll(x, rng.integers(5, NBIN - 5)))
            pf.append(a); pr.append(b); p0.append(c)
        pf, pr, p0 = map(np.array, (pf, pr, p0))
        out[name] = dict(c2g=int((bh(pf) < 0.05).sum()), g2c=int((bh(pr) < 0.05).sum()),
                         null=int((bh(p0) < 0.05).sum()), c2g_p05=float(np.nanmean(pf < 0.05)),
                         g2c_p05=float(np.nanmean(pr < 0.05)), null_p05=float(np.nanmean(p0 < 0.05)),
                         n=int(np.isfinite(pf).sum()))
    return out


def run(granger_only=False):
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ax = pd.read_csv(f"{R}/{DS}/axes.csv", index_col=0); bc = [str(b) for b in ax.index]
    xy = np.c_[ax.x.values, ax.y.values]
    rc = pd.Series(np.load(READS[0]), index=[bc_of(l) for l in open(READS[1]) if l.strip()]).reindex(bc).values

    V, names, _ = hic_features_mel(bc)                               # A/B, spots x 500 kb bins
    Ins, ich, ist = load_insulation(bc)                              # spots x 100 kb bins
    print(f"insulation {Ins.shape}, finite {np.isfinite(Ins).mean():.1%}", flush=True)
    I5 = to_ab_bins(Ins, ich, ist, names)                            # spots x 500 kb bins
    X, genes = mel_rna(bc); Xl = lognorm(sp.csr_matrix(X)).tocsc()
    b2g, b2n = bin_gene_map(names, genes)
    keep = [b for b in sorted(b2g) if np.isfinite(I5[:, b]).all()]
    S = np.column_stack([np.asarray(Xl[:, b2g[b]].mean(1)).ravel() for b in keep]).astype(np.float32)
    if SMOOTH_K:
        from scipy.spatial import cKDTree
        S = S[cKDTree(xy).query(xy, k=SMOOTH_K)[1]].mean(1)
    AB, IN = V[:, keep], I5[:, keep]
    print(f"{len(keep)} 500 kb bins with >=3 genes and complete insulation", flush=True)
    if not SMOOTH_K:                                                 # series cache read by the Fig 6 script
        np.savez_compressed(f"{OUT}/cache_series.npz", AB=AB.astype(np.float32), IN=IN.astype(np.float32), S=S,
                            names=names[keep], genes=np.array([",".join(b2n[b][:6]) for b in keep]), bc=np.array(bc))

    tag = ("_natural" if NATURAL else "") + (f"_deg{DEG}" if DEG > 1 else "") + (f"_smooth{SMOOTH_K}" if SMOOTH_K else "")
    summ = {}
    if granger_only: goto_granger = True
    else: goto_granger = False
    # ---- (1) insulation vs rho ----------------------------------------------------------
    ins_mean = np.nanmean(Ins, 1); ins_sd = np.nanstd(Ins, 1)
    summ = dict(reads_vs_ins_mean=float(spearmanr(rc, ins_mean)[0]), reads_vs_ins_sd=float(spearmanr(rc, ins_sd)[0]))
    per_bin = []
    for a in (() if goto_granger else AXES):
        d = ax[f"d_{a}"].values
        rot = rotated_fields(xy, d, NROT, rng)
        s = {}
        for nm, v in (("ins_mean", ins_mean), ("ins_sd", ins_sd)):
            r = spearmanr(d, v)[0]; nl = np.array([spearmanr(q, v)[0] for q in rot])
            s[nm] = dict(r=float(r), p_rot=float(((np.abs(nl) >= abs(r)).sum() + 1) / (NROT + 1)),
                         partial_reads=partial_spearman(v, d, rc))
        for nm, M in (("insulation", IN), ("AB", AB)):
            r = spear_mat(M, d); nl = np.array([spear_mat(M, q) for q in rot])
            p = ((np.abs(nl) >= np.abs(r)).sum(0) + 1) / (NROT + 1)
            s[f"{nm}_bins_sig"] = int((p < 0.05).sum()); s[f"{nm}_bins_n"] = int(len(p))
            per_bin.append(pd.DataFrame(dict(axis=a, layer=nm, bin=names[keep], r=r, p_rot=p)))
        summ[a] = s
        print(f"rho_{a}: {json.dumps(s)}", flush=True)
    if not goto_granger:
        pd.concat(per_bin).to_csv(f"{OUT}/bins_vs_rho.csv", index=False)
        np.save(f"{OUT}/spot_insulation_summary.npy", np.c_[ins_mean, ins_sd, rc])
        json.dump(summ, open(f"{OUT}/insulation_summary.json", "w"), indent=1)

    # ---- (2) Granger vs rho length --------------------------------------------------------
    N = len(bc); n_sub = int(round(min(FRACS) * N))
    layers = {"A/B": (AB, S), "insulation": (IN, S)}
    rows = []
    for a in AXES:
        d = ax[f"d_{a}"].values; order = np.argsort(d)
        for f in FRACS:
            w = int(round(f * N))
            starts = np.linspace(0, N - w, REPS).round().astype(int)
            for r_, s0 in enumerate(starts):
                win = order[s0:s0 + w]
                idx = np.sort(win) if NATURAL else np.sort(rng.choice(win, n_sub, replace=False))
                res = granger_window(idx, d, layers, rng)
                span = float(np.ptp(d[win]) / np.ptp(d))
                for L, v in res.items():
                    rows.append(dict(axis=a, frac=f, rep=r_, start=int(s0), rho_span=span, layer=L, **v))
                print(f"rho_{a} f={f:.2f} rep {r_}: " + " | ".join(
                    f"{L} c2g {v['c2g']} g2c {v['g2c']} null {v['null']}" for L, v in res.items()), flush=True)
    T = pd.DataFrame(rows); T.to_csv(f"{OUT}/granger_vs_length{tag}.csv", index=False)
    print(T.groupby(["axis", "layer", "frac"])[["c2g", "g2c", "null", "c2g_p05", "null_p05"]].mean().round(2).to_string())
    print(f"-> {OUT}")


def direction_tests(k_smooth=25):
    from scipy.spatial import cKDTree
    ax = pd.read_csv(f"{R}/{DS}/axes.csv", index_col=0); bc = [str(b) for b in ax.index]
    xy = np.c_[ax.x.values, ax.y.values]
    V, names, _ = hic_features_mel(bc); Ins, ich, ist = load_insulation(bc); I5 = to_ab_bins(Ins, ich, ist, names)
    X, genes = mel_rna(bc); Xl = lognorm(sp.csr_matrix(X)).tocsc(); b2g, _ = bin_gene_map(names, genes)
    keep = [b for b in sorted(b2g) if np.isfinite(I5[:, b]).all()]
    S = np.column_stack([np.asarray(Xl[:, b2g[b]].mean(1)).ravel() for b in keep]).astype(np.float32)
    Ss = S[cKDTree(xy).query(xy, k=k_smooth)[1]].mean(1)
    lag1 = lambda B: float(np.nanmedian(((B[1:] - B.mean(0)) * (B[:-1] - B.mean(0))).sum(0) / np.maximum(((B - B.mean(0)) ** 2).sum(0), 1e-12)))
    rows = []
    for a in AXES:
        for sgn, dname in ((1, "forward"), (-1, "reversed")):
            d = sgn * ax[f"d_{a}"].values; o = np.argsort(d); E = np.array_split(o, NBIN); rb = np.array([d[e].mean() for e in E])
            B = lambda C: detrend_poly(np.vstack([C[e].mean(0) for e in E]), rb, 3)
            for L, C in (("A/B", V[:, keep]), ("insulation", I5[:, keep])):
                Cb = B(C)
                for gname, G in (("raw genes", S), (f"{k_smooth}-NN genes", Ss)):
                    Gb = B(G)
                    pf, pr = np.array([pair_p(Gb[:, j], Cb[:, j])[:2] for j in range(len(keep))]).T
                    rows.append(dict(axis=a, direction=dname, layer=L, genes=gname, c2g=float(np.mean(pf < .05)),
                                     g2c=float(np.mean(pr < .05)), c2g_q=int((bh(pf) < .05).sum()), g2c_q=int((bh(pr) < .05).sum()),
                                     lag1_chromatin=lag1(Cb), lag1_genes=lag1(Gb)))
                    print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/direction_tests.csv", index=False)


if __name__ == "__main__":
    if "--direction" in sys.argv:
        direction_tests(); sys.exit()
    if "--smooth-genes" in sys.argv: SMOOTH_K = int(sys.argv[sys.argv.index("--smooth-genes") + 1])
    NATURAL = "--natural" in sys.argv
    if "--deg" in sys.argv: DEG = int(sys.argv[sys.argv.index("--deg") + 1])
    run(granger_only="--granger-only" in sys.argv)
