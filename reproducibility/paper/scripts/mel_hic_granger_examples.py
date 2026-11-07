"""Per-bin Granger tests along each Saturn axis, for Fig 6 examples (noise-matched genes, as Fig 6 d-f).

For every 500 kb bin (A/B compartment and insulation layer; 2,558 bins with >= 3 genes) and both axes, all spots, rho
forward and reversed: spots ordered by rho -> 30 equal-count windows -> window means of the chromatin value and of the
bin's gene score (mean log-expression of its genes, averaged over each spot's 25 spatial neighbours = noise-matched)
-> cubic detrend on rho -> Granger both directions, lags 1-3, Bonferroni over lags x directions (saturn_granger_rho.pair_p)
-> BH over bins per (axis, layer, direction of rho).
-> results/real_saturn/<DS>_insulation_granger/granger_per_bin.csv, granger_series.npz (binned series per axis)
Usage: python scripts/mel_hic_granger_examples.py [--ds mel_hic_abins_dr50_51_rnadr]
"""
import os, sys
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saturn_granger_rho import pair_p, bh
from domain_annot import R

DS = sys.argv[sys.argv.index("--ds") + 1] if "--ds" in sys.argv else "mel_hic_abins_dr50_51_rnadr"
G = f"{R}/{DS}_insulation_granger"; NBIN, K = 30, 25


def detrend3(V, r):
    z = (r - r.mean()) / r.std(); A = np.column_stack([z ** k for k in range(4)])
    return V - A @ np.linalg.lstsq(A, V, rcond=None)[0]


def main():
    C = np.load(f"{G}/cache_series.npz", allow_pickle=True); bc = list(C["bc"].astype(str))
    ax = pd.read_csv(f"{R}/{DS}/axes.csv", index_col=0).loc[bc]; xy = np.c_[ax.x.values, ax.y.values]
    S = np.asarray(C["S"], float); Ss = S[cKDTree(xy).query(xy, k=K)[1]].mean(1)
    names, genes = C["names"].astype(str), C["genes"].astype(str)
    rows, series = [], {}
    for a in ("RNA", "HiC"):
        for sgn, dname in ((1, "forward"), (-1, "reversed")):
            d = sgn * ax[f"d_{a}"].values; E = np.array_split(np.argsort(d), NBIN); rb = np.array([d[e].mean() for e in E])
            Gb = detrend3(np.vstack([Ss[e].mean(0) for e in E]), rb)
            if dname == "forward": series[f"{a}|genes"] = Gb; series[f"{a}|rho"] = rb
            for L, key in (("A/B", "AB"), ("insulation", "IN")):
                Cb = detrend3(np.vstack([np.asarray(C[key], float)[e].mean(0) for e in E]), rb)
                if dname == "forward": series[f"{a}|{key}"] = Cb
                P = np.array([pair_p(Gb[:, j], Cb[:, j]) for j in range(Cb.shape[1])], float)
                qf, qr = bh(P[:, 0]), bh(P[:, 1])
                rows.append(pd.DataFrame(dict(axis=a, rho=dname, layer=L, bin=names, genes=genes, p_c2g=P[:, 0], p_g2c=P[:, 1],
                                              lag_c2g=P[:, 2].astype(int), q_c2g=qf, q_g2c=qr)))
                print(f"{a} {dname} {L}: q<0.05 c2g {int((qf < .05).sum())}, g2c {int((qr < .05).sum())}", flush=True)
    T = pd.concat(rows, ignore_index=True); T.to_csv(f"{G}/granger_per_bin.csv", index=False)
    np.savez_compressed(f"{G}/granger_series.npz", **series, names=names, genes=genes)
    print("->", G)


if __name__ == "__main__":
    main()
