"""Does cis gene-peak gradient agreement live ALONG the isodepth, or ACROSS it?

grad(rho) comes from autograd through the Saturn isodepth net (GASTON's method). At each
spot it defines an orthonormal frame:

    u = grad(rho)/|grad(rho)|      along the axis   (perpendicular to the isodepth contour)
    v = rot90(u)                   across the axis  (along the contour)

Every feature's spatial gradient is projected onto it:

    a = <grad f, u>   how fast f changes ALONG  rho   -- the part Saturn models
    b = <grad f, v>   how fast f changes ACROSS rho   -- the part Saturn discards

For each (gene, cis peak, domain) the gene and peak are then correlated over the spots of
that domain, separately in a and in b. Distance to TSS is the calibration, as before.

If cis agreement is only in `a`, the isodepth captures it. If it is also in `b`, cis
regulation has structure the 1-D axis cannot represent.

Usage: python scripts/saturn_cis_gradient_decomp.py <dataset> [n_genes]
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/saturn_repo/src")
from lpc_gaston_fig4l import gaston_domains
from domain_k import kdom
from domain_annot import CFG, R
from saturn_cis_function_pairs import genes_and_peaks, WIN, MAXPK, KNN
from saturn_rho_gradient import rho_gradient
from saturn import cis as SCIS, gradient as SG


def run(ds, n_genes=1500, per_axis_scaling=False):
    """isodepth gradient by the chain rule through zscore_coords (Fig 2i); per_axis_scaling=True (--per-axis) is the
    pre-2026-10-22 run, which divided by the per-axis coordinate std and writes cis_gradient_decomp_per_axis.csv"""
    OUT = f"{R}/{ds}_cis_pairs"; os.makedirs(OUT, exist_ok=True)
    cfg = CFG[ds]
    ax = pd.read_csv(f"{R}/{ds}/axes.csv", index_col=0); bc = [str(b) for b in ax.index]
    z_ = dict(np.load(f"{R}/targets_{ds}.npz", allow_pickle=True))
    grho, xy = rho_gradient(ds, per_axis_scaling=per_axis_scaling); xy = xy.astype(np.float32)
    Xr, Xa, gn, pn, tp, kg, tss = genes_and_peaks(ds, n_genes, cfg, bc)
    _, pm = SCIS.peak_positions(pn)
    cis = SCIS.cis_pairs(kg, gn, pn, tp, tss, window=WIN, max_peaks=MAXPK)
    pairs = [(gi, s_) for gi, pk in cis.items() for s_ in pk]
    gi_a = np.array([p[0] for p in pairs]); pk_a = np.array([p[1] for p in pairs]); need = np.unique(pk_a)
    print(f"   {len(pairs)} cis pairs; gradients for {len(kg)} genes, {len(need)} peaks", flush=True)
    G = SCIS.smoothed_logcpm(Xr, kg, xy, KNN); P = SCIS.smoothed_logcpm(Xa, need, xy, KNN)
    gG = SG.spatial_gradients(G, xy); gP = SG.spatial_gradients(P, xy)
    gpos = {v: i for i, v in enumerate(kg)}; ppos = {v: i for i, v in enumerate(need)}
    ga = np.array([gpos[v] for v in gi_a]); pb = np.array([ppos[v] for v in pk_a])
    kb = (pm[pk_a] - np.array([tss[gn[i]][1] for i in gi_a])) / 1e3
    rows = []
    for axis in ["RNA", "ATAC"]:
        d = ax[f"d_{axis}"].values; lab = gaston_domains(d, z_[f"Y_{axis}"], L_=kdom(ds, axis))
        res, frac = SCIS.along_across_corr(gG, gP, grho[axis], ga, pb, lab)
        print(f"   d_{axis}: {100*frac:.0f}% of gene gradient ENERGY lies along rho", flush=True)
        for k, (ra, rb) in res.items():
            rows.append(pd.DataFrame(dict(dataset=ds, axis=axis, domain=k, gene=gn[gi_a], peak=pn[pk_a], kb=kb,
                                          r_along=np.round(ra, 4), r_across=np.round(rb, 4))))
    T = pd.concat(rows, ignore_index=True)
    T.to_csv(f"{OUT}/cis_gradient_decomp{'_per_axis' if per_axis_scaling else ''}.csv", index=False)
    print(f"-> {OUT}/cis_gradient_decomp.csv ({len(T)} rows)")
    return T


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1500, per_axis_scaling="--per-axis" in sys.argv)
