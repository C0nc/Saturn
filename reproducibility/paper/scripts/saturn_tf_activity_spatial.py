"""Spatial TF activity with the Saturn domain contours drawn on top.

The rho-binned version answers "TF activity as a function of isodepth". This one keeps
the same chromVAR machinery but computes the deviation z PER SPOT, so the map is a real
spatial field rather than a banded re-drawing of rho. Each TF then gets:

  spot z (raw) | spot z (kNN-smoothed) + d_RNA contours | fitted f(d_RNA)
               | spot z (smoothed) + d_ATAC contours    | fitted f(d_ATAC)

so you can see directly how much of the spatial TF field each isodepth explains.

Per-spot deviations reuse ONE permutation set across all TFs (every TF's motif-hit set
is the same size by construction -- the 99th percentile), which turns the chromVAR null
into a single matmul instead of NPERM x NTF column means.

Usage: python scripts/saturn_tf_activity_spatial.py <dataset> [n_tf]
"""
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "<workspace>/Hole/gaston_mix_mm/scripts")
sys.path.insert(0, "<workspace>/saturn_repo/src")
from prep_real2 import H
from lpc_gaston_fig4l import gaston_domains, boundaries
from domain_annot import CFG, R
from piecewise import fit_gauss
from saturn_tf_activity_rho import (PANEL, motif_hits, peak_gc, background_sets,
                                    deviations_matched, TOPK, NBIN, KDOM, T_LRT)
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "svg.fonttype": "none"})
NPERM_SPOT, KNN = 100, 15
from saturn import tfactivity as TA
from saturn.mechanism import smooth as _smooth


def dev_fast(M, hits, nperm=NPERM_SPOT, seed=0):
    """chromVAR z with one permutation set shared by every TF (saturn.tfactivity.deviations_fast)"""
    return TA.deviations_fast(M, hits, nperm=nperm, seed=seed)


def knn_smooth(V, xy, k=KNN):
    return _smooth(V, xy, k)


def pmap(a, xy, v, title, d=None, bnds=(), cmap="coolwarm"):
    lo, hi = np.nanpercentile(v, 2), np.nanpercentile(v, 98)
    m = max(abs(lo), abs(hi))
    sc = a.scatter(xy[:, 0], xy[:, 1], c=v, cmap=cmap, s=4, linewidths=0, vmin=-m, vmax=m)
    plt.colorbar(sc, ax=a, fraction=.046)
    if d is not None and len(bnds):
        a.tricontour(xy[:, 0], xy[:, 1], d, levels=sorted(bnds), colors="k", linewidths=1.3, alpha=.85)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); a.set_title(title, fontsize=7)


def run(ds, n_tf=8):
    OUT = f"{R}/{ds}_tf_activity"
    cfg = CFG[ds]
    ax = pd.read_csv(f"{R}/{ds}/axes.csv", index_col=0); bc = [str(b) for b in ax.index]
    xy = np.c_[ax.x.values, ax.y.values]
    z = dict(np.load(f"{R}/targets_{ds}.npz", allow_pickle=True))
    atac = cfg["atac"](); ai = {str(b): i for i, b in enumerate(atac.obs_names)}
    A = sp.csr_matrix(atac.X)[[ai[b] for b in bc]]
    peaks = np.array([str(p) for p in atac.var_names])
    tot = np.asarray(A.sum(0)).ravel().astype(float)
    keep = np.argsort(-tot)[:TOPK]; keep.sort()
    hits, tfn, valid = motif_hits(ds, peaks, keep)
    As = A[:, valid]
    M = TA.log_accessibility(As, dtype=np.float32)
    print(f"   spot-level matrix {M.shape}; computing chromVAR z per spot", flush=True)
    gc = peak_gc(ds, peaks, valid)
    acc = np.asarray(As.sum(0)).ravel().astype(float)
    Zs = deviations_matched(M, hits, background_sets(gc, acc, hits))
    Zsm = np.column_stack([knn_smooth(Zs[:, j], xy) for j in range(Zs.shape[1])])
    np.savez_compressed(f"{OUT}/tf_z_per_spot.npz", z=Zs, z_smooth=Zsm,
                        tfs=np.array(tfn, object), barcodes=np.array(bc, object))

    fitted, doms, bnds_, dd = {}, {}, {}, {}
    for axis in ["RNA", "ATAC"]:
        d = ax[f"d_{axis}"].values
        lab = gaston_domains(d, z[f"Y_{axis}"], L_=KDOM)
        groups, db, lb = TA.rho_bins(d, lab, NBIN)
        Zb = np.vstack([Zs[g].mean(0) for g in groups])
        sl, it = fit_gauss(np.nan_to_num(Zb), lb, db, t=T_LRT)
        fitted[axis] = np.column_stack([it[j][lab] + sl[j][lab] * d for j in range(len(tfn))])
        doms[axis], bnds_[axis], dd[axis] = lab, boundaries(d, lab), d

    F = pd.read_csv(f"{OUT}/tf_fits.csv")
    best = F.groupby("tf").R2.max().sort_values(ascending=False).head(n_tf).index.tolist()
    fig, AX = plt.subplots(len(best), 5, figsize=(23, 3.9 * len(best)), squeeze=False)
    fig.subplots_adjust(top=.93)
    for r, tf in enumerate(best):
        j = tfn.index(tf)
        ln = [k for k, v in PANEL[ds].items() if tf in v][0]
        pmap(AX[r, 0], xy, Zs[:, j], f"{tf} ({ln})\nchromVAR z per SPOT (raw)")
        for c0, axis in ((1, "RNA"), (3, "ATAC")):
            cc = np.corrcoef(Zsm[:, j], fitted[axis][:, j])[0, 1]
            pmap(AX[r, c0], xy, Zsm[:, j], f"{tf}: spot z, kNN{KNN}-smoothed\n"
                 f"d_{axis} domain contours", d=dd[axis], bnds=bnds_[axis])
            pmap(AX[r, c0 + 1], xy, fitted[axis][:, j],
                 f"{tf}: FITTED f(d_{axis})\nr(smoothed z, fitted) = {cc:+.2f}",
                 d=dd[axis], bnds=bnds_[axis])
    fig.suptitle(f"{ds}: spatial TF activity (chromVAR z per spot) with Saturn domain contours, "
                 f"and the fitted function of each isodepth", fontsize=12, y=.995)
    fig.tight_layout(rect=(0, 0, 1, 1 - .95 / (3.9 * len(best))))
    for e in ("png", "svg"):
        fig.savefig(f"{OUT}/tf_activity_spatial.{e}", dpi=200 if e == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {OUT}/tf_activity_spatial.png  (+ tf_z_per_spot.npz)")


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
