"""Top marker FUNCTIONS for each modality, screened independently and in parallel.

Unlike gene_function_panels.py, the ATAC side here is NOT restricted to the cis window
of any gene: peaks (or tiles) are screened exactly the way genes are -- GASTON's
piecewise Poisson fit along their OWN isodepth, keeping features sloped in at least two
domains -- and the best ones are reported on their own terms. Whatever gene a top peak
sits near is read off afterwards, as annotation, not as a selection criterion.

Two figures per dataset, each 8 rows x 5 columns, one row per feature:
    rho (Saturn isodepth) | domains | raw | fitted | fitted function
  genes_top8_functions.*  screened on d_RNA  against the RNA domains
  peaks_top8_functions.*  screened on d_ATAC against the ATAC domains

Usage: python scripts/marker_function_panels.py <dataset> [n_rows] [n_jobs]
"""
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
from joblib import Parallel, delayed
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
from lpc_gaston_fig4l import gaston_domains, boundaries, panel_space, panel_function, peak_class, L
from lpc_gene_isodepth import gaston_bins
from domain_annot import CFG, R, nearest_gene
from gaston.segmented_fit import pw_linear_fit
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "svg.fonttype": "none"})

NB = {"RNA": 30, "ATAC": 20}
KDOM, T_LRT = 4, 0.1
NCAND = {"RNA": 1500, "ATAC": 6000}       # features screened, ranked by total counts
MIN_COUNT = {"RNA": 200, "ATAC": 100}
CHUNK = 250


def _chunk(C, rest, lab, d, binidx, keepbin):
    """piecewise Poisson fit of one chunk of features; returns slopes and binned R2"""
    fit = pw_linear_fit(np.c_[C, rest], lab, d, None, None,
                        idx_kept=np.arange(C.shape[1]), umi_threshold=0, t=T_LRT)["all_cell_types"]
    slope, inter = fit[0], fit[1]
    db = np.array([d[binidx == b].mean() for b in keepbin])
    lb = np.array([np.bincount(lab[binidx == b], minlength=lab.max() + 1).argmax() for b in keepbin])
    r2 = np.empty(C.shape[1])
    lib = C.sum(1) + rest
    for i in range(C.shape[1]):
        y = np.array([C[binidx == b, i].sum() for b in keepbin]) / np.maximum(
            np.array([lib[binidx == b].sum() for b in keepbin]), 1)
        pred = np.exp(inter[i, lb] + slope[i, lb] * db)
        r2[i] = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    return slope, r2


def screen(Mat, lib, lab, d, feats, mod, n_jobs=8):
    """rank every feature by (number of sloped domains, binned R2), in parallel"""
    tot = np.asarray(Mat.sum(0)).ravel().astype(float)
    cand = np.argsort(-tot)[:NCAND[mod]]
    cand = cand[tot[cand] >= MIN_COUNT[mod]]
    binidx, _, keepbin = gaston_bins(d, NB[mod])
    parts = [cand[i:i + CHUNK] for i in range(0, len(cand), CHUNK)]
    res = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_chunk)(np.asarray(Mat[:, p].todense()).astype(float),
                        np.maximum(lib - np.asarray(Mat[:, p].sum(1)).ravel(), 0),
                        lab, d, binidx, keepbin) for p in parts)
    slope = np.vstack([r[0] for r in res]); r2 = np.concatenate([r[1] for r in res])
    df = pd.DataFrame(dict(feature=feats[cand], counts=tot[cand].astype(int),
                           n_sloped_domains=(np.abs(slope) > 1e-9).sum(1), R2=np.round(r2, 3),
                           slopes=[np.round(s, 2).tolist() for s in slope]))
    return df.sort_values(["n_sloped_domains", "R2"], ascending=False), cand


def draw(dataset, mod, top, Mat, lib, lab, d, xy, bnds, feats, OUT, note=None):
    n = len(top)
    fig, AX = plt.subplots(n, 5, figsize=(18.5, 3.0 * n), squeeze=False)
    for row, (_, rec) in enumerate(top.iterrows()):
        j = int(np.where(feats == rec.feature)[0][0])
        c_ = np.asarray(Mat[:, j].todense()).ravel().astype(float)
        fit = pw_linear_fit(np.c_[c_, np.maximum(lib - c_, 0)], lab, d, None, None,
                            idx_kept=np.array([0]), umi_threshold=0, t=T_LRT)["all_cell_types"]
        nm = rec.feature if note is None else f"{rec.feature}\n{note.get(rec.feature, '')}"
        panel_space(AX[row, 0], xy, d, f"rho_{mod} (Saturn isodepth)", "coolwarm", contour_d=d, bnds=bnds)
        panel_space(AX[row, 1], xy, lab, f"{mod} domains (K={lab.max() + 1})", None,
                    contour_d=d, bnds=bnds, discrete=True)
        panel_space(AX[row, 2], xy, L(c_ / lib), f"{nm}\nRAW", "magma", contour_d=d, bnds=bnds)
        panel_space(AX[row, 3], xy, L(np.exp(fit[1][0, lab] + fit[0][0, lab] * d)), f"{nm}\nFITTED",
                    "magma", contour_d=d, bnds=bnds)
        panel_function(AX[row, 4], c_, lib, d, lab, NB[mod], f"isodepth d_{mod}",
                       "log(rate x 1e6 + 1)",
                       f"{rec.feature}: {int(rec.counts)} counts, sloped in "
                       f"{int(rec.n_sloped_domains)}/{lab.max() + 1} domains, R2 {rec.R2:.2f}",
                       gfit=fit, grow=0)
        a_ = AX[row, 4]
        obs = [ln.get_ydata() for ln in a_.get_lines() if ln.get_linestyle() == "None"]
        obs += [c.get_offsets()[:, 1] for c in a_.collections if len(c.get_offsets())]
        if obs:
            v = np.concatenate([np.asarray(o, float).ravel() for o in obs])
            v = v[np.isfinite(v)]
            if len(v):
                pad = 0.1 * (v.max() - v.min() + 1e-9)
                a_.set_ylim(v.min() - pad, v.max() + pad)
    what = "genes" if mod == "RNA" else "peaks / tiles"
    fig.suptitle(f"{dataset}: top {n} {what} by multi-domain {mod} function "
                 f"(sloped in >=2 domains, ranked by R2) -- screened on d_{mod} alone, no cis constraint",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / (3.0 * n)))
    tag = "genes" if mod == "RNA" else "peaks"
    for e in ("png", "svg"):
        fig.savefig(f"{OUT}/{tag}_top{n}_functions.{e}", dpi=250 if e == "png" else None)
    plt.close(fig)
    print(f"   wrote {tag}_top{n}_functions", flush=True)


def run(dataset, n_rows=8, n_jobs=8, kdom=KDOM):
    cfg = CFG[dataset]
    OUT = f"{R}/{dataset}_marker_functions"; os.makedirs(OUT, exist_ok=True)
    ax = pd.read_csv(f"{R}/{dataset}/axes.csv", index_col=0)
    z = dict(np.load(f"{R}/targets_{dataset}.npz", allow_pickle=True))
    dR, dA = ax["d_RNA"].values, ax["d_ATAC"].values
    xy = np.c_[ax.x.values, ax.y.values]
    labR = gaston_domains(dR, z["Y_RNA"], L_=kdom); labA = gaston_domains(dA, z["Y_ATAC"], L_=kdom)
    bR, bA = boundaries(dR, labR), boundaries(dA, labA)
    idx = pd.Index([str(b) for b in ax.index])
    r = cfg["rna"]()[idx].copy(); a = cfg["atac"]()[idx].copy()
    X = sp.csc_matrix(r.X); vn = np.array([str(v) for v in r.var_names])
    lib = np.asarray(X.sum(1)).ravel().clip(1)
    A_ = sp.csc_matrix(a.X); pkn = np.array([str(v) for v in a.var_names])
    alib = np.asarray(A_.sum(1)).ravel().clip(1)
    print(f"{dataset}: RNA domains {np.bincount(labR).tolist()} | ATAC domains {np.bincount(labA).tolist()}",
          flush=True)

    fg = f"{OUT}/gene_screen.csv"
    if os.path.exists(fg) and os.environ.get("RESCREEN", "0") != "1":
        sg = pd.read_csv(fg); print("   RNA screen reused from cache", flush=True)
    else:
        sg, _ = screen(X, lib, labR, dR, vn, "RNA", n_jobs); sg.to_csv(fg, index=False)
    print(f"   RNA: {len(sg)} screened, {(sg.n_sloped_domains >= 2).sum()} sloped in >=2 domains", flush=True)

    fp = f"{OUT}/peak_screen.csv"
    if os.path.exists(fp) and os.environ.get("RESCREEN", "0") != "1":
        sp_ = pd.read_csv(fp); print("   ATAC screen reused from cache", flush=True)
        near_g = near_d = None
    else:
        sp_, _ = screen(A_, alib, labA, dA, pkn, "ATAC", n_jobs)
        near_g, near_d = nearest_gene(pkn, cfg["gtf"])
    pos = {p: i for i, p in enumerate(pkn)}
    if near_g is not None:
        sp_["nearest_gene"] = [near_g[pos[p]] for p in sp_.feature]
        sp_["kb_to_TSS"] = [round(near_d[pos[p]] / 1000, 1) for p in sp_.feature]
        sp_["class"] = [peak_class(near_d[pos[p]]) for p in sp_.feature]
        sp_.to_csv(fp, index=False)
    print(f"   ATAC: {len(sp_)} screened, {(sp_.n_sloped_domains >= 2).sum()} sloped in >=2 domains", flush=True)

    top_g = sg[sg.n_sloped_domains >= 2].head(n_rows)
    top_p = sp_[sp_.n_sloped_domains >= 2].head(n_rows)
    draw(dataset, "RNA", top_g, X, lib, labR, dR, xy, bR, vn, OUT)
    note = {p: f"{g} {k:+.1f} kb ({c})" for p, g, k, c in
            zip(top_p.feature, top_p.nearest_gene, top_p.kb_to_TSS, top_p["class"])}
    draw(dataset, "ATAC", top_p, A_, alib, labA, dA, xy, bA, pkn, OUT, note=note)
    print("written", OUT, flush=True)
    return sg, sp_


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8,
        int(sys.argv[3]) if len(sys.argv) > 3 else 8)
