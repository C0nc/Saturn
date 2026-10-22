"""Post-fit annotation + marker screening for the spatial TCR fit (rcc_tcr_spatial).

The TCR modality differs from every other one in two ways that the pipeline has to
respect rather than paper over:

  MASK    only 382 of 520 bins carry enough local TCR mass to be observed, so every
          TCR-side statistic runs on the observed subset and the domain segmentation
          is computed there too.
  DEPTH   the median bin holds ONE receptor. A per-bin binomial slope therefore has
          almost no power, so clone enrichment is ALSO tested at the domain level,
          where a domain pools dozens of receptors, with Fisher's exact test. Both are
          reported, together with the receptor counts behind them, so a slope that the
          p<0.1 gate passed on two receptors is visible as such.

RNA keeps the Poisson path. Written:
  rcc_tcr_spatial_domains/domain_annotation.csv, domain_maps.*
  rcc_tcr_spatial_marker_functions/{gene,tcr_feature,clone}_*.csv
                                   genes_top8_functions.*, tcr_features_functions.*
"""
import os, sys, json
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad
from scipy.stats import fisher_exact
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
from prep_real2 import H
from lpc_gaston_fig4l import gaston_domains, boundaries, panel_space
from lpc_gene_isodepth import gaston_bins
from domain_annot import R, markers, gradient_count
from marker_function_panels import draw as draw_poisson
from piecewise import fit_gauss, fit_binomial
from gaston.segmented_fit import pw_linear_fit
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "svg.fonttype": "none"})

DS, KDOM, T_LRT, NB, BIN_UM = "rcc_tcr_spatial", 4, 0.1, 20, 200.0


def clone_counts(bc):
    """raw expanded-clone counts per bin, and the bin's TCR total"""
    a = ad.read_h5ad(f"{H}/slide_gotags_rcc/rcc_1_a.h5ad")
    xy = np.asarray(a.obsm["spatial"], float)
    trb = np.array([("" if str(v) in ("nan", "") else str(v).split()[0]) for v in a.obs["TRB_cdr3"]],
                   dtype=object)
    key = np.floor(xy / BIN_UM).astype(np.int64)
    code = key[:, 0] * 1_000_000 + key[:, 1]
    uniq, inv = np.unique(code, return_inverse=True)
    pos = pd.Index([str(u) for u in uniq]).get_indexer(pd.Index(bc))
    vc = pd.Series(trb[trb != ""]).value_counts()
    clones = [c for c in vc.index if vc[c] >= 3]
    C = np.zeros((len(uniq), len(clones)))
    for j, c in enumerate(clones):
        np.add.at(C[:, j], inv[trb == c], 1.0)
    tot = np.zeros(len(uniq)); np.add.at(tot, inv[trb != ""], 1.0)
    return C[pos], tot[pos], np.array(clones)


def run(n_rows=8, kdom=KDOM):
    OD, OM = f"{R}/{DS}_domains", f"{R}/{DS}_marker_functions"
    os.makedirs(OD, exist_ok=True); os.makedirs(OM, exist_ok=True)
    ax = pd.read_csv(f"{R}/{DS}/axes.csv", index_col=0)
    z = dict(np.load(f"{R}/targets_{DS}.npz", allow_pickle=True))
    bc = [str(b) for b in ax.index]
    dR, dT = ax["d_RNA"].values, ax["d_TCR"].values
    xy = np.c_[ax.x.values, ax.y.values]
    m = z["mask_TCR"].astype(bool)
    labR = gaston_domains(dR, z["Y_RNA"], L_=kdom)
    labT_obs = gaston_domains(dT[m], z["Y_TCR"][m], L_=kdom)     # segment on observed bins only
    labT = np.full(len(dT), -1, int); labT[m] = labT_obs
    bR = boundaries(dR, labR); bT = boundaries(dT[m], labT_obs)
    print(f"{DS}: RNA domains {np.bincount(labR).tolist()} | "
          f"TCR domains {np.bincount(labT_obs).tolist()} (on {int(m.sum())}/{len(m)} observed bins)",
          flush=True)

    a = ad.read_h5ad(f"{H}/slide_gotags_rcc/rcc_1_a.h5ad")
    xyb = np.asarray(a.obsm["spatial"], float)
    key = np.floor(xyb / BIN_UM).astype(np.int64)
    code = key[:, 0] * 1_000_000 + key[:, 1]
    uniq, inv = np.unique(code, return_inverse=True)
    G = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(uniq), len(inv)))
    pos = pd.Index([str(u) for u in uniq]).get_indexer(pd.Index(bc))
    X = sp.csr_matrix((G @ sp.csr_matrix(a.X))[pos])
    vn = np.array([str(v) for v in a.var_names])
    lib = np.asarray(X.sum(1)).ravel().clip(1)

    # ---- RNA screen (Poisson) -------------------------------------------------
    gR = gradient_count(X, lib, labR, dR)
    tot = np.asarray(X.sum(0)).ravel().astype(float)
    cand = np.argsort(-tot)[:1500]; cand = cand[tot[cand] >= 100]
    C = np.asarray(X[:, cand].todense())
    fitG = pw_linear_fit(np.c_[C, np.maximum(lib - C.sum(1), 0)], labR, dR, None, None,
                         idx_kept=np.arange(len(cand)), umi_threshold=0, t=T_LRT)["all_cell_types"]
    bi, _, kp = gaston_bins(dR, NB)
    r2g = []
    for i in range(len(cand)):
        y = np.array([C[bi == b, i].sum() for b in kp]) / np.maximum(
            np.array([lib[bi == b].sum() for b in kp]), 1)
        db = np.array([dR[bi == b].mean() for b in kp])
        lb = np.array([np.bincount(labR[bi == b], minlength=labR.max() + 1).argmax() for b in kp])
        pred = np.exp(fitG[1][i, lb] + fitG[0][i, lb] * db)
        r2g.append(1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
    sg = pd.DataFrame(dict(feature=vn[cand], counts=tot[cand].astype(int),
                           n_sloped_domains=(np.abs(fitG[0]) > 1e-9).sum(1), R2=np.round(r2g, 3),
                           slopes=[np.round(fitG[0][i], 2).tolist() for i in range(len(cand))]))
    sg = sg.sort_values(["n_sloped_domains", "R2"], ascending=False)
    sg.to_csv(f"{OM}/gene_screen.csv", index=False)
    draw_poisson(DS, "RNA", sg[sg.n_sloped_domains >= 2].head(n_rows), X, lib, labR, dR, xy, bR, vn, OM)

    # ---- TCR target features (Gaussian: they are logit fields, not counts) -----
    Yt = np.asarray(z["Y_TCR"], float)[m]
    fnames = [k.replace("ann_frac_", "clone ") for k in z if k.startswith("ann_frac_")] + ["Gini-Simpson"]
    fnames = fnames[:Yt.shape[1]]
    slope, inter = fit_gauss(Yt, labT_obs, dT[m], t=T_LRT)
    sf = pd.DataFrame(dict(feature=fnames, n_sloped_domains=(np.abs(slope) > 1e-12).sum(1),
                           slopes=[np.round(s, 3).tolist() for s in slope]))
    sf.to_csv(f"{OM}/tcr_feature_screen.csv", index=False)
    print(sf.to_string(index=False), flush=True)

    # ---- clone-level enrichment: bin-level binomial AND domain-level Fisher ----
    Cc, tot_t, clones = clone_counts(bc)
    rows = []
    sl, it = fit_binomial(Cc[m], np.repeat(tot_t[m][:, None], Cc.shape[1], 1), labT_obs, dT[m], t=T_LRT)
    for j, c in enumerate(clones):
        for k in range(labT_obs.max() + 1):
            ink = labT_obs == k
            a_ = int(Cc[m][ink, j].sum()); b_ = int(tot_t[m][ink].sum() - a_)
            c_ = int(Cc[m][~ink, j].sum()); d_ = int(tot_t[m][~ink].sum() - c_)
            odds, p = fisher_exact([[a_, b_], [c_, d_]]) if (a_ + c_) else (np.nan, 1.0)
            rows.append(dict(clone=c[:14], domain=k, n_bins=int(ink.sum()),
                             receptors_in_domain=a_, receptors_total=int(Cc[m][:, j].sum()),
                             frac_in_domain=round(a_ / max(a_ + b_, 1), 3),
                             frac_elsewhere=round(c_ / max(c_ + d_, 1), 3),
                             fisher_p=round(float(p), 4), binom_slope=round(float(sl[j, k]), 2)))
    cl = pd.DataFrame(rows).sort_values("fisher_p")
    cl.to_csv(f"{OM}/clone_domain_enrichment.csv", index=False)
    print("\nclone x domain enrichment (top 10 by Fisher p):", flush=True)
    print(cl.head(10).to_string(index=False), flush=True)

    # ---- domain annotation ----------------------------------------------------
    gT = (np.abs(slope) > 1e-12).sum(0)
    ann = []
    for mod, lab, d_, idx_ in [("RNA", labR, dR, np.ones(len(dR), bool)), ("TCR", labT, dT, m)]:
        for k in range(kdom):
            ina = (lab == k) & idx_
            if not ina.any():
                continue
            row = dict(modality=mod, domain=k, n_bins=int(ina.sum()),
                       isodepth=f"[{d_[ina].min():.2f}, {d_[ina].max():.2f}]",
                       n_beads=round(float(np.asarray(z['ann_n_beads'], float)[ina].mean()), 1),
                       tcr_mass=round(float(np.asarray(z['ann_tcr_mass'], float)[ina].mean()), 2),
                       receptors=int(tot_t[ina].sum()),
                       gini=round(float(np.asarray(z['ann_gini_simpson'], float)[ina].mean()), 3))
            if mod == "RNA":
                row["gradient_feats"] = f"{int(gR[k])}/500"
                row["markers"] = "; ".join(f"{g}({lf:+.1f})" for g, lf, _ in markers(X, lib, lab, vn, k))
            else:
                row["gradient_feats"] = f"{int(gT[k])}/{Yt.shape[1]}"
                fr = Cc[ina].sum(0) / max(tot_t[ina].sum(), 1)
                o = np.argsort(-fr)[:4]
                row["markers"] = "; ".join(f"{clones[i][:12]}({fr[i]:.2f})" for i in o if fr[i] > 0)
            ann.append(row)
    A = pd.DataFrame(ann); A.to_csv(f"{OD}/domain_annotation.csv", index=False)
    for mod in ["RNA", "TCR"]:
        print(f"\n=== {DS} {mod} domains ===")
        print(A[A.modality == mod].drop(columns=["modality"]).to_string(index=False))

    # ---- maps + the three TCR feature functions -------------------------------
    fig, AX = plt.subplots(1, 2, figsize=(11, 4.6))
    panel_space(AX[0], xy, labR, "", None, contour_d=dR, bnds=bR, discrete=True)
    AX[0].set_title(f"{DS}: RNA domains (K={kdom}, all {len(dR)} bins)", fontsize=7, loc="left")
    AX[1].scatter(xy[~m, 0], xy[~m, 1], c="#e8e8e8", s=4, linewidths=0, label="no TCR mass")
    panel_space(AX[1], xy[m], labT_obs, "", None, contour_d=dT[m], bnds=bT, discrete=True)
    AX[1].set_title(f"{DS}: TCR domains (K={kdom}, {int(m.sum())} observed bins;\ngrey = unobserved)",
                    fontsize=7, loc="left")
    fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(f"{OD}/domain_maps.{e}", dpi=300 if e == "png" else None)
    plt.close(fig)

    nf = Yt.shape[1]
    fig, AX = plt.subplots(nf, 5, figsize=(18.5, 3.0 * nf), squeeze=False)
    bi2, _, kp2 = gaston_bins(dT[m], NB)
    for row in range(nf):
        y = Yt[:, row]
        ob = np.array([y[bi2 == b].mean() for b in kp2])
        db = np.array([dT[m][bi2 == b].mean() for b in kp2])
        lb = np.array([np.bincount(labT_obs[bi2 == b], minlength=kdom).argmax() for b in kp2])
        pred_sp = inter[row][labT_obs] + slope[row][labT_obs] * dT[m]
        panel_space(AX[row, 0], xy[m], dT[m], "rho_TCR (Saturn isodepth)", "coolwarm",
                    contour_d=dT[m], bnds=bT)
        panel_space(AX[row, 1], xy[m], labT_obs, f"TCR domains (K={kdom})", None,
                    contour_d=dT[m], bnds=bT, discrete=True)
        panel_space(AX[row, 2], xy[m], y, f"{fnames[row]}\nRAW (target)", "coolwarm",
                    contour_d=dT[m], bnds=bT)
        panel_space(AX[row, 3], xy[m], pred_sp, f"{fnames[row]}\nFITTED", "coolwarm",
                    contour_d=dT[m], bnds=bT)
        a_ = AX[row, 4]
        for k in range(kdom):
            sel = lb == k
            a_.plot(db[sel], ob[sel], "o", ms=3, color=f"C{k}")
            if sel.sum():
                xs = np.linspace(db[sel].min(), db[sel].max(), 20)
                a_.plot(xs, inter[row][k] + slope[row][k] * xs, color=f"C{k}", lw=1.6)
        a_.set_xlabel("isodepth d_TCR"); a_.set_ylabel("scaled target value")
        a_.set_title(f"{fnames[row]}: sloped in {int((np.abs(slope[row]) > 1e-12).sum())}/{kdom} domains")
        a_.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{DS}: TCR target features along d_TCR (Gaussian piecewise fit, F-gate p<{T_LRT}); "
                 f"observed bins only", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / (3.0 * nf)))
    for e in ("png", "svg"):
        fig.savefig(f"{OM}/tcr_features_functions.{e}", dpi=250 if e == "png" else None)
    plt.close(fig)
    print(f"\nwritten {OD} and {OM}", flush=True)
    return A, cl


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
