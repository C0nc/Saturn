"""Figure 4 — MERFISH isoform usage: spatial isoform heterogeneity is mostly cell-type composition,
with a small, credible core of within-cell-type regulation in the oligodendrocyte lineage.

RT&T-AMP-MERFISH (Cohen ... Zhuang, Cell 2026), section 5; Saturn fit `rtt_merfish_s5`
(50 um bins; RNA = gene GLM-PCA, Isoform = within-gene usage). Every segmented cell is mapped to
its bin, so it carries both rho and both domains plus its own cell-type label.

  a  the section, cells coloured by class
  b  fit quality: R2 full / own / cross
  c  isoform switches split into composition vs within-cell-type change (exact Kitagawa split)
  d  rho_RNA, RNA domains, rho_Isoform, Isoform domains (labels from anatomy + composition)
  e  Tcf7l2, Fezf2 on rho_RNA; usage of Kcnab3 and Zfp950 isoforms on rho_Isoform
  f  within-type examples: switch-isoform usage per domain in each cell class (bootstrap 95% CI)
  g  the same switches as a continuous function of rho inside each cell class: cells binned by
     per-class rho sextiles, usage with bootstrap 95% CI; per-class binomial slope per SD of rho, tested against
     the 200-rotation spatial null (rho rotated at bin level, re-read by every cell)
  h  usage change vs total-expression change per gene

Supplement (Saturn/final_fig4/supp): the spillover control for the glial within-type cases, and the
Tcf7l2 usage maps in oligodendrocytes vs neurons that g quantifies.

Mechanism (TSS / 3' end / splicing) is NOT shown: 99.7% of measured isoform pairs differ at their
ends by probe design and the switches match that background (p = 0.62); it goes to Supp.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/_style")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/saturn_repo/src")       # Saturn package (saturn_repo)
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/fig2")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
from paper_style import apply, save_dual, MODALITY, DOMAIN, CELLCLASS, CMAP
import fig2_lpc as F2

H = "/mnt/HDD2B/Users/yancui"
R = Path(f"{H}/Hole/gaston_mix_mm/results/real_saturn")
DS = "rtt_merfish_s5"
DEEP = R / f"{DS}_isoform_deep"
OUT = Path(f"{H}/Saturn/published_figures/fig4/fig4_isoform")
FINAL = Path(f"{H}/Saturn/final_fig4")
MODALITY = dict(MODALITY, Isoform=MODALITY["Splicing"])
C_RNA, C_ISO = MODALITY["RNA"], MODALITY["Isoform"]
GENES_E = [("Tcf7l2", "thalamic"), ("Fezf2", "deep-layer cortical")]
NB_RNA = 25                 # bins along rho_RNA, as in the dataset's screen (isoform_marker_panels.NB)
ISOS_E = [("Kcnab3", "ENSMUST00000142328"), ("Zfp950", "ENSMUST00000122927")]
EXAMPLES = [("Tcf7l2", "RNA", "ENSMUST00000111651"), ("Nkx2-2", "Isoform", "ENSMUST00000109970")]
H_EX = ("Tcf7l2", "RNA", "ENSMUST00000111651", "oligodendrocyte")
GLIA = {"Oligodendrocytes", "NFOLs", "Astrocytes", "Microglia", "OPCs"}


SHORT = {"thalamus / hypothalamus": "thal./hypothal.", "white matter & hippocampus": "WM & hippoc.",
         "ventricle / choroid plexus": "ventricle/CP", "upper-layer cortex": "upper cortex"}


def cell_class(t):
    if t in ("Oligodendrocytes", "NFOLs"): return "oligodendrocyte"
    if t == "OPCs": return "OPC"
    if t == "Astrocytes": return "astrocyte"
    if t == "Microglia": return "microglia"
    if t in ("Endothelial", "Pericytes", "SMCs", "VLMCs"): return "vascular"
    if t in ("Choroid Plexus", "Ependymal"): return "choroid/ependymal"
    if t == "Other": return "other"
    return "neuron"


# --------------------------------------------------------------------------- #
def load():
    import merfish_isoform_deep as MD
    from saturn.domains import gaston_domains
    C = MD.load_cells()
    import anndata as ad
    a = ad.read_h5ad(f"{MD.D}/cellxgene_ISO_labels.h5ad", backed="r")
    o = a.obs[a.obs["sample"].astype(str) == MD.SAMPLE]
    C["xy"] = o[["global_x", "global_y"]].values.astype(float)
    C["cls"] = np.array([cell_class(t) for t in C["ct"]])
    cdom, keep = MD.cell_domains(C); C["dom"], C["keep"] = cdom, keep
    ax = pd.read_csv(R / DS / "axes.csv", index_col=0); ax.index = ax.index.astype(str)
    z = dict(np.load(R / f"targets_{DS}.npz", allow_pickle=True))
    D = dict(xy=np.c_[ax.x.values, ax.y.values], bc=list(ax.index),
             d={"RNA": ax.d_RNA.values, "Isoform": ax.d_Isoform.values},
             lab={m: gaston_domains(ax[f"d_{m}"].values, z[f"Y_{m}"], n_domains=4) for m in ("RNA", "Isoform")},
             z=z, summary=json.load(open(R / DS / "summary.json")), C=C)
    # bin-level gene counts from cells (same binning as the fit) and isoform counts (cached, fit order)
    bidx = pd.Index(D["bc"]).get_indexer(pd.Index(C["bin"]))
    m = bidx >= 0
    G = sp.csr_matrix((np.ones(m.sum()), (bidx[m], np.where(m)[0])), shape=(len(D["bc"]), len(C["bin"])))
    D["X"] = {"RNA": sp.csc_matrix(G @ C["Xg"])}
    D["names"] = {"RNA": C["genes"]}
    D["lib"] = {"RNA": np.asarray(D["X"]["RNA"].sum(1)).ravel().clip(1)}
    from saturn_isoform_switch import iso_matrix
    Xi, isos, ig = iso_matrix(DS)
    D["Xi"], D["isos"], D["iso_gene"] = sp.csc_matrix(Xi), np.array([str(i).split(".")[0] for i in isos]), ig
    D["names_dom"] = label_domains(D)
    return D


def label_domains(D):
    z = D["z"]; reg = z["ann_ccf_region"].astype(str); out = {}
    fr = lambda c, lab, k: float(np.mean(z[c][lab == k]))
    lab = D["lab"]["RNA"]; K = list(range(4))
    cp = max(K, key=lambda k: fr("ann_frac_Choroid Plexus", lab, k)); K.remove(cp)
    cx = max(K, key=lambda k: fr("mk_Slc17a7", lab, k)); K.remove(cx)
    th = max(K, key=lambda k: fr("ann_frac_Thalamus", lab, k) + fr("ann_frac_Hypothalamus", lab, k)); K.remove(th)
    wm = K[0]
    assert fr("ann_frac_Oligodendrocytes", lab, wm) > fr("ann_frac_Oligodendrocytes", lab, cx)
    out["RNA"] = {th: "thalamus / hypothalamus", wm: "white matter & hippocampus", cp: "ventricle / choroid plexus", cx: "cortex"}
    lab = D["lab"]["Isoform"]; d = D["d"]["Isoform"]; K = list(range(4))
    cp = max(K, key=lambda k: fr("ann_frac_Choroid Plexus", lab, k)); K.remove(cp)
    sub = max(K, key=lambda k: fr("ann_frac_Thalamus", lab, k) + fr("ann_frac_Hypothalamus", lab, k)); K.remove(sub)
    up = max(K, key=lambda k: np.median(d[lab == k])); K.remove(up)
    out["Isoform"] = {sub: "subcortical", cp: "ventricle / choroid plexus", K[0]: "cortex", up: "upper-layer cortex"}
    return out


def dcol(D, m):
    return np.array([DOMAIN[D["names_dom"][m][k]] for k in range(4)])


# --------------------------------------------------------------------------- #
def panel_fit(ax, S):
    R2 = S["R2_whole"]; mods = ["RNA", "Isoform"]; x = np.arange(2); w = 0.26
    for i, (key, alpha) in enumerate([("full", 1.0), ("own_axis_only", 0.65), ("other_axes_only", 0.3)]):
        v = [R2[m][key] for m in mods]
        ax.bar(x + (i - 1) * w, v, w, color=[MODALITY[m] for m in mods], alpha=alpha, edgecolor="#1a1a1a", lw=0.4)
        for xx, vv in zip(x + (i - 1) * w, v):
            ax.text(xx, vv + 0.012, f"{vv:.2f}", ha="center", va="bottom", fontsize=5.2)
    ax.set_xticks(x); ax.set_xticklabels(mods)
    for tl, m in zip(ax.get_xticklabels(), mods): tl.set_color(MODALITY[m])
    ax.set_ylim(0, 0.6); ax.set_yticks([0, 0.3, 0.6]); ax.set_ylabel("variance explained (R²)", labelpad=1)
    ax.set_xlim(-0.55, 1.55)
    ax.legend(handles=[Patch(fc="#555555", alpha=a_, ec="#1a1a1a", lw=0.4, label=l_) for a_, l_ in
                       [(1, "full"), (0.65, "own axis"), (0.3, "cross axis")]],
              fontsize=5.2, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=0.9,
              handleheight=0.8, columnspacing=0.5, handletextpad=0.3, borderaxespad=0.1)


def panel_maps(fig, spec, D):
    from matplotlib.patches import Rectangle
    gs = GridSpecFromSubplotSpec(2, 4, subplot_spec=spec, height_ratios=[1, 0.3], wspace=0.18, hspace=0.02)
    first = None
    for j, m in enumerate(("RNA", "Isoform")):
        a1 = fig.add_subplot(gs[0, 2 * j]); a2 = fig.add_subplot(gs[0, 2 * j + 1]); first = first or a1
        d = D["d"][m]; lo, hi = np.percentile(d, [1, 99])
        sc = F2.spatial(a1, D["xy"], d, cmap=CMAP["score"], vmin=lo, vmax=hi, s=0.9)
        F2.cbar(fig, a1, sc, "isodepth", ticks=[round(lo, 1), round(hi, 1)])
        a1.set_title(f"ρ$_{{\\mathrm{{{m}}}}}$", color=MODALITY[m], fontsize=7, pad=2)
        F2.spatial(a2, D["xy"], None, colors=dcol(D, m)[D["lab"][m]], s=0.9)
        a2.set_title(f"{m} domains", color=MODALITY[m], fontsize=7, pad=2)
        lg = fig.add_subplot(gs[1, 2 * j:2 * j + 2]); lg.set_axis_off()
        names = D["names_dom"][m]
        order = np.argsort([np.median(d[D["lab"][m] == k]) for k in range(4)])
        for i, k in enumerate(order):
            n = int((D["lab"][m] == k).sum()); x_ = 0.02 + (i % 2) * 0.5; y_ = 0.72 - (i // 2) * 0.5
            lg.add_patch(Rectangle((x_, y_ - 0.12), 0.035, 0.3, color=DOMAIN[names[k]], transform=lg.transAxes))
            lg.text(x_ + 0.05, y_ + 0.03, f"{names[k]}  (n={n})", fontsize=5.6, va="center", transform=lg.transAxes)
    return first


def panel_cells(ax, D):
    C = D["C"]; order = ["other", "vascular", "choroid/ependymal", "microglia", "OPC", "astrocyte", "oligodendrocyte", "neuron"]
    for c in order:
        m = C["cls"] == c
        ax.scatter(C["xy"][m, 0], C["xy"][m, 1], s=0.12, lw=0, color=CELLCLASS[c], rasterized=True)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    x0, y0 = C["xy"][:, 0].min(), C["xy"][:, 1].min()
    ax.plot([x0, x0 + 1000], [y0 - 150] * 2, color="k", lw=1.5); ax.text(x0 + 500, y0 - 220, "1 mm", ha="center", va="top", fontsize=5.5)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=3, color=CELLCLASS[c], label=f"{c} ({(C['cls'] == c).sum():,})")
                       for c in order[::-1]], fontsize=5.1, loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0,
              handletextpad=0.1, labelspacing=0.25)
    ax.set_title(f"section 5: {len(C['cls']):,} cells, 36 types", fontsize=6.3, pad=2)


def panel_decomp(ax, D):
    Dm = pd.read_csv(DEEP / "switch_decomposition.csv")
    v = Dm.within_frac.clip(-0.25, 1.0).values; sig = Dm.within_sig.values
    rng = np.random.default_rng(0); x = rng.uniform(-0.25, 0.25, len(v))
    ax.scatter(x[~sig], v[~sig], s=5, lw=0, color="#b0b7bd", label="within-type CI spans 0")
    ax.scatter(x[sig], v[sig], s=5, lw=0, color=C_ISO, label="within-type significant")
    med = np.median(Dm.within_frac); ax.plot([-0.38, 0.38], [med, med], color="#1a1a1a", lw=1.3)
    ax.axhline(0, color="#9aa2aa", lw=0.5, ls=(0, (2, 2)))
    ax.set_xlim(-0.4, 1.9); ax.set_xticks([]); ax.set_ylim(-0.27, 1.05)
    ax.set_ylabel("within-cell-type share\nof the usage change", labelpad=1)
    ax.text(0.45, med, f"median {med:.2f}\n→ {100 * (1 - med):.0f}% composition", fontsize=5.5, va="center")
    ax.text(0.45, 0.85, f"{int(sig.sum())}/{len(Dm)} switches with\nsignificant within-type\ncomponent", fontsize=5.3, va="top")
    ax.legend(fontsize=5.1, loc="lower left", bbox_to_anchor=(0.33, 0.0), handletextpad=0.1, borderaxespad=0.1, markerscale=1.5)
    ax.set_title(f"{len(Dm)} isoform switches", fontsize=6.3, pad=2)
    return dict(median_within=float(med), n_sig=int(sig.sum()), n=len(Dm))


def panel_features(fig, spec, D):
    from saturn.piecewise import fit_binomial
    from saturn.domains import isodepth_bins as gaston_bins
    gs = GridSpecFromSubplotSpec(1, 4, subplot_spec=spec, wspace=0.55)
    stats, first = {}, None
    for i, (g, sub) in enumerate(GENES_E):
        a_, Fres = F2.panel_feature(fig, gs[i], D, "RNA", g, g, sub)
        first = first or a_; stats[g] = dict(R2=round(float(Fres["r2"]), 3), n_sloped=Fres["n_sloped"])
        rng_ = []
        for nb in (20, 25, 30, 35):                   # binned R2 is binning-sensitive for thalamic genes
            F2.NB["RNA"] = nb; rng_.append(round(float(F2.feature_function(D, "RNA", g)["r2"]), 2))
        F2.NB["RNA"] = NB_RNA; stats[g]["R2_bins_20_25_30_35"] = rng_
    for i, (g, iso) in enumerate(ISOS_E):
        gg = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2 + i], height_ratios=[1.05, 1], hspace=0.28)
        j = int(np.where(D["isos"] == iso)[0][0]); cols = np.where(D["iso_gene"] == D["iso_gene"][j])[0]
        s = np.asarray(D["Xi"][:, j].todense()).ravel(); n = np.asarray(D["Xi"][:, cols].sum(1)).ravel()
        d, lab = D["d"]["Isoform"], D["lab"]["Isoform"]
        sl, it = fit_binomial(s[:, None], n, lab, d, t=0.1); sl, it = sl[0], it[0]
        a1 = fig.add_subplot(gg[0])
        nb = cKDTree(D["xy"]).query(D["xy"], k=30)[1]
        u = s[nb].sum(1) / np.maximum(n[nb].sum(1), 1); u[n[nb].sum(1) == 0] = np.nan
        sc = F2.spatial(a1, D["xy"], u, cmap=CMAP["score"], vmin=np.nanpercentile(u, 2), vmax=np.nanpercentile(u, 98))
        F2.cbar(fig, a1, sc, "usage (30-NN)")
        a1.set_title(f"{g} isoform\n{iso[-6:]} usage", fontsize=6.5, pad=2, color=C_ISO)
        a2 = fig.add_subplot(gg[1]); cl = dcol(D, "Isoform")
        bi, _, kp = gaston_bins(d, 30)
        db = np.array([d[bi == b].mean() for b in kp]); ob = np.array([s[bi == b].sum() / max(n[bi == b].sum(), 1) for b in kp])
        lb = np.array([np.bincount(lab[bi == b], minlength=4).argmax() for b in kp]); nb_ = np.array([n[bi == b].sum() for b in kp])
        for k in range(4):
            m = (lb == k) & (nb_ >= 20)
            a2.scatter(db[m], ob[m], s=5, color=cl[k], lw=0.3, edgecolor="#1a1a1a", zorder=3)
            dk = d[lab == k]; xs = np.linspace(dk.min(), dk.max(), 20)
            a2.plot(xs, 1 / (1 + np.exp(-(it[k] + sl[k] * xs))), color=cl[k], lw=1.3)
        a2.set_ylim(-0.02, 1.02); a2.set_xlabel("ρ$_{\\mathrm{Isoform}}$", color=C_ISO, labelpad=1)
        a2.set_ylabel("usage", labelpad=1); a2.tick_params(labelsize=5.5, pad=1)
        a2.set_title(f"{int((np.abs(sl) > 1e-12).sum())}/4 domains sloped (binomial)", fontsize=5.8, pad=2)
        stats[g] = dict(isoform=iso, n_sloped=int((np.abs(sl) > 1e-12).sum()))
    return first, stats


def usage_by_domain(D, gene, axis, iso, cls, rng, nboot=300):
    import merfish_isoform_deep as MD
    C = D["C"]; a, tot = MD.usage_counts(C, gene, iso)
    out = []
    for k in range(4):
        m = C["keep"] & (C["dom"][axis] == k) & (tot > 0) & ((C["cls"] == cls) if cls != "all cells" else True)
        idx = np.where(m)[0]
        if tot[idx].sum() < 20: out.append((np.nan, np.nan, np.nan)); continue
        u = a[idx].sum() / tot[idx].sum()
        bs = [a[b].sum() / max(tot[b].sum(), 1) for b in (rng.choice(idx, len(idx)) for _ in range(nboot))]
        out.append((u, np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
    return np.array(out)


def panel_within(fig, spec, D):
    gs = GridSpecFromSubplotSpec(1, len(EXAMPLES), subplot_spec=spec, wspace=0.45)
    rng = np.random.default_rng(0); first = None; stats = {}
    for i, (g, axis, iso) in enumerate(EXAMPLES):
        a = fig.add_subplot(gs[i]); first = first or a
        order = np.argsort([np.median(D["d"][axis][D["lab"][axis] == k]) for k in range(4)])
        x = np.arange(4)
        for cls, col, ls in (("all cells", "#1a1a1a", "-"), ("oligodendrocyte", CELLCLASS["oligodendrocyte"], "-"),
                             ("OPC", CELLCLASS["OPC"], "-"), ("astrocyte", CELLCLASS["astrocyte"], "-"),
                             ("neuron", CELLCLASS["neuron"], ":")):
            U = usage_by_domain(D, g, axis, iso, cls, rng)[order]
            ok = np.isfinite(U[:, 0])
            a.errorbar(x[ok] + {"all cells": 0, "oligodendrocyte": -0.12, "OPC": 0.12, "astrocyte": 0.24, "neuron": -0.24}[cls],
                       U[ok, 0], yerr=[U[ok, 0] - U[ok, 1], U[ok, 2] - U[ok, 0]], color=col, lw=1.4 if cls == "all cells" else 1.0,
                       ls=ls, marker="o", ms=2.5, capsize=1.2, elinewidth=0.6, label=cls)
            stats[f"{g}|{cls}"] = [None if not np.isfinite(v) else round(float(v), 3) for v in U[:, 0]]
        names = [D["names_dom"][axis][k] for k in order]
        a.set_xticks(x); a.set_xticklabels([SHORT.get(n, n) for n in names], fontsize=5.3, rotation=30, ha="right",
                                           rotation_mode="anchor")
        for tl, n in zip(a.get_xticklabels(), names): tl.set_color(DOMAIN[n])
        a.set_ylim(0, 1.0); a.set_ylabel(f"usage of {iso[-6:]}", labelpad=1); a.tick_params(labelsize=5.5, pad=1)
        a.set_title(f"{g}  (domains of ρ$_{{\\mathrm{{{axis}}}}}$)", fontsize=6.3, pad=2,
                    color=C_RNA if axis == "RNA" else C_ISO)
        if i == len(EXAMPLES) - 1:
            a.legend(fontsize=5.1, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, handlelength=1.4, labelspacing=0.3)
    return first, stats


RHO_EX = [("Tcf7l2", "RNA", "ENSMUST00000111651", ["oligodendrocyte", "OPC", "neuron"]),
          ("Nkx2-2", "Isoform", "ENSMUST00000109970", ["astrocyte", "OPC", "neuron"])]
NQ, NROT_G = 6, 200          # per-class sextiles of rho: every class gets evenly filled points


def _logit_slope(a, n, x, iters=30):
    """binomial IRLS, logit(p) = b0 + b1 * x; returns b1"""
    b = np.zeros(2); X = np.c_[np.ones_like(x), x]
    p0 = np.clip(a.sum() / n.sum(), 1e-4, 1 - 1e-4); b[0] = np.log(p0 / (1 - p0))
    for _ in range(iters):
        eta = X @ b; p = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
        W = n * p * (1 - p); z = eta + (a - n * p) / np.maximum(W, 1e-9)
        bn = np.linalg.solve(X.T @ (W[:, None] * X) + 1e-8 * np.eye(2), X.T @ (W * z))
        if np.max(np.abs(bn - b)) < 1e-7: b = bn; break
        b = bn
    return float(b[1])


def panel_rho_within(fig, spec, D):
    """switch-isoform usage as a continuous function of rho INSIDE each cell class; slope tested
    against the rotation null (rho rotated at the bin level, re-read by every cell)"""
    import merfish_isoform_deep as MD
    from sproxseq_validate2 import rotated_fields
    C = D["C"]; rng = np.random.default_rng(0)
    cell_bin = pd.Index(D["bc"]).get_indexer(pd.Index(C["bin"]))
    gs = GridSpecFromSubplotSpec(1, len(RHO_EX), subplot_spec=spec, wspace=0.42)
    first, st = None, {}
    for i, (g, axis, iso, classes) in enumerate(RHO_EX):
        a_, tot = MD.usage_counts(C, g, iso)
        dbin = D["d"][axis]; sd = dbin.std()
        rot = rotated_fields(D["xy"], dbin, NROT_G, rng)
        ok = (cell_bin >= 0) & (tot > 0)
        dcell = np.full(len(tot), np.nan); dcell[ok] = dbin[cell_bin[ok]]
        ax = fig.add_subplot(gs[i]); first = first or ax
        for cls in classes + ["all cells"]:
            m = ok & ((C["cls"] == cls) if cls != "all cells" else True)
            idx = np.where(m)[0]; x = dcell[idx]; A = a_[idx].astype(float); N = tot[idx].astype(float)
            q = np.quantile(x, np.linspace(0, 1, NQ + 1))
            bi = np.clip(np.digitize(x, q[1:-1]), 0, NQ - 1)
            xb, yb, lo, hi = [], [], [], []
            for b in range(NQ):
                mb = bi == b
                if N[mb].sum() < 10: continue
                u = A[mb].sum() / N[mb].sum(); k = np.where(mb)[0]
                bs = [A[s].sum() / max(N[s].sum(), 1) for s in (rng.choice(k, len(k)) for _ in range(200))]
                xb.append(x[mb].mean()); yb.append(u); lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
            xb, yb, lo, hi = map(np.array, (xb, yb, lo, hi))
            col = "#1a1a1a" if cls == "all cells" else CELLCLASS[cls]
            ls = ":" if cls == "neuron" else "-"
            ax.fill_between(xb, lo, hi, color=col, alpha=0.12, lw=0)
            ax.plot(xb, yb, color=col, lw=1.4 if cls == "all cells" else 1.1, ls=ls, marker="o", ms=2.2)
            b1 = _logit_slope(A, N, (x - x.mean()) / sd)
            nl = np.array([_logit_slope(A, N, (r[cell_bin[idx]] - r[cell_bin[idx]].mean()) / sd) for r in rot])
            p = float(((np.abs(nl) >= abs(b1)).sum() + 1) / (NROT_G + 1))
            st[f"{g}|{cls}"] = dict(n_cells=int(len(idx)), reads=int(N.sum()), slope_per_sd=round(b1, 3), p_rot=p)
        ax.set_ylim(0, 1.0); ax.set_xlabel(f"ρ$_{{\\mathrm{{{axis}}}}}$", color=C_RNA if axis == "RNA" else C_ISO, labelpad=1)
        if i == 0: ax.set_ylabel(f"switch-isoform usage", labelpad=1)
        ax.tick_params(labelsize=5.5, pad=1)
        ax.set_title(f"{g} {iso[-6:]} along ρ$_{{\\mathrm{{{axis}}}}}$", fontsize=6.2, pad=2,
                     color=C_RNA if axis == "RNA" else C_ISO)
        txt = "\n".join(f"{c if c != 'oligodendrocyte' else 'oligo'} {st[f'{g}|{c}']['slope_per_sd']:+.2f} "
                        f"(p {st[f'{g}|{c}']['p_rot']:.2f})" for c in classes)
        ax.text(0.98, 0.03, "slope per SD of ρ\n" + txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=4.9 + 0.2,
                color="#3a3f4b")
    return first, st


def panel_spillover(ax, D):
    S = pd.read_csv(DEEP / "within_type_spillover_check.csv")
    S = S[S.type.isin(GLIA)].copy()
    has = np.isfinite(S.expr_ratio)
    xr = np.log2(S.expr_ratio.where(has, np.nan))
    lam = S.gene == "Lamp5"
    ax.scatter(xr[has & ~lam], S.neighbour_r[has & ~lam], s=10, color=C_ISO, lw=0.3, edgecolor="#1a1a1a", label="expressed in the glial type")
    ax.scatter(xr[has & lam], S.neighbour_r[has & lam], s=12, color="#B64342", marker="X", lw=0, label="Lamp5 (likely spillover)")
    xs = np.full((~has).sum(), np.log2(80)); ax.scatter(xs, S.neighbour_r[~has], s=9, color="#9aa2aa", lw=0, marker="s",
                                                         label="gene not on panel")
    ax.axvline(np.log2(0.3), color="#9aa2aa", lw=0.6, ls=(0, (2, 2)))
    ax.text(np.log2(80), 0.27, "gene not\non panel", fontsize=5.1, ha="center", va="bottom", color="#7f8a94")
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_xlabel("glia / neuron expression (log2)", labelpad=1); ax.set_ylabel("usage r with neighbouring\nneurons", labelpad=1)
    for g in ["Tcf7l2", "Nkx2-2", "Aldh9a1", "Lamp5"]:
        r = S[(S.gene == g) & has].iloc[:1]
        if len(r): ax.annotate(g, (np.log2(r.expr_ratio.iloc[0]), r.neighbour_r.iloc[0]), fontsize=5.2, xytext=(2, 2),
                               textcoords="offset points", style="italic")
    ax.legend(fontsize=5.0 + 0.1, loc="lower left", borderaxespad=0.1, handletextpad=0.1, labelspacing=0.25)
    ax.tick_params(labelsize=5.5, pad=1)
    ax.set_title("spillover control (glial cases)", fontsize=6.3, pad=2)
    return dict(n=len(S), lamp5=int(lam.sum()), expressed=int((has & ~lam & (S.expr_ratio >= 0.3)).sum()), no_panel=int((~has).sum()))


def panel_space_example(fig, spec, D):
    import merfish_isoform_deep as MD
    g, axis, iso, cls = H_EX; C = D["C"]
    a_, tot = MD.usage_counts(C, g, iso)
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.08)
    axs = []
    for i, c in enumerate([cls, "neuron"]):
        m = C["keep"] & (C["cls"] == c) & (tot > 0)
        idx = np.where(m)[0]; nb = cKDTree(C["xy"][idx]).query(C["xy"][idx], k=25)[1]
        u = a_[idx][nb].sum(1) / tot[idx][nb].sum(1)
        a = fig.add_subplot(gs[i]); axs.append(a)
        a.scatter(C["xy"][:, 0], C["xy"][:, 1], s=0.05, color="#e6e6e6", lw=0, rasterized=True)
        sc = a.scatter(C["xy"][idx, 0], C["xy"][idx, 1], c=u, cmap=CMAP["score"], vmin=0, vmax=1, s=1.3 if c != "neuron" else 0.7,
                       lw=0, rasterized=True)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); [s.set_visible(False) for s in a.spines.values()]
        a.set_title(f"{g} {iso[-6:]} usage\nin {c}s (n={len(idx):,})", fontsize=6.0, pad=2, color=CELLCLASS[c])
    cax = axs[1].inset_axes([1.03, 0.15, 0.05, 0.7]); cb = fig.colorbar(sc, cax=cax, ticks=[0, 0.5, 1])
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.5, length=1.5); cb.set_label("usage (25-NN)", fontsize=5.5)
    return axs[0]


def panel_usage_expr(ax, D):
    G = pd.read_csv(DEEP / "usage_vs_expression.csv")
    s = G[(G.axis == "Isoform") & (G.scope == "pooled")]
    pure = (s.usage_swing >= 0.2) & (s.expr_log2fc < 0.5); both = (s.usage_swing >= 0.2) & (s.expr_log2fc >= 0.5)
    ax.scatter(s.expr_log2fc[~pure & ~both], s.usage_swing[~pure & ~both], s=2.5, lw=0, color="#c9cdd2", rasterized=True)
    ax.scatter(s.expr_log2fc[both], s.usage_swing[both], s=3.5, lw=0, color="#7f8a94", label=f"usage + expression ({int(both.sum())})")
    ax.scatter(s.expr_log2fc[pure], s.usage_swing[pure], s=6, lw=0.3, edgecolor="#1a1a1a", color=C_ISO,
               label=f"usage only ({int(pure.sum())})")
    ax.axhline(0.2, color="#9aa2aa", lw=0.6, ls=(0, (2, 2))); ax.axvline(0.5, color="#9aa2aa", lw=0.6, ls=(0, (2, 2)))
    ax.set_xlabel("|log2 FC| total expression across domains", labelpad=1); ax.set_ylabel("max isoform usage swing", labelpad=1)
    ax.set_xlim(0, min(6, s.expr_log2fc.max() + 0.2)); ax.set_ylim(0, 1)
    ax.legend(fontsize=5.1, loc="upper right", borderaxespad=0.1, handletextpad=0.1, markerscale=1.5)
    ax.tick_params(labelsize=5.5, pad=1)
    ax.set_title(f"{len(s)} multi-isoform genes, ρ$_{{\\mathrm{{Isoform}}}}$ domains", fontsize=6.3, pad=2)
    return dict(n=len(s), usage_only=int(pure.sum()), both=int(both.sum()))


# --------------------------------------------------------------------------- #
def _single(fn):
    def b(fig, spec, D):
        ax = fig.add_subplot(spec); return ax, fn(ax, D)
    return b


PANELS = [
    ("a", (2.6, 1.6), _single(lambda ax, D: panel_cells(ax, D))),
    ("b", (1.45, 1.6), _single(lambda ax, D: panel_fit(ax, D["summary"]))),
    ("c", (1.6, 1.6), _single(panel_decomp)),
    ("d", (7.2, 1.5), lambda fig, spec, D: (panel_maps(fig, spec, D), None)),
    ("e", (7.2, 2.0), panel_features),
    ("f", (5.2, 1.7), panel_within),
    ("g", (4.2, 1.8), panel_rho_within),
    ("h", (2.3, 1.7), _single(panel_usage_expr)),
]

# moved to the supplement: the spillover control for the glial cases in f, and the two-map view that g quantifies
SUPP = [
    ("spillover", (2.2, 1.9), _single(panel_spillover)),
    ("space_example", (3.3, 1.7), lambda fig, spec, D: (panel_space_example(fig, spec, D), None)),
]


def export_panels(D, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    for L, size, build in PANELS:
        fig = plt.figure(figsize=size)
        spec = GridSpec(1, 1, figure=fig, left=0.14, right=0.86, top=0.84, bottom=0.2)[0]
        build(fig, spec, D); save_dual(fig, outdir / f"fig4{L}", write_manifest=False); plt.close(fig)
    print(f"-> {len(PANELS)} panels in {outdir}")


def export_supp(D, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, size, build in SUPP:
        fig = plt.figure(figsize=size)
        spec = GridSpec(1, 1, figure=fig, left=0.2, right=0.86, top=0.84, bottom=0.2)[0]
        _, out = build(fig, spec, D); stats[name] = out
        save_dual(fig, outdir / f"figS4_{name}", write_manifest=False); plt.close(fig)
    json.dump(stats, open(outdir / "figS4_stats.json", "w"), indent=1, default=str)
    print(f"-> {len(SUPP)} supplementary panels in {outdir}")


def main():
    apply(base_pt=6.5)
    F2.MODALITY["Isoform"] = C_ISO                    # F2.panel_feature colours by modality name
    F2.NB["RNA"] = NB_RNA
    D = load()
    fig = plt.figure(figsize=(7.2, 9.7))
    outer = GridSpec(5, 1, figure=fig, height_ratios=[1.55, 1.45, 2.0, 1.75, 1.65], hspace=0.7,
                     left=0.07, right=0.95, top=0.975, bottom=0.04)
    g1 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], width_ratios=[1.75, 0.95, 1.05], wspace=0.75)
    g4 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[3], width_ratios=[1.0, 0.32], wspace=0.1)
    g5 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[4], width_ratios=[1.75, 1.0], wspace=0.35)
    where = dict(a=g1[0], b=g1[1], c=g1[2], d=outer[1], e=outer[2], f=g4[0], g=g5[0], h=g5[1])
    nudge = dict(a=(-0.03, 0.015), b=(-0.06, 0.015), c=(-0.07, 0.015), d=(-0.03, 0.02), e=(-0.06, 0.03),
                 f=(-0.06, 0.03), g=(-0.07, 0.03), h=(-0.07, 0.03))
    res = {}
    for L, _, build in PANELS:
        ax, out = build(fig, where[L], D); res[L] = out; F2.letter(ax, L, *nudge[L])
    F2.draw_letters(fig)
    manifest = dict(dataset=DS, domains={m: {int(k): v for k, v in D["names_dom"][m].items()} for m in D["names_dom"]},
                    decomposition=res["c"], features=res["e"], within_examples=res["f"], within_along_rho=res["g"],
                    usage_vs_expression=res["h"],
                    moved_to_supplement=["spillover control for the glial within-type cases (was g)",
                                         "Tcf7l2 usage maps in oligodendrocytes vs neurons (was h; quantified by g)"],
                    not_shown=dict(mechanism="99.7% of measured isoform pairs differ at ends (probe design); switches match "
                                             "background p=0.62; neuronal 3'UTR control null (290/587, p=0.63)"),
                    sources=[str(R / DS), str(DEEP), str(R / f"{DS}_isoform_switch")])
    save_dual(fig, OUT, extra=manifest); save_dual(fig, FINAL / "fig4_isoform", extra=manifest); plt.close(fig)
    export_panels(D, FINAL / "panels")
    export_supp(D, FINAL / "supp")
    print(json.dumps(dict(domains=manifest["domains"], c=res["c"], g=res["g"], h=res["h"]), indent=1))


if __name__ == "__main__":
    main()
