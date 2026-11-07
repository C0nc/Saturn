"""Figure 3 (v2) — Fig 3 a-j unchanged + CellRank 2 on the Saturn axes and the transition-state comparison with Mellon.

  k  CellRank 2 fate flow (transition matrix projected on the tissue, streamlines; width = flow strength) over the tumour-fate probability, four kernels (each 0.8 PseudotimeKernel + 0.2 ConnectivityKernel,
     GPCCA, 2 terminal states; time oriented to increase with the melanocytic programme): DPT baseline, rho_RNA,
     rho_ATAC, and the two-kernel fusion 0.5 k_rhoRNA + 0.5 k_rhoATAC. Spearman of the fate with the melanocytic score,
     and the top lineage drivers
  (supplementary: the same arrows over the RNA domains; tumour-fate agreement along the fusion weight w; fate maps
   with lineage drivers)
  l  transition zones from the isodepth gradient (state change per spot step along grad rho, top 10%, connected zones;
     gradual = >= 3 spots wide along the gradient, annotated by what changes across them; share of state change along
     grad rho vs 200 rotations) vs the standard Saturn-free route: Mellon log-density on ATAC LSI / RNA PCA, lowest 10%
     = 'transition' (recall of the zone spots, depth of flagged spots, epidermis share)
  m  marker programmes (melanocytic, immune, stroma, CNV; z across tissue) across the largest unit of each method,
     in 5 bins of the method's own coordinate: Saturn gradient zone 1 by rho; Mellon patch 1 (largest connected patch of
     its lowest-density 10%) by log-density
  n  the m comparison for every zone / patch: best |Spearman| of the programmes with the method's coordinate minus a
     500-permutation null of the same size (filled: p < 0.05); % = flagged spots inside an ordered unit
Sources: Hole/gaston_mix_mm/results/real_saturn/melanoma_tissue_cellrank_kernels (mel_cellrank_kernels.py),
melanoma_tissue_gradient (mel_gradient_transition.py, mel_baseline_mellon.py).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, "<workspace>/Saturn/published_figures/_style")
sys.path.insert(0, "<workspace>/Saturn/published_figures/fig2")
sys.path.insert(0, "<workspace>/Saturn/published_figures/fig3")
from paper_style import apply, save_dual, MODALITY, METHOD, CMAP, PROGRAM
from scipy.stats import spearmanr
import fig2_lpc as F2
import fig3_melanoma as F3

H = "<workspace>"
CRK = F3.R / "melanoma_tissue_cellrank_kernels"
GRD = F3.R / "melanoma_tissue_gradient"
OUT = Path(f"{H}/Saturn/published_figures/fig3/fig3_melanoma_v2")
FINAL = Path(f"{H}/Saturn/final_fig3_v2")
C_RNA, C_ATAC = MODALITY["RNA"], MODALITY["ATAC"]
KERNELS = [("DPT", "DPT (baseline)", METHOD["DPT"]), ("rho_RNA", "ρ$_{\\mathrm{RNA}}$ kernel", C_RNA),
           ("rho_ATAC", "ρ$_{\\mathrm{ATAC}}$ kernel", C_ATAC), ("fusion_w0.50", "fusion (w = 0.5)", METHOD["CellRank fusion"])]
FLAG = 0.10


def _obs(D):
    o = pd.read_csv(CRK / "obs.csv", index_col=0).reindex(D["bc"])
    g = pd.read_csv(GRD / "spots.csv", index_col=0).reindex(D["bc"])
    b = pd.read_csv(GRD / "baseline_spots.csv", index_col=0).reindex(D["bc"])
    return o, g, b


def _drivers(k, n=3):
    gs = json.load(open(CRK / "drivers.json"))["drivers"][k]["genes"]
    return [g for g in gs if not (g.startswith(("MT-", "ENSG", "LINC", "MIR")) or "-AS" in g)][:n]


def panel_fate_maps(fig, spec, D):
    o, _, _ = _obs(D); S = pd.read_csv(CRK / "kernels_summary.csv").set_index("kernel")
    gs = GridSpecFromSubplotSpec(1, 4, subplot_spec=spec, wspace=0.12)
    axs, out = [], {}
    for j, (k, lab, col) in enumerate(KERNELS):
        ax = fig.add_subplot(gs[j]); axs.append(ax)
        v = o[f"tumour_fate_{k}"].values
        sc = F2.spatial(ax, D["xy"], v, cmap=CMAP["pseudotime"], vmin=np.nanpercentile(v, 2), vmax=np.nanpercentile(v, 98), s=0.9)
        r = float(S.loc[k, "tumour_fate_vs_orient"])
        ax.set_title(f"{lab}\nr = {r:+.2f} with melanocytic", fontsize=5.8, color=col, pad=2)
        ax.text(0.5, -0.04, ", ".join(_drivers(k)), transform=ax.transAxes, ha="center", va="top",
                fontsize=5, style="italic", color="#1a1a1a")
        out[k] = dict(r_melanocytic=r, drivers=_drivers(k, 6))
    cax = axs[-1].inset_axes([1.03, 0.1, 0.05, 0.8]); cb = fig.colorbar(sc, cax=cax, ticks=[])
    cb.outline.set_linewidth(0.4); cb.set_label("tumour-fate prob.", fontsize=5.3, labelpad=1)
    axs[0].text(-0.03, -0.04, "drivers:", transform=axs[0].transAxes, ha="right", va="top", fontsize=5, color="#5c6670")
    return axs[0], out


def _grid_flow(xy, V, pitch, n=40, sigma=2.0):
    """CellRank-style projection averaged on a regular grid (Gaussian weights, sigma in spot pitches); grid points
    farther than one pitch from any spot are masked. Returns grid axes and U, W, speed (n x n)"""
    from scipy.spatial import cKDTree
    gx = np.linspace(xy[:, 0].min(), xy[:, 0].max(), n); gy = np.linspace(xy[:, 1].min(), xy[:, 1].max(), n)
    X, Y = np.meshgrid(gx, gy); G = np.c_[X.ravel(), Y.ravel()]
    d, _ = cKDTree(xy).query(G)
    Wt = np.exp(-((G[:, None, :] - xy[None, :, :]) ** 2).sum(-1) / (2 * (sigma * pitch) ** 2))
    Vg = (Wt @ V) / Wt.sum(1, keepdims=True); Vg[d > pitch] = np.nan
    U, Wv = Vg[:, 0].reshape(n, n), Vg[:, 1].reshape(n, n)
    return gx, gy, U, Wv, np.hypot(U, Wv)


def panel_fate_arrows(fig, spec, D, background="fate"):
    """fate arrows (transition matrix projected on the tissue, mel_cellrank_arrows.py) over the tumour-fate
    probability of the same kernel (background='fate') or the RNA domains (background='domain')"""
    from paper_style import DOMAIN
    o, _, _ = _obs(D); S = pd.read_csv(CRK / "kernels_summary.csv").set_index("kernel")
    A = np.load(CRK / "arrows.npz", allow_pickle=True)
    idx = pd.Index([str(b) for b in A["barcodes"]]).get_indexer(pd.Index(D["bc"])); assert (idx >= 0).all()
    gs = GridSpecFromSubplotSpec(1, 4, subplot_spec=spec, wspace=0.12)
    axs, out = [], {}
    for j, (k, lab, col) in enumerate(KERNELS):
        ax = fig.add_subplot(gs[j]); axs.append(ax)
        if background == "fate":
            v = o[f"tumour_fate_{k}"].values
            sc = F2.spatial(ax, D["xy"], v, cmap=CMAP["pseudotime"], vmin=np.nanpercentile(v, 2),
                            vmax=np.nanpercentile(v, 98), s=2.2)
        else:
            nm = D["names_dom"]["RNA"]; c_ = [DOMAIN[nm[l]] for l in D["lab"]["RNA"]]
            F2.spatial(ax, D["xy"], None, colors=c_, s=2.2)
        gx, gy, U, Wv, sp_ = _grid_flow(D["xy"], A[f"V_{k}"][idx], PITCH)
        lw = 0.15 + 0.6 * np.clip(sp_ / np.nanpercentile(sp_, 95), 0, 1)
        ax.streamplot(gx, gy, U, Wv, color="#2a2f33", linewidth=lw, density=0.75, arrowsize=0.45,
                      arrowstyle="-|>", minlength=0.12, zorder=3)
        lo, hi = D["xy"].min(0) - PITCH / 2, D["xy"].max(0) + PITCH / 2      # streamlines must not stretch the frame
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        r = float(S.loc[k, "tumour_fate_vs_orient"])
        ax.set_title(f"{lab}\nr = {r:+.2f} with melanocytic", fontsize=5.8, color=col, pad=2)
        out[k] = dict(r_melanocytic=r, mean_flow=float(np.nanmean(sp_)))
    if background == "fate":
        cax = axs[-1].inset_axes([1.03, 0.1, 0.05, 0.8]); cb = fig.colorbar(sc, cax=cax, ticks=[])
        cb.outline.set_linewidth(0.4); cb.set_label("tumour-fate prob.", fontsize=5.3, labelpad=1)
    else:
        from matplotlib.patches import Patch as _P
        nm = D["names_dom"]["RNA"]
        axs[0].legend(handles=[_P(fc=DOMAIN[v], label=v) for v in nm.values()], fontsize=5, ncol=4, frameon=False,
                      loc="upper left", bbox_to_anchor=(0.0, -0.02), handlelength=0.9, columnspacing=0.8)
    return axs[0], out


def panel_fusion(ax, D):
    S = pd.read_csv(CRK / "kernels_summary.csv").set_index("kernel")
    w = [0, 0.25, 0.5, 0.75, 1.0]; ks = ["rho_ATAC", "fusion_w0.25", "fusion_w0.50", "fusion_w0.75", "rho_RNA"]
    y = [float(S.loc[k, "tumour_fate_vs_orient"]) for k in ks]
    ax.plot(w, y, "-", color=METHOD["CellRank fusion"], lw=1.0, zorder=2)
    ax.scatter(w, y, s=12, c=[C_ATAC, METHOD["CellRank fusion"], METHOD["CellRank fusion"], METHOD["CellRank fusion"], C_RNA],
               lw=0.3, edgecolor="#1a1a1a", zorder=3)
    dpt = float(S.loc["DPT", "tumour_fate_vs_orient"])
    ax.axhline(dpt, color=METHOD["DPT"], lw=0.8, ls="--"); ax.text(0.02, dpt + 0.02, "DPT", fontsize=5, color=METHOD["DPT"])
    ax.set_xticks(w); ax.set_xticklabels(["0\nATAC", "0.25", "0.5", "0.75", "1\nRNA"], fontsize=5.3)
    ax.set_xlabel("fusion weight on k$_{\\mathrm{RNA}}$", labelpad=1)
    ax.set_ylabel("tumour fate vs\nmelanocytic (Spearman)")
    ax.set_ylim(-0.05, 0.65)
    return dict(zip(ks, y), DPT=dpt)


ZONE_TAG = {("RNA", 212): "epidermis → tumour front", ("RNA", 929): "CNV↑, melanocytic loss",
            ("ATAC", 505): "tumour → immune/stroma", ("ATAC", 45): "high-CNV tumour",
            ("ATAC", 437): "CNV↑ tumour (CNV +0.66)", ("ATAC", 808): "immune-accessible edge"}


def panel_transition_maps(fig, spec, D):
    """gradient transition zones (annotated by what changes across them) vs Mellon lowest-density spots"""
    _, g, b = _obs(D); Z = pd.read_csv(GRD / "zones.csv"); A = json.load(open(GRD / "summary.json"))
    BS = pd.read_csv(GRD / "baseline_summary.csv")
    gs = GridSpecFromSubplotSpec(1, 4, subplot_spec=spec, wspace=0.1)
    axs, out = [], {}
    note = dict(fontsize=5.2, ha="center", va="top", linespacing=1.15)
    for j, m in enumerate(["ATAC", "RNA"]):
        ax = fig.add_subplot(gs[j]); axs.append(ax)
        zone = g[f"zone_{m}"].values; tr = g[f"transition_{m}"].values.astype(bool); sharp = (zone >= 0) & ~tr
        F2.spatial(ax, D["xy"], None, colors="#e6e6e6", s=1.1)
        ax.scatter(*D["xy"][sharp].T, s=1.1, lw=0, color="#9aa2aa", rasterized=True)
        ax.scatter(*D["xy"][tr].T, s=1.5, lw=0, color=MODALITY[m], rasterized=True)
        z = Z[(Z.axis == m) & (Z.kind == "gradual")].sort_values("n", ascending=False).reset_index(drop=True)
        for i, r in z.iterrows():
            c = D["xy"][zone == r.zone].mean(0)
            ax.text(*c, str(i + 1), fontsize=5.5, ha="center", va="center", fontweight="bold", color="#1a1a1a",
                    bbox=dict(fc="white", ec=MODALITY[m], lw=0.5, boxstyle="circle,pad=0.15"))
        a = A[m]
        ax.set_title(f"Saturn $\\nabla$ρ$_{{\\mathrm{{{m}}}}}$ transition zones\n{len(z)} gradual (≥3 spots wide), "
                     f"{int((Z[Z.axis == m].kind == 'sharp').sum())} sharp", fontsize=5.6, color=MODALITY[m], pad=2)
        out[f"gradient_{m}"] = dict(n_gradual=len(z), align=a["median_align"], null=a["null_align_median"], p=a["p_align"],
                                    zones={int(r.zone): ZONE_TAG.get((m, r.zone), "") for _, r in z.iterrows()})
    for j, (m, base, lab, bl) in enumerate([("ATAC", "LSI", "ATAC LSI", "LSI + Mellon"), ("RNA", "PCA", "RNA PCA", "PCA + Mellon")]):
        ax = fig.add_subplot(gs[2 + j]); axs.append(ax)
        low = b[f"low_{base}_10"].values.astype(bool); tr = g[f"transition_{m}"].values.astype(bool)
        F2.spatial(ax, D["xy"], None, colors="#e6e6e6", s=1.1)
        ax.scatter(*D["xy"][low].T, s=1.5, lw=0, color=METHOD["Mellon"], rasterized=True)
        ax.scatter(*D["xy"][tr].T, s=2.0, facecolor="none", edgecolor="#1a1a1a", lw=0.25, rasterized=True)
        for i, u in enumerate(_units(D, m)["Mellon"]["units"][:3]):
            ax.text(*D["xy"][u].mean(0), str(i + 1), fontsize=5.5, ha="center", va="center", fontweight="bold",
                    bbox=dict(fc="white", ec=METHOD["Mellon"], lw=0.5, boxstyle="circle,pad=0.15"))
        ax.set_title(f"Mellon on {lab}\nlowest-density 10% of spots", fontsize=5.6, color=METHOD["Mellon"], pad=2)
        rec = float((low & tr).sum() / tr.sum()); r_ = BS[(BS.modality == m) & (BS.low_fraction == 0.1)].iloc[0]
        out[f"mellon_{m}"] = dict(recall=rec, chance=float(r_.chance_recall), depth_low=int(r_.depth_low), depth_rest=int(r_.depth_rest))
    h = [Patch(fc=C_ATAC, label="gradual zone (ATAC)"), Patch(fc=C_RNA, label="gradual zone (RNA)"),
         Patch(fc="#9aa2aa", label="sharp interface (≤2 spots, spot mixing)"), Patch(fc=METHOD["Mellon"], label="Mellon low density"),
         Line2D([], [], marker="o", ls="", mfc="none", mec="#1a1a1a", mew=0.5, ms=3, label="gradual-zone spots (reference)")]
    axs[0].legend(handles=h, fontsize=5, ncol=5, loc="upper left", bbox_to_anchor=(0.0, -0.02), frameon=False,
                  handlelength=0.9, columnspacing=0.8, handletextpad=0.3)
    return axs[0], out


PROG = [("ann_mel_marker", "melanocytic", PROGRAM["melanocytic"]), ("ann_immune_marker", "immune", PROGRAM["immune"]),
        ("ann_stromal_marker", "stroma", PROGRAM["stroma"]), ("ann_cnv_burden_v2", "CNV burden", "#6B6B6B")]
NBIN, NPERM, MIN_UNIT, PITCH = 5, 500, 5, 129.0      # PITCH: DBiT spot spacing in the registered image frame


def _units(D, m):
    """the transition units each method proposes, with the coordinate it orders them by.
    Saturn: each connected gradient zone, ordered by rho_m.  Mellon: each connected patch (spots within 1.6 pitch)
    of its lowest-density 10%, ordered by log-density.  Units of >= MIN_UNIT spots, numbered by size."""
    from scipy.spatial import cKDTree
    from scipy.sparse.csgraph import connected_components
    import scipy.sparse as sp_
    _, g, b = _obs(D); base = {"ATAC": "LSI", "RNA": "PCA"}[m]; out = {}
    zone = g[f"zone_{m}"].values; tr = g[f"transition_{m}"].values.astype(bool)
    low = b[f"low_{base}_10"].values.astype(bool); idx = np.where(low)[0]
    pr = cKDTree(D["xy"][idx]).query_pairs(1.6 * PITCH, output_type="ndarray")
    _, l = connected_components(sp_.coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(idx),) * 2), directed=False)
    patch = np.full(len(low), -1); patch[idx] = l
    for meth, lab, sel, coord in (("Saturn", zone, tr, D["d"][m]), ("Mellon", patch, low, b[f"logdens_{base}"].values)):
        U = [(lab == u) & sel for u in np.unique(lab[sel])]
        U = sorted([u for u in U if u.sum() >= MIN_UNIT], key=lambda u: -u.sum())
        out[meth] = dict(units=U, coord=coord, n_flagged=int(sel.sum()))
    return out


def _zprog(D):
    return np.c_[[(np.asarray(D["z"][k], float) - np.nanmean(D["z"][k])) / np.nanstd(D["z"][k]) for k, _, _ in PROG]].T


def _best_r(M, c):
    from scipy.stats import rankdata
    rc = rankdata(c); rc = (rc - rc.mean()) / rc.std()
    RM = np.apply_along_axis(rankdata, 0, M); RM = (RM - RM.mean(0)) / RM.std(0)
    return float(np.max(np.abs(RM.T @ rc / len(c))))


def panel_profiles(fig, spec, D):
    """programme scores across the largest unit of each method, binned by the method's own coordinate"""
    M = _zprog(D); gs = GridSpecFromSubplotSpec(1, 4, subplot_spec=spec, wspace=0.45); axs, out = [], {}
    for j, (m, meth) in enumerate([("ATAC", "Saturn"), ("ATAC", "Mellon"), ("RNA", "Saturn"), ("RNA", "Mellon")]):
        U = _units(D, m)[meth]; s = U["units"][0]; c = U["coord"][s]
        ax = fig.add_subplot(gs[j]); axs.append(ax)
        q = np.quantile(c, np.linspace(0, 1, NBIN + 1)); bi = np.clip(np.digitize(c, q[1:-1]), 0, NBIN - 1)
        out[f"{m} {meth}"] = dict(n=int(s.sum()))
        for i, (k, lab, col) in enumerate(PROG):
            v = M[s, i]; mu = np.array([v[bi == b_].mean() for b_ in range(NBIN)])
            se = np.array([v[bi == b_].std() / np.sqrt(max((bi == b_).sum(), 1)) for b_ in range(NBIN)])
            ax.plot(range(NBIN), mu, "-o", color=col, ms=2, lw=0.9, label=lab)
            ax.fill_between(range(NBIN), mu - se, mu + se, color=col, alpha=0.15, lw=0)
            out[f"{m} {meth}"][lab] = round(float(spearmanr(v, c)[0]), 3)
        ax.axhline(0, color="#9aa2aa", lw=0.4, ls=":")
        ax.set_xticks([0, NBIN - 1])
        if meth == "Saturn":
            ax.set_xticklabels(["low ρ", "high ρ"], fontsize=5.2)
            ax.set_title(f"Saturn ρ$_{{\\mathrm{{{m}}}}}$ zone 1\n{int(s.sum())} spots", fontsize=5.5, color=MODALITY[m], pad=2)
        else:
            ax.set_xticklabels(["lowest", "higher"], fontsize=5.2)
            ax.set_title(f"Mellon {m} patch 1\n{int(s.sum())} spots", fontsize=5.5, color=METHOD["Mellon"], pad=2)
        if j == 0: ax.set_ylabel("programme score\n(z across tissue)")
        ax.set_xlabel("ρ" if meth == "Saturn" else "density", labelpad=0, fontsize=5.3)
    axs[0].legend(fontsize=5, loc="upper left", bbox_to_anchor=(0.0, -0.32), ncol=4, frameon=False, handlelength=1.2, columnspacing=0.8)
    return axs[0], out


def panel_order(ax, D):
    """the n comparison for EVERY unit: how strongly the method's own coordinate orders the programmes inside the unit
    (best |Spearman| over the 4 programmes) above a permutation null of the same unit size (coordinate shuffled
    within the unit, NPERM times).  Excess = observed - null median; filled = permutation p < 0.05."""
    M = _zprog(D); rng = np.random.default_rng(42); out = {}
    for j, m in enumerate(["ATAC", "RNA"]):
        Us = _units(D, m)
        for dx, meth, col in ((-0.17, "Saturn", MODALITY[m]), (0.17, "Mellon", METHOD["Mellon"])):
            U = Us[meth]; rows = []
            for i, s in enumerate(U["units"]):
                c = U["coord"][s]; o = _best_r(M[s], c)
                nul = np.array([_best_r(M[s], rng.permutation(c)) for _ in range(NPERM)])
                rows.append(dict(unit=i + 1, n=int(s.sum()), best_r=round(o, 3), null_median=round(float(np.median(nul)), 3),
                                 p=round(float((1 + (nul >= o).sum()) / (NPERM + 1)), 4)))
            R_ = pd.DataFrame(rows); ex = R_.best_r - R_.null_median; sig = R_.p < 0.05
            x = j + dx + np.random.default_rng(j).uniform(-0.06, 0.06, len(R_))
            ax.scatter(x[sig], ex[sig], s=R_.n[sig] * 0.35 + 3, color=col, lw=0.4, edgecolor="#1a1a1a", zorder=3)
            ax.scatter(x[~sig], ex[~sig], s=R_.n[~sig] * 0.35 + 3, facecolor="white", edgecolor=col, lw=0.7, zorder=3)
            frac = float(R_.n[sig].sum() / U["n_flagged"])
            ax.text(j + dx, 0.6, f"{frac:.0%}", ha="center", fontsize=5.3, color=col, fontweight="bold")
            out[f"{m}|{meth}"] = dict(units=rows, frac_flagged_spots_ordered=round(frac, 3))
    ax.axhline(0, color="#1a1a1a", lw=0.5)
    ax.set_xticks([-0.17, 0.17, 0.83, 1.17]); ax.set_xticklabels(["Saturn", "Mellon", "Saturn", "Mellon"], fontsize=5.2)
    for t, c in zip(ax.get_xticklabels(), [MODALITY["ATAC"], METHOD["Mellon"], MODALITY["RNA"], METHOD["Mellon"]]): t.set_color(c)
    ax.text(0, 0.7, "ATAC", ha="center", fontsize=5.8, color=MODALITY["ATAC"]); ax.text(1, 0.7, "RNA", ha="center", fontsize=5.8, color=MODALITY["RNA"])
    ax.axvline(0.5, color="#9aa2aa", lw=0.5, ls=":")
    ax.set_xlim(-0.55, 1.55); ax.set_ylim(-0.25, 0.8)
    ax.set_ylabel("ordering of programmes\nby the method (|r| − null)")
    ax.set_title("every zone / patch (size = spots)\nfilled: permutation p < 0.05\n% = flagged spots in an ordered unit", fontsize=5.2, pad=3)
    return out


PANELS = F3.PANELS + [
    ("k", (7.2, 1.75), panel_fate_arrows),
    ("l", (7.2, 1.9), panel_transition_maps),
    ("m", (5.0, 1.6), panel_profiles),
    ("n", (2.0, 1.7), F3._single(panel_order)),
]
SUPP = [("S3a", (7.2, 1.85), lambda fig, spec, D: panel_fate_arrows(fig, spec, D, background="domain")),
        ("S3b", (1.8, 1.6), F3._single(panel_fusion)),
        ("S3c", (5.4, 1.6), panel_fate_maps)]


def export_panels(D, outdir, zoom):
    outdir.mkdir(parents=True, exist_ok=True)
    for L, size, build in PANELS:
        fig = plt.figure(figsize=size)
        spec = GridSpec(1, 1, figure=fig, left=0.14, right=0.88, top=0.8, bottom=0.25)[0]
        if L == "i":
            F3.panel_istar_zoom(fig, spec, zoom)
        else:
            build(fig, spec, D)
        save_dual(fig, outdir / f"fig3{L}", write_manifest=False); plt.close(fig)
    print(f"-> {len(PANELS)} panels in {outdir}")


def main():
    apply(base_pt=6.5)
    D = F3.load(); F3.D_GLOBAL.update(D)
    fig = plt.figure(figsize=(7.2, 15.2))
    outer = GridSpec(8, 1, figure=fig, height_ratios=[1.5, 1.45, 2.0, 1.6, 1.9, 1.55, 1.75, 1.65], hspace=0.55,
                     left=0.07, right=0.95, top=0.98, bottom=0.04)
    g1 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], width_ratios=[1.6, 1.0, 1.0], wspace=0.45)
    g4 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[3], width_ratios=[1.85, 0.85, 0.28], wspace=0.25)
    g5 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[4], width_ratios=[1.35, 1.0, 1.05], wspace=0.3)
    g5b = GridSpecFromSubplotSpec(2, 1, subplot_spec=g5[1], height_ratios=[1, 0.25], hspace=0.05)
    g6 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[5], width_ratios=[3.2, 1.0], wspace=0.3)
    g8 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[7], width_ratios=[4.0, 1.45], wspace=0.35)
    where = dict(a=g1[0], b=g1[1], c=g1[2], d=outer[1], e=outer[2], f=g4[0], g=g4[1], h=g5[0], i=g5b[0], j=g5[2],
                 k=g6[0], l=g6[1], m=outer[6], n=g8[0], o=g8[1])
    nudge = dict(a=(-0.03, 0.015), b=(-0.06, 0.015), c=(-0.06, 0.015), d=(-0.03, 0.02), e=(-0.06, 0.03),
                 f=(-0.05, 0.03), g=(-0.05, 0.03), h=(-0.02, 0.02), i=(-0.02, 0.02), j=(-0.06, 0.03),
                 k=(-0.03, 0.025), l=(-0.07, 0.015), m=(-0.03, 0.025), n=(-0.06, 0.03), o=(-0.07, 0.02))
    res = {}
    for L, _, build in PANELS:
        if L == "i":
            ax = F3.panel_istar_zoom(fig, where["i"], res["h"]["zoom_px"]); out = None
        else:
            ax, out = build(fig, where[L], D)
        res[L] = out; F2.letter(ax, L, *nudge[L])
    F2.draw_letters(fig)
    old = json.load(open(F3.FINAL / "fig3_melanoma.json")) if (F3.FINAL / "fig3_melanoma.json").exists() else {}
    manifest = dict(old, cellrank_fate=res["k"], cellrank_fusion=res["l"], transition_vs_mellon=res["m"], zone_profiles=res["n"], ordering=res["o"], v2_sources=[str(CRK), str(GRD)])
    save_dual(fig, OUT, extra=manifest); save_dual(fig, FINAL / "fig3_melanoma", extra=manifest); plt.close(fig)
    export_panels(D, FINAL / "panels", res["h"]["zoom_px"])
    print(json.dumps(dict(m=res["m"], n=res["n"], o=res["o"]), indent=1, default=float))


if __name__ == "__main__":
    main()
