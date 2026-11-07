"""Figure 2 — LPC demyelinating lesion: transcription and chromatin order the inflammation differently.

Saturn fit `lpc10_fov` (spatial RNA + ATAC, LPC day 10, lesion-plus-distal-white-matter FOV).

  a  CODEX of the section: CD11c (activated myeloid) / MBP (myelin) / DAPI, FOV and the two
     CD11c+ foci used to place it
  b  fit quality: R2 full / own-axis only / cross-axes only per modality
  c  the two layers: lesion vs distal-WM focus on rho_RNA and on rho_ATAC
  d  rho_RNA, RNA domains, rho_ATAC, ATAC domains
  e  representative genes on rho_RNA (Trem2, Mbp) and peaks on rho_ATAC (Tyrobp promoter,
     Pak3 enhancer): spatial map + piecewise function along the own axis
  f  TF activity (GC/accessibility-matched chromVAR) along binned rho_ATAC
  g  SPI1 and POU3F3 activity in space
  h  cis gene-peak slope-sign concordance vs TSS distance, against a distal (>5 Mb) control
  i  gene-peak gradient agreement ALONG vs ACROSS rho vs TSS distance
  j  inflammatory-program genes: R2 on rho_RNA vs rho_ATAC

Domain colours are assigned by ANNOTATED label (composition-verified, asserted below),
never by domain index. All statistics are computed here from the pipeline outputs.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import mannwhitneyu, wilcoxon, spearmanr, chi2_contingency
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

sys.path.insert(0, "<workspace>/Saturn/published_figures/_style")
sys.path.insert(0, "<workspace>/saturn_repo/src")       # Saturn package (saturn_repo)
sys.path.insert(0, "<workspace>/Hole/gaston_mix_mm/scripts")
from paper_style import apply, save_dual, MODALITY, DOMAIN, REGION, LINEAGE, CMAP, PALETTE, ANCHOR

H = "<workspace>"
R = Path(f"{H}/Hole/gaston_mix_mm/results/real_saturn")
CODEX = Path(f"{H}/external_data/codex_lpc")
OUT = Path(f"{H}/Saturn/published_figures/fig2/fig2_lpc")
FINAL = Path(f"{H}/Saturn/final_fig2")          # assembled figure + one SVG/PNG per panel
DS, T_LRT, NB = "lpc10_fov", 0.1, {"RNA": 30, "ATAC": 20}
FEATS = [("RNA", "Trem2", "Trem2", "DAM receptor"),
         ("RNA", "Mbp", "Mbp", "myelin"),
         ("ATAC", "chr7:30412652-30413002", "Tyrobp promoter", "TREM2 adaptor (DAP12)"),
         ("ATAC", "chr2:91084599-91084795", "Spi1 +2 kb enhancer", "microglial PU.1")]
LINEAGE_ORDER = ["microglia", "stress", "astrocyte", "OL", "neuron"]
KB_EDGES = [0, 25, 100, 250, 500, 1000]
TF_MIN_R2 = 0.15            # TFs below this on rho_ATAC are left to the supplementary heatmap
C_RNA, C_ATAC = MODALITY["RNA"], MODALITY["ATAC"]
INK, GRID = PALETTE["black"], ANCHOR["stone"]
CODEX_CH = [("CD11c", "#B0473C", 1.0), ("MBP", "#5F9AA3", 0.8), ("DAPI", "#6A7595", 0.3)]   # stains on white (subtractive)
FOCUS = ANCHOR["sand"]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load():
    from saturn.domains import gaston_domains
    from domain_annot import CFG
    from domain_k import kdom
    ax = pd.read_csv(R / DS / "axes.csv", index_col=0)
    z = dict(np.load(R / f"targets_{DS}.npz", allow_pickle=True))
    D = dict(xy=np.c_[ax.x.values, ax.y.values], bc=[str(b) for b in ax.index],
             d={"RNA": ax.d_RNA.values, "ATAC": ax.d_ATAC.values},
             lab={m: gaston_domains(ax[f"d_{m}"].values, z[f"Y_{m}"], n_domains=kdom(DS, m)) for m in ("RNA", "ATAC")},
             region=pd.Series(z["ann_region"].astype(str)).replace({"distal_WM": "distal WM focus"}).values,
             micro=z["ann_microglia_z"].astype(float), myelin=z["ann_myelin_z"].astype(float),
             summary=json.load(open(R / DS / "summary.json")))
    idx = pd.Index(D["bc"])
    r = CFG[DS]["rna"]()[idx]; a = CFG[DS]["atac"]()[idx]
    D["X"] = {"RNA": sp.csc_matrix(r.X).astype(float), "ATAC": sp.csc_matrix(a.X).astype(float)}
    D["names"] = {"RNA": np.array([str(v) for v in r.var_names]), "ATAC": np.array([str(v) for v in a.var_names])}
    D["lib"] = {m: np.asarray(D["X"][m].sum(1)).ravel().clip(1) for m in D["X"]}
    D["names_dom"] = label_domains(D)
    return D


def label_domains(D):
    """map domain index -> annotated label from composition; asserts the reading holds.
    K per axis is the elbow K (domain_k.kdom): RNA 3, ATAC 2. At RNA K = 3 the periventricular band
    joins the thalamic end (355 of its 516 spots; the rest go to white matter)."""
    out = {}
    lab, reg = D["lab"]["RNA"], D["region"]
    K = range(lab.max() + 1)
    les = max(K, key=lambda k: (reg[lab == k] == "lesion").mean())
    rest = [k for k in K if k != les]
    wm = max(rest, key=lambda k: np.median(D["myelin"][lab == k])); rest.remove(wm)
    if len(rest) == 1:
        th = rest[0]
        assert np.median(D["micro"][lab == th]) < 0, "thalamic RNA domain should be microglia-poor"
        out["RNA"] = {les: "lesion core", wm: "white matter", th: "thalamic & periventricular"}
    else:
        pv = max(rest, key=lambda k: (reg[lab == k] == "distal WM focus").mean()); rest.remove(pv)
        th = rest[0]
        assert np.median(D["micro"][lab == th]) < 0, "thalamic RNA domain should be microglia-poor"
        out["RNA"] = {les: "lesion core", wm: "white matter", pv: "periventricular", th: "thalamic"}

    lab, d = D["lab"]["ATAC"], D["d"]["ATAC"]
    K = range(lab.max() + 1)
    inf = max(K, key=lambda k: np.median(D["micro"][lab == k]))
    assert (reg[lab == inf] == "distal WM focus").mean() > 0.1 and (reg[lab == inf] == "lesion").mean() > 0.2
    c0 = np.median(d[lab == inf])
    rest = sorted([k for k in K if k != inf], key=lambda k: abs(np.median(d[lab == k]) - c0))
    out["ATAC"] = {inf: "inflammatory chromatin", **dict(zip(rest, ["parenchymal", "transitional", "thalamic"]))}
    return out


def dcol(D, m):
    return np.array([DOMAIN[D["names_dom"][m][k]] for k in range(D["lab"][m].max() + 1)])


def feature_function(D, m, feat):
    from saturn.piecewise import fit_poisson
    from saturn.domains import isodepth_bins
    j = int(np.where(D["names"][m] == feat)[0][0])
    c = np.asarray(D["X"][m][:, j].todense()).ravel()
    lib, lab, d = D["lib"][m], D["lab"][m], D["d"][m]
    sl, it = (v[0] for v in fit_poisson(c[:, None], lib, lab, d, t=T_LRT))   # GASTON's per-domain Poisson fit
    bi, _, kp = isodepth_bins(d, NB[m])
    db = np.array([d[bi == b].mean() for b in kp])
    ob = np.log1p(np.array([c[bi == b].sum() / lib[bi == b].sum() for b in kp]) * 1e6)
    lb = np.array([np.bincount(lab[bi == b], minlength=lab.max() + 1).argmax() for b in kp])
    # R2 on the RATE scale, exactly as the pipeline's screen computes it (marker_function_panels._chunk),
    # so the value printed here matches the supplementary screen tables; the plot itself is on log scale
    y = np.array([c[bi == b].sum() for b in kp]) / np.maximum(np.array([lib[bi == b].sum() for b in kp]), 1)
    pr = np.exp(it[lb] + sl[lb] * db)
    r2 = 1 - ((y - pr) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    spot = np.log1p(np.exp(it[lab] + sl[lab] * d) * 1e4)      # fitted function at every spot, log CP10k
    return dict(c=c, lib=lib, sl=sl, it=it, db=db, ob=np.log1p(y * 1e6), lb=lb, r2=r2, n=int(c.sum()), spot=spot,
                n_sloped=int((np.abs(sl) > 1e-9).sum()))


def smooth(v, xy, k=8):
    from scipy.spatial import cKDTree
    _, nn = cKDTree(xy).query(xy, k=k)
    return v[nn].mean(1)


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #
LETTERS = []


def letter(ax, s, dx=-0.035, dy=0.012):
    """queued; drawn in figure coordinates once the layout is final"""
    LETTERS.append((ax, s, dx, dy))


def draw_letters(fig):
    for ax, s, dx, dy in LETTERS:
        b = ax.get_position()
        fig.text(b.x0 + dx, b.y1 + dy, s, fontsize=9, fontweight="bold", va="bottom", ha="left")


def spatial(ax, xy, v, cmap=None, colors=None, vmin=None, vmax=None, s=1.1):
    if colors is not None:
        ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=s, lw=0, rasterized=True)
        sc = None
    else:
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=v, cmap=cmap, s=s, lw=0, vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sd in ax.spines.values():
        sd.set_visible(False)
    return sc


def cbar(fig, ax, sc, label, ticks=None, on_top=False):
    cax = ax.inset_axes([1.02, 0.1, 0.045, 0.8] if not on_top else [1.03, 0.05, 0.05, 0.72])
    cb = fig.colorbar(sc, cax=cax, ticks=ticks)
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4)
    if on_top:                                   # label above the bar: never collides with tick labels
        cb.ax.set_title(label, fontsize=5.5, pad=2, loc="left")
    else:
        cb.set_label(label, fontsize=5.5, labelpad=1)


REG_CSV = Path(f"{H}/catalign_lpc10/results/LPC10S1_wm/LPC10S1_spots_in_codex.csv")   # Catalign DBiT->CODEX


def codex_rgb():
    """CODEX overview (level 3) as stains on white: each channel absorbs the complement of its colour"""
    z = np.load(CODEX / "overview_LPC10_S1_lvl3.npz")
    def nrm(a, lo=1, hi=99.6):
        a = a.astype(float); l, h_ = np.percentile(a[a > 0], [lo, hi])
        return np.clip((a - l) / (h_ - l + 1e-9), 0, 1)
    return np.clip(1 - sum(nrm(z[k])[..., None] * (1 - np.array(to_rgb(c))) * g for k, c, g in CODEX_CH), 0, 1)


def fov_on_codex(D, f=8.0):
    """registered positions of the Saturn FOV spots on CODEX (level-0 px / f), in D order, + outline polygon"""
    from scipy.spatial import ConvexHull
    S = pd.read_csv(REG_CSV, index_col=0).loc[D["bc"]]
    xy = np.c_[S.codex_lvl0_x.values, S.codex_lvl0_y.values] / f
    return xy, xy[ConvexHull(xy).vertices]


def _stain_key(ax):
    from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox
    keys = HPacker(children=[TextArea(k, textprops=dict(color=c, fontsize=6)) for k, c, _ in CODEX_CH],
                   align="baseline", pad=0, sep=4)                 # packed in points: never overlaps at any size
    ax.add_artist(AnchoredOffsetbox(loc="lower left", child=keys, frameon=False, pad=0.15, borderpad=0.2))


def _clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sd in ax.spines.values():
        sd.set_visible(False)


def panel_codex(ax, D):
    """whole CODEX section; dashed outline = where the DBiT FOV lands after Catalign registration"""
    from matplotlib.patches import Polygon
    rgb = codex_rgb()
    ax.imshow(rgb[::2, ::2], interpolation="bilinear")
    _, hull = fov_on_codex(D, f=16.0)                              # level 3 shown at half size
    ax.add_patch(Polygon(hull, closed=True, fill=False, ec=INK, lw=0.9, ls=(0, (3, 2))))
    top = hull[np.argmin(hull[:, 1])]
    ax.text(top[0] - 8, top[1] - 4, "DBiT FOV", color=INK, fontsize=6, va="bottom", ha="right")
    _clean(ax); _stain_key(ax)
    return dict(fov_outline_lvl3_half=hull.round(1).tolist())


def panel_codex_zoom(ax, D, pad=60):
    """CODEX around the FOV with the registered DBiT spots: lesion and distal-WM focus on the CD11c+ foci"""
    from matplotlib.patches import Polygon
    rgb = codex_rgb()
    xy, hull = fov_on_codex(D)                                     # level-3 px
    x0, y0 = (hull.min(0) - pad).astype(int); x1, y1 = (hull.max(0) + pad).astype(int)
    ax.imshow(rgb[y0:y1, x0:x1], extent=(x0, x1, y1, y0), interpolation="bilinear")
    reg = D["region"]
    for r_ in ("lesion", "distal WM focus"):
        m = reg == r_
        ax.scatter(xy[m, 0], xy[m, 1], s=0.5, lw=0, color=REGION[r_], alpha=0.75, rasterized=True)
    ax.add_patch(Polygon(hull, closed=True, fill=False, ec=INK, lw=0.7, ls=(0, (3, 2))))
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); _clean(ax)
    ax.set_title("DBiT spots on CODEX", fontsize=6, pad=2)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=2.6, color=REGION[r]) for r in ("lesion", "distal WM focus")],
              labels=["lesion", "distal WM focus"], fontsize=5.3, loc="upper center", bbox_to_anchor=(0.5, -0.01),
              ncol=1, handletextpad=0.1, borderaxespad=0, labelspacing=0.15, frameon=False)
    return {r_: int((reg == r_).sum()) for r_ in ("lesion", "distal WM focus")}


def build_codex(fig, spec, D):
    g = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[1.5, 1], wspace=0.22)
    a = fig.add_subplot(g[0]); b = fig.add_subplot(g[1])
    info = dict(overview=panel_codex(a, D), zoom=panel_codex_zoom(b, D))
    from matplotlib.patches import ConnectionPatch
    hull = np.array(info["overview"]["fov_outline_lvl3_half"])
    tip = hull[np.argmax(hull[:, 0])]                              # right corner of the FOV outline -> close-up
    fig.add_artist(ConnectionPatch(xyA=tip, coordsA=a.transData, xyB=(-0.02, 0.5), coordsB=b.transAxes,
                                   arrowstyle="-|>", mutation_scale=7, lw=0.9, color=INK))
    return a, info


def panel_fit(ax, S):
    R2 = S["R2_whole"]; mods = ["RNA", "ATAC"]
    x = np.arange(2); w = 0.26
    for i, (key, lab, alpha) in enumerate([("full", "full", 1.0), ("own_axis_only", "own axis", 0.65),
                                           ("other_axes_only", "cross axis", 0.3)]):
        v = [R2[m][key] for m in mods]
        ax.bar(x + (i - 1) * w, v, w, color=[MODALITY[m] for m in mods], alpha=alpha,
               edgecolor=INK, lw=0.4, label=lab)
        for xx, vv in zip(x + (i - 1) * w, v):
            ax.text(xx, vv + 0.015, f"{vv:.2f}", ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks(x); ax.set_xticklabels(mods)
    for tl, m in zip(ax.get_xticklabels(), mods):
        tl.set_color(MODALITY[m])
    ax.set_ylim(0, 0.8); ax.set_yticks([0, 0.4, 0.8]); ax.set_ylabel("variance explained (R²)")
    ax.set_xlim(-0.55, 1.55)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc="#555555", alpha=a_, ec=INK, lw=0.4, label=l_) for a_, l_ in
                       [(1, "full"), (0.65, "own axis"), (0.3, "cross axis")]],
              fontsize=5.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=0.9,
              handleheight=0.8, columnspacing=0.6, handletextpad=0.3, borderaxespad=0.1)


def panel_layers(ax, D):
    """where the two inflammatory foci sit on each axis.

    Each rho is converted to a percentile rank (oriented so the lesion sits at the low end)
    so the two axes share one scale. rho_RNA puts the lesion and the distal-WM focus at
    opposite ends; rho_ATAC puts both at the inflammatory end.
    """
    reg = D["region"]; rng = np.random.default_rng(1)
    order = ["lesion", "distal WM focus", "other"]
    auc, med = {}, {}
    for g, m in enumerate(("RNA", "ATAC")):
        v = D["d"][m].copy()
        if np.median(v[reg == "lesion"]) > np.median(v):
            v = -v
        pr = (np.argsort(np.argsort(v)) + 0.5) / len(v) * 100
        for i, r_ in enumerate(order):
            y = pr[reg == r_]
            if r_ == "other":
                y = rng.choice(y, 600, replace=False)
            x = g * 3.4 + i + rng.uniform(-0.28, 0.28, len(y))
            ax.scatter(x, y, s=1.0, lw=0, color=REGION[r_], alpha=0.55 if r_ != "other" else 0.8,
                       rasterized=True)
            mm = np.median(pr[reg == r_]); med[(m, r_)] = mm
            ax.plot([g * 3.4 + i - 0.34, g * 3.4 + i + 0.34], [mm, mm], color=INK, lw=1.1)
        a_ = v[reg == "lesion"]; b_ = v[reg == "distal WM focus"]
        u = mannwhitneyu(a_, b_)
        au = u.statistic / (len(a_) * len(b_)); au = max(au, 1 - au)
        auc[m] = (au, u.pvalue)
        ax.text(g * 3.4 + 1, 104, f"lesion vs focus\nAUC {au:.2f}", ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks([1, 4.4])
    ax.set_xticklabels(["ρ$_{\\mathrm{RNA}}$", "ρ$_{\\mathrm{ATAC}}$"])
    for tl, m in zip(ax.get_xticklabels(), ("RNA", "ATAC")):
        tl.set_color(MODALITY[m])
    ax.set_ylim(0, 100); ax.set_yticks([0, 50, 100]); ax.set_ylabel("position along axis (percentile)")
    ax.set_xlim(-0.6, 6.0)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=3, color=REGION[r]) for r in order],
              labels=order, fontsize=5.3, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
              handletextpad=0.1, columnspacing=0.5, borderaxespad=0)
    return auc, med


def region_contours(ax, D, lw=0.7):
    """outline of the two CD11c+ foci (lesion solid, distal WM focus dashed): spots rasterised on the DBiT grid,
    holes filled, largest piece only, lightly smoothed, contoured at 0.5"""
    if "region" not in D:                        # only the LPC data have CD11c+ foci (panel_maps is shared with Fig 3)
        return
    from scipy.ndimage import binary_fill_holes, binary_closing, gaussian_filter, label
    xy = D["xy"]; pitch = np.median(np.diff(np.unique(xy[:, 0])))
    ix = np.round((xy - xy.min(0)) / pitch).astype(int); n = ix.max(0) + 3
    gx = xy[:, 0].min() + (np.arange(n[0]) - 1) * pitch; gy = xy[:, 1].min() + (np.arange(n[1]) - 1) * pitch
    for r_, ls in (("lesion", "-"), ("distal WM focus", (0, (2.5, 1.5)))):
        M = np.zeros(n, bool); m = D["region"] == r_; M[ix[m, 0] + 1, ix[m, 1] + 1] = True
        M = binary_fill_holes(binary_closing(M, iterations=2))
        cc, ncc = label(M)                       # keep the main focus only; small satellites get no outline
        if ncc > 1:
            M = cc == np.bincount(cc.ravel())[1:].argmax() + 1
        ax.contour(gx, gy, gaussian_filter(M.astype(float), 0.8).T, levels=[0.5], colors=[INK], linewidths=lw,
                   linestyles=[ls])
        i_, j_ = np.nonzero(M)                   # name the focus just above its outline
        x_ = min(gx[i_].mean() + 0.3 * (xy[:, 0].max() - xy[:, 0].min()), xy[:, 0].max())   # stays inside the map
        ax.text(x_, gy[j_].max() + 1.5 * pitch, r_, fontsize=5, ha="right", va="bottom", color=INK,
                path_effects=[pe.withStroke(linewidth=1.2, foreground="white", alpha=0.85)], zorder=6)


def panel_maps(fig, spec, D):
    gs = GridSpecFromSubplotSpec(2, 4, subplot_spec=spec, height_ratios=[1, 0.28], wspace=0.18, hspace=0.02)
    out = []
    for j, m in enumerate(("RNA", "ATAC")):
        a1 = fig.add_subplot(gs[0, 2 * j]); a2 = fig.add_subplot(gs[0, 2 * j + 1])
        d = D["d"][m]; lo, hi = np.percentile(d, [1, 99])
        sc = spatial(a1, D["xy"], d, cmap=CMAP["score"], vmin=lo, vmax=hi)
        cbar(fig, a1, sc, "isodepth", ticks=[round(lo, 1), round(hi, 1)])
        region_contours(a1, D)
        a1.set_title(f"ρ$_{{\\mathrm{{{m}}}}}$", color=MODALITY[m], fontsize=7, pad=2)
        cols = dcol(D, m)[D["lab"][m]]
        spatial(a2, D["xy"], None, colors=cols)
        region_contours(a2, D)
        a2.set_title(f"{m} domains", color=MODALITY[m], fontsize=7, pad=2)
        lg = fig.add_subplot(gs[1, 2 * j:2 * j + 2]); lg.set_axis_off()
        names = D["names_dom"][m]
        order = np.argsort([np.median(d[D["lab"][m] == k]) for k in names])
        for i, k in enumerate(np.array(list(names))[order]):
            n = int((D["lab"][m] == k).sum())
            x_ = 0.02 + (i % 2) * 0.5; y_ = 0.72 - (i // 2) * 0.5
            lg.add_patch(Rectangle((x_, y_ - 0.12), 0.035, 0.3, color=DOMAIN[names[k]], transform=lg.transAxes))
            lg.text(x_ + 0.05, y_ + 0.03, f"{names[k]}  (n={n})", fontsize=5.8, va="center", transform=lg.transAxes)
        out.append((a1, a2))
    return out


def along(ax, m, d, lab, cols, bx, by, blab, curve, ylabel):
    """bin means (domain colour) and the fitted function per domain along the own axis"""
    for k in range(lab.max() + 1):
        s_ = blab == k
        ax.scatter(bx[s_], by[s_], s=6, color=cols[k], lw=0.3, edgecolor=INK, zorder=3)
        xs = np.linspace(d[lab == k].min(), d[lab == k].max(), 20)
        ax.plot(xs, curve(k, xs), color=cols[k], lw=1.3, zorder=2)
    for b in sorted([d[lab == k].max() for k in range(lab.max() + 1)])[:-1]:
        ax.axvline(b, color=GRID, lw=0.4, ls=(0, (2, 2)), zorder=1)
    ax.set_xlabel(f"ρ$_{{\\mathrm{{{m}}}}}$", color=MODALITY[m], labelpad=1)
    ax.set_ylabel(ylabel, labelpad=1); ax.tick_params(labelsize=5.5, pad=1)


def panel_feature(fig, spec, D, m, feat, title, sub):
    """one feature, three columns: ORIGINAL per-spot spatial distribution (log CP10k, no smoothing) | the FITTED
    function evaluated at every spot (same scale) | along the own axis: every spot (grey), bin means, fitted function"""
    gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, width_ratios=[1, 1, 1.05], wspace=0.95)
    F = feature_function(D, m, feat)
    lab, d = D["lab"][m], D["d"][m]; cols = dcol(D, m)
    v = np.log1p(F["c"] / F["lib"] * 1e4)
    lo, hi = np.percentile(np.r_[v, F["spot"]], [2, 99])
    a1, a2, a3 = (fig.add_subplot(gs[i]) for i in range(3))
    o = np.argsort(v)                                          # high values on top so sparse signal stays visible
    spatial(a1, D["xy"][o], v[o], cmap=CMAP["expression"], vmin=lo, vmax=hi, s=0.9)
    sc = spatial(a2, D["xy"], F["spot"], cmap=CMAP["expression"], vmin=lo, vmax=hi, s=0.9)
    cbar(fig, a2, sc, "log\nCP10k", on_top=True)
    a1.set_title(f"{title}\n{sub}", fontsize=6.2, pad=2, color=MODALITY[m], loc="left")
    a1.text(0.5, -0.04, "original", transform=a1.transAxes, ha="center", va="top", fontsize=5.5)
    a2.text(0.5, -0.04, "fitted function", transform=a2.transAxes, ha="center", va="top", fontsize=5.5)
    along(a3, m, d, lab, cols, F["db"], F["ob"], F["lb"],
          lambda k, xs: np.log1p(np.exp(F["it"][k] + F["sl"][k] * xs) * 1e6), "log(CPM+1)")
    a3.set_title(f"fitted function · R² {F['r2']:.2f}\n{F['n_sloped']}/{len(F['sl'])} domains sloped", fontsize=5.8, pad=2)
    return a1, F


def panel_tf_heat(fig, ax, cax_strip, D, ax_fit=None, strip_fit=None):
    b = pd.read_csv(R / f"{DS}_tf_activity" / "tf_bins.csv")
    f = pd.read_csv(R / f"{DS}_tf_activity" / "tf_fits.csv")
    b = b[b.axis == "ATAC"]; f = f[f.axis == "ATAC"].set_index("tf")
    Z = b.pivot(index="tf", columns="bin", values="z")
    rho = b.groupby("bin").rho.mean().loc[Z.columns].values
    dom = b.groupby("bin").domain.agg(lambda s: s.mode().iloc[0]).loc[Z.columns].values
    rows = []
    for L in LINEAGE_ORDER:
        tfs = [t for t in Z.index if f.loc[t, "lineage"] == L and f.loc[t, "R2"] >= TF_MIN_R2]
        tfs = sorted(tfs, key=lambda t: rho[np.argmax(Z.loc[t].values)])
        rows += tfs
    Zm = Z.loc[rows].values
    im = ax.imshow(Zm, aspect="auto", cmap=CMAP["score"], vmin=-3, vmax=3, interpolation="nearest")
    rax = ax
    if ax_fit is not None:                       # the same rows, the fitted piecewise function per bin
        Fm = b.pivot(index="tf", columns="bin", values="fitted").loc[rows, Z.columns].values
        ax_fit.imshow(Fm, aspect="auto", cmap=CMAP["score"], vmin=-3, vmax=3, interpolation="nearest")
        ax_fit.set_yticks([]); ax_fit.set_xticks([]); rax = ax_fit
        ax.set_xlabel("observed (bin z)", fontsize=5.8, labelpad=2)
        ax_fit.set_xlabel("fitted function", fontsize=5.8, labelpad=2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=5.5)
    ax.set_xticks([]); ax.tick_params(axis="y", length=0, pad=7)
    for i, t in enumerate(rows):
        ax.add_patch(Rectangle((-1.5, i - 0.5), 1.0, 1, color=LINEAGE[f.loc[t, "lineage"]], clip_on=False))
    for i, t in enumerate(rows):
        r2 = f.loc[t, "R2"]
        rax.text(Zm.shape[1] - 0.2, i, f" {r2:.2f}", fontsize=5.2, va="center", ha="left",
                fontweight="bold" if r2 >= 0.5 else "normal", color=INK if r2 >= 0.5 else "#8a939c")
    rax.text(Zm.shape[1] + 0.3, -0.9, "R²", fontsize=5.5, ha="left", va="bottom")
    cols = dcol(D, "ATAC")
    for st in [cax_strip] + ([strip_fit] if strip_fit is not None else []):
        st.imshow(dom[None, :], aspect="auto", cmap=ListedColormap(cols), vmin=-0.5, vmax=len(cols) - 0.5,
                  interpolation="nearest")
        st.set_xticks([]); st.set_yticks([])
        for sd in st.spines.values():
            sd.set_visible(False)
    ends = [D["names_dom"]["ATAC"][int(dom[0])], D["names_dom"]["ATAC"][int(dom[-1])]]
    cax_strip.set_title(f"ρ$_{{\\mathrm{{ATAC}}}}$, 60 equal-count bins:  {ends[0]}  →  {ends[1]}",
                        color=C_ATAC, fontsize=6, pad=2, loc="left")
    return im, [t for t in Z.index if t not in rows]


def tf_spot(D, tf):
    """per-spot chromVAR z (same motif hits and GC/accessibility-matched background as the bins) and its piecewise
    function of rho_ATAC fitted at SPOT level with the same domains and gate, so data and fit share one scale"""
    from saturn.piecewise import fit_gauss
    z = np.load(R / f"{DS}_tf_activity" / "tf_z_per_spot.npz", allow_pickle=True)
    order = pd.Index([str(b) for b in z["barcodes"]]).get_indexer(pd.Index(D["bc"]))
    assert (order >= 0).all()
    v = np.asarray(z["z"], float)[order, [str(t) for t in z["tfs"]].index(tf)]
    lab, d = D["lab"]["ATAC"], D["d"]["ATAC"]
    sl, it = fit_gauss(v[:, None], lab, d, t=T_LRT); sl, it = sl[0], it[0]
    fit = it[lab] + sl[lab] * d
    r2 = 1 - ((v - fit) ** 2).sum() / ((v - v.mean()) ** 2).sum()
    q = np.array_split(np.argsort(d), 30)
    return dict(v=v, fit=fit, sl=sl, it=it, r2=r2, bx=np.array([d[g].mean() for g in q]),
                by=np.array([v[g].mean() for g in q]),
                blab=np.array([np.bincount(lab[g], minlength=lab.max() + 1).argmax() for g in q]))


def panel_tf_space(fig, spec, D):
    """SPI1 / POU3F3: original per-spot activity | fitted function at every spot | along rho_ATAC"""
    gs = GridSpecFromSubplotSpec(2, 3, subplot_spec=spec, width_ratios=[1, 1, 1.05], wspace=1.0, hspace=0.55)
    lab, d = D["lab"]["ATAC"], D["d"]["ATAC"]; cols = dcol(D, "ATAC"); first = None
    for r, tf in enumerate(["SPI1", "POU3F3"]):
        T = tf_spot(D, tf); lim = np.percentile(np.abs(T["v"]), 98)
        a1, a2, a3 = (fig.add_subplot(gs[r, i]) for i in range(3))
        spatial(a1, D["xy"], T["v"], cmap=CMAP["score"], vmin=-lim, vmax=lim, s=0.9)
        sc = spatial(a2, D["xy"], T["fit"], cmap=CMAP["score"], vmin=-lim, vmax=lim, s=0.9)
        cbar(fig, a2, sc, "motif z", on_top=True)
        a1.set_title(f"{tf} activity", fontsize=6.2, pad=2, loc="left",
                     color=LINEAGE["microglia"] if tf == "SPI1" else LINEAGE["neuron"])
        a1.text(0.5, -0.04, "original (per spot)", transform=a1.transAxes, ha="center", va="top", fontsize=5.5)
        a2.text(0.5, -0.04, "fitted function", transform=a2.transAxes, ha="center", va="top", fontsize=5.5)
        along(a3, "ATAC", d, lab, cols, T["bx"], T["by"], T["blab"],
              lambda k, xs, T=T: T["it"][k] + T["sl"][k] * xs, "motif z")
        a3.set_title(f"fitted function · R² {T['r2']:.2f}", fontsize=5.8, pad=2)
        first = first or a1
    return first


def panel_tf_method(ax):
    """how TF activity is inferred: chromVAR deviations on Saturn isodepth bins, then a fitted function of rho"""
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    steps = [("ATAC peaks", "12,000 most-read peaks\nof the FOV"),
             ("motif hits", "JASPAR 2022 PWM scan,\nboth strands; hit = top 1%\nscore per TF (23 TFs)"),
             ("pool spots", "60 equal-count bins along\nρ$_{\\mathrm{ATAC}}$ (heatmap), or each\nspot alone (maps)"),
             ("chromVAR z", "motif-hit peaks vs 50\nbackground peak sets\nmatched on GC, accessibility"),
             ("fitted function", "piecewise-linear in ρ$_{\\mathrm{ATAC}}$,\none line per Saturn domain\n(F-test gate) → R²")]
    w, gap = 0.176, 0.026
    for i, (h, t) in enumerate(steps):
        x = 0.005 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.06), w, 0.86, boxstyle="round,pad=0,rounding_size=0.02",
                                    fc="white" if i < 4 else "#f4f6f8", ec=C_ATAC, lw=0.8, transform=ax.transAxes))
        ax.text(x + 0.012, 0.8, f"{i + 1}  {h}", fontsize=6, fontweight="bold", va="center", transform=ax.transAxes,
                color=INK)
        ax.text(x + 0.012, 0.62, t, fontsize=5.5, va="top", transform=ax.transAxes, color="#333333", linespacing=1.15)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.004, 0.49), (x + w + gap - 0.004, 0.49), transform=ax.transAxes,
                                         arrowstyle="-|>", mutation_scale=6, lw=0.8, color=INK))
    return ax


def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def panel_cosign(ax):
    P = pd.read_csv(R / f"{DS}_cis_pairs" / "cis_function_pairs.csv")
    stats = {}
    for m, ls in (("RNA", "-"), ("ATAC", "--")):
        s = P[P.axis == m]
        cis = s[s.kind == "cis"]; dist = s[s.kind == "distal"]
        b = np.digitize(np.abs(cis.kb), KB_EDGES[1:-1])
        xs, ys, lo, hi = [], [], [], []
        for i in range(len(KB_EDGES) - 1):
            g = cis[b == i].concordant
            xs.append(i); ys.append(g.mean())
            l_, h_ = wilson(g.sum(), len(g)); lo.append(l_); hi.append(h_)
        col = MODALITY[m]
        ax.plot(xs, ys, ls, color=col, marker="o", ms=2.5, lw=1.1, label=f"on ρ$_{{\\mathrm{{{m}}}}}$")
        ax.fill_between(xs, lo, hi, color=col, alpha=0.18, lw=0)
        base = dist.concordant.mean()
        ax.axhline(base, color=col, lw=0.7, ls=":", alpha=0.9)
        near = cis[np.abs(cis.kb) < KB_EDGES[1]].concordant
        tab = [[near.sum(), len(near) - near.sum()], [dist.concordant.sum(), len(dist) - dist.concordant.sum()]]
        stats[m] = dict(near=float(near.mean()), n_near=int(len(near)), distal=float(base),
                        n_distal=int(len(dist)), p=float(chi2_contingency(tab)[1]))
    ax.set_xticks(range(len(KB_EDGES) - 1))
    ax.set_xticklabels(["<25", "25–100", "100–250", "250–500", "500–1000"], rotation=30, ha="right", fontsize=5.5)
    ax.set_xlabel("gene–peak distance (kb)", labelpad=1)
    ax.set_ylabel("slope-sign concordance")
    from matplotlib.lines import Line2D
    h_, l_ = ax.get_legend_handles_labels()
    h_.append(Line2D([], [], color="#5c6670", ls=":", lw=0.9)); l_.append("distal control (>5 Mb)")
    ax.legend(h_, l_, fontsize=5.3, loc="upper right", handlelength=1.6, borderaxespad=0.1, labelspacing=0.25)
    return stats


def panel_gradient(ax):
    G = pd.read_csv(R / f"{DS}_cis_pairs" / "cis_gradient_decomp.csv")
    rng = np.random.default_rng(0); stats = {}
    for m, ls in (("RNA", "-"), ("ATAC", "--")):
        s = G[G.axis == m]; b = np.digitize(np.abs(s.kb), KB_EDGES[1:-1])
        for comp, col in (("r_along", MODALITY[m]), ("r_across", GRID)):
            xs, ys, lo, hi = [], [], [], []
            for i in range(len(KB_EDGES) - 1):
                v = s[comp].values[b == i]
                bs = [np.median(rng.choice(v, len(v))) for _ in range(200)]
                xs.append(i); ys.append(np.median(v))
                lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
            ax.plot(xs, ys, ls, color=col, marker="o", ms=2.3, lw=1.0,
                    label=f"{'along' if comp == 'r_along' else 'across'} ρ$_{{\\mathrm{{{m}}}}}$")
            ax.fill_between(xs, lo, hi, color=col, alpha=0.18, lw=0)
        stats[m] = dict(p_along_gt_across=float(wilcoxon(s.r_along, s.r_across, alternative="greater").pvalue),
                        rho_along_vs_kb=float(spearmanr(np.abs(s.kb), s.r_along).statistic),
                        n=int(len(s)))
    ax.set_xticks(range(len(KB_EDGES) - 1))
    ax.set_xticklabels(["<25", "25–100", "100–250", "250–500", "500–1000"], rotation=30, ha="right", fontsize=5.5)
    ax.set_xlabel("gene–peak distance (kb)", labelpad=1)
    ax.set_ylabel("gene–peak gradient agreement (r)")
    ax.legend(fontsize=5.2, loc="upper right", ncol=1, handlelength=1.5, borderaxespad=0.1, labelspacing=0.25)
    return stats


def panel_infl(ax):
    f = pd.read_csv(R / f"{DS}_posthoc" / "infl_gene_axis_fits.csv")
    col = np.where(f.R2_dATAC > f.R2_dRNA, C_ATAC, C_RNA)
    ax.scatter(f.R2_dRNA, f.R2_dATAC, s=9, c=col, lw=0.3, edgecolor=INK, zorder=3)
    ax.plot([0, 1], [0, 1], color=GRID, lw=0.6, ls=(0, (2, 2)), zorder=1)
    for g in ["Itgax", "Trem2", "Apoe", "Stat1"]:
        r = f[f.gene == g]
        if len(r):
            ax.annotate(g, (r.R2_dRNA.iloc[0], r.R2_dATAC.iloc[0]), fontsize=5.3, xytext=(2, 1),
                        textcoords="offset points", style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("R² on ρ$_{\\mathrm{RNA}}$", color=C_RNA, labelpad=1)
    ax.set_ylabel("R² on ρ$_{\\mathrm{ATAC}}$", color=C_ATAC, labelpad=1)
    p = wilcoxon(f.R2_dATAC, f.R2_dRNA, alternative="greater").pvalue
    k = int((f.R2_dATAC > f.R2_dRNA).sum())
    ax.set_title(f"inflammatory program genes\n{k}/{len(f)} better on ρ$_{{\\mathrm{{ATAC}}}}$, Wilcoxon p = {p:.2f}",
                 fontsize=5.8, pad=7)
    ax.set_aspect("equal")
    return dict(n=len(f), better_atac=k, p=float(p))


# --------------------------------------------------------------------------- #
def build_features(fig, spec, D):
    g3 = GridSpecFromSubplotSpec(2, 2, subplot_spec=spec, wspace=0.22, hspace=0.6)
    first, stats = None, {}
    for i, (m, feat, title, sub) in enumerate(FEATS):
        a_, F = panel_feature(fig, g3[i // 2, i % 2], D, m, feat, title, sub)
        stats[title] = dict(modality=m, feature=feat, R2=round(float(F["r2"]), 3),
                            n_sloped=F["n_sloped"], counts=F["n"])
        first = first or a_
    return first, stats


def build_tf(fig, spec, D):
    gh = GridSpecFromSubplotSpec(3, 5, subplot_spec=spec, height_ratios=[0.045, 1, 0.1],
                                 width_ratios=[1, 0.04, 1, 0.16, 0.05], hspace=0.05, wspace=0.02)
    strip = fig.add_subplot(gh[0, 0]); hm = fig.add_subplot(gh[1, 0])
    im, dropped = panel_tf_heat(fig, hm, strip, D, ax_fit=fig.add_subplot(gh[1, 2]), strip_fit=fig.add_subplot(gh[0, 2]))
    cax = fig.add_subplot(gh[1, 4]); cb = fig.colorbar(im, cax=cax, ticks=[-3, 0, 3])
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.5, length=1.5); cb.set_label("motif z", fontsize=5.5)
    key = fig.add_subplot(gh[2, :3]); key.set_axis_off()
    for i, L in enumerate(LINEAGE_ORDER):
        key.add_patch(Rectangle((0.01 + i * 0.2, -0.9), 0.012, 0.7, color=LINEAGE[L], transform=key.transAxes, clip_on=False))
        key.text(0.027 + i * 0.2, -0.55, L, fontsize=5.5, va="center", transform=key.transAxes)
    return strip, dropped


def build_tf_space(fig, spec, D):
    return panel_tf_space(fig, spec, D)


# one entry per panel: (letter, standalone size in inches, builder).  The assembled figure and
# the standalone panel files are drawn by the SAME builders, so they cannot drift apart.
def _single(fn):
    def b(fig, spec, D):
        ax = fig.add_subplot(spec); return ax, fn(ax, D)
    return b


PANELS = [
    ("a", (3.30, 1.45), build_codex),
    ("b", (1.45, 1.45), _single(lambda ax, D: panel_fit(ax, D["summary"]))),
    ("c", (1.85, 1.60), _single(panel_layers)),
    ("d", (7.20, 1.45), lambda fig, spec, D: (panel_maps(fig, spec, D)[0][0], None)),
    ("e", (7.20, 2.90), build_features),
    ("f", (4.10, 2.45), build_tf),
    ("g", (3.10, 2.45), lambda fig, spec, D: (build_tf_space(fig, spec, D), None)),
    ("h", (2.30, 1.45), _single(lambda ax, D: panel_cosign(ax))),
    ("i", (2.30, 1.45), _single(lambda ax, D: panel_gradient(ax))),
    ("j", (1.75, 1.45), _single(lambda ax, D: panel_infl(ax))),
]


def export_panels(D, outdir):
    """every panel as its own PNG (600 dpi) + SVG, no panel letter, same builder as the figure"""
    outdir.mkdir(parents=True, exist_ok=True)
    for L, size, build in PANELS:
        fig = plt.figure(figsize=size)
        spec = GridSpec(1, 1, figure=fig, left=0.12, right=0.9, top=0.86, bottom=0.18)[0]
        build(fig, spec, D)
        save_dual(fig, outdir / f"fig2{L}", write_manifest=False)
        plt.close(fig)
    print(f"-> {len(PANELS)} panels in {outdir}")


def main():
    """the assembled figure is built by fig2_lpc_v2.py (a-k from this module + SEACells l-m)"""
    import fig2_lpc_v2
    fig2_lpc_v2.main()


if __name__ == "__main__":
    main()
