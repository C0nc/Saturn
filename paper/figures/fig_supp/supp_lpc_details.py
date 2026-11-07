"""Supplementary Figures 5 and 6 -- LPC lesion (Fig 2): experimental and analysis details not shown in the main figure. Existing outputs only.

  S5  a whole section and FOV; b lesion labels (microglia-positive components of the DBiT score, lpc10_hotspots.npz); c CODEX CD11c per registered
      spot; d CD11c by label (two-sided Mann-Whitney vs other; CODEX was not used to make the labels); e,f RNA UMI and ATAC fragments in peaks per spot;
      g |Spearman| of each isodepth with annotations, depth, CODEX CD11c and the other isodepth
  S6  a slope-sign concordance vs gene-peak distance; b gene-peak gradient agreement along vs across the isodepth (fig2_lpc panels);
      c per-feature R2: unique own / shared / unique cross; d TF activity R2 on each isodepth; e R2 of the 37 inflammatory-program genes on each isodepth
Output: fig_supp/S5/figS5.* and fig_supp/S6/figS6.* on A4 sheets (png, svg, pdf, json), one PNG + SVG per panel in panels/
Usage: python fig_supp/supp_lpc_details.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

H = "/mnt/HDD2B/Users/yancui"; PF = Path(f"{H}/Saturn/published_figures")
for p in (f"{PF}/_style", f"{H}/saturn_repo/src", f"{PF}/fig2", f"{H}/Hole/gaston_mix_mm/scripts", f"{PF}/fig_supp"):
    sys.path.insert(0, p)
from paper_style import apply, save_panel_svg, MODALITY, REGION, LINEAGE, CMAP
from a4 import save_a4
import fig2_lpc as F
R = Path(f"{H}/Hole/gaston_mix_mm/results/real_saturn"); DS = "lpc10_fov"
CX = f"{H}/catalign_lpc10/results/LPC10S1_wm/LPC10S1_codex_per_spot.h5ad"
OUT = PF / "fig_supp"; W = 7.2
C_R, C_A = MODALITY["RNA"], MODALITY["ATAC"]
RHO = {"RNA": "ρ$_{\\mathrm{RNA}}$", "ATAC": "ρ$_{\\mathrm{ATAC}}$"}
STATS = {}
GN = {"complement_astro": "complement / astrocyte", "acute_IFN_NFkB": "acute IFN\n/ NF-κB"}


class Cv:
    def __init__(s, w, h):
        s.w, s.h = w, h; s.fig = plt.figure(figsize=(w, h))
    def ax(s, x, y, w, h):
        return s.fig.add_axes([x / s.w, 1 - (y + h) / s.h, w / s.w, h / s.h])
    def text(s, x, y, t, **k):
        k.setdefault("va", "top"); return s.fig.text(x / s.w, 1 - y / s.h, t, **k)


def fmt_p(p):
    if p == 0: return "$P$ < 1 × 10$^{-300}$"
    e = int(np.floor(np.log10(p))); return f"$P$ = {p / 10 ** e:.2f} × 10$^{{{e}}}$" if p < 0.001 else f"$P$ = {p:.3f}"


def load():
    ax = pd.read_csv(R / DS / "axes.csv", index_col=0); ax.index = ax.index.astype(str)
    z = dict(np.load(R / f"targets_{DS}.npz", allow_pickle=True)); h = dict(np.load(R / "lpc10_hotspots.npz", allow_pickle=True))
    C = ad.read_h5ad(CX); C.obs_names = C.obs_names.astype(str); C = C[ax.index]
    name = [v for v in C.var_names if v.upper().replace("-", "") in ("CD11C", "ITGAX")][0]
    x = C[:, name].X; x = np.asarray(x.todense() if hasattr(x, "todense") else x, float).ravel()
    reg = pd.Series(z["ann_region"].astype(str)).replace({"distal_WM": "distal WM focus"}).values
    L = F.load()                      # depth as in the manuscript's Methods: counts over all genes and called peaks
    assert L["bc"] == list(ax.index)
    D = dict(ax=ax, xy=ax[["x", "y"]].values, reg=reg, cd11c=x, depthR=L["lib"]["RNA"].astype(float), depthA=L["lib"]["ATAC"].astype(float),
             micro=z["ann_microglia_z"].astype(float), myelin=z["ann_myelin_z"].astype(float), dist=z["ann_dist_to_lesion"].astype(float), h=h)
    print("median RNA UMI", np.median(D["depthR"]), "ATAC fragments in peaks", np.median(D["depthA"]), "n", len(ax), pd.Series(reg).value_counts().to_dict(), flush=True)
    return D


def spatial(a, xy, v=None, colors=None, cmap=None, vmin=None, vmax=None, s=0.9):
    sc = a.scatter(xy[:, 0], xy[:, 1], c=colors if colors is not None else v, cmap=None if colors is not None else cmap,
                   vmin=vmin, vmax=vmax, s=s, lw=0, rasterized=True)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    for sp_ in a.spines.values(): sp_.set_visible(False)
    return sc


def cbar(cv, x, y, h, sc, label, ticks=None):
    cax = cv.ax(x, y + 0.05 * h, 0.05, 0.9 * h); cb = cv.fig.colorbar(sc, cax=cax, ticks=ticks); cb.outline.set_linewidth(0.3)
    cb.ax.tick_params(labelsize=5, length=1.5, width=0.4, pad=1); cb.set_label(label, fontsize=5, labelpad=1)


def clean(a):
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
    a.tick_params(labelsize=5, length=1.5, width=0.4, pad=1)


# ----------------------------------------------------------------------- panels: (cv, x, y, w, h, D)
def p_section(cv, x, y, w, h, D):
    hs = D["h"]; a = cv.ax(x, y, w, h); bc = hs["barcodes"].astype(str); keep = np.isin(bc, D["ax"].index)
    spatial(a, hs["xy"][~keep], colors="#E3E6E8", s=0.8)
    sc = spatial(a, hs["xy"][keep], hs["zmg"][keep], cmap=CMAP["score"], vmin=-5, vmax=5, s=0.8)
    a.set_title(f"whole section: {len(bc):,} spots\nFOV: {keep.sum():,} spots; grey = outside FOV", fontsize=5.5, pad=2)
    cbar(cv, x + w + 0.03, y, h, sc, "microglia score (z, clipped at ±5)", ticks=[-5, 0, 5])
    STATS["fov"] = dict(n_section=int(len(bc)), n_fov=int(keep.sum()))


def p_region(cv, x, y, w, h, D):
    a = cv.ax(x, y, w, h); o = np.argsort(D["reg"] != "other")[::1]
    col = np.array([REGION["other" if r == "other" else r] for r in D["reg"]])
    spatial(a, D["xy"][o], colors=col[o])
    a.set_title("lesion labels", fontsize=5.5, pad=2)
    cts = pd.Series(D["reg"]).value_counts()
    a.legend(handles=[Patch(fc=REGION[k], label=f"{k.replace(' focus', '')} ({cts[k]:,})") for k in ("lesion", "distal WM focus", "other")],
             fontsize=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=1, handlelength=0.8, handleheight=0.8,
             handletextpad=0.3, labelspacing=0.1, borderaxespad=0)


def p_cd11c(cv, x, y, w, h, D):
    a = cv.ax(x, y, w, h); v = np.log1p(D["cd11c"]); lo, hi = np.nanpercentile(v, [2, 99])
    sc = spatial(a, D["xy"], v, cmap=CMAP["expression"], vmin=lo, vmax=hi); a.set_title("CODEX CD11c per spot", fontsize=5.5, pad=2)
    cbar(cv, x + w + 0.03, y, h, sc, "log(1 + intensity)")


def p_cd11c_viol(cv, x, y, w, h, D):
    a = cv.ax(x, y, w, h); v = np.log1p(D["cd11c"]); order = ["lesion", "distal WM focus", "other"]; dat = [v[D["reg"] == k] for k in order]
    P = a.violinplot(dat, positions=range(3), widths=0.8, showextrema=False, bw_method=0.3)
    for b, k in zip(P["bodies"], order): b.set_facecolor(REGION[k]); b.set_alpha(0.75); b.set_edgecolor("#1a1a1a"); b.set_linewidth(0.3)
    for i, d_ in enumerate(dat):
        q1, m, q3 = np.percentile(d_, [25, 50, 75]); a.plot([i, i], [q1, q3], color="#1a1a1a", lw=1.2)
        a.scatter([i], [m], s=9, color="white", edgecolor="#1a1a1a", lw=0.5, zorder=3)
    p1 = mannwhitneyu(dat[0], dat[2]).pvalue; p2 = mannwhitneyu(dat[1], dat[2]).pvalue; top = max(d_.max() for d_ in dat)
    for i, p_ in enumerate((p1, p2)):
        yy = top * (1.05 + (1 - i) * 0.17); a.plot([i, i, 2, 2], [yy - top * 0.02, yy, yy, yy - top * 0.02], color="#333333", lw=0.5)
        a.text((i + 2) / 2, yy, fmt_p(p_), ha="center", va="bottom", fontsize=5)
    a.set_ylim(top=top * 1.42); a.set_xticks(range(3)); a.set_xticklabels(["lesion", "distal", "other"], fontsize=5.5)
    a.set_ylabel("CODEX CD11c, log(1 + intensity)", fontsize=5.5, labelpad=1); clean(a)
    a.set_title("protein not used to define the labels", fontsize=5.5, pad=2)
    STATS["cd11c"] = dict(medians_raw={k: round(float(np.median(D["cd11c"][D["reg"] == k])), 3) for k in order}, p_lesion_vs_other=float(p1), p_distal_vs_other=float(p2),
                      test="two-sided Mann-Whitney U")


def p_depth(key, label, title):
    def f(cv, x, y, w, h, D):
        a = cv.ax(x, y, w, h); v = np.log10(D[key].clip(1)); lo, hi = np.percentile(v, [1, 99])
        sc = spatial(a, D["xy"], v, cmap=CMAP["expression"], vmin=lo, vmax=hi); a.set_title(f"{title}, median {np.median(D[key]):,.0f}", fontsize=5.5, pad=2)
        cbar(cv, x + w + 0.03, y, h, sc, label)
    return f


def p_corr(cv, x, y, w, h, D):
    a = cv.ax(x, y, w, h); dR, dA = D["ax"].d_RNA.values, D["ax"].d_ATAC.values
    rows = [("microglia score", D["micro"]), ("CODEX CD11c", D["cd11c"]), ("myelin score", D["myelin"]), ("distance to lesion", D["dist"]),
            ("RNA UMI per spot", D["depthR"]), ("ATAC fragments in peaks", D["depthA"]), ("other isodepth", None)]
    Z = np.array([[spearmanr(dR, dA).statistic if v is None else spearmanr(d_, v).statistic for d_ in (dR, dA)] for _, v in rows])
    sgn = np.array([np.sign(spearmanr(dR, D["micro"]).statistic), np.sign(spearmanr(dA, D["micro"]).statistic)])
    im = a.imshow(np.abs(Z), aspect="auto", cmap=CMAP["recovery"], vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(Z):
        a.text(j, i, f"{abs(v):.2f}", ha="center", va="center", fontsize=5, color="white" if abs(v) > 0.5 else "#1a1a1a")
    a.set_xticks([0, 1]); a.set_xticklabels([RHO["RNA"], RHO["ATAC"]], fontsize=5.5); a.xaxis.tick_top(); a.tick_params(length=0, pad=1.5)
    for tl, c_ in zip(a.get_xticklabels(), (C_R, C_A)): tl.set_color(c_)
    a.set_yticks(range(len(rows))); a.set_yticklabels([r[0] for r in rows], fontsize=5.2)
    for s_ in a.spines.values(): s_.set_visible(False)
    STATS["correlations"] = {r[0]: dict(RNA=round(float(Z[i, 0]), 3), ATAC=round(float(Z[i, 1]), 3)) for i, r in enumerate(rows)}
    a.text(0.5, -0.03, "|Spearman r|", transform=a.transAxes, ha="center", va="top", fontsize=5, color="#444444")


def p_owncross(cv, x, y, w, h, D):
    a = cv.ax(x, y, w, h); T = pd.read_csv(R / f"{DS}_own_vs_cross" / "own_vs_cross.csv")
    comps = [("unique_own", 1.0, "unique own axis"), ("shared", 0.55, "shared"), ("unique_cross", 0.25, "unique cross axis")]; st = {}
    for i, m in enumerate(("RNA", "ATAC")):
        s = T[T.modality == m]; bottom = 0
        for c, al, _ in comps:
            v = float(np.clip(s[c], 0, None).median()); a.bar(i, v, 0.62, bottom=bottom, color=MODALITY[m], alpha=al, edgecolor="#1a1a1a", lw=0.4); bottom += v
        fr = 100 * (s.R2_own > s.R2_cross).mean(); st[m] = dict(n=int(len(s)), own_gt_cross_pct=round(float(fr), 1))
        a.text(i, bottom + 0.005, f"own>cross\n{fr:.0f}%", ha="center", va="bottom", fontsize=5)
    a.set_xticks([0, 1]); a.set_xticklabels(["RNA", "ATAC"], fontsize=5.5)
    for tl, c_ in zip(a.get_xticklabels(), (C_R, C_A)): tl.set_color(c_)
    a.set_xlim(-0.6, 1.6); a.set_ylim(0, 0.38); a.set_ylabel("median R² per feature", fontsize=5.5, labelpad=1); clean(a)
    a.legend(handles=[Patch(fc="#555555", alpha=a_, ec="#1a1a1a", lw=0.4, label=l_) for _, a_, l_ in comps], fontsize=5, frameon=False, loc="upper center", ncol=1,
             handlelength=0.9, handleheight=0.8, borderaxespad=0, labelspacing=0.1, handletextpad=0.3)
    STATS["own_cross"] = st


def p_cosign(cv, x, y, w, h, D):
    a = cv.ax(x, y, w, h); STATS["concordance"] = F.panel_cosign(a); clean(a); a.xaxis.label.set_size(5.5); a.yaxis.label.set_size(5.5)
    a.set_title("slope-sign concordance by distance", fontsize=5.5, pad=2)


def p_gradient(cv, x, y, w, h, D):
    a = cv.ax(x, y, w, h); STATS["gradient"] = F.panel_gradient(a); clean(a); a.xaxis.label.set_size(5.5); a.yaxis.label.set_size(5.5)
    a.set_title("gradient agreement, along vs across", fontsize=5.5, pad=2)


def p_tf(cv, x, y, w, h, D):
    a = cv.ax(x, y, w, h); T = pd.read_csv(R / f"{DS}_tf_activity" / "tf_fits.csv").pivot(index="tf", columns="axis", values="R2")
    lin = pd.read_csv(R / f"{DS}_tf_activity" / "tf_fits.csv").drop_duplicates("tf").set_index("tf").lineage
    a.scatter(T.RNA, T.ATAC, s=9, c=[LINEAGE[lin[t]] for t in T.index], lw=0.3, edgecolor="#1a1a1a", zorder=3)
    m = max(T.max()) * 1.08; a.plot([0, m], [0, m], color="#9aa2aa", lw=0.5, ls=(0, (2, 2)))
    lab = pd.concat([T.ATAC.nlargest(4), (T.RNA - T.ATAC).nlargest(3)]).index.unique()
    for t in lab:
        lft = t == "POU3F3"; a.annotate(t, (T.RNA[t], T.ATAC[t]), fontsize=5, xytext=(-2 if lft else 2, 1.5), ha="right" if lft else "left", textcoords="offset points")
    a.set_xlim(0, m); a.set_ylim(0, m); a.set_xlabel(f"R² on {RHO['RNA']}", fontsize=5.5, labelpad=1, color=C_R); a.set_ylabel(f"R² on {RHO['ATAC']}", fontsize=5.5, labelpad=1, color=C_A); clean(a)
    p = wilcoxon(T.ATAC, T.RNA).pvalue; k = int((T.ATAC > T.RNA).sum())
    a.set_title(f"TF activity: {k}/{len(T)} higher on {RHO['ATAC']}, {fmt_p(p)}", fontsize=5.3, pad=2)
    a.legend(handles=[Patch(fc=LINEAGE[l_], label=l_) for l_ in LINEAGE if l_ in set(lin)], fontsize=5, frameon=False, loc="lower right", handlelength=0.8, handleheight=0.8,
             handletextpad=0.3, labelspacing=0.1, borderaxespad=0)
    STATS["tf"] = dict(n=int(len(T)), higher_on_ATAC=k, p_two_sided_wilcoxon=float(p))


def p_infl(cv, x, y, w, h, D):
    a = cv.ax(x, y, w, h); f = pd.read_csv(R / f"{DS}_posthoc" / "infl_gene_axis_fits.csv")
    groups = list(pd.unique(f.group)); f["d"] = f.R2_dATAC - f.R2_dRNA; f = pd.concat([f[f.group == g].sort_values("d", ascending=False) for g in groups]).reset_index(drop=True)
    for i, r in f.iterrows(): a.plot([i, i], [r.R2_dRNA, r.R2_dATAC], color=C_A if r.d > 0 else C_R, lw=0.7, zorder=1)
    a.scatter(f.index, f.R2_dRNA, s=9, color=C_R, lw=0, zorder=2, label=RHO["RNA"]); a.scatter(f.index, f.R2_dATAC, s=9, color=C_A, lw=0, zorder=2, label=RHO["ATAC"])
    a.set_xticks(f.index); a.set_xticklabels(f.gene, rotation=60, ha="right", rotation_mode="anchor", fontsize=5, fontstyle="italic"); a.set_xlim(-0.7, len(f) - 0.3); a.set_ylim(-0.03, 1.13)
    a.set_ylabel("R² of the fitted function", fontsize=5.5, labelpad=1); clean(a); st = {}
    for g in groups:
        ix = f.index[f.group == g]; a.text((ix[0] + ix[-1]) / 2, 1.12, GN.get(g, g), linespacing=0.95, ha="center", va="top", fontsize=5.2, color="#333333")
        if ix[0] > 0: a.axvline(ix[0] - 0.5, color="#9aa2aa", lw=0.4, ls=(0, (2, 2)))
        s = f[f.group == g]; st[g] = dict(n=int(len(s)), better_ATAC=int((s.d > 0).sum()), med_R2_ATAC=round(float(s.R2_dATAC.median()), 2), med_R2_RNA=round(float(s.R2_dRNA.median()), 2))
    a.legend(fontsize=5, frameon=False, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, handletextpad=0.1, columnspacing=0.6, borderaxespad=0); STATS["inflammatory_genes"] = st


PAGES = dict(
    S5=dict(title="LPC lesion: field of view, lesion labels and data quality", H=5.2, panels=[
        ("a", "fov_selection", p_section, (1.6, 1.7), (0.1, 0.5), 0.02), ("b", "lesion_labels", p_region, (1.25, 1.7), (2.45, 0.5), 2.35),
        ("c", "codex_cd11c", p_cd11c, (1.25, 1.7), (3.85, 0.5), 3.75), ("d", "cd11c_by_label", p_cd11c_viol, (1.25, 1.7), (5.9, 0.5), 5.5),
        ("e", "rna_depth", p_depth("depthR", "log$_{10}$ UMI", "RNA UMI\nper spot"), (1.25, 1.7), (0.1, 3.1), 0.02),
        ("f", "atac_depth", p_depth("depthA", "log$_{10}$ fragments", "ATAC fragments in peaks\nper spot"), (1.25, 1.7), (2.0, 3.1), 1.92),
        ("g", "isodepth_correlations", p_corr, (1.5, 1.7), (5.2, 3.1), 3.9)]),
    S6=dict(title="LPC lesion: gene-peak coupling, own and cross terms, TFs and inflammatory genes", H=7.95, panels=[
        ("a", "concordance_by_distance", p_cosign, (2.8, 1.8), (0.5, 0.5), 0.02), ("b", "gradient_along_across", p_gradient, (2.8, 1.8), (4.2, 0.5), 3.7),
        ("c", "own_vs_cross", p_owncross, (1.6, 2.1), (0.5, 3.0), 0.02), ("d", "tf_r2_two_axes", p_tf, (2.1, 2.1), (3.0, 3.0), 2.5),
        ("e", "inflammatory_genes", p_infl, (6.5, 1.55), (0.5, 5.65), 0.02)]))


def main():
    global STATS
    apply(base_pt=6.5); D = load()
    for tag, P in PAGES.items():
        STATS.clear(); stem = OUT / tag; (stem / "panels").mkdir(parents=True, exist_ok=True)
        cv = Cv(W, P["H"]); cv.text(0.02, 0.0, P["title"], fontsize=7, fontweight="bold")
        for L, nm, fn, (w, h), (x, y), lx in P["panels"]:
            fn(cv, x, y, w, h, D); cv.text(lx, y - 0.25, L, fontsize=9, fontweight="bold")
        save_a4(cv.fig, stem / f"fig{tag}", extra=dict(stats=dict(STATS))); plt.close(cv.fig)
        print(tag, json.dumps(STATS, default=str)[:1500], flush=True)
        for L, nm, fn, (w, h), _, _ in P["panels"]:
            ml = 0.1 if fn in (p_section, p_region, p_cd11c) or nm.endswith("depth") else (1.3 if fn is p_corr else 0.45)
            q = Cv(min(7.2, ml + w + 0.75), h + 1.3); fn(q, ml, 0.4, w, h, D); p = stem / "panels" / f"fig{tag}{L}_{nm}"
            save_panel_svg(q.fig, p); q.fig.savefig(p.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03); plt.close(q.fig)


if __name__ == "__main__":
    main()
