"""Supplementary Figures 8 and 12-15 on A4 sheets, from the main figures' own builders (no new analysis).

  S8  (Fig 3)  a own vs cross split per feature, b CellRank fusion-weight sweep, c top lineage driver genes of each kernel (fate arrows and maps are in main Fig 3)
  S12 (Fig 7)  a Saturn vs GASTON R2, b cross-term R2 gain, c cross-explained genes, d Hi-C long genes: all five datasets
  S13 (Fig 7)  a local cross-gain maps, b example genes: the two datasets not in the main figure
  S14 (Fig 7)  gradient comparison, all five datasets     S15 (Fig 7)  how far the steepness reading goes
Output: fig_supp/S<n>/figS<n>.{png,svg,pdf,json}, one SVG per panel in panels/
Usage: python fig_supp/supp_extra_pages.py [S8] [fig7]
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

PF = Path("<workspace>/Saturn/published_figures"); OUT = PF / "fig_supp"
for p in ("_style", "fig2", "fig3", "fig7", "fig_supp"):
    sys.path.insert(0, str(PF / p))
from paper_style import apply, save_panel_svg
from a4 import save_a4, auto_letters


def finish(fig, tag, groups):
    """letter every sub-plot; groups = [(builder id, its axes)] -> {builder id: 'first-last'} for the panel file names"""
    L = auto_letters(fig); save_a4(fig, OUT / tag / f"fig{tag}"); plt.close(fig); out = {}
    for k, ax in groups:
        ls = sorted(L[a] for a in ax if a in L); out[k] = ls[0] if len(ls) == 1 else f"{ls[0]}-{ls[-1]}"
    return out


def s8():
    import fig3_parts as P, json
    from fig3_melanoma_v2 import CRK, KERNELS
    F3 = P.F3; D = F3.load(); F3.D_GLOBAL.update(D); res = {}
    DR = json.load(open(CRK / "drivers.json"))["drivers"]

    def table(ax):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        for sd in ax.spines.values(): sd.set_visible(False)
        ax.set_title("top lineage driver genes per kernel", fontsize=6, pad=2)
        ax.plot([0, 1], [0.86, 0.86], color="#1a1a1a", lw=0.6); ax.text(0, 0.9, "kernel", fontsize=5.5); ax.text(0.3, 0.9, "driver genes (ranked)", fontsize=5.5)
        for i, (k, lab, col) in enumerate(KERNELS):
            y = 0.68 - i * 0.21; gs = DR[k]["genes"][:6]
            ax.text(0, y, lab, fontsize=5.5, color=col, va="center")
            ax.text(0.3, y, ", ".join(gs[:3]) + ",\n" + ", ".join(gs[3:]), fontsize=5.5, va="center", fontstyle="italic", linespacing=1.05)

    fig = plt.figure(figsize=(7.2, 2.2))
    g0 = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.5, 2.2], wspace=0.55, left=0.07, right=0.98, top=0.88, bottom=0.2)
    P.build(fig, "c", g0[0], D, res); P.build(fig, "S3b", g0[1], D, res); table(fig.add_subplot(g0[2]))
    L = auto_letters(fig); save_a4(fig, OUT / "S8" / "figS8"); plt.close(fig)
    (OUT / "S8" / "panels").mkdir(parents=True, exist_ok=True)
    for L_, nm, size, x0 in (("c", "a_own_vs_cross", (1.75, 2.0), 0.22), ("S3b", "b_cellrank_fusion_weight", P.SIZE["S3b"], 0.2)):
        fig = plt.figure(figsize=size); sp_ = GridSpec(1, 1, figure=fig, left=x0, right=0.95, top=0.84, bottom=0.22)[0]
        P.build(fig, L_, sp_, D, res); save_panel_svg(fig, OUT / "S8" / "panels" / f"figS8{nm}"); plt.close(fig)
    fig = plt.figure(figsize=(3.3, 1.5)); table(fig.add_axes([0.03, 0.05, 0.94, 0.8])); save_panel_svg(fig, OUT / "S8" / "panels" / "figS8c_cellrank_driver_genes"); plt.close(fig)


def fig7():
    import fig7_cross_term as M
    F2 = M.F2; orig = F2.cbar
    def cbar(fig, ax, sc, label, ticks=None, on_top=False):       # supplement pages only: lift the title clear of the top tick
        orig(fig, ax, sc, label, ticks=ticks, on_top=on_top)
        if on_top:
            cb = sc.colorbar; cb.ax.set_title(label, fontsize=5.5, pad=5.5, loc="left")
    F2.cbar = cbar
    D = M.load(); names = {L: nm for L, nm, *_ in M.SUPP}; sizes = {L: sz for L, _, sz, *_ in M.SUPP}
    def page(tag, size, layout, mapping):
        fig = plt.figure(figsize=size); where = layout(fig); groups = []
        for L, new, nd in mapping:
            b, ks = M.BUILD[L]; D["keys"] = D["sets"][ks]; before = set(fig.axes); b(fig, where[L], D); groups.append((L, [a for a in fig.axes if a not in before]))
        lab = finish(fig, tag, groups); (OUT / tag / "panels").mkdir(parents=True, exist_ok=True)
        for L, new, _ in mapping:
            b, ks = M.BUILD[L]; D["keys"] = D["sets"][ks]; fig = plt.figure(figsize=sizes[L])
            b(fig, GridSpec(1, 1, figure=fig, **M.panel_box(L))[0], D); save_panel_svg(fig, OUT / tag / "panels" / f"fig{tag}{lab[L]}_{names[L]}"); plt.close(fig)
    def l12(fig):
        o = GridSpec(2, 1, figure=fig, height_ratios=[3.0, 2.2], hspace=0.42, left=0.14, right=0.93, top=0.96, bottom=0.03)
        g1 = GridSpecFromSubplotSpec(1, 2, subplot_spec=o[0], width_ratios=[1.05, 1], wspace=0.45); g2 = GridSpecFromSubplotSpec(1, 2, subplot_spec=o[1], width_ratios=[1.4, 1], wspace=0.75)
        return {"S6a": g1[0], "S6b": g1[1], "S6c": g2[0], "S6d": g2[1]}
    def l13(fig):
        o = GridSpec(2, 1, figure=fig, height_ratios=[3.6, 4.0], hspace=0.3, left=0.1, right=0.9, top=0.97, bottom=0.02)
        return {"S6e": o[0], "S6f": o[1]}
    page("S12", (7.2, 5.9), l12, [("S6a", "a", (-0.07, 0.02)), ("S6b", "b", (-0.06, 0.02)), ("S6c", "c", (-0.13, 0.02)), ("S6d", "d", (-0.07, 0.02))])
    page("S13", (7.2, 8.2), l13, [("S6e", "a", (-0.06, 0.012)), ("S6f", "b", (-0.05, 0.045))])
    page("S14", (7.2, 9.4), M.supp, [("S7", "a", (-0.04, 0.01))])
    page("S15", (7.2, 5.6), M.supp8, [("S8", "a", (-0.06, 0.02))])


if __name__ == "__main__":
    apply(base_pt=6.5); a = sys.argv[1:] or ["S8", "fig7"]
    if "S8" in a: s8()
    if "fig7" in a: fig7()
