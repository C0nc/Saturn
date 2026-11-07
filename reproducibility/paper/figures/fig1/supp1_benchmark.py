"""Supplementary Figure 1 -- benchmark results not shown in Fig 1 (v5 grid, the 6 methods of Fig 1). Hard-grid violins are in Fig 1d,e.

  a  per condition (mean of seeds), per method: recovery and leak of the RNA and ATAC axes, easy and hard grids
  b  axis recovery over every easy-grid run, one-sided paired Wilcoxon vs Saturn
  c  cross-axis leak over the same runs, same test
  d  cross-coupling read off each method's axes vs the truth, all runs (Saturn's built-in coupling too)
  e  axis duplication (|corr| between a method's two axes) minus the true |corr|, all runs
Metrics are read from the benchmark tables, never recomputed. Drawn with the same code as the older figS1a/figS1b.
Output: fig1/supp/S1/figS1.{png,svg,pdf,json} on an A4 sheet, one PNG + SVG per panel in fig1/supp/S1/panels/
Usage: python published_figures/fig1/supp1_benchmark.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

sys.path.insert(0, "<workspace>/Saturn/published_figures/_style")
sys.path.insert(0, "<workspace>/Saturn/published_figures/fig1")
from paper_style import apply, save_panel_svg, METHOD, MODALITY, CMAP
from a4 import save_a4, auto_letters
import supp_fig1_benchmark as B
V = B.V
METRICS, cond_label = B.METRICS, B.cond_label
ORDER = [m for m in B.ORDER if m != "GASTON (global-scaled)"]      # as Fig 1: the global-scaled ablation arm is not shown
V.ORDER = ORDER
OUT = Path("<workspace>/Saturn/published_figures/fig1/supp/S1")
NOTE_A = ("x labels: angle between the two true axes; cross-coupling RNA←ATAC / ATAC←RNA; (hard grid) coupled features. "
          "Colour range: recovery 0.5–1, leak 0–0.3.")
NOTE_V = "Brackets: one-sided paired Wilcoxon, Saturn vs each method over identical runs (* P < 0.05, ** P < 0.01, *** P < 0.001, ns = not significant)."


def heatmaps(fig, spec):
    gs = GridSpecFromSubplotSpec(4, 2, spec, width_ratios=[12, 8], hspace=0.28, wspace=0.05)
    key = ["angle", "cross_RA", "cross_AR", "coupled_frac"]
    T = fig._T; first = None
    for i, (col, lab, better) in enumerate(METRICS):
        for j, grid in enumerate(("easy", "hard")):
            S = T[T.grid == grid].groupby(key + ["method"])[col].mean().unstack("method")[ORDER]
            a = fig.add_subplot(gs[i, j]); first = first or a
            cmap = CMAP["recovery"] if better == "high" else CMAP["recovery"] + "_r"
            vmin, vmax = (0.5, 1.0) if better == "high" else (0.0, 0.3)
            a.imshow(S.T.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            for (r, c), v in np.ndenumerate(S.T.values):
                if np.isfinite(v):
                    f = (v - vmin) / (vmax - vmin)
                    a.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=5,
                           color="white" if (f > 0.6 if better == "high" else f < 0.4) else "#1a1a1a")
            a.set_yticks(range(len(ORDER))); a.set_yticklabels(ORDER if j == 0 else [], fontsize=5.5)
            for tl, m in zip(a.get_yticklabels(), ORDER):
                tl.set_color(V.ink(METHOD[m]))
            a.set_xticks(range(len(S)))
            a.set_xticklabels([cond_label(r, grid) for r in S.reset_index().itertuples()] if i == 3 else [], fontsize=5)
            a.tick_params(length=0, pad=1.5)
            for s in a.spines.values(): s.set_visible(False)
            a.set_title(f"{lab}, {grid} grid", fontsize=6.2, pad=2, color=MODALITY["RNA" if col.endswith("R") else "ATAC"])
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    y = min(a.get_tightbbox(r).y0 for a in fig.axes[:8]) / fig.bbox.height
    fig.text(first.get_position().x0, y - 0.006, NOTE_A, fontsize=5.3, color="#444444", va="top")
    return first


def violin(a, T, col, lab, better):
    Lf, key = V.long_form(T[T.grid == "easy"])
    V.violins(a, Lf, key, col, lab, better)
    n_runs = Lf.groupby(key).ngroups // Lf.axis.nunique()
    a.set_title(f"easy grid ({n_runs} runs × 2 axes)", fontsize=6.4, pad=2)
    a.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); a.tick_params(axis="x", labelsize=6)


def coupling(a, T):
    for m in ORDER:
        S = T[T.method == m]
        a.scatter(S.true_RA, S.contrib_RA, s=5, lw=0, color=METHOD[m], alpha=0.75, label=m)
    S = T[(T.method == "Saturn") & T.builtin_RA.notna()]
    a.scatter(S.true_RA, S.builtin_RA, s=12, marker="x", lw=0.7, color="#1a1a1a", label="Saturn built-in coupling")
    lim = [0, max(T.true_RA.max(), T.contrib_RA.max()) * 1.05]; a.plot(lim, lim, color="#9aa2aa", lw=0.6, ls=(0, (2, 2)))
    a.set_xlabel("true cross-coupling RNA ← ATAC", labelpad=1); a.set_ylabel("coupling read off\nthe method's axes", labelpad=1)
    a.set_title("cross-coupling recovery, all runs", fontsize=6.4, pad=2)
    a.legend(fontsize=5, frameon=False, loc="lower right", markerscale=1.5, handletextpad=0.1, labelspacing=0.15)
    for s in ("top", "right"): a.spines[s].set_visible(False)


def duplication(a, T):
    dup = [T[T.method == m].axis_dup.values - T[T.method == m].true_corr.abs().values for m in ORDER]
    bp = a.boxplot(dup, positions=range(len(ORDER)), widths=0.6, showfliers=False, patch_artist=True,
                   medianprops=dict(color="#1a1a1a", lw=0.8), whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5))
    for b, m in zip(bp["boxes"], ORDER):
        b.set_facecolor(METHOD[m]); b.set_alpha(0.7); b.set_edgecolor("#1a1a1a"); b.set_linewidth(0.4)
    a.axhline(0, color="#9aa2aa", lw=0.6)
    a.set_xticks(range(len(ORDER))); a.set_xticklabels([V.TICK.get(m, m) for m in ORDER], rotation=40, ha="right", rotation_mode="anchor", fontsize=6)
    for tl, m in zip(a.get_xticklabels(), ORDER): tl.set_color(V.ink(METHOD[m]))
    a.set_ylabel("axis duplication − true |corr|", labelpad=1)
    a.set_title("axis duplication, all runs (0 = as in the truth)", fontsize=6.4, pad=2)
    for s in ("top", "right"): a.spines[s].set_visible(False)


def letter(fig, ax, s, dx=0.05, dy=0.012):
    fig.canvas.draw(); b = ax.get_position()
    fig.text(b.x0 - dx, b.y1 + dy, s, fontsize=9, fontweight="bold")


def full(T):
    fig = plt.figure(figsize=(7.2, 9.8)); fig._T = T
    outer = GridSpec(3, 1, figure=fig, height_ratios=[4.4, 2.3, 2.3], hspace=0.42, left=0.14, right=0.985, top=0.975, bottom=0.075)
    ah = heatmaps(fig, outer[0])
    row = GridSpecFromSubplotSpec(1, 2, outer[1], wspace=0.25); row2 = GridSpecFromSubplotSpec(1, 2, outer[2], wspace=0.25)
    ab = fig.add_subplot(row[0]); violin(ab, T, "rec", "axis recovery (|Spearman|)", "high")
    ac = fig.add_subplot(row[1]); violin(ac, T, "leak", "cross-axis leak", "low")
    ad = fig.add_subplot(row2[0]); coupling(ad, T)
    ae = fig.add_subplot(row2[1]); duplication(ae, T)
    auto_letters(fig)
    fig.text(0.14, 0.008, NOTE_V, fontsize=5.3, color="#444444")
    save_a4(fig, OUT / "figS1", extra=dict(panels={"a-h": "per-condition heatmaps", "i": "easy recovery violins", "j": "easy leak violins",
              "k": "cross-coupling recovery", "l": "axis duplication"}))
    plt.close(fig)


def panel(name, size, draw, box, L):
    fig = plt.figure(figsize=size)
    gs = GridSpec(1, 1, figure=fig, **box)
    ax = draw(fig, gs[0]); auto_letters(fig, start=ord(L[0]) - 97)
    p = OUT / "panels" / name
    save_panel_svg(fig, p); fig.savefig(p.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig); print("->", p)


def main():
    apply(base_pt=6.5); T = B.load(); T = T[T.method.isin(ORDER)]
    print(T.groupby(["grid", "method"]).size().unstack())
    full(T)
    def single(f, *a):
        def d(fig, s):
            ax = fig.add_subplot(s); f(ax, T, *a); return ax
        return d
    def hm(fig, s):
        fig._T = T; return heatmaps(fig, s)
    box = dict(left=0.17, right=0.98, top=0.92, bottom=0.27)
    panel("figS1a-h_all_conditions", (7.2, 4.6), hm, dict(left=0.14, right=0.99, top=0.96, bottom=0.075), "a")
    fp = (3.5, 2.5)
    panel("figS1i_easy_axis_recovery", fp, single(violin, "rec", "axis recovery (|Spearman|)", "high"), box, "i")
    panel("figS1j_easy_cross_axis_leak", fp, single(violin, "leak", "cross-axis leak", "low"), box, "j")
    panel("figS1k_cross_coupling_recovery", fp, single(coupling), dict(left=0.17, right=0.98, top=0.92, bottom=0.14), "k")
    panel("figS1l_axis_duplication", fp, single(duplication), box, "l")


if __name__ == "__main__":
    main()
