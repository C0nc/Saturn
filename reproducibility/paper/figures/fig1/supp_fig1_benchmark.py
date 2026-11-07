"""Supplementary Figure 1 — every benchmark result behind Fig 1 (v5 grid, 48 runs x 7 methods).

  figS1a_all_conditions      per condition (seed-averaged), per method: axis recovery and cross-axis leak of the RNA
                             and ATAC axes, for the easy grid (angle x coupling pairs, 40 % coupled features) and the
                             hard grid (angle x coupling x coupled fraction); values printed in every cell
  figS1b_distributions       a/b recovery and leak over every run of the easy grid (Fig 1 shows the hard grid),
                             c/d the hard grid with the global-scaled GASTON ablation included, one-sided paired
                             Wilcoxon vs Saturn; e cross-coupling read off each method's axes vs the truth (Saturn's
                             built-in coupling too); f axis duplication (|corr| between a method's two axes) vs truth
  figS1c/d_maps_{easy,hard}  ground truth and every method's recovered RNA and ATAC axes, every condition (seed 0)
Metrics are read from the benchmark tables (results/bench_v5_full/bench_full_s*.csv + bench_full_fill.csv), never
recomputed. PNG at 600 dpi (maps are raster).
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import wilcoxon

sys.path.insert(0, "<workspace>/Saturn/published_figures/_style")
sys.path.insert(0, "<workspace>/Saturn/published_figures/fig1")
from paper_style import apply, check_min_font, rasterize_points, METHOD, MODALITY, CMAP
import fig1_violin_maps as V
from fig1_model_benchmark import RES, G

OUT = Path("<workspace>/Saturn/published_figures/fig1/supp")
ORDER = ["Saturn", "GASTON", "GASTON (global-scaled)", "MEFISTO", "MISO", "SCIGMA", "SpatialGlue"]
KEY = {"GASTON": "GASTON (default)"}
V.ORDER, V.KEY = ORDER, KEY
V.stars = lambda p: "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"     # short label, no overlap
METRICS = [("rec_R", "RNA axis recovery", "high"), ("rec_A", "ATAC axis recovery", "high"),
           ("leak_R", "RNA axis leak", "low"), ("leak_A", "ATAC axis leak", "low")]


def load():
    T = pd.concat([pd.read_csv(f) for f in glob.glob(str(RES / "bench_full_s*.csv"))] + [pd.read_csv(RES / "bench_full_fill.csv")])
    T["method"] = T.method.replace({v: k for k, v in KEY.items()})
    return T.drop_duplicates(["grid", "angle", "cross_RA", "cross_AR", "coupled_frac", "seed", "method"], keep="last")


def save(fig, name):
    check_min_font(fig); rasterize_points(fig); OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.05); plt.close(fig); print("->", OUT / f"{name}.png")


def cond_label(r, grid):
    s = f"{int(r.angle)}°\n{r.cross_RA:g}/{r.cross_AR:g}"
    return s if grid == "easy" else s + f"\n{int(r.coupled_frac * 100)}%"


# --------------------------------------------------------------------------- S1a
def page_conditions(T):
    fig = plt.figure(figsize=(7.2, 9.6))
    gs = GridSpec(4, 2, figure=fig, width_ratios=[12, 8], hspace=0.62, wspace=0.12, left=0.13, right=0.93, top=0.95, bottom=0.04)
    fig.suptitle("Every condition of the Fig 1 benchmark (mean of seeds): recovery (higher = better), leak (lower = better)",
                 fontsize=7, fontweight="bold", y=0.99)
    key = ["angle", "cross_RA", "cross_AR", "coupled_frac"]
    for i, (col, lab, better) in enumerate(METRICS):
        for j, grid in enumerate(("easy", "hard")):
            S = T[T.grid == grid].groupby(key + ["method"])[col].mean().unstack("method")[ORDER]
            a = fig.add_subplot(gs[i, j])
            cmap = CMAP["recovery"] if better == "high" else CMAP["recovery"] + "_r"
            vmin, vmax = (0.5, 1.0) if better == "high" else (0.0, 0.3)
            a.imshow(S.T.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            for (r, c), v in np.ndenumerate(S.T.values):
                if np.isfinite(v):
                    dark = (v - vmin) / (vmax - vmin) > 0.6 if better == "high" else (v - vmin) / (vmax - vmin) < 0.4
                    a.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=5, color="white" if dark else "#1a1a1a")
            a.set_yticks(range(len(ORDER)))
            a.set_yticklabels(ORDER if j == 0 else [], fontsize=5.5)
            for tl, m in zip(a.get_yticklabels(), ORDER):
                tl.set_color(V.ink(METHOD[m])); tl.set_fontweight("bold" if m == "Saturn" else "normal")
            a.set_xticks(range(len(S))); a.set_xticklabels([cond_label(r, grid) for r in S.reset_index().itertuples()], fontsize=5)
            a.tick_params(length=0, pad=1.5)
            a.set_title(f"{lab} · {grid} grid", fontsize=6.2, pad=2, color=MODALITY["RNA" if col.endswith("R") else "ATAC"])
    fig.text(0.13, 0.012, "x labels: angle between the two true axes; cross-coupling RNA←ATAC / ATAC←RNA; (hard) coupled "
             "features", fontsize=5.3, color="#444444")
    save(fig, "figS1a_all_conditions")


# --------------------------------------------------------------------------- S1b
def page_distributions(T):
    fig = plt.figure(figsize=(7.2, 7.4))
    gs = GridSpec(3, 2, figure=fig, hspace=0.75, wspace=0.3, left=0.08, right=0.98, top=0.95, bottom=0.08)
    fig.suptitle("Fig 1 benchmark: all runs, all methods", fontsize=7, fontweight="bold", y=0.99)
    L = []
    for r, grid in enumerate(("easy", "hard")):
        Lf, key = V.long_form(T[T.grid == grid])
        for c, (col, lab, better) in enumerate((("rec", "axis recovery (|Spearman|)", "high"), ("leak", "cross-axis leak", "low"))):
            a = fig.add_subplot(gs[r, c]); V.violins(a, Lf, key, col, lab, better)
            a.set_title(f"{grid} grid ({Lf.groupby(key).ngroups} runs × axes)", fontsize=6.4, pad=2)
            L.append(a)
    a = fig.add_subplot(gs[2, 0]); L.append(a)
    for m in ORDER:
        S = T[T.method == m]
        a.scatter(S.true_RA, S.contrib_RA, s=5, lw=0, color=METHOD[m], alpha=0.75, label=m)
    S = T[(T.method == "Saturn") & T.builtin_RA.notna()]
    a.scatter(S.true_RA, S.builtin_RA, s=12, marker="x", lw=0.7, color="#1a1a1a", label="Saturn built-in coupling")
    lim = [0, max(T.true_RA.max(), T.contrib_RA.max()) * 1.05]; a.plot(lim, lim, color="#9aa2aa", lw=0.6, ls=(0, (2, 2)))
    a.set_xlabel("true cross-coupling RNA ← ATAC", labelpad=1); a.set_ylabel("coupling read off the method's axes", labelpad=1)
    a.set_title("cross-coupling recovery", fontsize=6.4, pad=2)
    a.legend(fontsize=5, frameon=False, loc="lower right", markerscale=1.5, handletextpad=0.1, labelspacing=0.15)
    a = fig.add_subplot(gs[2, 1]); L.append(a)
    dup = [T[T.method == m].axis_dup.values - T[T.method == m].true_corr.abs().values for m in ORDER]
    bp = a.boxplot(dup, positions=range(len(ORDER)), widths=0.6, showfliers=False, patch_artist=True,
                   medianprops=dict(color="#1a1a1a", lw=0.8), whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5))
    for b, m in zip(bp["boxes"], ORDER):
        b.set_facecolor(METHOD[m]); b.set_alpha(0.7); b.set_edgecolor("#1a1a1a"); b.set_linewidth(0.4)
    a.axhline(0, color="#9aa2aa", lw=0.6)
    a.set_xticks(range(len(ORDER))); a.set_xticklabels(ORDER, rotation=40, ha="right", rotation_mode="anchor", fontsize=5.5)
    for tl, m in zip(a.get_xticklabels(), ORDER): tl.set_color(V.ink(METHOD[m]))
    a.set_ylabel("axis duplication − true |corr|", labelpad=1)
    a.set_title("do the two recovered axes collapse onto each other? (0 = as in the truth)", fontsize=6.0, pad=2)
    fig.canvas.draw()
    for ax, s in zip(L, "abcdef"):
        b = ax.get_position(); fig.text(b.x0 - 0.05, b.y1 + 0.012, s, fontsize=9, fontweight="bold")
    fig.text(0.08, 0.005, "brackets: one-sided paired Wilcoxon, Saturn vs each method over identical runs "
             "(* p < 0.05, ** < 0.01, *** < 0.001, ns = not significant)", fontsize=5.3, color="#444444")
    save(fig, "figS1b_distributions")


# --------------------------------------------------------------------------- S1c / S1d
def page_maps(T, grid, letter):
    runs = sorted({p.stem[:-3] for p in (RES / "axes").glob(f"{grid}_*_s0.npz")})
    fig = plt.figure(figsize=(7.2, 0.92 * 2 * len(runs) + 0.4))
    gs = GridSpec(2 * len(runs), len(ORDER) + 2, figure=fig, width_ratios=[1] * (len(ORDER) + 1) + [0.08],
                  wspace=0.08, hspace=0.42, top=1 - 0.35 / fig.get_size_inches()[1], bottom=0.01, left=0.1, right=0.97)
    Tg = T[T.grid == grid]
    for i, r in enumerate(runs):
        a_, c_, f_ = r.split("_")[1:]
        cra = float(c_[1:]); car = Tg[Tg.cross_RA == cra].cross_AR.iloc[0]
        lab = f"{a_[1:]}°, c {cra:g}/{car:g}" + ("" if grid == "easy" else f", {int(float(f_) * 100)}%")
        im = V.maps(fig, gs, f"{r}_s0", Tg, 2 * i, i == 0, label=lab, note=True)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[:2, -1]), ticks=[-2, 0, 2]); cb.ax.tick_params(labelsize=5.5)
    cb.set_label("isodepth (z)", fontsize=5.5)
    fig.suptitle(f"All {grid} conditions (seed 0): ground truth and every method's recovered axes "
                 "(rec = |Spearman| with the truth; leak = residual contamination by the other true axis)",
                 fontsize=6.4, fontweight="bold", y=0.998)
    save(fig, f"figS1{letter}_maps_{grid}")


def main():
    apply(base_pt=6.5); T = load()
    print(T.groupby(["grid", "method"]).size().unstack())
    page_conditions(T); page_distributions(T); page_maps(T, "easy", "c"); page_maps(T, "hard", "d")


if __name__ == "__main__":
    main()
