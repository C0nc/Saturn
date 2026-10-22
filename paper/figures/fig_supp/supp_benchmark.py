"""Figure 1 — Saturn simulation benchmarks (reproducible, seed-controlled runs).

Two benchmarks, two views each:

  BENCH A — noise robustness      (3 sim regimes x 10 noise scenarios x 2 seeds)
    (a) method ranking, paired-Wilcoxon brackets vs Saturn
    (b) per-situation recovery heatmap: 8 methods x 10 noise scenarios

  BENCH B — modality x correlation pressure test  (M x corr x 2 seeds)
    (c) method ranking, paired-Wilcoxon brackets vs Saturn
    (d) per-situation recovery heatmap: 8 methods x (M, corr) grid

Metric throughout: sign-invariant Spearman |rho| between recovered and true
per-modality isodepth.

Sources
  bench A: Hole/gaston_mix_mm/results/noise_saturn/runs.csv                 (960 rows)
  bench B: Hole/gaston_mix_mm/results/pressure_test_v2_3x4_final.csv        (576 rows)

Excluded by project decision: the `multiscale` sim regime; the superseded
54-config pressure run (pressure_test_all.csv); MULTISPATI-PCA / SpatialNMF.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/_style")
from paper_style import apply, save_dual, clean_ax, METHOD, PALETTE, CMAP

RES = Path("/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/results")
OUT = Path("/mnt/HDD2B/Users/yancui/Saturn/published_figures/fig_supp/supp_benchmark")
KEEP_REGIMES = ["smooth", "domains", "cascade"]

NOISE_ORDER = ["clean", "lane_med", "lane_hi", "ambient_med", "ambient_hi",
                "depth_low", "depth_verylow", "compound", "dropout_med", "dropout_hi"]
NOISE_LABEL = {
    "clean": "clean", "lane_med": "lane med", "lane_hi": "lane hi",
    "ambient_med": "ambient med", "ambient_hi": "ambient hi",
    "depth_low": "depth low", "depth_verylow": "depth v.low",
    "compound": "compound", "dropout_med": "dropout med", "dropout_hi": "dropout hi",
}


def stars(p: float) -> str:
    if p < 1e-4:  return "****"
    if p < 1e-3:  return "***"
    if p < 1e-2:  return "**"
    if p < 5e-2:  return "*"
    return "ns"


def sig_bracket(ax, x1, x2, y, label, *, tick=0.010, lw=0.7, pad=0.004,
                  color="#1a1a1a"):
    fs = plt.rcParams["font.size"] - 1
    ax.plot([x1, x1, x2, x2], [y, y + tick, y + tick, y],
             lw=lw, c=color, clip_on=False, solid_joinstyle="miter")
    ax.text((x1 + x2) / 2, y + tick + pad, label, ha="center", va="bottom",
             fontsize=fs, color=color, clip_on=False,
             fontweight="bold" if label != "ns" else "normal")


def load():
    a = pd.read_csv(RES / "noise_saturn" / "runs.csv")
    a = a[a.regime.isin(KEEP_REGIMES)].copy()
    b = pd.read_csv(RES / "pressure_test_v2_3x4_final.csv").copy()
    return a, b


# --------------------------------------------------------------------------- #
# Shared panel builders
# --------------------------------------------------------------------------- #
def ranking_panel(ax, piv, title, note, ylim_top):
    """Bar ranking with paired-Wilcoxon brackets vs Saturn."""
    order = piv.mean().sort_values(ascending=False).index.tolist()
    means = piv[order].mean().values
    sems = (piv[order].std() / np.sqrt(piv[order].notna().sum())).values
    xs = np.arange(len(order))

    ax.bar(xs, means, yerr=sems,
            color=[METHOD.get(m, PALETTE["neutral"]) for m in order],
            edgecolor="black", linewidth=0.5, zorder=3,
            error_kw=dict(lw=0.6, capsize=1.6, capthick=0.6))

    sat = piv["Saturn"]
    top = float((means + sems).max())
    step = (ylim_top - top - 0.05) / 7.0
    for lvl, (i, m) in enumerate((i, m) for i, m in enumerate(order) if m != "Saturn"):
        ok = sat.notna() & piv[m].notna()
        _, p = wilcoxon(sat[ok], piv[m][ok])
        sig_bracket(ax, 0, i, top + 0.03 + lvl * step, stars(p))

    ax.set_xticks(xs)
    ax.set_xticklabels(order, rotation=38, ha="right")
    ax.set_ylabel("Isodepth recovery  |$\\rho$|")
    ax.set_ylim(0, ylim_top)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title(title, loc="left", fontweight="bold", pad=3)
    ax.margins(x=0.02)
    clean_ax(ax)
    ax.text(0.985, 0.575, note, transform=ax.transAxes, ha="right", va="bottom",
             fontsize=plt.rcParams["font.size"] - 1.5, color="#555555",
             linespacing=1.3)
    return order


def heatmap_panel(ax, mat, col_labels, row_order, title, xlabel,
                    group_edges=None, group_labels=None):
    """Per-situation recovery heatmap: methods (rows) x situations (columns)."""
    data = mat.loc[row_order, col_labels].values
    im = ax.imshow(data, cmap=CMAP["recovery"], aspect="auto",
                    norm=Normalize(vmin=0.4, vmax=1.0))

    fs = plt.rcParams["font.size"] - 1.5
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center",
                     fontsize=fs, color="white" if v > 0.78 else "#1a1a1a")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=38, ha="right")
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order)
    # emphasise the Saturn row
    sat_i = list(row_order).index("Saturn")
    for lbl in ax.get_yticklabels():
        if lbl.get_text() == "Saturn":
            lbl.set_fontweight("bold")
            lbl.set_color(METHOD["Saturn"])
    ax.add_patch(plt.Rectangle((-0.5, sat_i - 0.5), data.shape[1], 1,
                                 fill=False, ec=METHOD["Saturn"], lw=1.4,
                                 zorder=5, clip_on=False))

    if group_edges:
        for e in group_edges:
            ax.axvline(e - 0.5, color="white", lw=1.6)
        if group_labels:
            for (lo, hi), lab in group_labels:
                ax.text((lo + hi) / 2, -0.74, lab, ha="center", va="bottom",
                         fontsize=plt.rcParams["font.size"] - 0.5,
                         fontweight="bold", color="#1a1a1a")

    ax.set_xlabel(xlabel, labelpad=1)
    ax.set_title(title, loc="left", fontweight="bold", pad=17 if group_labels else 3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.030, pad=0.015)
    cb.set_label("recovery |$\\rho$|", labelpad=2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, width=0.6)
    return im


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def panel_a(ax, dfa):
    piv = dfa.pivot_table(index=["regime", "noise", "seed", "modality"],
                            columns="method", values="recovery")
    n = int(piv["Saturn"].notna().sum())
    return ranking_panel(
        ax, piv, "a   Noise robustness — method ranking",
        f"n = {n} paired runs\n3 regimes × 10 noise\n× 2 seeds × 2 modalities",
        ylim_top=1.42)


def panel_b(ax, dfa, row_order):
    mat = dfa.pivot_table(index="method", columns="noise", values="recovery")
    heatmap_panel(ax, mat, NOISE_ORDER, row_order,
                    "b   Recovery in each noise scenario",
                    "corruption severity  →")
    ax.set_xticklabels([NOISE_LABEL[n] for n in NOISE_ORDER], rotation=38, ha="right")


def panel_c(ax, dfb):
    piv = dfb.pivot_table(index=["M", "corr", "seed", "modality"],
                            columns="method", values="recovery")
    n = int(piv["Saturn"].notna().sum())
    return ranking_panel(
        ax, piv, "c   Modality × correlation pressure test — method ranking",
        f"n = {n} paired runs\nM = 2,3,4  ×  corr = .3,.4,.5,.6\n× 2 seeds",
        ylim_top=1.42)


def panel_d(ax, dfb, row_order):
    dfb = dfb.copy()
    dfb["cell"] = dfb["corr"].map(lambda c: f"{c:g}")
    Ms = sorted(dfb.M.unique())
    corrs = sorted(dfb["corr"].unique())
    cols, edges, glabels = [], [], []
    mat = {}
    pos = 0
    for M in Ms:
        lo = pos
        for c in corrs:
            key = f"M{M}·{c:g}"
            cols.append(key)
            sub = dfb[(dfb.M == M) & (dfb["corr"] == c)]
            for meth, v in sub.groupby("method").recovery.mean().items():
                mat.setdefault(meth, {})[key] = v
            pos += 1
        glabels.append(((lo, pos - 1), f"M = {M}"))
        if M != Ms[-1]:
            edges.append(pos)
    mat = pd.DataFrame(mat).T
    heatmap_panel(ax, mat, cols, row_order,
                    "d   Recovery in each modality × correlation setting",
                    "modality count M  ·  inter-modality correlation",
                    group_edges=edges, group_labels=glabels)
    ax.set_xticklabels([c.split("·")[1] for c in cols], rotation=0, ha="center")


# --------------------------------------------------------------------------- #
def main():
    apply(base_pt=7)
    dfa, dfb = load()

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.2))
    order_a = panel_a(axes[0, 0], dfa)
    panel_b(axes[0, 1], dfa, order_a)
    order_c = panel_c(axes[1, 0], dfb)
    panel_d(axes[1, 1], dfb, order_c)

    fig.tight_layout(pad=1.1, w_pad=1.6, h_pad=2.4)
    man = save_dual(fig, OUT, dpi=600, extra=dict(
        bench_a=str(RES / "noise_saturn" / "runs.csv"),
        bench_b=str(RES / "pressure_test_v2_3x4_final.csv"),
        regimes_kept=KEEP_REGIMES,
        metric="sign-invariant Spearman |rho| vs true per-modality isodepth",
        stats="paired Wilcoxon signed-rank vs Saturn; ****p<1e-4 ***p<1e-3 **p<0.01 *p<0.05",
        excluded="multiscale regime; pressure_test_all.csv (54-config); denoising panels",
    ))
    print("saved:", man["png"], "\n      ", man["svg"])


if __name__ == "__main__":
    main()
