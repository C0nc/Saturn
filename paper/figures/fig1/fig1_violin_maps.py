"""Figure 1 companions: violins with significance brackets, and ground truth vs every method's recovered axes.

  (standalone violins / representative maps are now Fig 1 panels d-e / c, exported by fig1_model_benchmark.py)
  fig1_violins          axis recovery and cross-axis leak over the hard grid (24 runs x 2 axes), violin + points,
                        bracket Saturn vs each method = one-sided paired Wilcoxon (recovery: Saturn higher;
                        leak: Saturn lower); n.s. shown explicitly
  fig1_recovery_maps    representative hard condition (same PICK as fig1_model_benchmark.py): ground truth and
                        each method's two axes on the 30 x 30 grid, rec / leak under each tile
  fig1_recovery_maps_all  every hard condition (seed 0): one row pair per condition, recovery under each tile

GASTON = the official package at its published default (its own per-component z-scoring); the global-scaled ablation
arm in the benchmark tables is not shown.
Metrics are read from the benchmark tables, never recomputed. Axis sign is aligned to the truth for display only.
Usage: python published_figures/fig1/fig1_violin_maps.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/_style")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/fig1")
from paper_style import apply, save_dual, METHOD, MODALITY, CMAP
from fig1_model_benchmark import RES, PICK, G, long_form, stars, zscore
import glob

OUT = Path("/mnt/HDD2B/Users/yancui/Saturn/published_figures/fig1")
ORDER = ["Saturn", "GASTON", "MEFISTO", "MISO", "SCIGMA", "SpatialGlue"]
KEY = {"GASTON": "GASTON (default)"}                 # figure label -> key in tables / npz
LABEL, TICK = {}, {}                                 # GASTON = the official package at its published default


def ink(c, max_l=0.62):
    """Text colour for a swatch: light theme colours are darkened (same hue) so labels stay readable."""
    import colorsys
    from matplotlib.colors import to_rgb, to_hex
    h, l, sat = colorsys.rgb_to_hls(*to_rgb(c))
    return to_hex(colorsys.hls_to_rgb(h, min(l, max_l), sat))


def load_table():
    T = pd.concat([pd.read_csv(f) for f in glob.glob(str(RES / "bench_full_s*.csv"))])
    T = T[T.grid == "hard"].copy()
    T["method"] = T.method.replace({v: k for k, v in KEY.items()})
    return T


def violins(ax, L, key, col, ylabel, better):
    wide = L.pivot_table(index=key, columns="method", values=col)
    rng = np.random.default_rng(0)
    data = [wide[m].dropna().values for m in ORDER]
    parts = ax.violinplot(data, positions=range(len(ORDER)), widths=0.8, showextrema=False, bw_method=0.3)
    for b, m in zip(parts["bodies"], ORDER):
        b.set_facecolor(METHOD[m]); b.set_edgecolor("none"); b.set_alpha(0.35)
    for i, (m, v) in enumerate(zip(ORDER, data)):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=4, lw=0, color=METHOD[m], alpha=0.8, zorder=2)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([i, i], [q1, q3], color="#1a1a1a", lw=1.6, zorder=3)
        ax.scatter([i], [med], s=12, color="white", edgecolor="#1a1a1a", lw=0.7, zorder=4)
    top = max(np.max(v) for v in data); step = 0.09
    for j, m in enumerate(ORDER[1:], start=1):
        pair = wide[["Saturn", m]].dropna()
        p = wilcoxon(pair["Saturn"], pair[m], alternative="greater" if better == "high" else "less").pvalue
        y = top + 0.05 + (j - 1) * step
        ax.plot([0, 0, j, j], [y - 0.02, y, y, y - 0.02], color="#333333", lw=0.6)
        ax.text(j / 2, y - 0.004, f"{stars(p)}  (n={len(pair)})" if stars(p) == "n.s." else stars(p),
                ha="center", va="bottom", fontsize=6)
    ax.set_ylim(-0.03, top + 0.07 + (len(ORDER) - 1) * step)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0] if top > 0.8 else np.round(np.linspace(0, top, 5), 2))
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([TICK.get(m, m) for m in ORDER], rotation=40, ha="right", rotation_mode="anchor", fontsize=6.5)
    for tl, m in zip(ax.get_xticklabels(), ORDER): tl.set_color(ink(METHOD[m]))
    ax.set_ylabel(ylabel, fontsize=7); ax.set_xlim(-0.6, len(ORDER) - 0.4)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


def tile(a, v, edge, lw):
    im = a.imshow(zscore(v).reshape(G, G), cmap=CMAP["score"], origin="lower", vmin=-2, vmax=2, interpolation="nearest")
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_visible(True); s.set_edgecolor(edge); s.set_linewidth(lw)
    return im


def maps(fig, gs, run, T, row0, first, label=None, note=True):
    z = np.load(RES / "axes" / f"{run}.npz", allow_pickle=True); t = np.asarray(z["true"], float)
    ang, cra, cf, sd = run.split("_")[1:]
    R = T[(T.angle == int(ang[1:])) & (T.cross_RA == float(cra[1:])) & (T.coupled_frac == float(cf))
          & (T.seed == int(sd[1:]))].set_index("method")
    im = None
    for k, (ax_lab, mc) in enumerate([("RNA axis", MODALITY["RNA"]), ("ATAC axis", MODALITY["ATAC"])]):
        for c, m in enumerate(["Ground truth"] + ORDER):
            a = fig.add_subplot(gs[row0 + k, c])
            if m == "Ground truth":
                im = tile(a, t[:, k], "#1a1a1a", 1.2)
            else:
                v = np.asarray(z[KEY.get(m, m)], float)[:, k]; v = v if np.corrcoef(v, t[:, k])[0, 1] >= 0 else -v
                im = tile(a, v, METHOD["Saturn"] if m == "Saturn" else "#9aa2aa", 1.2 if m == "Saturn" else 0.5)
                if m in R.index:
                    rec = float(R.loc[m, "rec_R" if k == 0 else "rec_A"]); lk = float(R.loc[m, "leak_R" if k == 0 else "leak_A"])
                    a.text(0.5, -0.05, f"rec {rec:.2f}" + (f" · leak {lk:.2f}" if note else ""), transform=a.transAxes,
                           ha="center", va="top", fontsize=5.5, color=METHOD["Saturn"] if m == "Saturn" else "#333333")
            if first and k == 0:
                a.set_title(LABEL.get(m, m), fontsize=6.5, fontweight="bold", pad=3,
                            color="#1a1a1a" if m == "Ground truth" else ink(METHOD[m]))
            if c == 0:
                a.set_ylabel(ax_lab if label is None else f"{label}\n{ax_lab}", fontsize=6, color=mc)
    return im


def main():
    """supplementary panel: every hard condition (seed 0). Panels b-d of Fig 1 are exported by fig1_model_benchmark.py."""
    from paper_style import save_panel_svg
    apply(base_pt=7)
    T = load_table()
    runs = sorted({p.stem[:-3] for p in (RES / "axes").glob("hard_*_s0.npz")})
    fig = plt.figure(figsize=(7.2, 1.05 * 2 * len(runs)))
    gs = GridSpec(2 * len(runs), len(ORDER) + 2, figure=fig, width_ratios=[1] * (len(ORDER) + 1) + [0.06],
                  wspace=0.08, hspace=0.35, top=0.975, bottom=0.01)
    for i, r in enumerate(runs):
        a_, c_, f_ = r.split("_")[1:]
        im = maps(fig, gs, f"{r}_s0", T, 2 * i, i == 0, label=f"{a_[1:]}°, c {c_[1:]}, {int(float(f_) * 100)}%", note=False)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[:2, -1]), ticks=[-2, 0, 2]); cb.set_label("isodepth (z)", fontsize=6)
    fig.suptitle("All hard conditions (seed 0): angle, cross-coupling, coupled features", fontsize=7, y=0.99)
    out = save_panel_svg(fig, OUT / "panels" / "figS1_all_hard_conditions_recovered_axes"); plt.close(fig)
    print("->", out)


if __name__ == "__main__":
    main()
