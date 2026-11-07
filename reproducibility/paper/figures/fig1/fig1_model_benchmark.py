"""Figure 1 — Saturn: one isodepth per modality, and what it recovers on the hard grid.

  (a) model schematic: spot coordinates -> per-modality isodepth MLPs -> M x M additive
      pathways (diagonal = own axis, off-diagonal = cross) -> MSE on each modality's target
  (b) how the benchmark data are generated (simulate_v5, drawn from the simulator itself; fig1_sim_diagram.py)
  (c) ground truth vs every method's recovered isodepths, one representative hard condition
  (d) axis recovery over every hard run: violins, brackets = one-sided paired Wilcoxon vs Saturn
  (e) cross-axis leak over the same runs, same test
GASTON = the official package at its published default; the global-scaled ablation arm is not shown.
Panels (c)-(e) come from fig1_violin_maps.py.
Output: published_figures/fig1/fig1.{png,svg,json}  (Saturn low-saturation theme)

Benchmark: v5 hard grid (two true axes rotated 45 or 90 deg apart, cross-modality
coupling 0.8 / 0.95, 60 / 90 % coupled features, 3 seeds -> 24 runs x 2 axes).
GASTON is run at its published default.

Metrics are read from the benchmark's own result tables, never recomputed:
  rec  = sign-invariant |Spearman| between recovered and true axis
  leak = the benchmark's residual contamination by the OTHER true axis. A raw
         corr(recovered_1, true_2) is NOT leak: at 45 deg the two true axes already
         correlate at 0.68.

Panel (b, c) condition = hard_a45_c0.8_0.6_s0, the run whose per-method recovery sits
closest to the per-method means over all hard runs (not hand-picked).

Sources
  results/bench_v5_full/bench_full_s*.csv     per-run metrics
  results/bench_v5_full/axes/hard_*.npz       recovered axes (900 spots x 2) per method
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/_style")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/saturn_repo/src")       # Saturn package (saturn_repo)
from paper_style import apply, save_dual, save_panel_svg, METHOD, METHOD_ORDER_MM, MODALITY, CMAP
import matplotlib as _mpl
CMAP_CAT = _mpl.colormaps[CMAP["cluster"]]

RES = Path("/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/results/bench_v5_full")
OUT = Path("/mnt/HDD2B/Users/yancui/Saturn/published_figures/fig1/fig1")
PICK = "hard_a45_c0.8_0.6_s0"
NAME = {"GASTON": "GASTON (default)"}          # label in figure -> key in the result tables
G = 30
C_RNA, C_ATAC = MODALITY["RNA"], MODALITY["ATAC"]


def load_table():
    T = pd.concat([pd.read_csv(f) for f in glob.glob(str(RES / "bench_full_s*.csv"))])
    T = T[T.grid == "hard"].copy()
    T["method"] = T.method.replace({v: k for k, v in NAME.items()})
    return T


def long_form(T):
    """one row per (run, axis): rec and leak, so methods pair on identical runs"""
    key = ["angle", "cross_RA", "coupled_frac", "seed"]
    a = T[key + ["method", "rec_R", "leak_R"]].rename(columns={"rec_R": "rec", "leak_R": "leak"}).assign(axis="RNA")
    b = T[key + ["method", "rec_A", "leak_A"]].rename(columns={"rec_A": "rec", "leak_A": "leak"}).assign(axis="ATAC")
    return pd.concat([a, b]), key + ["axis"]


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < .05 else "n.s."


# --------------------------------------------------------------------------- #
# (a) schematic
# --------------------------------------------------------------------------- #
def box(ax, x, y, w, h, fc, ec, lw=0.8, r=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, transform=ax.transAxes))


def arrow(ax, x0, y0, x1, y1, lw=0.9, color="#333333"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), transform=ax.transAxes,
                                 arrowstyle="-|>", mutation_scale=7, lw=lw, color=color))


def ramp(ax, x, y, w, h, angle):
    """a tiny isodepth tile: a coolwarm ramp at the given angle"""
    g = np.linspace(-1, 1, 24)
    X, Y = np.meshgrid(g, g)
    Z = np.cos(np.deg2rad(angle)) * X + np.sin(np.deg2rad(angle)) * Y
    ins = ax.inset_axes([x, y, w, h])
    ins.imshow(Z, cmap=CMAP["score"], origin="lower", interpolation="bilinear")
    ins.set_xticks([]); ins.set_yticks([])
    for s in ins.spines.values():
        s.set_linewidth(0.6); s.set_color("#333333")


def schematic(ax):
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fs, fss = 7, 6

    # spots
    ax.text(0.005, 0.93, "Spots", fontsize=fs, fontweight="bold", transform=ax.transAxes)
    gx, gy = np.meshgrid(np.linspace(0.015, 0.105, 6), np.linspace(0.30, 0.78, 6))
    ax.scatter(gx.ravel(), gy.ravel(), s=5, color="#5c6670", transform=ax.transAxes, lw=0)
    ax.text(0.005, 0.16, "coordinates (x, y)\none position per spot", fontsize=fss,
            transform=ax.transAxes, va="top", color="#333333")
    arrow(ax, 0.125, 0.54, 0.165, 0.54)

    # per-modality isodepth networks
    ax.text(0.17, 0.93, "Per-modality isodepth", fontsize=fs, fontweight="bold", transform=ax.transAxes)
    for yy, col, m in [(0.60, C_RNA, "RNA"), (0.30, C_ATAC, "ATAC")]:
        box(ax, 0.17, yy, 0.10, 0.19, fc="white", ec=col, lw=1.1)
        ax.text(0.22, yy + 0.095, f"MLP$_{{\\mathrm{{{m}}}}}$", fontsize=fs, ha="center", va="center",
                color=col, transform=ax.transAxes)
        arrow(ax, 0.272, yy + 0.095, 0.297, yy + 0.095, color=col)
        ramp(ax, 0.30, yy + 0.01, 0.055, 0.17, 20 if m == "RNA" else 110)
        ax.text(0.362, yy + 0.095, f"ρ$_{{\\mathrm{{{m}}}}}$", fontsize=fs, va="center",
                color=col, transform=ax.transAxes)
    ax.text(0.17, 0.16, "one scalar field per modality\n(general M; M = 2 shown)", fontsize=fss,
            transform=ax.transAxes, va="top", color="#333333")
    arrow(ax, 0.40, 0.54, 0.435, 0.54)

    # M x M additive pathways
    ax.text(0.44, 0.93, "M × M additive pathways", fontsize=fs, fontweight="bold", transform=ax.transAxes)
    x0, y0, cw, ch, gap = 0.475, 0.30, 0.075, 0.20, 0.012
    ax.text(x0 + cw / 2, 0.82, "from ρ$_{\\mathrm{RNA}}$", fontsize=fss, ha="center", color=C_RNA, transform=ax.transAxes)
    ax.text(x0 + cw * 1.5 + gap, 0.82, "from ρ$_{\\mathrm{ATAC}}$", fontsize=fss, ha="center", color=C_ATAC, transform=ax.transAxes)
    ax.text(x0 - 0.008, y0 + ch * 1.5 + gap, "Y$_{\\mathrm{RNA}}$", fontsize=fss, ha="right", va="center", color=C_RNA, transform=ax.transAxes)
    ax.text(x0 - 0.008, y0 + ch / 2, "Y$_{\\mathrm{ATAC}}$", fontsize=fss, ha="right", va="center", color=C_ATAC, transform=ax.transAxes)
    t = np.linspace(0, 1, 30)
    curves = {(0, 0): np.clip(1.6 * t - 0.2, 0, None) ** 0.9, (1, 1): np.clip(1.4 * t - 0.1, 0, None),
              (0, 1): 0.15 + 0.12 * t, (1, 0): 0.2 - 0.08 * t}
    for r_ in (0, 1):                       # r_ = target row (0 = RNA, top)
        for c_ in (0, 1):                   # c_ = source axis
            own = r_ == c_
            bx = x0 + c_ * (cw + gap); by = y0 + (1 - r_) * (ch + gap)
            src = C_RNA if c_ == 0 else C_ATAC
            box(ax, bx, by, cw, ch, fc="#f4f6f8" if own else "white", ec=src if own else "#b8bfc6",
                lw=1.2 if own else 0.6)
            yv = curves[(r_, c_)]
            ax.plot(bx + 0.008 + t * (cw - 0.016), by + 0.03 + yv / 1.3 * (ch - 0.06),
                    color=src, lw=1.4 if own else 0.8, transform=ax.transAxes)
    ax.text(x0 - 0.04, 0.16, "diagonal = own axis (anchored)\noff-diagonal = cross (λ = 0.03)",
            fontsize=fss, transform=ax.transAxes, va="top", color="#333333")
    arrow(ax, 0.641, 0.54, 0.661, 0.54)

    # reconstruction
    ax.text(0.665, 0.93, "Reconstruction", fontsize=fs, fontweight="bold", transform=ax.transAxes)
    for yy, col, m, lik in [(0.60, C_RNA, "RNA", "GLM-PCs"), (0.30, C_ATAC, "ATAC", "LSI")]:
        box(ax, 0.665, yy, 0.195, 0.19, fc="white", ec=col, lw=1.0)
        ax.text(0.669, yy + 0.125,
                f"Ŷ$_{{\\mathrm{{{m}}}}}$ = b + Σ$_k$ P$_{{\\mathrm{{{m}}}\\leftarrow k}}$(ρ$_k$)",
                fontsize=fss, color=col, transform=ax.transAxes, va="center")
        ax.text(0.669, yy + 0.05, f"MSE on {lik}", fontsize=fss, color="#333333",
                transform=ax.transAxes, va="center")
    ax.text(0.665, 0.16, "3-stage fit, held-out\nrestart acceptance", fontsize=fss,
            transform=ax.transAxes, va="top", color="#333333")
    arrow(ax, 0.865, 0.54, 0.892, 0.54)

    # segmentation
    ax.text(0.897, 0.93, "Post-hoc", fontsize=fs, fontweight="bold", transform=ax.transAxes)
    cols = [CMAP_CAT(i) for i in range(4)]          # theme categorical colours
    for i, c in enumerate(cols):
        ax.add_patch(Rectangle((0.899 + i * 0.024, 0.50), 0.024, 0.08, fc=c, ec="white", lw=0.4,
                               transform=ax.transAxes))
    ax.text(0.899, 0.66, "DP segmentation\nof each ρ", fontsize=fss, transform=ax.transAxes, color="#333333")
    ax.text(0.899, 0.44, "domains · marker\nfunctions of ρ", fontsize=fss, transform=ax.transAxes,
            va="top", color="#333333")


# --------------------------------------------------------------------------- #
# (b) recovered axes
# --------------------------------------------------------------------------- #
def zscore(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def axes_panel(fig, spec, T):
    z = np.load(RES / "axes" / f"{PICK}.npz", allow_pickle=True)
    t = np.asarray(z["true"], float)
    row = T.set_index("method")
    ang, cra, cf, sd = PICK.split("_")[1:]
    row = T[(T.angle == int(ang[1:])) & (T.cross_RA == float(cra[1:])) &
            (T.coupled_frac == float(cf)) & (T.seed == int(sd[1:]))].set_index("method")
    cols = ["Ground truth"] + METHOD_ORDER_MM
    gs = GridSpecFromSubplotSpec(2, len(cols) + 1, subplot_spec=spec, wspace=0.08, hspace=0.34,
                                 width_ratios=[1] * len(cols) + [0.07])
    im = None
    for r, (k, lab, mc) in enumerate([(0, "axis 1 · RNA", C_RNA), (1, "axis 2 · ATAC", C_ATAC)]):
        for c, m in enumerate(cols):
            a = fig.add_subplot(gs[r, c])
            if m == "Ground truth":
                v = t[:, k]
            else:
                v = np.asarray(z[NAME.get(m, m)], float)[:, k]
                v = v if np.corrcoef(v, t[:, k])[0, 1] >= 0 else -v      # isodepth sign is arbitrary
            im = a.imshow(zscore(v).reshape(G, G), cmap=CMAP["score"], origin="lower",
                          vmin=-2, vmax=2, interpolation="nearest")
            a.set_xticks([]); a.set_yticks([])
            ec, lw = ("#1a1a1a", 1.3) if m == "Ground truth" else \
                     (METHOD["Saturn"], 1.3) if m == "Saturn" else ("#9aa2aa", 0.5)
            for s in a.spines.values():
                s.set_visible(True); s.set_edgecolor(ec); s.set_linewidth(lw)
            if r == 0:
                a.set_title(m, fontsize=7, color="#1a1a1a" if m == "Ground truth" else METHOD[m],
                            fontweight="bold", pad=3)
            if c == 0:
                a.set_ylabel(lab, fontsize=7, color=mc)
            if m != "Ground truth":
                rec = float(row.loc[m, "rec_R" if k == 0 else "rec_A"])
                lk = float(row.loc[m, "leak_R" if k == 0 else "leak_A"])
                a.text(0.5, -0.07, f"rec {rec:.2f}  leak {lk:.2f}", transform=a.transAxes,
                       ha="center", va="top", fontsize=6,
                       color=METHOD["Saturn"] if m == "Saturn" else "#333333")
    cax = fig.add_subplot(gs[:, -1])
    cb = fig.colorbar(im, cax=cax, ticks=[-2, 0, 2])
    cb.set_label("isodepth (z)", fontsize=6.5); cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_linewidth(0.5)


# --------------------------------------------------------------------------- #
# (c, d) per-run strips
# --------------------------------------------------------------------------- #
def strips(ax, L, key, col, ylabel, better):
    rng = np.random.default_rng(0)
    wide = L.pivot_table(index=key, columns="method", values=col)
    for i, m in enumerate(METHOD_ORDER_MM):
        v = wide[m].dropna().values
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=6, lw=0, color=METHOD[m], alpha=0.55, zorder=2)
        med = np.median(v)
        ax.plot([i - 0.3, i + 0.3], [med, med], color="#1a1a1a", lw=1.3, zorder=3)
        if m != "Saturn":
            pair = wide[["Saturn", m]].dropna()
            alt = "greater" if better == "high" else "less"
            p = wilcoxon(pair["Saturn"], pair[m], alternative=alt).pvalue
            ax.text(i, 1.075, stars(p), ha="center", va="bottom", fontsize=6.5)
        ax.text(i, -0.14, f"{med:.2f}", ha="center", va="top", fontsize=6, color="#333333")
    ax.set_xticks(range(len(METHOD_ORDER_MM)))
    ax.set_xticklabels(METHOD_ORDER_MM, rotation=30, ha="right", fontsize=6.5)
    for tl, m in zip(ax.get_xticklabels(), METHOD_ORDER_MM):
        tl.set_color(METHOD[m])
    ax.set_ylim(-0.05, 1.13); ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(axis="x", pad=11)
    ax.set_xlim(-0.6, len(METHOD_ORDER_MM) - 0.4)


def main():
    import fig1_violin_maps as VM                      # violins + map tiles (imports this module's helpers)
    apply(base_pt=7)
    T = VM.load_table()
    L, key = long_form(T)
    n_run = int((T.method == "Saturn").sum())

    import fig1_sim_diagram as SD
    fig = plt.figure(figsize=(7.2, 9.2))
    outer = GridSpec(4, 2, figure=fig, height_ratios=[1.0, 1.2, 1.25, 1.45], hspace=0.42, wspace=0.3,
                     top=0.97, bottom=0.06, left=0.08, right=0.98)
    a = fig.add_subplot(outer[0, :]); schematic(a)
    SD.sim_panel(fig, outer[1, :])
    gb = GridSpecFromSubplotSpec(2, len(VM.ORDER) + 2, subplot_spec=outer[2, :], wspace=0.08, hspace=0.3,
                                 width_ratios=[1] * (len(VM.ORDER) + 1) + [0.07])
    im = VM.maps(fig, gb, PICK, T, 0, True)
    cb = fig.colorbar(im, cax=fig.add_subplot(gb[:, -1]), ticks=[-2, 0, 2])
    cb.set_label("isodepth (z)", fontsize=6); cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_linewidth(0.5)
    ac = fig.add_subplot(outer[3, 0]); VM.violins(ac, L, key, "rec", "axis recovery\n|Spearman| with true axis", "high")
    ad = fig.add_subplot(outer[3, 1]); VM.violins(ad, L, key, "leak", "cross-axis leak\n(lower is better)", "low")
    ac.set_title(f"{n_run} runs × 2 axes, hard grid", fontsize=7)
    ad.set_title("one-sided paired Wilcoxon vs Saturn", fontsize=7)
    a.text(0.0, 1.02, "a", transform=a.transAxes, fontsize=9, fontweight="bold", va="bottom")
    for ax_, l in ((ac, "d"), (ad, "e")):
        ax_.text(-0.2, 1.06, l, transform=ax_.transAxes, fontsize=9, fontweight="bold")
    bs = outer[1, :].get_position(fig)
    fig.text(bs.x0 - 0.06, bs.y1 + 0.03, "b", fontsize=9, fontweight="bold")
    fig.text(0.5, bs.y1 + 0.03, "Simulation benchmark: generative model (drawn from the simulator, same condition as c)",
             fontsize=7, ha="center", color="#333333")
    bb = outer[2, :].get_position(fig)
    fig.text(bb.x0 - 0.06, bb.y1 + 0.03, "c", fontsize=9, fontweight="bold")
    fig.text(0.5, bb.y1 + 0.03, f"Ground truth vs recovered isodepths, representative hard condition ({PICK})",
             fontsize=7, ha="center", color="#333333")

    save_dual(fig, OUT, extra=dict(
        panels={"a": "schematic", "b": f"simulation generative model, drawn from simulate_v5 at {PICK}",
                "c": f"ground truth + every method's axes, {PICK}",
                "d": "recovery violins + one-sided paired Wilcoxon vs Saturn",
                "e": "leak violins + one-sided paired Wilcoxon vs Saturn"},
        n_runs=n_run, pairing="(angle, cross_RA, coupled_frac, seed, axis)",
        sources=[str(RES / "bench_full_s*.csv"), str(RES / "axes" / f"{PICK}.npz")],
        methods=VM.ORDER, gaston="published default only", theme="saturn_theme"))
    print(f"-> {OUT}.png / .svg / .json")


PANEL_DIR = OUT.parent / "panels"


def export_panels():
    """every sub-panel as its own named SVG (fig1<letter>_<name>.svg); map tiles are embedded as 600-dpi raster,
    everything else vector. Same builders as the assembled figure."""
    import fig1_violin_maps as VM
    apply(base_pt=7)
    T = VM.load_table(); L, key = long_form(T); n_run = int((T.method == "Saturn").sum())
    fig = plt.figure(figsize=(7.2, 2.0)); schematic(fig.add_subplot(111))
    save_panel_svg(fig, PANEL_DIR / "fig1a_model_schematic"); plt.close(fig)
    import fig1_sim_diagram as SD
    fig = plt.figure(figsize=(7.2, 2.3))
    SD.sim_panel(fig, GridSpec(1, 1, figure=fig, left=0.04, right=0.99, top=0.82, bottom=0.02)[0])
    save_panel_svg(fig, PANEL_DIR / "fig1b_simulation_generative_model"); plt.close(fig)
    fig = plt.figure(figsize=(7.2, 2.4))
    gb = GridSpec(2, len(VM.ORDER) + 2, figure=fig, width_ratios=[1] * (len(VM.ORDER) + 1) + [0.07], wspace=0.08, hspace=0.3)
    im = VM.maps(fig, gb, PICK, T, 0, True)
    cb = fig.colorbar(im, cax=fig.add_subplot(gb[:, -1]), ticks=[-2, 0, 2]); cb.set_label("isodepth (z)", fontsize=6)
    save_panel_svg(fig, PANEL_DIR / "fig1c_ground_truth_vs_recovered_axes"); plt.close(fig)
    for letter_, col, ylab, better, name, title in (
            ("d", "rec", "axis recovery\n|Spearman| with true axis", "high", "axis_recovery_violins", f"{n_run} runs × 2 axes, hard grid"),
            ("e", "leak", "cross-axis leak\n(lower is better)", "low", "cross_axis_leak_violins", "one-sided paired Wilcoxon vs Saturn")):
        fig, ax = plt.subplots(figsize=(3.5, 3.0)); VM.violins(ax, L, key, col, ylab, better); ax.set_title(title, fontsize=7)
        save_panel_svg(fig, PANEL_DIR / f"fig1{letter_}_{name}"); plt.close(fig)
    print(f"-> panel SVGs in {PANEL_DIR}")


if __name__ == "__main__":
    main()
    export_panels()
