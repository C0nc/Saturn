"""Two example bins, chromatin -> gene along the isodepth: chromatin (left axis) and gene score (right axis) over 30 windows, all spots.
Dots = window means, line = running mean of 3 windows for display. P and lag from granger_per_bin.csv: Granger chromatin -> gene on cubic-detrended
series, min over lags 1-3, Bonferroni over lags x directions (mel_hic_granger_examples.py)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "<workspace>/Saturn/published_figures/_style")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from paper_style import apply, save_panel_svg, MODALITY
apply(base_pt=6.5)
import matplotlib as mpl, matplotlib._mathtext as _mt
mpl.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial",
                     "mathtext.sf": "Arial", "mathtext.default": "regular", "mathtext.fallback": None})
_mt.SHRINK_FACTOR = 0.85
R = Path("<workspace>/Hole/gaston_mix_mm/results/real_saturn"); DS = "mel_hic_abins_dr50_51_rnadr"; G = R / f"{DS}_insulation_granger"
OUT = Path("<workspace>/Saturn/published_figures/fig6/panels/fig6_granger_example_lines")
EX = [("RNA", "insulation", "IN", "chr2:232500000-233000000", "insulation score", "#C49A5A", "#8a6a2f"),
      ("HiC", "A/B", "AB", "chr1:246500000-247000000", "A/B score", "#9A5A55", "#9A5A55")]
C = np.load(G / "cache_series.npz", allow_pickle=True); bc = list(C["bc"].astype(str))
ax0 = pd.read_csv(R / DS / "axes.csv", index_col=0).loc[bc]; xy = np.c_[ax0.x.values, ax0.y.values]
S = np.asarray(C["S"], float); Ss = S[cKDTree(xy).query(xy, k=25)[1]].mean(1); names = list(C["names"].astype(str)); gl = C["genes"].astype(str)
T = pd.read_csv(G / "granger_per_bin.csv"); T = T[T.rho == "forward"]
AXC = {"RNA": MODALITY["RNA"], "HiC": MODALITY["HiC"]}; GENE = MODALITY["RNA"]


def pfmt(p):
    if p >= 0.001: return f"P = {p:.3f}"
    e = int(np.floor(np.log10(p))); return f"P = {p / 10 ** e:.2f} × 10$^{{{e}}}$".replace("-", "−")


def smooth(v):
    k = np.ones(3) / 3; o = np.convolve(v, k, mode="same"); o[0], o[-1] = v[:2].mean(), v[-2:].mean(); return o


fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.1)); fig.subplots_adjust(left=0.15, right=0.85, top=0.88, bottom=0.12, hspace=0.72)
for ax, (m, layer, key, b, ylab, col, ink) in zip(axes, EX):
    j = names.index(b); d = ax0[f"d_{m}"].values; E = np.array_split(np.argsort(d), 30); rb = np.array([d[e].mean() for e in E])
    c = np.array([np.asarray(C[key], float)[e, j].mean() for e in E]); g = np.array([Ss[e, j].mean() for e in E])
    r = T[(T.axis == m) & (T.layer == layer) & (T.bin == b)].iloc[0]
    ax.scatter(rb, c, s=5, color=col, alpha=0.45, lw=0, zorder=2); ax.plot(rb, smooth(c), color=col, lw=1.4, zorder=3)
    a2 = ax.twinx(); a2.scatter(rb, g, s=5, color=GENE, alpha=0.45, lw=0, zorder=2); a2.plot(rb, smooth(g), color=GENE, lw=1.4, zorder=3)
    ax.set_ylabel(ylab, color=ink, labelpad=1, fontsize=6); a2.set_ylabel("gene score", color=GENE, labelpad=2, fontsize=6)
    ax.tick_params(axis="y", labelcolor=ink, labelsize=5.5); a2.tick_params(axis="y", labelcolor=GENE, labelsize=5.5); ax.tick_params(axis="x", labelsize=5.5)
    a2.spines["right"].set_visible(True); a2.spines["top"].set_visible(False); ax.spines["top"].set_visible(False)
    ax.set_xlabel(f"ρ$_{{{m}}}$ (30 windows)", color=AXC[m], labelpad=1, fontsize=6)
    c_, se = b.split(":"); s0, e0 = (int(x) / 1e6 for x in se.split("-")); gs = ", ".join(gl[j].split(",")[:4])
    ax.text(0.5, 1.2, f"{c_} {s0:g}–{e0:g} Mb · {gs}", transform=ax.transAxes, ha="center", va="bottom", fontsize=6)
    ax.text(0.5, 1.04, f"{layer if layer == 'A/B' else 'insulation'} → gene, {pfmt(r.p_c2g)}, lag {int(r.lag_c2g)}", transform=ax.transAxes, ha="center", va="bottom", fontsize=6)
    print(m, layer, b, dict(p_c2g=r.p_c2g, q_c2g=r.q_c2g, lag=int(r.lag_c2g), p_g2c=r.p_g2c), gl[j])
save_panel_svg(fig, OUT)
fig.savefig(OUT.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03); fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
print(OUT)
