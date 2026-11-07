"""Direction of Granger coupling per domain, same bar layout as the full-axis panel.
Data: Hole/gaston_mix_mm/scripts/mel_hic_granger_direction_domains.py. Bar label = % of bins [bins with q < 0.05]; tick = mean over 100 circular shifts."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/_style")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from paper_style import apply, save_panel_svg, MODALITY
apply(base_pt=6.5)
import matplotlib as mpl, matplotlib._mathtext as _mt
mpl.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial",
                     "mathtext.sf": "Arial", "mathtext.default": "regular", "mathtext.fallback": None})
_mt.SHRINK_FACTOR = 0.85
G = Path("/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/results/real_saturn/mel_hic_abins_dr50_51_rnadr_insulation_granger")
T = pd.read_csv(G / "granger_direction_domains.csv")
AXC = {"RNA": MODALITY["RNA"], "HiC": MODALITY["HiC"]}
GROUPS = [("RNA", "epidermis"), ("RNA", "melanocytic tumour"), ("RNA", "plasma / immune"), ("RNA", "deep tumour"), ("HiC", "non-tumour"), ("HiC", "tumour")]
OUT = Path("/mnt/HDD2B/Users/yancui/Saturn/published_figures/fig6/panels/fig6h_granger_direction_domains")
rows = [(a, dom, rd) for a, dom in GROUPS for rd in ("forward", "reversed")]
ys, y = [], 0
for i, r in enumerate(rows):
    ys.append(y); y -= 1.0
    if i % 2 == 1: y -= 0.35
    if i == 7: y -= 0.35
fig, axes = plt.subplots(1, 2, figsize=(4.2, 5.0), sharey=True)
fig.subplots_adjust(left=0.32, right=0.99, top=0.95, bottom=0.14, wspace=0.08)
xmax = max(T.c2g.max(), T.g2c.max()) * 1.55
for ax, (layer, nm) in zip(axes, (("insulation", "insulation"), ("A/B", "A/B compartment"))):
    for yy, (a, dom, rd) in zip(ys, rows):
        s = T[(T.axis == a) & (T.domain == dom) & (T.layer == layer) & (T.reading == rd)].iloc[0]
        for dy, v, q, n, kw in ((0.19, s.c2g, s.c2g_q, s.null_c2g, dict(color=AXC[a], edgecolor="#1a1a1a", lw=0.3)), (-0.19, s.g2c, s.g2c_q, s.null_g2c, dict(color="white", edgecolor=AXC[a], lw=0.7))):
            ax.barh(yy + dy, v, 0.36, **kw)
            ax.plot([n, n], [yy + dy - 0.18, yy + dy + 0.18], color="#1a1a1a" if kw["color"] == "white" else "#d8d8d8", lw=0.7, zorder=3)
            ax.text(v + xmax * 0.012, yy + dy, f"{v:.1f} [{int(q)}]", fontsize=5, va="center", ha="left")
    lag = T[(T.layer == layer) & (T.reading == "forward")].lag_chrom.median()
    ax.set_title(f"{nm}\nlag-1 chromatin {lag:.2f}", fontsize=5.8, pad=2); ax.set_xlabel("bins, P < 0.05 (%)", labelpad=1)
    ax.set_xlim(0, xmax); ax.tick_params(axis="x", labelsize=5.5)
axes[0].set_yticks(ys); axes[0].set_yticklabels([f"{d} {'forward' if r == 'forward' else 'reversed'}" for _, d, r in rows], fontsize=5.3)
for tl, (a, _, _) in zip(axes[0].get_yticklabels(), rows): tl.set_color(AXC[a])
axes[1].tick_params(axis="y", left=False, labelleft=False)
axes[0].set_ylim(min(ys) - 0.6, 0.6)
for ax in axes: ax.axhline((ys[7] + ys[8]) / 2, color="#9aa2aa", lw=0.4, ls=(0, (2, 2)))
fig.text(0.005, (0.95 + 0.14) / 2 + 0.17, "ρ$_{RNA}$ domains", rotation=90, color=AXC["RNA"], fontsize=6, va="center")
fig.text(0.005, 0.14 + 0.1, "ρ$_{HiC}$ domains", rotation=90, color=AXC["HiC"], fontsize=6, va="center")
lg = T[T.reading == "forward"].lag_gene.median()
fig.legend(handles=[Patch(fc="#777777", ec="#1a1a1a", lw=0.3, label="chromatin → gene"), Patch(fc="white", ec="#777777", lw=0.7, label="gene → chromatin"),
                    Line2D([], [], color="#1a1a1a", lw=0.7, label="shifted-null mean")], loc="lower center", ncol=3, frameon=False, fontsize=5.1, bbox_to_anchor=(0.55, 0.0), handlelength=1.2, columnspacing=0.8,
           title=f"label: % [bins with q < 0.05]; n = 2,558 bins; all spots per domain; lag-1 genes {lg:.2f}", title_fontsize=5)
save_panel_svg(fig, OUT)
fig.savefig(OUT.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03); fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
print(OUT)
