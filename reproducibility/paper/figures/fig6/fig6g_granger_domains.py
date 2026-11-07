"""Fig 6 Granger panel: significant chromatin-gene pairs per domain of each isodepth vs the circular-shift null.
Data: Hole/gaston_mix_mm/scripts/mel_hic_granger_domains.py -> granger_domains_{counts,summary}.csv
One violin pair per domain of each axis (200 subsamples of 500 spots); bracket = BH q over the 12 groups."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/_style")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from paper_style import apply, save_panel_svg, MODALITY, DOMAIN
apply(base_pt=6.5)
import matplotlib as mpl, matplotlib._mathtext as _mt
mpl.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial",
                     "mathtext.sf": "Arial", "mathtext.default": "regular", "mathtext.fallback": None})
_mt.SHRINK_FACTOR = 0.85
G = Path("/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/results/real_saturn/mel_hic_abins_dr50_51_rnadr_insulation_granger")
C, T = pd.read_csv(G / "granger_domains_counts.csv", keep_default_na=False), pd.read_csv(G / "granger_domains_summary.csv")
AXC = {"RNA": MODALITY["RNA"], "HiC": MODALITY["HiC"]}
GROUPS = [("RNA", "epidermis"), ("RNA", "melanocytic tumour"), ("RNA", "plasma / immune"), ("RNA", "deep tumour"),
          ("HiC", "non-tumour"), ("HiC", "tumour")]
XPOS = [0, 1, 2, 3, 4.5, 5.75]
OUT = Path("/mnt/HDD2B/Users/yancui/Saturn/published_figures/fig6/panels/fig6g_granger_domains")


def violin(ax, v, x, color, filled):
    p = ax.violinplot(v, positions=[x], widths=0.36, showextrema=False)
    for b in p["bodies"]:
        b.set_facecolor(color if filled else "white"); b.set_edgecolor(color if filled else "#6b6b6b"); b.set_alpha(1); b.set_linewidth(0.6)
    ax.plot([x - 0.1, x + 0.1], [np.median(v)] * 2, color="white" if filled else "#1a1a1a", lw=0.8, zorder=3)


fig, axes = plt.subplots(2, 1, figsize=(3.6, 3.4), sharex=True, sharey=True)
fig.subplots_adjust(left=0.13, right=0.99, top=0.9, bottom=0.26, hspace=0.3)
top = C[C.kind == "observed"].n_sig.max()
for ax, (layer, nm) in zip(axes, (("A/B", "A/B compartment–gene"), ("insulation", "insulation–gene"))):
    for (a, dom), x in zip(GROUPS, XPOS):
        s = C[(C.axis == a) & (C.domain == dom) & (C.layer == layer)]
        o, n = s[s.kind == "observed"].n_sig.values, s[s.kind == "null"].n_sig.values
        violin(ax, o, x - 0.2, AXC[a], True); violin(ax, n, x + 0.2, "#6b6b6b", False)
        r = T[(T.axis == a) & (T.domain == dom) & (T.layer == layer)].iloc[0]
        h = max(o.max(), n.max()) + 8
        ax.plot([x - 0.2, x - 0.2, x + 0.2, x + 0.2], [h, h + 5, h + 5, h], color="#1a1a1a", lw=0.5)
        ax.text(x, h + 7, f"{np.median(o):.0f} vs {np.median(n):.0f}\nq = {r.q_bh:.3f}", ha="center", va="bottom", fontsize=5, linespacing=1.0)
    ax.set_title(nm, fontsize=6, pad=2, loc="left"); ax.set_ylabel("significant pairs (n = 2,558)", labelpad=1, fontsize=6)
    ax.axvline(3.75, color="#9aa2aa", lw=0.4, ls=(0, (2, 2)))
    ax.set_ylim(50, top + 85); ax.tick_params(axis="y", labelsize=5.5)
axes[1].set_xticks(XPOS); axes[1].set_xticklabels([d for _, d in GROUPS], rotation=30, ha="right", fontsize=5.5)
for tl, (a, _) in zip(axes[1].get_xticklabels(), GROUPS): tl.set_color(AXC[a])
fig.text(0.13 + 0.86 * 1.8 / 6.75, 0.955, "ρ$_{RNA}$ domains", color=AXC["RNA"], fontsize=6, ha="center", va="bottom")
fig.text(0.13 + 0.86 * 5.35 / 6.75, 0.955, "ρ$_{HiC}$ domains", color=AXC["HiC"], fontsize=6, ha="center", va="bottom")
fig.legend(handles=[Patch(fc="#777777", ec="#1a1a1a", lw=0.3, label="observed"), Patch(fc="white", ec="#6b6b6b", lw=0.6, label="circular-shift null")],
           loc="lower center", ncol=2, frameon=False, fontsize=5.1, bbox_to_anchor=(0.55, -0.005), handlelength=1.2, columnspacing=0.8,
           title="labels: median observed vs null; median pairs at per-pair BH q < 0.05 = 0 in every group", title_fontsize=5)
save_panel_svg(fig, OUT)
fig.savefig(OUT.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
print(OUT)
