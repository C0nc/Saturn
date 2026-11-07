"""Fig 6 reversal panel: net direction of Granger pairs per domain with each isodepth read forward and reversed.
Data: Hole/gaston_mix_mm/scripts/mel_hic_granger_domains_reversal.py -> granger_domains_reversal_{counts,summary}.csv
net = chromatin->gene pairs minus gene->chromatin pairs. Grey band = central 95% of the null net; bracket = BH q of the
forward-vs-reversed change against the circular-shift null."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/_style")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from paper_style import apply, save_panel_svg, MODALITY
apply(base_pt=6.5)
import matplotlib as mpl, matplotlib._mathtext as _mt
mpl.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial",
                     "mathtext.sf": "Arial", "mathtext.default": "regular", "mathtext.fallback": None})
_mt.SHRINK_FACTOR = 0.85
G = Path("/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/results/real_saturn/mel_hic_abins_dr50_51_rnadr_insulation_granger")
C, T = pd.read_csv(G / "granger_domains_reversal_counts.csv", keep_default_na=False), pd.read_csv(G / "granger_domains_reversal_summary.csv")
C = C[C.kind == "observed"]
AXC = {"RNA": MODALITY["RNA"], "HiC": MODALITY["HiC"]}
GROUPS = [("RNA", "epidermis"), ("RNA", "melanocytic tumour"), ("RNA", "plasma / immune"), ("RNA", "deep tumour"),
          ("HiC", "non-tumour"), ("HiC", "tumour")]
XPOS = [0, 1, 2, 3, 4.5, 5.75]
OUT = Path("/mnt/HDD2B/Users/yancui/Saturn/published_figures/fig6/panels/fig6h_granger_reversal")
fmt = lambda v: f"{v:g}".replace("-", "−")


def violin(ax, v, x, color, filled):
    p = ax.violinplot(v, positions=[x], widths=0.36, showextrema=False)
    for b in p["bodies"]:
        b.set_facecolor(color if filled else "white"); b.set_edgecolor(color); b.set_alpha(1); b.set_linewidth(0.6); b.set_zorder(2)
    ax.plot([x - 0.1, x + 0.1], [np.median(v)] * 2, color="white" if filled else "#1a1a1a", lw=0.8, zorder=3)


fig, axes = plt.subplots(2, 1, figsize=(3.6, 3.4), sharex=True, sharey=True)
fig.subplots_adjust(left=0.13, right=0.99, top=0.9, bottom=0.26, hspace=0.3)
hi = C.net.max()
for ax, (layer, nm) in zip(axes, (("A/B", "A/B compartment–gene"), ("insulation", "insulation–gene"))):
    ax.axhline(0, color="#1a1a1a", lw=0.5, zorder=1)
    for (a, dom), x in zip(GROUPS, XPOS):
        s = C[(C.axis == a) & (C.domain == dom) & (C.layer == layer)]
        f, b = s[s.reading == "forward"].net.values, s[s.reading == "reversed"].net.values
        r = T[(T.axis == a) & (T.domain == dom) & (T.layer == layer)].iloc[0]
        ax.add_patch(Rectangle((x - 0.42, r.null_lo), 0.84, r.null_hi - r.null_lo, fc="#dfe2e5", ec="none", zorder=0))
        violin(ax, f, x - 0.2, AXC[a], True); violin(ax, b, x + 0.2, AXC[a], False)
        h = max(f.max(), b.max(), r.null_hi) + 6
        ax.plot([x - 0.2, x - 0.2, x + 0.2, x + 0.2], [h, h + 4, h + 4, h], color="#1a1a1a", lw=0.5)
        ax.text(x, h + 6, f"{fmt(r.net_fwd)} vs {fmt(r.net_rev)}\nq = {r.q_bh:.3f}", ha="center", va="bottom", fontsize=5, linespacing=1.0)
    ax.set_title(nm, fontsize=6, pad=2, loc="left"); ax.set_ylabel("net pairs (chromatin → gene\nminus gene → chromatin)", labelpad=1, fontsize=5.5)
    ax.axvline(3.75, color="#9aa2aa", lw=0.4, ls=(0, (2, 2)))
    ax.set_ylim(C.net.min() - 8, hi + 62); ax.tick_params(axis="y", labelsize=5.5)
axes[1].set_xticks(XPOS); axes[1].set_xticklabels([d for _, d in GROUPS], rotation=30, ha="right", fontsize=5.5)
for tl, (a, _) in zip(axes[1].get_xticklabels(), GROUPS): tl.set_color(AXC[a])
fig.text(0.13 + 0.86 * 1.8 / 6.75, 0.955, "ρ$_{RNA}$ domains", color=AXC["RNA"], fontsize=6, ha="center", va="bottom")
fig.text(0.13 + 0.86 * 5.35 / 6.75, 0.955, "ρ$_{HiC}$ domains", color=AXC["HiC"], fontsize=6, ha="center", va="bottom")
fig.legend(handles=[Patch(fc="#777777", ec="#1a1a1a", lw=0.3, label="forward"), Patch(fc="white", ec="#555555", lw=0.6, label="reversed"),
                    Patch(fc="#dfe2e5", ec="none", label="95% of null")],
           loc="lower center", ncol=3, frameon=False, fontsize=5.1, bbox_to_anchor=(0.55, -0.005), handlelength=1.2, columnspacing=0.8,
           title="labels: median net forward vs reversed; q: change on reversal vs null", title_fontsize=5)
save_panel_svg(fig, OUT)
fig.savefig(OUT.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
print(OUT)
