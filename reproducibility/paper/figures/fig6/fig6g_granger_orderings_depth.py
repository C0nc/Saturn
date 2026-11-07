"""Depth removed. Fig 6 Granger panel with all comparisons: pairs along rho vs shifted chromatin, a random spatial direction, a random spot
order and the other isodepth. Data: Hole/gaston_mix_mm/scripts/mel_hic_granger_domains_orderings.py
Violins = 200 subsamples of 500 spots per domain; brackets = one-sided empirical q of rho vs each comparison, BH over 48 tests."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/_style")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from paper_style import apply, save_panel_svg, MODALITY
apply(base_pt=6.5)
import matplotlib as mpl, matplotlib._mathtext as _mt
mpl.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial",
                     "mathtext.sf": "Arial", "mathtext.default": "regular", "mathtext.fallback": None})
_mt.SHRINK_FACTOR = 0.85
G = Path("/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/results/real_saturn/mel_hic_abins_dr50_51_rnadr_insulation_granger")
C, T = pd.read_csv(G / "granger_domains_orderings_counts_depth.csv"), pd.read_csv(G / "granger_domains_orderings_summary_depth.csv")
AXC = {"RNA": MODALITY["RNA"], "HiC": MODALITY["HiC"]}; OTHER = {"RNA": "HiC", "HiC": "RNA"}
GROUPS = [("RNA", "epidermis"), ("RNA", "melanocytic tumour"), ("RNA", "plasma / immune"), ("RNA", "deep tumour"),
          ("HiC", "non-tumour"), ("HiC", "tumour")]
XPOS = [0, 1, 2, 3, 4.25, 5.25]; W = 0.17
ORDER = ["rho", "shift", "direction", "random", "other"]; OFF = dict(zip(ORDER, (-2 * W, -W, 0, W, 2 * W)))
STY = dict(shift=("white", "#6b6b6b"), direction=("#E3CFA3", "#8a7445"), random=("#D9DDE1", "#7d858c"))
OUT = Path("/mnt/HDD2B/Users/yancui/Saturn/published_figures/fig6/panels/fig6g_granger_orderings_depth")


def violin(ax, v, x, fc, ec, line):
    p = ax.violinplot(v, positions=[x], widths=0.15, showextrema=False)
    for b in p["bodies"]:
        b.set_facecolor(fc); b.set_edgecolor(ec); b.set_alpha(1); b.set_linewidth(0.5)
    ax.plot([x - 0.05, x + 0.05], [np.median(v)] * 2, color=line, lw=0.8, zorder=3)


fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, sharey=True)
fig.subplots_adjust(left=0.065, right=0.995, top=0.93, bottom=0.17, hspace=0.2)
top = C[ORDER].values.max()
for ax, (layer, nm) in zip(axes, (("A/B", "A/B compartment–gene"), ("insulation", "insulation–gene"))):
    ax.axhline(128, color="#9aa2aa", lw=0.5, ls=(0, (3, 2)), zorder=0)
    for (a, dom), x in zip(GROUPS, XPOS):
        s = C[(C.axis == a) & (C.domain == dom) & (C.layer == layer)]
        violin(ax, s.rho.values, x + OFF["rho"], AXC[a], AXC[a], "white")
        for c in ("shift", "direction", "random"):
            violin(ax, s[c].values, x + OFF[c], STY[c][0], STY[c][1], "#1a1a1a")
        violin(ax, s.other.values, x + OFF["other"], AXC[OTHER[a]], AXC[OTHER[a]], "white")
        h = s[ORDER].values.max() + 8
        for i, c in enumerate(("shift", "direction", "random", "other")):
            r = T[(T.axis == a) & (T.domain == dom) & (T.layer == layer) & (T.comparison == c)].iloc[0]
            y = h + i * 42; x0, x1 = x + OFF["rho"], x + OFF[c]
            ax.plot([x0, x0, x1, x1], [y, y + 6, y + 6, y], color="#1a1a1a", lw=0.4)
            ax.text((x0 + x1) / 2, y + 7, f"{r.q_bh:.3f}" + ("*" if r.q_bh < .05 else ""), ha="center", va="bottom", fontsize=5)
    ax.set_title(nm, fontsize=6, pad=2, loc="left"); ax.set_ylabel("nominally significant pairs\n(P < 0.05; n = 2,558)", labelpad=1, fontsize=6)
    ax.axvline(3.625, color="#9aa2aa", lw=0.4, ls=(0, (2, 2)))
    ax.set_ylim(40, top + 190); ax.tick_params(axis="y", labelsize=5.5)
axes[1].set_xticks(XPOS); axes[1].set_xticklabels([d for _, d in GROUPS], rotation=0, ha="center", fontsize=5.5)
for tl, (a, _) in zip(axes[1].get_xticklabels(), GROUPS): tl.set_color(AXC[a])
fig.text(0.065 + 0.93 * 1.9 / 6.1, 0.955, "ρ$_{RNA}$ domains", color=AXC["RNA"], fontsize=6, ha="center", va="bottom")
fig.text(0.065 + 0.93 * 5.1 / 6.1, 0.955, "ρ$_{HiC}$ domains", color=AXC["HiC"], fontsize=6, ha="center", va="bottom")
fig.legend(handles=[Patch(fc="#777777", ec="#1a1a1a", lw=0.3, label="along the domain's isodepth (ρ)"), Patch(fc="white", ec="#6b6b6b", lw=0.5, label="ρ, chromatin shifted"),
                    Patch(fc=STY["direction"][0], ec=STY["direction"][1], lw=0.5, label="random spatial direction"), Patch(fc=STY["random"][0], ec=STY["random"][1], lw=0.5, label="random spot order"),
                    Patch(fc="#555555", ec="none", label="other isodepth (colour of that axis)")],
           loc="lower center", ncol=5, frameon=False, fontsize=5.1, bbox_to_anchor=(0.52, 0.0), handlelength=1.2, columnspacing=0.9,
           title="brackets: q of ρ vs each comparison (BH over 48 tests; * q < 0.05); depth removed; dashed line: 5% of pairs", title_fontsize=5)
save_panel_svg(fig, OUT)
fig.savefig(OUT.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
print(OUT)
