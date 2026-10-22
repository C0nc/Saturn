"""Figure 4 as two pages, each sub-panel also as its own named SVG (Saturn low-saturation theme).

  part 1  a-e   overview: cells, fit quality, composition vs within-type split, isodepths + domains, fitted functions
  part 2  f-n   usage along rho inside cell classes, isoform-only changes, the amygdala/olfactory | cortex switch,
                and isoform switching along the Sst interneuron state transition
Builders: fig4_isoform_sv.py (a-j) and fig4_isoform_v3.py (k-n); panel e = original | fitted function in space |
bin means + fitted function, for gene expression (Poisson) and isoform usage (binomial).
-> published_figures/fig4/fig4_part{1,2}.{png,svg,json} + panels/fig4<letter>_<name>.svg
Usage: python published_figures/fig4/fig4_parts.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/_style")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/fig2")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/fig4")
from paper_style import apply, save_dual, save_panel_svg
import fig2_lpc as F2
import fig4_isoform_sv as F4S
import fig4_isoform_v3 as V3

OUT = Path("/mnt/HDD2B/Users/yancui/Saturn/published_figures/fig4")
NAMES = dict(a="cells_by_class", b="fit_quality_R2", c="composition_vs_within_split", d="isodepths_and_domains",
             e="fitted_functions", f="usage_along_rho_by_class", g="isoform_only_changes", h="switch_scatter",
             i="switch_curves_at_boundary", j="switch_composition_vs_within", k="sst_state_in_space",
             l="sst_state_by_domain", m="sst_isoform_switches", n="sst_switch_screen")
SIZE = {L: s for L, s, _ in V3.PANELS}
BUILD = {L: b for L, _, b in V3.PANELS}


def page(D, res, letters, size, layout, nudge, stem):
    fig = plt.figure(figsize=size); where = layout(fig)
    F2.LETTERS.clear()
    for L in letters:
        ax, out = BUILD[L](fig, where[L], D); res[L] = out; F2.letter(ax, L, *nudge[L])
    F2.draw_letters(fig)
    save_dual(fig, OUT / stem, extra=dict(panels=",".join(letters))); plt.close(fig)


def part1(fig):
    o = GridSpec(3, 1, figure=fig, height_ratios=[1.55, 1.45, 3.0], hspace=0.6, left=0.07, right=0.91, top=0.97, bottom=0.03)
    g1 = GridSpecFromSubplotSpec(1, 3, subplot_spec=o[0], width_ratios=[1.75, 0.95, 1.05], wspace=0.75)
    return dict(a=g1[0], b=g1[1], c=g1[2], d=o[1], e=o[2])


def part2(fig):
    o = GridSpec(4, 1, figure=fig, height_ratios=[1.6, 1.45, 1.75, 1.6], hspace=0.72, left=0.07, right=0.91, top=0.97, bottom=0.05)
    g4 = GridSpecFromSubplotSpec(1, 2, subplot_spec=o[0], width_ratios=[1.0, 0.2], wspace=0.1)
    g5 = GridSpecFromSubplotSpec(1, 2, subplot_spec=o[1], width_ratios=[1.0, 0.22], wspace=0.1)
    g6 = GridSpecFromSubplotSpec(1, 3, subplot_spec=o[2], width_ratios=[0.95, 1.75, 0.95], wspace=0.62)
    g7 = GridSpecFromSubplotSpec(1, 4, subplot_spec=o[3], width_ratios=[1.05, 0.85, 2.5, 1.0], wspace=0.75)
    return dict(f=g4[0], g=g5[0], h=g6[0], i=g6[1], j=g6[2], k=g7[0], l=g7[1], m=g7[2], n=g7[3])


def main():
    apply(base_pt=6.5)
    F2.MODALITY["Isoform"] = F4S.C_ISO
    D = F4S.load(); res = {}
    page(D, res, "abcde", (7.2, 7.4), part1,
         dict(a=(-0.03, 0.015), b=(-0.06, 0.015), c=(-0.07, 0.015), d=(-0.03, 0.02), e=(-0.06, 0.03)), "fig4_part1")
    page(D, res, "fghijklmn", (7.2, 8.3), part2,
         dict(f=(-0.06, 0.03), g=(-0.06, 0.03), h=(-0.07, 0.03), i=(-0.06, 0.03), j=(-0.1, 0.03),
              k=(-0.03, 0.025), l=(-0.07, 0.02), m=(-0.06, 0.025), n=(-0.07, 0.02)), "fig4_part2")
    pdir = OUT / "panels"; pdir.mkdir(exist_ok=True)
    for L in "abcdefghijklmn":
        fig = plt.figure(figsize=SIZE[L])
        spec = GridSpec(1, 1, figure=fig, left=0.14, right=0.86, top=0.82, bottom=0.22)[0]
        BUILD[L](fig, spec, D); save_panel_svg(fig, pdir / f"fig4{L}_{NAMES[L]}"); plt.close(fig)
    json.dump(dict(features=res["e"]), open(OUT / "fig4_parts_stats.json", "w"), indent=1, default=str)
    print("->", OUT)


if __name__ == "__main__":
    main()
