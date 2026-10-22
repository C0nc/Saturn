"""Figure 3 as three pages, each sub-panel also as its own named SVG (Saturn low-saturation theme).

  part 1  a-e   overview: H&E + FOV, fit quality, own/cross contribution, isodepths + domains, fitted functions
  part 2  f-j   chromatin-transcription coupling (function-matched peaks, chromatin potential) and iStar interface
  part 3  k-n   CellRank 2 fate arrows on the Saturn axes and transition zones vs Mellon
  figS3         the arrows over the RNA domains, the fusion-weight sweep, fate maps with lineage drivers
Builders: fig3_melanoma.py (a-j) and fig3_melanoma_v2.py (k-o); panel e uses Fig 2's three-column feature layout
(original | fitted function in space | bin means + fitted function).
-> published_figures/fig3/fig3_part{1,2,3}.{png,svg,json} + panels/fig3<letter>_<name>.svg
Usage: python published_figures/fig3/fig3_parts.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/_style")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/fig2")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/fig3")
from paper_style import apply, save_dual, save_panel_svg
import fig2_lpc as F2
import fig3_melanoma as F3
import fig3_melanoma_v2 as V2

OUT = Path("/mnt/HDD2B/Users/yancui/Saturn/published_figures/fig3")
NAMES = dict(a="he_registered_fov", b="fit_quality_R2", c="own_cross_contribution", d="isodepths_and_domains",
             e="fitted_functions", f="function_matched_peaks", g="chromatin_potential", h="istar_tumour_stroma",
             i="istar_interface_zoom", j="istar_interface_stats", k="cellrank_fate_arrows",
             l="transition_zones_vs_mellon", m="zone_programme_profiles", n="zone_ordering_vs_null",
             S3a="cellrank_fate_arrows_on_domains", S3b="cellrank_fusion_weight", S3c="cellrank_fate_maps_drivers")
SIZE = {**{L: s for L, s, _ in V2.PANELS + V2.SUPP}, "e": (7.2, 2.9)}
BUILD = {L: b for L, _, b in V2.PANELS + V2.SUPP}


def build(fig, L, spec, D, res):
    if L == "i":
        return F3.panel_istar_zoom(fig, spec, res["h"]["zoom_px"]), None
    return BUILD[L](fig, spec, D)


def page(D, res, letters, size, layout, nudge, stem):
    fig = plt.figure(figsize=size); where = layout(fig)
    F2.LETTERS.clear()
    for L in letters:
        ax, out = build(fig, L, where[L], D, res); res[L] = out; F2.letter(ax, L, *nudge[L])
    F2.draw_letters(fig)
    save_dual(fig, OUT / stem, extra=dict(panels=",".join(letters))); plt.close(fig)


def part1(fig):
    o = GridSpec(3, 1, figure=fig, height_ratios=[1.5, 1.45, 3.0], hspace=0.45, left=0.07, right=0.95, top=0.97, bottom=0.03)
    g1 = GridSpecFromSubplotSpec(1, 3, subplot_spec=o[0], width_ratios=[1.6, 1.0, 1.0], wspace=0.45)
    return dict(a=g1[0], b=g1[1], c=g1[2], d=o[1], e=o[2])


def part2(fig):
    o = GridSpec(2, 1, figure=fig, height_ratios=[1.6, 1.9], hspace=0.55, left=0.07, right=0.95, top=0.94, bottom=0.07)
    g4 = GridSpecFromSubplotSpec(1, 3, subplot_spec=o[0], width_ratios=[1.85, 0.85, 0.28], wspace=0.25)
    g5 = GridSpecFromSubplotSpec(1, 3, subplot_spec=o[1], width_ratios=[1.35, 1.0, 1.05], wspace=0.3)
    g5b = GridSpecFromSubplotSpec(2, 1, subplot_spec=g5[1], height_ratios=[1, 0.25], hspace=0.05)
    return dict(f=g4[0], g=g4[1], h=g5[0], i=g5b[0], j=g5[2])


def part3(fig):
    o = GridSpec(3, 1, figure=fig, height_ratios=[1.75, 1.75, 1.65], hspace=0.6, left=0.07, right=0.95, top=0.95, bottom=0.06)
    g8 = GridSpecFromSubplotSpec(1, 2, subplot_spec=o[2], width_ratios=[4.0, 1.45], wspace=0.35)
    return dict(k=o[0], l=o[1], m=g8[0], n=g8[1])


def supp(fig):
    o = GridSpec(2, 1, figure=fig, height_ratios=[1.85, 1.6], hspace=0.6, left=0.07, right=0.95, top=0.93, bottom=0.08)
    g = GridSpecFromSubplotSpec(1, 2, subplot_spec=o[1], width_ratios=[1.0, 3.2], wspace=0.35)
    return {"S3a": o[0], "S3b": g[0], "S3c": g[1]}


def main():
    apply(base_pt=6.5)
    D = F3.load(); F3.D_GLOBAL.update(D); res = {}
    page(D, res, "abcde", (7.2, 6.9), part1,
         dict(a=(-0.03, 0.015), b=(-0.06, 0.015), c=(-0.06, 0.015), d=(-0.03, 0.02), e=(-0.06, 0.03)), "fig3_part1")
    page(D, res, "fghij", (7.2, 4.3), part2,
         dict(f=(-0.05, 0.03), g=(-0.05, 0.03), h=(-0.02, 0.02), i=(-0.02, 0.02), j=(-0.06, 0.03)), "fig3_part2")
    page(D, res, "klmn", (7.2, 6.2), part3,
         dict(k=(-0.03, 0.025), l=(-0.03, 0.025), m=(-0.06, 0.03), n=(-0.07, 0.02)), "fig3_part3")
    page(D, res, ["S3a", "S3b", "S3c"], (7.2, 4.0), supp,
         {"S3a": (-0.03, 0.025), "S3b": (-0.07, 0.015), "S3c": (-0.03, 0.025)}, "figS3_cellrank")
    pdir = OUT / "panels"; pdir.mkdir(exist_ok=True)
    for L in list("abcdefghijklmn") + ["S3a", "S3b", "S3c"]:
        fig = plt.figure(figsize=SIZE[L])
        spec = GridSpec(1, 1, figure=fig, left=0.12, right=0.9, top=0.84, bottom=0.2)[0]
        build(fig, L, spec, D, res); save_panel_svg(fig, pdir / f"fig3{L}_{NAMES[L]}"); plt.close(fig)
    json.dump({k: v for k, v in res.items() if k in "efgj"}, open(OUT / "fig3_parts_stats.json", "w"), indent=1, default=str)
    print("->", OUT)


if __name__ == "__main__":
    main()
