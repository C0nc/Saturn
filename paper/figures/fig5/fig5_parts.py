"""Figure 5 (Sprox-seq tonsil) as three pages, each sub-panel also as its own named SVG (Saturn low-saturation theme).

  part 1  a-g   fits, held-out complexes tracking rho, D1 isodepths vs the authors' GC pseudotime, zone order
  part 2  h-k   complex dynamics along rho vs along GC pseudotime, rotation null, per-follicle replication, examples
  part 3  l-p   latest D1 analysis: rho_RNA = crypt surface -> lymphoid interior, and the held-out complex layer
                re-organises along it (isotype-corrected specificity): CD9 tetraspanin web at the crypt
                lymphoepithelium vs lineage-receptor clusters in the interior; method-free geometry; B1 replication
Builders: fig5_sproxseq.py (a-k) and fig5_D1_crypt_niche.py (l-p, its panels a-e).
-> published_figures/fig5/fig5_part{1,2,3}.{png,svg,json} + panels/fig5<letter>_<name>.svg
Usage: python published_figures/fig5/fig5_parts.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/_style")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/fig2")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/fig5")
from paper_style import apply, save_dual, save_panel_svg
import fig2_lpc as F2
import fig5_sproxseq as S
import fig5_D1_crypt_niche as CN

OUT = Path("/mnt/HDD2B/Users/yancui/Saturn/published_figures/fig5")
NAMES = dict(a="gc_zones_D1", b="fit_quality_R2", c="heldout_complexes_tracking_rho", d="isodepths_vs_gc_pseudotime",
             e="rho_vs_dpt_follicle", f="zone_order_along_rho", g="rho_vs_dpt_all_sections",
             h="complex_change_rho_vs_dpt", i="agreement_vs_rotation_null", j="per_follicle_replication",
             k="example_complexes_along_rho", l="crypt_axis_tissue_rho_RNA", m="crypt_axis_programmes",
             n="CD9_web_vs_crypt_distance", o="D1_B1_replication", p="complex_specificity_heatmap")
CNMAP = dict(l=CN.pa, m=CN.pb, n=CN.pd_, o=CN.pe)                 # crypt-niche builders, renamed in figure order
SIZE = {**{L: s for L, s, _ in S.PANELS}, "l": (2.3, 2.0), "m": (3.0, 2.0), "n": (2.4, 2.2), "o": (2.2, 2.2), "p": (2.4, 2.9)}


def build(fig, L, spec, D, DC):
    if L in CNMAP:
        ax = fig.add_subplot(spec); CNMAP[L](ax, DC); return ax, None
    if L == "p":
        g = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[1, 0.06], wspace=0.08)
        ax = fig.add_subplot(g[0]); CN.pc(ax, DC, fig.add_subplot(g[1])); return ax, None
    return dict((k, b) for k, _, b in S.PANELS)[L](fig, spec, D)


def page(D, DC, letters, size, layout, nudge, stem):
    fig = plt.figure(figsize=size); where = layout(fig); F2.LETTERS.clear()
    for L in letters:
        ax, _ = build(fig, L, where[L], D, DC); F2.letter(ax, L, *nudge[L])
    F2.draw_letters(fig); save_dual(fig, OUT / stem, extra=dict(panels=",".join(letters))); plt.close(fig)


def part1(fig):
    o = GridSpec(3, 1, figure=fig, height_ratios=[1.5, 1.35, 1.6], hspace=0.62, left=0.08, right=0.96, top=0.96, bottom=0.07)
    g1 = GridSpecFromSubplotSpec(1, 3, subplot_spec=o[0], width_ratios=[1.6, 1.0, 1.0], wspace=0.75)
    g3 = GridSpecFromSubplotSpec(1, 3, subplot_spec=o[2], width_ratios=[1.2, 0.9, 1.2], wspace=0.55)
    return dict(a=g1[0], b=g1[1], c=g1[2], d=o[1], e=g3[0], f=g3[1], g=g3[2])


def part2(fig):
    o = GridSpec(2, 1, figure=fig, height_ratios=[2.0, 1.55], hspace=0.6, left=0.08, right=0.96, top=0.95, bottom=0.08)
    g4 = GridSpecFromSubplotSpec(1, 3, subplot_spec=o[0], width_ratios=[1.25, 1.15, 1.0], wspace=0.6)
    return dict(h=g4[0], i=g4[1], j=g4[2], k=o[1])


def part3(fig):
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 0.9, 0.04], height_ratios=[1, 1.15], wspace=0.55, hspace=0.45,
                  left=0.06, right=0.96, top=0.95, bottom=0.08)
    return dict(l=gs[0, 0], m=gs[0, 1], n=gs[1, 0], o=gs[1, 1], p=GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1, 2:])[0])


def main():
    apply(base_pt=6.5)
    D, DC = S.load(), CN.load()
    page(D, DC, "abcdefg", (7.2, 5.6), part1,
         dict(a=(-0.03, 0.015), b=(-0.07, 0.03), c=(-0.07, 0.015), d=(-0.03, 0.02), e=(-0.07, 0.015),
              f=(-0.07, 0.015), g=(-0.07, 0.015)), "fig5_part1")
    page(D, DC, "hijk", (7.2, 4.6), part2,
         dict(h=(-0.07, 0.015), i=(-0.09, 0.015), j=(-0.08, 0.015), k=(-0.07, 0.03)), "fig5_part2")
    page(D, DC, "lmnop", (7.2, 5.4), part3,
         dict(l=(-0.03, 0.02), m=(-0.06, 0.02), n=(-0.07, 0.02), o=(-0.07, 0.02), p=(-0.16, 0.02)), "fig5_part3")
    pdir = OUT / "panels"; pdir.mkdir(exist_ok=True)
    for L in "abcdefghijklmnop":
        fig = plt.figure(figsize=SIZE[L])
        spec = GridSpec(1, 1, figure=fig, left=0.2 if L not in "p" else 0.38, right=0.8, top=0.84, bottom=0.2)[0]
        build(fig, L, spec, D, DC); save_panel_svg(fig, pdir / f"fig5{L}_{NAMES[L]}"); plt.close(fig)
    print("->", OUT)


if __name__ == "__main__":
    main()
