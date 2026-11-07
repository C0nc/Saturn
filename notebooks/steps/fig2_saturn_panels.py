"""Fig. 2 without the SEACells comparison: panels a-j, through the figure's own builders (fig2_lpc.py, fig2_lpc_v2.py), unmodified.

Part 1 (a-g) uses the same layout as fig2_lpc_v2.main, so it should match the published page. Part 2 keeps the upper row (h-j) of
the published grid and leaves out k-l. Writes fig2_part1, fig2_part2 and panels/ under the workspace's published_figures/fig2.
"""
import os, sys
ROOT = os.environ["SATURN_ROOT"]
sys.path.insert(0, f"{ROOT}/Saturn/published_figures/fig2")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import fig2_lpc_v2 as V2
from paper_style import apply, save_dual

F2 = V2.F2
apply(base_pt=6.5)
D = F2.load()
PANELS = [p for p in V2.PANELS if p[0] not in ("k", "l")]
res = {}
# part 1: a-g, as in fig2_lpc_v2.main
fig = plt.figure(figsize=(7.2, 9.3))
outer = GridSpec(4, 1, figure=fig, height_ratios=[1.45, 1.45, 3.0, 2.45], hspace=0.42, left=0.07, right=0.95, top=0.975, bottom=0.03)
g1 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], width_ratios=[2.2, 0.85, 1.1], wspace=0.40)
g4 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[3], width_ratios=[1.25, 1], wspace=0.22)
where = dict(a=g1[0], b=g1[1], c=g1[2], d=outer[1], e=outer[2], f=g4[0], g=g4[1])
nudge = dict(a=(-0.03, 0.012), b=(-0.06, 0.012), c=(-0.06, 0.012), d=(-0.03, 0.02), e=(-0.06, 0.03), f=(-0.075, 0.02), g=(-0.03, 0.02))
F2.LETTERS.clear()
for L, _, build in PANELS:
    if L in where:
        ax, out = build(fig, where[L], D); res[L] = out; F2.letter(ax, L, *nudge[L])
F2.draw_letters(fig)
save_dual(fig, V2.OUT.parent / "fig2_part1", extra=dict(panels="a-g", features=res["e"], tf_heatmap_dropped=res["f"]))
plt.close(fig)
# part 2: h-j, upper row of the published grid
fig = plt.figure(figsize=(7.2, 3.6))
outer = GridSpec(2, 1, figure=fig, height_ratios=[1.4, 1.45], hspace=0.6, left=0.07, right=0.95, top=0.94, bottom=0.14)
g5 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.5, width_ratios=[1.1, 1.1, 0.85])
where = dict(h=g5[0], i=g5[1], j=g5[2])
nudge = dict(h=(-0.085, 0.02), i=(-0.085, 0.02), j=(-0.06, 0.03))
F2.LETTERS.clear()
for L, _, build in PANELS:
    if L in where:
        ax, out = build(fig, where[L], D); res[L] = out; F2.letter(ax, L, *nudge[L])
F2.draw_letters(fig)
save_dual(fig, V2.OUT.parent / "fig2_part2", extra=dict(panels="h-j", gradient=res["i"]))
plt.close(fig)
V2.PANELS = PANELS
V2.export_panels(D)
