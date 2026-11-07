#!/usr/bin/env python
"""Fig. 1 with Saturn only: schematic (a), simulation diagram (b), truth vs recovered maps (c), recovery (d), leak (e).

Every panel is drawn by the figure's own unmodified functions (fig1_model_benchmark.schematic, fig1_sim_diagram.sim_panel,
fig1_violin_maps.maps / violins), which read the workspace benchmark tables. The method list of fig1_violin_maps is set to
Saturn alone, so violins() draws no bracket or Wilcoxon test. Only the page layout below is new.
Output: published_figures/fig1/fig1_saturn_only.{png,svg,json} and fig1/panels_saturn_only/*.svg in the workspace.
"""
import os, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

PF = Path(os.environ["SATURN_ROOT"]).expanduser().resolve() / "Saturn/published_figures"
sys.path[:0] = [str(PF / "_style"), str(PF / "fig1")]
import fig1_model_benchmark as MB, fig1_violin_maps as VM, fig1_sim_diagram as SD
from paper_style import apply, save_dual, save_panel_svg

VM.ORDER[:] = ["Saturn"]
OUT = PF / "fig1/fig1_saturn_only"; PANELS = PF / "fig1/panels_saturn_only"
REC = ("rec", "axis recovery\n|Spearman| with true axis", "high")
LEAK = ("leak", "cross-axis leak\n(lower is better)", "low")


def maps_panel(fig, spec):
    gs = GridSpecFromSubplotSpec(2, 3, subplot_spec=spec, width_ratios=[1, 1, 0.07], wspace=0.08, hspace=0.3)
    im = VM.maps(fig, gs, MB.PICK, T, 0, True)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[:, -1]), ticks=[-2, 0, 2]); cb.set_label("isodepth (z)", fontsize=6)
    cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_linewidth(0.5)


apply(base_pt=7)
T = VM.load_table(); L, key = MB.long_form(T); n_run = int((T.method == "Saturn").sum())
assert n_run == 24, n_run
title = f"{n_run} runs × 2 axes, hard grid"

fig = plt.figure(figsize=(7.2, 8.0))
outer = GridSpec(3, 1, figure=fig, height_ratios=[1.0, 1.3, 1.5], hspace=0.42, top=0.97, bottom=0.06, left=0.08, right=0.98)
a = fig.add_subplot(outer[0]); MB.schematic(a)
SD.sim_panel(fig, outer[1])
low = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], width_ratios=[1.3, 0.75, 0.75], wspace=0.85)
maps_panel(fig, low[0])
d = fig.add_subplot(low[1]); VM.violins(d, L, key, *REC[:2], REC[2]); d.set_title(title, fontsize=7)
e = fig.add_subplot(low[2]); VM.violins(e, L, key, *LEAK[:2], LEAK[2]); e.set_title(title, fontsize=7)
a.text(0.0, 1.02, "a", transform=a.transAxes, fontsize=9, fontweight="bold", va="bottom")
b_pos = outer[1].get_position(fig); c_pos = low[0].get_position(fig)
fig.text(b_pos.x0 - 0.06, b_pos.y1 + 0.045, "b", fontsize=9, fontweight="bold")
fig.text(0.5, b_pos.y1 + 0.045, "Simulation benchmark: generative model (drawn from the simulator, same condition as c)", fontsize=7, ha="center", color="#333333")
fig.text(c_pos.x0 - 0.06, c_pos.y1 + 0.035, "c", fontsize=9, fontweight="bold")
fig.text(c_pos.x0 + 0.01, c_pos.y1 + 0.035, f"Representative hard condition ({MB.PICK})", fontsize=7, color="#333333")
for ax_, l in ((d, "d"), (e, "e")):
    ax_.text(-0.45, 1.06, l, transform=ax_.transAxes, fontsize=9, fontweight="bold")
save_dual(fig, OUT, extra=dict(panels={"a": "schematic", "b": f"simulation generative model at {MB.PICK}", "c": f"truth and Saturn axes, {MB.PICK}",
                                       "d": "Saturn recovery, 24 runs x 2 axes", "e": "Saturn leak, 24 runs x 2 axes"},
                               n_runs=n_run, sources=[str(MB.RES / "bench_full_s*.csv"), str(MB.RES / "axes" / f"{MB.PICK}.npz")], methods=VM.ORDER))
plt.close(fig)

fig = plt.figure(figsize=(7.2, 2.0)); MB.schematic(fig.add_subplot(111))
save_panel_svg(fig, PANELS / "fig1a_model_schematic"); plt.close(fig)
fig = plt.figure(figsize=(7.2, 2.3)); SD.sim_panel(fig, GridSpec(1, 1, figure=fig, left=0.04, right=0.99, top=0.82, bottom=0.02)[0])
save_panel_svg(fig, PANELS / "fig1b_simulation_generative_model"); plt.close(fig)
fig = plt.figure(figsize=(2.6, 2.4)); maps_panel(fig, GridSpec(1, 1, figure=fig)[0])
save_panel_svg(fig, PANELS / "fig1c_ground_truth_vs_recovered_axes"); plt.close(fig)
for letter, spec, name in (("d", REC, "axis_recovery_violin"), ("e", LEAK, "cross_axis_leak_violin")):
    fig, ax = plt.subplots(figsize=(1.8, 2.6)); VM.violins(ax, L, key, spec[0], spec[1], spec[2]); ax.set_title(title, fontsize=7)
    save_panel_svg(fig, PANELS / f"fig1{letter}_{name}"); plt.close(fig)
print(f"-> {OUT}.png, panels in {PANELS}")
