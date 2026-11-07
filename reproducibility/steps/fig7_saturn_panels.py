"""Fig 7 without the GASTON benchmark, from the figure's own builder in the workspace, unmodified.

    fig7_saturn_panels.py steep    write fig7_steep.json (builder's steep_cases); saturn_steep_transition.py reads it
    fig7_saturn_panels.py examples write fig7_examples.npz (builder's examples), in its own process: rebuilding it imports domain_annot.py,
                                   which sets font.family to Arial and would change the font list in the SVG text of the figure process
    fig7_saturn_panels.py          parts 1-3, supplements S6-S8, one SVG per panel

The builder's main() and load() read gaston_default_r2/{summary,restarts}.csv, which only panels a and S6a use. Here load() runs
with those two reads returning empty tables, and main() is replayed without a and S6a: same layouts, same panel builders, same
nudges, same random streams (no panel draws random numbers). Panel a's slot in part 1 and S6a's in S6 stay empty.
-> published_figures/fig7/fig7_part{1,2,3}.{png,svg,json}, figS{6,7,8}*.{png,svg,json}, panels/, fig7_saturn_stats.json
"""
import json, os, sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = os.environ["SATURN_ROOT"]
sys.path.insert(0, f"{ROOT}/Saturn/published_figures/fig7")
import fig7_cross_term as F7

DROPPED = ("a", "S6a")


def load_without_bench():
    read = pd.read_csv
    pd.read_csv = lambda p, *a, **k: pd.DataFrame() if "gaston_default_r2" in str(p) else read(p, *a, **k)
    try:
        return F7.load()
    finally:
        pd.read_csv = read


def main():
    F7.apply(base_pt=6.5)
    D = load_without_bench(); res = {}
    print("main-figure datasets:", D["sets"]["main"], flush=True)
    F7.page(D, res, ["b", "c"], (7.2, 8.6), F7.part1, dict(b=(-0.06, 0.02), c=(-0.06, 0.012)), "fig7_part1")
    F7.page(D, res, ["d", "e", "f"], (7.2, 8.6), F7.part2, dict(d=(-0.13, 0.02), e=(-0.08, 0.02), f=(-0.05, 0.045)), "fig7_part2")
    F7.page(D, res, ["g", "h", "i"], (7.2, 7.75), F7.part3, dict(g=(0.0, 0.052), h=(-0.09, 0.075), i=(0.0, 0.052)), "fig7_part3")
    F7.page(D, res, ["S6b", "S6c", "S6d", "S6e", "S6f"], (7.2, 14.0), F7.supp6,
            {"S6b": (-0.06, 0.02), "S6c": (-0.13, 0.02), "S6d": (-0.07, 0.02), "S6e": (-0.06, 0.012), "S6f": (-0.05, 0.045)}, "figS6_all_datasets")
    F7.page(D, res, ["S7"], (7.2, 9.4), F7.supp, {"S7": (-0.04, 0.01)}, "figS7_gradient")
    F7.page(D, res, ["S8"], (7.2, 5.6), F7.supp8, {"S8": (-0.06, 0.02)}, "figS8_steepness_checks")
    pdir = F7.OUT.parent / "panels"; pdir.mkdir(exist_ok=True)
    for L, nm, size, b, ks in F7.PANELS + F7.SUPP:
        if L in DROPPED: continue
        D["keys"] = D["sets"][ks]; fig = plt.figure(figsize=size)
        spec = GridSpec(1, 1, figure=fig, **F7.panel_box(L))[0]
        b(fig, spec, D); stem = pdir / (f"fig{L}_{nm}" if L.startswith("S") else f"fig7{L}_{nm}"); F7.save_panel_svg(fig, stem)
        if L in "ghi":
            fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
    F7.export_gi_block(D, pdir)
    json.dump(res, open(F7.OUT.parent / "fig7_saturn_stats.json", "w"), indent=1, default=str)
    print("->", F7.OUT.parent)


if __name__ == "__main__":
    if sys.argv[1:] == ["steep"]:
        F7.steep_cases()
    elif sys.argv[1:] == ["examples"]:
        F7.examples()
    else:
        main()
