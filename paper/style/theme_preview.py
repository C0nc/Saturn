"""Preview + checks for the Saturn theme: old vs new swatches per palette, colormaps, colour-blind separation.

Separation = minimum pairwise CAM02-UCS distance inside each categorical palette, under normal vision and
simulated deuteranomaly / protanomaly (severity 100). Rough guide: < 10 hard to tell apart, > 20 comfortable.
Usage: python published_figures/_style/theme_preview.py [old_module new_module]
"""
import sys, itertools
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from colorspacious import cspace_convert
sys.path.insert(0, str(Path(__file__).parent))
import importlib
import saturn_theme as TH

OLD = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "paper_style_v1_saturated")
NEW = importlib.import_module(sys.argv[2] if len(sys.argv) > 2 else "paper_style")
SETS = ["MODALITY", "METHOD", "DOMAIN", "GCZONE", "CELLCLASS", "PROGRAM", "LINEAGE", "LESION", "GENOTYPE", "REGION"]


def sep(cols):
    out = {}
    for name, cvd in (("normal", None), ("deutan", "deuteranomaly"), ("protan", "protanomaly")):
        rgb = np.array([to_rgb(c) for c in cols])
        if cvd:
            rgb = np.clip(cspace_convert(rgb, {"name": "sRGB1+CVD", "cvd_type": cvd, "severity": 100}, "sRGB1"), 0, 1)
        J = cspace_convert(rgb, "sRGB1", "CAM02-UCS")
        out[name] = min(np.linalg.norm(a - b) for a, b in itertools.combinations(J, 2)) if len(J) > 1 else np.nan
    return out


def uniq(d):
    seen, keep = set(), {}
    for k, v in d.items():
        if v.lower() not in seen: seen.add(v.lower()); keep[k] = v
    return keep


def main():
    rows = [(s, uniq(getattr(OLD, s)), uniq(getattr(NEW, s))) for s in SETS]
    n = sum(max(len(o), len(nw)) for _, o, nw in rows)
    fig = plt.figure(figsize=(9, 0.2 * n + 3.6))
    ax = fig.add_axes([0, 0.2, 1, 0.8]); ax.set_axis_off(); y = 0
    for s, o, nw in rows:
        so, sn = sep(list(o.values())), sep(list(nw.values()))
        ax.text(0, y, f"{s}   min ΔE old {so['normal']:.0f}/{so['deutan']:.0f}/{so['protan']:.0f}  ->  "
                      f"new {sn['normal']:.0f}/{sn['deutan']:.0f}/{sn['protan']:.0f}  (normal/deutan/protan)",
                fontsize=7, fontweight="bold", va="center"); y -= 1
        for k in nw:
            ax.add_patch(plt.Rectangle((0.0, y - 0.4), 0.06, 0.8, color=o.get(k, "#ffffff")))
            ax.add_patch(plt.Rectangle((0.07, y - 0.4), 0.06, 0.8, color=nw[k]))
            ax.text(0.14, y, f"{k}   {o.get(k, '-')} -> {nw[k]}", fontsize=6, va="center"); y -= 1
        print(s, {k: round(v, 1) for k, v in sn.items()}, "| old", {k: round(v, 1) for k, v in so.items()})
    ax.set_xlim(0, 1); ax.set_ylim(y, 1)
    grad = np.linspace(0, 1, 256)[None]
    for i, (k, o) in enumerate([("score", "coolwarm"), ("pseudotime", "viridis"), ("recovery", "Blues"), ("cluster", "tab10")]):
        for j, cm in enumerate([o, NEW.CMAP[k]]):
            a = fig.add_axes([0.03 + 0.24 * i, 0.1 - 0.06 * j, 0.21, 0.04])
            a.imshow(grad if k != "cluster" else np.arange(10)[None], cmap=cm, aspect="auto"); a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{k}: {cm}", fontsize=6, pad=2)
    out = Path(__file__).parent.parent / "_palettes" / "saturn_theme_preview.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); print("->", out)


if __name__ == "__main__":
    main()
