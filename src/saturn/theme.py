"""The Saturn manuscript colour theme: low-saturation anchors, colormaps and the locked term colours.

  ANCHOR     named theme colours
  CMAPS      matplotlib colormaps, registered on import (each also as ``<name>_r``):
               saturn_div   navy -> mist -> rust, diverging: isodepth, scores, correlation
               saturn_seq   night -> sea -> mist: pseudotime
               saturn_mono  mist -> navy: 0-1 quality, gains
               saturn_cat   10 categorical colours: domains, clusters
  CMAP       role -> colormap name, as used in the paper figures
  MODALITY   modality -> colour, the same in every figure
  REGION     LPC lesion regions
  set_theme  matplotlib rcParams of the paper figures (Arial, small text, no top/right spines)
"""
from __future__ import annotations
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

ANCHOR = {
    "night": "#14244A", "navy": "#2B3F6B", "slate": "#6A7595", "sea": "#3E7987", "teal": "#5F9AA3",
    "aqua": "#9CC5C9", "mist": "#E6EBEA", "stone": "#959D9E", "silver": "#B8C5C6", "blush": "#E2C7C9",
    "rose": "#B98A8E", "rust": "#9A5A55", "ochre": "#C49A5A", "sand": "#D9C49A", "plum": "#7E6A8C",
    "sage": "#7E9A7A", "ink": "#3E4243",
}

CATEGORICAL = ["#2B3F6B", "#C49A5A", "#5F9AA3", "#B98A8E", "#7E6A8C", "#7E9A7A", "#9A5A55", "#9CC5C9", "#D9C49A", "#6A7595"]


def _lin(name, cols):
    return LinearSegmentedColormap.from_list(name, cols, N=256)


CMAPS = {
    "saturn_div": _lin("saturn_div", ["#1F3461", "#4F7F93", "#A9C4C8", "#EEF0EE", "#E2C3C0", "#C08A85", "#8A4E4B"]),
    "saturn_seq": _lin("saturn_seq", ["#14244A", "#2B4A72", "#3E7987", "#7FB0B5", "#C9DCDB", "#EEF1EF"]),
    "saturn_mono": _lin("saturn_mono", ["#EEF1EF", "#B8CBCD", "#6E8FA6", "#2B3F6B", "#14244A"]),
    "saturn_cat": ListedColormap(CATEGORICAL, name="saturn_cat"),
}

CMAP = {
    "score": "saturn_div",          # isodepth, scores, correlation, binary contrasts (centre it)
    "expression": "saturn_seq_r",   # counts, CP10k, usage: low = mist, high = night
    "distance": "saturn_seq_r",
    "pseudotime": "saturn_seq",
    "recovery": "saturn_mono",      # 0-1 quality, R2, gains
    "cluster": "saturn_cat",
}

MODALITY = {"RNA": "#2B3F6B", "ATAC": "#C49A5A", "Protein": "#7E6A8C", "Isoform": "#6A7FA6", "Hi-C": "#9A5A55",
            "H&E": "#7E9A7A"}
REGION = {"lesion": "#9D6262", "distal WM focus": "#B38EAF", "distal_WM": "#B38EAF", "other": "#D3D7D7"}
REFERENCE = "#959D9E"               # dashed reference lines (chance level, raw control)


def register():
    for k, c in CMAPS.items():
        for name, cm in ((k, c), (k + "_r", c.reversed())):
            if name not in mpl.colormaps:
                mpl.colormaps.register(cm, name=name)


def set_theme(base_pt: float = 7):
    """Apply the paper's matplotlib settings: Arial (with fallbacks), base_pt text, no top/right spines, theme colour cycle."""
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": base_pt, "axes.titlesize": base_pt, "axes.labelsize": base_pt, "xtick.labelsize": base_pt - 1,
        "ytick.labelsize": base_pt - 1, "legend.fontsize": base_pt - 1, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL), "image.cmap": "saturn_div",
        "figure.dpi": 150, "savefig.dpi": 300, "svg.fonttype": "none", "pdf.fonttype": 42,
    })


register()
