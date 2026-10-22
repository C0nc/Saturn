"""Saturn manuscript theme: low-saturation palette + colormaps anchored on two reference covers.

  night   cover 1: deep navy field with silver line art   -> #0A1C49 navy, #6A7595 silver-slate
  ukiyo   cover 2: woodblock sea and sky                    -> #3E7987 sea teal, #5DA2AD, #85C6CC aqua,
                                                               #DAE1E1 mist, #B8C5C6 / #959D9E stone,
                                                               #E2C7C9 blush horizon, #3E4243 ink
Warm accents (needed so RNA/ATAC-type pairs stay distinguishable) are pulled to the same low chroma.

  ANCHOR      the named theme colours
  mute(hex)   carries any legacy colour into the theme: keeps its hue, caps chroma, lifts lightness
  CMAPS       registered matplotlib colormaps (name -> LinearSegmentedColormap), all with a _r variant:
                saturn_div   navy -> sea -> mist -> blush -> rust      diverging (rho, scores, correlation)
                saturn_seq   night -> sea -> mist                      sequential (pseudotime, density)
                saturn_mono  mist -> navy                              0-1 quality metrics (recovery)
                saturn_cat   10 categorical theme colours (ListedColormap)
"""
from __future__ import annotations
import colorsys
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb, to_hex

ANCHOR = {
    "night":     "#14244A",   # cover-1 field, lifted slightly for print
    "navy":      "#2B3F6B",
    "slate":     "#6A7595",   # cover-1 silver line art
    "sea":       "#3E7987",   # cover-2 sea
    "teal":      "#5F9AA3",
    "aqua":      "#9CC5C9",   # cover-2 sky, desaturated
    "mist":      "#E6EBEA",   # cover-2 cloud / paper
    "stone":     "#959D9E",
    "silver":    "#B8C5C6",
    "blush":     "#E2C7C9",   # cover-2 horizon
    "rose":      "#B98A8E",
    "rust":      "#9A5A55",
    "ochre":     "#C49A5A",
    "sand":      "#D9C49A",
    "plum":      "#7E6A8C",
    "sage":      "#7E9A7A",
    "ink":       "#3E4243",   # cover-2 rock ink
}


def mute(hex_: str, max_sat: float = 0.28, l_lo: float = 0.30, l_hi: float = 0.82, toward: str = "#8E9AA6",
         blend: float = 0.12) -> str:
    """Carry a legacy colour into the theme: keep its hue, cap HLS saturation (lighter colours get a lower cap),
    keep lightness in [l_lo, l_hi], then blend a little toward the theme's cool grey so every hue shares the
    same undertone."""
    h, l, s = colorsys.rgb_to_hls(*to_rgb(hex_))
    if s < 0.08:                           # greys stay greys, tinted toward cover-2 stone
        return to_hex(colorsys.hls_to_rgb(0.5, min(max(l, 0.35), 0.88), 0.05))
    s = min(s, max_sat if l < 0.6 else max_sat * 0.85)
    l = min(max(l, l_lo), l_hi)
    rgb = colorsys.hls_to_rgb(h, l, s); g = to_rgb(toward)
    return to_hex(tuple((1 - blend) * a + blend * b for a, b in zip(rgb, g)))


def _lin(name, cols):
    return LinearSegmentedColormap.from_list(name, cols, N=256)


CMAPS = {
    "saturn_div":  _lin("saturn_div", ["#1F3461", "#4F7F93", "#A9C4C8", "#EEF0EE", "#E2C3C0", "#C08A85", "#8A4E4B"]),
    "saturn_seq":  _lin("saturn_seq", ["#14244A", "#2B4A72", "#3E7987", "#7FB0B5", "#C9DCDB", "#EEF1EF"]),
    "saturn_mono": _lin("saturn_mono", ["#EEF1EF", "#B8CBCD", "#6E8FA6", "#2B3F6B", "#14244A"]),
    "saturn_cat":  ListedColormap(["#2B3F6B", "#C49A5A", "#5F9AA3", "#B98A8E", "#7E6A8C", "#7E9A7A",
                                   "#9A5A55", "#9CC5C9", "#D9C49A", "#6A7595"], name="saturn_cat"),
}


def register():
    for k, c in CMAPS.items():
        for name, cm in ((k, c), (k + "_r", c.reversed())):
            if name not in mpl.colormaps:
                mpl.colormaps.register(cm, name=name)


register()
