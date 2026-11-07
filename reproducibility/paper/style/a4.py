"""Save a finished figure on an exact A4 sheet at true size (nothing rescaled): content keeps its physical size, centred
horizontally, `top` inches from the top edge. Refuses content that leaves the sheet or text below 5 pt at print size."""
from __future__ import annotations
import json
from pathlib import Path
import re
import matplotlib.pyplot as plt
from matplotlib import _mathtext
from paper_style import check_min_font, rasterize_points

# Mathtext draws sub/superscripts at 70 %, so a 5-pt "rho_RNA" prints its subscript at 3.5 pt, and check_min_font cannot see it. Keep
# scripts at full size (still lowered / raised) in every script that saves through this module; main-figure scripts do not import it.
_mathtext.SHRINK_FACTOR = 1.0
# paper_style sets Arial for normal text only; mathtext (rho, sub/superscripts) would otherwise be DejaVu Sans
plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.sf": "Arial", "mathtext.it": "Arial:italic",
                     "mathtext.bf": "Arial:bold", "mathtext.fallback": None})

A4_W, A4_H = 210 / 25.4, 297 / 25.4
MARGIN = 0.25                        # inches of clear paper required on every side


def auto_letters(fig, axes=None, min_w=0.45, min_h=0.3, start=0):
    """one bold letter per sub-plot, row by row, left to right, above the plot's own title and left of its y label. Returns {axes: letter}."""
    W, H = fig.get_size_inches(); fig.canvas.draw(); r = fig.canvas.get_renderer()
    A = [a for a in (fig.axes if axes is None else axes) if a.get_visible() and a.axison]
    A = [a for a in A if a.get_position().width * W >= min_w and a.get_position().height * H >= min_h]
    A.sort(key=lambda a: -a.get_position().y1); rows = []
    for a in A:
        if rows and abs(rows[-1][0].get_position().y1 - a.get_position().y1) * H < 0.3:
            rows[-1].append(a)
        else:
            rows.append([a])
    out = {}; i = start
    for row in rows:
        for a in sorted(row, key=lambda a: a.get_position().x0):
            top = a.get_window_extent(r).y1
            for t in (a.title, a._left_title, a._right_title):
                if t.get_text().strip():
                    top = max(top, t.get_window_extent(r).y1)
            x = a.get_tightbbox(r).x0
            fig.text(x / fig.bbox.width, (top + 1) / fig.bbox.height, chr(97 + i), fontsize=9, fontweight="bold", ha="left", va="bottom")
            out[a] = chr(97 + i); i += 1
    return out


def svg_below_min(path, min_pt=5.0):
    """every font size actually written to the SVG below min_pt -> {size: count}; sizes there include sub/superscripts"""
    out = {}
    for x in re.findall(r"font(?:-size)?:\s*(?:italic |oblique |bold |normal |\d00 )*([\d.]+)px", Path(path).read_text(errors="ignore")):
        if float(x) < min_pt - 0.005:
            out[float(x)] = out.get(float(x), 0) + 1
    return out


def save_a4(fig, stem, top=0.5, dpi=600, extra=None):
    w, h = fig.get_size_inches()
    if h + top + MARGIN > A4_H or w + 2 * MARGIN > A4_W:
        raise ValueError(f"{w:.2f} x {h:.2f} in does not fit A4 with these margins")
    mx, my = (A4_W - w) / 2, A4_H - top - h
    for ax in fig.axes:
        p = ax.get_position(original=True)
        ax.set_position([(p.x0 * w + mx) / A4_W, (p.y0 * h + my) / A4_H, p.width * w / A4_W, p.height * h / A4_H])
    for t in fig.texts:
        x, y = t.get_position(); t.set_position(((x * w + mx) / A4_W, (y * h + my) / A4_H))
    fig.set_size_inches(A4_W, A4_H)
    check_min_font(fig, page_width_in=A4_W); rasterize_points(fig)
    fig.canvas.draw(); bb = fig.get_tightbbox(fig.canvas.get_renderer())      # inches
    if bb.x0 < MARGIN or bb.y0 < MARGIN or bb.x1 > A4_W - MARGIN or bb.y1 > A4_H - MARGIN:
        raise ValueError(f"content leaves the A4 margins: x {bb.x0:.2f}-{bb.x1:.2f}, y {bb.y0:.2f}-{bb.y1:.2f}")
    p = Path(stem); p.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": "standard"}):        # paper_style sets tight, which would crop the sheet away
        for ext in ("png", "svg", "pdf"):
            fig.savefig(p.with_suffix("." + ext), dpi=dpi)
    low = svg_below_min(p.with_suffix(".svg"))
    if low:
        raise ValueError(f"{p}: text below 5 pt in the saved SVG: {low}")
    if "font-family: 'DejaVu Sans'" in p.with_suffix(".svg").read_text(errors="ignore"):
        raise ValueError(f"{p}: non-Arial text in the saved SVG")
    p.with_suffix(".json").write_text(json.dumps(dict(page="A4", size_in=[round(A4_W, 3), round(A4_H, 3)], dpi=dpi,
        content_extent_in=dict(x=[round(bb.x0, 2), round(bb.x1, 2)], y=[round(bb.y0, 2), round(bb.y1, 2)]), **(extra or {})), indent=1, default=str))
    print("->", p, f"content x {bb.x0:.2f}-{bb.x1:.2f}, y {bb.y0:.2f}-{bb.y1:.2f} of {A4_W:.2f} x {A4_H:.2f} in", flush=True)
