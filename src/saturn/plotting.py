"""Diagnostic figures: domain maps and the 5-column function figure.

Function figure contract (one row per feature):
    rho (isodepth) | domains | RAW | FITTED | fitted function
The function panel shows binned observed values as points and the per-domain fit as lines, coloured
by domain. Keep all five columns: the FITTED map is what shows whether the function reproduces the
spatial pattern. These are review figures; paper figures go through the house style (Arial >= 5 pt,
PNG 600 dpi + SVG, domain colours locked by LABEL, not by index).
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from . import piecewise as pw
from .theme import CATEGORICAL, ANCHOR, set_theme  # noqa: F401  (registers the saturn_* colormaps)


def _domain_color(k, colors=None):
    if colors is not None:
        return colors[k] if isinstance(colors, dict) else colors[int(k) % len(colors)]
    return CATEGORICAL[int(k) % len(CATEGORICAL)]


def spot_size(ax, xy, fill=1.2):
    """Marker area (pt^2) and shape that tile the spot grid at the current axes size, so no white lines show
    between spots. Square markers on an axis-aligned grid (DBiT, binned MERFISH), round ones otherwise (Visium)."""
    xy = np.asarray(xy, float)
    sub = xy[np.random.default_rng(0).choice(len(xy), min(len(xy), 3000), replace=False)]
    dist, j = cKDTree(xy).query(sub, k=2)
    pitch = np.median(dist[:, 1])
    v = xy[j[:, 1]] - sub
    on_axis = np.mean(np.minimum(np.abs(v[:, 0]), np.abs(v[:, 1])) < 0.05 * pitch) > 0.8
    fig = ax.figure; pos = ax.get_position()
    w, h = pos.width * fig.get_figwidth() * 72, pos.height * fig.get_figheight() * 72
    span = np.ptp(xy, 0) + pitch
    side = pitch * min(w / span[0], h / span[1]) * fill
    return (side * (1.0 if on_axis else 1.2)) ** 2, ("s" if on_axis else "o")


def spatial(ax, xy, val, title="", cmap="saturn_div", contour_d=None, bnds=(), discrete=False, s=None, colors=None,
            vmin=None, vmax=None, center=None, legend=False):
    """Map one value per spot. Continuous values use cmap with 1-99 percentile limits (or vmin/vmax; center=0 makes
    them symmetric). discrete=True colours integer labels with the theme's categorical colours, or with ``colors``
    (a list, or a dict label -> colour). s=None sizes the markers to tile the spot grid."""
    xy = np.asarray(xy, float)
    if s is None:
        s, marker = spot_size(ax, xy)
    else:
        marker = "o"
    if discrete:
        val = np.asarray(val)
        for k in np.unique(val):
            m = val == k
            ax.scatter(xy[m, 0], xy[m, 1], s=s, marker=marker, color=_domain_color(k, colors), label=str(k),
                       rasterized=True, lw=0)
        if legend:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), markerscale=4 / max(np.sqrt(s), 1e-3), handletextpad=0.2)
        sc = None
    else:
        val = np.asarray(val, float)
        lo, hi = np.nanpercentile(val, [1, 99])
        lo, hi = (lo if vmin is None else vmin), (hi if vmax is None else vmax)
        if center is not None:
            r = max(abs(lo - center), abs(hi - center)); lo, hi = center - r, center + r
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=val, s=s, marker=marker, cmap=cmap, vmin=lo, vmax=hi, rasterized=True, lw=0)
    if contour_d is not None and len(bnds):
        try:
            ax.tricontour(xy[:, 0], xy[:, 1], contour_d, levels=sorted(bnds), colors=ANCHOR["ink"], linewidths=0.6)
        except Exception:
            pass
    ax.set_title(title, fontsize=7); ax.set_aspect("equal"); ax.axis("off")
    return sc


def function_row(axes, xy, d, lab, y, slope_j, inter_j, name, bnds=(), lib=None, link="identity", n_bins=25,
                 unit="value"):
    """Draw one feature into five axes. y = raw values (counts when lib is given)."""
    raw = y / np.maximum(lib, 1) if lib is not None else y
    fitted = pw.predict(slope_j[None], inter_j[None], lab, d, link)[:, 0]
    r2, db, ob, pred, lb = pw.binned_r2(y, d, lab, slope_j, inter_j, n_bins, lib=lib, link=link)
    spatial(axes[0], xy, d, "rho (isodepth)", contour_d=d, bnds=bnds)
    spatial(axes[1], xy, lab, "domains", contour_d=d, bnds=bnds, discrete=True)
    spatial(axes[2], xy, raw, f"{name}\nRAW", cmap="saturn_seq_r", contour_d=d, bnds=bnds)
    spatial(axes[3], xy, fitted, f"{name}\nFITTED", cmap="saturn_seq_r", contour_d=d, bnds=bnds)
    a = axes[4]
    for k in range(int(lab.max()) + 1):
        m = lb == k
        a.plot(db[m], ob[m], "o", ms=2.5, color=_domain_color(k))
        if m.any():
            xs = np.linspace(db[m].min(), db[m].max(), 20)
            eta = inter_j[k] + slope_j[k] * xs
            a.plot(xs, np.exp(eta) if link == "log" else (1 / (1 + np.exp(-eta)) if link == "logit" else eta),
                   color=_domain_color(k), lw=1.4)
    a.set_ylim(ob.min() - 0.05 * np.ptp(ob), ob.max() + 0.05 * np.ptp(ob))   # lines extrapolate: clip to data
    a.set_xlabel("isodepth"); a.set_ylabel(unit); a.set_title(f"{name}: sloped in {pw.n_sloped(slope_j[None])[0]} "
                                                             f"domains, R2 {r2:.2f}", fontsize=7)
    a.spines[["top", "right"]].set_visible(False)
    return r2


def function_figure(xy, d, lab, Ymat, slope, inter, names, idx, bnds=(), lib=None, link="identity", unit="value"):
    """rows = features idx (columns of Ymat; slope/inter rows aligned to Ymat columns)."""
    fig, AX = plt.subplots(len(idx), 5, figsize=(14, 2.6 * len(idx)), squeeze=False)
    for r, j in enumerate(idx):
        y = np.asarray(Ymat[:, j].todense()).ravel() if hasattr(Ymat, "todense") else np.asarray(Ymat)[:, j]
        function_row(AX[r], xy, d, lab, y, slope[j], inter[j], names[j], bnds, lib, link, unit=unit)
    fig.tight_layout()
    return fig
