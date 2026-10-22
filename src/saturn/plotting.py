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

from . import piecewise as pw


def spatial(ax, xy, val, title="", cmap="coolwarm", contour_d=None, bnds=(), discrete=False, s=4):
    xy = np.asarray(xy)
    if discrete:
        val = np.asarray(val).astype(int)
        for k in np.unique(val):
            m = val == k; ax.scatter(xy[m, 0], xy[m, 1], s=s, color=f"C{k}", label=f"D{k}", rasterized=True, lw=0)
    else:
        lo, hi = np.nanpercentile(val, [1, 99])
        ax.scatter(xy[:, 0], xy[:, 1], c=val, s=s, cmap=cmap, vmin=lo, vmax=hi, rasterized=True, lw=0)
    if contour_d is not None and len(bnds):
        try:
            ax.tricontour(xy[:, 0], xy[:, 1], contour_d, levels=sorted(bnds), colors="k", linewidths=0.6)
        except Exception:
            pass
    ax.set_title(title, fontsize=7); ax.set_aspect("equal"); ax.axis("off")


def function_row(axes, xy, d, lab, y, slope_j, inter_j, name, bnds=(), lib=None, link="identity", n_bins=25,
                 unit="value"):
    """Draw one feature into five axes. y = raw values (counts when lib is given)."""
    raw = y / np.maximum(lib, 1) if lib is not None else y
    fitted = pw.predict(slope_j[None], inter_j[None], lab, d, link)[:, 0]
    r2, db, ob, pred, lb = pw.binned_r2(y, d, lab, slope_j, inter_j, n_bins, lib=lib, link=link)
    spatial(axes[0], xy, d, "rho (isodepth)", contour_d=d, bnds=bnds)
    spatial(axes[1], xy, lab, "domains", contour_d=d, bnds=bnds, discrete=True)
    spatial(axes[2], xy, raw, f"{name}\nRAW", contour_d=d, bnds=bnds)
    spatial(axes[3], xy, fitted, f"{name}\nFITTED", contour_d=d, bnds=bnds)
    a = axes[4]
    for k in range(int(lab.max()) + 1):
        m = lb == k
        a.plot(db[m], ob[m], "o", ms=2.5, color=f"C{k}")
        if m.any():
            xs = np.linspace(db[m].min(), db[m].max(), 20)
            eta = inter_j[k] + slope_j[k] * xs
            a.plot(xs, np.exp(eta) if link == "log" else (1 / (1 + np.exp(-eta)) if link == "logit" else eta),
                   color=f"C{k}", lw=1.4)
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
