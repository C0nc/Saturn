"""Saturn logo: a planet whose bands are isodepth contours, with one ring. Writes logo.{png,svg} and logo_mark.{png,svg} here."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Arc

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"], "svg.fonttype": "none"})
NAVY, MIST, RUST, OCHRE, SLATE = "#2B3F6B", "#E6EBEA", "#9A5A55", "#C49A5A", "#6A7595"
DIV = LinearSegmentedColormap.from_list("saturn_div", [NAVY, "#5F9AA3", MIST, "#B98A8E", RUST])
TILT = 22                      # ring tilt, degrees
R, RX, RY = 1.0, 1.95, 0.52    # planet radius and ring half-axes


def mark(ax):
    """the planet, the back half of the ring behind it, the front half over it"""
    ring = dict(xy=(0, 0), width=2 * RX, height=2 * RY, angle=TILT, color=OCHRE, lw=6.5, capstyle="round")
    ax.add_patch(Arc(theta1=0, theta2=180, zorder=1, **ring))
    g = np.linspace(-1.05, 1.05, 700); X, Y = np.meshgrid(g, g)
    t = np.deg2rad(TILT); u = -X * np.sin(t) + Y * np.cos(t) + 0.18 * np.sin(2.2 * (X * np.cos(t) + Y * np.sin(t)))
    im = ax.imshow(np.round(u * 3.5) / 3.5, extent=(-1.05, 1.05, -1.05, 1.05), origin="lower", cmap=DIV.reversed(), vmin=-1.05, vmax=1.05,
                   interpolation="bilinear", zorder=2)
    disc = Circle((0, 0), R, transform=ax.transData, fc="none", ec="none", zorder=2); ax.add_patch(disc); im.set_clip_path(disc)
    cs = ax.contour(X, Y, u, levels=(np.arange(-3, 4) + 0.5) / 3.5, colors="white", linewidths=1.5, linestyles="solid", zorder=3)
    cs.set_clip_path(disc)
    ax.add_patch(Arc(theta1=180, theta2=360, zorder=4, **ring))
    ax.set_aspect("equal"); ax.axis("off")


def save(fig, stem):
    for ext in ("png", "svg"):
        fig.savefig(f"{HERE}/{stem}.{ext}", dpi=600 if ext == "png" else None, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


fig, ax = plt.subplots(figsize=(2.4, 1.9)); mark(ax); ax.set_xlim(-2.2, 2.2); ax.set_ylim(-1.45, 1.45); save(fig, "logo_mark")
fig = plt.figure(figsize=(2.4, 2.3)); ax = fig.add_axes([0, 0.17, 1, 0.83]); mark(ax); ax.set_xlim(-2.2, 2.2); ax.set_ylim(-1.45, 1.45)
fig.text(0.5, 0.1, "Saturn", ha="center", va="center", fontsize=25, fontweight="bold", color=SLATE); save(fig, "logo")
