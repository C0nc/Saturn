"""Supplementary Figures 2 and 3 -- ground truth and every method's recovered RNA and ATAC axes, every condition (seed 0).

  S2  easy grid (12 conditions), S3  hard grid (8 conditions); the 6 methods of Fig 1, equal square maps, one shared colour range.
  Under each map: recovery / leak, read from the benchmark tables, never recomputed. Axis sign aligned to the truth for display only.
Output: fig1/supp/S2/figS2.{png,svg,json}, fig1/supp/S3/figS3.{png,svg,json}
Usage: python published_figures/fig1/supp23_maps.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "<workspace>/Saturn/published_figures/_style")
sys.path.insert(0, "<workspace>/Saturn/published_figures/fig1")
from paper_style import apply, METHOD, MODALITY
from a4 import save_a4
import supp_fig1_benchmark as B
V = B.V
from fig1_model_benchmark import RES
ROOT = Path("<workspace>/Saturn/published_figures/fig1/supp")
ORDER = [m for m in B.ORDER if m != "GASTON (global-scaled)"]
COLS = ["Ground truth"] + ORDER
W, TILE, PITCH_X, LAB = 7.2, 0.43, 0.443, 0.42
ROW = TILE + 0.1                     # one map + its recovery / leak line
TOP = 0.2                            # method header
GAP_COND = 0.07


def run_key(r):
    a_, c_, f_ = r.split("_")[1:]
    return int(a_[1:]), float(c_[1:]), float(f_)


def group(fig, FH, x0, y0, run, Tg, header, letter):
    """two rows (RNA, ATAC) of one condition; x0, y0 = top-left in inches"""
    z = np.load(RES / "axes" / f"{run}_s0.npz", allow_pickle=True); t = np.asarray(z["true"], float)
    ang, cra, cf = run_key(run)
    R = Tg[(Tg.angle == ang) & (Tg.cross_RA == cra) & (Tg.coupled_frac == cf) & (Tg.seed == 0)].set_index("method")
    im = None
    for k, (lab, col) in enumerate((("RNA", MODALITY["RNA"]), ("ATAC", MODALITY["ATAC"]))):
        y = y0 + k * ROW
        fig.text((x0 + LAB - 0.05) / W, 1 - (y + TILE / 2) / FH, lab, rotation=90, ha="center", va="center", fontsize=5.5, color=col)
        for c, m in enumerate(COLS):
            a = fig.add_axes([(x0 + LAB + c * PITCH_X) / W, 1 - (y + TILE) / FH, TILE / W, TILE / FH])
            if m == "Ground truth":
                im = V.tile(a, t[:, k], "#1a1a1a", 0.9)
            else:
                v = np.asarray(z[V.KEY.get(m, m)], float)[:, k]; v = v if np.corrcoef(v, t[:, k])[0, 1] >= 0 else -v
                im = V.tile(a, v, METHOD["Saturn"] if m == "Saturn" else "#9aa2aa", 0.9 if m == "Saturn" else 0.4)
                rec, lk = float(R.loc[m, "rec_R" if k == 0 else "rec_A"]), float(R.loc[m, "leak_R" if k == 0 else "leak_A"])
                a.text(0.5, -0.03, f"{rec:.2f} / {lk:.2f}", transform=a.transAxes, ha="center", va="top", fontsize=5,
                       color=METHOD["Saturn"] if m == "Saturn" else "#333333")
            if header and k == 0:
                a.set_title("truth" if m == "Ground truth" else m, fontsize=5.2, pad=2,
                            color="#1a1a1a" if m == "Ground truth" else V.ink(METHOD[m]))
    car = Tg[Tg.cross_RA == cra].cross_AR.iloc[0]
    s = f"{ang}°, {cra:g}/{car:g}" + ("" if run.startswith("easy") else f", {int(cf * 100)}%")
    fig.text((x0 + 0.12) / W, 1 - (y0 + ROW - 0.05) / FH, s, rotation=90, ha="center", va="center", fontsize=5.5)
    fig.text((x0 + 0.04) / W, 1 - (y0 - 0.02) / FH, letter, fontsize=9, fontweight="bold", va="top")
    return im


def page(T, grid, tag):
    runs = sorted({p.stem[:-3] for p in (RES / "axes").glob(f"{grid}_*_s0.npz")}, key=run_key)
    Tg = T[T.grid == grid]; n = int(np.ceil(len(runs) / 2)); step = 2 * ROW + GAP_COND
    FH = TOP + n * step + 0.46
    fig = plt.figure(figsize=(W, FH)); im = None
    for i, r in enumerate(runs):
        col, row = i // n, i % n
        im = group(fig, FH, col * (LAB + 7 * PITCH_X + 0.13), TOP + row * step, r, Tg, row == 0, chr(97 + i))
    ybar = TOP + n * step + 0.05
    cax = fig.add_axes([(LAB) / W, 1 - (ybar + 0.07) / FH, 1.0 / W, 0.07 / FH])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-2, 0, 2]); cb.outline.set_linewidth(0.3)
    cb.ax.tick_params(labelsize=5, length=1.5, width=0.4, pad=1)
    fig.text((LAB + 1.08) / W, 1 - (ybar + 0.035) / FH, "isodepth (z)", fontsize=5.3, va="center")
    fig.text((LAB + 1.75) / W, 1 - (ybar + 0.035) / FH,
             "Under each map: axis recovery (|Spearman|) / cross-axis leak. Seed 0.", fontsize=5.3, va="center", color="#444444")
    fig.text(LAB / W, 1 - (ybar + 0.25) / FH, "Row labels: angle between the two true axes, cross-coupling RNA←ATAC / ATAC←RNA"
             + ("." if grid == "easy" else ", coupled features."), fontsize=5.3, va="center", color="#444444")
    save_a4(fig, ROOT / tag / f"fig{tag}"); plt.close(fig); print("->", ROOT / tag, f"{FH:.2f} in", flush=True)


def main():
    apply(base_pt=6.5); T = B.load(); T = T[T.method.isin(ORDER)]
    page(T, "easy", "S2"); page(T, "hard", "S3")


if __name__ == "__main__":
    main()
