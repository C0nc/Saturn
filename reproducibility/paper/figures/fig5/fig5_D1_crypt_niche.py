"""Fig 5 (draft): along Saturn's rho_RNA in tonsil section D1 -- crypt surface -> lymphoid interior -- the held-out
protein-complex layer re-organises: a CD9 tetraspanin web at the crypt lymphoepithelium.

  a  D1 tissue coloured by t_RNA (rank of rho_RNA, crypt epithelium = 0); crypt-epithelium spots ringed
  b  what the axis is: tonsil programmes (depth-regressed module scores) per t_RNA octile
  c  method-free geometry: CD9-web specificity by distance to the crypt epithelium, D1 (solid) and B1 (dashed),
     with two interior-rising complexes for contrast
  d  replication: slope along t_RNA, D1 vs B1, every complex; D1 rotation-null hits coloured
  e  isotype-corrected complex specificity per t_RNA octile, the 12 steepest surface and 12 steepest interior
     complexes among those passing the rotation null in D1 (row-centred)
Inputs: results/real_saturn/sproxseq_crypt_niche/ (scripts/sproxseq_D1_crypt_niche.py), sproxseq_D1_explain/spots.csv
-> fig5_D1_crypt_niche.{png,svg,json} + panels/fig5D1{a..e}_*.svg
Usage: python published_figures/fig5/fig5_D1_crypt_niche.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
sys.path.insert(0, "<workspace>/Saturn/published_figures/_style")
sys.path.insert(0, "<workspace>/Hole/gaston_mix_mm/scripts")
from paper_style import apply, save_dual, save_panel_svg, clean_ax, CMAP, ANCHOR
from sproxseq_ccc import spots

R = Path("<workspace>/Hole/gaston_mix_mm/results/real_saturn"); N = R / "sproxseq_crypt_niche"
OUT = Path("<workspace>/Saturn/published_figures/fig5")
SURF, INT = ANCHOR["rust"], ANCHOR["navy"]
PROG = {"epithelium": ANCHOR["ochre"], "CCL20 (epithelial chemokine)": ANCHOR["rust"], "myeloid": ANCHOR["rose"],
        "IgA plasma": ANCHOR["plum"], "CCR6 (memory B)": ANCHOR["sage"], "naive mantle": ANCHOR["sand"],
        "T zone": ANCHOR["teal"], "GC": ANCHOR["night"]}
PLAB = {"CCL20 (epithelial chemokine)": "CCL20", "CCR6 (memory B)": "CCR6 memory B"}
WEB = ["CD9:HLADPRQ", "HLADPRQ:CD9", "CD9:ICAM1", "CD9:CD32", "CD147:CD9", "CD9:CD19"]
CONTRAST = ["CD20:CD20", "HLADPRQ:HLADPRQ"]


def nice(c):
    return c.replace("HLADPRQ", "MHC-II").replace("CD279", "PD-1").replace("PDL2", "PD-L2")


def load():
    sp = spots("D1"); T = pd.read_csv(N / "complexes.csv", index_col=0)
    return dict(sp=sp, T=T, prof=pd.read_csv(N / "profiles_D1.csv", index_col=0),
                mod=pd.read_csv(N / "modules_D1.csv", index_col=0),
                geo={s: pd.read_csv(N / f"geometry_{s}.csv", index_col=0) for s in ("D1", "B1")})


def pa(ax, D):
    sp = D["sp"]; epi = sp.compartment.values == "epithelium"
    sc = ax.scatter(sp.x, sp.y, c=sp.t_RNA, cmap=CMAP["pseudotime"], s=2.2, lw=0)
    ax.scatter(sp.x[epi], sp.y[epi], s=3.2, facecolor="none", edgecolor=SURF, lw=0.35)
    clean_ax(ax, spatial=True); ax.invert_yaxis()
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.01, ticks=[0, 1]); cb.ax.set_yticklabels(["crypt", "interior"])
    cb.ax.tick_params(labelsize=5.5); ax.set_title("D1: Saturn ρ$_{\\mathrm{RNA}}$ (rank)", fontsize=6.5)
    ax.text(0.01, 0.01, "ringed: crypt epithelium", transform=ax.transAxes, fontsize=5.5, color=SURF)


def pb(ax, D):
    M = D["mod"]; x = np.arange(len(M))
    for k, c in PROG.items():
        ax.plot(x, M[k], color=c, lw=1.0, marker="o", ms=2, label=PLAB.get(k, k))
    ax.axhline(0, color="#999999", lw=0.4)
    ax.set_xticks([0, len(M) - 1]); ax.set_xticklabels(["crypt", "interior"]); ax.set_xlim(-0.3, len(M) - 0.7)
    ax.legend(fontsize=5.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), handlelength=1.2)
    ax.set_xlabel("ρ$_{\\mathrm{RNA}}$ octile", fontsize=6); ax.set_ylabel("module score (z)", fontsize=6); clean_ax(ax)


def pc(ax, D, cax):
    T, P = D["T"], D["prof"]
    sig = T[T.p_D1 < 0.05]
    rows = list(sig.sort_values("slope_D1").index[:12]) + list(sig.sort_values("slope_D1").index[::-1][:12][::-1])
    M = P[rows].T.values; M = M - M.mean(1, keepdims=True)
    im = ax.imshow(M, cmap=CMAP["score"], vmin=-0.5, vmax=0.5, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([nice(r) for r in rows], fontsize=5.5)
    for tl, r in zip(ax.get_yticklabels(), rows): tl.set_color(SURF if T.loc[r, "slope_D1"] < 0 else INT)
    ax.set_xticks([0, M.shape[1] - 1]); ax.set_xticklabels(["crypt", "interior"]); ax.set_xlabel("ρ$_{\\mathrm{RNA}}$ octile", fontsize=6)
    ax.axhline(11.5, color="white", lw=1.2)
    cb = plt.colorbar(im, cax=cax, ticks=[-0.5, 0, 0.5]); cb.ax.tick_params(labelsize=5.5)
    cb.set_label("specificity, log2 O/E\nminus isotype (row-centred)", fontsize=5.5)


def pd_(ax, D):
    x = np.arange(5); lab = ["epith.", "<2", "2-4", "4-7", ">7"]
    for s, ls in (("D1", "-"), ("B1", "--")):
        G = D["geo"][s]
        for c in WEB: ax.plot(x, G[c].values, ls=ls, color=SURF, lw=0.8, alpha=0.85)
        for c in CONTRAST: ax.plot(x, G[c].values - G[c].values.mean() + 0.6, ls=ls, color=INT, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_xlabel("distance to crypt epithelium (spots)", fontsize=6)
    ax.set_ylabel("specificity (log2, isotype-corrected)", fontsize=6)
    ax.text(0.98, 0.97, "CD9 web: CD9 with MHC-II, ICAM-1,\nFcγRIIB, CD147, CD19", transform=ax.transAxes, ha="right",
            va="top", fontsize=5.5, color=SURF)
    ax.text(0.98, 0.06, "CD20:CD20, MHC-II:MHC-II (centred)", transform=ax.transAxes, ha="right", fontsize=5.5, color=INT)
    ax.plot([], [], "k-", lw=0.8, label="D1"); ax.plot([], [], "k--", lw=0.8, label="B1 (replication)")
    ax.legend(fontsize=5.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.8)); clean_ax(ax)


def pe(ax, D):
    T = D["T"]; s = T.p_D1 < 0.05
    ax.scatter(T.slope_D1[~s], T.slope_B1[~s], s=3, lw=0, color=ANCHOR["silver"])
    ax.scatter(T.slope_D1[s], T.slope_B1[s], s=4, lw=0, color=np.where(T.slope_D1[s] < 0, SURF, INT))
    ax.axhline(0, color="#999999", lw=0.4); ax.axvline(0, color="#999999", lw=0.4)
    same = np.mean(np.sign(T.slope_D1[s]) == np.sign(T.slope_B1[s]))
    ax.text(0.03, 0.97, f"D1 hits (n={int(s.sum())}): {same:.0%} same\ndirection in B1, rotation p = 0.05",
            transform=ax.transAxes, va="top", fontsize=5.5)
    ax.set_xlabel("slope along ρ$_{\\mathrm{RNA}}$, D1", fontsize=6); ax.set_ylabel("slope along ρ$_{\\mathrm{RNA}}$, B1", fontsize=6); clean_ax(ax)


PANELS = [("a", "tissue_rho_RNA", pa, (2.3, 2.0)), ("b", "axis_programmes", pb, (3.0, 2.0)),
          ("c", "CD9_web_vs_crypt_distance", pd_, (2.4, 2.2)), ("d", "D1_B1_replication", pe, (2.2, 2.2)),
          ("e", "complex_specificity_heatmap", pc, (2.4, 2.9))]


def main():
    apply(base_pt=6.5); D = load()
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 0.9, 0.04], height_ratios=[1, 1.15], wspace=0.55, hspace=0.45,
                  left=0.04, right=0.96, top=0.96, bottom=0.08)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
           fig.add_subplot(gs[1, 2])]
    for (k, _, f, _), ax in zip(PANELS, axs):
        f(ax, D, fig.add_subplot(gs[1, 3])) if k == "e" else f(ax, D)
        ax.text(-0.12, 1.06, k, transform=ax.transAxes, fontsize=9, fontweight="bold")
    save_dual(fig, OUT / "fig5_D1_crypt_niche"); plt.close(fig)
    for k, nm, f, size in PANELS:
        fig = plt.figure(figsize=size)
        if k == "e":
            g = GridSpec(1, 2, figure=fig, width_ratios=[1, 0.06], left=0.38, right=0.82, top=0.95, bottom=0.12)
            f(fig.add_subplot(g[0]), D, fig.add_subplot(g[1]))
        else:
            ax = fig.add_axes([0.2, 0.18, 0.55 if k == "b" else 0.72, 0.75]); f(ax, D)
        print("->", save_panel_svg(fig, OUT / "panels" / f"fig5D1{k}_{nm}")); plt.close(fig)


if __name__ == "__main__":
    main()
