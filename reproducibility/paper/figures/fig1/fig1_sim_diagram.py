"""Figure 1 panel: how the v5 benchmark data are generated (simulate_v5, the generative model Saturn assumes).

Every tile and curve is drawn from the simulator itself, run on the representative condition PICK
(hard_a45_c0.8_0.6_s0), so the diagram shows the actual data behind panels c-e, not a cartoon.

  1 two isodepths      curved ramps d_RNA, d_ATAC on a 30 x 30 jittered grid; gradients theta apart
  2 domain curves      per-feature piecewise-linear curves of ONE axis, breakpoints shared by the axis
                       (4 domains; 30% jumps, 30% flat slopes)
  3 additive log-rate  RNA gene  eta = a + s(d_RNA) + c(d_ATAC);  ATAC peak  eta = b + t(d_ATAC) + e(d_RNA)
                       own amplitude 1, cross amplitude cross_RA / cross_AR, on a coupled fraction of features
  4 counts             RNA negative binomial (theta 10), ATAC Poisson
  5 targets            RNA GLM-PCs (10), ATAC LSI (15, component 1 dropped) -> the input of every method
Usage: imported by fig1_model_benchmark.py (sim_panel)
"""
from __future__ import annotations
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, "<workspace>/Saturn/published_figures/_style")
sys.path.insert(0, "<workspace>/saturn_repo/src")       # Saturn package (saturn_repo)
from saturn.sim import SimConfig, simulate                          # noqa: E402
from paper_style import MODALITY, CMAP                             # noqa: E402

C_RNA, C_ATAC = MODALITY["RNA"], MODALITY["ATAC"]
CFG = dict(axis_angle_deg=45, cross_RA=0.8, cross_AR=0.2, coupled_frac=0.6, rna_libsize=1000.0, seed=0)
GRID = ("hard grid: θ $\\in$ {45°, 90°} × cross$_{\\mathrm{RNA←ATAC}}$ $\\in$ {0.8, 0.95} × coupled features $\\in$ {60, 90%} "
        "× 3 seeds = 24 runs   ·   cross$_{\\mathrm{ATAC←RNA}}$ = 0.2, RNA depth 1,000, 500 genes, 1,000 peaks")


def _z(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def _tile(ax, v, n, cmap=None, edge="#9aa2aa", title=None, tc="#333333"):
    ax.imshow(_z(v).reshape(n, n), cmap=cmap or CMAP["score"], origin="lower", vmin=-2, vmax=2,
              interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_edgecolor(edge); s.set_linewidth(0.6)
    if title:
        ax.set_title(title, fontsize=6, color=tc, pad=2)


def _pick(F, coupled=None):
    """the feature with the largest between-domain jump, so the step is visible (coupled only if a mask is given)"""
    idx = np.arange(F.shape[1]) if coupled is None else np.where(coupled)[0]
    return idx[np.argmax(np.abs(np.diff(np.sort(F[:, idx], 0), axis=0)).max(0))]


def _curves(ax, d, F, bps, col, xlabel, k=3):
    o = np.argsort(d); rng = np.random.default_rng(1)
    for i, g in enumerate(rng.choice(F.shape[1], k, replace=False)):
        ax.plot(d[o], F[o, g] + 2.6 * (i - 1), color=col, lw=0.9, alpha=1 - 0.25 * i)
    for b in bps:
        ax.axvline(b, color="#9aa2aa", lw=0.5, ls="--")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel, fontsize=6, color=col, labelpad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _op(fig, ax, s):
    ax.set_axis_off(); ax.text(0.5, 0.5, s, fontsize=8, ha="center", va="center", transform=ax.transAxes)


def _arrow(fig, a0, a1):
    p0, p1 = a0.get_position(), a1.get_position()
    y = (p0.y0 + p0.y1) / 2
    fig.patches.append(FancyArrowPatch((p0.x1 + 0.004, y), (p1.x0 - 0.004, y), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=6, lw=0.8, color="#333333"))


def sim_panel(fig, spec):
    d = simulate(SimConfig(**CFG)); n = d.config.n_side
    dR, dA = d.d[:, 0], d.d[:, 1]; P = d.parts
    g = _pick(P["c"], d.coupled["R"]); p = _pick(P["e"], d.coupled["A"])
    # columns: iso | curves | s + c = eta | counts | targets  (narrow gaps hold arrows / operators)
    W = [1, 0.32, 1.5, 0.32, 1, 0.22, 1, 0.22, 1, 0.32, 1, 0.32, 1]
    gs = GridSpecFromSubplotSpec(3, len(W), subplot_spec=spec, width_ratios=W, height_ratios=[1, 1, 0.16],
                                 wspace=0.05, hspace=0.42)
    ax = {}
    for r, (m, col) in enumerate((("RNA", C_RNA), ("ATAC", C_ATAC))):
        own, crs, eta_own, eta_crs = ("s", "c", dR, dA) if m == "RNA" else ("t", "e", dA, dR)
        f = g if m == "RNA" else p
        ax[r, 0] = fig.add_subplot(gs[r, 0]); _tile(ax[r, 0], d.d[:, r], n, edge=col)
        ax[r, 0].set_ylabel(f"d$_{{\\mathrm{{{m}}}}}$", fontsize=6.5, color=col, labelpad=1)
        ax[r, 2] = fig.add_subplot(gs[r, 2])
        _curves(ax[r, 2], d.d[:, r], P[own], d.breakpoints["R" if r == 0 else "A"], col, f"d$_{{\\mathrm{{{m}}}}}$")
        oth = "ATAC" if m == "RNA" else "RNA"
        ax[r, 4] = fig.add_subplot(gs[r, 4]); _tile(ax[r, 4], P[own][:, f], n, edge=col)
        ax[r, 6] = fig.add_subplot(gs[r, 6]); _tile(ax[r, 6], P[crs][:, f], n, edge=MODALITY[oth])
        ax[r, 8] = fig.add_subplot(gs[r, 8]); _tile(ax[r, 8], P[own][:, f] + P[crs][:, f], n, edge="#1a1a1a")
        _op(fig, fig.add_subplot(gs[r, 5]), "+"); _op(fig, fig.add_subplot(gs[r, 7]), "=")
        cnt = d.counts_R[:, f] if m == "RNA" else d.counts_A[:, f]
        lib = (d.counts_R if m == "RNA" else d.counts_A).sum(1)
        ax[r, 10] = fig.add_subplot(gs[r, 10]); _tile(ax[r, 10], np.log1p(1e3 * cnt / lib), n, cmap=CMAP["pseudotime"], edge=col)
        Y = d.Y_R if m == "RNA" else d.Y_A
        k = int(np.argmax([abs(np.corrcoef(Y[:, j], d.d[:, r])[0, 1]) for j in range(Y.shape[1])]))
        ax[r, 12] = fig.add_subplot(gs[r, 12]); _tile(ax[r, 12], Y[:, k], n, edge=col)
        lab = (f"own s(d$_{{\\mathrm{{RNA}}}}$)", f"cross c(d$_{{\\mathrm{{ATAC}}}}$)", "log-rate η (gene)") if m == "RNA" \
            else (f"own t(d$_{{\\mathrm{{ATAC}}}}$)", f"cross e(d$_{{\\mathrm{{RNA}}}}$)", "log-rate η (peak)")
        for c_, t_ in zip((4, 6, 8), lab):
            ax[r, c_].set_xlabel(t_, fontsize=5.5, labelpad=1.5, color="#333333")
        ax[r, 10].set_xlabel("NB counts (θ 10)" if m == "RNA" else "Poisson counts", fontsize=5.5, labelpad=1.5)
        ax[r, 12].set_xlabel(f"GLM-PC{k + 1}" if m == "RNA" else f"LSI {k + 2}", fontsize=5.5, labelpad=1.5)
    fig.canvas.draw_idle()
    for r in (0, 1):
        for a0, a1 in ((0, 2), (2, 4), (8, 10), (10, 12)):
            _arrow(fig, ax[r, a0], ax[r, a1])
    heads = {0: "1  two isodepths\n(θ = 45° apart)", 2: "2  piecewise-linear curves\n(4 domains / axis, jumps)",
             6: "3  additive log-rate: own + cross\n(cross 0.8 / 0.2 · coupled 60%)", 10: "4  counts",
             12: "5  targets\n(input to all methods)"}
    for c_, t_ in heads.items():
        ax[0, c_].set_title(t_, fontsize=6, color="#1a1a1a", pad=3)
    foot = fig.add_subplot(gs[2, :]); foot.set_axis_off()
    foot.text(0.5, 0.0, GRID, fontsize=5.5, ha="center", va="bottom", color="#333333", transform=foot.transAxes)
    return ax
