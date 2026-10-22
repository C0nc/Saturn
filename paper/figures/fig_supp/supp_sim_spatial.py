"""Figure 1 (spatial companion) — what the simulation benchmark actually is.

  (a) Ground-truth isodepth fields: 2 modalities x 3 sim regimes
      (smooth / domains / cascade). This is the signal every method must recover.
  (b) The noise ladder: first PC of the log-normalised observed counts under
      increasing corruption. PC1 is what a linear method has to work with, so
      this panel shows the signal actually degrading (total counts would not —
      the isodepth lives in gene-level structure, not in depth).
  (c) Recovered isodepth in tissue space on a deliberately discriminating
      config (dropout-hi), annotated with recovery |rho|.

Reruns the simulator and the recovery methods live (seed-controlled), so the
maps correspond exactly to the quantitative panels in fig1_benchmark.py.
"""
from __future__ import annotations
import os, sys, warnings, contextlib, io
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr
from sklearn.decomposition import PCA as skPCA

HOLE = Path("/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm")
sys.path.insert(0, str(HOLE / "src"))
sys.path.insert(0, str(HOLE / "sim"))
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/_style")

from simulate_v2 import SimConfigV2, NoiseSpec, simulate_v2      # noqa
import baselines as BASE                                          # noqa
from baselines_ext import recover_gaston_single_per_mod           # noqa
from gastonmix_mm.model import GastonMixMM                        # noqa
from gastonmix_mm.train import train_model, build_knn, get_isodepth  # noqa
from paper_style import apply, save_dual, METHOD, CMAP            # noqa

OUT = Path("/mnt/HDD2B/Users/yancui/Saturn/published_figures/fig_supp/supp_sim_spatial")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CFG = dict(iso_hidden=[128, 128], dec_hidden=[128], enc_dim=16, sigma=3.0)
TRAIN_CFG = dict(n_epochs=1800, lr=2e-3, smooth_weight=2.0, coupling_weight=0.0,
                  self_recon_weight=3.0, var_floor_weight=1.0, n_restarts=3, verbose=0)

REGIMES = ["smooth", "domains", "cascade"]
NOISE_LADDER = [
    ("clean",      NoiseSpec(libsize=2000)),
    ("depth low",  NoiseSpec(libsize=500)),
    ("compound",   NoiseSpec(libsize=500, dropout_p=0.3,
                               lane_sigma=0.3, ambient_alpha=0.1)),
    ("dropout hi", NoiseSpec(libsize=2000, dropout_p=0.6)),
]
# Panel (c) config. dropout-hi is where the benchmark shows the widest spread,
# so it discriminates. Among the 6 (regime x seed) draws at dropout-hi, this one
# was selected as REPRESENTATIVE — chosen to minimise total deviation from each
# method's own benchmark marginal mean at dropout-hi (Saturn .90, GASTON .81,
# PCA .51), not to favour any method. Realised here: .909 / .781 / .447.
SHOWCASE = dict(regime="cascade", noise=NOISE_LADDER[3][1], noise_name="dropout hi",
                  M=2, corr=0.4, seed=1)
N_SIDE, GENES = 30, 60


def seed_all(seed: int):
    """Must run BEFORE GastonMixMM(...) — the positional-encoding B matrix is
    drawn in __init__, so seeding afterwards cannot reproduce a fit. Without
    this the panel silently inherits whatever RNG state earlier panels left
    behind, which can land the fit in a folded basin."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_saturn(d, seed=0):
    seed_all(seed)
    coords = torch.tensor(d.coords)
    targets = [torch.tensor(c) for c in d.counts]
    libs = [torch.tensor(c.sum(1, keepdims=True) / c.sum(1).mean().clip(min=1))
            for c in d.counts]
    knn = build_knn(d.coords, k=6)
    model = GastonMixMM([c.shape[1] for c in d.counts], likelihood="poisson", **MODEL_CFG)
    model, _ = train_model(model, coords, targets, libsize=libs, knn_idx=knn,
                            device=DEV, **TRAIN_CFG)
    return get_isodepth(model, coords, device=DEV)


def best_factor(factors_m, rho_true_m):
    F = np.asarray(factors_m)
    if F.ndim == 1:
        F = F[:, None]
    scores = [abs(spearmanr(F[:, j], rho_true_m).correlation) for j in range(F.shape[1])]
    j = int(np.nanargmax(scores))
    return F[:, j], float(scores[j])


def align(vec, ref):
    r = spearmanr(vec, ref).correlation
    return -vec if (r is not None and r < 0) else vec


def pc1_of(counts):
    """First PC of log-normalised counts — the signal a linear method sees."""
    lib = counts.sum(1, keepdims=True).clip(min=1)
    X = np.log1p(counts / lib * 1e4)
    X = X - X.mean(0)
    return skPCA(n_components=1, random_state=0).fit_transform(X)[:, 0]


def scatter_field(ax, coords, v, title=None, cmap=None, s=5.5):
    cmap = cmap or CMAP["score"]
    lo, hi = np.nanpercentile(v, [2, 98])
    ax.scatter(coords[:, 0], coords[:, 1], c=v, cmap=cmap, s=s,
                vmin=lo, vmax=hi, linewidths=0, marker="s", rasterized=True)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, pad=2, fontsize=plt.rcParams["font.size"])


def section(ax, text):
    """Bold section header anchored to the first axis of a block."""
    ax.text(0.0, 1.30, text, transform=ax.transAxes, ha="left", va="bottom",
             fontweight="bold", fontsize=plt.rcParams["font.size"] + 1,
             clip_on=False)


def main():
    apply(base_pt=7)
    fig = plt.figure(figsize=(7.5, 7.4))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[2.0, 1.0, 1.0],
                    hspace=0.55, left=0.07, right=0.99, top=0.93, bottom=0.04)

    # ---------------- (a) ground-truth isodepth per regime ------------------ #
    print("[a] ground-truth fields")
    ax00 = None
    for j, regime in enumerate(REGIMES):
        cfg = SimConfigV2(n_side=N_SIDE, M=2, corr=0.4, genes_per_mod=GENES,
                            regime=regime, noise=NOISE_LADDER[0][1], seed=0)
        d = simulate_v2(cfg)
        gsa = gs[0].subgridspec(2, 3, hspace=0.10, wspace=0.05)
        for m in range(2):
            ax = fig.add_subplot(gsa[m, j])
            scatter_field(ax, d.coords, d.rho_true[:, m],
                            title=regime if m == 0 else None)
            if j == 0:
                ax.set_ylabel(f"modality {m+1}", labelpad=2)
                if m == 0:
                    ax00 = ax
    section(ax00, "a   Ground-truth isodepth fields — three simulation regimes")

    # ---------------- (b) noise ladder -------------------------------------- #
    print("[b] noise ladder (PC1 of log-normalised counts)")
    gsb = gs[1].subgridspec(1, 4, wspace=0.05)
    ax_b0 = None
    for j, (name, spec) in enumerate(NOISE_LADDER):
        cfg = SimConfigV2(n_side=N_SIDE, M=2, corr=0.4, genes_per_mod=GENES,
                            regime=SHOWCASE["regime"], noise=spec, seed=SHOWCASE["seed"])
        d = simulate_v2(cfg)
        v = align(pc1_of(d.counts[0]), d.rho_true[:, 0])
        r = abs(spearmanr(v, d.rho_true[:, 0]).correlation)
        ax = fig.add_subplot(gsb[0, j])
        scatter_field(ax, d.coords, v, title=name)
        ax.text(0.5, -0.06, f"PC1 vs truth  |$\\rho$| = {r:.2f}",
                 transform=ax.transAxes, ha="center", va="top",
                 fontsize=plt.rcParams["font.size"] - 0.5, color="#555555")
        if j == 0:
            ax.set_ylabel("PC1 of observed\nexpression", labelpad=2, linespacing=1.2)
            ax_b0 = ax
    section(ax_b0, "b   Signal degradation under increasing corruption "
                     "(cascade regime, modality 1)")

    # ---------------- (c) recovered isodepth -------------------------------- #
    print("[c] recovery on showcase config — running methods")
    cfg = SimConfigV2(n_side=N_SIDE, M=SHOWCASE["M"], corr=SHOWCASE["corr"],
                        genes_per_mod=GENES, regime=SHOWCASE["regime"],
                        noise=SHOWCASE["noise"], seed=SHOWCASE["seed"])
    d = simulate_v2(cfg)
    truth = d.rho_true[:, 0]
    panels = [("ground truth", truth, None)]

    v = align(run_saturn(d, seed=SHOWCASE["seed"])[:, 0], truth)
    panels.append(("Saturn", v, abs(spearmanr(v, truth).correlation)))
    print(f"    Saturn        |rho| = {panels[-1][2]:.3f}")

    try:
        seed_all(SHOWCASE["seed"])
        Z = recover_gaston_single_per_mod(d.counts, d.coords, epochs=500,
                                            seed=SHOWCASE["seed"])
        v, sc = best_factor(Z[0], truth)
        panels.append(("GASTON-single", align(v, truth), sc))
        print(f"    GASTON-single |rho| = {sc:.3f}")
    except Exception as e:
        print("    GASTON-single failed:", e)

    for label, fn in [("PCA", BASE.recover_pca), ("NMF", BASE.recover_nmf)]:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                F = fn(d.counts, 8)
            v, sc = best_factor(F[0], truth)
            panels.append((label, align(v, truth), sc))
            print(f"    {label:13s} |rho| = {sc:.3f}")
        except Exception as e:
            print(f"    {label} failed:", e)

    gsc = gs[2].subgridspec(1, len(panels), wspace=0.05)
    ax_c0 = None
    for j, (name, v, score) in enumerate(panels):
        ax = fig.add_subplot(gsc[0, j])
        scatter_field(ax, d.coords, v, title=name)
        if score is not None:
            is_sat = name == "Saturn"
            ax.text(0.5, -0.06, f"|$\\rho$| = {score:.3f}", transform=ax.transAxes,
                     ha="center", va="top", fontsize=plt.rcParams["font.size"] - 0.5,
                     color=METHOD["Saturn"] if is_sat else "#555555",
                     fontweight="bold" if is_sat else "normal")
        if j == 0:
            ax.set_ylabel("isodepth\n(modality 1)", labelpad=2, linespacing=1.2)
            ax_c0 = ax
    section(ax_c0,
             f"c   Recovered isodepth in tissue space — one representative simulation "
             f"({SHOWCASE['regime']} regime, {SHOWCASE['noise_name']} noise, "
             f"M = {SHOWCASE['M']}, corr = {SHOWCASE['corr']}, seed {SHOWCASE['seed']})")
    ax_c0.text(0.0, 1.16,
                "config chosen so every method sits closest to its own benchmark mean "
                "at dropout-hi — not selected to favour any method",
                transform=ax_c0.transAxes, ha="left", va="bottom",
                fontsize=plt.rcParams["font.size"] - 1.5, color="#555555",
                clip_on=False)

    man = save_dual(fig, OUT, dpi=600, extra=dict(
        simulator=str(HOLE / "sim" / "simulate_v2.py"),
        model_cfg=MODEL_CFG, train_cfg=TRAIN_CFG,
        showcase={k: str(v) for k, v in SHOWCASE.items()},
        n_side=N_SIDE, genes_per_mod=GENES, device=DEV,
        recovery={n: s for n, _, s in panels if s is not None},
    ))
    print("saved:", man["png"], "\n      ", man["svg"])


if __name__ == "__main__":
    main()
