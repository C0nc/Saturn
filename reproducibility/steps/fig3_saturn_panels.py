"""Fig 3 without comparison methods: parts 1-3 and one SVG per panel, from the figure's own builders in the workspace.

Parts 1 and 2 (panels a-j) call fig3_parts.page() with the published layouts and builders, unchanged.
Part 3 is mixed in the published figure, so the Saturn part is drawn here:
  k   fig3_melanoma_v2.panel_fate_arrows, unchanged, with the DPT baseline removed from its kernel list (its last column stays empty)
  l-n the Saturn branches of panel_transition_maps, panel_profiles and panel_order, copied line by line; Mellon columns are dropped
These three builders cannot run as they are because they read mel_baseline_mellon.py's output, which is not produced here.
Dropped: Mellon columns in l, m and n; DPT column in k; supplementary figS3, which has its own DPT reference and is not Fig 3.

panel_order in the published figure draws Saturn ATAC, Mellon ATAC, Saturn RNA, Mellon RNA permutations from one random stream,
so without Mellon the rho_RNA permutation p values are drawn from a different part of that stream (rho_ATAC ones are unchanged).
-> published_figures/fig3/fig3_part{1,2,3}.{png,svg,json}, panels/fig3<letter>_<name>.svg, fig3_saturn_stats.json
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch
from scipy.stats import spearmanr

ROOT = os.environ["SATURN_ROOT"]
sys.path.insert(0, f"{ROOT}/Saturn/published_figures/fig3")
import fig3_parts as P
import fig3_melanoma as F3
import fig3_melanoma_v2 as V2
import fig2_lpc as F2
from paper_style import apply, save_panel_svg, MODALITY

V2.KERNELS = [k for k in V2.KERNELS if k[0] != "DPT"]


def _obs(D):
    o = pd.read_csv(V2.CRK / "obs.csv", index_col=0).reindex(D["bc"])
    g = pd.read_csv(V2.GRD / "spots.csv", index_col=0).reindex(D["bc"])
    return o, g, None


V2._obs = _obs


def units(D, m):
    """V2._units, Saturn branch: each connected gradient zone, ordered by rho_m, >= MIN_UNIT spots, numbered by size"""
    _, g, _ = _obs(D)
    zone = g[f"zone_{m}"].values; tr = g[f"transition_{m}"].values.astype(bool)
    U = [(zone == u) & tr for u in np.unique(zone[tr])]
    U = sorted([u for u in U if u.sum() >= V2.MIN_UNIT], key=lambda u: -u.sum())
    return dict(units=U, coord=D["d"][m], n_flagged=int(tr.sum()))


def panel_zones(fig, spec, D):
    """V2.panel_transition_maps, Saturn columns"""
    _, g, _ = _obs(D); Z = pd.read_csv(V2.GRD / "zones.csv"); A = json.load(open(V2.GRD / "summary.json"))
    gs = GridSpecFromSubplotSpec(1, 4, subplot_spec=spec, wspace=0.1); axs, out = [], {}
    for j, m in enumerate(["ATAC", "RNA"]):
        ax = fig.add_subplot(gs[j]); axs.append(ax)
        zone = g[f"zone_{m}"].values; tr = g[f"transition_{m}"].values.astype(bool); sharp = (zone >= 0) & ~tr
        F2.spatial(ax, D["xy"], None, colors="#e6e6e6", s=1.1)
        ax.scatter(*D["xy"][sharp].T, s=1.1, lw=0, color="#9aa2aa", rasterized=True)
        ax.scatter(*D["xy"][tr].T, s=1.5, lw=0, color=MODALITY[m], rasterized=True)
        z = Z[(Z.axis == m) & (Z.kind == "gradual")].sort_values("n", ascending=False).reset_index(drop=True)
        for i, r in z.iterrows():
            c = D["xy"][zone == r.zone].mean(0)
            ax.text(*c, str(i + 1), fontsize=5.5, ha="center", va="center", fontweight="bold", color="#1a1a1a",
                    bbox=dict(fc="white", ec=MODALITY[m], lw=0.5, boxstyle="circle,pad=0.15"))
        a = A[m]
        ax.set_title(f"Saturn $\\nabla$ρ$_{{\\mathrm{{{m}}}}}$ transition zones\n{len(z)} gradual (≥3 spots wide), "
                     f"{int((Z[Z.axis == m].kind == 'sharp').sum())} sharp", fontsize=5.6, color=MODALITY[m], pad=2)
        out[f"gradient_{m}"] = dict(n_gradual=len(z), align=a["median_align"], null=a["null_align_median"], p=a["p_align"],
                                    zones={int(r.zone): V2.ZONE_TAG.get((m, r.zone), "") for _, r in z.iterrows()})
    h = [Patch(fc=V2.C_ATAC, label="gradual zone (ATAC)"), Patch(fc=V2.C_RNA, label="gradual zone (RNA)"),
         Patch(fc="#9aa2aa", label="sharp interface (≤2 spots, spot mixing)")]
    axs[0].legend(handles=h, fontsize=5, ncol=3, loc="upper left", bbox_to_anchor=(0.0, -0.02), frameon=False,
                  handlelength=0.9, columnspacing=0.8, handletextpad=0.3)
    return axs[0], out


def panel_profiles(fig, spec, D):
    """V2.panel_profiles, Saturn columns"""
    M = V2._zprog(D); gs = GridSpecFromSubplotSpec(1, 4, subplot_spec=spec, wspace=0.45); axs, out = [], {}
    for j, m in enumerate(["ATAC", "RNA"]):
        U = units(D, m); s = U["units"][0]; c = U["coord"][s]
        ax = fig.add_subplot(gs[j]); axs.append(ax)
        q = np.quantile(c, np.linspace(0, 1, V2.NBIN + 1)); bi = np.clip(np.digitize(c, q[1:-1]), 0, V2.NBIN - 1)
        out[f"{m} Saturn"] = dict(n=int(s.sum()))
        for i, (k, lab, col) in enumerate(V2.PROG):
            v = M[s, i]; mu = np.array([v[bi == b_].mean() for b_ in range(V2.NBIN)])
            se = np.array([v[bi == b_].std() / np.sqrt(max((bi == b_).sum(), 1)) for b_ in range(V2.NBIN)])
            ax.plot(range(V2.NBIN), mu, "-o", color=col, ms=2, lw=0.9, label=lab)
            ax.fill_between(range(V2.NBIN), mu - se, mu + se, color=col, alpha=0.15, lw=0)
            out[f"{m} Saturn"][lab] = round(float(spearmanr(v, c)[0]), 3)
        ax.axhline(0, color="#9aa2aa", lw=0.4, ls=":")
        ax.set_xticks([0, V2.NBIN - 1]); ax.set_xticklabels(["low ρ", "high ρ"], fontsize=5.2)
        ax.set_title(f"Saturn ρ$_{{\\mathrm{{{m}}}}}$ zone 1\n{int(s.sum())} spots", fontsize=5.5, color=MODALITY[m], pad=2)
        if j == 0: ax.set_ylabel("programme score\n(z across tissue)")
        ax.set_xlabel("ρ", labelpad=0, fontsize=5.3)
    axs[0].legend(fontsize=5, loc="upper left", bbox_to_anchor=(0.0, -0.32), ncol=4, frameon=False, handlelength=1.2, columnspacing=0.8)
    return axs[0], out


def order_rows(M, U, rng):
    rows = []
    for i, s in enumerate(U["units"]):
        c = U["coord"][s]; o = V2._best_r(M[s], c)
        nul = np.array([V2._best_r(M[s], rng.permutation(c)) for _ in range(V2.NPERM)])
        rows.append(dict(unit=i + 1, n=int(s.sum()), best_r=round(o, 3), null_median=round(float(np.median(nul)), 3),
                         p=round(float((1 + (nul >= o).sum()) / (V2.NPERM + 1)), 4)))
    return pd.DataFrame(rows)


def panel_order(ax, D):
    """V2.panel_order, Saturn points"""
    M = V2._zprog(D); rng = np.random.default_rng(42); out = {}
    for j, m in enumerate(["ATAC", "RNA"]):
        U = units(D, m); col = MODALITY[m]
        R_ = order_rows(M, U, rng); ex = R_.best_r - R_.null_median; sig = R_.p < 0.05
        x = j + np.random.default_rng(j).uniform(-0.06, 0.06, len(R_))
        ax.scatter(x[sig], ex[sig], s=R_.n[sig] * 0.35 + 3, color=col, lw=0.4, edgecolor="#1a1a1a", zorder=3)
        ax.scatter(x[~sig], ex[~sig], s=R_.n[~sig] * 0.35 + 3, facecolor="white", edgecolor=col, lw=0.7, zorder=3)
        frac = float(R_.n[sig].sum() / U["n_flagged"])
        ax.text(j, 0.6, f"{frac:.0%}", ha="center", fontsize=5.3, color=col, fontweight="bold")
        out[f"{m}|Saturn"] = dict(units=R_.to_dict("records"), frac_flagged_spots_ordered=round(frac, 3))
    ax.axhline(0, color="#1a1a1a", lw=0.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["ATAC", "RNA"], fontsize=5.2)
    for t, c in zip(ax.get_xticklabels(), [MODALITY["ATAC"], MODALITY["RNA"]]): t.set_color(c)
    ax.axvline(0.5, color="#9aa2aa", lw=0.5, ls=":")
    ax.set_xlim(-0.55, 1.55); ax.set_ylim(-0.25, 0.8)
    ax.set_ylabel("ordering of programmes\nby ρ (|r| − null)")
    ax.set_title("every zone (size = spots)\nfilled: permutation p < 0.05\n% = flagged spots in an ordered unit", fontsize=5.2, pad=3)
    return out


P.BUILD.update(l=panel_zones, m=panel_profiles, n=F3._single(panel_order))
P.NAMES.update(l="transition_zones")


def main():
    apply(base_pt=6.5)
    D = F3.load(); F3.D_GLOBAL.update(D); res = {}
    P.page(D, res, "abcde", (7.2, 6.9), P.part1,
           dict(a=(-0.03, 0.015), b=(-0.06, 0.015), c=(-0.06, 0.015), d=(-0.03, 0.02), e=(-0.06, 0.03)), "fig3_part1")
    P.page(D, res, "fghij", (7.2, 4.3), P.part2,
           dict(f=(-0.05, 0.03), g=(-0.05, 0.03), h=(-0.02, 0.02), i=(-0.02, 0.02), j=(-0.06, 0.03)), "fig3_part2")
    P.page(D, res, "klmn", (7.2, 6.2), P.part3,
           dict(k=(-0.03, 0.025), l=(-0.03, 0.025), m=(-0.06, 0.03), n=(-0.07, 0.02)), "fig3_part3")
    pdir = P.OUT / "panels"; pdir.mkdir(exist_ok=True)
    for L in "abcdefghijklmn":
        fig = plt.figure(figsize=P.SIZE[L])
        spec = GridSpec(1, 1, figure=fig, left=0.12, right=0.9, top=0.84, bottom=0.2)[0]
        P.build(fig, L, spec, D, res); save_panel_svg(fig, pdir / f"fig3{L}_{P.NAMES[L]}"); plt.close(fig)
    json.dump({k: v for k, v in res.items() if k in "efgjklmn"}, open(P.OUT / "fig3_saturn_stats.json", "w"), indent=1, default=str)
    print("->", P.OUT)


if __name__ == "__main__":
    main()
