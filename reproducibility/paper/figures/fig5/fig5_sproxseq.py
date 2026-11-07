"""Figure 5 — Sprox-seq tonsil: the isodepth predicts the held-out protein-complex layer and
recovers germinal-centre development from the tissue alone.

Sprox-seq (Wang, Xia ... Tay, Cell 2026), human tonsil, sections A1/B1/D1. Saturn p46 fits:
RNA (GLM-PCA) + 46 protein-probe marginals (CLR+PCA) as input. The 1,678 PLA products
(protein complexes / proximities) are NEVER in the loss -- they are the held-out layer.
Complex statistic = the authors' Pearson residual (interaction beyond both proteins' abundance).

  a  D1 tissue: follicle / GC zones from the authors' marker lists
  b  fit quality, three sections
  c  held-out complexes that track rho across the whole tissue (200-rotation spatial null)
  d  D1: rho_RNA, rho_Protein, the authors' GC pseudotime (DPT, re-run their way), rho_RNA in follicles
  e  rho_RNA vs DPT inside follicle/GC spots, by zone
  f  rho_RNA by zone: naive follicle -> dark zone -> light zone
  g  rho vs DPT for both axes in all three sections (rotation null)
  h  every complex: change along rho vs change along DPT inside follicle/GC (D1 rho_RNA)
  i  that agreement against the rotation null, both axes, all sections
  j  per-follicle replication
  k  example complexes along rho inside follicle/GC

Rho signs are arbitrary; each is oriented to increase with DPT (sproxseq_gc_trajectory.py).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/_style")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/fig2")
from paper_style import apply, save_dual, MODALITY, GCZONE, CMAP
import fig2_lpc as F2

H = "/mnt/HDD2B/Users/yancui"
R = Path(f"{H}/Hole/gaston_mix_mm/results/real_saturn")
G = R / "sproxseq_gc_trajectory"
OUT = Path(f"{H}/Saturn/published_figures/fig5/fig5_sproxseq")
FINAL = Path(f"{H}/Saturn/final_fig5")
SEC, AXIS = "D1", "RNA"                    # main example: the axis/section pair with full zone order
SECS, MODS = ["A1", "B1", "D1"], ["RNA", "Protein"]
ZNAME = {"NBC": "naive follicle", "DZ": "dark zone", "LZ": "light zone", "other": "extrafollicular"}
ZORDER = ["NBC", "DZ", "LZ"]
EXAMPLES = [("HLADPRQ:HLADPRQ", "MHC-II clustering"), ("CD81:CD9", "tetraspanin web"),
            ("CD35:VCAM1", "FDC contact"), ("ITGA4:VCAM1", "VLA-4–VCAM1")]
C_RNA, C_PROT = MODALITY["RNA"], MODALITY["Protein"]


def load():
    D = dict(summary=json.load(open(G / "summary.json")), whole=pd.read_csv(G / "whole_tissue.csv"),
             cx=pd.read_csv(G / "complex_dynamics.csv"), fol=pd.read_csv(G / "follicles.csv"),
             null=dict(np.load(G / "agreement_null.npz")),
             fit={s: json.load(open(R / f"sproxseq_{s}_p46" / "summary.json"))["R2_whole"] for s in SECS})
    sp_ = pd.read_csv(G / f"spots_{SEC}.csv", index_col=0)
    ax = pd.read_csv(R / f"sproxseq_{SEC}_p46" / "axes.csv", index_col=0)
    assert list(ax.index.astype(str)) == list(sp_.index.astype(str))
    D.update(spots=sp_, xy=np.c_[ax.x.values, ax.y.values],
             res=np.load(G / f"residuals_{SEC}.npy"),
             res_prod=pd.read_csv(G / f"residual_products_{SEC}.csv").iloc[:, 0].astype(str).values)
    return D


def srow(D, sec, m):
    return next(s for s in D["summary"] if s["section"] == sec and s["axis"] == m)


def spatial(ax, xy, v=None, colors=None, cmap=None, vmin=None, vmax=None, s=1.6):
    return F2.spatial(ax, xy, v, cmap=cmap, colors=colors, vmin=vmin, vmax=vmax, s=s)


# --------------------------------------------------------------------------- #
def panel_zones(ax, D):
    z = D["spots"].zone.values
    cols = np.array([GCZONE[ZNAME[k]] for k in z])
    spatial(ax, D["xy"], colors=cols, s=1.8)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=3.2, color=GCZONE[ZNAME[k]],
                              label=f"{ZNAME[k]} ({(z == k).sum()})") for k in ZORDER + ["other"]],
              fontsize=5.3, loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0, handletextpad=0.1,
              labelspacing=0.3)
    ax.set_title(f"tonsil {SEC}: zones from the authors' markers", fontsize=6.3, pad=2)


def panel_fit(ax, D):
    x = np.arange(3); w = 0.2
    for j, m in enumerate(MODS):
        for i, (key, alpha) in enumerate([("full", 1.0), ("own_axis_only", 0.5)]):
            v = [D["fit"][s][m][key] for s in SECS]
            ax.bar(x + (2 * j + i - 1.5) * w, v, w, color=MODALITY[m], alpha=alpha, edgecolor="#1a1a1a", lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels(SECS); ax.set_ylim(0, 0.85); ax.set_yticks([0, 0.4, 0.8])
    ax.set_ylabel("variance explained (R²)", labelpad=1)
    ax.legend(handles=[Patch(fc=MODALITY[m], alpha=a_, ec="#1a1a1a", lw=0.4, label=f"{m} {l_}")
                       for m in MODS for a_, l_ in [(1, "full"), (0.5, "own")]],
              fontsize=5.0 + 0.2, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=0.9,
              columnspacing=0.5, handletextpad=0.3, borderaxespad=0.1)


def panel_whole(ax, D):
    W = D["whole"]; x = np.arange(3); w = 0.36
    for j, m in enumerate(MODS):
        v = [100 * W[(W.section == s) & (W.axis == m)].frac_sig.iloc[0] for s in SECS]
        ax.bar(x + (j - 0.5) * w, v, w, color=MODALITY[m], edgecolor="#1a1a1a", lw=0.4, label=f"ρ$_{{\\mathrm{{{m}}}}}$")
        for xx, vv in zip(x + (j - 0.5) * w, v): ax.text(xx, vv + 0.8, f"{vv:.0f}", ha="center", fontsize=5.2)
    ax.axhline(5, color="#9aa2aa", lw=0.7, ls=(0, (2, 2))); ax.text(-0.62, 6.0, "chance", fontsize=5.2, color="#7f8a94", ha="left")
    ax.set_xticks(x); ax.set_xticklabels(SECS); ax.set_ylim(0, 40)
    ax.set_ylabel("held-out complexes tracking ρ (%)", labelpad=1)
    ax.legend(fontsize=5.3, loc="upper left", handlelength=0.9, borderaxespad=0.1)
    ax.set_title(f"~850 complexes, never in the fit", fontsize=6.3, pad=2)
    return {f"{r.section}|{r.axis}": f"{r.n_sig}/{r.n_complex}" for r in W.itertuples()}


def panel_maps(fig, spec, D):
    gs = GridSpecFromSubplotSpec(1, 4, subplot_spec=spec, wspace=0.35)
    S = D["spots"]; keep = S.zone.isin(ZORDER).values; out = None
    for i, (key, title, col) in enumerate([("rho_RNA", "ρ$_{\\mathrm{RNA}}$", C_RNA), ("rho_Protein", "ρ$_{\\mathrm{Protein}}$", C_PROT),
                                           ("dpt", "GC pseudotime (authors' DPT)", "#1a1a1a"),
                                           ("rho_RNA_f", "ρ$_{\\mathrm{RNA}}$ in follicle/GC", C_RNA)]):
        a = fig.add_subplot(gs[i]); out = out or a
        if key == "rho_RNA_f" or key == "dpt":
            spatial(a, D["xy"], colors=np.full(len(S), "#ececec"), s=1.6)
            v = S["dpt" if key == "dpt" else "rho_RNA"].values[keep]
            lo, hi = np.nanpercentile(v, [1, 99])
            sc = a.scatter(D["xy"][keep, 0], D["xy"][keep, 1], c=v, cmap=CMAP["pseudotime"] if key == "dpt" else CMAP["score"],
                           vmin=lo, vmax=hi, s=1.6, lw=0, rasterized=True)
        else:
            v = S[key].values; lo, hi = np.percentile(v, [1, 99])
            sc = spatial(a, D["xy"], v, cmap=CMAP["score"], vmin=lo, vmax=hi)
        F2.cbar(fig, a, sc, "pseudotime" if key == "dpt" else "isodepth", ticks=[round(lo, 2), round(hi, 2)])
        a.set_title(title, fontsize=6.5, pad=2, color=col)
    return out


def panel_scatter(ax, D):
    S = D["spots"]; keep = S.zone.isin(ZORDER).values
    for k in ZORDER:
        m = (S.zone == k).values
        ax.scatter(S.rho_RNA[m], S.dpt[m], s=2.5, lw=0, color=GCZONE[ZNAME[k]], alpha=0.8, rasterized=True)
    s = srow(D, SEC, AXIS)
    ax.set_xlabel("ρ$_{\\mathrm{RNA}}$", color=C_RNA, labelpad=1); ax.set_ylabel("GC pseudotime (DPT)", labelpad=1)
    ax.set_xlim(*np.percentile(S.rho_RNA[keep], [1, 99]) + np.array([-0.2, 0.2]))
    ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=2.8, color=GCZONE[ZNAME[k]], label=ZNAME[k]) for k in ZORDER],
              fontsize=5.1, loc="lower right", handletextpad=0.1, borderaxespad=0.1, labelspacing=0.25)
    ax.text(0.03, 0.97, f"Spearman {s['r_dpt']:.2f}\nrotation p {s['p_rot_dpt']:.2f}", transform=ax.transAxes,
            va="top", fontsize=5.5)
    ax.set_title(f"{keep.sum()} follicle/GC spots", fontsize=6.3, pad=2)
    return dict(r=s["r_dpt"], p=s["p_rot_dpt"], n=int(keep.sum()))


def panel_box(ax, D):
    S = D["spots"]
    data = [S.rho_RNA[S.zone == k].values for k in ZORDER]
    bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False, medianprops=dict(color="#1a1a1a", lw=1))
    for b, k in zip(bp["boxes"], ZORDER): b.set_facecolor(GCZONE[ZNAME[k]]); b.set_edgecolor("#1a1a1a"); b.set_lw(0.5)
    for w_ in bp["whiskers"] + bp["caps"]: w_.set_lw(0.5)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels([ZNAME[k].replace(" ", "\n") for k in ZORDER], fontsize=5.5)
    ax.set_ylabel("ρ$_{\\mathrm{RNA}}$", color=C_RNA, labelpad=1)
    ax.set_title("zone order along ρ", fontsize=6.3, pad=2)
    return {ZNAME[k]: round(float(np.median(d)), 3) for k, d in zip(ZORDER, data)}


def stars(p):
    return "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def panel_dpt_all(ax, D):
    x = np.arange(3); w = 0.36
    for j, m in enumerate(MODS):
        for i, s in enumerate(SECS):
            r = srow(D, s, m); xx = i + (j - 0.5) * w
            ax.bar(xx, r["r_dpt"], w, color=MODALITY[m], edgecolor="#1a1a1a", lw=0.4, alpha=1 if r["p_rot_dpt"] < 0.05 else 0.35)
            ax.text(xx, r["r_dpt"] + 0.02, stars(r["p_rot_dpt"]), ha="center", fontsize=5.2)
    ax.set_xticks(x); ax.set_xticklabels(SECS); ax.set_ylim(0, 0.85); ax.set_yticks([0, 0.4, 0.8])
    ax.set_ylabel("|Spearman| ρ vs GC pseudotime", labelpad=1)
    ax.legend(handles=[Patch(fc=MODALITY[m], ec="#1a1a1a", lw=0.4, label=f"ρ$_{{\\mathrm{{{m}}}}}$") for m in MODS],
              fontsize=5.3, loc="upper left", handlelength=0.9, borderaxespad=0.1, ncol=2, columnspacing=0.6)
    ax.set_title("rotation null, 200 rotations", fontsize=6.3, pad=2)


def panel_agree(ax, D):
    c = D["cx"]; c = c[(c.section == SEC) & (c.axis == AXIS)]
    sig = c.p_rot < 0.05
    ax.scatter(c.r_rho[~sig], c.r_dpt[~sig], s=2.5, lw=0, color="#c9cdd2", rasterized=True, label="ns along ρ")
    ax.scatter(c.r_rho[sig], c.r_dpt[sig], s=3.5, lw=0, color=C_RNA, rasterized=True, label="changes along ρ (p<0.05)")
    ax.axhline(0, color="#9aa2aa", lw=0.5); ax.axvline(0, color="#9aa2aa", lw=0.5)
    lim = 0.75; ax.plot([-lim, lim], [-lim, lim], color="#9aa2aa", lw=0.5, ls=(0, (2, 2)))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    for name, lab in EXAMPLES:
        r_ = c[c.complex == name].sort_values("reads").iloc[-1]
        ax.scatter(r_.r_rho, r_.r_dpt, s=12, facecolor="none", edgecolor="#1a1a1a", lw=0.7)
        off = {"MHC-II clustering": (-4, 4), "tetraspanin web": (-4, -8), "FDC contact": (4, -6), "VLA-4–VCAM1": (4, 1)}[lab]
        ax.annotate({"MHC-II clustering": "MHC-II", "tetraspanin web": "CD81:CD9"}.get(lab, lab), (r_.r_rho, r_.r_dpt), fontsize=5.1, xytext=off, textcoords="offset points",
                    ha="right" if off[0] < 0 else "left")
    s = srow(D, SEC, AXIS)
    ax.text(0.03, 0.97, f"agreement r {s['agreement']:.2f}\n{s['n_complex_sig']} change along ρ\n"
                        f"{100 * s['same_sign_as_dpt']:.0f}% same sign", transform=ax.transAxes, va="top", ha="left", fontsize=5.3)
    ax.set_xlabel("complex change along ρ$_{\\mathrm{RNA}}$ (Spearman)", labelpad=1)
    ax.set_ylabel("change along GC pseudotime", labelpad=1)
    ax.legend(fontsize=5.0 + 0.1, loc="lower right", markerscale=2, handletextpad=0.1, borderaxespad=0.1)
    ax.set_title(f"{len(c)} held-out complexes, {SEC}", fontsize=6.3, pad=2)
    return dict(agreement=s["agreement"], n_sig=s["n_complex_sig"], same_sign=s["same_sign_as_dpt"])


def panel_null(ax, D):
    ys = []
    for i, s in enumerate(SECS):
        for j, m in enumerate(MODS):
            y = i * 2.4 + j; ys.append((y, f"{s} ρ$_{{\\mathrm{{{m}}}}}$", m))
            nl = D["null"][f"{s}|{m}"]
            vp = ax.violinplot(nl, positions=[y], vert=False, widths=0.8, showextrema=False)
            for b in vp["bodies"]: b.set_facecolor("#c9cdd2"); b.set_edgecolor("none"); b.set_alpha(1)
            r = srow(D, s, m)
            ax.scatter(r["agreement"], y, s=16, color=MODALITY[m], edgecolor="#1a1a1a", lw=0.4, zorder=3,
                       alpha=1 if r["agreement_null_p"] < 0.05 else 0.4)
            ax.text(1.02, y, f"p {r['agreement_null_p']:.2f}", fontsize=5.1, va="center")
    ax.set_yticks([y for y, _, _ in ys]); ax.set_yticklabels([l for _, l, _ in ys], fontsize=5.4)
    for tl, (_, _, m) in zip(ax.get_yticklabels(), ys): tl.set_color(MODALITY[m])
    ax.axvline(0, color="#9aa2aa", lw=0.5); ax.set_xlim(-1, 1.25); ax.set_xticks([-1, 0, 1]); ax.invert_yaxis()
    ax.set_xlabel("agreement (ρ vs DPT across complexes)", labelpad=1)
    ax.set_title("vs rotated ρ (grey)", fontsize=6.3, pad=2)


def panel_follicles(ax, D):
    F = D["fol"]; i = 0; ticks = []
    for s in SECS:
        for m in MODS:
            f = F[(F.section == s) & (F.axis == m)]
            x = np.full(len(f), i) + np.random.default_rng(int(10 * i)).uniform(-0.15, 0.15, len(f))
            ax.scatter(x, f.r, s=np.clip(f.n / 8, 3, 30), color=MODALITY[m], edgecolor="#1a1a1a", lw=0.3, alpha=0.85)
            ax.plot([i - 0.25, i + 0.25], [np.median(f.r)] * 2, color="#1a1a1a", lw=1)
            ticks.append((i, s)); i += 1
        i += 0.5
    ax.axhline(0, color="#9aa2aa", lw=0.5)
    ax.set_xticks([np.mean([t for t, s_ in ticks if s_ == s]) for s in SECS]); ax.set_xticklabels(SECS)
    ax.set_ylabel("Spearman ρ vs DPT\nwithin one follicle", labelpad=1); ax.set_ylim(-1, 1)
    ax.set_title("each follicle/GC (size = spots)", fontsize=6.3, pad=2)


def panel_examples(fig, spec, D):
    S = D["spots"]; keep = S.zone.isin(ZORDER).values
    d = S.rho_RNA.values[keep]; z = S.zone.values[keep]
    c = D["cx"]; c = c[(c.section == SEC) & (c.axis == AXIS)]
    gs = GridSpecFromSubplotSpec(1, len(EXAMPLES), subplot_spec=spec, wspace=0.42)
    q = np.quantile(d, np.linspace(0, 1, 13)); bi = np.clip(np.digitize(d, q[1:-1]), 0, 11)
    first, st = None, {}
    for i, (name, lab) in enumerate(EXAMPLES):
        r_ = c[c.complex == name].sort_values("reads").iloc[-1]
        j = int(np.where(D["res_prod"] == r_["product"])[0][0]); y = D["res"][keep, j]
        a = fig.add_subplot(gs[i]); first = first or a
        xb = np.array([d[bi == b].mean() for b in range(12)]); yb = np.array([y[bi == b].mean() for b in range(12)])
        eb = np.array([y[bi == b].std() / np.sqrt((bi == b).sum()) for b in range(12)])
        zb = [pd.Series(z[bi == b]).value_counts().index[0] for b in range(12)]
        a.plot(xb, yb, color="#9aa2aa", lw=0.8, zorder=1)
        a.errorbar(xb, yb, yerr=eb, fmt="none", ecolor="#1a1a1a", elinewidth=0.5, capsize=0, zorder=2)
        a.scatter(xb, yb, s=13, c=[GCZONE[ZNAME[k]] for k in zb], edgecolor="#1a1a1a", lw=0.4, zorder=3)
        a.set_xlabel("ρ$_{\\mathrm{RNA}}$", color=C_RNA, labelpad=1)
        if i == 0: a.set_ylabel("Pearson residual\n(beyond abundance)", labelpad=1)
        a.set_title(f"{name.replace('HLADPRQ', 'HLA-DP/DR/DQ')}\n{lab}", fontsize=6.0, pad=2)
        a.text(0.97, 0.04 if r_.r_rho > 0 else 0.96, f"ρ r {r_.r_rho:+.2f} (p {r_.p_rot:.2f})\nDPT r {r_.r_dpt:+.2f}",
               transform=a.transAxes, fontsize=5.1, ha="right", va="bottom" if r_.r_rho > 0 else "top")
        a.tick_params(labelsize=5.5, pad=1)
        st[name] = dict(product=r_["product"], r_rho=round(float(r_.r_rho), 3), p=float(r_.p_rot), r_dpt=round(float(r_.r_dpt), 3))
    return first, st


# --------------------------------------------------------------------------- #
def _single(fn):
    def b(fig, spec, D):
        ax = fig.add_subplot(spec); return ax, fn(ax, D)
    return b


PANELS = [
    ("a", (2.6, 1.6), _single(panel_zones)),
    ("b", (1.6, 1.6), _single(panel_fit)),
    ("c", (1.7, 1.6), _single(panel_whole)),
    ("d", (7.2, 1.45), lambda fig, spec, D: (panel_maps(fig, spec, D), None)),
    ("e", (1.9, 1.7), _single(panel_scatter)),
    ("f", (1.5, 1.7), _single(panel_box)),
    ("g", (1.9, 1.7), _single(panel_dpt_all)),
    ("h", (2.5, 2.1), _single(panel_agree)),
    ("i", (2.2, 2.1), _single(panel_null)),
    ("j", (1.9, 2.1), _single(panel_follicles)),
    ("k", (7.2, 1.6), panel_examples),
]


def export_panels(D, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    for L, size, build in PANELS:
        fig = plt.figure(figsize=size)
        spec = GridSpec(1, 1, figure=fig, left=0.17, right=0.85, top=0.84, bottom=0.2)[0]
        build(fig, spec, D); save_dual(fig, outdir / f"fig5{L}", write_manifest=False); plt.close(fig)
    print(f"-> {len(PANELS)} panels in {outdir}")


def main():
    apply(base_pt=6.5)
    D = load()
    fig = plt.figure(figsize=(7.2, 9.7))
    outer = GridSpec(5, 1, figure=fig, height_ratios=[1.5, 1.35, 1.6, 2.0, 1.55], hspace=0.62,
                     left=0.08, right=0.96, top=0.975, bottom=0.05)
    g1 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], width_ratios=[1.6, 1.0, 1.0], wspace=0.75)
    g3 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], width_ratios=[1.2, 0.9, 1.2], wspace=0.55)
    g4 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[3], width_ratios=[1.25, 1.15, 1.0], wspace=0.6)
    where = dict(a=g1[0], b=g1[1], c=g1[2], d=outer[1], e=g3[0], f=g3[1], g=g3[2], h=g4[0], i=g4[1], j=g4[2], k=outer[4])
    nudge = dict(a=(-0.03, 0.015), b=(-0.07, 0.03), c=(-0.07, 0.015), d=(-0.03, 0.02), e=(-0.07, 0.015),
                 f=(-0.07, 0.015), g=(-0.07, 0.015), h=(-0.07, 0.015), i=(-0.09, 0.015), j=(-0.08, 0.015), k=(-0.07, 0.03))
    res = {}
    for L, _, build in PANELS:
        ax, out = build(fig, where[L], D); res[L] = out; F2.letter(ax, L, *nudge[L])
    F2.draw_letters(fig)
    manifest = dict(section=SEC, axis=AXIS, whole_tissue=res["c"], rho_vs_dpt=res["e"], zone_medians=res["f"],
                    agreement=res["h"], examples=res["k"], per_axis=D["summary"],
                    notes=dict(zones="authors' marker lists (their Leiden labels are not deposited)",
                               dpt="re-run as in paper_code/A1_B1_D1_code6.3 (root = most naive follicle spot)",
                               axis_choice="D1 rho_RNA shown as the main example; all six section/axis pairs in g, i, j"),
                    sources=[str(G), str(R / f"sproxseq_{SEC}_p46")])
    save_dual(fig, OUT, extra=manifest); save_dual(fig, FINAL / "fig5_sproxseq", extra=manifest); plt.close(fig)
    export_panels(D, FINAL / "panels")
    print(json.dumps(dict(c=res["c"], e=res["e"], f=res["f"], h=res["h"]), indent=1, default=str))


if __name__ == "__main__":
    main()
