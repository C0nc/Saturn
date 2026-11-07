"""Figure 4 (v3) — Fig 4 v2 a-j unchanged + isoform switching along a cell-state transition inside one cell type.

Inhibitory Sst interneurons (594 cells in fitted bins) span the amygdala/olfactory | cortex boundary of rho_Isoform
(the KA|KC switch boundary of h-j; domains 1|2 at the elbow K = 3).
  k  Sst cells in space, coloured by the RNA state score t = P(cortex-like | scVI, Sst only), cross-validated logistic
     regression amygdala/olfactory vs cortex domain; transitional cells (0.25 <= t <= 0.75) ringed
  l  t per rho_Isoform domain, and how well each coordinate follows rho_Isoform (t vs RNA-only DPT, Spearman)
  m  isoform usage in Sst cells along t: 10 quantile windows (pooled reads) + 4-parameter sigmoid (binomial ML),
     switch point t0; q from 500 permutations of t (BH); within-domain q permutes t inside each domain
  n  all testable isoforms (>= 80 reads in Sst cells): usage change across t vs permutation q
Source: Hole/gaston_mix_mm/results/real_saturn/rtt_sst_transition/ (rtt_sst_isoform_transition.py).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

sys.path.insert(0, "<workspace>/Saturn/published_figures/_style")
sys.path.insert(0, "<workspace>/Saturn/published_figures/fig2")
sys.path.insert(0, "<workspace>/Saturn/published_figures/fig4")
from paper_style import apply, save_dual, DOMAIN, CMAP
import fig2_lpc as F2
import fig4_isoform_sv as F4S

H = "<workspace>"
SST = Path(f"{H}/Hole/gaston_mix_mm/results/real_saturn/rtt_sst_transition")
OUT = Path(f"{H}/Saturn/published_figures/fig4/fig4_isoform_v3")
FINAL = Path(f"{H}/Saturn/final_fig4_v3")
C_ISO = F4S.C_ISO
SW = [("Kcnab3", "ENSMUST00000142328"), ("Vbp1", "ENSMUST00000033540"), ("Coq8b", "ENSMUST00000003860")]
NWIN = 10


def _sst():
    return pd.read_csv(SST / "sst_states.csv", index_col=0), pd.read_csv(SST / "isoform_dynamics.csv"), \
        json.load(open(SST / "cellrank_summary.json"))


def panel_sst_space(ax, D):
    S, _, cr = _sst(); C = D["C"]
    ax.scatter(*C["xy"].T, s=0.05, lw=0, color="#e3e3e3", rasterized=True)
    o = np.argsort(S.t_state.values)
    sc = ax.scatter(S.x.values[o], S.y.values[o], c=S.t_state.values[o], cmap=CMAP["pseudotime"], vmin=0, vmax=1, s=1.6,
                    lw=0, rasterized=True)
    tr = S.transitional.values
    ax.scatter(S.x[tr], S.y[tr], s=3.2, facecolor="none", edgecolor="#1a1a1a", lw=0.2, rasterized=True)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    F2.cbar(ax.figure, ax, sc, "t", ticks=[0, 0.5, 1])
    ax.set_title(f"{cr['n_cells']} Sst interneurons\nCV AUC {cr['cv_auc_2v3']:.2f} (amyg. vs cortex)", fontsize=5.8, pad=2)
    ax.text(0.0, 0.0, f"○ transitional ({cr['n_transitional']})", transform=ax.transAxes, fontsize=5)
    return dict(n=cr["n_cells"], cv_auc=cr["cv_auc_2v3"], n_transitional=cr["n_transitional"])


def panel_sst_domain(ax, D):
    S, _, cr = _sst(); rng = np.random.default_rng(0)
    K = len(F4S.NAMES["Isoform"])
    for k in range(K):
        v = S.t_state[S.domain_iso == k].values
        if len(v) < 3: continue
        nm = F4S.NAMES["Isoform"][k]; col = DOMAIN[nm]
        vp = ax.violinplot(v, [k], widths=0.75, showextrema=False)
        for b in vp["bodies"]: b.set_facecolor(col); b.set_edgecolor("none"); b.set_alpha(0.55)
        ax.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=0.8, lw=0, color="#1a1a1a", alpha=0.5, rasterized=True)
        ax.plot([k - 0.28, k + 0.28], [v.mean()] * 2, color="#1a1a1a", lw=1.1)
    ax.axhspan(0.25, 0.75, color="#9aa2aa", alpha=0.12, lw=0)
    ax.set_xticks(range(K)); ax.set_xticklabels([f'{F4S.SHORT[F4S.NAMES["Isoform"][k]]} ({(S.domain_iso == k).sum()})' for k in range(K)], rotation=35, ha="right", fontsize=5.2)
    for t, k in zip(ax.get_xticklabels(), range(K)):
        t.set_bbox(dict(fc=DOMAIN[F4S.NAMES["Isoform"][k]], ec="none", alpha=0.35, pad=0.6))
    ax.set_ylim(-0.03, 1.03); ax.set_ylabel("RNA state score t")
    ax.set_title(f"vs ρ$_{{\\mathrm{{Isoform}}}}$: t {cr['t_vs_rho']:+.2f}, DPT {cr['dpt_vs_rho']:+.2f}", fontsize=5.6, pad=3)
    return dict(t_by_domain=cr["t_by_domain"], t_vs_rho=cr["t_vs_rho"], dpt_vs_rho=cr["dpt_vs_rho"])


def panel_sst_switch(fig, spec, D):
    S, T, _ = _sst(); t = S.t_state.values
    q = np.quantile(t, np.linspace(0, 1, NWIN + 1)); bi = np.clip(np.digitize(t, q[1:-1]), 0, NWIN - 1)
    tw = np.array([t[bi == b].mean() for b in range(NWIN)])
    gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.25); axs, out = [], {}
    xs = np.linspace(0, 1, 200)
    for j, (g, iso) in enumerate(SW):
        ax = fig.add_subplot(gs[j]); axs.append(ax); r = T[T.isoform == iso].iloc[0]
        uw = np.array([float(v) for v in r.uw.split(";")])
        ax.axvspan(0.25, 0.75, color="#9aa2aa", alpha=0.12, lw=0)
        ax.scatter(tw, uw, s=8, color=C_ISO, lw=0.3, edgecolor="#1a1a1a", zorder=3)
        ax.plot(xs, r.u_lo + (r.u_hi - r.u_lo) / (1 + np.exp(-r.steep * (xs - r.t0))), color=C_ISO, lw=1.1, zorder=2)
        ax.axvline(r.t0, color="#B64342", lw=0.7, ls="--")
        ax.text(r.t0 - 0.02, 0.97, f"t$_0$ {r.t0:.2f}", color="#B64342", fontsize=5, ha="right", va="top")
        ax.set_ylim(0, 1); ax.set_xlim(-0.02, 1.02); ax.set_xticks([0, 0.5, 1])
        ax.set_title(f"{g} {iso[-6:]}\nq {r.q:.3f}\nq$_{{\\mathrm{{in\\,domain}}}}$ {r.q_within_dom:.3f}", fontsize=5.6, pad=2,
                     color=C_ISO if r.state_coupled else "#5c6670", style="italic" if False else "normal")
        ax.set_xlabel("t  (amygdala → cortex state)" if j == 1 else "", labelpad=1)
        if j == 0: ax.set_ylabel("usage in Sst cells")
        else: ax.set_yticklabels([])
        out[f"{g} {iso}"] = dict(u_lo=round(r.u_lo, 3), u_hi=round(r.u_hi, 3), t0=round(r.t0, 3), q=round(r.q, 4),
                                 q_within_domain=round(r.q_within_dom, 4), state_coupled=bool(r.state_coupled),
                                 expr_ratio_sst_vs_nbr=round(r.expr_ratio_sst_vs_nbr, 2))
    return axs[0], out


def panel_sst_volcano(ax, D):
    _, T, _ = _sst()
    y = -np.log10(T.q.clip(1e-3)); sw = T.switch.values.astype(bool); sc_ = T.state_coupled.values.astype(bool)
    ax.scatter(T.du[~sw], y[~sw], s=6, color="#b0b7bd", lw=0)
    ax.scatter(T.du[sw], y[sw], s=12, color=C_ISO, lw=0.3, edgecolor="#1a1a1a", zorder=3)
    ax.scatter(T.du[sc_], y[sc_], s=34, facecolor="none", edgecolor="#B64342", lw=0.7, zorder=4)
    for _, r in T[sw].iterrows():
        ax.annotate(r.gene, (r.du, -np.log10(max(r.q, 1e-3))), fontsize=5, style="italic", xytext=(3, -1), textcoords="offset points")
    ax.axhline(-np.log10(0.1), color="#9aa2aa", lw=0.5, ls=":"); ax.axvline(0, color="#9aa2aa", lw=0.5)
    ax.set_xlabel("usage change along t", labelpad=1); ax.set_ylabel("−log$_{10}$ q (permutation)")
    ax.set_xlim(-0.55, 0.55)
    ax.set_title(f"{len(T)} testable isoforms, {int(sw.sum())} switches", fontsize=5.6, pad=3)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C_ISO, mec="#1a1a1a", mew=0.3, ms=3, label="switch"),
                       Line2D([], [], marker="o", ls="", mfc="none", mec="#B64342", mew=0.7, ms=4.5, label="holds within domain")],
              fontsize=5, loc="upper right", frameon=False, handletextpad=0.2, borderaxespad=0.1)
    ax.set_ylim(0, 2.6)
    return dict(n_tested=len(T), n_switch=int(sw.sum()), n_state_coupled=int(sc_.sum()))


PANELS = F4S.PANELS + [
    ("k", (1.8, 1.7), F2._single(panel_sst_space)),
    ("l", (1.5, 1.7), F2._single(panel_sst_domain)),
    ("m", (3.6, 1.5), panel_sst_switch),
    ("n", (1.6, 1.6), F2._single(panel_sst_volcano)),
]


def export_panels(D, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    for L, size, build in PANELS:
        fig = plt.figure(figsize=size)
        spec = GridSpec(1, 1, figure=fig, left=0.16, right=0.84, top=0.8, bottom=0.24)[0]
        build(fig, spec, D); save_dual(fig, outdir / f"fig4{L}", write_manifest=False); plt.close(fig)
    print(f"-> {len(PANELS)} panels in {outdir}")


def main():
    apply(base_pt=6.5)
    F2.MODALITY["Isoform"] = C_ISO
    D = F4S.load()
    fig = plt.figure(figsize=(7.2, 13.4))
    outer = GridSpec(7, 1, figure=fig, height_ratios=[1.55, 1.45, 2.0, 1.6, 1.45, 1.75, 1.6], hspace=0.72,
                     left=0.07, right=0.91, top=0.98, bottom=0.035)
    g1 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], width_ratios=[1.75, 0.95, 1.05], wspace=0.75)
    g4 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[3], width_ratios=[1.0, 0.2], wspace=0.1)
    g5 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[4], width_ratios=[1.0, 0.22], wspace=0.1)
    g6 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[5], width_ratios=[0.95, 1.75, 0.95], wspace=0.62)
    g7 = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[6], width_ratios=[1.05, 0.85, 2.5, 1.0], wspace=0.75)
    where = dict(a=g1[0], b=g1[1], c=g1[2], d=outer[1], e=outer[2], f=g4[0], g=g5[0], h=g6[0], i=g6[1], j=g6[2],
                 k=g7[0], l=g7[1], m=g7[2], n=g7[3])
    nudge = dict(a=(-0.03, 0.015), b=(-0.06, 0.015), c=(-0.07, 0.015), d=(-0.03, 0.02), e=(-0.06, 0.03),
                 f=(-0.06, 0.03), g=(-0.06, 0.03), h=(-0.07, 0.03), i=(-0.06, 0.03), j=(-0.1, 0.03),
                 k=(-0.03, 0.025), l=(-0.07, 0.02), m=(-0.06, 0.025), n=(-0.07, 0.02))
    res = {}
    for L, _, build in PANELS:
        ax, out = build(fig, where[L], D); res[L] = out; F2.letter(ax, L, *nudge[L])
    F2.draw_letters(fig)
    manifest = dict(fit=str(F4S.FIT), domains=F4S.NAMES, domain_composition=D["desc"], split_whole=res["c"],
                    features=res["e"], per_type_along_rho=res["f"], isoform_only=res["g"], switch_23=res["h"],
                    switch_split=res["j"], sst_state=res["k"], sst_domain=res["l"], sst_switches=res["m"],
                    sst_screen=res["n"], sources=[str(F4S.FIT), str(F4S.TAB), str(SST)])
    save_dual(fig, OUT, extra=manifest); save_dual(fig, FINAL / "fig4_isoform", extra=manifest); plt.close(fig)
    export_panels(D, FINAL / "panels")
    print(json.dumps(dict(k=res["k"], l=res["l"], m=res["m"], n=res["n"]), indent=1, default=str))


if __name__ == "__main__":
    main()
