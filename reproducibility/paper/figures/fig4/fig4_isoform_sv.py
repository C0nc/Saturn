"""Figure 4 (v2) — MERFISH isoform usage on the SPLISOSM-filtered whole-slice fit.

RT&T-AMP-MERFISH (Cohen ... Zhuang, Cell 2026), section 5. Saturn fit rtt_merfish_s5_iso100_sv_pca_shrunk,
no restarts, lam_cross 0.3: 100 um bins; RNA = gene GLM-PCA; Isoform = PCA of shrunk CLR isoform-ratio vectors of the
1,071 genes with spatially variable usage (SPLISOSM HSIC-IR, FDR < 0.05). Every cell is mapped to its bin.

  a  the section, cells coloured by class
  b  fit quality: R2 full / own / cross
  c  whole slice: usage change between rho_Isoform domains split into composition vs within-cell-type (Kitagawa),
     top 40 isoforms by deviance explained along rho_Isoform
  d  rho_RNA, RNA domains, rho_Isoform, Isoform domains (GASTON DP, elbow K: RNA 4, Isoform 3; labels from cell-type composition)
  e  fitted functions, each as original per-bin map | fitted function at every bin | bin means + fit (as Fig 2e):
     Tcf7l2, Fezf2 expression on rho_RNA (Poisson piecewise); Kcnab3 018614 and Mef2c 198199 usage on rho_Isoform
     (binomial piecewise, LRT gate t = 0.1)
  f  usage along rho_Isoform inside each cell class: per-class rho sextiles, bootstrap 95% CI; slope per SD of rho
     with the 200-rotation null
  g  same rho_Isoform domain, expression vs usage: per isoform, expression log2 FC vs usage change across the domain;
     isoform-only = usage sloped (quasi-binomial q < 0.05, |du| >= 0.15) while expression is flat
  h  the amygdala/olfactory | cortex boundary (domains KA|KC = 1|2 at the elbow K): usage on each side, switches coloured by within-type share
  i  switch isoforms across the KA|KC boundary: binned usage + per-domain binomial fits
  j  composition vs within-type split of the top switches, with usage in neurons only
Tables: Hole/gaston_mix_mm/results/real_saturn/<fit>/isodepth/ (rtt_isoform_sv_isodepth.py, rtt_isoform_sv_fig4_prep.py).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, "<workspace>/Saturn/published_figures/_style")
sys.path.insert(0, "<workspace>/saturn_repo/src")       # Saturn package (saturn_repo)
sys.path.insert(0, "<workspace>/Saturn/published_figures/fig2")
sys.path.insert(0, "<workspace>/Saturn/published_figures/fig4")
sys.path.insert(0, "<workspace>/Hole/gaston_mix_mm/scripts")
from paper_style import apply, save_dual, MODALITY, DOMAIN, CELLCLASS, CMAP
import fig2_lpc as F2
import fig4_isoform as F4
import rtt_isoform_sv_isodepth as I
import merfish_isoform_deep as MD

H = "<workspace>"
FIT = Path(I.FIT); TAB = Path(I.OUT)
OUT = Path(f"{H}/Saturn/published_figures/fig4/fig4_isoform_sv")
FINAL = Path(f"{H}/Saturn/final_fig4_v2")
MODALITY = dict(MODALITY, Isoform=MODALITY["Splicing"])
C_RNA, C_ISO = MODALITY["RNA"], MODALITY["Isoform"]
NAMES = {"RNA": {0: "thalamus / hypothalamus", 1: "white matter & hippocampus", 2: "ventricle / choroid plexus", 3: "cortex"},
         "Isoform": {0: "subcortical & white matter", 1: "amygdala / olfactory cortex", 2: "cortex"}}   # elbow K: RNA 4, Isoform 3
KA, KC = 1, 2          # rho_Isoform domains either side of the switch boundary: amygdala / olfactory cortex | cortex
SHORT = {"thalamus / hypothalamus": "thal./hypoth.", "white matter & hippocampus": "WM & hippoc.",
         "ventricle / choroid plexus": "ventricle/CP", "amygdala / olfactory cortex": "amyg./olf. ctx", "subcortical": "subcortical", "subcortical & white matter": "subcort. & WM",
         "cortex": "cortex"}
GENES_E = [("Tcf7l2", "thalamic"), ("Fezf2", "deep-layer cortical")]
ISOS_E = [("Kcnab3", "ENSMUST00000018614"), ("Mef2c", "ENSMUST00000198199")]
RHO_EX = [("Lamp5", "ENSMUST00000143777"), ("Mef2c", "ENSMUST00000198199"), ("Kcnab3", "ENSMUST00000018614")]
SWITCH_EX = [("Kcnab3", "ENSMUST00000018614"), ("Pax3", "ENSMUST00000172555"), ("Ace", "ENSMUST00000001964")]
CLASSES = ["neuron", "oligodendrocyte", "astrocyte", "OPC"]
NQ, NROT = 6, 200


# --------------------------------------------------------------------------- #
def load():
    C, B = I.load(); lab, desc = I.domains(B)
    for m in lab:                                     # labels must still describe the domains they name
        assert len(desc[m]) == len(NAMES[m]), (m, len(desc[m]))
        top = max(desc[m]); assert "Cortex L2/3 IT" in desc[m][top]["top_types"] or "Cortex Subplate" in desc[m][top]["top_types"], desc[m][top]
    t0 = desc["Isoform"][0]["top_types"]                # merged subcortical + white matter: oligodendrocyte-led, with thalamus
    assert "Oligodendrocytes" in t0 and ("Thalamus" in t0 or "Hypothalamus" in t0), t0
    assert "Subplate" in desc["Isoform"][KA]["top_types"]
    dg = json.load(open(FIT / "diag.json"))
    D = dict(xy=B["xy"], d=B["d"], lab=lab, desc=desc, names_dom=NAMES, summary=dict(R2_whole=dg["R2_whole"]), C=C, B=B,
             X={"RNA": B["Xg"].astype(float)}, names={"RNA": B["genes"]}, lib={"RNA": B["lib"]})
    return D


def dcol(D, m):
    return np.array([DOMAIN[NAMES[m][k]] for k in range(len(NAMES[m]))])


# --------------------------------------------------------------------------- #
def panel_split_whole(ax, D):
    S = pd.read_csv(TAB / "composition_vs_within.csv")
    v = S.within_frac.clip(-0.25, 1.0).values; sig = S.within_sig.values.astype(bool)
    x = np.random.default_rng(0).uniform(-0.25, 0.25, len(v))
    ax.scatter(x[~sig], v[~sig], s=5, lw=0, color="#b0b7bd", label="within-type CI spans 0")
    ax.scatter(x[sig], v[sig], s=5, lw=0, color=C_ISO, label="within-type significant")
    med = float(np.median(S.within_frac)); ax.plot([-0.38, 0.38], [med, med], color="#1a1a1a", lw=1.3)
    ax.axhline(0, color="#9aa2aa", lw=0.5, ls=(0, (2, 2)))
    ax.set_xlim(-0.4, 1.9); ax.set_xticks([]); ax.set_ylim(-0.27, 1.05)
    ax.set_ylabel("within-cell-type share\nof the usage change", labelpad=1)
    ax.text(0.45, med, f"median {med:.2f}\n→ {100 * (1 - med):.0f}% composition", fontsize=5.5, va="center")
    ax.text(0.45, 0.85, f"{int(sig.sum())}/{len(S)} with significant\nwithin-type component", fontsize=5.3, va="top")
    ax.legend(fontsize=5.1, loc="lower left", bbox_to_anchor=(0.33, 0.0), handletextpad=0.1, borderaxespad=0.1, markerscale=1.5)
    ax.set_title(f"top {len(S)} isoforms along ρ$_{{\\mathrm{{Isoform}}}}$", fontsize=6.3, pad=2)
    return dict(median_within=med, n_sig=int(sig.sum()), n=len(S))


def panel_maps(fig, spec, D):
    gs = GridSpecFromSubplotSpec(2, 4, subplot_spec=spec, height_ratios=[1, 0.3], wspace=0.18, hspace=0.02); first = None
    for j, m in enumerate(("RNA", "Isoform")):
        a1 = fig.add_subplot(gs[0, 2 * j]); a2 = fig.add_subplot(gs[0, 2 * j + 1]); first = first or a1
        d = D["d"][m]; lo, hi = np.percentile(d, [1, 99])
        sc = F2.spatial(a1, D["xy"], d, cmap=CMAP["score"], vmin=lo, vmax=hi, s=0.9)
        F2.cbar(fig, a1, sc, "isodepth", ticks=[round(lo, 1), round(hi, 1)])
        a1.set_title(f"ρ$_{{\\mathrm{{{m}}}}}$", color=MODALITY[m], fontsize=7, pad=2)
        F2.spatial(a2, D["xy"], None, colors=dcol(D, m)[D["lab"][m]], s=0.9)
        a2.set_title(f"{m} domains", color=MODALITY[m], fontsize=7, pad=2)
        lg = fig.add_subplot(gs[1, 2 * j:2 * j + 2]); lg.set_axis_off()
        for i, k in enumerate(range(len(NAMES[m]))):
            n = int((D["lab"][m] == k).sum()); x_ = 0.0 + (i % 2) * 0.52; y_ = 0.72 - (i // 2) * 0.5
            lg.add_patch(Rectangle((x_, y_ - 0.12), 0.035, 0.3, color=DOMAIN[NAMES[m][k]], transform=lg.transAxes))
            lg.text(x_ + 0.05, y_ + 0.03, f"{k} {NAMES[m][k]} ({n})", fontsize=5.4, va="center", transform=lg.transAxes)
    return first


def iso_counts(D, iso):
    B = D["B"]; C = D["C"]
    j = int(np.where(C["iso"] == iso)[0][0]); cols = np.where(C["iso_gene"] == C["iso_gene"][j])[0]
    return np.asarray(B["Xi"][:, j].todense()).ravel().astype(float), np.asarray(B["Xi"][:, cols].sum(1)).ravel().astype(float)


def usage_fit(D, s, n):
    """binomial piecewise fit of isoform usage along rho_Isoform (LRT gate t = 0.1): slopes, intercepts per domain"""
    from saturn.piecewise import fit_binomial
    d, lab = D["d"]["Isoform"], D["lab"]["Isoform"]
    sl, it = fit_binomial(s[:, None], n, lab, d, t=0.1)
    return sl[0], it[0]


def usage_curve(a2, D, s, n, doms=None, nb=30, show_bnd=False):
    from saturn.domains import isodepth_bins as gaston_bins
    d, lab = D["d"]["Isoform"], D["lab"]["Isoform"]; cl = dcol(D, "Isoform")
    sl, it = usage_fit(D, s, n)
    bi, _, kp = gaston_bins(d, nb)
    db = np.array([d[bi == b].mean() for b in kp]); ob = np.array([s[bi == b].sum() / max(n[bi == b].sum(), 1) for b in kp])
    lb = np.array([np.bincount(lab[bi == b], minlength=lab.max() + 1).argmax() for b in kp]); nb_ = np.array([n[bi == b].sum() for b in kp])
    for k in (doms if doms is not None else range(lab.max() + 1)):
        m = (lb == k) & (nb_ >= 20)
        a2.scatter(db[m], ob[m], s=5, color=cl[k], lw=0.3, edgecolor="#1a1a1a", zorder=3)
        dk = d[lab == k]; xs = np.linspace(dk.min(), dk.max(), 20)
        a2.plot(xs, 1 / (1 + np.exp(-(it[k] + sl[k] * xs))), color=cl[k], lw=1.3)
    for b in sorted([d[lab == k].max() for k in range(lab.max() + 1)])[:-1]:
        a2.axvline(b, color=F2.GRID, lw=0.4, ls=(0, (2, 2)), zorder=1)
    if show_bnd:
        a2.axvline((d[lab == KA].max() + d[lab == KC].min()) / 2, color="#1a1a1a", lw=0.5, ls=(0, (2, 2)))
    a2.set_ylim(-0.02, 1.02); a2.set_xlabel("ρ$_{\\mathrm{Isoform}}$", color=C_ISO, labelpad=1)
    a2.set_ylabel("usage", labelpad=1); a2.tick_params(labelsize=5.5, pad=1)
    return int((np.abs(sl) > 1e-12).sum())


def panel_iso_feature(fig, spec, D, g, iso):
    """one isoform, three columns as Fig 2e: ORIGINAL per-bin usage (reads of the isoform / reads of its gene, no
    smoothing; bins without reads blank) | the fitted binomial function at every bin | bin means + fitted function"""
    gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, width_ratios=[1, 1, 1.05], wspace=0.95)
    s, n = iso_counts(D, iso); d, lab = D["d"]["Isoform"], D["lab"]["Isoform"]
    sl, it = usage_fit(D, s, n); fit = 1 / (1 + np.exp(-(it[lab] + sl[lab] * d)))
    u = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    a1, a2, a3 = (fig.add_subplot(gs[i]) for i in range(3))
    ok = np.isfinite(u)
    F2.spatial(a1, D["xy"][~ok], None, colors="#ececec", s=0.9)
    F2.spatial(a1, D["xy"][ok], u[ok], cmap=CMAP["expression"], vmin=0, vmax=1, s=0.9)
    sc = F2.spatial(a2, D["xy"], fit, cmap=CMAP["expression"], vmin=0, vmax=1, s=0.9)
    F2.cbar(fig, a2, sc, "usage", ticks=[0, 0.5, 1], on_top=True)
    a1.set_title(f"{g} isoform {iso[-6:]}\nusage", fontsize=6.2, pad=2, color=C_ISO, loc="left")
    a1.text(0.5, -0.04, "original", transform=a1.transAxes, ha="center", va="top", fontsize=5.5)
    a2.text(0.5, -0.04, "fitted function", transform=a2.transAxes, ha="center", va="top", fontsize=5.5)
    k = usage_curve(a3, D, s, n)
    a3.set_title(f"fitted function (binomial)\n{k}/{len(NAMES['Isoform'])} domains sloped", fontsize=5.8, pad=2)
    return a1, dict(isoform=iso, n_sloped=k, reads=int(s.sum()))


def panel_features(fig, spec, D):
    """2 x 2: genes on rho_RNA (Fig 2e builder, Poisson) and isoform usage on rho_Isoform (binomial)"""
    F2.NB["RNA"] = 25
    gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=spec, wspace=0.22, hspace=0.6); stats, first = {}, None
    D_ = dict(D, names_dom=NAMES)
    F2.dcol = lambda D__, m: dcol(D__, m)
    for i, (g, sub) in enumerate(GENES_E):
        a_, Fres = F2.panel_feature(fig, gs[0, i], D_, "RNA", g, g, sub)
        first = first or a_; stats[g] = dict(R2=round(float(Fres["r2"]), 3), n_sloped=Fres["n_sloped"])
    for i, (g, iso) in enumerate(ISOS_E):
        _, stats[g] = panel_iso_feature(fig, gs[1, i], D, g, iso)
    return first, stats


def panel_rho_within(fig, spec, D):
    from sproxseq_validate2 import rotated_fields
    C, B = D["C"], D["B"]; rng = np.random.default_rng(0)
    rot = rotated_fields(D["xy"], D["d"]["Isoform"], NROT, np.random.default_rng(1))
    gs = GridSpecFromSubplotSpec(1, len(RHO_EX), subplot_spec=spec, wspace=0.42); first, st = None, {}
    for i, (g, iso) in enumerate(RHO_EX):
        ax = fig.add_subplot(gs[i]); first = first or ax
        S = I.per_type(C, B, g, iso, "Isoform", rot, rng, None); st[g] = S
        a_, tot = MD.usage_counts(C, g, iso); ok = (C["cbin"] >= 0) & (tot > 0); dbin = D["d"]["Isoform"]
        for cls in CLASSES + ["all cells"]:
            if cls not in S: continue
            idx = np.where(ok & ((C["cls"] == cls) if cls != "all cells" else True))[0]
            x = dbin[C["cbin"][idx]]; A_, N_ = a_[idx].astype(float), tot[idx].astype(float)
            q = np.quantile(x, np.linspace(0, 1, NQ + 1)); bi = np.clip(np.digitize(x, q[1:-1]), 0, NQ - 1); xb, yb, lo, hi = [], [], [], []
            for b in range(NQ):
                mb = np.where(bi == b)[0]
                if N_[mb].sum() < 10: continue
                bs = [A_[s_].sum() / max(N_[s_].sum(), 1) for s_ in (rng.choice(mb, len(mb)) for _ in range(200))]
                xb.append(x[mb].mean()); yb.append(A_[mb].sum() / N_[mb].sum()); lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
            col = "#1a1a1a" if cls == "all cells" else CELLCLASS[cls]; ls = ":" if cls == "neuron" else "-"
            ax.fill_between(xb, lo, hi, color=col, alpha=0.12, lw=0)
            ax.plot(xb, yb, color=col, lw=1.4 if cls == "all cells" else 1.1, ls=ls, marker="o", ms=2.2, label=cls)
        ax.set_ylim(0, 1.42); ax.set_yticks([0, 0.5, 1.0]); ax.set_xlabel("ρ$_{\\mathrm{Isoform}}$", color=C_ISO, labelpad=1)
        if i == 0: ax.set_ylabel("isoform usage", labelpad=1)
        ax.tick_params(labelsize=5.5, pad=1); ax.set_title(f"{g} {iso[-6:]} along ρ$_{{\\mathrm{{Isoform}}}}$", fontsize=6.2, pad=2, color=C_ISO)
        txt = "\n".join(f"{'oligo' if c == 'oligodendrocyte' else c} {S[c]['slope_per_sd']:+.2f} (p {S[c]['p_rot']:.2f})"
                        for c in CLASSES if c in S)
        ax.text(0.02, 0.99, "slope / SD ρ (rotation p)\n" + txt, transform=ax.transAxes, ha="left", va="top",
                fontsize=5, color="#3a3f4b", linespacing=1.05)
        if i == len(RHO_EX) - 1:
            ax.legend(fontsize=5.1, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, handlelength=1.4, labelspacing=0.3)
    return first, st


def panel_iso_only(fig, spec, D):
    T = pd.read_csv(TAB / "domain_slopes.csv"); T = T[(T.gene_reads >= 50) & T.gene_on_panel & (T.expr_counts >= 20)]
    K = len(NAMES["Isoform"]); gs = GridSpecFromSubplotSpec(1, K, subplot_spec=spec, wspace=0.12); first, st = None, {}
    for k in range(K):
        ax = fig.add_subplot(gs[k]); first = first or ax; t = T[T.domain == k]
        cat = np.where(t.isoform_only, "isoform only", np.where(t.usage_sloped, "usage + expression",
                       np.where(~t.expr_flat, "expression only", "neither")))
        for c, col, sz, z in (("neither", "#d9dcdf", 2, 1), ("expression only", "#9aa2aa", 2.5, 2),
                              ("usage + expression", "#4d5560", 4, 3), ("isoform only", C_ISO, 6, 4)):
            m = cat == c
            ax.scatter(t.expr_log2fc[m].clip(-4, 4), t.usage_du[m], s=sz, lw=0, color=col, zorder=z, rasterized=True, label=c)
        ax.axvspan(-0.5, 0.5, color=DOMAIN[NAMES["Isoform"][k]], alpha=0.12, lw=0, zorder=0)
        ax.axhline(0.15, color="#9aa2aa", lw=0.5, ls=(0, (2, 2))); ax.axhline(-0.15, color="#9aa2aa", lw=0.5, ls=(0, (2, 2)))
        ax.set_xlim(-4.2, 4.2); ax.set_ylim(-0.9, 0.9); ax.tick_params(labelsize=5.5, pad=1)
        ax.set_xlabel("expression log2 FC" if k else "expression log2 FC across domain", labelpad=1)
        if k == 0: ax.set_ylabel("usage change\nacross domain", labelpad=1)
        else: ax.set_yticklabels([])
        ax.set_title(f"{k} {SHORT[NAMES['Isoform'][k]]}\n{int(t.isoform_only.sum())} of {int(t.usage_sloped.sum())} sloped are isoform-only", fontsize=5.7, pad=2,
                     color="#3a6f9e" if NAMES["Isoform"][k] == "white matter & hippocampus" else DOMAIN[NAMES["Isoform"][k]])
        st[k] = dict(tested=len(t), usage_sloped=int(t.usage_sloped.sum()), isoform_only=int(t.isoform_only.sum()))
        if k == K - 1:
            ax.legend(fontsize=5.0, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, markerscale=1.6, labelspacing=0.3)
    return first, st


def panel_switch_scatter(ax, D):
    S = pd.read_csv(TAB / "switch_23.csv"); e = (S.reads_2 >= 100) & (S.reads_3 >= 100); S = S[e]
    ax.scatter(S.usage_2, S.usage_3, s=2, lw=0, color="#d0d4d8", rasterized=True, label=f"isoforms ({len(S)})")
    sw = S[S.switch]
    ax.plot([0, 1], [0, 1], color="#9aa2aa", lw=0.5, ls=(0, (2, 2)))
    ax.axhline(0.5, color="#e3e5e8", lw=0.4); ax.axvline(0.5, color="#e3e5e8", lw=0.4)
    has = sw.within_frac.notna()
    ax.scatter(sw.usage_2[~has], sw.usage_3[~has], s=6, lw=0, color="#7f8a94", label=f"switches ({len(sw)})")
    sc = ax.scatter(sw.usage_2[has], sw.usage_3[has], s=10, lw=0.3, edgecolor="#1a1a1a",
                    c=sw.within_frac[has].clip(0, 1), cmap=CMAP["score"], vmin=0, vmax=1, zorder=3, label="top 30, split")
    for g, iso in SWITCH_EX:
        r = S[S.isoform == iso]
        if len(r): ax.annotate(g, (r.usage_2.iloc[0], r.usage_3.iloc[0]), fontsize=5.3, style="italic", xytext=(3, -6), textcoords="offset points")
    cb = plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.02); cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.2, length=1.5)
    cb.set_label("within-type share", fontsize=5.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.tick_params(labelsize=5.5, pad=1)
    ax.set_xlabel(f"usage in {KA} amyg./olf. ctx", labelpad=1, color="#a77a2c"); ax.set_ylabel(f"usage in {KC} cortex", labelpad=1, color=DOMAIN["cortex"])
    ax.set_title(f"{KA} | {KC}: {len(sw)} switches\n({int(sw.flip.sum())} flips; grey = not split)", fontsize=5.8, pad=2, loc="left")
    return dict(n=len(S), switches=len(sw), flips=int(sw.flip.sum()))


def boundary_curve(ax, D, s, n, nq=10):
    """domains KA and KC on a RANK scale of rho (domain KA spans a narrow rho interval); fits drawn through the same map"""
    from saturn.piecewise import fit_binomial
    d, lab = D["d"]["Isoform"], D["lab"]["Isoform"]; cl = dcol(D, "Isoform")
    sl, it = fit_binomial(s[:, None], n, lab, d, t=0.1); sl, it = sl[0], it[0]
    w = np.sort(d[(lab == KA) | (lab == KC)]); rk = lambda v: np.searchsorted(w, v) / len(w)
    for k in (KA, KC):
        m = np.where(lab == k)[0]; x = d[m]; q = np.quantile(x, np.linspace(0, 1, nq + 1)); bi = np.clip(np.digitize(x, q[1:-1]), 0, nq - 1)
        xb = np.array([x[bi == b].mean() for b in range(nq)]); nb_ = np.array([n[m][bi == b].sum() for b in range(nq)])
        ub = np.array([s[m][bi == b].sum() for b in range(nq)]) / np.maximum(nb_, 1)
        ok = nb_ >= 15
        ax.scatter(rk(xb[ok]), ub[ok], s=6, color=cl[k], lw=0.3, edgecolor="#1a1a1a", zorder=3)
        xs = np.quantile(x, np.linspace(0.01, 0.99, 40)); ax.plot(rk(xs), 1 / (1 + np.exp(-(it[k] + sl[k] * xs))), color=cl[k], lw=1.3)
    ax.axvline(((lab == KA).sum()) / len(w), color="#1a1a1a", lw=0.5, ls=(0, (2, 2)))
    ax.set_ylim(-0.02, 1.02); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel(f"ρ$_{{\\mathrm{{Isoform}}}}$ rank ({KA} | {KC})", color=C_ISO, labelpad=1); ax.tick_params(labelsize=5.5, pad=1)


def panel_switch_curves(fig, spec, D):
    gs = GridSpecFromSubplotSpec(1, len(SWITCH_EX), subplot_spec=spec, wspace=0.42); first = None
    d, lab = D["d"]["Isoform"], D["lab"]["Isoform"]; win = (lab == KA) | (lab == KC)
    for i, (g, iso) in enumerate(SWITCH_EX):
        ax = fig.add_subplot(gs[i]); first = first or ax; s, n = iso_counts(D, iso)
        boundary_curve(ax, D, s, n)
        if i == 0: ax.set_ylabel("usage (10 ρ bins / domain)", labelpad=1)
        if i != len(SWITCH_EX) // 2: ax.set_xlabel("")
        ax.set_title(f"{g} {iso[-6:]}", fontsize=6.2, pad=2, color=C_ISO)
    return first


def panel_switch_split(ax, D):
    S = pd.read_csv(TAB / "switch_23.csv"); S = S[S.switch & S.within_frac.notna()].head(12).iloc[::-1]
    y = np.arange(len(S))
    ax.barh(y, S.composition, color="#b0b7bd", height=0.7, label="composition")
    ax.barh(y, S.within, left=np.where(np.sign(S.within) == np.sign(S.composition), S.composition, 0), color=C_ISO, height=0.7,
            label="within cell type")
    ax.errorbar(np.where(np.sign(S.within) == np.sign(S.composition), S.composition, 0) + S.within, y,
                xerr=[S.within - S.within_ci_lo, S.within_ci_hi - S.within], fmt="none", ecolor="#1a1a1a", elinewidth=0.5, capsize=1)
    for yy, (_, r) in zip(y, S.iterrows()):
        ax.text(1.03, yy, f"{r.neuron_usage_2:.2f}→{r.neuron_usage_3:.2f}", transform=ax.get_yaxis_transform(), fontsize=5,
                va="center", ha="left", color="#7a5a86")
    ax.axvline(0, color="#1a1a1a", lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels([f"{g} {i[-6:]}" for g, i in zip(S.gene, S.isoform)], fontsize=5.1, style="italic")
    ax.set_xlabel(f"usage change {KA} → {KC}", labelpad=1); ax.tick_params(axis="x", labelsize=5.5, pad=1)
    ax.text(1.03, 1.0, f"neurons\n{KA}→{KC}", transform=ax.transAxes, fontsize=5.0, ha="left", va="bottom", color="#7a5a86")
    ax.legend(fontsize=5.0, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, borderaxespad=0.1, handlelength=1.0, columnspacing=0.6)
    ax.set_title("top switches: composition vs within", fontsize=6.3, pad=12)
    return S[["gene", "isoform", "composition", "within", "within_frac", "neuron_usage_2", "neuron_usage_3"]].round(3).to_dict("records")


def panel_fit(ax, D):
    F4.panel_fit(ax, D["summary"]); ax.set_ylim(0, 0.75); ax.set_yticks([0, 0.25, 0.5, 0.75])


# --------------------------------------------------------------------------- #
def _single(fn):
    def b(fig, spec, D):
        ax = fig.add_subplot(spec); return ax, fn(ax, D)
    return b


PANELS = [
    ("a", (2.6, 1.6), _single(lambda ax, D: F4.panel_cells(ax, D))),
    ("b", (1.45, 1.6), _single(panel_fit)),
    ("c", (1.6, 1.6), _single(panel_split_whole)),
    ("d", (7.2, 1.5), lambda fig, spec, D: (panel_maps(fig, spec, D), None)),
    ("e", (7.2, 2.9), panel_features),
    ("f", (5.6, 1.7), panel_rho_within),
    ("g", (6.4, 1.6), panel_iso_only),
    ("h", (2.0, 1.9), _single(panel_switch_scatter)),
    ("i", (4.0, 1.5), lambda fig, spec, D: (panel_switch_curves(fig, spec, D), None)),
    ("j", (2.4, 2.0), _single(panel_switch_split)),
]


def export_panels(D, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    for L, size, build in PANELS:
        fig = plt.figure(figsize=size)
        spec = GridSpec(1, 1, figure=fig, left=0.14, right=0.86, top=0.84, bottom=0.2)[0]
        build(fig, spec, D); save_dual(fig, outdir / f"fig4{L}", write_manifest=False); plt.close(fig)
    print(f"-> {len(PANELS)} panels in {outdir}")


def main():
    apply(base_pt=6.5)
    F2.MODALITY["Isoform"] = C_ISO
    D = load()
    fig = plt.figure(figsize=(7.2, 11.6))
    outer = GridSpec(6, 1, figure=fig, height_ratios=[1.55, 1.45, 2.0, 1.6, 1.45, 1.75], hspace=0.72,
                     left=0.07, right=0.91, top=0.978, bottom=0.035)
    g1 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], width_ratios=[1.75, 0.95, 1.05], wspace=0.75)
    g4 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[3], width_ratios=[1.0, 0.2], wspace=0.1)
    g5 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[4], width_ratios=[1.0, 0.22], wspace=0.1)
    g6 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[5], width_ratios=[0.95, 1.75, 0.95], wspace=0.62)
    where = dict(a=g1[0], b=g1[1], c=g1[2], d=outer[1], e=outer[2], f=g4[0], g=g5[0], h=g6[0], i=g6[1], j=g6[2])
    nudge = dict(a=(-0.03, 0.015), b=(-0.06, 0.015), c=(-0.07, 0.015), d=(-0.03, 0.02), e=(-0.06, 0.03),
                 f=(-0.06, 0.03), g=(-0.06, 0.03), h=(-0.07, 0.03), i=(-0.06, 0.03), j=(-0.1, 0.03))
    res = {}
    for L, _, build in PANELS:
        ax, out = build(fig, where[L], D); res[L] = out; F2.letter(ax, L, *nudge[L])
    F2.draw_letters(fig)
    manifest = dict(fit=str(FIT), domains=NAMES, domain_composition=D["desc"], split_whole=res["c"], features=res["e"],
                    per_type_along_rho=res["f"], isoform_only=res["g"], switch_23=res["h"], switch_split=res["j"],
                    sources=[str(FIT), str(TAB)])
    save_dual(fig, OUT, extra=manifest); save_dual(fig, FINAL / "fig4_isoform", extra=manifest); plt.close(fig)
    export_panels(D, FINAL / "panels")
    print(json.dumps(dict(c=res["c"], g=res["g"], h=res["h"]), indent=1, default=str))


if __name__ == "__main__":
    main()
