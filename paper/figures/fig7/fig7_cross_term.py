"""Figure 7 — Saturn fits every figure dataset better than per-modality GASTON, and its cross pathways say where, and
for which genes, one modality's axis explains the other's variance.

Datasets = the five figure fits (Fig 2 LPC10 RNA+ATAC, Fig 3 melanoma RNA+ATAC, Fig 4 MERFISH RNA+isoform, Fig 5
Sprox-seq B1 RNA+protein, Fig 6 melanoma RNA+Hi-C).
  part 1
  a  whole-section R2 per modality: Saturn (one fit, seed 42) vs default GASTON fitted on that modality alone, same
     target (best-loss of 30 restarts; range = all restarts)          [scripts/gaston_default_r2.py]
  b  Saturn R2 split into the own-axis part and the cross-term gain (cross pathways held at their mean)
  c  per dataset: rho_RNA | rho_X | where the other axis explains RNA (local cross gain, 25-NN) | where rho_RNA explains X
  part 2
  d  RNA genes whose expression the other axis explains beyond rho_RNA (hinge model, partial F, BH 5%, dR2 >= 0.01)
  e  potential regulation, ATAC: a cross-explained gene's own +-50 kb accessibility follows the same rho_ATAC profile,
     vs the gene paired with a random other gene's window
  f  potential regulation, Hi-C: cross-explained genes are long; length does not predict the own-axis fit
  g  one example gene per dataset: expression | its cross component (the rho_X part of the fit) at every spot |
     partial residual (expression minus own-axis part) along rho_X, with the cis window for ATAC
  h  what the rho_ATAC-steep zone marks (the ATAC datasets): where rho_ATAC changes faster than rho_RNA (each relative
     to its own median) | the top switching cell type there | the cell types and TFs whose boundary is sharpest in
     the ATAC-steep zone vs the RNA-steep zone (top 3 each way, rotation p < 0.05): cortical neurons in the LPC
     section; melanocytic tumour / tolerant CD8 (IRF8, TEAD1) in melanoma, while invasive / neural-crest tumour
     switch where rho_RNA is steep                                            [scripts/saturn_steep_followup.py]
  i  what the ATAC cross term reflects: share of each gene's dR2 from rho_ATAC absorbed by its own cis chromatin, TF
     activity (chromVAR), cell-type composition, all three, or a rotated (spatially scrambled) block of equal width
     [scripts/saturn_cross_mechanism.py]
  j  transect across the main rho_ATAC-steep zone, along rho_ATAC's gradient: both axes and the RNA gain, and the top
     switching cell type and TF from h: the top cell type rises across the zone
     [scripts/saturn_steep_transition.py]
  supplement S7: the same gradient comparison (direction and steepness) for all five datasets -- no consistent
     relation with the local cross gain under a rotation null
  supplement S8: how far the steepness reading goes (LPC, melanoma): steepness vs RNA gain by quintile and away from
     the edge; depth / edge partial correlation; CODEX protein boundaries (LPC, same section, never fitted) follow
     rho_RNA, not rho_ATAC; the model's steepness RATIO is not reproduced by the raw targets' gradients
Analysis: Hole/gaston_mix_mm/scripts/saturn_cross_term.py
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
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/saturn_repo/src")       # Saturn package (saturn_repo)
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/fig2")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
from paper_style import apply, save_dual, save_panel_svg, MODALITY, CMAP, PALETTE, METHOD
import fig2_lpc as F2
from saturn import crossterm as CT

H = "/mnt/HDD2B/Users/yancui"
R = Path(f"{H}/Hole/gaston_mix_mm/results/real_saturn"); X7 = R / "saturn_cross_term"
OUT = Path(f"{H}/Saturn/published_figures/fig7/fig7_cross_term")
KEYS = ["lpc10", "melanoma", "merfish", "sprox", "hic"]
XMOD = dict(lpc10="ATAC", melanoma="ATAC", merfish="Isoform", sprox="Protein", hic="HiC")
XLAB = dict(ATAC="ATAC", Isoform="isoform", Protein="protein", HiC="Hi-C")
TITLE = dict(lpc10="Fig 2 · LPC lesion", melanoma="Fig 3 · melanoma", merfish="Fig 4 · brain MERFISH",
             sprox="Fig 5 · tonsil Sprox-seq", hic="Fig 6 · melanoma Hi-C")
EX = dict(lpc10="Trem2", melanoma="IFI44L", merfish="Calb2", sprox="IGHG1", hic="NRG3")
DSNAME = dict(lpc10="lpc10_fov", melanoma="melanoma_tissue", merfish="rtt_merfish_s5_iso100_sv_pca_shrunk",
              sprox="sproxseq_B1_p46_dr", hic="mel_hic_abins_dr50_51_rnadr")
DR2_MIN = 0.01                      # 1% of a gene's variance: with thousands of spots tiny gains pass FDR alone
C_SAT, C_GAS = METHOD.get("Saturn", "#14244A"), "#9AA2AA"


def mcol(m):
    return MODALITY.get(m, "#6A7FA6")


def sub(m):
    return f"ρ$_{{\\mathrm{{{m}}}}}$"


# --------------------------------------------------------------------------- #
def examples():
    """per-spot expression, own and cross parts of the example gene (and its +-50 kb window for ATAC); cached"""
    f = X7 / "fig7_examples.npz"
    if f.exists():
        return dict(np.load(f, allow_pickle=True))
    import saturn_cross_term as SC                      # dataset loaders (rna, cis windows)
    from saturn import crossterm as CT
    out = {}
    for k in KEYS:
        T = pd.read_csv(X7 / f"{k}_spots.csv", index_col=0); bc = list(T.index.astype(str)); g = EX[k]
        Xm, genes = SC.rna(k, bc); col = np.asarray(genes) == g; assert col.sum() >= 1, (k, g)
        col = col & (np.cumsum(col) == 1)
        y = CT.logcpm(Xm, col)[:, 0]; ys = [y]
        if XMOD[k] == "ATAC":
            _, W, _ = SC.cis_test(k, bc, np.array([g]), np.zeros((len(bc), 1)), np.array([True])); ys.append(W[:, 0])
        for j, v in enumerate(ys):
            Bo, Bx = CT.hinge(T.rho_RNA.values), CT.hinge(T.rho_X.values); N = len(v)
            A = np.c_[np.ones(N), Bo, Bx]; c = np.linalg.lstsq(A, v, rcond=None)[0]; no = 1 + Bo.shape[1]
            own, cr = A[:, :no] @ c[:no], Bx @ c[no:]
            nm = "gene" if j == 0 else "cis"
            out[f"{k}|{nm}|y"], out[f"{k}|{nm}|own"], out[f"{k}|{nm}|cross"] = v, own, cr - cr.mean()
    np.savez_compressed(f, **out)
    return out


STEEP_CASES = ("lpc10", "melanoma")
STEEP_DS = dict(lpc10="lpc10_fov", melanoma="melanoma_tissue")


def steepness(T):
    """log((|grad rho_X| / median) / (|grad rho_RNA| / median)): > 0 where the other axis changes faster"""
    return CT.steepness(T)


def steep_cases():
    """per case: depth-controlled steepness-gain rho, and the cross-explained genes ranked by how much of their cross
    component's variance sits inside the steepest-rho_X quintile (variance there / overall variance); cached"""
    f = X7 / "fig7_steep.json"
    if f.exists():
        return json.load(open(f))
    from scipy.stats import spearmanr
    import saturn_cross_term as SC
    res = CT.rank_resid                                   # ranks with the ranks of log depth regressed out
    out = {}
    for k in STEEP_CASES:
        T = pd.read_csv(X7 / f"{k}_spots.csv", index_col=0); bc = list(T.index.astype(str)); lg = steepness(T)
        Xm, _ = SC.rna(k, bc); dep = np.log(np.asarray(Xm.sum(1)).ravel())
        z = np.load(X7 / f"{k}_profiles.npz", allow_pickle=True); P, g = z["prof"], z["genes"].astype(str)
        G = pd.read_csv(X7 / f"{k}_genes.csv").set_index("gene").reindex(g)
        keep = ((G.q < 0.05) & (G.dR2_cross >= 2 * DR2_MIN) & (G.detect >= 0.2)).values \
            & ~pd.Series(g).str.match(r"ENSG|LINC|MIR\d|Gm\d|.*Rik$|.*-AS|.*-DT$|mt-|MT-|RP[LS]\d|GAPDHP").values
        P, g = P[:, keep], g[keep]; top = lg >= np.quantile(lg, 0.8)
        vr = P[top].var(0) / P.var(0); o = np.argsort(-vr)
        out[k] = dict(rho=float(spearmanr(lg, T.gain_RNA)[0]), rho_depth=float(spearmanr(res(lg, dep), res(T.gain_RNA.values, dep))[0]),
                      genes=list(g[o[:12]]), var_ratio=[float(v) for v in vr[o[:12]]], n_genes=int(keep.sum()))
    json.dump(out, open(f, "w"), indent=1)
    return out


def load():
    D = dict(sp={k: pd.read_csv(X7 / f"{k}_spots.csv", index_col=0) for k in KEYS},
             S={k: json.load(open(X7 / f"{k}_summary.json")) for k in KEYS},
             G={k: pd.read_csv(X7 / f"{k}_genes.csv") for k in KEYS},
             cis={k: pd.read_csv(X7 / f"{k}_cis.csv") for k in ("lpc10", "melanoma")},
             L=pd.read_csv(X7 / "hic_length.csv"),
             bench=pd.read_csv(R / "gaston_default_r2" / "summary.csv"),
             rest=pd.read_csv(R / "gaston_default_r2" / "restarts.csv"))
    D["ex"] = examples(); D["steep"] = steep_cases()
    D["tr"] = {k: dict(F=pd.read_csv(X7 / f"{k}_transition_fits.csv").set_index("feature"),
                       P=pd.read_csv(X7 / f"{k}_transition_profiles.csv"),
                       sp=pd.read_csv(X7 / f"{k}_transition_spots.csv"),
                       S=json.load(open(X7 / f"{k}_transition_summary.json"))) for k in STEEP_CASES}
    FU = X7 / "steep_followup"
    D["fu"] = dict(S=json.load(open(FU / "followup.json")), codex=pd.read_csv(FU / "lpc10_codex.csv"),
                   sw={k: pd.read_csv(FU / f"{k}_zone_switch.csv") for k in STEEP_CASES})
    D["mech"] = {k: dict(G=pd.read_csv(X7 / f"{k}_mech_genes.csv"), dr=pd.read_csv(X7 / f"{k}_mech_drivers.csv"),
                         B=pd.read_csv(X7 / f"{k}_mech_boundary.csv"), S=json.load(open(X7 / f"{k}_mech_summary.json")))
                 for k in STEEP_CASES}
    return D


# --------------------------------------------------------------------------- #
def panel_bench(ax, D):
    B = D["bench"]; key = {v: k for k, v in DSNAME.items()}; y = 0; ticks = []
    for k in KEYS:
        for _, r in B[B.dataset == DSNAME[k]].iterrows():
            rs = D["rest"][(D["rest"].dataset == r.dataset) & (D["rest"].modality == r.modality)].R2
            ax.plot([rs.min(), rs.max()], [y, y], color=C_GAS, lw=2.2, alpha=0.5, solid_capstyle="round")
            ax.scatter(r.GASTON_best_loss, y, s=14, facecolor="white", edgecolor="#5c6670", lw=0.8, zorder=3)
            ax.scatter(r.Saturn, y, s=18, color=mcol(r.modality), edgecolor="#1a1a1a", lw=0.3, zorder=4)
            ticks.append((y, f"{XLAB.get(r.modality, r.modality)}", mcol(r.modality))); y += 1
        ax.text(1.0, y - 1.5, TITLE[k].split(" · ")[1], transform=ax.get_yaxis_transform(), fontsize=5.2, ha="right",
                va="center", color="#555555")
        y += 0.6
    ax.set_yticks([t[0] for t in ticks]); ax.set_yticklabels([t[1] for t in ticks], fontsize=5.5)
    for tl, t in zip(ax.get_yticklabels(), ticks): tl.set_color(t[2])
    ax.invert_yaxis(); ax.set_xlabel("whole-section R²", labelpad=1); ax.set_xlim(0.1, 0.9)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", color="#777777", mec="#1a1a1a", mew=0.3, ms=4, label="Saturn (one fit)"),
                       Line2D([], [], ls="", marker="o", mfc="white", mec="#5c6670", ms=4, label="GASTON, best-loss restart"),
                       Line2D([], [], color=C_GAS, lw=2.2, alpha=0.5, label="GASTON, 30 restarts")],
              fontsize=5, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, handletextpad=0.3,
              columnspacing=0.8)
    ax.set_title("Saturn vs per-modality GASTON (same targets)", fontsize=6.2, pad=2)
    wins = int((B.Saturn > B.GASTON_best_loss).sum())
    return dict(saturn_wins=f"{wins}/{len(B)}", rows=B[["dataset", "modality", "Saturn", "GASTON_best_loss"]].to_dict("records"))


def panel_split(ax, D):
    x = 0; ticks = []; out = {}
    for k in KEYS:
        S = D["S"][k]
        for nm, m in (("RNA", "RNA"), ("X", XMOD[k])):
            f_, o_ = S[f"R2_{nm}"]["full"], S[f"R2_{nm}"]["own_only"]
            ax.bar(x, o_, 0.7, color=mcol(m), alpha=0.45, edgecolor="#1a1a1a", lw=0.3)
            ax.bar(x, f_ - o_, 0.7, bottom=o_, color=mcol(m), edgecolor="#1a1a1a", lw=0.3)
            ticks.append((x, XLAB.get(m, m), mcol(m))); out[f"{k}|{m}"] = dict(own=round(o_, 3), cross_gain=round(f_ - o_, 3)); x += 1
        ax.text(x - 1.5, -0.19, TITLE[k].split(" · ")[1].replace(" ", "\n", 1), transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=5, color="#555555")
        x += 0.5
    ax.set_xticks([t[0] for t in ticks]); ax.set_xticklabels([t[1] for t in ticks], rotation=50, ha="right", fontsize=5.3)
    for tl, t in zip(ax.get_xticklabels(), ticks): tl.set_color(t[2])
    ax.set_ylabel("Saturn R²", labelpad=1); ax.set_ylim(0, 0.9)
    ax.legend(handles=[Patch(fc="#9aa2aa", alpha=0.45, ec="#1a1a1a", lw=0.3, label="own axis"),
                       Patch(fc="#5c6670", ec="#1a1a1a", lw=0.3, label="+ other axis (cross term)")],
              fontsize=5, frameon=False, loc="upper right")
    ax.set_title("what the cross term adds", fontsize=6.2, pad=2)
    return out


def panel_gain_maps(fig, spec, D):
    """one row per dataset: rho_RNA | rho_X | local cross gain of RNA (from rho_X) | of X (from rho_RNA), shared scale per row"""
    gs = GridSpecFromSubplotSpec(len(KEYS), 1, subplot_spec=spec, hspace=0.45); first, out = None, {}
    for i, k in enumerate(KEYS):
        T = D["sp"][k]; xy = T[["x", "y"]].values; m = XMOD[k]
        g4 = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[i], wspace=0.55)
        ax = [fig.add_subplot(g4[j]) for j in range(4)]; first = first or ax[0]
        s = 3.0 if len(T) < 2500 else 1.6
        for a, col, mm in ((ax[0], "rho_RNA", "RNA"), (ax[1], "rho_X", m)):
            lo, hi = np.percentile(T[col], [1, 99]); sc = F2.spatial(a, xy, T[col], cmap=CMAP["score"], vmin=lo, vmax=hi, s=s)
            F2.cbar(fig, a, sc, "isodepth", ticks=[])
            a.set_title(sub(mm), fontsize=6.2, pad=2, color=mcol(mm))
        hi = max(np.percentile(T.gain_RNA, 98), np.percentile(T.gain_X, 98))
        for a, col, t, c in ((ax[2], "gain_RNA", f"RNA explained by {sub(m)}", "RNA"), (ax[3], "gain_X", f"{XLAB[m]} explained by {sub('RNA')}", m)):
            sc = F2.spatial(a, xy, T[col].clip(0, None), cmap=CMAP["recovery"], vmin=0, vmax=hi, s=s)
            F2.cbar(fig, a, sc, "local R²\ngain", on_top=True)
            a.set_title(t, fontsize=5.6, pad=2, color=mcol(c))
        ax[0].text(-0.08, 0.5, TITLE[k], transform=ax[0].transAxes, rotation=90, ha="right", va="center", fontsize=6.0,
                   fontweight="bold")
        out[k] = dict(mean_gain_RNA=round(float(T.gain_RNA.mean()), 3), mean_gain_X=round(float(T.gain_X.mean()), 3))
    return first, out


def panel_ngenes(ax, D):
    out = {}
    for i, k in enumerate(KEYS):
        G = D["G"][k]; hit = (G.q < 0.05) & (G.dR2_cross >= DR2_MIN); f = hit.mean(); m = XMOD[k]
        ax.barh(i, 100 * f, 0.65, color=mcol(m), edgecolor="#1a1a1a", lw=0.3)
        ax.text(100 * f + 1, i, f"{int(hit.sum())}/{len(G)}", va="center", fontsize=5)
        out[k] = dict(n=int(hit.sum()), tested=len(G), n_fdr_only=int((G.q < 0.05).sum()))
    ax.set_yticks(range(len(KEYS))); ax.set_yticklabels([f"{TITLE[k].split(' · ')[1]} ({sub(XMOD[k])})" for k in KEYS], fontsize=5.3)
    ax.invert_yaxis(); ax.set_xlabel("RNA genes explained by the other axis\nbeyond ρ$_{\\mathrm{RNA}}$ (%; FDR 5%, ΔR² ≥ 0.01)", labelpad=1)
    ax.set_xlim(0, 50); ax.set_title("cross-explained genes", fontsize=6.2, pad=2)
    return out


def panel_cis(ax, D):
    out = {}
    for i, k in enumerate(("lpc10", "melanoma")):
        c = D["cis"][k].concordance.values; S = D["S"][k]["cis"]
        vp = ax.violinplot(c, positions=[i], widths=0.7, showextrema=False)
        for b in vp["bodies"]: b.set_facecolor(mcol("ATAC")); b.set_alpha(0.55); b.set_edgecolor("#1a1a1a"); b.set_linewidth(0.3)
        ax.scatter(i, c.mean(), s=14, color="#1a1a1a", zorder=3)
        ax.errorbar(i + 0.38, S["ctrl_mean"], yerr=2 * S["ctrl_sd"], fmt="o", mfc="white", mec="#5c6670", ecolor="#5c6670", ms=3.5, lw=0.7)
        ax.text(i, 1.08, f"+{S['z']:.0f} SD", ha="center", fontsize=5.2)
        out[k] = dict(n=S["n_genes"], mean=round(S["mean_obs"], 3), control=round(S["ctrl_mean"], 3), z=round(S["z"], 1))
    ax.axhline(0, color="#9aa2aa", lw=0.4); ax.set_ylim(-1, 1.18)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["LPC lesion", "melanoma"], fontsize=5.5)
    ax.set_ylabel("gene vs own ±50 kb chromatin\n(ρ$_{\\mathrm{ATAC}}$ profile, r)", labelpad=1)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", color="#1a1a1a", ms=3.5, label="cis window (mean)"),
                       Line2D([], [], ls="", marker="o", mfc="white", mec="#5c6670", ms=3.5, label="random window ±2 SD")],
              fontsize=5, frameon=False, loc="lower left", handletextpad=0.2)
    ax.set_title("cis chromatin follows ρ$_{\\mathrm{ATAC}}$", fontsize=6.2, pad=2)
    return out


def panel_length(ax, D):
    L = D["L"].copy(); L["kb"] = L.length / 1e3; L = L[L.kb > 0]
    top = L.nlargest(200, "dR2_cross").kb; S = D["S"]["hic"]["length"]
    parts = ax.boxplot([L.kb, top], positions=[0, 1], widths=0.55, showfliers=False, patch_artist=True,
                       medianprops=dict(color="#1a1a1a", lw=0.8), whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5))
    for b, c in zip(parts["boxes"], ("#d5d9dc", mcol("HiC"))):
        b.set_facecolor(c); b.set_edgecolor("#1a1a1a"); b.set_linewidth(0.4)
    ax.set_yscale("log"); ax.set_xticks([0, 1]); ax.set_xticklabels(["all tested", "top 200\ncross genes"], fontsize=5.5)
    ax.set_ylabel("gene length (kb)", labelpad=1)
    ax.text(0.03, 0.98, f"median {S['median_all_kb']:.0f} → {S['median_top_kb']:.0f} kb\np {S['p']:.0e}\n"
            f"length vs cross gain ρ {S['rho_cross']:.2f}\nlength vs own-axis R² ρ {S['rho_own']:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=5)
    ax.set_ylim(top=ax.get_ylim()[1] * 40)
    ax.set_title("ρ$_{\\mathrm{HiC}}$ explains long genes", fontsize=6.2, pad=2)
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in S.items()}


def panel_examples(fig, spec, D):
    """row = example gene: log expression | cross component at every spot | partial residual along rho_X (+ cis window)"""
    gs = GridSpecFromSubplotSpec(len(KEYS), 1, subplot_spec=spec, hspace=0.75); first, out = None, {}
    E = D["ex"]
    for i, k in enumerate(KEYS):
        T = D["sp"][k]; xy = T[["x", "y"]].values; m = XMOD[k]; g = EX[k]; s = 3.0 if len(T) < 2500 else 1.6
        g3 = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[i], width_ratios=[1, 1, 1.3], wspace=0.75)
        a1, a2, a3 = (fig.add_subplot(g3[j]) for j in range(3)); first = first or a1
        y, own, cr = E[f"{k}|gene|y"], E[f"{k}|gene|own"], E[f"{k}|gene|cross"]
        lo, hi = np.percentile(y, [2, 99]); sc = F2.spatial(a1, xy, y, cmap=CMAP["expression"], vmin=lo, vmax=hi, s=s)
        F2.cbar(fig, a1, sc, "log\nCP10k", on_top=True)
        a1.set_title(f"{g}", fontsize=6.4, pad=2, color=mcol("RNA"), loc="left", fontstyle="italic")
        a1.text(0, 1.3, TITLE[k], transform=a1.transAxes, fontsize=6.0, fontweight="bold", va="bottom")
        a1.text(0.5, -0.04, "expression", transform=a1.transAxes, ha="center", va="top", fontsize=5.3)
        v = np.percentile(np.abs(cr), 99); sc = F2.spatial(a2, xy, cr, cmap=CMAP["score"], vmin=-v, vmax=v, s=s)
        F2.cbar(fig, a2, sc, "log\nCP10k", on_top=True)
        a2.set_title(f"part explained by {sub(m)}", fontsize=5.8, pad=2, color=mcol(m), loc="left")
        a2.text(0.5, -0.04, "cross component", transform=a2.transAxes, ha="center", va="top", fontsize=5.3)
        d = T.rho_X.values; nb = 25; q = np.quantile(d, np.linspace(0, 1, nb + 1)); bi = np.clip(np.digitize(d, q[1:-1]), 0, nb - 1)
        xb = np.array([d[bi == j].mean() for j in range(nb)]); o = np.argsort(d)
        series = [("gene", g, mcol("RNA"))] + ([("cis", "±50 kb chromatin", mcol("ATAC"))] if f"{k}|cis|y" in E else [])
        for nm, lab, col in series:
            r = E[f"{k}|{nm}|y"] - E[f"{k}|{nm}|own"]; c = E[f"{k}|{nm}|cross"]
            rb = np.array([r[bi == j].mean() for j in range(nb)]); mu, sd = rb.mean(), rb.std() + 1e-12
            a3.scatter(xb, (rb - mu) / sd, s=6, color=col, lw=0, alpha=0.8)
            a3.plot(d[o], (c[o] + r.mean() - mu) / sd, color=col, lw=1.3, label=lab)
        a3.set_xlabel(sub(m), color=mcol(m), labelpad=1); a3.set_ylabel("partial residual (z)", labelpad=1)
        a3.tick_params(labelsize=5.3, pad=1)
        Gk = D["G"][k].set_index("gene").loc[g]
        a3.set_title(f"ΔR² from {sub(m)} {Gk.dR2_cross:.2f} (own-axis R² {Gk.R2_own:.2f})", fontsize=5.6, pad=2)
        a3.legend(fontsize=5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), handlelength=1.2)
        out[k] = dict(gene=g, dR2_cross=round(float(Gk.dR2_cross), 3), R2_own=round(float(Gk.R2_own), 3), q=float(Gk.q))
        if f"{k}|cis|y" in E:
            out[k]["cis_concordance"] = round(float(D["cis"][k].set_index("gene").concordance.get(g, np.nan)), 3)
    return first, out




def short(f):
    """compact cell-type / TF label"""
    t = f.split(":", 1)[-1]
    for a, b in (("intratelencephalic projecting glutamatergic neuron", "IT neuron"),
                 ("corticothalamic-projecting glutamatergic cortical neuron", "CT neuron"),
                 ("extratelencephalic projecting glutamatergic cortical neuron", "ET neuron"),
                 ("glutamatergic cortical neuron", "neuron"), ("GABAergic cortical interneuron", "interneuron"),
                 ("Melanoma_NeuralCrest", "neural-crest tumour"), ("Melanoma_", ""), ("Deletional tolerance", "tolerant"), ("Precursor exhaustion", "precursor-exh. T"),
                 ("oligodendrocyte precursor cell", "OPC"), ("cell", "")):
        t = t.replace(a, b)
    t = t.strip(); t = t + " tumour" if t in ("Melanocytic", "Transitory", "Invasive") else t
    return t + (" (TF)" if f.startswith("TF:") else "")


def top_switch(D, k, n=3, alpha=0.05):
    """cell types / TFs whose boundary is sharpest in the ATAC-steep zone (top n) and in the RNA-steep zone (top n)"""
    W = D["fu"]["sw"][k]; W = W[W.p_rot < alpha]
    return pd.concat([W[W.log2_sharpness > 0].nlargest(n, "log2_sharpness"), W[W.log2_sharpness < 0].nsmallest(n, "log2_sharpness")])


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def panel_steep(fig, spec, D):
    """two cases: where rho_ATAC is steeper | the top switching cell type | what switches sharply in each zone"""
    gs = GridSpecFromSubplotSpec(len(STEEP_CASES), 1, subplot_spec=spec, hspace=0.55); first, out = None, {}
    for i, k in enumerate(STEEP_CASES):
        T = D["sp"][k]; xy = T[["x", "y"]].values; m = XMOD[k]; s = 2.4 if len(T) < 2500 else 1.2
        lg = steepness(T); W = top_switch(D, k)
        g3 = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[i], width_ratios=[1, 1, 1.35], wspace=1.25)
        a0, a1, a2 = (fig.add_subplot(g3[j]) for j in range(3)); first = first or a0
        v = np.percentile(np.abs(lg), 98)
        sc = F2.spatial(a0, xy, lg, cmap=CMAP["score"], vmin=-v, vmax=v, s=s); F2.cbar(fig, a0, sc, "log steepness\nATAC / RNA", on_top=True)
        a0.set_title(f"{sub(m)} steeper (red) / {sub('RNA')} steeper (blue)", fontsize=5.6, pad=2, loc="left")
        a0.text(0, 1.2, TITLE[k], transform=a0.transAxes, fontsize=6.2, fontweight="bold", va="bottom")
        f0 = W.iloc[0]; C = (MC_comp(k, list(T.index.astype(str))) if f0.kind == "type" else MC_tf(k, list(T.index.astype(str))))[f0.feature]
        sc = F2.spatial(a1, xy, C.values, cmap=CMAP["expression"] if f0.kind == "type" else CMAP["score"], s=s,
                        vmin=np.nanpercentile(C, 2), vmax=np.nanpercentile(C, 98))
        F2.cbar(fig, a1, sc, "proportion" if f0.kind == "type" else "activity (z)", on_top=True)
        top = lg >= np.quantile(lg, 0.8)
        a1.scatter(*xy[top].T, s=s * 0.18, c="#1a1a1a", lw=0, rasterized=True)
        a1.set_title(f"{short(f0.kind + ':' + f0.feature)}; black = {sub(m)}-steep zone", fontsize=5.6, pad=2, loc="left")
        y = np.arange(len(W)); cc = np.where(W.log2_sharpness > 0, mcol(m), mcol("RNA"))
        a2.barh(y, W.log2_sharpness, 0.62, color=cc, edgecolor="none"); a2.axvline(0, color="#1a1a1a", lw=0.5)
        for yy, (x, p) in enumerate(zip(W.log2_sharpness, W.p_rot)):
            a2.text(x + (0.04 if x > 0 else -0.04), yy, stars(p), ha="left" if x > 0 else "right", va="center", fontsize=5.5)
        a2.set_yticks(y); a2.set_yticklabels([short(f"{kd}:{f}") for kd, f in zip(W.kind, W.feature)], fontsize=5.5); a2.invert_yaxis()
        lim = np.abs(W.log2_sharpness).max() * 1.3; a2.set_xlim(-lim if (W.log2_sharpness < 0).any() else -0.15, lim)
        a2.set_xlabel(f"boundary sharpness, {sub(m)}-steep vs\n{sub('RNA')}-steep zone (log2)", labelpad=1)
        a2.set_title("what switches in each zone (rotation p)", fontsize=5.8, pad=2)
        out[k] = {f"{kd}:{f}": dict(log2=round(float(x), 2), p_rot=round(float(p), 3))
                  for kd, f, x, p in zip(W.kind, W.feature, W.log2_sharpness, W.p_rot)}
    return first, out


def MC_comp(k, bc):
    import saturn_cross_mechanism as MC
    return MC.composition(k, bc)


def MC_tf(k, bc):
    import saturn_cross_mechanism as MC
    return MC.tf_activity(k, bc)


BLOCKS = [("cis", "own cis\nchromatin"), ("TF", "TF activity\n(chromVAR)"), ("comp", "cell-type\ncomposition"),
          ("all", "all three"), ("null", "rotated\nblock (null)")]
MECH_EX = dict(lpc10=["Trem2", "C1qa", "Ctsd", "Apoe", "Ttr", "Enpp2"], melanoma=["IFI44L", "MX1", "TNFSF10", "COL1A1", "DUSP6"])


def panel_mech(fig, spec, D):
    """left: share of the rho_ATAC term absorbed by each source, across all cross genes; right: example genes"""
    g2 = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[1.0, 1.45], wspace=0.5)
    ax = fig.add_subplot(g2[0]); out = {}
    w = 0.36
    for i, k in enumerate(STEEP_CASES):
        G = D["mech"][k]["G"]; pos = np.arange(len(BLOCKS)) + (i - 0.5) * (w + 0.04)
        data = [G[f"absorbed_{b}"].dropna().clip(-0.5, 1.2).values for b, _ in BLOCKS]
        bp = ax.boxplot(data, positions=pos, widths=w, showfliers=False, patch_artist=True,
                        medianprops=dict(color="#1a1a1a", lw=0.8), whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5))
        for bx in bp["boxes"]:
            bx.set_facecolor(mcol("ATAC")); bx.set_alpha(0.85 if i == 0 else 0.4); bx.set_edgecolor("#1a1a1a"); bx.set_linewidth(0.4)
        out[k] = {b: round(float(np.nanmedian(G[f"absorbed_{b}"])), 3) for b, _ in BLOCKS} | dict(n=len(G))
    ax.axhline(0, color="#9aa2aa", lw=0.4)
    ax.set_xticks(range(len(BLOCKS))); ax.set_xticklabels([l for _, l in BLOCKS], fontsize=5)
    ax.set_ylabel("share of the gene's ΔR² from ρ$_{\\mathrm{ATAC}}$\nabsorbed by the source", labelpad=1)
    ax.set_ylim(-0.5, 1.4); ax.set_yticks(np.arange(-0.4, 1.01, 0.2))
    ax.legend(handles=[Patch(fc=mcol("ATAC"), alpha=0.85, ec="#1a1a1a", lw=0.4, label=f"LPC lesion ({out['lpc10']['n']} genes)"),
                       Patch(fc=mcol("ATAC"), alpha=0.4, ec="#1a1a1a", lw=0.4, label=f"melanoma ({out['melanoma']['n']} genes)")],
              fontsize=5, frameon=False, loc="upper left", ncol=2)
    ax.set_title("what the ATAC cross term reflects", fontsize=6.2, pad=2)
    # right: example genes heatmap + named drivers
    rows, lab = [], []
    for k in STEEP_CASES:
        G = D["mech"][k]["G"].set_index("gene"); dr = D["mech"][k]["dr"].set_index("gene")
        for g in MECH_EX[k]:
            rows.append([G.loc[g, f"absorbed_{b}"] for b, _ in BLOCKS])
            t1 = dr.loc[g, "type1"].rsplit(" (", 1); f1 = dr.loc[g, "tf1"].rsplit(" (", 1)
            lab.append((k, g, f"{short(t1[0])} ({t1[1]}; {f1[0]} ({f1[1]}", G.loc[g, "dR2_ATAC"]))
    Mx = np.array(rows, float)
    a2 = fig.add_subplot(g2[1])
    im = a2.imshow(Mx.clip(0, 1), cmap=CMAP["recovery"], vmin=0, vmax=1, aspect="auto")
    for (r, c), v in np.ndenumerate(Mx):
        a2.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=5, color="white" if v > 0.6 else "#1a1a1a")
    a2.set_xticks(range(len(BLOCKS))); a2.set_xticklabels([l for _, l in BLOCKS], fontsize=5)
    a2.xaxis.tick_top()
    a2.set_yticks(range(len(lab))); a2.set_yticklabels([f"{g}  (ΔR² {d:.2f})" for _, g, _, d in lab], fontsize=5.2)
    for tl in a2.get_yticklabels(): tl.set_fontstyle("italic")
    nl = len(MECH_EX["lpc10"]); a2.axhline(nl - 0.5, color="white", lw=2)
    for r, (k, g, t, _) in enumerate(lab):
        a2.text(len(BLOCKS) - 0.4, r, t, fontsize=5, va="center", ha="left")
    a2.text(len(BLOCKS) - 0.4, -1.25, "top cell type (r); top TF (r)", fontsize=5, ha="left", va="bottom", color="#555555")
    for r0, r1, k in ((0, nl, "lpc10"), (nl, len(lab), "melanoma")):
        a2.annotate(TITLE[k].split(" · ")[1], xy=(-0.3, (r0 + r1 - 1) / 2), xycoords=("axes fraction", "data"), rotation=90,
                    fontsize=5.6, fontweight="bold", ha="center", va="center")
    out["examples"] = {g: dict(zip([b for b, _ in BLOCKS], np.round(Mx[i], 2).tolist()), drivers=t)
                       for i, (_, g, t, _) in enumerate(lab)}
    return ax, out


TR_COL = ["#2B3F6B", "#C49A5A", "#7E9A7A", "#9A5A55", "#7E6A8C", "#5F9AA3"]
TR = dict(
    lpc10=dict(types=[("type:macrophage", "macrophage"), ("type:microglial cell", "microglia"),
                      ("type:leptomeningeal cell", "leptomeningeal"),
                      ("type:L2/3 intratelencephalic projecting glutamatergic neuron", "L2/3 IT neuron"),
                      ("type:L6 corticothalamic-projecting glutamatergic cortical neuron", "L6 CT neuron")],
               genes=[("gene:Trem2", "Trem2"), ("TF:SPI1", "SPI1 motif"), ("gene:Ttr", "Ttr"), ("gene:Enpp2", "Enpp2"),
                      ("gene:Pcsk2", "Pcsk2"), ("gene:Ntrk3", "Ntrk3")],
               ends=("lesion side", "cortex")),
    melanoma=dict(types=[("type:Melanoma_NeuralCrest", "neural-crest"), ("type:Melanoma_Invasive", "invasive"),
                         ("type:CAFs myCAF-like", "myCAF"), ("type:Melanoma_Melanocytic", "melanocytic"),
                         ("type:Deletional tolerance CD8", "tolerant CD8")],
                  genes=[("gene:PMEL", "PMEL"), ("gene:MITF", "MITF"), ("TF:IRF1", "IRF1 motif"), ("gene:BCAN", "BCAN"),
                         ("gene:FOSB", "FOSB"), ("gene:ANXA1", "ANXA1")],
                  ends=("tumour core", "section edge")))


def panel_transition(fig, spec, D):
    """per case: transect map | both axes + RNA gain along it | the top switching cell type and TF of h"""
    gs = GridSpecFromSubplotSpec(len(STEEP_CASES), 1, subplot_spec=spec, hspace=0.95); first, out = None, {}
    for i, k in enumerate(STEEP_CASES):
        T = D["sp"][k]; xy = T[["x", "y"]].values; Q = D["tr"][k]; F, P, St, S = Q["F"], Q["P"], Q["sp"], Q["S"]
        cfg = TR[k]; z0, z1 = S["s_zone"]; s = 2.4 if len(T) < 2500 else 1.2
        W = top_switch(D, k); W = W[W.log2_sharpness > 0]
        feats = [f"type:{W[W.kind == 'type'].feature.iloc[0]}", f"TF:{W[W.kind == 'TF'].feature.iloc[0]}"] if (W.kind == "TF").any() \
            else [f"type:{f}" for f in W[W.kind == "type"].feature.iloc[:2]]
        g3 = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[i], width_ratios=[0.8, 1, 1], wspace=0.6)
        a0 = fig.add_subplot(g3[0]); first = first or a0
        F2.spatial(a0, xy, None, colors="#e3e5e8", s=s)
        b = St.band.values; F2.spatial(a0, xy[b], St.s.values[b], cmap=CMAP["pseudotime"], s=s)
        zone = St.zone.values; a0.scatter(*xy[zone & b].T, s=s * 0.25, c="#1a1a1a", lw=0, rasterized=True)
        u = np.array(S["u"]); c0 = xy[zone].mean(0); L = (St.s.values[b].max() - St.s.values[b].min()) * S["spot_spacing"]
        st = c0 + u * St.s.values[b].min() * S["spot_spacing"]
        a0.annotate("", xy=st + u * L, xytext=st, arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#9A5A55", mutation_scale=6))
        a0.text(0.5, -0.04, "colour = position; black = steep zone", transform=a0.transAxes, ha="center", va="top", fontsize=5)
        a0.set_title("transect along ∂ρ$_{\\mathrm{ATAC}}$", fontsize=5.6, pad=2, loc="left")
        a0.text(0, 1.22, TITLE[k], transform=a0.transAxes, fontsize=6.2, fontweight="bold", va="bottom")
        def z(f):
            y = P[f]; return (y - y.mean()) / (y.std() + 1e-12)
        a1 = fig.add_subplot(g3[1])
        for f, lab, col, ls in (("axis:rho_ATAC", sub("ATAC"), mcol("ATAC"), "-"), ("axis:rho_RNA", sub("RNA"), mcol("RNA"), "-"),
                                ("axis:RNA gain", "RNA gain from " + sub("ATAC"), "#9aa2aa", "--")):
            a1.plot(P.s, z(f), ls, lw=1.3, color=col, label=lab)
        a1.set_title("both axes and the RNA gain\nalong the transect", fontsize=5.8, pad=2)
        a2 = fig.add_subplot(g3[2])
        for j, f in enumerate(feats):
            col = (mcol("ATAC"), "#5F9AA3")[j]
            a2.plot(P.s, z(f), "-", lw=1.2, color=col, label=short(f)); a2.scatter(P.s, z(f), s=2.5, color=col, lw=0)
        a2.set_title("top switching cell type / TF (from h)", fontsize=5.8, pad=2)
        for ax in (a1, a2):
            ax.axvspan(z0, z1, color=mcol("ATAC"), alpha=0.13, lw=0)
            ax.set_xlabel("position along transect (spots)", labelpad=1); ax.set_ylabel("z (along transect)", labelpad=1)
            ax.tick_params(labelsize=5.3, pad=1)
            ax.legend(fontsize=5.3, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.28), ncol=1, handlelength=1.4, labelspacing=0.2)
        a1.text(0.02, 0.97, cfg["ends"][0], transform=a1.transAxes, fontsize=5, va="top", color="#555555")
        a1.text(0.98, 0.03, cfg["ends"][1], transform=a1.transAxes, fontsize=5, ha="right", va="bottom", color="#555555")
        out[k] = {lab: dict(at=round(float(F.loc[f, "steep_at"]), 1), ci=[round(float(F.loc[f, "steep_lo"]), 1), round(float(F.loc[f, "steep_hi"]), 1)])
                  for f, lab in [("axis:rho_ATAC", "rho_ATAC"), ("axis:rho_RNA", "rho_RNA"), ("axis:RNA gain", "RNA gain")] + [(f, short(f)) for f in feats]} \
            | dict(zone=[round(z0, 1), round(z1, 1)], n_band=S["n_band"])
    return first, out


def panel_supp_gradient(fig, spec, D):
    """S7: misalignment and steepness maps + the signed steepness test with its rotation null"""
    gs = GridSpecFromSubplotSpec(len(KEYS), 4, subplot_spec=spec, hspace=0.5, wspace=0.6, width_ratios=[1, 1, 1, 1.2])
    first, out = None, {}
    for i, k in enumerate(KEYS):
        T = D["sp"][k]; xy = T[["x", "y"]].values; m = XMOD[k]; s = 3.0 if len(T) < 2500 else 1.6
        lg = np.log((T.grad_X / T.grad_X.median()) / (T.grad_RNA / T.grad_RNA.median()))
        a0 = fig.add_subplot(gs[i, 0]); first = first or a0
        sc = F2.spatial(a0, xy, T.misalign, cmap=CMAP["recovery"], vmin=0, vmax=1, s=s); F2.cbar(fig, a0, sc, "1 − |cos|", on_top=True)
        a0.set_title("direction difference", fontsize=5.6, pad=2)
        a0.text(-0.08, 0.5, TITLE[k], transform=a0.transAxes, rotation=90, ha="right", va="center", fontsize=5.8, fontweight="bold")
        a1 = fig.add_subplot(gs[i, 1]); v = np.percentile(np.abs(lg), 98)
        sc = F2.spatial(a1, xy, lg, cmap=CMAP["score"], vmin=-v, vmax=v, s=s); F2.cbar(fig, a1, sc, "log steepness\nX / RNA", on_top=True)
        a1.set_title(f"{sub(m)} steeper (red) / {sub('RNA')} steeper", fontsize=5.4, pad=2)
        a2 = fig.add_subplot(gs[i, 2]); hi = max(np.percentile(T.gain_RNA, 98), 1e-3)
        sc = F2.spatial(a2, xy, T.gain_RNA.clip(0, None), cmap=CMAP["recovery"], vmin=0, vmax=hi, s=s)
        F2.cbar(fig, a2, sc, "local R²\ngain", on_top=True); a2.set_title(f"RNA explained by {sub(m)}", fontsize=5.6, pad=2)
        a3 = fig.add_subplot(gs[i, 3]); S = D["S"][k]
        for nm, col, lab in (("RNA", mcol("RNA"), f"RNA gain vs {sub(m)} steeper"), ("X", mcol(m), f"{XLAB[m]} gain vs {sub('RNA')} steeper")):
            st = S[f"steeper_{nm}"]; a3.plot(range(1, 6), st["gain_by_quintile"], "-o", ms=2.5, lw=1, color=col,
                                              label=f"{lab}: ρ {st['rho']:.2f}, rot. p {st['p_rot']:.2f}")
            out[f"{k}|{nm}"] = dict(rho=round(st["rho"], 3), p_rot=round(st["p_rot"], 3))
        a3.set_xlabel("steepness quintile", labelpad=1); a3.set_ylabel("local R² gain", labelpad=1)
        a3.legend(fontsize=5, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.32), handlelength=1.0)
        a3.tick_params(labelsize=5.3, pad=1)
    return first, out


def panel_supp_steep(fig, spec, D):
    """S8: steepness vs RNA gain (quintiles; away from the edge) | CODEX protein boundaries vs each axis (LPC) |
    model steepness vs the raw targets' gradients"""
    FU = D["fu"]["S"]; CX = D["fu"]["codex"]; NM = dict(lpc10="LPC", melanoma="melanoma")
    gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=spec, hspace=0.75, wspace=0.55); out = {}
    ax = fig.add_subplot(gs[0, 0]); first = ax
    for j, k in enumerate(STEEP_CASES):
        St = D["S"][k]["steeper_RNA"]
        ax.plot(range(1, 6), FU[k]["gain_by_quintile"], "-o", ms=2.5, lw=1, color=(mcol("RNA"), mcol("ATAC"))[j],
                label=f"{NM[k]}: ρ {St['rho']:.2f}, rot. p {St['p_rot']:.2f}; partial ρ {FU[k]['controls']['rho_partial']:.2f}")
    ax.set_xticks(range(1, 6)); ax.set_xticklabels(["RNA\nsteeper", "", "", "", "ATAC\nsteeper"])
    ax.set_ylabel("RNA local R² gain", labelpad=1); ax.set_title("RNA gain by steepness quintile", fontsize=6, pad=2)
    ax.legend(fontsize=5.2, frameon=False, loc="upper left", handlelength=1.2)
    ax.text(0, -0.24, "partial ρ: log RNA / ATAC depth and edge distance removed", transform=ax.transAxes, fontsize=5, color="#555555")
    ax = fig.add_subplot(gs[0, 1]); F2.letter(ax, "b", -0.06, 0.02); cuts = [0, 3, 6, 10]
    for j, k in enumerate(STEEP_CASES):
        E = FU[k]["edge"]; cs = [c for c in cuts if f"interior>{c}" in E]; x = np.array([cuts.index(c) for c in cs]) + (j - 0.5) * 0.36
        r = [E[f"interior>{c}"] for c in cs]
        ax.bar(x, [w["rho"] for w in r], 0.34, color=(mcol("RNA"), mcol("ATAC"))[j], label=NM[k])
        for xi, w in zip(x, r):
            ax.text(xi, w["rho"] + 0.015, f"{w['p_rot']:.2f}", ha="center", va="bottom", fontsize=5)
        out[f"edge|{k}"] = {c: dict(n=w["n"], rho=round(w["rho"], 3), p_rot=round(w["p_rot"], 3)) for c, w in zip(cs, r)}
    ax.set_xticks(range(4)); ax.set_xticklabels([">0", ">3", ">6", ">10"]); ax.set_ylim(0, 0.85)
    ax.set_xlabel("spots kept: > n spots from the edge", labelpad=1); ax.set_ylabel("ρ, steepness vs RNA gain", labelpad=1)
    ax.set_title("not an edge effect (numbers = rotation p)", fontsize=6, pad=2); ax.legend(fontsize=5.2, frameon=False, loc="upper left")
    ax = fig.add_subplot(gs[1, 0]); F2.letter(ax, "c", -0.06, 0.02)
    grp = list(dict.fromkeys(CX.boundary))
    for j, (a, lab) in enumerate((("grad_RNA", f"|∂{sub('RNA')}|"), ("grad_ATAC", f"|∂{sub('ATAC')}|"))):
        sb = CX[CX.axis == a].set_index("boundary").loc[grp]; x = np.arange(len(grp)) + (j - 0.5) * 0.36
        ax.bar(x, sb.rho, 0.34, color=(mcol("RNA"), mcol("ATAC"))[j], label=lab)
        for xi, (r, p) in zip(x, zip(sb.rho, sb.p_rot)):
            ax.text(xi, r + 0.008, f"{p:.2f}", ha="center", va="bottom", fontsize=5)
    ax.set_xticks(range(len(grp))); ax.set_xticklabels([g.replace(" (", "\n(").replace(" / ", "/\n").replace("cortical neuron layers", "neuron\nlayers") for g in grp], fontsize=5.3)
    ax.set_ylim(0, 0.34); ax.set_ylabel("ρ with protein boundary strength", labelpad=1)
    ax.set_title("LPC, CODEX (same section, not fitted):\nneuron-layer boundaries follow ρ$_{\\mathrm{RNA}}$", fontsize=6, pad=2)
    ax.legend(fontsize=5.2, frameon=False, loc="upper left", ncol=2)
    out["codex"] = {f"{b}|{a}": dict(rho=round(r, 3), p_rot=round(p, 3)) for b, a, r, p in zip(CX.boundary, CX.axis, CX.rho, CX.p_rot)}
    ax = fig.add_subplot(gs[1, 1]); F2.letter(ax, "d", -0.06, 0.02)
    labs = [("gradATAC", f"|∂{sub('ATAC')}|"), ("gradRNA", f"|∂{sub('RNA')}|"), ("ratio", "steepness\nratio")]
    for j, k in enumerate(STEEP_CASES):
        ax.bar(np.arange(3) + (j - 0.5) * 0.36, [FU[k]["data"][a] for a, _ in labs], 0.34, color=(mcol("RNA"), mcol("ATAC"))[j], label=NM[k])
        out[f"data|{k}"] = {a: round(FU[k]["data"][a], 3) for a, _ in labs}
    ax.axhline(0, color="#1a1a1a", lw=0.5); ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in labs]); ax.set_ylim(-0.1, 0.45)
    ax.set_ylabel("Spearman ρ, model vs raw targets", labelpad=1)
    ax.set_title("each axis's gradient is in the data;\nthe ratio is not", fontsize=6, pad=2); ax.legend(fontsize=5.2, frameon=False, loc="upper right")
    return first, out


# --------------------------------------------------------------------------- #
def _single(fn):
    def b(fig, spec, D):
        ax = fig.add_subplot(spec); return ax, fn(ax, D)
    return b


PANELS = [
    ("a", "saturn_vs_gaston_R2", (3.2, 3.0), _single(panel_bench)),
    ("b", "cross_term_R2_gain", (3.0, 2.4), _single(panel_split)),
    ("c", "local_cross_gain_maps", (7.2, 8.6), panel_gain_maps),
    ("d", "cross_explained_genes", (2.6, 2.0), _single(panel_ngenes)),
    ("e", "cis_chromatin_concordance", (2.0, 2.2), _single(panel_cis)),
    ("f", "hic_long_genes", (1.9, 2.2), _single(panel_length)),
    ("g", "cross_examples", (6.6, 9.4), panel_examples),
    ("h", "what_atac_steep_zone_marks", (7.2, 4.2), panel_steep),
    ("i", "what_atac_cross_term_reflects", (7.2, 2.9), panel_mech),
    ("j", "steep_zone_transition", (7.2, 4.6), panel_transition),
]
SUPP = [("S7", "gradient_vs_cross_gain", (7.2, 8.6), panel_supp_gradient),
        ("S8", "steepness_checks", (7.2, 5.6), panel_supp_steep)]
BUILD = {L: b for L, _, _, b in PANELS + SUPP}


def page(D, res, letters, size, layout, nudge, stem):
    fig = plt.figure(figsize=size); where = layout(fig); F2.LETTERS.clear()
    for L in letters:
        ax, out = BUILD[L](fig, where[L], D); res[L] = out; F2.letter(ax, "a" if L.startswith("S") else L, *nudge[L])
    F2.draw_letters(fig); save_dual(fig, OUT.parent / stem, extra=dict(panels=",".join(letters), stats={L: res[L] for L in letters if res.get(L)}))
    plt.close(fig)


def part1(fig):
    o = GridSpec(2, 1, figure=fig, height_ratios=[2.8, 8.6], hspace=0.28, left=0.1, right=0.93, top=0.975, bottom=0.02)
    g1 = GridSpecFromSubplotSpec(1, 2, subplot_spec=o[0], width_ratios=[1.05, 1], wspace=0.45)
    return dict(a=g1[0], b=g1[1], c=o[1])


def part2(fig):
    o = GridSpec(2, 1, figure=fig, height_ratios=[2.1, 9.4], hspace=0.22, left=0.08, right=0.9, top=0.975, bottom=0.02)
    g1 = GridSpecFromSubplotSpec(1, 3, subplot_spec=o[0], width_ratios=[1.35, 1, 0.95], wspace=0.75)
    return dict(d=g1[0], e=g1[1], f=g1[2], g=o[1])


def part3(fig):
    o = GridSpec(3, 1, figure=fig, height_ratios=[4.0, 2.6, 4.2], hspace=0.42, left=0.1, right=0.95, top=0.97, bottom=0.06)
    return dict(h=o[0], i=o[1], j=o[2])


def supp(fig):
    return {"S7": GridSpec(1, 1, figure=fig, left=0.08, right=0.92, top=0.95, bottom=0.04)[0]}


def supp8(fig):
    return {"S8": GridSpec(1, 1, figure=fig, left=0.1, right=0.97, top=0.93, bottom=0.1)[0]}


def main():
    apply(base_pt=6.5)
    D = load(); res = {}
    page(D, res, ["a", "b", "c"], (7.2, 11.6), part1, dict(a=(-0.07, 0.02), b=(-0.06, 0.02), c=(-0.06, 0.012)), "fig7_part1")
    page(D, res, ["d", "e", "f", "g"], (7.2, 11.8), part2,
         dict(d=(-0.13, 0.02), e=(-0.08, 0.02), f=(-0.07, 0.02), g=(-0.05, 0.03)), "fig7_part2")
    page(D, res, ["h", "i", "j"], (7.2, 12.0), part3, dict(h=(-0.06, 0.06), i=(-0.08, 0.06), j=(-0.06, 0.06)), "fig7_part3")
    page(D, res, ["S7"], (7.2, 9.4), supp, {"S7": (-0.04, 0.01)}, "figS7_gradient")
    page(D, res, ["S8"], (7.2, 5.6), supp8, {"S8": (-0.06, 0.02)}, "figS8_steepness_checks")
    pdir = OUT.parent / "panels"; pdir.mkdir(exist_ok=True)
    for L, nm, size, b in PANELS + SUPP:
        fig = plt.figure(figsize=size)
        spec = GridSpec(1, 1, figure=fig, left=0.3 if L in "ad" else 0.14, right=0.88, top=0.9, bottom=0.18 if L in "bdef" else 0.04)[0]
        b(fig, spec, D); save_panel_svg(fig, pdir / (f"fig{L}_{nm}" if L.startswith("S") else f"fig7{L}_{nm}")); plt.close(fig)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
