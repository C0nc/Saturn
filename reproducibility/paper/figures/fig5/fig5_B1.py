"""Figure 5 — Sprox-seq tonsil, section B1: two isodepths and the held-out protein-complex layer along them.

Saturn fit sproxseq_B1_p46_dr (RNA GLM-PCA + 46 protein probes, log depth regressed). The 1,678 PLA products are never in
the loss. Complex statistic = isotype-corrected specificity: pooled log2((O+1)/(E+1)) minus the partners' mouse-IgG
isotype log2 O/E (E = per-spot independence expectation); without the isotype term a rare partner inflates any complex.
Tables: Hole/gaston_mix_mm/results/real_saturn/sproxseq_B1_fig5/ (scripts/sproxseq_B1_fig5_prep.py).

part 1
  a  B1 compartments from marker genes
  b  fit quality: R2 full / own axis / cross axis
  c  held-out complexes whose specificity changes along each axis (slope over 8 bins vs 200 rotations; chance 5%)
  d  rho_RNA, RNA domains, rho_Protein, Protein domains (GASTON DP, elbow K; labels from marker programmes)
  e  representative features, each original map | fitted function at every spot | bin means + fitted function:
     genes on rho_RNA (Poisson), proteins (A/B probes summed) on rho_Protein (Poisson)
part 2
  f  RNA programmes along rho_RNA (octiles)          g  protein markers along rho_Protein (octiles)
  h  complex specificity along rho_RNA, the 12 steepest at each end among rotation-null hits (row-centred)
  i  the same along rho_Protein
  j  cell-cell contact complexes as fitted functions: 7-spot pooled specificity | fitted function at every spot |
     30 equal-count bins + per-domain fit (Gaussian piecewise, F-test gate t = 0.1)
-> published_figures/fig5/fig5_part{1,2}.{png,svg,json} + panels/fig5<letter>_<name>.svg
Usage: python published_figures/fig5/fig5_B1.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp_
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/_style")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/saturn_repo/src")       # Saturn package (saturn_repo)
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/fig2")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
from paper_style import apply, save_dual, save_panel_svg, MODALITY, DOMAIN, CMAP, ANCHOR, PALETTE
import fig2_lpc as F2

H = "/mnt/HDD2B/Users/yancui"
R = Path(f"{H}/Hole/gaston_mix_mm/results/real_saturn"); T5 = R / "sproxseq_B1_fig5"; FIT = "sproxseq_B1_p46_dr"
OUT = Path(f"{H}/Saturn/published_figures/fig5")
C_RNA, C_PROT, INK = MODALITY["RNA"], MODALITY["Protein"], PALETTE["black"]
NAMES = {"RNA": {0: "crypt epithelium", 1: "subepithelial IgA plasma", 2: "mantle & T zone", 3: "germinal centre"},
         "Protein": {0: "T / Tfh–FDC", 1: "naive mantle B", 2: "CD9+ CD38+ surface"}}
DOMAIN.update({"crypt epithelium": ANCHOR["ochre"], "subepithelial IgA plasma": ANCHOR["plum"],
               "mantle & T zone": ANCHOR["sage"], "germinal centre": ANCHOR["night"],
               "T / Tfh–FDC": ANCHOR["teal"], "naive mantle B": ANCHOR["sand"], "CD9+ CD38+ surface": ANCHOR["rust"]})
COMP = {"epithelium": ANCHOR["ochre"], "EF": ANCHOR["rose"], "T": ANCHOR["teal"], "NBC": ANCHOR["sand"],
        "DZ": ANCHOR["night"], "LZ": ANCHOR["slate"]}
COMP_NAME = {"epithelium": "crypt epithelium", "EF": "extrafollicular", "T": "T zone", "NBC": "naive follicle",
             "DZ": "GC dark zone", "LZ": "GC light zone"}
FEATS = [("RNA", "CXCL13", "CXCL13", "surface / mantle end"), ("RNA", "BCL6", "BCL6", "germinal-centre end"),
         ("Protein", "LFA1", "LFA-1 protein", "T-zone end"), ("Protein", "CD19", "CD19 protein", "B-follicle end")]
# chosen from a fit screen (Poisson piecewise R2): crypt epithelial genes (KRT13/KRT5/CCL20) are too sparse at the
# narrow crypt end for a stable fit (R2 < 0); CD3 protein R2 -0.51
CCC_EX = [("RNA", "ITGA4:VCAM1", "VLA-4 – VCAM-1"), ("RNA", "VCAM1:CD23", "VCAM-1 – CD23"),
          ("Protein", "HLADPRQ:CD3", "MHC-II – CD3 (T–APC)"), ("Protein", "ICAM1:CD3", "ICAM-1 – CD3")]
PROT_MK = [("CD3", ANCHOR["teal"]), ("CD279", ANCHOR["sea"]), ("CD35", ANCHOR["rose"]), ("IgD", ANCHOR["sand"]),
           ("CD20", ANCHOR["ochre"]), ("CD38", ANCHOR["rust"]), ("CD9", ANCHOR["plum"])]
PLAB = {"CD279": "PD-1", "HLADPRQ": "MHC-II"}
RNA_PROG = {"epithelium": ANCHOR["ochre"], "CCL20 (epithelial chemokine)": ANCHOR["rust"], "myeloid": ANCHOR["rose"],
            "IgA plasma": ANCHOR["plum"], "CCR6 (memory B)": ANCHOR["sage"], "naive mantle": ANCHOR["sand"],
            "T zone": ANCHOR["teal"], "GC": ANCHOR["night"]}


def nice(c):
    for k, v in PLAB.items(): c = c.replace(k, v)
    return c


# --------------------------------------------------------------------------- #
def load():
    from domain_annot import CFG                                     # noqa: F401  (registers paths)
    from sproxseq_gc_trajectory import rna
    sp = pd.read_csv(T5 / "spots.csv", index_col=0); bc = list(sp.index)
    a = rna("B1", bc); a.var_names_make_unique()
    P = np.load(T5 / "proteins.npz", allow_pickle=True); assert list(P["barcodes"]) == bc
    Xp = sp_.csc_matrix(P["counts"].astype(float))
    D = dict(sp=sp, bc=bc, xy=sp[["x", "y"]].values.astype(float),
             d={"RNA": sp.rho_RNA.values, "Protein": sp.rho_Protein.values},
             lab={"RNA": sp.dom_RNA.values.astype(int), "Protein": sp.dom_Protein.values.astype(int)},
             X={"RNA": sp_.csc_matrix(a.X).astype(float), "Protein": Xp},
             names={"RNA": np.array([str(v) for v in a.var_names]), "Protein": np.array(P["targets"]).astype(str)},
             names_dom=NAMES, summary=json.load(open(R / FIT / "summary.json")), S=json.load(open(T5 / "summary.json")),
             a=a, prot=pd.DataFrame(P["counts"], columns=np.array(P["targets"]).astype(str), index=bc))
    D["lib"] = {m: np.asarray(D["X"][m].sum(1)).ravel().clip(1) for m in D["X"]}
    L = np.load(T5 / "layers.npz", allow_pickle=True)
    D["layers"] = dict(O=L["O"], E=L["E"], IO=L["IO"], IE=L["IE"], name=list(L["pairs"]), tg=list(L["tg"]))
    return D


def dcol(D, m):
    return np.array([DOMAIN[NAMES[m][k]] for k in range(len(NAMES[m]))])


def octile_bins(t, nb=8):
    return np.minimum((np.argsort(np.argsort(t)) * nb) // len(t), nb - 1)


# --------------------------------------------------------------------------- #
def panel_comp(ax, D):
    z = D["sp"].compartment.values
    for k in ["EF", "T", "NBC", "LZ", "DZ", "epithelium"]:
        m = z == k
        F2.spatial(ax, D["xy"][m], None, colors=COMP[k], s=1.6)
    ax.legend(handles=[Patch(fc=COMP[k], label=f"{COMP_NAME[k]} ({int((z == k).sum())})") for k in COMP],
              fontsize=5.2, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, handlelength=0.9)
    ax.set_title("tonsil B1: compartments from marker genes", fontsize=6.5, pad=2)


def panel_fit(ax, D):
    R2 = D["summary"]["R2_whole"]; mods = ["RNA", "Protein"]; x = np.arange(2); w = 0.26
    for i, (key, alpha) in enumerate([("full", 1.0), ("own_axis_only", 0.65), ("other_axes_only", 0.3)]):
        v = [R2[m][key] for m in mods]
        ax.bar(x + (i - 1) * w, v, w, color=[MODALITY[m] for m in mods], alpha=alpha, edgecolor=INK, lw=0.4)
        for xx, vv in zip(x + (i - 1) * w, v):
            ax.text(xx, vv + 0.015, f"{vv:.2f}", ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks(x); ax.set_xticklabels(mods)
    for tl, m in zip(ax.get_xticklabels(), mods): tl.set_color(MODALITY[m])
    ax.set_ylim(0, 0.8); ax.set_yticks([0, 0.4, 0.8]); ax.set_ylabel("variance explained (R²)"); ax.set_xlim(-0.55, 1.55)
    ax.legend(handles=[Patch(fc="#555555", alpha=a_, ec=INK, lw=0.4, label=l_) for a_, l_ in
                       [(1, "full"), (0.65, "own axis"), (0.3, "cross axis")]], fontsize=5.5, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=0.9, columnspacing=0.6, handletextpad=0.3, borderaxespad=0.1)


def panel_heldout(ax, D):
    S = D["S"]; mods = ["RNA", "Protein"]; v = [100 * S[m]["frac_p05"] for m in mods]
    ax.bar(range(2), v, 0.55, color=[MODALITY[m] for m in mods], edgecolor=INK, lw=0.4)
    for i, vv in enumerate(v): ax.text(i, vv + 0.6, f"{vv:.1f}%", ha="center", fontsize=5.8)
    ax.axhline(5, color="#9aa2aa", lw=0.7, ls="--"); ax.text(0.5, 5.6, "chance", fontsize=5.5, color="#7a828a", ha="center")
    ax.set_xticks(range(2)); ax.set_xticklabels(["ρ$_{\\mathrm{RNA}}$", "ρ$_{\\mathrm{Protein}}$"])
    for tl, m in zip(ax.get_xticklabels(), mods): tl.set_color(MODALITY[m])
    ax.set_ylabel("held-out complexes changing\nalong ρ (rotation p < 0.05, %)"); ax.set_ylim(0, 28); ax.set_xlim(-0.6, 1.6)
    ax.set_title(f"{S['RNA']['n_complexes']} complexes, never in the fit\n(isotype-corrected specificity)", fontsize=6, pad=2)


def panel_maps(fig, spec, D):
    gs = GridSpecFromSubplotSpec(2, 4, subplot_spec=spec, height_ratios=[1, 0.32], wspace=0.18, hspace=0.02); first = None
    for j, m in enumerate(("RNA", "Protein")):
        a1, a2 = fig.add_subplot(gs[0, 2 * j]), fig.add_subplot(gs[0, 2 * j + 1]); first = first or a1
        d = D["d"][m]; lo, hi = np.percentile(d, [1, 99])
        sc = F2.spatial(a1, D["xy"], d, cmap=CMAP["score"], vmin=lo, vmax=hi, s=1.2)
        F2.cbar(fig, a1, sc, "isodepth", ticks=[round(lo, 1), round(hi, 1)])
        end = ("crypt → interior" if m == "RNA" else "T zone → B follicle")
        a1.set_title(f"ρ$_{{\\mathrm{{{m}}}}}$  ({end})", color=MODALITY[m], fontsize=6.5, pad=2)
        F2.spatial(a2, D["xy"], None, colors=dcol(D, m)[D["lab"][m]], s=1.2)
        a2.set_title(f"{m} domains", color=MODALITY[m], fontsize=6.5, pad=2)
        lg = fig.add_subplot(gs[1, 2 * j:2 * j + 2]); lg.set_axis_off()
        for i, k in enumerate(range(len(NAMES[m]))):
            n = int((D["lab"][m] == k).sum()); x_ = (i % 2) * 0.52; y_ = 0.72 - (i // 2) * 0.45
            lg.add_patch(Rectangle((x_, y_ - 0.12), 0.035, 0.3, color=DOMAIN[NAMES[m][k]], transform=lg.transAxes))
            lg.text(x_ + 0.05, y_ + 0.03, f"{k} {NAMES[m][k]} ({n})", fontsize=5.5, va="center", transform=lg.transAxes)
    return first


def panel_features(fig, spec, D):
    F2.NB.update(RNA=25, Protein=25); F2.dcol = dcol
    gs = GridSpecFromSubplotSpec(len(FEATS), 1, subplot_spec=spec, hspace=0.55); first, st = None, {}   # one feature per row
    for i, (m, feat, title, sub) in enumerate(FEATS):
        a_, Fr = F2.panel_feature(fig, gs[i], D, m, feat, title, sub)
        first = first or a_; st[title] = dict(R2=round(float(Fr["r2"]), 3), n_sloped=Fr["n_sloped"])
    return first, st


def panel_rna_prog(ax, D):
    from sproxseq_B1_rna_axis import MOD, resid
    import scanpy as sc
    a = D["a"].copy(); ld = np.log1p(np.asarray(a.X.sum(1)).ravel()); sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    b = octile_bins(D["d"]["RNA"])
    for k, c in RNA_PROG.items():
        g = [x for x in MOD[k] if x in a.var_names]
        v = resid(np.asarray(a[:, g].X.todense()), [ld, D["sp"].mito.values]).mean(1); v = (v - v.mean()) / v.std()
        ax.plot(range(8), [v[b == i].mean() for i in range(8)], "-o", color=c, ms=2, lw=1.0,
                label=k.replace(" (epithelial chemokine)", "").replace(" (memory B)", " memory B"))
    ax.axhline(0, color="#9aa2aa", lw=0.4); ax.set_xticks([0, 7]); ax.set_xticklabels(["crypt", "interior"])
    ax.set_xlabel("ρ$_{\\mathrm{RNA}}$ octile", color=C_RNA); ax.set_ylabel("programme score (z)")
    ax.legend(fontsize=5.3, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), handlelength=1.2)
    F2.clean_ax(ax) if hasattr(F2, "clean_ax") else None


def panel_prot_prog(ax, D):
    P = D["prot"]; L = np.log1p(P.div(P.sum(1), axis=0) * 1e4); Z = (L - L.mean()) / L.std()
    b = octile_bins(D["d"]["Protein"])
    for t, c in PROT_MK:
        ax.plot(range(8), [Z[t].values[b == i].mean() for i in range(8)], "-o", color=c, ms=2, lw=1.0, label=nice(t))
    ax.axhline(0, color="#9aa2aa", lw=0.4); ax.set_xticks([0, 7]); ax.set_xticklabels(["T zone", "B / surface"])
    ax.set_xlabel("ρ$_{\\mathrm{Protein}}$ octile", color=C_PROT); ax.set_ylabel("protein (CLR-like z)")
    ax.legend(fontsize=5.3, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), handlelength=1.2)


def _heat(fig, spec, D, m, n_end):
    T = pd.read_csv(T5 / f"complexes_{m}.csv", index_col=0); Pf = pd.read_csv(T5 / f"profiles_{m}.csv", index_col=0)
    sig = T[T.p < 0.05].sort_values("slope")
    rows = list(sig.index[:n_end]) + list(sig.index[::-1][:n_end][::-1])
    k = 8; g = np.array_split(np.arange(len(Pf)), k)                    # 30 profile bins -> 8 for display
    M = np.array([[Pf[r].values[i].mean() for i in g] for r in rows]); M -= M.mean(1, keepdims=True)
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[1, 0.05], wspace=0.08)
    ax = fig.add_subplot(gs[0]); im = ax.imshow(M, cmap=CMAP["score"], vmin=-0.5, vmax=0.5, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([nice(r) for r in rows], fontsize=5.3)
    lo_c, hi_c = (ANCHOR["ochre"], ANCHOR["night"]) if m == "RNA" else (ANCHOR["teal"], ANCHOR["rust"])
    for tl, r in zip(ax.get_yticklabels(), rows): tl.set_color(lo_c if T.loc[r, "slope"] < 0 else hi_c)
    ax.axhline(n_end - 0.5, color="white", lw=1.2)
    ax.set_xticks([0, k - 1]); ax.set_xticklabels(["crypt", "interior"] if m == "RNA" else ["T zone", "B / surface"])
    ax.set_xlabel(f"ρ$_{{\\mathrm{{{m}}}}}$ octile", color=MODALITY[m])
    pct = 100 * D["S"][m]["frac_p05"]
    ax.set_title(f"{len(sig)} complexes change along ρ$_{{\\mathrm{{{m}}}}}$ ({pct:.1f}%, chance 5%)", fontsize=6, pad=2)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[1]), ticks=[-0.5, 0, 0.5]); cb.ax.tick_params(labelsize=5.5)
    cb.set_label("specificity, log2 O/E minus\nisotype (row-centred)", fontsize=5.5)
    return ax


def ccc_function(D, m, cname, k=7, nb=30):
    from scipy.spatial import cKDTree
    from saturn.piecewise import fit_gauss
    from sproxseq_D1_crypt_niche import spec
    L = D["layers"]; j = L["name"].index(cname); a, b = cname.split(":")
    one = lambda H: spec(H, [(a, b)], L["O"][:, [j]], L["E"][:, [j]], L["IO"], L["IE"], L["tg"])[:, 0]
    xy, d, lab = D["xy"], D["d"][m], D["lab"][m]; n = len(d)
    nn = cKDTree(xy).query(xy, k=k)[1]
    Hk = sp_.csr_matrix((np.ones(n * k), (nn.ravel(), np.repeat(np.arange(n), k))), shape=(n, n)).toarray()
    raw = one(Hk)                                                         # 7-spot pooled, per spot
    g = np.minimum((np.argsort(np.argsort(d)) * nb) // n, nb - 1); Hb = np.eye(nb)[g]
    by = one(Hb); bx = np.array([d[g == i].mean() for i in range(nb)])
    bl = np.array([np.bincount(lab[g == i], minlength=lab.max() + 1).argmax() for i in range(nb)])
    sl, it = fit_gauss(by[:, None], bl, bx, t=0.1); sl, it = sl[0], it[0]
    pred = it[bl] + sl[bl] * bx; r2 = 1 - ((by - pred) ** 2).sum() / ((by - by.mean()) ** 2).sum()
    return dict(raw=raw, fit=it[lab] + sl[lab] * d, bx=bx, by=by, bl=bl, sl=sl, it=it, r2=r2)


def panel_ccc(fig, spec, D):
    gs = GridSpecFromSubplotSpec(len(CCC_EX), 1, subplot_spec=spec, hspace=0.55); first, st = None, {}
    T = {m: pd.read_csv(T5 / f"complexes_{m}.csv", index_col=0) for m in ("RNA", "Protein")}
    for i, (m, cname, title) in enumerate(CCC_EX):
        C = ccc_function(D, m, cname); g3 = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[i],
                                                                    width_ratios=[1, 1, 1.05], wspace=1.25)
        a1, a2, a3 = (fig.add_subplot(g3[q]) for q in range(3)); first = first or a1
        lo, hi = np.percentile(np.r_[C["raw"], C["fit"]], [2, 98])
        F2.spatial(a1, D["xy"], C["raw"], cmap=CMAP["score"], vmin=lo, vmax=hi, s=0.9)
        sc = F2.spatial(a2, D["xy"], C["fit"], cmap=CMAP["score"], vmin=lo, vmax=hi, s=0.9)
        F2.cbar(fig, a2, sc, "log2 O/E\n− isotype", on_top=True)
        a1.set_title(f"{title}\n{nice(cname)}", fontsize=6.0, pad=2, color=MODALITY[m], loc="left")
        a1.text(0.5, -0.04, "original (7-spot pooled)", transform=a1.transAxes, ha="center", va="top", fontsize=5.3)
        a2.text(0.5, -0.04, "fitted function", transform=a2.transAxes, ha="center", va="top", fontsize=5.5)
        d, lab = D["d"][m], D["lab"][m]
        F2.along(a3, m, d, lab, dcol(D, m), C["bx"], C["by"], C["bl"], lambda k, xs, C=C: C["it"][k] + C["sl"][k] * xs,
                 "specificity")
        p = float(T[m].loc[cname, "p"])
        a3.set_title(f"R² {C['r2']:.2f} · rotation p {p:.3f}", fontsize=5.6, pad=2)
        st[cname] = dict(axis=m, R2_bins=round(float(C["r2"]), 3), slope_p=p)
    return first, st


def _single(fn):
    def b(fig, spec, D):
        ax = fig.add_subplot(spec); return ax, fn(ax, D)
    return b


PANELS = [
    ("a", "compartments", (2.6, 1.6), _single(panel_comp)),
    ("b", "fit_quality_R2", (1.5, 1.6), _single(panel_fit)),
    ("c", "heldout_complexes_along_rho", (1.5, 1.6), _single(panel_heldout)),
    ("d", "isodepths_and_domains", (7.2, 1.55), lambda fig, spec, D: (panel_maps(fig, spec, D), None)),
    ("e", "fitted_functions", (6.0, 7.6), panel_features),
    ("f", "rna_programmes_along_rho_RNA", (2.9, 1.7), _single(panel_rna_prog)),
    ("g", "protein_markers_along_rho_Protein", (2.9, 1.7), _single(panel_prot_prog)),
    ("h", "complexes_along_rho_RNA", (2.6, 2.7), lambda fig, spec, D: (_heat(fig, spec, D, "RNA", 12), None)),
    ("i", "complexes_along_rho_Protein", (2.6, 2.7), lambda fig, spec, D: (_heat(fig, spec, D, "Protein", 10), None)),
    ("j", "ccc_fitted_functions", (6.0, 7.6), panel_ccc),
]
BUILD = {L: b for L, _, _, b in PANELS}


def page(D, res, letters, size, layout, nudge, stem):
    fig = plt.figure(figsize=size); where = layout(fig); F2.LETTERS.clear()
    for L in letters:
        ax, out = BUILD[L](fig, where[L], D); res[L] = out; F2.letter(ax, L, *nudge[L])
    F2.draw_letters(fig); save_dual(fig, OUT / stem, extra=dict(panels=",".join(letters), stats={L: res[L] for L in letters if res.get(L)}))
    plt.close(fig)


def part1(fig):
    o = GridSpec(3, 1, figure=fig, height_ratios=[1.5, 1.55, 7.2], hspace=0.35, left=0.07, right=0.95, top=0.975, bottom=0.02)
    g1 = GridSpecFromSubplotSpec(1, 3, subplot_spec=o[0], width_ratios=[2.1, 1.0, 0.9], wspace=0.9)
    return dict(a=g1[0], b=g1[1], c=g1[2], d=o[1], e=o[2])


def part2(fig):
    o = GridSpec(3, 1, figure=fig, height_ratios=[1.6, 2.7, 7.2], hspace=0.35, left=0.08, right=0.95, top=0.98, bottom=0.02)
    g1 = GridSpecFromSubplotSpec(1, 2, subplot_spec=o[0], wspace=0.75)
    g2 = GridSpecFromSubplotSpec(1, 2, subplot_spec=o[1], wspace=0.75)
    return dict(f=g1[0], g=g1[1], h=g2[0], i=g2[1], j=o[2])


def main():
    apply(base_pt=6.5)
    F2.MODALITY["Protein"] = C_PROT
    D = load(); res = {}
    page(D, res, "abcde", (7.2, 11.4), part1,
         dict(a=(-0.03, 0.015), b=(-0.07, 0.03), c=(-0.07, 0.03), d=(-0.03, 0.02), e=(-0.06, 0.03)), "fig5_part1")
    page(D, res, "fghij", (7.2, 12.6), part2,
         dict(f=(-0.07, 0.02), g=(-0.07, 0.02), h=(-0.16, 0.02), i=(-0.16, 0.02), j=(-0.06, 0.03)), "fig5_part2")
    pdir = OUT / "panels"; pdir.mkdir(exist_ok=True)
    for L, nm, size, b in PANELS:
        fig = plt.figure(figsize=size)
        spec = GridSpec(1, 1, figure=fig, left=0.3 if L in "hi" else 0.16, right=0.8 if L in "fg" else 0.88,
                        top=0.84, bottom=0.2)[0]
        b(fig, spec, D); save_panel_svg(fig, pdir / f"fig5{L}_{nm}"); plt.close(fig)
    print(json.dumps({k: res[k] for k in ("e", "j")}, indent=1, default=str))


if __name__ == "__main__":
    main()
