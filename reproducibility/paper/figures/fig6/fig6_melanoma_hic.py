"""Figure 6 — Melanoma spatial Hi-C + RNA: chromatin structure varies along the isodepths, and reading chromatin and
expression over a LONGER stretch of rho reveals more Granger coupling between them -- whose direction a spatial axis
cannot resolve.

SpHIC117 melanoma, Saturn `mel_hic_abins_dr50_51_rnadr` (seed 42; 6,108 spots, DBiT chip rows 50-51 dropped).
  RNA  GLM-PCA, cubic in log RNA depth regressed out
  HiC  the chromatin layers themselves: A/B compartment (500 kb) + Higashi single-spot TAD insulation (averaged into
       the same 500 kb bins), each bin regressed on a cubic in log Hi-C reads, z-scored, the two layers weighted
       equally, PCA -> 10 PCs  [scripts/mel_hic_abins_prep.py --rna-dr]
  (replaces the Fast-Higashi fit mel_hic_fh_dr50_51: Hi-C R2 0.19 -> 0.59; without RNA depth regression the RNA axis
   fell onto library depth, Spearman 0.54)
Gene side of every bin = mean log-expression of its >= 3 protein-coding genes (2,558 bins).

  a  rho_RNA | RNA domains (marker-annotated) | rho_HiC | Hi-C domains (tumour / non-tumour; reads shown)
  c  (bottom row) chromatin bins that track each axis beyond a 200-rotation null, Hi-C reads partialled out
     (rho_HiC is BUILT from these bins, so its bars are in-sample; rho_RNA's are the out-of-modality test)
  b  fitted functions, each original map | fitted function at every spot | bin means + fit:
     genes on rho_RNA (Poisson), A/B compartment bins and insulation bins on rho_HiC (Gaussian, F-test gate t = 0.1)
  d  chromatin-expression Granger coupling vs the fraction of rho read (all spots, cubic detrend, noise-matched genes)
  e  the same with every window cut to the SAME number of spots (length effect only)
  g  examples: bins with a per-bin Granger test passing FDR along rho whose direction flips when rho is reversed
     (chromatin per spot | noise-matched gene score per spot | detrended window series)  [mel_hic_granger_examples.py]
  h  cis consistency: slope signs of gene / A/B / insulation in the same 500 kb bin, per domain, vs a random genome-wide background
     (same gene, chromatin of a bin > 1 Mb away)  [mel_hic_cis_threeway.py]      i  three cis-consistent example bins: fitted function maps
     (genes | A/B | insulation) + the three fits along rho
  f  direction: chromatin->gene vs gene->chromatin, rho forward / reversed (noise-matched genes)

Analysis: Hole/gaston_mix_mm/scripts/mel_hic_insulation_granger.py and mel_hic_bins_partial.py (--ds of this fit).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.stats import spearmanr

sys.path.insert(0, "<workspace>/Saturn/published_figures/_style")
sys.path.insert(0, "<workspace>/saturn_repo/src")       # Saturn package (saturn_repo)
sys.path.insert(0, "<workspace>/Saturn/published_figures/fig2")
sys.path.insert(0, "<workspace>/Hole/gaston_mix_mm/scripts")
from paper_style import apply, save_dual, save_panel_svg, MODALITY, DOMAIN, CMAP
import fig2_lpc as F2

H = "<workspace>"
R = Path(f"{H}/Hole/gaston_mix_mm/results/real_saturn")
DS = "mel_hic_abins_dr50_51_rnadr"
G = R / f"{DS}_insulation_granger"
OUT = Path(f"{H}/Saturn/published_figures/fig6/fig6_melanoma_hic")
FINAL = Path(f"{H}/Saturn/final_fig6")
C_RNA, C_HIC = MODALITY["RNA"], MODALITY["HiC"]
AXC = {"RNA": C_RNA, "HiC": C_HIC}
LAYER_LS = {"A/B": "-", "insulation": "--"}
DOM_MARKERS = {"epidermis": ["KRT10", "KRT1", "KRT14"], "plasma / immune": ["IGHG1", "IGHA1", "CD74"],
               "melanocytic tumour": ["DCT", "OCA2", "PMEL"], "deep tumour": ["FRMD5", "DPP6", "KCNMA1"]}
# elbow K = 3 on rho_RNA: plasma / immune (954 spots) and melanocytic tumour (1,726) form one domain
DOM_MARKERS_K3 = {"epidermis": DOM_MARKERS["epidermis"],
                  "melanocytic tumour & immune": DOM_MARKERS["melanocytic tumour"] + DOM_MARKERS["plasma / immune"],
                  "deep tumour": DOM_MARKERS["deep tumour"]}
N_EX = 3
TUMOUR = ["DCT", "PMEL", "MLANA", "TYR", "FRMD5", "DPP6", "KCNMA1"]
DOMAIN.update({"tumour": "#4C436A", "non-tumour": "#DCCB9F", "epidermis–tumour transition": "#B6ADC9",
               "deep tumour": "#5F9AA3"})                 # deep tumour was nearly the melanocytic-tumour purple


def _csv(f):
    return pd.read_csv(f) if Path(f).exists() else None          # Granger tables may still be running (preview)


def load():
    from saturn.domains import gaston_domains
    from continuous_marker_panels import mel_rna
    from saturn.domains import dp_error_curve as curve, kneedle
    ax = pd.read_csv(R / DS / "axes.csv", index_col=0); bc = [str(b) for b in ax.index]
    z = dict(np.load(R / f"targets_{DS}.npz", allow_pickle=True))
    C = dict(np.load(G / "cache_series.npz", allow_pickle=True))
    assert list(C["bc"].astype(str)) == bc
    D = dict(xy=np.c_[ax.x.values, ax.y.values], d={m: ax[f"d_{m}"].values for m in ("RNA", "HiC")},
             lab=gaston_domains(ax["d_RNA"].values, z["Y_RNA"], n_domains=kneedle(curve(ax["d_RNA"].values, z["Y_RNA"]))),
             labH=gaston_domains(ax["d_HiC"].values, z["Y_HiC"], n_domains=kneedle(curve(ax["d_HiC"].values, z["Y_HiC"]))), C=C,
             bins=pd.read_csv(G / "bins_vs_rho_partial.csv"),
             main=_csv(G / "granger_vs_length_natural_deg3_smooth25.csv"),
             ctrl=_csv(G / "granger_vs_length_deg3_smooth25.csv"),
             direc=_csv(G / "direction_tests.csv"),
             fit=json.load(open(R / DS / "summary.json"))["R2_whole"])
    X, genes = mel_rna(bc)
    lib = np.asarray(X.sum(1)).ravel().clip(1)
    D["names_dom"] = label_domains(D, X, genes, lib)
    D["X"], D["genes"], D["libr"] = X, np.asarray(genes).astype(str), lib
    HG = f"{H}/SpHIC117/HIC/higashi"
    fl = [l.strip().split("/")[-1].split(".")[0] for l in open(f"{HG}/filelist.txt")]
    D["reads"] = np.load(f"{HG}/read_count_all.npy")[pd.Index(fl).get_indexer(pd.Index(bc))].astype(float)   # log
    gi = {g: i for i, g in enumerate(genes)}; t = np.log1p(np.asarray(X[:, [gi[g] for g in TUMOUR if g in gi]].sum(1)).ravel() / lib * 1e4)
    K = D["labH"].max() + 1; mt = [t[D["labH"] == k].mean() for k in range(K)]
    D["names_H"] = ({int(np.argmax(mt)): "tumour", int(np.argmin(mt)): "non-tumour"} if K == 2 else
                    {k: n for k, n in zip(np.argsort(mt), ["non-tumour", "epidermis–tumour transition", "tumour"][:K])})
    return D


def label_domains(D, X, genes, lib):
    """assign each RNA domain the marker programme it is most enriched for (z across domains)"""
    gi = {g: i for i, g in enumerate(genes)}; lab = D["lab"]; K = lab.max() + 1
    sc = {}
    for name, gs in (DOM_MARKERS if K == 4 else DOM_MARKERS_K3).items():
        cols = [gi[g] for g in gs if g in gi]
        v = np.log1p(np.asarray(X[:, cols].sum(1)).ravel() / lib * 1e4)
        m = np.array([v[lab == k].mean() for k in range(K)]); sc[name] = (m - m.mean()) / m.std()
    S = pd.DataFrame(sc)
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-S.values)
    out = {int(k): S.columns[j] for k, j in zip(r, c)}
    for k, n in out.items():
        assert S.loc[k, n] > 0, f"domain {k} not enriched for {n} ({S.loc[k, n]:.2f})"
    return out


def dcol(D, m="RNA"):
    nm = D["names_dom"] if m == "RNA" else D["names_H"]
    return np.array([DOMAIN[nm[k]] for k in range(len(nm))])


# --------------------------------------------------------------------------- #
def panel_rho(fig, ax, D, m):
    d = D["d"][m]; lo, hi = np.percentile(d, [1, 99])
    sc = F2.spatial(ax, D["xy"], d, cmap=CMAP["score"], vmin=lo, vmax=hi, s=1.5)
    F2.cbar(fig, ax, sc, "isodepth", ticks=[np.ceil(lo * 10) / 10, np.floor(hi * 10) / 10])
    ax.set_title(f"ρ$_{{\\mathrm{{{m}}}}}$  (R² {D['fit'][m]['full']:.2f})", color=AXC[m], fontsize=6.8, pad=2)


def panel_domains(ax, D):
    F2.spatial(ax, D["xy"], None, colors=dcol(D)[D["lab"]], s=1.5)
    order = np.argsort([np.median(D["d"]["RNA"][D["lab"] == k]) for k in range(D["lab"].max() + 1)])
    ax.legend(handles=[Line2D([], [], ls="", marker="s", ms=3.2, color=DOMAIN[D["names_dom"][k]],
                              label=f"{D['names_dom'][k]} ({(D['lab'] == k).sum()})") for k in order],
              fontsize=5.2, loc="upper center", bbox_to_anchor=(0.5, -0.02), borderaxespad=0, handletextpad=0.1,
              labelspacing=0.2, frameon=False)
    ax.set_title("RNA domains (along ρ$_{\\mathrm{RNA}}$)", color=C_RNA, fontsize=6.5, pad=2)


def panel_domains_hic(ax, D):
    lab = D["labH"]; F2.spatial(ax, D["xy"], None, colors=dcol(D, "HiC")[lab], s=1.5)
    order = np.argsort([np.median(D["d"]["HiC"][lab == k]) for k in range(lab.max() + 1)])
    ax.legend(handles=[Line2D([], [], ls="", marker="s", ms=3.2, color=DOMAIN[D["names_H"][k]],
                              label=f"{D['names_H'][k]} ({(lab == k).sum()}; reads {np.exp(np.median(D['reads'][lab == k]) - np.median(D['reads'])):.2f}×)")
                       for k in order], fontsize=5.2, loc="upper center", bbox_to_anchor=(0.5, -0.02), borderaxespad=0,
              handletextpad=0.1, labelspacing=0.2, frameon=False)
    ax.set_title("Hi-C domains (along ρ$_{\\mathrm{HiC}}$)", color=C_HIC, fontsize=6.5, pad=2)
    return {D["names_H"][k]: dict(n=int((lab == k).sum()), median_log_reads=round(float(np.median(D["reads"][lab == k])), 2))
            for k in range(lab.max() + 1)}


def panel_track(ax, D):
    B = D["bins"]; x = np.arange(2); w = 0.36; out = {}
    for j, L in enumerate(("AB", "insulation")):
        v = []
        for m in ("RNA", "HiC"):
            s = B[(B.axis == m) & (B.layer == L)]; v.append(100 * (s.p_rot < 0.05).mean())
            out[f"{m}|{L}"] = f"{int((s.p_rot < 0.05).sum())}/{len(s)}"
        ax.bar(x + (j - 0.5) * w, v, w, color=[C_RNA, C_HIC], edgecolor="#1a1a1a", lw=0.4, hatch=None if L == "AB" else "////",
               alpha=1 if L == "AB" else 0.6)
        for xx, vv in zip(x + (j - 0.5) * w, v): ax.text(xx, vv + 0.8, f"{vv:.0f}", ha="center", fontsize=5.2)
    ax.axhline(5, color="#9aa2aa", lw=0.7, ls=(0, (2, 2)))
    ax.set_xticks(x); ax.set_xticklabels(["ρ$_{\\mathrm{RNA}}$", "ρ$_{\\mathrm{HiC}}$"])
    for tl, m in zip(ax.get_xticklabels(), ("RNA", "HiC")): tl.set_color(AXC[m])
    ax.set_ylim(0, 92); ax.set_ylabel("500 kb bins tracking ρ (%)", labelpad=1)
    ax.text(1, -0.24, "in-sample", fontsize=5, ha="center", va="top", color=C_HIC, transform=ax.get_xaxis_transform())
    ax.legend(handles=[Patch(fc="#777777", ec="#1a1a1a", lw=0.4, label="A/B compartment"),
                       Patch(fc="#bbbbbb", ec="#1a1a1a", lw=0.4, hatch="////", label="TAD insulation"),
                       Line2D([], [], color="#9aa2aa", lw=0.7, ls=(0, (2, 2)), label="chance (5%)")],
              fontsize=5.2, loc="upper left", handlelength=1.2, borderaxespad=0.1)
    ax.set_title("rotation null, Hi-C reads partialled", fontsize=6.3, pad=2)
    return out


GENES_E = [("KRT10", "epidermis"), ("DCT", "melanocytic tumour")]


MAP_S = 1.3                                                      # spot size on every map of the fitted-function block
COL_W = [1, 0.05, 1, 0.98, 0.88]                                  # original | gap | fitted | colourbar room | along rho
COL_GAP = 0.28                                                   # between the two features of a row


def _gene_feature(fig, cells, Dg, g, sub):
    """one gene: original | fitted function at every spot | bin means + fit along rho_RNA (same content as F2.panel_feature)"""
    F = F2.feature_function(Dg, "RNA", g)
    v = np.log1p(F["c"] / F["lib"] * 1e4); lo, hi = np.percentile(np.r_[v, F["spot"]], [2, 99])
    a1, a2, a3 = (fig.add_subplot(c) for c in cells); o = np.argsort(v)      # high on top: sparse signal stays visible
    F2.spatial(a1, Dg["xy"][o], v[o], cmap=CMAP["expression"], vmin=lo, vmax=hi, s=MAP_S)
    sc = F2.spatial(a2, Dg["xy"], F["spot"], cmap=CMAP["expression"], vmin=lo, vmax=hi, s=MAP_S)
    F2.cbar(fig, a2, sc, "log\nCP10k", on_top=True)
    a1.set_title(f"{g}\n{sub}", fontsize=6.0, pad=2, color=C_RNA, loc="left")
    a1.text(0.5, -0.04, "original", transform=a1.transAxes, ha="center", va="top", fontsize=5.5)
    a2.text(0.5, -0.04, "fitted function", transform=a2.transAxes, ha="center", va="top", fontsize=5.5)
    F2.along(a3, "RNA", Dg["d"]["RNA"], Dg["lab"]["RNA"], F2.dcol(Dg, "RNA"), F["db"], F["ob"], F["lb"],
             lambda k, xs: np.log1p(np.exp(F["it"][k] + F["sl"][k] * xs) * 1e6), "log(CPM+1)")
    a3.set_title(f"fitted function · R² {F['r2']:.2f}\n{F['n_sloped']}/{len(F['sl'])} domains sloped", fontsize=5.8, pad=2)
    return a1, dict(R2=round(float(F["r2"]), 3), n_sloped=F["n_sloped"])


def _pick_bins(D, layer, m, n=2):
    """the n chromatin bins of one layer that track axis m most strongly (rotation p < 0.05)"""
    B = D["bins"]; s_ = B[(B.axis == m) & (B.layer == layer) & (B.p_rot < 0.05)]
    return list(s_.reindex(s_.r.abs().sort_values(ascending=False).index).bin[:n])


def _bin_feature(fig, cells, D, layer, m, b):
    """one chromatin bin: original per-spot map | fitted function at every spot | bin means + fit along rho_m"""
    from saturn.piecewise import fit_gauss
    key, lname = {"AB": ("AB", "A/B compartment"), "insulation": ("IN", "insulation")}[layer]
    B = D["bins"]; s_ = B[(B.axis == m) & (B.layer == layer) & (B.p_rot < 0.05)]
    names = list(D["C"]["names"].astype(str)); V = D["C"][key]; gl = D["C"]["genes"].astype(str)
    d = D["d"][m]; lab = D["lab"] if m == "RNA" else D["labH"]; cl = dcol(D, m); nb = 25
    q = np.quantile(d, np.linspace(0, 1, nb + 1)); bi = np.clip(np.digitize(d, q[1:-1]), 0, nb - 1)
    j = names.index(b); v = np.asarray(V[:, j], float)
    sl, it = fit_gauss(v[:, None], lab, d, t=0.1); sl, it = sl[0], it[0]; fit = it[lab] + sl[lab] * d
    xb = np.array([d[bi == k].mean() for k in range(nb)]); yb = np.array([v[bi == k].mean() for k in range(nb)])
    lb = np.array([np.bincount(lab[bi == k], minlength=lab.max() + 1).argmax() for k in range(nb)])
    pr = it[lb] + sl[lb] * xb; r2 = 1 - ((yb - pr) ** 2).sum() / ((yb - yb.mean()) ** 2).sum()
    a1, a2, a3 = (fig.add_subplot(c) for c in cells)
    lo, hi = np.percentile(v, [2, 98])
    F2.spatial(a1, D["xy"], v, cmap=CMAP["score"], vmin=lo, vmax=hi, s=MAP_S)
    sc = F2.spatial(a2, D["xy"], fit, cmap=CMAP["score"], vmin=lo, vmax=hi, s=MAP_S)
    F2.cbar(fig, a2, sc, "A/B" if layer == "AB" else "insulation", on_top=True)
    c_, se = b.split(":"); s0, e0 = (int(x) / 1e6 for x in se.split("-"))
    a1.set_title(f"{lname} {c_} {s0:g}–{e0:g} Mb\n{', '.join(gl[j].split(',')[:3])}", fontsize=6.0, pad=2,
                 color=C_HIC, loc="left")
    a1.text(0.5, -0.04, "original (per spot)", transform=a1.transAxes, ha="center", va="top", fontsize=5.5)
    a2.text(0.5, -0.04, "fitted function", transform=a2.transAxes, ha="center", va="top", fontsize=5.5)
    F2.along(a3, m, d, lab, cl, xb, yb, lb, lambda k, xs, sl=sl, it=it: it[k] + sl[k] * xs,
             "compartment score" if layer == "AB" else "insulation score")
    r = s_.set_index("bin").loc[b]
    a3.set_title(f"fitted function · R² {r2:.2f}\nrotation p {r.p_rot:.3f}", fontsize=5.8, pad=2)
    return a1, dict(R2_bins=round(float(r2), 3), r=round(float(r.r), 3), p=float(r.p_rot), genes=gl[j])


def panel_features(fig, spec, D, letters=None):
    """three aligned rows on ONE grid, so every map square is the same size:
    genes on rho_RNA (Poisson piecewise) | A/B compartment bins on rho_HiC | insulation bins on rho_HiC (Gaussian piecewise)"""
    import scipy.sparse as sp_
    F2.NB["RNA"] = 25; F2.MODALITY["HiC"] = C_HIC
    Dg = dict(D, lab={"RNA": D["lab"]}, names_dom={"RNA": D["names_dom"]}, X={"RNA": sp_.csc_matrix(D["X"]).astype(float)},
              names={"RNA": D["genes"]}, lib={"RNA": D["libr"]})
    F2.dcol = lambda D_, m: dcol(D, m)
    gs = GridSpecFromSubplotSpec(3, 11, subplot_spec=spec, width_ratios=COL_W + [COL_GAP] + COL_W, wspace=0, hspace=0.85)
    rows = [[("gene", g, sub) for g, sub in GENES_E],
            [("AB", b, None) for b in _pick_bins(D, "AB", "HiC")],
            [("insulation", b, None) for b in _pick_bins(D, "insulation", "HiC")]]
    first, st = None, {}
    for i, row in enumerate(rows):
        for j, (kind, name, sub) in enumerate(row):
            c0 = 6 * j; cells = (gs[i, c0], gs[i, c0 + 2], gs[i, c0 + 4])
            if kind == "gene":
                a_, out = _gene_feature(fig, cells, Dg, name, sub); st[name] = out
            else:
                a_, out = _bin_feature(fig, cells, D, kind, "HiC", name); st[f"{kind}|{name}"] = out
            first = first or a_
            if j == 0 and letters:
                a_.text(-0.14, 1.32, letters[i], transform=a_.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
    return first, st


def _length_panel(ax, T, title, pooled=True):
    st = {}
    for m in ("RNA", "HiC"):
        for L, key in (("A/B", "AB"), ("insulation", "IN")):
            s = T[(T.axis == m) & (T.layer == L)]
            y = (s.c2g_p05 + s.g2c_p05) / 2 if pooled else s.c2g_p05
            g = pd.DataFrame(dict(frac=s.frac, y=y)).groupby("frac").y
            mu, se = g.mean() * 100, g.sem() * 100
            ax.plot(mu.index, mu.values, color=AXC[m], ls=LAYER_LS[L], lw=1.1, marker="o", ms=2.5)
            ax.fill_between(mu.index, mu - se, mu + se, color=AXC[m], alpha=0.12, lw=0)
            r, p = spearmanr(s.frac, y); st[f"{m}|{L}"] = dict(spearman=round(float(r), 2), p=float(p),
                                                              first=round(float(mu.iloc[0]), 2), last=round(float(mu.iloc[-1]), 2))
    nl = T.groupby("frac").null_p05.mean() * 100
    ax.plot(nl.index, nl.values, color="#9aa2aa", lw=1.0, ls=":", marker="o", ms=2)
    ax.set_xlabel("fraction of ρ read", labelpad=1); ax.set_ylabel("bins with Granger coupling (%)", labelpad=1)
    ax.set_xticks([0.1, 0.35, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0.1", "0.35", "0.5", "0.75", "1"])
    ax.set_ylim(0, None); ax.set_title(title, fontsize=6.2, pad=2)
    return st


def panel_length(ax, D):
    st = _length_panel(ax, D["main"], "all spots per window")
    ax.legend(handles=[Line2D([], [], color=C_RNA, lw=1.1, label="ρ$_{\\mathrm{RNA}}$"),
                       Line2D([], [], color=C_HIC, lw=1.1, label="ρ$_{\\mathrm{HiC}}$"),
                       Line2D([], [], color="#555555", lw=1.1, ls="-", label="A/B"),
                       Line2D([], [], color="#555555", lw=1.1, ls="--", label="insulation"),
                       Line2D([], [], color="#9aa2aa", lw=1.0, ls=":", label="shifted null")],
              fontsize=5.1, loc="lower right", handlelength=1.6, borderaxespad=0.1, labelspacing=0.2, ncol=2,
              columnspacing=0.8)
    return st


def panel_ctrl(ax, D):
    return _length_panel(ax, D["ctrl"], "same spots per window (length only)")


def panel_direction(fig, spec, D):
    """direction test for both chromatin layers, side by side on one scale (noise-matched genes). Bar label = % of the
    2,558 bins with P < 0.05 [bins with BH q < 0.05 in that direction]"""
    T = D["direc"]; T = T[T.genes == "25-NN genes"]; out = []
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.08)
    first = None
    rows = [(m, d) for m in ("RNA", "HiC") for d in ("forward", "reversed")]; y = np.arange(4)[::-1]
    for c, (layer, nm) in enumerate((("insulation", "insulation"), ("A/B", "A/B compartment"))):
        ax = fig.add_subplot(gs[c], sharey=first); first = first or ax
        lag = T[(T.layer == layer) & (T.direction == "forward")].lag1_chromatin.median()
        for yy, (m, dname) in zip(y, rows):
            s = T[(T.axis == m) & (T.layer == layer) & (T.direction == dname)].iloc[0]
            for dy, v, q, kw in ((0.19, s.c2g, s.c2g_q, dict(color=AXC[m], edgecolor="#1a1a1a", lw=0.3)),
                                 (-0.19, s.g2c, s.g2c_q, dict(color="white", edgecolor=AXC[m], lw=0.8))):
                ax.barh(yy + dy, 100 * v, 0.36, **kw)
                ax.text(100 * v + 0.3, yy + dy, f"{100 * v:.1f} [{int(q)}]", fontsize=5, va="center", ha="left")
            out.append(dict(layer=layer, axis=m, direction=dname, c2g=round(float(s.c2g), 4), g2c=round(float(s.g2c), 4),
                            c2g_q=int(s.c2g_q), g2c_q=int(s.g2c_q)))
        ax.set_xlim(0, 17.5); ax.set_xticks([0, 5, 10])
        ax.set_title(f"{nm}\nlag-1 chromatin {lag:.2f}", fontsize=5.8, pad=2)
        ax.set_xlabel("bins, P < 0.05 (%)", labelpad=1)
        if c == 0:
            ax.set_yticks(y); ax.set_yticklabels([f"ρ$_{{\\mathrm{{{m}}}}}$ {d_}" for m, d_ in rows], fontsize=6)
            for tl, (m, _) in zip(ax.get_yticklabels(), rows): tl.set_color(AXC[m])
        else:
            ax.tick_params(axis="y", left=False, labelleft=False); ax.spines["left"].set_visible(True)
    lg = T[T.direction == "forward"].lag1_genes.median()
    fig.legend(handles=[Patch(fc="#777777", ec="#1a1a1a", lw=0.3, label="chromatin → gene"),
                        Patch(fc="white", ec="#777777", lw=0.8, label="gene → chromatin")],
               fontsize=5.1, loc="upper center", bbox_to_anchor=(0.5, spec.get_position(fig).y0 - 0.17), ncol=2, handlelength=1.2,
               frameon=False, columnspacing=0.8, borderaxespad=0,
               title=f"label: % [bins with q < 0.05]; n = 2,558 bins; lag-1 genes {lg:.2f}", title_fontsize=5.1)
    return first, out


# --------------------------------------------------------------------------- #
GRANGER_EX = [("RNA", "insulation", "chr4:158500000-159000000"), ("RNA", "insulation", "chr2:232500000-233000000"),
              ("HiC", "A/B", "chr1:152000000-152500000")]   # FDR < 0.05 forward AND direction flips with rho reversed


def panel_granger_examples(fig, spec, D):
    """bins with a significant per-bin Granger test along rho (noise-matched genes, cubic detrend, 30 windows, BH over
    bins; scripts/mel_hic_granger_examples.py). Row = chromatin per spot | noise-matched gene score per spot (25-NN) |
    both detrended window series along rho (z), with the forward and reversed-rho tests"""
    from scipy.spatial import cKDTree
    T = pd.read_csv(G / "granger_per_bin.csv"); Sr = np.load(G / "granger_series.npz", allow_pickle=True)
    names = list(D["C"]["names"].astype(str)); gl = D["C"]["genes"].astype(str)
    Ss = np.asarray(D["C"]["S"], float)[cKDTree(D["xy"]).query(D["xy"], k=25)[1]].mean(1)
    gs = GridSpecFromSubplotSpec(len(GRANGER_EX), 1, subplot_spec=spec, hspace=0.6); first, st = None, {}
    for i, (m, layer, b) in enumerate(GRANGER_EX):
        key = "AB" if layer == "A/B" else "IN"; j = names.index(b)
        g3 = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[i], width_ratios=[1, 1, 1.25], wspace=1.0)
        a1, a2, a3 = (fig.add_subplot(g3[k]) for k in range(3)); first = first or a1
        v = np.asarray(D["C"][key][:, j], float); lo, hi = np.percentile(v, [2, 98])
        sc = F2.spatial(a1, D["xy"], v, cmap=CMAP["score"], vmin=lo, vmax=hi, s=0.6)
        F2.cbar(fig, a1, sc, "A/B" if key == "AB" else "insulation", on_top=True)
        g = Ss[:, j]; lo, hi = np.percentile(g, [2, 98])
        sc = F2.spatial(a2, D["xy"], g, cmap=CMAP["expression"], vmin=lo, vmax=hi, s=0.6)
        F2.cbar(fig, a2, sc, "log\nexpr.", on_top=True)
        c_, se = b.split(":"); s0, e0 = (int(x) / 1e6 for x in se.split("-"))
        a1.set_title(f"{layer} {c_} {s0:g}–{e0:g} Mb", fontsize=6.0, pad=2, color=C_HIC, loc="left")
        a2.set_title(f"genes (noise-matched)\n{', '.join(gl[j].split(',')[:4])}", fontsize=5.6, pad=2, color=C_RNA, loc="left")
        rb = Sr[f"{m}|rho"]; z = lambda x: (x - x.mean()) / (x.std() + 1e-12)
        a3.plot(rb, z(Sr[f"{m}|{key}"][:, j]), "-o", color=C_HIC, ms=2, lw=1.0, label=layer)
        a3.plot(rb, z(Sr[f"{m}|genes"][:, j]), "-o", color=C_RNA, ms=2, lw=1.0, label="genes")
        a3.axhline(0, color="#9aa2aa", lw=0.4); a3.set_xlabel(f"ρ$_{{\\mathrm{{{m}}}}}$ (30 windows)", color=AXC[m], labelpad=1)
        a3.set_ylabel("detrended (z)", labelpad=1); a3.tick_params(labelsize=5.5, pad=1)
        f_ = T[(T.axis == m) & (T.layer == layer) & (T.bin == b) & (T.rho == "forward")].iloc[0]
        r_ = T[(T.axis == m) & (T.layer == layer) & (T.bin == b) & (T.rho == "reversed")].iloc[0]
        lead = "chromatin → gene" if f_.q_c2g < f_.q_g2c else "gene → chromatin"
        qv = min(f_.q_c2g, f_.q_g2c); prv = r_.p_g2c if lead.startswith("chromatin") else r_.p_c2g
        a3.set_title(f"{lead} along ρ, q {qv:.3f}\nρ reversed: opposite direction, p {prv:.4f}", fontsize=5.6, pad=2)
        a3.legend(fontsize=5.2, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), handlelength=1.2)
        for a_ in (a1, a2): a_.text(0.5, -0.04, "per spot", transform=a_.transAxes, ha="center", va="top", fontsize=5.3)
        st[b] = dict(axis=m, layer=layer, lead=lead, q=float(qv), reversed_p=float(prv), genes=gl[j])
    return first, st


CIS_EX = [("RNA", "chr1:153000000-153500000", "epidermal (S100A7/A12, PGLYRP3/4)"),
          ("RNA", "chr3:122500000-123000000", "interferon (PARP9/14/15, DTX3L)"),
          ("HiC", "chr21:41000000-41500000", "interferon (MX1, MX2)")]
C_AB, C_IN = "#9A5A55", "#C49A5A"


def panel_cis(fig, spec, D):
    """cis consistency of gene / A/B / insulation slopes per domain vs a random background (scripts/mel_hic_cis_threeway.py):
    left = same-sign share among units where both are sloped (filled) vs the same gene with chromatin of a random genome-wide bin outside its
    cis window (open, mean +- 2 SD of 20 draws); right = cis excess per domain"""
    P = pd.read_csv(G / "cis_threeway_pairs.csv")
    lab = {"gene-AB": "gene–A/B", "gene-IN": "gene–insulation", "AB-IN": "A/B–insulation",
           "three-way (gene=A/B, insulation opposite)": "three-way"}
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[1.2, 1], wspace=0.45)
    a1 = fig.add_subplot(gs[0]); out = {}
    for i, (key, nm) in enumerate(lab.items()):
        for dx, m in ((-0.15, "RNA"), (0.15, "HiC")):
            r = P[(P.axis == m) & (P.domain == "all") & (P.pair == key)].iloc[0]
            a1.errorbar(i + dx, r.control, yerr=2 * r.ctrl_sd, fmt="o", mfc="white", mec=AXC[m], ecolor=AXC[m], ms=3.5, lw=0.7)
            a1.scatter(i + dx, r.same_sign, s=16, color=AXC[m], edgecolor="#1a1a1a", lw=0.3, zorder=3)
            zsc = (r.same_sign - r.control) / max(r.ctrl_sd, 1e-9); out[f"{m}|{nm}"] = dict(obs=round(r.same_sign, 3),
                                                                                         control=round(r.control, 3), z=round(zsc, 1))
            if zsc > 3: a1.text(i + dx, r.same_sign + 0.03, f"+{zsc:.0f}SD", fontsize=5, ha="center", color=AXC[m])
    a1.set_xticks(range(len(lab))); a1.set_xticklabels(list(lab.values()), rotation=30, ha="right", fontsize=5.6)
    a1.set_ylabel("same slope sign\n(units sloped in both)", labelpad=1); a1.set_ylim(0, 1.05)
    a1.legend(handles=[Line2D([], [], ls="", marker="o", color="#555555", mec="#1a1a1a", mew=0.3, ms=4, label="cis (same bin)"),
                       Line2D([], [], ls="", marker="o", mfc="white", mec="#555555", ms=4, label="random background ±2 SD"),
                       Line2D([], [], ls="", marker="s", color=C_RNA, ms=4, label="along ρ$_{\\mathrm{RNA}}$"),
                       Line2D([], [], ls="", marker="s", color=C_HIC, ms=4, label="along ρ$_{\\mathrm{HiC}}$")],
              fontsize=5, loc="lower left", frameon=False, ncol=2, handletextpad=0.2, columnspacing=0.6)
    a1.set_title("cis consistency per domain, all bins", fontsize=6, pad=2)
    a2 = fig.add_subplot(gs[1]); x = 0; ticks = []
    for m, nm_dom in (("RNA", D["names_dom"]), ("HiC", D["names_H"])):
        S_ = P[(P.axis == m) & (P.domain != "all")]
        for k in sorted(S_.domain.unique(), key=int):
            for dx, key, mk in ((-0.15, "gene-AB", "o"), (0.15, "three-way (gene=A/B, insulation opposite)", "D")):
                r = S_[(S_.domain == k) & (S_.pair == key)].iloc[0]
                a2.bar(x + dx, 100 * (r.same_sign - r.control), 0.28, color=DOMAIN[nm_dom[int(k)]], edgecolor="#1a1a1a", lw=0.3,
                       hatch=None if key == "gene-AB" else "////")
            ticks.append((x, nm_dom[int(k)], AXC[m])); x += 1
        x += 0.5
    a2.axhline(0, color="#1a1a1a", lw=0.5); a2.margins(y=0.12)
    a2.set_xticks([t[0] for t in ticks]); a2.set_xticklabels([t[1] for t in ticks], rotation=40, ha="right", fontsize=5.2)
    for tl, t in zip(a2.get_xticklabels(), ticks): tl.set_color(t[2])
    a2.set_ylabel("cis − background (percentage points)", labelpad=1)
    a2.legend(handles=[Patch(fc="#bbbbbb", ec="#1a1a1a", lw=0.3, label="gene–A/B"),
                       Patch(fc="#bbbbbb", ec="#1a1a1a", lw=0.3, hatch="////", label="three-way")], fontsize=5, frameon=False,
              loc="upper right")
    a2.set_title("per domain (ρ$_{\\mathrm{RNA}}$ | ρ$_{\\mathrm{HiC}}$)", fontsize=6, pad=2)
    return a1, out


def panel_cis_examples(fig, spec, D):
    """bins where gene and A/B slope the same way and insulation the opposite way, in more than one domain. Row = the
    fitted function at every spot of the bin's genes | its A/B compartment | its insulation (per-domain Gaussian
    piecewise along rho_m, F-test gate t = 0.1) | the three fitted functions (lines) over 25 equal-count windows
    (dots), each z-scored"""
    from saturn.piecewise import fit_gauss
    names = list(D["C"]["names"].astype(str)); gl = D["C"]["genes"].astype(str)
    # one grid for all rows: map | colourbar room | map | colourbar room | map | room + y label | along rho | legend
    gs = GridSpecFromSubplotSpec(len(CIS_EX), 8, subplot_spec=spec, width_ratios=[1, 0.4, 1, 0.4, 1, 0.7, 0.9, 0.5],
                                 wspace=0, hspace=0.6)
    first = None
    for i, (m, b, title) in enumerate(CIS_EX):
        j = names.index(b); d = D["d"][m]; lab = D["lab"] if m == "RNA" else D["labH"]
        q = np.quantile(d, np.linspace(0, 1, 26)); bi = np.clip(np.digitize(d, q[1:-1]), 0, 24)
        xb = np.array([d[bi == k].mean() for k in range(25)])
        a = fig.add_subplot(gs[i, 6])
        c_, se = b.split(":"); s0, e0 = (int(x) / 1e6 for x in se.split("-"))
        for c, (key, col, nm, unit, cm) in enumerate((("S", C_RNA, "genes", "log\nexpr.", "expression"),
                                                     ("AB", C_AB, "A/B", "A/B", "score"),
                                                     ("IN", C_IN, "insulation", "insulation", "score"))):
            v = np.asarray(D["C"][key][:, j], float)
            yb = np.array([v[bi == k].mean() for k in range(25)]); mu, sd = yb.mean(), yb.std()   # z of window means
            sl, it = fit_gauss(v[:, None], lab, d, t=0.1); sl, it = sl[0], it[0]; fit = it[lab] + sl[lab] * d
            am = fig.add_subplot(gs[i, 2 * c]); first = first or am
            lo, hi = np.percentile(fit, [1, 99]); hi = hi if hi > lo else lo + 1e-6
            sc = F2.spatial(am, D["xy"], fit, cmap=CMAP[cm], vmin=lo, vmax=hi, s=1.6)
            F2.cbar(fig, am, sc, unit, on_top=True)
            am.set_title(nm, fontsize=6.0, pad=2, color=col, loc="left")
            am.text(0.5, -0.04, "fitted function", transform=am.transAxes, ha="center", va="top", fontsize=5.5)
            if c == 0:
                am.text(0, 1.14, f"{c_} {s0:g}–{e0:g} Mb · {title}", transform=am.transAxes, fontsize=6.2, ha="left",
                        va="bottom", fontweight="bold")
            a.scatter(xb, (yb - mu) / sd, s=5, color=col, lw=0, alpha=0.7)
            for k in range(lab.max() + 1):
                dk = d[lab == k]; xs = np.linspace(dk.min(), dk.max(), 20)
                a.plot(xs, (it[k] + sl[k] * xs - mu) / sd, color=col, lw=1.3, label=nm if k == 0 else None)
        for bd in sorted([d[lab == k].max() for k in range(lab.max() + 1)])[:-1]:
            a.axvline(bd, color=F2.GRID, lw=0.4, ls=(0, (2, 2)))
        a.set_title(f"along ρ$_{{\\mathrm{{{m}}}}}$", fontsize=5.8, pad=2, color=AXC[m])
        a.set_xlabel(f"ρ$_{{\\mathrm{{{m}}}}}$", color=AXC[m], labelpad=1); a.tick_params(labelsize=5.3, pad=1)
        a.set_ylabel("z (25 windows)", labelpad=1)
        a.legend(fontsize=5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), handlelength=1.2)
    return first, None


def panel_maps(fig, spec, D):
    """rho_RNA | RNA domains | rho_HiC | Hi-C domains in one panel (as Figs 2-5)"""
    gs = GridSpecFromSubplotSpec(1, 4, subplot_spec=spec, wspace=0.6)
    a1 = fig.add_subplot(gs[0]); panel_rho(fig, a1, D, "RNA")
    panel_domains(fig.add_subplot(gs[1]), D)
    panel_rho(fig, fig.add_subplot(gs[2]), D, "HiC")
    out = panel_domains_hic(fig.add_subplot(gs[3]), D)
    return a1, out


def _single(fn):
    def b(fig, spec, D):
        ax = fig.add_subplot(spec); return ax, fn(ax, D)
    return b


PANELS = [
    ("a", (7.2, 1.7), panel_maps),
    ("b", (7.2, 4.65), panel_features),
    ("c", (1.8, 1.8), _single(panel_track)),
    ("d", (2.5, 2.1), _single(panel_length)),
    ("e", (2.3, 2.1), _single(panel_ctrl)),
    ("f", (3.0, 2.3), panel_direction),
    ("g", (6.0, 5.4), panel_granger_examples),
    ("h", (6.0, 2.0), panel_cis),
    ("i", (7.2, 5.6), panel_cis_examples),
]


NAMES = dict(a="isodepths_and_domains", b="fitted_functions", c="chromatin_bins_tracking_rho",
             d="granger_vs_rho_length", e="granger_same_spots", f="granger_direction", g="granger_examples",
             h="cis_threeway_consistency", i="cis_threeway_examples")


BLOCK_BOX = dict(left=0.06, right=0.99, top=0.92, bottom=0.07)     # the fitted-function block needs the full print width


def export_panels(D, outdir=OUT.parent / "panels"):
    """every sub-panel as its own named SVG (fig6<letter>_<name>.svg); dense spots rasterized inside, text vector"""
    outdir.mkdir(parents=True, exist_ok=True)
    for L, size, build in PANELS:
        fig = plt.figure(figsize=size)
        box = BLOCK_BOX if L in "bi" else dict(left=0.2, right=0.82, top=0.84, bottom=0.2)
        spec = GridSpec(1, 1, figure=fig, **box)[0]
        build(fig, spec, D); save_panel_svg(fig, outdir / f"fig6{L}_{NAMES[L]}")
        if L == "i":
            fig.savefig(outdir / f"fig6{L}_{NAMES[L]}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
    fig = plt.figure(figsize=(7.2, 4.65))                          # the fitted-function block on its own, rows lettered c-e
    spec = GridSpec(1, 1, figure=fig, **BLOCK_BOX)[0]
    panel_features(fig, spec, D, letters="cde")
    save_panel_svg(fig, outdir / "fig6cde_fitted_functions")
    fig.savefig(outdir / "fig6cde_fitted_functions.png", dpi=600, bbox_inches="tight", pad_inches=0.03); plt.close(fig)
    print(f"-> {len(PANELS) + 1} panels in {outdir}")


def main():
    apply(base_pt=6.5)
    D = load()
    fig = plt.figure(figsize=(7.2, 21.7))
    outer = GridSpec(6, 1, figure=fig, height_ratios=[1.45, 4.35, 1.75, 5.2, 2.0, 5.1], hspace=0.25,
                     left=0.08, right=0.96, top=0.985, bottom=0.03)
    g3 = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[2], width_ratios=[0.75, 1.05, 1.0, 1.1], wspace=0.75)
    where = dict(a=outer[0], b=outer[1], c=g3[0], d=g3[1], e=g3[2], f=g3[3], g=outer[3], h=outer[4], i=outer[5])
    nudge = dict(a=(-0.03, 0.015), b=(-0.06, 0.02), c=(-0.08, 0.02), d=(-0.07, 0.02), e=(-0.07, 0.02), f=(-0.12, 0.02),
                 g=(-0.06, 0.015), h=(-0.07, 0.02), i=(-0.07, 0.02))
    res = {}
    for L, _, build in PANELS:
        ax, out = build(fig, where[L], D); res[L] = out; F2.letter(ax, L, *nudge[L])
    F2.draw_letters(fig)
    manifest = dict(dataset=DS, domains_RNA={int(k): v for k, v in D["names_dom"].items()}, domains_HiC=res["a"],
                    bins_tracking=res["c"], fitted_functions=res["b"], length_all_spots=res["d"],
                    length_same_spots=res["e"], direction=res["f"], granger_examples=res["g"], cis_threeway=res["h"],
                    notes=dict(tracking="Spearman of rank residuals on rank(Hi-C reads); 200-rotation null residualised the same way",
                               granger="lags 1-3, Bonferroni over lags x directions, cubic detrend on rho, circular-shift null; "
                                       "genes averaged over 25 spatial neighbours to match chromatin smoothness",
                               direction="chromatin->gene asymmetry persists with rho reversed and shrinks when gene noise is "
                                         "matched: attributed to noise asymmetry, not to a lead along rho"),
                    sources=[str(G), str(R / DS)])
    save_dual(fig, OUT, extra=manifest); save_dual(fig, FINAL / "fig6_melanoma_hic", extra=manifest); plt.close(fig)
    export_panels(D)
    print(json.dumps(dict(domains_RNA=manifest["domains_RNA"], domains_HiC=res["a"], b=res["b"]), indent=1, default=str))


if __name__ == "__main__":
    main()
