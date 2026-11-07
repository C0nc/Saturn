"""Figure 3 — Melanoma (spatial RNA + ATAC): two axes, co-patterned chromatin, and a histology-
sharpened tumour/stroma interface.

Saturn fit `melanoma_tissue` (in-tissue QC, log-depth regressed from the RNA target).

  a  adjacent H&E with the DBiT field of view registered onto it (similarity transform fitted
     on tissue masks; 0.137 um/px H&E, 40 um DBiT pitch)
  b  fit quality: R2 full / own axis / cross axis
  c  modality contribution per feature: unique-own / shared / unique-cross (commonality)
  d  rho_RNA, RNA domains, rho_ATAC, ATAC domains (labels from composition, asserted)
  e  representative genes (TYR, KRT10) on rho_RNA and peaks (TYR -10 kb, PHGDH +12.7 kb) on rho_ATAC: original
     per-spot map | fitted function at every spot | bin means + fitted function along the own axis (as Fig 2e)
  f  process genes and their top function-matched peaks along rho_RNA (genome-wide; mostly trans): fitted functions
     (lines) over the observed values (dots = 20 equal-count bins of the smoothed z)
  g  chromatin potential examples (DORC vs gene, SHARE-seq style), vs the gene's own random-peak band
  h  tumour - stroma contrast at spot resolution vs after iStar super-resolution (8 um grid, adjacent H&E)
  i  zoom on a strong tumour/stroma interface
  j  the super-resolved interface is sharper (10-90% width across strong interfaces, spot resolution vs iStar) and
     accurate (iStar vs measured contrast at 10% held-out spots, all and at interfaces; no other upsampler is compared)

Spot maps are drawn in the registered image frame (x = BF column, y = -BF row) so they line up
with the H&E panels. Domain colours are keyed by annotated label.
"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, cv2
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter, minimum_filter, map_coordinates
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle, Patch
from PIL import Image

sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/_style")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/saturn_repo/src")       # Saturn package (saturn_repo)
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/published_figures/fig2")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
from paper_style import apply, save_dual, MODALITY, DOMAIN, PROGRAM, CMAP
import fig2_lpc as F2                                   # shared panel builders (spatial, cbar, feature fn)

H = "/mnt/HDD2B/Users/yancui"
R = Path(f"{H}/Hole/gaston_mix_mm/results/real_saturn")
DS = "melanoma_tissue"
IST = R / "melanoma_tissue_istar_feat"
ISTH = R / "melanoma_tissue_istar_feat_holdout10"
CPD = R / "melanoma_tissue_chromatin_potential"
OUT = Path(f"{H}/Saturn/published_figures/fig3/fig3_melanoma")
FINAL = Path(f"{H}/Saturn/final_fig3")
Image.MAX_IMAGE_PIXELS = None
FEATS = [("RNA", "TYR", "TYR", "melanocytic"), ("RNA", "KRT10", "KRT10", "epidermis"),
         ("ATAC", "chr11:89165000-89170000", "TYR −10 kb enhancer", "melanocytic chromatin"),
         ("ATAC", "chr1:119670000-119675000", "PHGDH +12.7 kb enhancer", "serine synthesis")]
PEAK_STYLE = [("o", (0, (3, 1.2))), ("s", (0, (1, 1))), ("^", (0, (4, 1, 1, 1)))]   # matched peak 1/2/3
MATCH_GENES = [("TYR", "melanocytic"), ("STAT1", "interferon"), ("KRT10", "epidermis"), ("FN1", "stroma")]
CP_EX = [("STAT1", 0, "ahead"), ("SERPINE1", 0, "behind")]
TUM = ["TYR", "PMEL", "MLANA", "DCT", "SOX10", "MITF"]; STR = ["COL1A1", "COL1A2", "COL3A1"]
C_RNA, C_ATAC = MODALITY["RNA"], MODALITY["ATAC"]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load():
    from saturn.domains import gaston_domains
    from domain_annot import CFG
    from domain_k import kdom
    ax = pd.read_csv(R / DS / "axes.csv", index_col=0)
    z = dict(np.load(R / f"targets_{DS}.npz", allow_pickle=True))
    pos = pd.read_csv(f"{H}/Yufan_Melanoma/spatial/tissue_positions_list.csv", header=None,
                      names=["bc", "in", "r", "c", "py", "px"]).set_index("bc").loc[ax.index]
    D = dict(bc=[str(b) for b in ax.index], xy=np.c_[pos.px.values, -pos.py.values],
             d={"RNA": ax.d_RNA.values, "ATAC": ax.d_ATAC.values},
             lab={m: gaston_domains(ax[f"d_{m}"].values, z[f"Y_{m}"], n_domains=kdom(DS, m)) for m in ("RNA", "ATAC")},
             summary=json.load(open(R / DS / "summary.json")), z=z, pos=pos)
    idx = pd.Index(D["bc"])
    r = CFG[DS]["rna"]()[idx]; a = CFG[DS]["atac"]()[idx]
    D["X"] = {"RNA": sp.csc_matrix(r.X).astype(float), "ATAC": sp.csc_matrix(a.X).astype(float)}
    D["names"] = {"RNA": np.array([str(v) for v in r.var_names]), "ATAC": np.array([str(v) for v in a.var_names])}
    D["lib"] = {m: np.asarray(D["X"][m].sum(1)).ravel().clip(1) for m in D["X"]}
    D["names_dom"] = label_domains(D)
    return D


def _lcp(D, m, name):
    j = int(np.where(D["names"][m] == name)[0][0])
    return np.log1p(np.asarray(D["X"][m][:, j].todense()).ravel() / D["lib"][m] * 1e4)


def label_domains(D):
    out = {}
    lab = D["lab"]["RNA"]; K = list(range(lab.max() + 1))
    mean = lambda v, L, k: float(np.mean(v[L == k]))
    krt, pmel, stat, col, jun = (_lcp(D, "RNA", g) for g in ("KRT10", "PMEL", "STAT1", "COL1A1", "JUN"))
    epi = max(K, key=lambda k: mean(krt, lab, k)); K.remove(epi)
    ifn = max(K, key=lambda k: mean(stat + col, lab, k)); K.remove(ifn)
    mel = max(K, key=lambda k: mean(pmel, lab, k)); K.remove(mel)
    st = K[0]
    assert mean(jun, lab, st) >= min(mean(jun, lab, k) for k in (epi, mel)), "AP-1 domain should be JUN-high"
    out["RNA"] = {epi: "epidermis", mel: "melanocytic tumour", ifn: "IFN-stromal", st: "AP-1 stress"}

    lab = D["lab"]["ATAC"]; K = list(range(lab.max() + 1))
    imm = max(K, key=lambda k: np.median(D["z"]["ann_immune_marker"][lab == k])); K.remove(imm)
    phg = _lcp(D, "ATAC", "chr1:119670000-119675000")
    met = max(K, key=lambda k: mean(phg, lab, k)); K.remove(met)
    if len(K) == 1:                                   # elbow K = 3: the low-signal domain (RNA depth 0.26x) is merged
        out["ATAC"] = {imm: "immune-accessible", met: "metabolic tumour", K[0]: "mixed tumour-stroma"}
    else:
        low, mix = sorted(K, key=lambda k: (lab == k).sum())
        out["ATAC"] = {imm: "immune-accessible", met: "metabolic tumour", low: "tumour, low-signal", mix: "mixed tumour-stroma"}
    return out


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #
def melanocytic_score(D, k=8):
    """mean z of the melanocytic genes (TUM) per spot, averaged over each spot and its 7 nearest neighbours"""
    X = np.column_stack([_lcp(D, "RNA", g) for g in TUM])
    return F2.smooth(((X - X.mean(0)) / X.std(0)).mean(1), D["xy"], k=k)


def he_crop(x0, x1, y0, y1, step=8):
    """H&E at 1/step of full resolution for a window given in thumb (1/32) px; cached next to the thumb"""
    f = R / "melanoma_tissue_istar" / f"he_fov_step{step}.png"
    if not f.exists():
        import tifffile, zarr
        zz = zarr.open(tifffile.TiffFile(f"{H}/Yufan_Melanoma/HE_HMelanoma.tif").aszarr(), mode="r")
        Image.fromarray(np.asarray(zz[y0 * 32:y1 * 32:step, x0 * 32:x1 * 32:step])).save(f)
    return np.asarray(Image.open(f))


def panel_he(fig, spec, D):
    """left: whole adjacent H&E with the registered DBiT field box; arrow; right: the field enlarged, spots = melanocytic score"""
    from matplotlib.patches import ConnectionPatch
    geo = json.load(open(IST / "geometry.json"))
    Fm = np.array(geo["registration"]["bf_to_he_full"])
    thumb = R / "melanoma_tissue_istar" / "he_thumb32.png"
    if not thumb.exists():
        import tifffile, zarr
        zz = zarr.open(tifffile.TiffFile(f"{H}/Yufan_Melanoma/HE_HMelanoma.tif").aszarr(), mode="r")
        Image.fromarray(np.asarray(zz[::32, ::32])).save(thumb)
    he = np.asarray(Image.open(thumb))
    box = (np.array([[0, 0, 1], [6382, 0, 1], [6382, 6382, 1], [0, 6382, 1], [0, 0, 1]]) @ Fm.T)[:, :2] / 32
    L = (np.c_[D["pos"].px.values, D["pos"].py.values, np.ones(len(D["pos"]))] @ Fm.T)[:, :2] / 32
    v = melanocytic_score(D); lim = np.floor(np.percentile(np.abs(v), 98) * 10) / 10
    g = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[1, 1], wspace=0.28)
    a0 = fig.add_subplot(g[0]); a1 = fig.add_subplot(g[1])
    a0.imshow(he)
    a0.plot(box[:, 0], box[:, 1], color="#111111", lw=0.8)
    bar = 2000 / geo["um_he"] / 32                                  # 2 mm scale bar
    a0.plot([20, 20 + bar], [he.shape[0] - 25] * 2, color="k", lw=1.5)
    a0.text(20 + bar / 2, he.shape[0] - 32, "2 mm", ha="center", va="bottom", fontsize=5.5)
    a0.set_title("adjacent H&E, registered DBiT field", fontsize=6, pad=2)
    pad = 12
    x0, y0 = np.floor(box.min(0) - pad).astype(int).clip(0); x1, y1 = np.ceil(box.max(0) + pad).astype(int)
    x1, y1 = min(x1, he.shape[1]), min(y1, he.shape[0])
    a1.imshow(he_crop(x0, x1, y0, y1), extent=(x0, x1, y1, y0), interpolation="bilinear")
    sc = a1.scatter(L[:, 0], L[:, 1], c=v, cmap=CMAP["score"], vmin=-lim, vmax=lim, s=1.9, lw=0, rasterized=True)
    a1.plot(box[:, 0], box[:, 1], color="#111111", lw=0.8)
    a1.set_xlim(x0, x1); a1.set_ylim(y1, y0)
    a1.set_title("melanocytic score", fontsize=6, pad=2)
    F2.cbar(fig, a1, sc, "mean z, 6 genes", ticks=[-lim, 0, lim])
    for a_ in (a0, a1):
        a_.set_xticks([]); a_.set_yticks([])
        for s in a_.spines.values(): s.set_visible(False)
    fig.add_artist(ConnectionPatch(xyA=(box[:4, 0].max(), box[:4, 1].mean()), coordsA=a0.transData, xyB=(-0.02, 0.5),
                                   coordsB=a1.transAxes, arrowstyle="-|>", mutation_scale=7, lw=0.9, color="#111111"))
    return a0, dict(melanocytic_score_lim=float(lim), genes=TUM)


def panel_owncross(ax):
    T = pd.read_csv(R / f"{DS}_own_vs_cross" / "own_vs_cross.csv")
    mods = ["RNA", "ATAC"]; comps = [("unique_own", 1.0, "unique own axis"), ("shared", 0.55, "shared"),
                                     ("unique_cross", 0.25, "unique cross axis")]
    for i, m in enumerate(mods):
        s = T[T.modality == m]; bottom = 0
        for c, al, _ in comps:
            v = float(np.clip(s[c], 0, None).median())
            ax.bar(i, v, 0.6, bottom=bottom, color=MODALITY[m], alpha=al, edgecolor="#1a1a1a", lw=0.4)
            bottom += v
        ax.text(i, bottom + 0.01, f"own>cross\n{100 * (s.R2_own > s.R2_cross).mean():.0f}%", ha="center",
                va="bottom", fontsize=5.5)
    ax.set_xticks(range(2)); ax.set_xticklabels(mods)
    for tl, m in zip(ax.get_xticklabels(), mods): tl.set_color(MODALITY[m])
    ax.set_ylabel("median R² per feature"); ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 0.62)
    ax.legend(handles=[Patch(fc="#555555", alpha=a_, ec="#1a1a1a", lw=0.4, label=l_) for _, a_, l_ in comps],
              fontsize=5.2, loc="upper right", ncol=1, handlelength=0.9, handleheight=0.8, borderaxespad=0.1,
              labelspacing=0.2)
    ax.set_ylim(0, 0.75)


def fitted_curves(D, m, axis, names):
    """gene/peak curves exactly as the chromatin-potential scripts: kNN-15 smoothed z, Gaussian piecewise fit"""
    from saturn_cis_function_pairs import smooth, zz, curves, NGRID
    from saturn.piecewise import fit_gauss
    M = np.column_stack([_lcp(D, m_, n) for m_, n in names])
    Z = zz(smooth(M, D["xy"]))
    d = D["d"][axis]; lab = D["lab"][axis]
    grid = np.linspace(np.percentile(d, 1), np.percentile(d, 99), NGRID)
    sl, it = fit_gauss(Z, lab, d, t=0.1)
    q = np.array_split(np.argsort(d), 20)                        # observed (smoothed z) per equal-count bin
    obs = (np.array([d[g].mean() for g in q]), np.array([Z[g].mean(0) for g in q]))
    return grid, curves(sl, it, d, lab, grid), obs


def domain_bands(ax, D, axis, grid, alpha=0.12):
    lab, d = D["lab"][axis], D["d"][axis]
    kg = lab[np.argmin(np.abs(d[None, :] - grid[:, None]), axis=1)]
    for k in np.unique(kg):
        g = grid[kg == k]
        ax.axvspan(g.min(), g.max(), color=DOMAIN[D["names_dom"][axis][k]], alpha=alpha, lw=0)


def panel_matched(fig, spec, D):
    from continuous_marker_panels import nearest, GTF_HS
    Mt = pd.read_csv(CPD / "function_matched_peaks.csv")
    gs = GridSpecFromSubplotSpec(1, len(MATCH_GENES), subplot_spec=spec, wspace=0.35)
    first, stats = None, {}
    for i, (g, prog) in enumerate(MATCH_GENES):
        top = Mt[(Mt.axis == "RNA") & (Mt.gene == g)].sort_values("rank").head(3)
        grid, C, (bx, by) = fitted_curves(D, None, "RNA", [("RNA", g)] + [("ATAC", p) for p in top.peak])
        coord = pd.Series(top.peak.values).str.extract(r"(chr[\w]+):(\d+)-(\d+)")
        ng, _ = nearest(coord, GTF_HS)
        a = fig.add_subplot(gs[i]); first = first or a
        domain_bands(a, D, "RNA", grid)
        a.plot(grid, C[:, 0], color=PROGRAM[prog], lw=1.6, label=g, zorder=3)
        a.scatter(bx, by[:, 0], s=5, color=PROGRAM[prog], lw=0.3, edgecolor="#1a1a1a", zorder=4)
        hs = []
        for j in range(len(top)):                 # one shape + dash per matched peak, same ATAC colour
            mk, ls = PEAK_STYLE[j]
            a.plot(grid, C[:, j + 1], color=C_ATAC, lw=0.7, ls=ls, alpha=0.9, zorder=2)
            a.scatter(bx, by[:, j + 1], s=5, marker=mk, facecolor="white", edgecolor=C_ATAC, lw=0.5, zorder=2)
            hs.append(Line2D([], [], color=C_ATAC, lw=0.7, ls=ls, marker=mk, ms=2.6, mfc="white", mec=C_ATAC, mew=0.5,
                             label=f"{coord[0][j]} {int(coord[1][j]) / 1e6:.0f} Mb r {top.similarity.iloc[j]:.2f}"))
        a.set_title(f"{g}  ({prog})", fontsize=6.3, pad=2, color=PROGRAM[prog])
        lo, hi = a.get_ylim(); a.set_ylim(lo, hi + 0.5 * (hi - lo))          # headroom for the peak key
        lg = a.legend(handles=hs, fontsize=5, loc="upper left", frameon=True, handlelength=1.3, handletextpad=0.2,
                      labelspacing=0.12, borderaxespad=0.15, borderpad=0.2, framealpha=0.85, edgecolor="none")
        for t in lg.get_texts():
            t.set_color(C_ATAC)
        a.set_xlabel("ρ$_{\\mathrm{RNA}}$", color=C_RNA, labelpad=1); a.tick_params(labelsize=5.5, pad=1)
        if i == 0: a.set_ylabel("fitted value (z)", labelpad=1)
        stats[g] = dict(top_peaks=top.peak.tolist(), similarity=top.similarity.round(4).tolist(),
                        relation=top.relation.tolist(), nearest_gene=list(ng))
    return first, stats


def panel_cp(fig, spec, D):
    cv = np.load(CPD / "dorc_curves_RNA.npz", allow_pickle=True)
    genes = list(cv["genes"]); grid = cv["grid"]; kg = cv["kgrid"]
    P = pd.read_csv(CPD / "dorc_cp_pairs.csv")
    gs = GridSpecFromSubplotSpec(1, len(CP_EX), subplot_spec=spec, wspace=0.6)
    first, stats = None, {}
    for i, (g, k, word) in enumerate(CP_EX):
        a = fig.add_subplot(gs[i]); first = first or a
        j = genes.index(g); q = cv["random_q"][:, j, :]
        domain_bands(a, D, "RNA", grid, alpha=0.08)
        sel = kg == k
        a.axvspan(grid[sel].min(), grid[sel].max(), color=DOMAIN[D["names_dom"]["RNA"][k]], alpha=0.22, lw=0)
        a.fill_between(grid, q[:, 0], q[:, 2], color="#9aa2aa", alpha=0.35, lw=0, label="random peak sets (5-95%)")
        a.plot(grid, cv["gene"][:, j], color=C_RNA, lw=1.5, label=f"{g} expression")
        a.plot(grid, cv["dorc"][:, j], color=C_ATAC, lw=1.5, label=f"{g} DORC accessibility")
        r_ = P[(P.axis == "RNA") & (P.gene == g) & (P.domain == k)].iloc[0]
        pct = 100 * r_.pct_vs_random if word == "ahead" else 100 * (1 - r_.pct_vs_random)
        a.set_title(f"{g}\nchromatin {word}\nnet CP {r_.net:+.2f}\n({'>' if word == 'ahead' else '<'}{pct:.0f}% of random)",
                    fontsize=5.6, pad=2)
        lo_, hi_ = grid[sel].min(), grid[sel].max(); w_ = hi_ - lo_
        a.set_xlim(lo_ - 0.6 * w_, hi_ + 0.6 * w_)
        m_ = (lo_ + hi_) / 2; yv = np.interp(m_, grid, cv["dorc"][:, j])
        a.annotate("", xy=(m_ + r_.cp_dorc * w_, yv), xytext=(m_, yv),
                   arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#111111", mutation_scale=6))
        a.set_xlabel("ρ$_{\\mathrm{RNA}}$", color=C_RNA, labelpad=1); a.tick_params(labelsize=5.5, pad=1)
        if i == 0: a.set_ylabel("fitted value (z)", labelpad=1)
        if i == len(CP_EX) - 1:
            h_, l_ = a.get_legend_handles_labels()
            a.legend(h_, ["random peak sets 5-95%", "expression", "DORC accessibility"], fontsize=5.1,
                     loc="upper left", bbox_to_anchor=(1.02, 1.0), handlelength=1.2, borderaxespad=0, labelspacing=0.3)
        stats[g] = dict(domain=int(k), domain_label=D["names_dom"]["RNA"][k], net_cp=float(r_.net),
                        cp_dorc=float(r_.cp_dorc), cp_random_median=float(r_.cp_random_median),
                        pct_vs_random=float(r_.pct_vs_random), n_dorc_peaks=int(r_.n_peaks),
                        dorc_peaks=str(cv["dorc_peaks"][j]))
    return first, stats


# ---- iStar ------------------------------------------------------------------
def ts_maps():
    z = np.load(IST / "tumour_stroma_maps.npz")
    return z["iStar"], z["interpolated"], z["tissue"]


def strong_interface_profiles(Ti, Tl, tis, R_=6):
    ok = tis & np.isfinite(Tl) & np.isfinite(Ti)
    Tf = np.where(ok, Tl, 0.0)
    hi = maximum_filter(np.where(ok, Tl, -9), size=2 * R_ + 1); lo = minimum_filter(np.where(ok, Tl, 9), size=2 * R_ + 1)
    s = np.sign(Tf)
    E = ok & ((np.roll(s, 1, 0) != s) | (np.roll(s, 1, 1) != s)) & np.roll(ok, 1, 0) & np.roll(ok, 1, 1) & (hi > 1) & (lo < -1)
    gy, gx = np.gradient(Tf); t = np.arange(-R_, R_ + 0.01, 0.25)
    W = {"interpolated": [], "iStar": []}; prof = {"interpolated": [], "iStar": []}
    for y, x in zip(*np.where(E)):
        g = np.array([gy[y, x], gx[y, x]]); n = np.linalg.norm(g)
        if n < 1e-6: continue
        dvec = g / n; py, px = y + t * dvec[0], x + t * dvec[1]
        pr = {k: map_coordinates(np.where(ok, T, np.nan), [py, px], order=1, mode="nearest") for k, T in (("interpolated", Tl), ("iStar", Ti))}
        if any(np.isnan(v).any() for v in pr.values()): continue
        ww = {}
        for k, v in pr.items():
            lo_, hi_ = np.percentile(v, 5), np.percentile(v, 95); a_ = lo_ + 0.1 * (hi_ - lo_); b_ = lo_ + 0.9 * (hi_ - lo_)
            if not v[0] < v[-1]: break
            i1, i2 = np.argmax(v >= a_), np.argmax(v >= b_)
            if i2 <= i1: break
            ww[k] = (t[i2] - t[i1]) * 8.0
        if len(ww) == 2:
            for k in ww: W[k].append(ww[k]); prof[k].append(pr[k])
    return E, t * 8.0, {k: np.array(v) for k, v in W.items()}, {k: np.array(v) for k, v in prof.items()}


def heldout_ts():
    L = pd.read_csv(ISTH / "locs-all.tsv", sep="\t", index_col=0)
    C = pd.read_csv(ISTH / "cnts-all.tsv", sep="\t", index_col=0).loc[L.index]
    held = pd.read_csv(ISTH / "heldout_spots.txt", header=None)[0].astype(str).values
    te = L.index.isin(held); tr = ~te; xy = L[["x", "y"]].values
    nb = cKDTree(xy[tr]).query(xy, k=6)[1]
    i = (L.y.values / 16).astype(int); j = (L.x.values / 16).astype(int)
    Zo, Zi, Zk, Zl = {}, {}, {}, {}
    for g in TUM + STR:
        n = "RNA_" + g; v = C[n].values; mu, sd_ = v[tr].mean(), v[tr].std()
        y = pickle.load(open(ISTH / "cnts-super" / f"{n}.pickle", "rb"))
        p = np.array([y[max(a - 1, 0):a + 2, max(b - 1, 0):b + 2].mean() for a, b in zip(i, j)])
        a_, b_ = np.linalg.lstsq(np.c_[p[tr], np.ones(tr.sum())], v[tr], rcond=None)[0]
        li = griddata(xy[tr], v[tr], xy, method="linear"); li = np.where(np.isnan(li), griddata(xy[tr], v[tr], xy, method="nearest"), li)
        for Dd, val in ((Zo, v), (Zi, a_ * p + b_), (Zk, v[tr][nb].mean(1)), (Zl, li)): Dd[g] = (val - mu) / sd_
    sc = lambda Dd: np.mean([Dd[g] for g in TUM], 0) - np.mean([Dd[g] for g in STR], 0)
    obs, ist, knn, lin = sc(Zo), sc(Zi), sc(Zk), sc(Zl)
    nbt = cKDTree(xy[tr]).query(xy, k=9)[1]; o_tr = obs[tr]; lo, hi = np.percentile(o_tr, [30, 70])
    bnd = np.array([(o_tr[k] < lo).any() and (o_tr[k] > hi).any() for k in nbt])
    rng = np.random.default_rng(0); res = {}
    for lab_, m in (("all", te), ("boundary", te & bnd), ("interior", te & ~bnd)):
        idx = np.where(m)[0]; r = lambda v_, s_: np.corrcoef(v_[idx[s_]], obs[idx[s_]])[0, 1]
        B = rng.integers(0, len(idx), (2000, len(idx))); full = np.arange(len(idx))
        res[lab_] = dict(n=int(m.sum()), **{k: float(r(v_, full)) for k, v_ in (("iStar", ist), ("knn6", knn), ("linear", lin))},
                         ci=[float(np.percentile([r(ist, s_) - r(knn, s_) for s_ in B], q)) for q in (2.5, 97.5)],
                         ci_iStar=[float(np.percentile([r(ist, s_) for s_ in B], q)) for q in (2.5, 97.5)])
    return res


def panel_istar_maps(fig, spec):
    Ti, Tl, tis = ts_maps(); he = np.asarray(Image.open(IST / "he.jpg"))
    geo = json.load(open(IST / "geometry.json")); win = np.array(geo["window_iStar"]); G = Ti.shape[0]
    E, _, _, _ = strong_interface_profiles(Ti, Tl, tis)
    dens = cv2.boxFilter(E.astype(float), -1, (31, 31)); cy, cx = np.unravel_index(np.argmax(dens), dens.shape)
    zoom = ((cx - 28) * 16, (cx + 28) * 16, (cy - 28) * 16, (cy + 28) * 16)
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.06)
    xs = np.arange(G) * 16 + 8
    axs = []
    for c, (T, t) in enumerate([(Tl, "spot resolution (40 µm)"), (Ti, "iStar super-resolution (8 µm)")]):
        a = fig.add_subplot(gs[c]); axs.append(a)
        a.imshow(he, alpha=0.3)
        im = a.imshow(np.where(tis, T, np.nan), cmap=CMAP["score"], vmin=-2, vmax=2, extent=(0, G * 16, G * 16, 0), interpolation="nearest")
        a.contour(xs, xs, np.where(tis, T, np.nan), levels=[0], colors="k", linewidths=0.35)
        a.add_patch(Rectangle((zoom[0], zoom[2]), zoom[1] - zoom[0], zoom[3] - zoom[2], fill=False, ec="#111111", lw=0.8))
        a.set_xlim(win[:, 0].min(), win[:, 0].max()); a.set_ylim(win[:, 1].max(), win[:, 1].min())
        a.set_xticks([]); a.set_yticks([]); [s_.set_visible(False) for s_ in a.spines.values()]
        a.set_title(t, fontsize=6.3, pad=2)
    cax = axs[1].inset_axes([1.02, 0.1, 0.04, 0.8]); cb = fig.colorbar(im, cax=cax, ticks=[-2, 0, 2])
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.5, length=1.5)
    cb.set_label("tumour − stroma (z)", fontsize=5.5, labelpad=1)
    return axs[0], dict(zoom_px=list(map(int, zoom)))


def panel_istar_zoom(fig, spec, zoom):
    Ti, Tl, tis = ts_maps(); he = np.asarray(Image.open(IST / "he.jpg")); G = Ti.shape[0]; xs = np.arange(G) * 16 + 8
    gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.06)
    axs = []
    for c, (T, t) in enumerate([(None, "H&E"), (Tl, "spots"), (Ti, "iStar")]):
        a = fig.add_subplot(gs[c]); axs.append(a)
        a.imshow(he, alpha=1.0 if T is None else 0.45)
        if T is not None:
            a.imshow(np.where(tis, T, np.nan), cmap=CMAP["score"], vmin=-2, vmax=2, extent=(0, G * 16, G * 16, 0),
                     interpolation="nearest", alpha=0.8)
            a.contour(xs, xs, np.where(tis, T, np.nan), levels=[0], colors="k", linewidths=0.7)
        a.set_xlim(zoom[0], zoom[1]); a.set_ylim(zoom[3], zoom[2]); a.set_xticks([]); a.set_yticks([])
        for s_ in a.spines.values(): s_.set_linewidth(0.8)
        a.set_title(t, fontsize=6.3, pad=2)
    bar = 100 / 0.5                                                   # 100 um at 0.5 um/px
    axs[0].plot([zoom[0] + 20, zoom[0] + 20 + bar], [zoom[3] - 25] * 2, color="k", lw=1.5)
    axs[0].text(zoom[0] + 20 + bar / 2, zoom[3] - 35, "100 µm", ha="center", va="bottom", fontsize=5.5)
    return axs[0]


def panel_istar_stats(fig, spec):
    Ti, Tl, tis = ts_maps()
    E, tt, W, prof = strong_interface_profiles(Ti, Tl, tis)
    ho = heldout_ts()
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.55)
    a = fig.add_subplot(gs[0])
    for k, col, nm in (("interpolated", "#9aa2aa", "spot resolution"), ("iStar", C_RNA, "iStar")):
        P = prof[k]; P = (P - P.min(1, keepdims=True)) / np.maximum(np.ptp(P, 1, keepdims=True), 1e-9)
        a.plot(tt, np.median(P, 0), color=col, lw=1.5, label=f"{nm}  {np.median(W[k]):.0f} µm")
        a.fill_between(tt, np.percentile(P, 25, 0), np.percentile(P, 75, 0), color=col, alpha=0.2, lw=0)
    p = mannwhitneyu(W["iStar"], W["interpolated"], alternative="less").pvalue
    a.set_xlabel("distance across interface (µm)", labelpad=1); a.set_ylabel("stroma → tumour (scaled)", labelpad=1)
    a.set_title(f"10–90% width, n={len(W['iStar'])} profiles\np={p:.1g}", fontsize=5.8, pad=2)
    a.legend(fontsize=5.2, loc="upper left", handlelength=1.2, borderaxespad=0.1)
    a.tick_params(labelsize=5.5, pad=1)
    b = fig.add_subplot(gs[1]); x = np.arange(2)
    for xi, s in zip(x, ("all", "boundary")):                     # iStar alone: accuracy at spots it never saw
        lo, hi = ho[s]["ci_iStar"]
        b.bar(xi, ho[s]["iStar"], 0.55, color=C_RNA, edgecolor="#1a1a1a", lw=0.4)
        b.errorbar(xi, ho[s]["iStar"], yerr=[[ho[s]["iStar"] - lo], [hi - ho[s]["iStar"]]], color="#1a1a1a", lw=0.7, capsize=2)
        b.text(xi, hi + 0.03, f"r {ho[s]['iStar']:.2f}", ha="center", va="bottom", fontsize=5.3)
    b.set_xticks(x); b.set_xticklabels([f"all\n(n={ho['all']['n']})", f"interface\n(n={ho['boundary']['n']})"], fontsize=5.5)
    b.set_ylabel("iStar vs measured, held-out r", labelpad=1); b.tick_params(labelsize=5.5, pad=1)
    b.set_ylim(0, 1.05); b.set_yticks([0, 0.5, 1.0]); b.set_xlim(-0.6, 1.6)
    b.set_title("accuracy at unseen spots\n(95% bootstrap CI)", fontsize=5.8, pad=2)
    return a, dict(width_um={k: float(np.median(v)) for k, v in W.items()}, width_p=float(p),
                   n_profiles=int(len(W["iStar"])), heldout=ho)


# --------------------------------------------------------------------------- #
def _outside_band():
    """gene-domain DORC cases outside their own random-set 90% band, vs the 10% expected by chance"""
    from scipy.stats import binomtest
    P = pd.read_csv(CPD / "dorc_cp_pairs.csv"); k = int(((P.pct_vs_random >= 0.95) | (P.pct_vs_random <= 0.05)).sum())
    return f"{k}/{len(P)} ({100 * k / len(P):.1f}%) vs 10% chance, binomial p {binomtest(k, len(P), 0.1, alternative='greater').pvalue:.1e}"


D_GLOBAL = {}


def _single(fn):
    def b(fig, spec, D):
        ax = fig.add_subplot(spec); return ax, fn(ax, D)
    return b


def build_d(fig, spec, D):
    return F2.panel_maps(fig, spec, D)[0][0], None


def build_e(fig, spec, D):
    F2.FEATS = FEATS
    return F2.build_features(fig, spec, D)


PANELS = [
    ("a", (3.4, 1.55), panel_he),
    ("b", (1.45, 1.55), _single(lambda ax, D: (F2.panel_fit(ax, D["summary"]), ax.set_ylim(0, 1.0),
                                                ax.set_yticks([0, 0.5, 1.0])))),
    ("c", (1.35, 1.55), _single(lambda ax, D: panel_owncross(ax))),
    ("d", (7.2, 1.5), build_d),
    ("e", (7.2, 2.0), build_e),
    ("f", (4.6, 1.55), panel_matched),
    ("g", (2.5, 1.55), panel_cp),
    ("h", (3.3, 1.75), lambda fig, spec, D: panel_istar_maps(fig, spec)),
    ("i", (2.1, 0.9), None),                           # needs the zoom box from h; built in main
    ("j", (2.6, 1.3), lambda fig, spec, D: panel_istar_stats(fig, spec)),
]


def export_panels(D, outdir, zoom):
    """each panel as its own 600 dpi PNG + SVG, same builders as the assembled figure"""
    outdir.mkdir(parents=True, exist_ok=True)
    for L, size, build in PANELS:
        fig = plt.figure(figsize=size)
        spec = GridSpec(1, 1, figure=fig, left=0.12, right=0.88, top=0.84, bottom=0.18)[0]
        if L == "i":
            panel_istar_zoom(fig, spec, zoom)
        else:
            build(fig, spec, D)
        save_dual(fig, outdir / f"fig3{L}", write_manifest=False); plt.close(fig)
    print(f"-> {len(PANELS)} panels in {outdir}")


def main():
    apply(base_pt=6.5)
    D = load(); D_GLOBAL.update(D)
    fig = plt.figure(figsize=(7.2, 9.7))
    outer = GridSpec(5, 1, figure=fig, height_ratios=[1.5, 1.45, 2.0, 1.6, 1.9], hspace=0.5,
                     left=0.07, right=0.95, top=0.975, bottom=0.04)
    g1 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], width_ratios=[1.6, 1.0, 1.0], wspace=0.45)
    g4 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[3], width_ratios=[1.85, 0.85, 0.28], wspace=0.25)
    g5 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[4], width_ratios=[1.35, 1.0, 1.05], wspace=0.3)
    g5b = GridSpecFromSubplotSpec(2, 1, subplot_spec=g5[1], height_ratios=[1, 0.25], hspace=0.05)
    where = dict(a=g1[0], b=g1[1], c=g1[2], d=outer[1], e=outer[2], f=g4[0], g=g4[1], h=g5[0], i=g5b[0], j=g5[2])
    nudge = dict(a=(-0.03, 0.015), b=(-0.06, 0.015), c=(-0.06, 0.015), d=(-0.03, 0.02), e=(-0.06, 0.03),
                 f=(-0.05, 0.03), g=(-0.05, 0.03), h=(-0.02, 0.02), i=(-0.02, 0.02), j=(-0.06, 0.03))
    res = {}
    for L, _, build in PANELS:
        if L == "i":
            ax = panel_istar_zoom(fig, where["i"], res["h"]["zoom_px"]); out = None
        else:
            ax, out = build(fig, where[L], D)
        res[L] = out; F2.letter(ax, L, *nudge[L])
    F2.draw_letters(fig)
    manifest = dict(
        dataset=DS, domains={m: {int(k): v for k, v in D["names_dom"][m].items()} for m in D["names_dom"]},
        features=res["e"], function_matched=res["f"], chromatin_potential_examples=res["g"],
        istar=dict(zoom=res["h"], **res["j"]),
        not_shown_but_recorded=dict(
            chromatin_potential_domain_summary=pd.read_csv(CPD / "dorc_cp_domain_summary.csv").to_dict("records"),
            gene_domain_cases_outside_random_band=_outside_band(),
            function_matching_cis_fraction_top10={"RNA": 0.001, "ATAC": 0.0}),
        sources=[str(R / DS), str(CPD), str(IST), str(ISTH), f"{H}/Yufan_Melanoma/HE_HMelanoma.tif"])
    save_dual(fig, OUT, extra=manifest); save_dual(fig, FINAL / "fig3_melanoma", extra=manifest)
    plt.close(fig)
    export_panels(D, FINAL / "panels", res["h"]["zoom_px"])
    print(json.dumps(dict(domains=manifest["domains"], istar=manifest["istar"]), indent=1, default=str))


if __name__ == "__main__":
    main()
