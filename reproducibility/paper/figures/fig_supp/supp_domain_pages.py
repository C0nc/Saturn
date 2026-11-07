"""Supplementary domain-annotation figures -- how every domain was annotated, both modalities of one dataset per A4 page.

  S4 LPC lesion (Fig 2), S7 melanoma DBiT (Fig 3), S9 brain MERFISH (Fig 4), S10 tonsil Sprox-seq (Fig 5), S11 melanoma spatial Hi-C (Fig 6)
Per modality: choice of K (DP error curve) | top 3 markers per domain | fitted functions of the top 2 markers per domain.
S9 also shows cell-type composition per domain (all types), because only the MERFISH domains were named from composition.
Domains, names, colours, markers and fits come from supp_domains.py / supp_domain_specs.py, i.e. the main figures' own.
Output: fig_supp/S<n>/figS<n>.{png,svg,pdf,json} on an A4 sheet, and one PNG + SVG per panel in fig_supp/S<n>/panels/
S11 also shows further chromatin-to-gene example bins in the style of main Fig 6h.
Usage: python fig_supp/supp_domain_pages.py [S4 S7 S9 S10 S11]
"""
from __future__ import annotations
import json, string, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PF = Path("<workspace>/Saturn/published_figures")
sys.path.insert(0, str(PF / "fig_supp"))
import supp_domains as SD
import supp_domain_specs as SPEC
from paper_style import apply, save_panel_svg, MODALITY, CMAP
from a4 import save_a4
from saturn.domains import dp_error_curve, kneedle, boundaries
import saturn_theme  # noqa: F401  (registers saturn_* colormaps)

TITLE = dict(fig2="LPC lesion (LPC10 FOV)", fig3="Melanoma (DBiT RNA + ATAC)", fig4="Brain MERFISH (RNA + isoform usage)",
             fig5="Tonsil Sprox-seq B1 (RNA + protein)", fig6="Melanoma spatial Hi-C (RNA + A/B compartments)")
SHORT = dict(fig2="LPC", fig3="melanoma", fig4="MERFISH", fig5="tonsil", fig6="melanomaHiC")
PAGES = dict(S4=["fig2"], S7=["fig3"], S9=["fig4"], S10=["fig5"], S11=["fig6"])
PALE = ("#d6dfe7", "#e6dfd2", "#dce1e1", "#dccb9f")
W, H1, FEAT, GROW = 7.2, 1.05, 0.68, 0.93
TILE = 0.46


def ink(c):
    return "#555555" if c.lower() in PALE else c


class Cv:
    """axes and text placed in inches from the top-left of the figure"""
    def __init__(s, w, h):
        s.w, s.h = w, h; s.fig = plt.figure(figsize=(w, h))
    def ax(s, x, y, w, h):
        return s.fig.add_axes([x / s.w, 1 - (y + h) / s.h, w / s.w, h / s.h])
    def text(s, x, y, t, **k):
        k.setdefault("va", "top"); return s.fig.text(x / s.w, 1 - y / s.h, t, **k)
    def letter(s, x, y, t):
        s.text(x, y, t, fontsize=9, fontweight="bold")


# --------------------------------------------------------------------------- data
def prep(S, m):
    """everything the panels draw, computed once"""
    K = int(m.lab.max()) + 1; col = [m.colors[k] for k in range(K)]
    M = SD.markers(m); order = list(np.argsort([np.median(m.d[m.lab == k]) for k in range(K)]))
    used, fits = set(), []
    for k in range(K):
        top = [f for f in M[M.domain == k].feature if f not in used][:SD.N_FIT]; used |= set(top)
        fits += [(k, f) for f in top]
    u = dp_error_curve(m.d, m.Y)
    feats = [f for k in order for f in list(M[M.domain == k].feature[:3])]
    jj = [int(np.where(m.feats == f)[0][0]) for f in feats]
    if m.kind == "gauss":
        Z = np.array([[np.nanmean(SD.values(m, j)[m.lab == k]) - np.nanmean(SD.values(m, j)) for k in order] for j in jj])
        zlab, vl = "Δ mean", max(np.nanpercentile(np.abs(Z), 98), 0.2)
    else:
        def rate(j, msk):
            if m.kind == "poisson":
                return SD.column(m.X, j)[msk].sum() / m.lib[msk].sum()
            return SD.column(m.X, j)[msk].sum() / max(SD.column(m.N, j)[msk].sum(), 1e-9)
        Z = np.array([[np.log2((rate(j, m.lab == k) + 1e-12) / (rate(j, np.ones(len(m.d), bool)) + 1e-12)) for k in order] for j in jj])
        zlab, vl = "log2 vs\nsection", 2.0
    G = []
    for k, f in fits:
        j = int(np.where(m.feats == f)[0][0]); y = SD.values(m, j); sl, it, link = SD.fit(m, j)
        db, ob, lb = SD.binned(m, j, y); pr = SD.scale(it[lb] + sl[lb] * db, link)
        r2 = 1 - ((ob - pr) ** 2).sum() / max(((ob - ob.mean()) ** 2).sum(), 1e-12)
        G.append(dict(k=k, f=f, y=y, spot=SD.scale(it[m.lab] + sl[m.lab] * m.d, link), sl=sl, it=it, link=link, db=db, ob=ob, lb=lb, r2=r2))
    return dict(S=S, m=m, K=K, col=col, order=order, u=u, ke=kneedle(u), feats=feats, Z=Z, zlab=zlab, vl=vl, G=G)


# --------------------------------------------------------------------------- panels
def elbow(cv, x, y, w, h, P):
    m, K, u = P["m"], P["K"], P["u"]; a = cv.ax(x, y, w, h); ks = np.arange(1, len(u) + 1)
    a.plot(ks, u, "-o", ms=2, lw=0.8, color=MODALITY.get(m.name, "#333333"))
    a.scatter([K], [u[K - 1]], s=22, facecolor="none", edgecolor="#1a1a1a", lw=0.7, zorder=3)
    a.set_xlabel("K (domains)", labelpad=1, fontsize=5.5); a.set_ylabel("unexplained\ntarget variance", labelpad=1, fontsize=5.5)
    a.set_title(f"elbow K = {P['ke']}; used K = {K}", fontsize=5.6, pad=2); a.set_xticks(ks[::2]); a.tick_params(labelsize=5, pad=1)
    for s in ("top", "right"): a.spines[s].set_visible(False)


def heat(cv, x, y, w, h, P):
    m, K, order, col = P["m"], P["K"], P["order"], P["col"]; a = cv.ax(x, y, w, h)
    im = a.imshow(P["Z"].T, aspect="auto", cmap=CMAP["score"], vmin=-P["vl"], vmax=P["vl"])
    a.set_yticks(range(K)); a.set_yticklabels([m.names[k] for k in order], fontsize=5.2)
    for tl, k in zip(a.get_yticklabels(), order): tl.set_color(ink(col[k]))
    a.set_xticks(range(len(P["feats"]))); a.set_xticklabels([m.label(f) if m.label else f for f in P["feats"]], rotation=60, ha="right", rotation_mode="anchor", fontsize=5)
    a.tick_params(length=1.5, pad=1.5)
    for i in range(1, K): a.axvline(3 * i - 0.5, color="white", lw=1)
    for s in a.spines.values(): s.set_visible(False)
    a.set_title("top 3 markers per domain", fontsize=5.8, pad=2)
    cax = cv.ax(x + w + 0.05, y, 0.06, h); cb = cv.fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=5, length=1.5, width=0.4, pad=1)
    cb.outline.set_linewidth(0.3); cb.ax.set_title(P["zlab"], fontsize=5, pad=2, loc="left")


COMP_H, COMP_ROW = 0.85, 1.2
EXTRA = ["#D98C5F", "#4F7F5B", "#A85C8A", "#3F7C9E", "#8A8A3C", "#D66F6F", "#5B4B8A", "#7FC4A0", "#BBAAD4", "#8C6D46"]
PAL = PF / "_palettes" / "celltype_stack.json"


def type_order(S):
    """every type, largest mean proportion first; no 'other' class"""
    C = S.comp.fillna(0); C.columns = [str(c).replace("_", " ") for c in C.columns]
    return C, list(C.mean().sort_values(ascending=False).index)


def colours(names):
    """sticky per type name; 36 distinct: theme colours, then light and dark partners"""
    cm = plt.get_cmap(CMAP["cluster"]); base = [matplotlib.colors.to_hex(cm(i)) for i in range(cm.N)] + EXTRA
    rgb = [np.array(matplotlib.colors.to_rgb(c)) for c in base]
    base = base + [matplotlib.colors.to_hex(0.55 * c + 0.45) for c in rgb] + [matplotlib.colors.to_hex(0.6 * c) for c in rgb]
    pal = json.loads(PAL.read_text()) if PAL.exists() else {}
    used = set(pal.values())
    for n in names:
        if n not in pal:
            c = next(b for b in base if b not in used); pal[n] = c; used.add(c)
    PAL.write_text(json.dumps(pal, indent=1)); return pal


def composition_row(cv, x, y, P, C, names, pal):
    m, K, order, col = P["m"], P["K"], P["order"], P["col"]; a = cv.ax(x, y, 2.05, COMP_H); left = np.zeros(K)
    F = pd.DataFrame({k: C[m.lab == k].mean() for k in order}).T[names]; F = F.div(F.sum(1), axis=0)
    for n in names:
        a.barh(np.arange(K), F[n].values, 0.78, left=left, color=pal[n], edgecolor="white", lw=0.2); left += F[n].values
    a.set_ylim(K - 0.5, -0.5); a.set_xlim(0, 1); a.set_xticks([0, 0.5, 1]); a.set_xticklabels(["0", "0.5", "1"], fontsize=5)
    a.set_xlabel("proportion of cells", labelpad=1, fontsize=5.5); a.tick_params(length=1.5, pad=1); a.set_yticks(range(K))
    a.set_yticklabels([m.names[k] for k in order], fontsize=5.2)
    for tl, k in zip(a.get_yticklabels(), order): tl.set_color(ink(col[k]))
    for s in ("top", "right"): a.spines[s].set_visible(False)
    a.set_title(f"cell-type composition per domain (all {len(names)} types)", fontsize=5.8, pad=2)
    lg = cv.ax(x + 2.13, y - 0.1, 0.01, 0.01); lg.axis("off")
    lg.legend([plt.Rectangle((0, 0), 1, 1, color=pal[n]) for n in names], [n[:27] for n in names], fontsize=5, frameon=False, loc="upper left", ncol=3,
              bbox_to_anchor=(0, 1), borderaxespad=0, borderpad=0, handlelength=0.7, handleheight=0.7, handletextpad=0.25, labelspacing=0.02, columnspacing=0.5)


# S11: more chromatin-to-gene example bins, same style as main Fig 6h. Rule: lowest forward chromatin-to-gene P per layer and axis, excluding the Fig 6h bins
LINE_EX = [("RNA", "insulation", "IN", "chr4:158500000-159000000"), ("RNA", "insulation", "IN", "chr7:140500000-141000000"), ("HiC", "insulation", "IN", "chr1:168000000-168500000"),
           ("RNA", "A/B", "AB", "chr22:28500000-29000000"), ("RNA", "A/B", "AB", "chr7:140500000-141000000"), ("HiC", "A/B", "AB", "chr3:52500000-53000000")]


def line_examples(cv, y, letters):
    from scipy.spatial import cKDTree
    R_ = Path("<workspace>/Hole/gaston_mix_mm/results/real_saturn"); DS = "mel_hic_abins_dr50_51_rnadr"; G = R_ / f"{DS}_insulation_granger"
    C = np.load(G / "cache_series.npz", allow_pickle=True); bc = list(C["bc"].astype(str)); ax0 = pd.read_csv(R_ / DS / "axes.csv", index_col=0).loc[bc]
    xy = np.c_[ax0.x.values, ax0.y.values]; S = np.asarray(C["S"], float); Ss = S[cKDTree(xy).query(xy, k=25)[1]].mean(1)
    names = list(C["names"].astype(str)); gl = C["genes"].astype(str); T = pd.read_csv(G / "granger_per_bin.csv"); T = T[T.rho == "forward"]; out = []
    smooth = lambda v: np.r_[v[:2].mean(), np.convolve(v, np.ones(3) / 3, mode="valid"), v[-2:].mean()]
    for i, ((m, layer, key, b), L) in enumerate(zip(LINE_EX, letters)):
        xp, yp = 0.42 + (0 if i < 3 else 3.6), y + 0.5 + (i % 3) * 1.05
        a = cv.ax(xp, yp, 2.62, 0.56); j = names.index(b); d = ax0[f"d_{m}"].values; E = np.array_split(np.argsort(d), 30); rb = np.array([d[e].mean() for e in E])
        c = np.array([np.asarray(C[key], float)[e, j].mean() for e in E]); g = np.array([Ss[e, j].mean() for e in E]); r = T[(T.axis == m) & (T.layer == layer) & (T.bin == b)].iloc[0]
        col, ink, gc = ("#C49A5A", "#8a6a2f", MODALITY["RNA"]) if key == "IN" else ("#9A5A55", "#9A5A55", MODALITY["RNA"])
        a.scatter(rb, c, s=2.5, color=col, alpha=0.45, lw=0, zorder=2); a.plot(rb, smooth(c), color=col, lw=0.9, zorder=3)
        a2 = a.twinx(); a2.scatter(rb, g, s=2.5, color=gc, alpha=0.45, lw=0, zorder=2); a2.plot(rb, smooth(g), color=gc, lw=0.9, zorder=3)
        a.set_ylabel("insulation" if key == "IN" else "A/B", color=ink, labelpad=0.5, fontsize=5); a2.set_ylabel("gene score", color=gc, labelpad=1, fontsize=5)
        a.tick_params(axis="y", labelcolor=ink, labelsize=5, pad=0.8, length=1.5, width=0.4); a2.tick_params(axis="y", labelcolor=gc, labelsize=5, pad=0.8, length=1.5, width=0.4); a.tick_params(axis="x", labelsize=5, pad=0.8, length=1.5, width=0.4)
        a.locator_params(axis="y", nbins=3); a2.locator_params(axis="y", nbins=3); a.locator_params(axis="x", nbins=6)
        a.spines["top"].set_visible(False); a2.spines["top"].set_visible(False); a2.spines["right"].set_visible(True)
        a.set_xlabel(f"ρ$_{{\\mathrm{{{m}}}}}$ (30 windows)", color=MODALITY[m], labelpad=0.5, fontsize=5)
        c_, se = b.split(":"); s0, e0 = (int(x) / 1e6 for x in se.split("-")); e_ = int(np.floor(np.log10(r.p_c2g)))
        ptxt = f"P = {r.p_c2g / 10 ** e_:.2f} × 10$^{{{e_}}}$".replace("-", "−")
        a.text(-0.16, 1.42, L, transform=a.transAxes, fontsize=9, fontweight="bold", va="center", ha="left")
        a.text(0, 1.3, f"{c_} {s0:g}–{e0:g} Mb · {', '.join(gl[j].split(',')[:4])}", transform=a.transAxes, fontsize=5, va="bottom")
        a.text(0, 1.03, f"{layer} → gene, {ptxt}, q = {r.q_c2g:.3f}, lag {int(r.lag_c2g)}", transform=a.transAxes, fontsize=5, va="bottom")
        out.append(dict(letter=L, axis=m, layer=layer, bin=b, genes=str(gl[j]), p=float(r.p_c2g), q=float(r.q_c2g), lag=int(r.lag_c2g), p_gene_to_chromatin=float(r.p_g2c)))
    return out


def functions(cv, x0, y0, P, letters=None):
    S, m, K, col = P["S"], P["m"], P["K"], P["col"]; stats = []; XP = 1.76
    for i, g in enumerate(P["G"]):
        x = x0 + (i % 4) * XP; y = y0 + (i // 4) * GROW + 0.24
        a1, a2 = cv.ax(x, y, TILE, TILE), cv.ax(x + TILE + 0.02, y, TILE, TILE); a3 = cv.ax(x + 1.22, y, 0.5, TILE)
        lo, hi = np.nanpercentile(g["y"], [2, 99]); cmap = CMAP["expression"] if m.kind != "gauss" else CMAP["score"]
        SD.spatial(a1, S.xy, g["y"], cmap=cmap, vmin=lo, vmax=hi, s=S.s * 0.6)
        SD.spatial(a2, S.xy, g["spot"], cmap=cmap, vmin=lo, vmax=hi, s=S.s * 0.6)
        cv.text(x + 0.14, y - 0.25, m.label(g["f"]) if m.label else g["f"], fontsize=5.6, fontstyle="italic" if m.name == "RNA" else "normal")
        if letters:
            cv.letter(x, y - 0.3, letters[i])
        cv.text(x, y - 0.13, f"marker of {m.names[g['k']]}", fontsize=5, color=ink(col[g["k"]]))
        a1.text(0.5, -0.03, "observed", transform=a1.transAxes, ha="center", va="top", fontsize=5)
        a2.text(0.5, -0.03, "fitted", transform=a2.transAxes, ha="center", va="top", fontsize=5)
        a3.scatter(g["db"], g["ob"], s=3.5, c=[col[q] for q in g["lb"]], lw=0.15, edgecolor="#1a1a1a", zorder=3)
        for q in range(K):
            dk = m.d[m.lab == q]
            if len(dk) >= 3:
                xs = np.linspace(dk.min(), dk.max(), 30); a3.plot(xs, SD.scale(g["it"][q] + g["sl"][q] * xs, g["link"]), color=col[q], lw=0.9)
        for b in boundaries(m.d, m.lab):
            a3.axvline(b, color="#9aa2aa", lw=0.3, ls=(0, (2, 2)))
        a3.set_xlabel(f"ρ$_{{\\mathrm{{{m.name}}}}}$", labelpad=0, fontsize=5); a3.set_ylabel(m.unit.replace("A/B compartment (z)", "A/B (z)"), labelpad=0.5, fontsize=5)
        a3.tick_params(labelsize=5, pad=0.8, length=1.5, width=0.4); a3.locator_params(axis="both", nbins=3)
        for s in ("top", "right"): a3.spines[s].set_visible(False)
        ns = int((np.abs(g["sl"]) > 1e-9).sum())
        a3.set_title(f"R² {g['r2']:.2f}\nsloped in {ns}/{K}", fontsize=5, pad=1.5, linespacing=0.95)
        stats.append(dict(domain=m.names[g["k"]], feature=g["f"], R2_binned=round(float(g["r2"]), 3), n_sloped=ns))
    return stats


def g_rows(P):
    return int(np.ceil(len(P["G"]) / 4))


def block_h(P):
    return 0.3 + H1 + FEAT + g_rows(P) * GROW


# --------------------------------------------------------------------------- output
def panel_out(cv, stem, name):
    p = stem / "panels" / name
    save_panel_svg(cv.fig, p); cv.fig.savefig(p.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03); plt.close(cv.fig)


def make_page(tag):
    apply(base_pt=6.5); stem = PF / "fig_supp" / tag; (stem / "panels").mkdir(parents=True, exist_ok=True)
    data = []
    for f in PAGES[tag]:
        S = getattr(SPEC, f)()
        data.append(dict(f=f, S=S, P=[prep(S, m) for m in S.mods])); print("prepared", f, flush=True)
    letters = iter(string.ascii_lowercase)
    for n_page, d in enumerate(data, 1):
        hascomp = d["f"] == "fig4"; hasg = d["f"] == "fig6"
        FH = 0.22 + sum(block_h(P) + 0.1 for P in d["P"]) + 0.05 + (2 * COMP_ROW if hascomp else 0) + (3.6 if hasg else 0)
        if hascomp:
            C, tnames = type_order(d["S"]); pal = colours(tnames)
        cv = Cv(W, FH); y = 0.02; meta = {}
        cv.text(0.02, y, TITLE[d["f"]], fontsize=7, fontweight="bold"); y += 0.22
        for P in d["P"]:
            m = P["m"]; yt = y + 0.3
            cv.text(0.02, y, f"{m.name} domains", fontsize=6.2, color=MODALITY.get(m.name, "#333333"))
            lc, le = next(letters), next(letters); lcomp = next(letters) if hascomp else None; lg = [next(letters) for _ in P["G"]]
            cv.letter(0.02, y + 0.12, lc); elbow(cv, 0.42, yt, 0.8, H1, P)
            cv.letter(1.33, y + 0.12, le); heat(cv, 2.2, yt, 4.45, H1, P)
            yg = yt + H1 + FEAT
            if hascomp:
                cv.letter(0.02, yg - 0.17, lcomp); composition_row(cv, 1.35, yg - 0.05, P, C, tnames, pal); yg += COMP_ROW
            st = functions(cv, 0.14, yg, P, lg)
            meta[f"{d['f']}_{m.name}"] = dict(letters=[lc, le, *([lcomp] if hascomp else []), *lg], K=P["K"], elbow_K=int(P["ke"]), fitted=st)
            y += block_h(P) + 0.1 + (COMP_ROW if hascomp else 0)
            # one file per panel
            base = f"fig{tag}{{}}_{SHORT[d['f']]}_{m.name}_{{}}"
            q = Cv(1.4, 1.6); elbow(q, 0.42, 0.2, 0.9, 1.1, P); panel_out(q, stem, base.format(lc, "choice_of_K"))
            q = Cv(6.2, 2.0); heat(q, 1.1, 0.2, 4.45, 1.1, P); panel_out(q, stem, base.format(le, "top_markers"))
            q = Cv(7.2, g_rows(P) * GROW + 0.06); functions(q, 0.14, 0.06, P, lg); panel_out(q, stem, base.format(f"{lg[0]}-{lg[-1]}", "fitted_functions"))
            if hascomp:
                q = Cv(7.2, 1.75); composition_row(q, 1.35, 0.2, P, C, tnames, pal); panel_out(q, stem, base.format(lcomp, "composition_stacked"))
        if hasg:
            ll = [next(letters) for _ in LINE_EX]
            cv.text(0.02, y, "Further chromatin-to-gene example bins", fontsize=6.2, color="#333333")
            meta["line_examples"] = line_examples(cv, y, ll)
            q = Cv(7.2, 3.8); line_examples(q, 0.02, ll); panel_out(q, stem, f"fig{tag}{ll[0]}-{ll[-1]}_granger_example_lines")
        save_a4(cv.fig, stem / f"fig{tag}", extra=dict(panels=meta)); plt.close(cv.fig)


if __name__ == "__main__":
    for t in sys.argv[1:] or list(PAGES):
        make_page(t)
