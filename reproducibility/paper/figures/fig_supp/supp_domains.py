"""Supplementary domain-annotation pages, one per figure and modality: how every domain was called and named.

    python fig_supp/supp_domains.py [fig2 fig3 fig4 fig5 fig6]

Each page (same layout for every likelihood):
  a  the isodepth rho of this modality
  b  the domains used in the main figure (GASTON DP on the Saturn targets, K = elbow), named as in the main figure
  c  how K was chosen: DP error vs K (saturn.domains.dp_error_curve), elbow (kneedle) and the K used
  d  spots and median sequencing depth per domain (a domain must not be a coverage band)
  e  top markers per domain: one-vs-rest enrichment (counts: Poisson z; continuous / usage: Welch t), shown as the
     domain-mean of each marker z-scored across domains; cell-type composition per domain where available
  f  fitted functions of the top 2 markers per domain, GASTON style: observed at every spot | the per-domain
     piecewise function evaluated at every spot | along rho: bin means (domain colour) + fitted function per domain
     (Poisson / Gaussian / binomial behind GASTON's LRT gate, saturn.piecewise); R2 on binned values
Outputs figN/supp/figN_domains_<modality>.{png,json} (600 dpi PNG; spatial layers are raster anyway) and
figN/supp/figN_domain_markers_<modality>.csv (top 8 markers per domain with statistics).
"""
from __future__ import annotations
import json, sys
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

H = "/mnt/HDD2B/Users/yancui"; PF = Path(f"{H}/Saturn/published_figures")
for p in (f"{PF}/_style", f"{H}/saturn_repo/src", f"{PF}/fig2", f"{H}/Hole/gaston_mix_mm/scripts"):
    sys.path.insert(0, p)
from paper_style import apply, check_min_font, rasterize_points, MODALITY, CMAP
from saturn import piecewise as PW, annotate as AN
from saturn.domains import dp_error_curve, kneedle, isodepth_bins, boundaries

N_MARK, N_FIT, NB = 8, 2, 25


@dataclass
class Mod:
    """one modality of one figure, in the figure's own domain labels"""
    name: str                         # RNA, ATAC, Isoform, Protein, HiC
    d: np.ndarray                     # isodepth
    lab: np.ndarray                   # domain index 0..K-1 (as in the main figure)
    names: dict                       # index -> annotated name
    colors: dict                      # index -> hex
    kind: str                         # poisson | gauss | binom
    feats: np.ndarray                 # feature names
    X: object                         # poisson: counts (spots x feats, sparse); gauss: values; binom: successes
    Y: np.ndarray                     # Saturn targets (for the DP error curve)
    depth: np.ndarray                 # per-spot sequencing depth (for d)
    depth_label: str = "median depth"
    lib: np.ndarray = None            # poisson exposure
    N: object = None                  # binom trials (spots x feats)
    label: callable = None            # pretty feature label
    unit: str = ""                    # y label of the function plots
    min_count: int = 100
    min_lfc: float = 0.5              # poisson markers: minimum log2 fold change
    keep: np.ndarray = None           # features eligible as markers (default all)


@dataclass
class Spec:
    fig: str
    title: str
    xy: np.ndarray
    mods: list
    s: float = 1.2                    # spatial point size
    comp: pd.DataFrame = None         # per-spot cell-type composition (optional)
    comp_label: str = "cell-type composition"
    notes: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- helpers
def spatial(ax, xy, v=None, colors=None, cmap=None, vmin=None, vmax=None, s=1.2):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=colors if colors is not None else v, cmap=None if colors is not None else cmap,
                    vmin=vmin, vmax=vmax, s=s, lw=0, rasterized=True)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sd in ax.spines.values():
        sd.set_visible(False)
    return sc


def cbar(fig, ax, sc, label, ticks=None):
    cax = ax.inset_axes([1.03, 0.08, 0.05, 0.7]); cb = fig.colorbar(sc, cax=cax, ticks=ticks)
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.2, length=1.5, width=0.4)
    cb.ax.set_title(label, fontsize=5.2, pad=2, loc="left")


def column(M, j):
    c = M[:, j]
    return np.asarray(c.todense()).ravel().astype(float) if hasattr(c, "todense") else np.asarray(c, float).ravel()


def markers(m: Mod):
    """top N_MARK one-vs-rest markers per domain -> DataFrame(domain, rank, feature, stat, effect)"""
    rows = []; K = m.lab.max() + 1
    keep = np.ones(len(m.feats), bool) if m.keep is None else m.keep
    for k in range(K):
        if m.kind == "poisson":
            Xk = m.X[:, np.where(keep)[0]]
            res = AN.poisson_markers(Xk, m.lib, m.lab, m.feats[keep], k, n=N_MARK, min_count=m.min_count, min_lfc=m.min_lfc)
            for r, (f, lfc, tot) in enumerate(res):
                rows.append(dict(domain=k, name=m.names[k], rank=r + 1, feature=f, effect=lfc, stat_name="log2FC",
                                 total_count=tot))
        else:
            V = usage(m)[:, keep] if m.kind == "binom" else np.asarray(m.X, float)[:, keep]
            res = AN.continuous_markers(V, m.lab, m.feats[keep], k, n=len(m.feats[keep]))
            res = sorted([r for r in res if r[1] > 0], key=lambda r: -r[2])[:N_MARK]   # enriched in the domain, by t
            for r, (f, diff, t) in enumerate(res):
                rows.append(dict(domain=k, name=m.names[k], rank=r + 1, feature=f, effect=diff,
                                 stat_name="mean difference", t=t))
    M = pd.DataFrame(rows)
    if len(M) and m.label is not None:
        M.insert(4, "label", [m.label(f) for f in M.feature])
    return M


def usage(m: Mod):
    S = m.X.toarray() if hasattr(m.X, "toarray") else np.asarray(m.X, float)
    N = m.N.toarray() if hasattr(m.N, "toarray") else np.asarray(m.N, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        U = S / N
    return np.where(N > 0, U, np.nan_to_num(np.nanmean(np.where(N > 0, U, np.nan), 0))[None, :] * np.ones_like(U))


def values(m: Mod, j):
    """observed per-spot value of feature j on the plotting scale"""
    if m.kind == "poisson":
        return np.log1p(column(m.X, j) / m.lib * 1e4)
    if m.kind == "binom":
        return usage(m)[:, j]
    return np.asarray(m.X, float)[:, j]


def fit(m: Mod, j):
    """per-domain fit of feature j; returns slope, inter (domains,), link"""
    if m.kind == "poisson":
        sl, it = PW.fit_poisson(column(m.X, j)[:, None], m.lib, m.lab, m.d)
        return sl[0], it[0], "log"
    if m.kind == "binom":
        sl, it = PW.fit_binomial(column(m.X, j)[:, None], column(m.N, j)[:, None], m.lab, m.d)
        return sl[0], it[0], "logit"
    sl, it = PW.fit_gauss(np.asarray(m.X, float)[:, [j]], m.lab, m.d)
    return sl[0], it[0], "identity"


def scale(eta, link):
    """fitted function on the plotting scale"""
    if link == "log":
        return np.log1p(np.exp(eta) * 1e4)
    if link == "logit":
        return 1 / (1 + np.exp(-eta))
    return eta


def binned(m: Mod, j, y):
    bi, _, kp = isodepth_bins(m.d, NB); kp = [b for b in kp if (bi == b).sum() >= 3]
    db = np.array([m.d[bi == b].mean() for b in kp])
    lb = np.array([np.bincount(m.lab[bi == b], minlength=m.lab.max() + 1).argmax() for b in kp])
    if m.kind == "poisson":
        c = column(m.X, j); ob = np.log1p(np.array([c[bi == b].sum() / m.lib[bi == b].sum() for b in kp]) * 1e4)
    elif m.kind == "binom":
        s, n = column(m.X, j), column(m.N, j)
        ob = np.array([s[bi == b].sum() / max(n[bi == b].sum(), 1e-9) for b in kp])
    else:
        ob = np.array([y[bi == b].mean() for b in kp])
    return db, ob, lb


# --------------------------------------------------------------------------- page
def page(S: Spec, m: Mod, out_dir: Path):
    K = int(m.lab.max()) + 1; col = [m.colors[k] for k in range(K)]
    M = markers(m); M.to_csv(out_dir / f"{S.fig}_domain_markers_{m.name}.csv", index=False)
    top, used = {}, set()                                # fitted panels: top markers, each feature shown once
    for k in range(K):
        top[k] = [f for f in M[M.domain == k].feature if f not in used][:N_FIT]; used |= set(top[k])
    fits = [(k, f) for k in range(K) for f in top[k]]
    nrow_f = int(np.ceil(len(fits) / 2))
    fig = plt.figure(figsize=(7.2, 4.6 + 1.32 * nrow_f))
    o = GridSpec(3, 1, figure=fig, height_ratios=[1.9, 2.3, 1.32 * nrow_f], hspace=0.55,
                 left=0.07, right=0.97, top=0.925, bottom=0.03)
    fig.suptitle(f"{S.title} · {m.name} domains: how they were called and named", fontsize=7.2, fontweight="bold", y=0.995)
    if m.name in S.notes:
        fig.text(0.07, 0.968, "naming rule: " + S.notes[m.name], fontsize=5.4, va="top", color="#444444", wrap=True)
    L = []                                               # (ax, letter)
    # ---- row 1: rho | domains | K | depth
    g1 = GridSpecFromSubplotSpec(1, 4, subplot_spec=o[0], wspace=0.75, width_ratios=[1, 1, 0.9, 0.95])
    a = fig.add_subplot(g1[0]); lo, hi = np.nanpercentile(m.d, [1, 99])
    sc = spatial(a, S.xy, m.d, cmap=CMAP["score"], vmin=lo, vmax=hi, s=S.s); cbar(fig, a, sc, "isodepth", ticks=[])
    a.set_title(f"ρ$_{{\\mathrm{{{m.name}}}}}$", color=MODALITY.get(m.name, "#333333"), fontsize=6.5, pad=2); L.append((a, "a"))
    a = fig.add_subplot(g1[1]); spatial(a, S.xy, colors=np.array(col)[m.lab], s=S.s)
    order = np.argsort([np.median(m.d[m.lab == k]) for k in range(K)])
    a.legend(handles=[Line2D([], [], ls="", marker="s", ms=3, color=col[k], label=f"{m.names[k]} ({(m.lab == k).sum()})")
                      for k in order], fontsize=5.1, loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False,
             handletextpad=0.1, labelspacing=0.15, borderaxespad=0)
    a.set_title(f"{K} domains (low → high ρ)", fontsize=6.3, pad=2); L.append((a, "b"))
    a = fig.add_subplot(g1[2]); u = dp_error_curve(m.d, m.Y); ke = kneedle(u); ks = np.arange(1, len(u) + 1)
    a.plot(ks, u, "-o", ms=2.2, lw=0.9, color=MODALITY.get(m.name, "#333333"))
    a.scatter([K], [u[K - 1]], s=26, facecolor="none", edgecolor="#1a1a1a", lw=0.8, zorder=3)
    a.set_xlabel("K (domains)", labelpad=1); a.set_ylabel("unexplained target variance", labelpad=1)
    a.set_title(f"elbow K = {ke}; used K = {K}", fontsize=6.0, pad=2); a.set_xticks(ks[::2]); L.append((a, "c"))
    a = fig.add_subplot(g1[3]); x = np.arange(K)
    med = [np.median(m.depth[m.lab == k]) for k in order]
    a.bar(x, med, 0.65, color=[col[k] for k in order], edgecolor="#1a1a1a", lw=0.3)
    for xi, k in zip(x, order):
        a.text(xi, med[list(order).index(k)], f"{(m.lab == k).sum()}", ha="center", va="bottom", fontsize=5)
    a.set_xticks(x); a.set_xticklabels([str(k) for k in order], fontsize=5.3); a.set_xlabel("domain (low → high ρ)", labelpad=1)
    a.set_ylabel(m.depth_label, labelpad=1); a.set_title("depth (n spots on bars)", fontsize=6.0, pad=2)
    L.append((a, "d"))
    # ---- row 2: marker heatmap | composition
    hasc = S.comp is not None
    g2 = GridSpecFromSubplotSpec(1, 2 if hasc else 1, subplot_spec=o[1], wspace=0.9, width_ratios=[1.25, 1] if hasc else [1])
    a = fig.add_subplot(g2[0]); feats = [f for k in order for f in list(M[M.domain == k].feature[:3])]
    jj = [int(np.where(m.feats == f)[0][0]) for f in feats]
    if m.kind == "gauss":                                  # continuous: domain mean minus section mean
        Z = np.array([[np.nanmean(values(m, j)[m.lab == k]) - np.nanmean(values(m, j)) for k in order] for j in jj])
        zlab, vl = "Δ mean", max(np.nanpercentile(np.abs(Z), 98), 0.2)
    else:                                                  # counts / usage: log2 domain rate over section rate
        def rate(j, msk):
            if m.kind == "poisson":
                return column(m.X, j)[msk].sum() / m.lib[msk].sum()
            return column(m.X, j)[msk].sum() / max(column(m.N, j)[msk].sum(), 1e-9)
        Z = np.array([[np.log2((rate(j, m.lab == k) + 1e-12) / (rate(j, np.ones(len(m.d), bool)) + 1e-12)) for k in order] for j in jj])
        zlab, vl = "log2 vs\nsection", 2.0
    im = a.imshow(Z.T, aspect="auto", cmap=CMAP["score"], vmin=-vl, vmax=vl)
    a.set_yticks(range(K)); a.set_yticklabels([m.names[k] for k in order], fontsize=5.2)
    for tl, k in zip(a.get_yticklabels(), order): tl.set_color(col[k] if col[k].lower() not in ("#d6dfe7", "#e6dfd2", "#dce1e1") else "#555555")
    lab_ = [m.label(f) if m.label else f for f in feats]
    a.set_xticks(range(len(feats))); a.set_xticklabels(lab_, rotation=60, ha="right", fontsize=5.0)
    for i in range(1, K):
        a.axvline(3 * i - 0.5, color="white", lw=1.2)
    cb = fig.colorbar(im, ax=a, fraction=0.03, pad=0.02); cb.ax.tick_params(labelsize=5); cb.outline.set_linewidth(0.3)
    cb.ax.set_title(zlab, fontsize=5.2, pad=2, loc="left")
    a.set_title("top 3 markers per domain (enrichment in each domain)", fontsize=6.0, pad=2); L.append((a, "e"))
    if hasc:
        a = fig.add_subplot(g2[1]); C = S.comp.fillna(0)
        mean = pd.DataFrame({k: C[m.lab == k].mean() for k in order}).T
        tops = mean.max().nlargest(8).index; mean = mean[tops]
        left = np.zeros(K)
        cm = plt.get_cmap(CMAP["cluster"])
        for i, t in enumerate(tops):
            a.barh(np.arange(K), mean[t].values, 0.7, left=left, color=cm(i % 10), edgecolor="white", lw=0.3, label=t[:28])
            left += mean[t].values
        a.set_yticks(range(K)); a.set_yticklabels([m.names[k] for k in order], fontsize=5.2); a.invert_yaxis()
        a.set_xlabel("mean proportion", labelpad=1); a.set_title(S.comp_label + " per domain (top 8)", fontsize=6.0, pad=2)
        a.legend(fontsize=5.1, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.02),
                 handlelength=0.8, labelspacing=0.15)
        L.append((a, "f"))
    # ---- rows: fitted functions
    gf = GridSpecFromSubplotSpec(nrow_f, 2, subplot_spec=o[2], wspace=0.28, hspace=0.75)
    stats = []
    for i, (k, f) in enumerate(fits):
        j = int(np.where(m.feats == f)[0][0]); y = values(m, j); sl, it, link = fit(m, j)
        db, ob, lb = binned(m, j, y); pr = scale(it[lb] + sl[lb] * db, link)
        r2 = 1 - ((ob - pr) ** 2).sum() / max(((ob - ob.mean()) ** 2).sum(), 1e-12)
        gi = GridSpecFromSubplotSpec(1, 3, subplot_spec=gf[i // 2, i % 2], wspace=0.45, width_ratios=[1, 1, 1.35])
        a1, a2, a3 = (fig.add_subplot(gi[q]) for q in range(3))
        lo, hi = np.nanpercentile(y, [2, 99])
        cmap = CMAP["expression"] if m.kind != "gauss" else CMAP["score"]
        sc = spatial(a1, S.xy, y, cmap=cmap, vmin=lo, vmax=hi, s=S.s * 0.8)
        spot = scale(it[m.lab] + sl[m.lab] * m.d, link)
        spatial(a2, S.xy, spot, cmap=cmap, vmin=lo, vmax=hi, s=S.s * 0.8)
        nm = m.label(f) if m.label else f
        a1.set_title(nm, fontsize=5.8, pad=7, loc="left", fontstyle="italic" if m.name == "RNA" else "normal")
        a1.text(0, 1.02, f"marker of {m.names[k]}", transform=a1.transAxes, fontsize=5.1, va="bottom",
                color=col[k] if col[k].lower() not in ("#d6dfe7", "#e6dfd2", "#dce1e1", "#dccb9f") else "#555555")
        a1.text(0.5, -0.03, "observed", transform=a1.transAxes, ha="center", va="top", fontsize=5)
        a2.text(0.5, -0.03, "fitted", transform=a2.transAxes, ha="center", va="top", fontsize=5)
        a3.scatter(db, ob, s=6, c=[col[q] for q in lb], lw=0.2, edgecolor="#1a1a1a", zorder=3)
        for q in range(K):
            dk = m.d[m.lab == q]
            if len(dk) < 3:
                continue
            xs = np.linspace(dk.min(), dk.max(), 30); a3.plot(xs, scale(it[q] + sl[q] * xs, link), color=col[q], lw=1.3)
        for b in boundaries(m.d, m.lab):
            a3.axvline(b, color="#9aa2aa", lw=0.4, ls=(0, (2, 2)))
        a3.set_xlabel(f"ρ$_{{\\mathrm{{{m.name}}}}}$", labelpad=0.5, fontsize=5.5); a3.set_ylabel(m.unit, labelpad=0.5, fontsize=5.3)
        a3.tick_params(labelsize=5, pad=1); a3.set_title(f"R² {r2:.2f}; sloped in {int((np.abs(sl) > 1e-9).sum())}/{K}", fontsize=5.3, pad=1.5)
        if i == 0:
            L.append((a1, "g"))
        stats.append(dict(domain=m.names[k], feature=f, R2_binned=round(float(r2), 3), n_sloped=int((np.abs(sl) > 1e-9).sum())))
    fig.canvas.draw()
    for ax, s in L:
        b = ax.get_position(); fig.text(b.x0 - 0.04, b.y1 + 0.008, s, fontsize=9, fontweight="bold", va="bottom")
    check_min_font(fig); rasterize_points(fig)
    stem = out_dir / f"{S.fig}_domains_{m.name}"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.05); plt.close(fig)
    meta = dict(figure=S.fig, modality=m.name, K=K, elbow_K=int(ke), dp_error=[round(float(v), 4) for v in u],
                domains={m.names[k]: dict(n=int((m.lab == k).sum()), median_depth=round(float(np.median(m.depth[m.lab == k])), 2),
                                          rho_median=round(float(np.median(m.d[m.lab == k])), 3),
                                          top_markers=list(M[M.domain == k].feature[:N_MARK])) for k in range(K)},
                fitted=stats, kind=m.kind, notes=S.notes)
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
    print("->", stem.with_suffix(".png"), flush=True)
    return meta


def build(S: Spec):
    out = PF / S.fig / "supp"; out.mkdir(parents=True, exist_ok=True)
    return [page(S, m, out) for m in S.mods]


if __name__ == "__main__":
    import supp_domain_specs as SPEC
    apply(base_pt=6.5)
    for f in sys.argv[1:] or ["fig2", "fig3", "fig4", "fig5", "fig6"]:
        build(getattr(SPEC, f)())
