"""Figure 2 (v2) — Fig 2 a-j + the SEACells metacell benchmark of cis gene-peak coupling.
Saturn low-saturation theme (2026-10-15). Panels e and g = per feature: original per-spot spatial distribution |
fitted function evaluated at every spot | bin means + fitted function along the own axis. f = observed vs fitted TF
activity along rho_ATAC. Two pages: part 1 = a-g, part 2 = h-l (letters run across both).
Output: published_figures/fig2/fig2_part1.{png,svg,json}, fig2_part2.{png,svg,json} + panels/fig2<letter>_<name>.svg; repro copy of the
assembled figure in final_fig2_v2/.

  k  near-minus-distal concordance gap (|TSS| < 25 kb vs > 1 Mb same chromosome), on the SAME
     (gene, peak, domain) triples (count read from cis_function_pairs.csv): Saturn fitted functions (both sloped) vs SEACells
     metacells (Persad 2023) on ATAC LSI and RNA PCA, all pairs and the p < 0.05 gate
  l  the same gated gap per Saturn domain
Source: Hole/gaston_mix_mm/results/real_saturn/lpc10_fov_seacells/ (lpc10_seacells_cis_benchmark.py).
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
from paper_style import apply, save_dual, save_panel_svg, MODALITY, DOMAIN, METHOD, PALETTE, ANCHOR
import fig2_lpc as F2

H = "/mnt/HDD2B/Users/yancui"
SC = F2.R / "lpc10_fov_seacells"
OUT = Path(f"{H}/Saturn/published_figures/fig2/fig2")          # figure home (published_figures/figN/figN)
FINAL = Path(f"{H}/Saturn/final_fig2_v2")
METH = [("Saturn fitted functions", "Saturn", "Saturn"), ("SEACells-LSI", "SEACells-LSI", "SEACells (ATAC LSI)"),
        ("SEACells-PCA", "SEACells-PCA", "SEACells (RNA PCA)")]
DOMS = {"RNA": ["lesion core", "white matter", "thalamic & periventricular"],      # elbow K: RNA 3, ATAC 2
        "ATAC": ["inflammatory chromatin", "parenchymal"]}
SHORT = {"lesion core": "lesion core", "white matter": "white matter", "periventricular": "periventric.",
         "thalamic": "thalamic", "thalamic & periventricular": "thal. & periventric.", "inflammatory chromatin": "inflamm. chrom.", "parenchymal": "parenchymal",
         "transitional": "transitional"}


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def panel_seacells_overall(ax, D):
    S = pd.read_csv(SC / "summary.csv")
    bw, out = 0.26, {}
    for j, m in enumerate(["RNA", "ATAC"]):
        rows = [("Saturn fitted functions", "both sloped", 1.0, "Saturn"),
                ("SEACells-LSI", "p<0.05", 1.0, "SEACells-LSI"), ("SEACells-PCA", "p<0.05", 1.0, "SEACells-PCA")]
        for i, (meth, filt, al, ck) in enumerate(rows):
            r = S[(S.method == meth) & (S.axis == m) & (S["filter"] == filt)].iloc[0]
            x = j + (i - 1) * bw
            ax.bar(x, r.gap, bw * 0.92, color=METHOD[ck], alpha=al, lw=0.4, edgecolor=PALETTE["black"])
            ax.text(x, max(r.gap, 0) + 0.004, stars(r.p), ha="center", va="bottom", fontsize=5, rotation=90)
            out[f"{m}|{meth}|{filt}"] = dict(gap=float(r.gap), p=float(r.p))
    ax.axhline(0, color=PALETTE["black"], lw=0.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["ρ$_{\\mathrm{RNA}}$ domains", "ρ$_{\\mathrm{ATAC}}$ domains"])
    for t, m in zip(ax.get_xticklabels(), ["RNA", "ATAC"]):
        t.set_color(MODALITY[m])
    ax.set_ylabel("near − distal concordance")
    ax.set_ylim(-0.02, 0.26); ax.set_xlim(-0.55, 1.55)        # legend sits above the tallest bar (0.18)
    h = [Patch(fc=METHOD["Saturn"], ec=PALETTE["black"], lw=0.4, label="Saturn (gene & peak sloped)"),
         Patch(fc=METHOD["SEACells-LSI"], ec=PALETTE["black"], lw=0.4, label="SEACells ATAC-LSI (r p < 0.05)"),
         Patch(fc=METHOD["SEACells-PCA"], ec=PALETTE["black"], lw=0.4, label="SEACells RNA-PCA (r p < 0.05)")]
    ax.legend(handles=h, fontsize=5, loc="upper right", ncol=1, handlelength=1.2, columnspacing=0.6,
              borderaxespad=0.1, labelspacing=0.2)
    n = len(pd.read_csv(F2.R / "lpc10_fov_cis_pairs" / "cis_function_pairs.csv", usecols=["gene"]))
    ax.set_title(f"cis gene–peak coupling, same {n:,} triples", fontsize=5.8, pad=3); out["n_triples"] = n
    return out


def panel_seacells_domain(ax, D):
    S = pd.read_csv(SC / "per_domain.csv")
    sat = S[S.method == "Saturn fitted functions"]
    xs, labs, cols, out = [], [], [], {}
    x, cen, ahead = 0, {}, []
    for m in ["RNA", "ATAC"]:
        x0 = x
        for d in DOMS[m]:
            vals = {"Saturn": sat[(sat.axis == m) & (sat.domain == d)]}
            for k in ["SEACells-LSI", "SEACells-PCA"]:
                vals[k] = S[(S.method == f"{k} (p<0.05)") & (S.axis == m) & (S.domain == d)]
            for k, dx, mk in (("Saturn", -0.2, "o"), ("SEACells-LSI", 0, "s"), ("SEACells-PCA", 0.2, "D")):
                r = vals[k]
                if len(r) == 0 or not np.isfinite(r.gap.iloc[0]):
                    ax.text(x + dx, 0.012, "n/a", fontsize=5, ha="center", rotation=90, color=ANCHOR["stone"]); continue
                g = float(r.gap.iloc[0])
                ax.plot([x + dx, x + dx], [0, g], color=METHOD[k], lw=0.9, zorder=2)
                ax.scatter(x + dx, g, s=11 if k == "Saturn" else 8, marker=mk, color=METHOD[k], lw=0.3,
                           edgecolor=PALETTE["black"], zorder=3)
                out[f"{m}|{d}|{k}"] = round(g, 3)
            if all(f"{m}|{d}|{k}" in out for k in ("Saturn", "SEACells-LSI", "SEACells-PCA")):
                ahead.append(out[f"{m}|{d}|Saturn"] > max(out[f"{m}|{d}|SEACells-LSI"], out[f"{m}|{d}|SEACells-PCA"]))
            xs.append(x); labs.append(SHORT[d]); cols.append(DOMAIN[d]); x += 1
        cen[m] = (x0 + x - 1) / 2; x += 0.6
    ax.axhline(0, color=PALETTE["black"], lw=0.5)
    ax.axvline(cen["RNA"] + (len(DOMS["RNA"]) - 1) / 2 + 0.8, color=ANCHOR["stone"], lw=0.5, ls=":")
    ax.set_xticks(xs); ax.set_xticklabels(labs, rotation=35, ha="right", fontsize=5.3)
    for t, c in zip(ax.get_xticklabels(), cols):
        t.set_color(PALETTE["black"]); t.set_bbox(dict(fc=c, ec="none", alpha=0.35, pad=0.6))
    ax.text(cen["RNA"], 0.245, "ρ$_{\\mathrm{RNA}}$ domains", color=MODALITY["RNA"], ha="center", fontsize=5.8)
    ax.text(cen["ATAC"], 0.245, "ρ$_{\\mathrm{ATAC}}$ domains", color=MODALITY["ATAC"], ha="center", fontsize=5.8)
    ax.set_ylim(-0.27, 0.38); ax.set_ylabel("near − distal concordance")
    h = [Line2D([], [], marker=mk, ls="", color=METHOD[k], mec=PALETTE["black"], mew=0.3, ms=3.5, label=l)
         for k, mk, l in (("Saturn", "o", "Saturn fitted functions"), ("SEACells-LSI", "s", "SEACells ATAC-LSI (p < 0.05)"),
                          ("SEACells-PCA", "D", "SEACells RNA-PCA (p < 0.05)"))]
    ax.legend(handles=h, fontsize=5, loc="upper center", ncol=3, handletextpad=0.2, borderaxespad=0.1,
              labelspacing=0.2, frameon=False)
    ax.set_title(f"per domain, significance-gated: Saturn ahead in {sum(ahead)} of {len(ahead)} domains",
                 fontsize=5.8, pad=3)
    return out


PANELS = F2.PANELS + [
    ("k", (2.6, 1.7), F2._single(panel_seacells_overall)),
    ("l", (4.2, 1.7), F2._single(panel_seacells_domain)),
]


PANEL_NAMES = dict(a="codex_overview", b="fit_quality_R2", c="lesion_vs_focus_positions", d="isodepths_and_domains",
                   e="fitted_functions", f="tf_activity_observed_vs_fitted", g="tf_activity_space_fitted",
                   h="cis_slope_concordance", i="gene_peak_gradient_agreement", j="inflammatory_genes_R2",
                   k="seacells_cis_gap", l="seacells_cis_gap_per_domain")
PANEL_DIR = OUT.parent / "panels"


def export_panels(D, outdir=PANEL_DIR):
    """every sub-panel as its own named SVG (fig2<letter>_<name>.svg); dense spatial points embedded as 600-dpi
    raster inside the SVG, everything else vector. Same builders as the assembled figure."""
    outdir.mkdir(parents=True, exist_ok=True)
    for L, size, build in PANELS:
        fig = plt.figure(figsize=size)
        spec = GridSpec(1, 1, figure=fig, left=0.1, right=0.93, top=0.9, bottom=0.14)[0]
        build(fig, spec, D); save_panel_svg(fig, outdir / f"fig2{L}_{PANEL_NAMES[L]}"); plt.close(fig)
    print(f"-> {len(PANELS)} panel SVGs in {outdir}")


def main():
    apply(base_pt=6.5)
    D = F2.load()
    res = {}
    # part 1: a-g
    fig = plt.figure(figsize=(7.2, 9.3))
    outer = GridSpec(4, 1, figure=fig, height_ratios=[1.45, 1.45, 3.0, 2.45], hspace=0.42,
                     left=0.07, right=0.95, top=0.975, bottom=0.03)
    g1 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], width_ratios=[1.55, 0.9, 1.15], wspace=0.42)
    g4 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[3], width_ratios=[1.25, 1], wspace=0.22)
    where = dict(a=g1[0], b=g1[1], c=g1[2], d=outer[1], e=outer[2], f=g4[0], g=g4[1])
    nudge = dict(a=(-0.03, 0.012), b=(-0.06, 0.012), c=(-0.06, 0.012), d=(-0.03, 0.02), e=(-0.06, 0.03),
                 f=(-0.075, 0.02), g=(-0.03, 0.02))
    F2.LETTERS.clear()
    for L, _, build in PANELS:
        if L in where:
            ax, out = build(fig, where[L], D); res[L] = out; F2.letter(ax, L, *nudge[L])
    F2.draw_letters(fig)
    save_dual(fig, OUT.parent / "fig2_part1", extra=dict(panels="a-g", features=res["e"], tf_heatmap_dropped=res["f"]))
    plt.close(fig)
    # part 2: h-l
    fig = plt.figure(figsize=(7.2, 3.6))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.4, 1.45], hspace=0.6, left=0.07, right=0.95, top=0.94,
                     bottom=0.14)
    g5 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.5, width_ratios=[1.1, 1.1, 0.85])
    g6 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.25, width_ratios=[1.0, 1.55])
    where = dict(h=g5[0], i=g5[1], j=g5[2], k=g6[0], l=g6[1])
    nudge = dict(h=(-0.085, 0.02), i=(-0.085, 0.02), j=(-0.06, 0.03), k=(-0.07, 0.02), l=(-0.06, 0.02))
    F2.LETTERS.clear()
    for L, _, build in PANELS:
        if L in where:
            ax, out = build(fig, where[L], D); res[L] = out; F2.letter(ax, L, *nudge[L])
    F2.draw_letters(fig)
    old = json.load(open(F2.FINAL / "fig2_lpc.json")) if (F2.FINAL / "fig2_lpc.json").exists() else {}
    manifest = dict(old, panels="h-l", seacells_overall=res["k"], seacells_per_domain=res["l"],
                    seacells_purity={"LSI": dict(RNA=0.525, ATAC=0.649), "PCA": dict(RNA=0.764, ATAC=0.715)},
                    seacells_source=str(SC))
    save_dual(fig, OUT.parent / "fig2_part2", extra=manifest); plt.close(fig)
    export_panels(D)


if __name__ == "__main__":
    main()
