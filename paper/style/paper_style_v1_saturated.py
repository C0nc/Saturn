"""Project-local paper-figure style.

Rules encoded here (from user's standing requirements — see
memory/feedback_paper_figure_standards.md):
  * Arial only, minimum 5 pt on every text element; prefer as large as legible.
  * Every figure exported at 600 dpi PNG **AND** SVG (from the same fig object).
  * Palette locked per term — never randomize across scripts. Modality colors +
    dataset colors defined here so every panel across every figure uses the
    same hex for the same word.
  * `savefig` sets `svg.fonttype='none'` and `pdf.fonttype=42` so text stays
    editable in Illustrator/Inkscape.

Adapted from ChenLiu-1996/figures4papers scientific-figure-making skill
(published_figures/figures4papers/scientific-figure-making/), with Arial-first
substitutions and dual-export enforcement.

Usage:
    from paper_style import (
        apply, save_dual, PALETTE, MODALITY, DOMAIN, LESION, GENOTYPE, CMAP,
    )
    apply(base_pt=7)                                # or 8 for bigger panels
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    ...
    save_dual(fig, Path('figX/panel_a'))            # writes .png @600 + .svg
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json
import matplotlib
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Palettes — the source of truth for cross-figure color alignment.
# --------------------------------------------------------------------------- #

# General-purpose semantic palette (from ChenLiu figures4papers).
PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1":        "#DDF3DE",
    "green_2":        "#AADCA9",
    "green_3":        "#8BCF8B",
    "red_1":          "#F6CFCB",
    "red_2":          "#E9A6A1",
    "red_strong":     "#B64342",
    "neutral":        "#CFCECE",
    "highlight":      "#FFD700",
    "teal":           "#42949E",
    "violet":         "#9A4D8E",
    "gold":           "#DAA520",
    "magenta":        "#C1428E",
    "black":          "#1a1a1a",
    "grey_50":        "#7f7f7f",
}

# Modality → color (locked across every figure in this paper).
# Matches Saturn architecture diagram color anchors.
MODALITY = {
    "RNA":      "#0F4D92",   # deep blue
    "ATAC":     "#E48B23",   # amber orange
    "Protein":  "#9A4D8E",   # violet
    "HE":       "#4A8F3D",   # deep green
    "H&E":      "#4A8F3D",
    "HiC":      "#B64342",   # red
    "Hi-C":     "#B64342",
    "DNAm":     "#42949E",   # teal
    "methyl":   "#42949E",
    "TCR":      "#DAA520",   # gold
    "Mutation": "#C1428E",   # magenta
    "MUT":      "#C1428E",
    "Splicing": "#3775BA",
    "lncRNA":   "#7C5FA0",
}

# Benchmark methods → color. Semantic convention (figures4papers house style):
#   blue    = proposed method (Saturn)
#   teal    = closest relative / ablation of proposed (GASTON-single)
#   reds    = competing published baselines
#   neutral = classical matrix-factorization references
METHOD = {
    "Saturn":          "#0F4D92",   # proposed — anchor blue
    "GASTON-Mix-MM":   "#0F4D92",
    "Saturn (ours)":   "#0F4D92",
    "GASTON-single":   "#42949E",   # closest relative — teal
    "GASTON":          "#42949E",
    "GASTON (global-scaled)": "#8CC5CB",   # GASTON on Saturn's globally scaled targets -- light teal
    "SpatialPCA":      "#B64342",   # published spatial baselines — reds
    "STAMP":           "#E9A6A1",
    "NSF":             "#C96A57",
    "MEFISTO":         "#9A4D8E",   # violet — published multi-modal baseline
    "SCIGMA":          "#8C2D2B",   # published multi-modal spatial integration — dark red
    "SpatialGlue":     "#D4726A",   # published multi-modal spatial integration — salmon
    "MISO":            "#8A7560",   # published multi-modal spatial integration — warm grey-brown
    "PCA":             "#909090",   # classical references — greys
    "NMF":             "#BFBFBF",
    "SpatialNMF":      "#8F8F8F",
    "MULTISPATI-PCA":  "#767676",
    "RAW":             "#D8D8D8",   # un-denoised control, not a method
    # downstream-analysis baselines (Fig 2-4 v2): metacells, trajectory, density
    "SEACells-LSI":    "#A86F3C",   # metacells on ATAC LSI -- brown
    "SEACells-PCA":    "#D9A66B",   # metacells on RNA PCA -- light brown
    "DPT":             "#8C8C8C",   # diffusion pseudotime kernel -- grey
    "CellRank fusion": "#5B4E9E",   # w*k_rhoRNA + (1-w)*k_rhoATAC -- indigo
    "Mellon":          "#C27BA0",   # density baseline -- mauve
}

# Canonical left-to-right ordering for benchmark bars — keep identical in every panel.
METHOD_ORDER = [
    "Saturn", "GASTON-single", "SpatialPCA", "STAMP", "NSF",
    "MEFISTO", "PCA", "NMF",
]

# Hard-grid multi-modal benchmark (Fig 1): proposed, closest relative, then
# published multi-modal spatial methods.
METHOD_ORDER_MM = ["Saturn", "GASTON", "MEFISTO", "MISO", "SCIGMA", "SpatialGlue"]

# LPC domain / anatomical region colors — align with paper_figures/figure2 usage.
DOMAIN = {
    "cortex":       "#5A8DBA",
    "hippocampus":  "#8FBF6E",
    "striatum":     "#D08A5C",
    "thalamus":     "#B48EAD",
    "CC":           "#B64342",   # corpus callosum / lesion tissue
    "lesion":       "#B64342",
    "peri_lesion":  "#E9A6A1",
    "control":      "#3775BA",
    "far":          "#CFCECE",
    # LPC10 FOV (Fig 2) -- keyed by ANNOTATED label, never by domain index
    "lesion core":            "#B64342",   # = lesion
    "white matter":           "#6FA8DC",
    "periventricular":        "#93C47D",
    "thalamic":               "#B48EAD",   # = thalamus; RNA and ATAC neuronal ends share it
    "inflammatory chromatin": "#E07B67",
    "parenchymal":            "#D5D5D5",
    "transitional":           "#F2C57C",
    # melanoma (Fig 3) -- keyed by ANNOTATED label
    "epidermis":              "#E6B85C",
    "melanocytic tumour":     "#4B3B6B",
    "IFN-stromal":            "#3C9D8B",
    "AP-1 stress":            "#E0784C",
    "immune-accessible":      "#C4453C",
    "tumour, low-signal":     "#B9A7D6",
    "mixed tumour-stroma":    "#9FC3B6",
    "metabolic tumour":       "#7A5AA6",
    # MERFISH isoform brain section (Fig 4) -- same tissue => same colour as LPC terms
    "thalamus / hypothalamus":    "#B48EAD",   # = thalamic
    "white matter & hippocampus": "#6FA8DC",   # = white matter
    "ventricle / choroid plexus": "#93C47D",   # = periventricular
    "cortex":                     "#5A8DBA",
    "subcortical":                "#CDB3CB",
    "upper-layer cortex":         "#2E5E8C",
    "amygdala / olfactory cortex": "#E8B36A",  # SV-isoform fit (Fig 4 v2): subplate, amygdala, olfactory areas
    # melanoma spatial Hi-C (Fig 6) -- shared terms keep their Fig 3 colour
    "plasma / immune":            "#C4453C",   # = immune-accessible
    "deep tumour":                "#2E2447",
    # elbow-K domains (2026-10-14): merged domains keep the colour of their majority part
    "thalamic & periventricular": "#B48EAD",   # Fig 2 RNA K = 3: 633 thalamic + 355 periventricular spots
    "subcortical & white matter": "#C7B4D3",   # Fig 4 Isoform K = 3: subcortical + white matter & hippocampus
    "melanocytic tumour & immune": "#6E4A7E",  # Fig 6 RNA K = 3: melanocytic tumour + plasma / immune
}

# Tonsil germinal-centre zones (Sprox-seq; Fig 5) -- called from the authors' marker lists
GCZONE = {
    "naive follicle":   "#F2C57C",   # IGHD/FCER2/TCL1A
    "dark zone":        "#2E5E8C",   # CXCR4/AICDA/MME/FOXP1
    "light zone":       "#E07B67",   # CD83/LMO2/BCL2A1
    "extrafollicular":  "#DDDDDD",
}

# Broad cell classes (MERFISH cell_label -> class; Fig 4)
CELLCLASS = {
    "neuron":             "#B48EAD",   # = LINEAGE neuron
    "oligodendrocyte":    "#6FA8DC",   # = LINEAGE OL
    "OPC":                "#A7C7E7",
    "astrocyte":          "#5AAE61",   # = LINEAGE astrocyte
    "microglia":          "#D6604D",   # = LINEAGE microglia
    "vascular":           "#9A9A9A",
    "choroid/ependymal":  "#93C47D",   # = ventricle / choroid plexus
    "other":              "#DDDDDD",
}

# Gene programmes (melanoma process genes; Fig 3)
PROGRAM = {
    "melanocytic":  "#4B3B6B",   # = melanocytic tumour
    "interferon":   "#3C9D8B",   # = IFN-stromal
    "epidermis":    "#E6B85C",   # = epidermis
    "stroma":       "#A6761D",
    "immune":       "#C4453C",   # = immune-accessible
}

# Tissue regions called independently of Saturn (CODEX / marker-defined).
REGION = {
    "lesion":           "#B64342",
    "distal WM focus":  "#C77DBA",   # second CD11c+ focus in distal white matter
    "other":            "#D5D5D5",
}

# Cell lineages (TF-activity lineage strips, cell-type programs).
LINEAGE = {
    "microglia":       "#D6604D",
    "neuron":          "#B48EAD",   # = thalamic / neuronal
    "astrocyte":       "#5AAE61",
    "OL":              "#6FA8DC",   # = white matter
    "stress":          "#8C8C8C",
}

# Lesion state (LPC).
LESION = {
    "lesion":     "#B64342",
    "peri":       "#E9A6A1",
    "cortex":     "#5A8DBA",
    "distal":     "#CFCECE",
}

# Genotype states.
GENOTYPE = {
    "WT":   "#3775BA",
    "MUT":  "#C1428E",
    "unknown": "#CFCECE",
    "not_genotyped": "#CFCECE",
}

# Colormaps — locked per data-type semantics (matches
# feedback_unified_viz_palette.md convention).
CMAP = {
    "score":    "coolwarm",      # ρ, lesion score, expression score
    "distance": "viridis_r",     # dist_to_lesion, dist_to_center — near=bright
    "density":  "coolwarm",      # Mellon log-density on ρ
    "pseudotime": "viridis",
    "cluster":  "tab10",
    "cluster_large": "tab20",
    "binary":   "RdBu_r",
    "correlation": "RdBu_r",
    "recovery":  "Blues",       # 0-1 quality metric, higher = deeper blue
}


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Style:
    """Publication rcParams. Defaults match user's standing rules."""
    base_pt: float = 7.0                      # ≥5 pt everywhere; 7 is a safe default
    axes_linewidth: float = 0.8               # spines
    tick_length: float = 3.0
    grid_alpha: float = 0.25
    font_family: tuple = ("Arial", "DejaVu Sans", "Helvetica")


def apply(base_pt: float = 7.0, style: Style | None = None) -> Style:
    """Apply publication rcParams. Call once per script, before plotting."""
    if style is None:
        style = Style(base_pt=float(base_pt))
    assert style.base_pt >= 5, f"base_pt ({style.base_pt}) below 5 pt minimum"

    matplotlib.use("Agg", force=False)
    rc = plt.rcParams
    rc["font.family"] = "sans-serif"
    rc["font.sans-serif"] = list(style.font_family)
    rc["font.size"] = style.base_pt
    rc["axes.titlesize"] = style.base_pt + 1
    rc["axes.labelsize"] = style.base_pt
    rc["xtick.labelsize"] = style.base_pt - 0.5
    rc["ytick.labelsize"] = style.base_pt - 0.5
    rc["legend.fontsize"] = style.base_pt - 0.5
    rc["figure.titlesize"] = style.base_pt + 2

    # Print / vector fidelity
    rc["svg.fonttype"] = "none"     # keep text editable in SVG
    rc["pdf.fonttype"] = 42
    rc["ps.fonttype"] = 42
    rc["axes.linewidth"] = style.axes_linewidth
    rc["xtick.major.width"] = style.axes_linewidth
    rc["ytick.major.width"] = style.axes_linewidth
    rc["xtick.major.size"] = style.tick_length
    rc["ytick.major.size"] = style.tick_length
    rc["axes.spines.top"] = False
    rc["axes.spines.right"] = False
    rc["axes.grid"] = False
    rc["grid.alpha"] = style.grid_alpha
    rc["legend.frameon"] = False
    rc["figure.dpi"] = 150
    rc["savefig.dpi"] = 600
    rc["savefig.bbox"] = "tight"
    rc["figure.autolayout"] = False
    return style


# --------------------------------------------------------------------------- #
# Dual export — the enforced PNG@600dpi + SVG pair.
# --------------------------------------------------------------------------- #

def save_dual(fig, out_stem: str | Path, *, dpi: int = 600,
                write_manifest: bool = True, extra: dict | None = None,
                pad_inches: float = 0.05) -> dict:
    """Save `fig` as both PNG (≥300 dpi, default 600) and SVG. Optionally drop a
    small JSON manifest recording font sizes / palette in effect.
    """
    p = Path(out_stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    png = p.with_suffix(".png")
    svg = p.with_suffix(".svg")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    fig.savefig(svg, bbox_inches="tight", pad_inches=pad_inches)

    manifest = dict(
        png=str(png), svg=str(svg), dpi=dpi,
        font_size_pt=plt.rcParams["font.size"],
        font_family=plt.rcParams["font.sans-serif"],
        axes_linewidth=plt.rcParams["axes.linewidth"],
        svg_fonttype=plt.rcParams["svg.fonttype"],
        pdf_fonttype=plt.rcParams["pdf.fonttype"],
        modality_palette=MODALITY,
        domain_palette=DOMAIN,
    )
    if extra:
        manifest.update(extra)
    if write_manifest:
        (p.with_suffix(".json")).write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def clean_ax(ax, *, spatial: bool = False):
    """Standard axis cleanup. `spatial=True` for tissue-space panels."""
    if spatial:
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
        ax.invert_yaxis()
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def color(term: str, default: str = "#7f7f7f") -> str:
    """Look up a term across all locked palettes. Case-insensitive."""
    for table in (METHOD, MODALITY, DOMAIN, LESION, GENOTYPE, PALETTE):
        if term in table:
            return table[term]
    tl = term.lower()
    for table in (METHOD, MODALITY, DOMAIN, LESION, GENOTYPE, PALETTE):
        for k, v in table.items():
            if k.lower() == tl:
                return v
    return default
