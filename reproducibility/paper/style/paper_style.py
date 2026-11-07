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
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
import saturn_theme  # noqa: F401  registers the saturn_* colormaps
from saturn_theme import ANCHOR


# --------------------------------------------------------------------------- #
# Palettes — the source of truth for cross-figure color alignment.
# Saturn theme (2026-10-15): low-saturation colours anchored on two reference covers
# (saturn_theme.py). Pre-theme values: paper_style_v1_saturated.py. Muted colours that collapsed for
# colour-blind readers inside one figure were split by LIGHTNESS (white matter light, thalamic dark, ...).
# --------------------------------------------------------------------------- #

# General-purpose semantic palette (from ChenLiu figures4papers).
PALETTE = {
    "blue_main":      "#2B3F6B",
    "blue_secondary": "#4E6A97",
    "green_1":        "#E3ECE2",
    "green_2":        "#BCCDB9",
    "green_3":        "#7E9A7A",
    "red_1":          "#E2C7C9",
    "red_2":          "#B98A8E",
    "red_strong":     "#9A5A55",
    "neutral":        "#CDD3D3",
    "highlight":      "#D9C49A",
    "teal":           "#5F9AA3",
    "violet":         "#7E6A8C",
    "gold":           "#C49A5A",
    "magenta":        "#A4728A",
    "black":          "#1E2430",
    "grey_50":        "#7C8486",
}

# Modality → color (locked across every figure in this paper).
# Matches Saturn architecture diagram color anchors.
MODALITY = {
    "RNA":      "#2B3F6B",   # deep blue
    "ATAC":     "#C49A5A",   # amber orange
    "Protein":  "#7E6A8C",   # violet
    "HE":       "#7E9A7A",   # deep green
    "H&E":      "#7E9A7A",
    "HiC":      "#9A5A55",   # red
    "Hi-C":     "#9A5A55",
    "Splicing": "#6A7FA6",
    "lncRNA":   "#9089AC",
}

# Benchmark methods → color. Semantic convention (figures4papers house style):
#   blue    = proposed method (Saturn)
#   teal    = closest relative / ablation of proposed (GASTON-single)
#   reds    = competing published baselines
#   neutral = classical matrix-factorization references
METHOD = {
    "Saturn":          "#14244A",   # proposed — anchor blue
    "GASTON-Mix-MM":   "#14244A",
    "Saturn (ours)":   "#14244A",
    "GASTON-single":   "#5F9AA3",   # closest relative — teal
    "GASTON":          "#5F9AA3",
    "GASTON (global-scaled)": "#9CC5C9",   # GASTON on Saturn's globally scaled targets -- light teal
    "SpatialPCA":      "#9A5A55",   # published spatial baselines — reds
    "STAMP":           "#E2C7C9",
    "NSF":             "#B27A6E",
    "MEFISTO":         "#7E6A8C",   # violet — published multi-modal baseline
    "SCIGMA":          "#6E3E3B",   # published multi-modal spatial integration — dark red
    "SpatialGlue":     "#D4AFAF",   # published multi-modal spatial integration — salmon
    "MISO":            "#9C8E76",   # published multi-modal spatial integration — warm grey-brown
    "PCA":             "#8C9496",   # classical references — greys
    "NMF":             "#BCC3C4",
    "SpatialNMF":      "#9AA2A4",
    "MULTISPATI-PCA":  "#6F7779",
    "RAW":             "#DDE2E2",   # un-denoised control, not a method
    # downstream-analysis baselines (Fig 2-4 v2): metacells, trajectory, density
    "SEACells-LSI":    "#9C7A55",   # metacells on ATAC LSI -- brown
    "SEACells-PCA":    "#D9C49A",   # metacells on RNA PCA -- light brown
    "DPT":             "#8C9496",   # diffusion pseudotime kernel -- grey
    "CellRank fusion": "#5E5A86",   # w*k_rhoRNA + (1-w)*k_rhoATAC -- indigo
    "Mellon":          "#A4728A",   # density baseline -- mauve
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
    "cortex":       "#7E9CBA",
    "hippocampus":  "#91B17F",
    "striatum":     "#AF917E",
    "thalamus":     "#86698A",
    "CC":           "#9D6262",   # corpus callosum / lesion tissue
    "lesion":       "#9D6262",
    "peri_lesion":  "#CBB5B5",
    "control":      "#5D7B9C",
    "far":          "#CCD1D1",
    # LPC10 FOV (Fig 2) -- keyed by ANNOTATED label, never by domain index
    "lesion core":            "#9D6262",   # = lesion
    "white matter":           "#D6DFE7",
    "periventricular":        "#97B48D",
    "thalamic":               "#86698A",   # = thalamus; RNA and ATAC neuronal ends share it
    "inflammatory chromatin": "#B49691",
    "parenchymal":            "#E6DFD2",
    "transitional":           "#C1B7A6",
    # melanoma (Fig 3) -- keyed by ANNOTATED label
    "epidermis":              "#DCCB9F",
    "melanocytic tumour":     "#534771",
    "IFN-stromal":            "#568D84",
    "AP-1 stress":            "#A9705A",
    "immune-accessible":      "#844A48",
    "tumour, low-signal":     "#B6ADC9",
    "mixed tumour-stroma":    "#9DBEB4",
    "metabolic tumour":       "#7D64A4",
    # MERFISH isoform brain section (Fig 4) -- same tissue => same colour as LPC terms
    "thalamus / hypothalamus":    "#86698A",   # = thalamic
    "white matter & hippocampus": "#D6DFE7",   # = white matter
    "ventricle / choroid plexus": "#97B48D",   # = periventricular
    "cortex":                     "#7E9CBA",
    "subcortical":                "#C5B0C7",
    "upper-layer cortex":         "#34506E",
    "amygdala / olfactory cortex": "#B8AA97",  # SV-isoform fit (Fig 4 v2): subplate, amygdala, olfactory areas
    # melanoma spatial Hi-C (Fig 6) -- shared terms keep their Fig 3 colour
    "plasma / immune":            "#844A48",   # = immune-accessible
    "deep tumour":                "#4C436A",
    # elbow-K domains (2026-10-14): merged domains keep the colour of their majority part
    "thalamic & periventricular": "#86698A",   # Fig 2 RNA K = 3: 633 thalamic + 355 periventricular spots
    "subcortical & white matter": "#C4B6CF",   # Fig 4 Isoform K = 3: subcortical + white matter & hippocampus
    "melanocytic tumour & immune": "#725483",  # Fig 6 RNA K = 3: melanocytic tumour + plasma / immune
}

# Tonsil germinal-centre zones (Sprox-seq; Fig 5) -- called from the authors' marker lists
GCZONE = {
    "naive follicle":   "#D9C49A",   # IGHD/FCER2/TCL1A
    "dark zone":        "#2B3F6B",   # CXCR4/AICDA/MME/FOXP1
    "light zone":       "#B98A8E",   # CD83/LMO2/BCL2A1
    "extrafollicular":  "#DCE1E1",
}

# Broad cell classes (MERFISH cell_label -> class; Fig 4)
CELLCLASS = {
    "neuron":             "#86698A",   # = LINEAGE neuron
    "oligodendrocyte":    "#D6DFE7",   # = LINEAGE OL
    "OPC":                "#B4C2CF",
    "astrocyte":          "#67A56F",   # = LINEAGE astrocyte
    "microglia":          "#AC7F79",   # = LINEAGE microglia
    "vascular":           "#959F9F",
    "choroid/ependymal":  "#97B48D",   # = ventricle / choroid plexus
    "other":              "#DBDFDF",
}

# Gene programmes (melanoma process genes; Fig 3)
PROGRAM = {
    "melanocytic":  "#534771",   # = melanocytic tumour
    "interferon":   "#568D84",   # = IFN-stromal
    "epidermis":    "#DCCB9F",   # = epidermis
    "stroma":       "#7F6F52",
    "immune":       "#844A48",   # = immune-accessible
}

# Tissue regions called independently of Saturn (CODEX / marker-defined).
REGION = {
    "lesion":           "#9D6262",
    "distal WM focus":  "#B38EAF",   # second CD11c+ focus in distal white matter
    "other":            "#D3D7D7",
}

# Cell lineages (TF-activity lineage strips, cell-type programs).
LINEAGE = {
    "microglia":       "#AC7F79",
    "neuron":          "#86698A",   # = thalamic / neuronal
    "astrocyte":       "#67A56F",
    "OL":              "#D6DFE7",   # = white matter
    "stress":          "#869292",
}

# Lesion state (LPC).
LESION = {
    "lesion":     "#9D6262",
    "peri":       "#CBB5B5",
    "cortex":     "#7E9CBA",
    "distal":     "#CCD1D1",
}

# Genotype states.
GENOTYPE = {
    "WT":   "#5D7B9C",
    "unknown": "#CCD1D1",
    "not_genotyped": "#CCD1D1",
}

# Colormaps — locked per data-type semantics (matches
# feedback_unified_viz_palette.md convention).
CMAP = {
    "score":    "saturn_div",    # rho, lesion score, expression score (diverging, navy <-> rust)
    "distance": "saturn_seq_r",  # dist_to_lesion, dist_to_center -- near = light
    "density":  "saturn_div",    # Mellon log-density on rho
    "pseudotime": "saturn_seq",
    "expression": "saturn_seq_r",  # per-spot counts / CP10k: low = mist, high = night (sparse signal stays visible)
    "cluster":  "saturn_cat",
    "cluster_large": "tab20",    # > 10 clusters: no muted 20-colour set yet
    "binary":   "saturn_div",
    "correlation": "saturn_div",
    "recovery":  "saturn_mono",  # 0-1 quality metric, higher = deeper navy
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

MIN_PT = 5.0          # manuscript rule: no text below 5 pt at print size
PAGE_WIDTH_IN = 7.2   # double-column width; a wider figure gets shrunk on the page, and so does its text


def check_min_font(fig, min_pt: float = MIN_PT, page_width_in: float = PAGE_WIDTH_IN):
    """Refuse to save a figure with any visible text below `min_pt` at PRINT size.

    Print size = font size x min(1, page_width / figure width): a 10-in figure placed on a 7.2-in page is
    scaled to 72 %, so its 6-pt labels print at 4.3 pt. Raises ValueError listing every offending string."""
    from matplotlib.text import Text
    scale = min(1.0, page_width_in / fig.get_size_inches()[0])
    bad = []
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip() and t.get_fontsize() * scale < min_pt - 1e-6:
            bad.append(f"{t.get_text()[:40]!r}: {t.get_fontsize():.1f} pt -> {t.get_fontsize() * scale:.2f} pt printed")
    if bad:
        raise ValueError(f"{len(bad)} text element(s) below the {min_pt} pt minimum "
                         f"(figure width {fig.get_size_inches()[0]:.2f} in, print scale {scale:.2f}):\n  "
                         + "\n  ".join(bad[:30]))


def save_dual(fig, out_stem: str | Path, *, dpi: int = 600,
                write_manifest: bool = True, extra: dict | None = None,
                pad_inches: float = 0.05) -> dict:
    """Save `fig` as both PNG (≥300 dpi, default 600) and SVG. Optionally drop a
    small JSON manifest recording font sizes / palette in effect.
    """
    check_min_font(fig)
    rasterize_points(fig)
    p = Path(out_stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    png = p.with_suffix(".png")
    svg = p.with_suffix(".svg")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    fig.savefig(svg, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)   # dpi = resolution of rasterized parts

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


RASTER_MIN_POINTS = 300   # scatter layers with more points than this are embedded as a 600-dpi image in the SVG


def rasterize_points(fig, min_points: int = RASTER_MIN_POINTS):
    """Embed dense point layers (spatial scatters, big point clouds), images and meshes as high-resolution
    raster INSIDE the SVG; axes, text, lines, bars and small point sets stay vector. Keeps panel SVGs small
    and stable to open while every label remains editable."""
    from matplotlib.collections import PathCollection, QuadMesh, PolyCollection
    from matplotlib.image import AxesImage
    for ax in fig.axes:
        for c in ax.collections:
            if isinstance(c, PathCollection) and len(c.get_offsets()) > min_points:
                c.set_rasterized(True)
            elif isinstance(c, (QuadMesh,)) or (isinstance(c, PolyCollection) and len(c.get_paths()) > min_points):
                c.set_rasterized(True)
        for im in ax.images:
            im.set_rasterized(True)


def save_panel_svg(fig, out_path: str | Path, *, dpi: int = 600, pad_inches: float = 0.03) -> Path:
    """One sub-panel as a standalone SVG (the manuscript's panel deliverable): min-font check, dense points
    rasterized at `dpi` inside the SVG, everything else vector, text kept as editable text."""
    check_min_font(fig)
    rasterize_points(fig)
    p = Path(out_path).with_suffix(".svg")
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    return p


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
