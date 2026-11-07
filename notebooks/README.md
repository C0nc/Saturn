# Notebooks

There are two kinds of notebook here. Both only cover Saturn; baseline methods are not included.

| kind | files | use it to |
|---|---|---|
| Tutorials | `tutorials/tutorial_*.ipynb` | learn the API: every step is an inline `saturn.*` call with text and plots |
| Figure reproduction | `fig*.ipynb` | rerun a manuscript figure from its inputs and compare the result with the published files |

All notebooks are stored with their outputs, so they can be read without running them.

## Tutorials

| tutorial | data | shows | CPU time |
|---|---|---|---|
| [`tutorial_lpc_rna_atac`](tutorials/tutorial_lpc_rna_atac.ipynb) | DBiT ARC-seq, LPC lesion, 3,425 spots | spot selection, RNA and ATAC targets, fit, domains, annotation, feature functions, own vs cross axis, TF activity, gene and cis peak comparison, cross-term decomposition | about 6 min |
| [`tutorial_merfish_isoform`](tutorials/tutorial_merfish_isoform.ipynb) | RT&T-MERFISH, 2,721 bins of 100 µm | isoform usage target, fit, domains, binomial fits along the axis, isoform switches between domains, one cell type across a boundary, cross-term decomposition | 9 to 21 min |
| [`tutorial_sproxseq_rna_protein`](tutorials/tutorial_sproxseq_rna_protein.ipynb) | Sprox-seq tonsil, section B1, 3,204 spots | RNA and protein targets, fit, domains, held-out protein complexes tested against a rotation null, cell-cell contact complexes, cross pathways | about 6 min |

Run one with:

```bash
SATURN_DATA=/path/to/data PYTHON=python bash tutorials/run_tutorial.sh tutorial_lpc_rna_atac /path/to/workdir
```

Input paths are relative to `SATURN_DATA` and are set in the first code cell of each tutorial. Each tutorial loads
the published RNA target and compares its fit with the published axes in a visible cell. `tutorials/src/` holds the
source of each notebook.

## Figure reproduction

| notebook | figure | inputs | how it compares with the published result |
|---|---|---|---|
| `fig1_simulation` | Fig. 1 | [`inputs/fig1.json`](inputs/fig1.json) | all 24 fits identical |
| `fig2_lpc` | Fig. 2 | [`inputs/fig2.json`](inputs/fig2.json) | fit identical; part 1 pixel-identical |
| `fig3_melanoma` | Fig. 3 | [`inputs/fig3.json`](inputs/fig3.json) | axes, tables and part 1–2 panels identical; iStar maps are read from the saved run, not rerun |
| `fig4_merfish_isoform` | Fig. 4 | [`inputs/fig4.json`](inputs/fig4.json) | axes and tables identical |
| `fig5_sproxseq` | Fig. 5 | [`inputs/fig5.json`](inputs/fig5.json) | fits identical; panels pixel-identical |
| `fig6_melanoma_hic` | Fig. 6 | [`inputs/fig6.json`](inputs/fig6.json) | fits identical; composite differs in 0.42% of pixels |
| `fig7_cross_term` | Fig. 7, Supplementary Figs. 6-8 | [`inputs/fig7.json`](inputs/fig7.json) | all tables and arrays identical; panel SVGs identical (see below) |

To run one:

```bash
python notebooks/setup_workspace.py --root ~/saturn_workspace --figs fig4 --link-from /path/to/data_root
SATURN_ROOT=~/saturn_workspace SATURN_PUBLISHED=/path/to/published bash notebooks/test_nb.sh fig4_merfish_isoform
```

`setup_workspace.py` copies the manuscript code from `paper/` into the workspace and replaces the lab path prefix.
It writes nothing outside the workspace. `inputs/figN.json` lists each input file and its source. A step is skipped
when its output already exists, so a stopped run can be resumed. `src/` holds the notebook sources (built with
`build.py`), and `steps/` holds the helper scripts and pinned files.

## What is pinned or cannot be rerun

**The saved RNA target is pinned.** `saturn.preprocess.rna_glmpc_real` starts GLM-PCA from an unseeded random draw,
so the RNA target cannot be regenerated. The notebooks load the saved target. A fit from a fresh draw gives a
similar R² but can give different axes. The absolute correlation with the published axes (RNA axis / other axis)
was:

| dataset | fresh draw vs published |
|---|---|
| LPC10 | 0.28 / 0.28 and 0.98 / 0.94 (two draws) |
| Melanoma RNA+ATAC, penalty 1 | 0.46 / 0.29 |
| Melanoma RNA+ATAC, penalty 20 | 0.98 / 0.49 and 0.96 / 0.68 (two draws) |
| MERFISH | 0.905 / 0.998 |
| Sprox-seq B1 | 0.77 / 0.88 |
| Melanoma Hi-C | 0.289 / 0.986 |

**Thread settings are part of the fit** (seed 42 in every case):

| dataset | settings |
|---|---|
| LPC10 | `threads=8`, `OMP_NUM_THREADS=8` |
| Melanoma RNA+ATAC, Hi-C | `threads=8`, `OMP_NUM_THREADS` unset |
| MERFISH | `threads=8`, `refit_all_epochs=0`, `n_restarts=0`, `lam_cross=0.3` |
| Sprox-seq | `threads=4`, `OMP_NUM_THREADS` unset |

**Steps that are not rerun.** These come from other environments or live queries and are read as inputs:

- Fig. 2: the k-l SEACells panels are dropped; the published CellCharter labels are read, and a rerun on LPC10 reaches
  ARI 0.67 to 0.70 against them.
- Fig. 3: `melanoma_with_cnv_in_tissue.h5ad` (inferCNV with a live BioMart query), `starfysh_decon.h5ad` (Starfysh), and
  the iStar maps behind panels a and h-j (GPU training without fixed seeds); the Mellon/DPT panels are dropped.
- Fig. 6: Higashi embeddings.
- Fig. 7: `LPC10_lesion_FOV.h5mu`, `starfysh_composition.csv`, `LPC10S1_codex_per_spot.h5ad`; panels a and
  Supplementary Fig. 6a are dropped. `domain_annot` imports `gaston`.

Supplementary figures other than 6 to 8 are not covered.

**Corrected on 2026-11-07.** The MERFISH cross-term table used 50 µm bins looked up with 100 µm barcodes, which sent 531 of
2,721 bins to the wrong row. `saturn_cross_term.py` now loads 100 µm bins and stops if a barcode is unmatched. The table now has
5,252 of 13,374 genes at q < 0.05 and ΔR² ≥ 0.01 (787 of 8,487 before). Main Fig. 7 did not change.

## Known issues in the code

- **`isoform_usage_pca`** has no option for the gene weight used by the published MERFISH target. The MERFISH tutorial
  gives the recipe inline.
- **`summary.json`** of the LPC10 and Sprox-seq fits has wrong values for `axis_depth_corr` (LPC10: 0.09 / 0.053, actual
  0.117 / 0.121) and the wrong sign for `axis_axis_corr`.
- **`SaturnFit.save`** writes float32, so a saved axis differs from the in-memory axis in the last digits.
