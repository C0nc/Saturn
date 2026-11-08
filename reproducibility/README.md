# Reproducibility

One notebook per main figure reproduces its analysis and draws its panels, step by step, with the `saturn` API. The
code is written out in the cells: targets, fit, domains, feature functions and every downstream test, followed by one
cell per panel, lettered as in the manuscript. The notebooks are stored with their outputs. The tutorials are in
`docs/tutorials/`.

| notebook | figure | panels drawn | run time |
|---|---|---|---|
| `fig1_simulation` | Fig. 1 | c–e (Saturn only) | about 2 min |
| `fig2_lpc` | Fig. 2 | b–n | about 8 min |
| `fig3_melanoma` | Fig. 3 | b–o | about 7 min |
| `fig4_merfish_isoform` | Fig. 4 | b–o | about 2 min |
| `fig5_sproxseq` | Fig. 5 | b–h | about 4 min |
| `fig6_melanoma_hic` | Fig. 6 | b–h | about 15 min |
| `fig7_cross_term` | Fig. 7, Supplementary Figs. 12–15 | Fig. 7b–j, Supplementary Figs. 12b–d, 13–15 | about 12 min |

Run times are for a fresh run on 8 CPU threads, most of it the Saturn fit. Panels that are schematics are not drawn.
The notebooks do not run comparison methods, so these panels are not drawn either: Fig. 1d–e baselines, Fig. 3l–o DPT
and Mellon, Fig. 7a and Supplementary Fig. 12a (GASTON).

## Running a notebook

Each notebook reads its inputs from one folder, `saturn_workspace/`, next to the notebook (or the folder named by
`SATURN_WORKSPACE`), and writes its results to `saturn_workspace/results/figN/`. `workspace_files/figN.json` lists every
input and where it comes from (GEO accession, reference genome, or the tool that made it). `make_workspace.py` links
them into a workspace:

```bash
python reproducibility/make_workspace.py fig4 /path/to/data_root workdir/saturn_workspace
PYTHON=python bash reproducibility/run_figure.sh fig4_merfish_isoform workdir
```

Or open the notebook in Jupyter from `workdir`. The fit and other slow steps are saved in `results/` and reused when a
notebook is run again; delete that folder to recompute them. `_src/` holds the source of each notebook (`build.py` turns it
into the `.ipynb`).

**Results read from `published/`.** Steps that need another conda environment, a GPU or a live query are not rerun; their
outputs sit in `saturn_workspace/published/`, and the notebook names the tool and the manuscript script that made them:
CellCharter and Catalign (Fig. 2), the SEACells benchmark (Fig. 2m,n), the tumour call with inferCNV and Starfysh, iStar and
the Mellon patches (Fig. 3), SPLISOSM, scVI and CellRank (Fig. 4), Higashi (Fig. 6), and cell2location, Starfysh and Catalign
(Fig. 7). Fig. 7 reads the fits of Figs. 2–6 from there, so it runs on its own.

**The saved RNA target is read, not rebuilt.** `saturn.preprocess.rna_glmpc_real` starts GLM-PCA from an unseeded random
draw, so the RNA target cannot be regenerated. Each notebook reads the saved RNA target and rebuilds everything else. A fit
from a fresh draw gives a similar R² but can give different axes. The absolute correlation with the published axes (RNA axis /
other axis) was:

| dataset | fresh draw vs published |
|---|---|
| LPC10 | 0.28 / 0.28 and 0.98 / 0.94 (two draws) |
| Melanoma RNA+ATAC, penalty 1 | 0.46 / 0.29 |
| Melanoma RNA+ATAC, penalty 20 | 0.98 / 0.49 and 0.96 / 0.68 (two draws) |
| MERFISH | 0.905 / 0.998 |
| Sprox-seq B1 | 0.77 / 0.88 |
| Melanoma Hi-C | 0.289 / 0.986 |

**Thread settings are part of the fit** (seed 42 in every case). Each notebook sets them in its first cell, before torch is
imported:

| dataset | settings |
|---|---|
| LPC10 | `threads=8`, `OMP_NUM_THREADS=8` |
| Melanoma RNA+ATAC, Hi-C | `threads=8`, `OMP_NUM_THREADS` unset |
| MERFISH | `threads=8`, `refit_all_epochs=0`, `n_restarts=0`, `lam_cross=0.3` |
| Sprox-seq | `threads=4`, `OMP_NUM_THREADS` unset |

With these settings the notebooks give the published axes exactly (largest difference 0, or 2e-7 where a saved float32
axis is read back), and the tables and the numbers in the panels are the published ones.

**Corrected on 2026-11-07.** The MERFISH cross-term table used 50 µm bins looked up with 100 µm barcodes, which sent 531 of
2,721 bins to the wrong row. The table now uses 100 µm bins and stops if a barcode is unmatched. It has 5,252 of 13,374
genes at q < 0.05 and ΔR² ≥ 0.01 (787 of 8,487 before). Main Fig. 7 did not change.

## Known issues in the code

- **`isoform_usage_pca`** has no option for the gene weight used by the published MERFISH target. The Fig. 4 notebook and
  the MERFISH tutorial give the recipe inline.
- **`summary.json`** of the LPC10 and Sprox-seq fits has wrong values for `axis_depth_corr` (LPC10: 0.09 / 0.053, actual
  0.117 / 0.121) and the wrong sign for `axis_axis_corr`.
- **`SaturnFit.save`** writes float32, so a saved axis differs from the in-memory axis in the last digits.

## Manuscript code

`paper/` holds the analysis scripts and figure builders as they were run for the manuscript. See its
[README](https://github.com/C0nc/Saturn/blob/main/reproducibility/paper/README.md).
