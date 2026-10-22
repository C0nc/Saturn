# Manuscript scripts (provenance)

These are the original scripts behind the manuscript figures, copied verbatim. They contain the
lab's absolute data paths, and they import the research copy of Saturn (`Saturn/src`) through
`sys.path`, so they do not run outside that file system. Use them to see exactly how a number was
made. Use the `saturn` package for new analyses: its functions were factored out of these scripts
and check identical on LPC10 (see the top-level README).

| step | script | library equivalent |
|---|---|---|
| targets: shared helpers, all datasets | `scripts/fit/prep_real2.py` | `saturn.preprocess` |
| targets: LPC10 FOV (Fig 2) | `scripts/fit/lpc10_fov_prep.py` | `rna_glmpc_real`, `atac_spectral` |
| targets: melanoma tissue (Fig 3) | `scripts/fit/melanoma_tissue_prep.py` | same |
| fit + summary + draft figure | `scripts/fit/run_real2.py` (`--dataset X --seeds 1 --seed0 42`, `SAT_THREADS`) | `saturn.fit` |
| Saturn vs GASTON R² (Fig 7a) | `scripts/fit/gaston_default_r2.py` | `saturn.heldout_r2(gaston=True)` |
| domains + annotation (RNA + ATAC) | `scripts/postfit/domain_annot.py`, `lpc10_fov_domain_annot.py` | `saturn.domains`, `saturn.annotate.annotate_counts` |
| GASTON DP + Poisson function panels | `scripts/postfit/lpc_gaston_fig4l.py`, `lpc_gene_isodepth.py` | `domains.gaston_domains`, `piecewise.fit_poisson` |
| per-domain fitters | `scripts/postfit/piecewise.py` | `saturn.piecewise` |
| continuous / isoform / TCR modalities | `scripts/postfit/{continuous,isoform,tcr}_marker_panels.py` | `annotate.screen(kind=...)`, `plotting.function_figure` |
| parallel marker screen, per-gene cis panels | `scripts/postfit/{marker,gene}_function_panels.py` | `annotate.screen` |
| cross term, cis, steepness, length (Fig 7 a–h) | `scripts/fig7/saturn_cross_term.py` | `saturn.crossterm` |
| mechanism (Fig 7 i) | `scripts/fig7/saturn_cross_mechanism.py` | `saturn.mechanism` |
| transition (Fig 7 j) | `scripts/fig7/saturn_steep_transition.py` | `saturn.transition` |
| Fig 7 assembly | `scripts/fig7/fig7_cross_term.py` | none |
| figure style | `style/paper_style.py`, `style/saturn_theme.py` | none |

The Figs 2 and 3 end-to-end reproduction packages are kept with the figures
(`published_figures/fig2/repo`, `fig3/repo`). Each has a `run_figN.sh`, a pinned environment and a
`REPRO_CHECK.md`.

Published fit environments: `spatialmultivelo` conda env, CPU.
- LPC10: `SAT_THREADS=8 OMP_NUM_THREADS=8`.
- Melanoma: `SAT_THREADS=8` with `OMP_NUM_THREADS` unset.
- Every fit used seed 42.
