```{include} ../reproducibility/README.md
```

## Fit settings and checks

- **The RNA target depends on an unseeded draw.** `rna_glmpc_real` starts GLM-PCA from an unseeded random state. R²
  stays similar across draws but the axes can change, so seed `np.random` yourself and compare several draws. The
  published fits reproduce only from the saved targets (see the section above).
- **Seed and threads together define a fit.** CPU fits depend on the thread count, because reduction
  order changes. Published fits:
  - LPC10 with `SAT_THREADS=8 OMP_NUM_THREADS=8`.
  - Melanoma with `SAT_THREADS=8` and `OMP_NUM_THREADS` unset.

  The same seed at 4 threads gave LPC10 ρ_RNA r = 0.86 against the published axis; R² was nearly
  unchanged.
- `saturn.repro.set_all_seeds` is called inside `train_saturn`. GPU runs set the cuBLAS workspace
  and deterministic algorithms.
- **The manuscript code runs on this package.** Every analysis script behind Figs 1–7 and every figure
  builder imports `saturn` (see `reproducibility/paper/`). Checked against the published outputs (2026-10-21):
  - **fits**: LPC10 and melanoma refit through `saturn.fit` give the published axes exactly (max |diff| 0)
    and the same coupling matrices;
  - **analyses** (domain K, domain table, marker screens, cis pairs, gradient decomposition, own vs
    cross, TF activity, Fig 7 cross term, mechanism and transition): byte-identical files. The only
    exception is three Fig 7 mechanism files, which differ at 1e-13;
  - **figures**: all 17 PNGs of Figs 1–7 re-render pixel-identical.
- **Analysis steps are thread-sensitive too.** Run post-hoc scripts with `OMP_NUM_THREADS` unset, as
  published. With 8 BLAS threads, 342 of 175k curve lags in the cis-pair table flip between exactly
  tied shifts.
- **Fixed 2026-10-22: coordinate scaling of ρ gradients.** The model is fitted on `zscore_coords` (one
  scalar scale), but the first manuscript runs scaled x and y separately; Fig 2i also evaluated the
  network at per-axis z-scored positions.
  - The package and the manuscript scripts now use the correct chain rule; Fig 2i and Fig 7 were rerun.
  - `rho_gradient(per_axis_scaling=True)`, `spot_decomposition(per_axis_sd=True)` and the scripts'
    `--per-axis` flag reproduce the earlier numbers.
  - Effect, nil on near-square sections (melanoma, Hi-C):
    - LPC10 Fig 7 steepness test: ρ 0.388, p 0.085 → ρ 0.393, p 0.075;
    - Fig 2i "agreement along > across ρ": ATAC axis Wilcoxon p 1e-59 → 2e-125 (holds); RNA axis
      p 6e-18 → 1e-5, with the median r along 0.017 vs across 0.018 (no longer a clear difference);
    - MERFISH and Sprox-seq (S7): rotation p values move by < 0.09, all still non-significant except
      MERFISH steeper-RNA (ρ −0.52, p 0.02, unchanged).

## Manuscript code

`reproducibility/paper/` holds the analysis scripts and figure builders behind every panel. See its [README](https://github.com/C0nc/Saturn/blob/main/reproducibility/paper/README.md).
