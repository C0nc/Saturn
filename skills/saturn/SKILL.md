---
name: saturn
description: End-to-end guide for Saturn (GASTON extended to M spatial modalities, per-modality isodepths + additive cross pathways) — per-modality targets, fitting and QC, GASTON-style domain annotation, per-domain feature functions for every likelihood, and the cross-term analyses (local gain, per-gene cross test, cis concordance, steepness, mechanism absorption, steep-zone transition). Use when fitting Saturn on any spatial multi-omics data, annotating its domains, or explaining what one modality's axis contributes to another.
metadata:
  type: workflow
---

# Saturn

Saturn = GASTON with one isodepth per modality and additive cross pathways:

```
Y_m,j(x) = base_m,j + Σ_k P_{m←k,j}( d_k(x) )        d_k = ReLU MLP over z-scored coords
```

- Repo: `github.com/C0nc/Saturn` (private). Local copy: `/mnt/HDD2B/Users/yancui/saturn_repo`.
- Import name: `saturn`, version 1.0.0.
- Full process: `docs/analysis_pipeline.md`. Function reference: `docs/api.md`. Read these when a
  step needs more than this page.
- **Environment.** Use a fresh env: `pip install -e ".[all]"`, plus CPU torch. The conda env
  `spatialmultivelo` already has the OLD research package `saturn` 0.1.0 (`SaturnModel`,
  `GastonMixMM`) installed editable. To use the new one there without installing, set
  `PYTHONPATH=/mnt/HDD2B/Users/yancui/saturn_repo/src`; never pip-install both into one env.
- **Device.** CPU. The networks are tiny, so a GPU is slower (kernel-launch bound).

## Pipeline at a glance

| step | call | key setting |
|---|---|---|
| 0 spots | choose the region so the biology dominates the variance | biggest lever: LPC10 R² 0.13 → 0.64 |
| 1 targets | `rna_glmpc_real(X, K=10)`; `atac_spectral(tiles, 15)` or `lsi_real(X, 15)`; `methyl_pca`, `compartment_pca`, `protein_clr_pca`, `isoform_usage_pca`, `tcr_kmer_lsi`; `zscore_coords` | recompute on the selected spots; global scaling; drop depth components |
| 2 fit | `saturn.fit(targets, coords_z, mods, seeds=[42], threads=8, depth={...})` | defaults are the published settings |
| 3 QC | `res.r2`, `heldout_r2(gaston=True)`, `res.axis_depth_corr`, `res.seed_stability`, `res.axis_axis_corr`, `orient()` | see the checklist below |
| 4 domains | `domains.gaston_domains(d_m, Y_m, n_domains=4)` | each axis with its own targets; 20 buckets for sparse ATAC |
| 5 annotate | `annotate.annotate_counts(...)` / `annotate_continuous(...)` | composition + markers + gradient count + median depth |
| 6 functions | `annotate.screen(kind="poisson" / "gauss" / "binomial")` → `plotting.function_figure` | keep features sloped in ≥ 2 domains; report N(≥2)/N |
| 7 cross term | `crossterm.spot_decomposition`, `gene_cross_test`, `cis_windows` + `cis_concordance` | B = linear + 5 hinges; BH; ±50 kb windows |
| 8 steepness | `crossterm.steepness_test(T)` | rotation null; partial out depth |
| 9 mechanism | `mechanism.absorption(..., blocks={"TF":..., "comp":...}, cis=...)`, `drivers`, `boundary_switch` | 15-NN smoothed, 6-column blocks; rotate blocks separately for the null |
| 10 transition | `transition.transect` → `transition_table` | DBSCAN zone, transect along ∇ρ_X, steepest-change bootstrap CI |

## Choosing a reduction

Build 2–4 candidates on the fitted spots and score each with `saturn.check_target(Y, depth, coords,
labels=other_modality_clusters)`:
- `depth_leak` should be < ~0.3;
- `label_knn` and `moran` should be high;
- pick K at the `elbow`; a flat spectrum is a warning.

Break ties with `heldout_r2`. Methylation: use `methyl_pca(..., depth_mode="regress")`. On E11 it had
depth leak 0.07, against 0.69–0.82 for scbs PCA, M-values, ALLCools, Amethyst and MethylVI.
Coverage is the number of covered regions. See the README table.

## Fitting: what the defaults mean (do not change them casually)

- **Stage 1** (30 % of 10,000 epochs): cross pathways off, giving M independent GASTONs. This
  prevents a discrete label swap.
- **Stage 2**: cross pathways on, zero-initialised. The own-axis anchor is on the **residual**
  Y − cross. The full anchor gave coupling 0.09 against a true 0.42.
- **Stage 3**: 4 restart rounds per head. A restart is accepted only if the fit on **20 % held-out
  spots** improves by > 0.5 %.
- **Refit**: 3,000 epochs on all spots.
- `lam_cross=0.03`, `standardize_d=False`, widths (20, 20), `lam_tail=0`.
- `saturn.fit` keeps the seed with the lowest whole-section loss.
- **Seed and thread count together define a fit.** Pin `threads` and `OMP_NUM_THREADS` and record
  them.
  - LPC10 published: 8 / 8. Melanoma: 8 / unset.
  - At 4 threads, LPC10 ρ_RNA r = 0.86 against the published axis.

## QC checklist (before any biology)

1. **R²**: full against own-axis-only. The gap is what the cross term adds. Saturn should beat
   GASTON on held-out spots.
2. **Depth**: |Spearman(axis, log depth)|. Pearson on raw depth understates it (−0.41 vs −0.71).
3. **Seed stability** (≥ 3 seeds): ≳ 0.8. Otherwise ensemble the seeds.
4. **Axis-axis correlation** > ~0.8 means the coupling is not identifiable (spnb67).
5. **Kurtosis** of an axis > ~6 means a "cliff" axis, which is unstable.
6. **Orient** every axis against a marker before comparing anything. Sign is arbitrary: an unanchored
   cross-section replication read −0.83 and was +0.83.
7. **Coupling [m, k]** is a variance share, not causality.

## Domain annotation recipe

For each domain, build the table and name it from **all** of these columns:
- composition: annotation fractions and means;
- Poisson-z markers (counts) or Welch-t markers (continuous);
- for ATAC: nearest-gene markers with class (P / p / l) and kb to the TSS, plus a gene-level column
  from peaks summed into genes;
- `gradient_feats`: features sloped *inside* the domain;
- `median_depth`.

Then:
- Use anatomical names (lesion core / rim / distal WM, melanocytic tumour / invasive front / immune
  niche) and pass them as `labels=`.
- Domain indices are not stable across fits. Paper colours are mapped by label.

## Likelihood per modality (behind GASTON's LRT gate, p < 0.1)

| data | fitter |
|---|---|
| counts with a library (RNA, ATAC) | `piecewise.fit_poisson` (GASTON's own) |
| continuous signed (methylation, Hi-C A/B, protein CLR) | `fit_gauss` |
| successes / trials (isoform usage, clone composition) | `fit_binomial` |
| over-dispersed counts | check `dispersion()`. About 1 means Poisson; NB is not implemented |

A wrong likelihood changes the answer. Gaussian fits on isoform CLR make every isoform of a
varying gene look sloped. The binomial gives 117/800 isoforms against 915/1,500 genes.

## Cross-term analyses: what was found (for calibration)

- **Saturn vs per-modality GASTON**, whole-section R², Saturn wins 9 of 10:
  - LPC10 0.643 / 0.674 vs 0.607 / 0.588;
  - melanoma 0.786 / 0.822 vs 0.729 / 0.747;
  - MERFISH RNA is the exception (0.577 vs 0.611).
- **cis concordance** (gene cross profile vs its own ±50 kb accessibility):
  - LPC10: 0.39 vs 0.12 random windows;
  - melanoma: 0.59 vs 0.33.

  The cross term is chromatin acting on its own gene.
- **Steepness**: gain rises with ρ_X steepness (LPC10 0.12 → 0.62 by quintile, ρ 0.39), but rotation
  p = 0.085. Treat it as a trend: two cases in the main figure, the rest in the supplement.
- **Absorption**, median, LPC10 / melanoma:
  - composition 87 % / 59 %;
  - TF 43 % / 18 %;
  - cis 8 % / 19 %;
  - null 20 % / 15 %.

  Composition is partly circular (proportions come from the same RNA). TF activity (chromVAR,
  ATAC-only) is independent.
- **Transition**: each axis puts its steepness on a different tissue boundary. ρ_RNA is steep on the
  lesion / invasive edge; ρ_ATAC on the cortex-neuron / melanocytic switch. Where ρ_RNA is flat but
  cells switch, only the ATAC pathway can fit the RNA.
  - LPC10: a transient interface (leptomeninges, Ttr/Enpp2, peak RNA gain) sits between the two.

## Traps

- **Reductions.** Computed on a larger section, they carry that section's variance. Recompute after
  every spot selection.
- **Depth components.** LSI / spectral component 1 is depth. Check components 2–3 too (umb1031 had
  two).
- **Methylation coverage** is the number of covered regions, not Seurat `nCount_DNAm` (a sum of
  residuals).
- **uint32 counts.** `np.argsort(-tot)` wraps around; cast to float. `np.percentile` returns NaN on
  NaN; use `nanpercentile`.
- **Promoter distance.** Measure it to the tile edge, not the midpoint.
- **Per-domain lines** extrapolate. Clip function panels to the observed range.
- **Coupling** is unidentifiable when axes are collinear. Sparse chromatin gives about 0–5 % held-out
  R² for every method.
- **Mechanism null.** Rotate each 6-column block separately; a 12-column null is unfair.
- **Melanoma steep zone.** It sits at the section edge with about 2.5× the depth. Lead with gene and
  TF evidence there.

## Figures

- Diagnostic: `saturn.plotting`, five columns `ρ | domains | RAW | FITTED | function`.
- Paper: `Saturn/published_figures/figN/`, using `_style/paper_style.py` with `apply()` and
  `save_dual()`.
  - Arial ≥ 5 pt, PNG 600 dpi + SVG, one SVG per sub-panel.
  - Saturn low-saturation theme; sticky palette.
  - `coolwarm` for ρ and scores, `viridis` for distance.
- The older diagnostic pipeline for many datasets lives in `/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts`
  (see the `saturn-postfit-figures` skill). The library functions were factored out of it and check
  identical on LPC10.
