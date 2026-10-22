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
| 4 domains | `k, _ = domains.choose_k(d_m, Y_m)`, then `domains.gaston_domains(d_m, Y_m, n_domains=k)` | K per axis at the DP-error elbow (published: LPC10 3/2, melanoma 4/3); each axis with its own targets |
| 5 annotate | `annotate.annotate_counts(...)` / `annotate_continuous(...)` | composition + markers + gradient count + median depth |
| 6 functions | `annotate.screen(kind="poisson" / "gauss" / "binomial")` → `plotting.function_figure` | keep features sloped in ≥ 2 domains; report N(≥2)/N |
| 7 cross term | `crossterm.spot_decomposition`, `gene_cross_test`, `cis_windows` + `cis_concordance` | B = linear + 5 hinges; BH; ±50 kb windows |
| 8 steepness | `crossterm.steepness_test(T)` | rotation null; partial out depth |
| 9 mechanism | `mechanism.absorption(..., blocks={"TF":..., "comp":...}, cis=...)`, `drivers`, `boundary_switch` | 15-NN smoothed, 6-column blocks; rotate blocks separately for the null |
| 10 transition | `transition.transect` → `transition_table` | DBSCAN zone, transect along ∇ρ_X, steepest-change bootstrap CI |

## Choosing the input reduction (do this before every fit)

Saturn never sees raw counts. Each modality enters as an (N × K) embedding, and **the reduction
decides what the isodepth can find**:
- a reduction that leaks depth gives a depth axis;
- a flat spectrum gives a poor fit.

### Defaults per modality

| modality | call | K | notes / rejected alternatives |
|---|---|---|---|
| RNA counts | `rna_glmpc_real(X, K=10)` | 10 | Poisson GLM-PCA over 3k Pearson-residual HVGs, depth as offset; falls back to residual PCA if GLM-PCA diverges |
| ATAC fragments | `atac_spectral(tiles, n_comps=15)` | 14 | SnapATAC2 5 kb tiles → select_features(250k) → spectral, component 1 (depth) dropped. Published LPC10 / melanoma target |
| ATAC / CUT&Tag peak matrix | `lsi_real(X, K=15)` | 10–15 | TF-IDF → SVD, component 1 dropped. Check components 2–3 for depth too (umb1031 had two). Very sparse marks (median < ~200 fragments): fewer components |
| DNA methylation | `methyl_pca(resid, K=10, depth=n_covered_regions, depth_mode="regress")` | 10 | see the worked example below |
| Hi-C | `compartment_pca(AB, K=10)` | 10 | A/B scores per bin. **Fast-Higashi rejected**: R² 0.12–0.20 vs 0.37, flat spectrum (top 10 = 8 % of variance) |
| protein / ADT | `protein_clr_pca(X, K=10)` | ≤ 10 | CLR per spot, all proteins |
| long-read isoforms | `isoform_usage_pca(X, iso_gene, K=10, depth=d)` | 10 | shrunken within-gene usage → CLR → depth regressed. Models usage, not expression |
| TCR / BCR | `tcr_kmer_lsi(tra, trb)` + `mask` | 8 | sequence identity had no spatial signal at our depth. Clone composition is the alternative |
| other continuous (H&E foundation-model features, mechanics) | centred PCA → `scale_target(P, "global")` | 10 | do NOT Otsu-mask H&E patches |
| coordinates | `zscore_coords(xy)` | — | one common scale (GASTON's per-axis z-score distorts geometry) |

### Worked example: methylation (E11 spatial-DMT, 8 reductions)

| reduction | depth leak ↓ | RNA-cluster 15-NN ↑ | Moran's I ↑ |
|---|---|---|---|
| scbs residual PCA (paper default) | 0.82 | 0.52 | 0.61 |
| residual PCA, drop depth PCs | 0.22 | 0.41 | 0.45 |
| **residual PCA, log coverage regressed (chosen)** | **0.07** | 0.47 | 0.51 |
| ALLCools-style β / cell mean | 0.75 | 0.46 | 0.55 |
| M-values | 0.81 | 0.48 | 0.57 |
| 100 kb windows (Amethyst-style) | 0.78 | 0.46 | 0.50 |
| MethylVI | 0.69 | 0.45 | 0.30 |

Coverage alone predicts the RNA clusters at 0.24 (chance 0.21). The reductions that keep the
coverage component score higher on the cluster test partly *because* of coverage, so do not pick on
`label_knn` alone.

Methylation traps:
- Coverage is the number of covered regions (`nFeature_DNAm`), NOT `nCount_DNAm`, which is a sum of
  residuals.
- Never log-transform fractions.
- A published methylation PCA with a hard elbow is acceptable: E11 MPCA components 1–10 hold 85 %,
  and component 11 drops 16×.

Script: `paper/scripts/fit/meth_dr_bench.py`.

### Procedure for a new modality or dataset

1. Build 2–4 candidate reductions **on exactly the spots you will fit**. Recompute after any change
   of spot selection: LPC10 R² went 0.13 → 0.64 from region choice plus recomputed reductions.
2. Score each candidate:
   ```python
   for name, Y in candidates.items():
       print(name, saturn.check_target(Y, depth=depth, coords=xy, labels=rna_clusters))
   ```
   | metric | want | meaning |
   |---|---|---|
   | `depth_leak` | < ~0.3 | max \|Spearman\| of the top 5 components with log depth. If it is high, drop that component (`lsi_real(drop_first=False)`) or regress depth |
   | `label_knn` | high | predicts labels from ANOTHER modality (RNA clusters), never this modality's own clusters (circular) |
   | `moran` | high | spatial structure an isodepth can fit |
   | `var_share`, `elbow` | clear elbow | set K at the elbow. No elbow (flat spectrum) means expect a poor 1-D fit |
3. Break ties with one quick Saturn fit per candidate and compare `saturn.heldout_r2`. This is the
   deciding criterion.
4. Keep **global scaling** (the default in every builder). Never z-score each component: on
   simulations, z-scored GLM-PCs made a single isodepth find the WRONG axis (own 0.43 vs other 0.89).
5. Save `Y_<mod>` and `depth_<mod>` with the fit, so later steps can recheck the depth confound per
   domain.

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
