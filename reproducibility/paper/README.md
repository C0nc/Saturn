# Manuscript code, aligned with the Saturn package

Everything behind the manuscript figures, copied from the lab file system by
`figures/_tools/sync_paper.py`:
- `scripts/`: 78 analysis scripts;
- `figures/`: the figure builders;
- `style/`: the house figure style.

**All Saturn steps go through the `saturn` package of this repo.** Only dataset loaders and
figure-specific analyses stay in the scripts:
- the fit, `saturn.fit` / `train_saturn`;
- targets, `saturn.preprocess`;
- domains and the per-axis K, `saturn.domains`;
- per-domain fits, `saturn.piecewise`;
- annotation, `saturn.annotate`;
- gradients, `saturn.gradient`;
- cis pairs, `saturn.cis`;
- TF activity, `saturn.tfactivity`;
- cross term, mechanism and transition, `saturn.crossterm` / `mechanism` / `transition`.

The copies keep the lab's absolute paths: they document how every number was made and run only on that
server. To rerun a figure elsewhere, use the notebooks in [`../notebooks/`](../notebooks/README.md), which copy this
code into a workspace and replace the paths. The self-contained, runnable packages for Figs 2 and 3 are
`figures/fig2/run_fig2.sh` and
`figures/fig3/run_fig3.sh` (on the server: `published_figures/figN/repo/`). Each repo pins a Saturn commit
in `SATURN_VERSION` and reads the package from `$SATURN_SRC`.

## Checked on 2026-10-21 against the published outputs

| what | result |
|---|---|
| LPC10 and melanoma refit with `saturn.fit` (seed 42, published thread settings) | axes identical (max \|diff\| 0), coupling identical |
| domain K table, LPC10 domain table, marker screens, cis pairs, gradient decomposition, own vs cross, TF activity (bins and per spot), inflammatory-gene fits | byte-identical |
| Fig 7 cross term (5 datasets), cis windows, gene length, transition | byte-identical |
| Fig 7 mechanism | identical to 1e-13 |
| figures: every PNG of Figs 1–7 (17 files) | pixel-identical |

Run the analysis scripts with `OMP_NUM_THREADS` unset (as published). Run the fits with the recorded
thread settings:
- LPC10: `SAT_THREADS=8 OMP_NUM_THREADS=8`;
- melanoma: `SAT_THREADS=8`, OMP unset.

## Figure → builder → analysis

| figure | builder (entry point) | analyses (scripts/) | fit / targets |
|---|---|---|---|
| 1 model + benchmark | `fig1/fig1_model_benchmark.py` | `bench_v5_full.py` (7 methods × 48 configs), `simulate_v5.py` / `saturn.sim` | — |
| 2 LPC lesion | `fig2/fig2_lpc.py` (assembles `fig2_lpc_v2.py`) | `lpc10_fov_domain_annot.py`, `marker_function_panels.py`, `saturn_tf_activity_rho.py`, `saturn_tf_activity_spatial.py`, `saturn_cis_function_pairs.py`, `saturn_cis_gradient_decomp.py`, `lpc10_fov_posthoc.py`, `lpc10_seacells_cis_benchmark.py` | `lpc10_fov_prep.py`, `run_real2.py --dataset lpc10_fov` |
| 3 melanoma | `fig3/fig3_parts.py` (uses `fig3_melanoma.py`, `fig3_melanoma_v2.py`) | `saturn_own_vs_cross.py`, `marker_function_panels.py`, `melanoma_chromatin_potential*.py`, `melanoma_istar_*.py`, `mel_cellrank_kernels.py`, `mel_cellrank_arrows.py`, `mel_gradient_transition.py`, `mel_baseline_mellon.py` | `melanoma_tissue_prep.py`, `run_real2.py --dataset melanoma_tissue` |
| 4 MERFISH isoform | `fig4/fig4_parts.py` (uses `fig4_isoform_sv.py`, `fig4_isoform_v3.py`) | `rtt_isoform_sv_isodepth.py`, `rtt_isoform_sv_fig4_prep.py`, `rtt_sst_isoform_transition.py`, `merfish_isoform_deep.py` | `rtt_isoform_ratio_pca.py`, `rtt_isoform_refit_nocollapse.py` |
| 5 Sprox-seq | `fig5/fig5_B1.py` | `sproxseq_B1_fig5_prep.py`, `sproxseq_D1_crypt_niche.py`, `sproxseq_gc_trajectory.py` | `sproxseq_prep_depthreg.py`, `run_real2.py` |
| 6 melanoma Hi-C | `fig6/fig6_melanoma_hic.py` | `mel_hic_insulation_granger.py`, `mel_hic_bins_partial.py`, `mel_hic_cis_threeway.py`, `mel_hic_granger_examples.py` | `mel_hic_abins_prep.py`, `run_real2.py` |
| 7 cross term (+ S6, S7, S8) | `fig7/fig7_cross_term.py` | `saturn_cross_term.py`, `saturn_cross_mechanism.py`, `saturn_steep_followup.py`, `saturn_steep_transition.py`, `gaston_default_r2.py` | (fits above) |
| domain annotation supplements (Figs 2–6) | `fig_supp/supp_domains.py` + `supp_domain_specs.py` | each figure's own `load()`; markers `saturn.annotate`, fits `saturn.piecewise`, K `saturn.domains` | — |
| Fig 1 benchmark supplement | `fig1/supp_fig1_benchmark.py` | `bench_v5_full` tables | — |
| domain K (all figures) | — | `domain_k_elbow.py` → `domain_k.py` | — |

## Notes

- **Superseded builders.**
  - `fig5/fig5_parts.py` and `fig5_sproxseq.py` build the earlier A1-section Fig 5; the current Fig 5 is
    `fig5_B1.py`. Running `fig5_parts.py` overwrites `fig5_part{1,2}`.
  - `fig3/fig3_melanoma_v2.py` run on its own stops at its manifest (`KeyError: 'o'`); `fig3_parts.py` is
    the entry point.
  - `fig_supp/supp_sim_spatial.py` uses the retired GASTON-Mix-MM model, not Saturn.
- **Gradient scaling (rerun 2026-10-22).** The first Fig 2i and Fig 7 runs scaled x and y separately
  when converting ρ gradients to raw coordinates. The scripts now use the package's correct chain rule
  and Fig 2i / Fig 7 were rerun with it; `--per-axis` on `saturn_cis_gradient_decomp.py` /
  `saturn_cross_term.py` reproduces the earlier numbers. The effect is listed in the top-level README.
  The byte-identity table above refers to the pre-fix run.
