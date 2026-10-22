# Saturn

**Saturn extends GASTON to M spatial modalities.** It learns one isodepth (a smooth 1-D
"topographic" coordinate over the tissue) per modality, and explains every modality by its own axis
*plus* every other modality's axis through additive piecewise-linear pathways:

```
Y_m,j(x) = base_m,j + Σ_k P_{m←k, j}( d_k(x) ) + noise          k over ALL modalities
```

- `d_m(x)` is the isodepth of modality *m*: a small ReLU network over z-scored coordinates.
- `P_{m←m}` is the modality's own program (GASTON's piecewise-linear expression function).
- `P_{m←k}`, k ≠ m, is how modality *m* is shaped by modality *k*'s axis, e.g. chromatin shaping RNA.
- With M = 1 this is exactly GASTON.

From one fit you get:
- per-modality isodepths ρ_m;
- a directed **coupling matrix**: the share of each modality's predicted variance carried by each axis;
- per-spot own and cross terms. Those feed the post-fit analyses here:
  - domains;
  - markers;
  - fitted functions;
  - where and why the cross term helps.

| | Saturn | GASTON (one per modality) |
|---|---|---|
| Hard simulation: RNA axis recovery, mean \|Spearman\| over 24 configs | **0.997** | 0.687 (default) / 0.841 (global-scaled) |
| *Whole-section R² of the targets, RNA / second modality:* | | |
| LPC10 RNA / ATAC | **0.643 / 0.674** | 0.607 / 0.588 |
| Melanoma RNA / ATAC | **0.786 / 0.822** | 0.729 / 0.747 |
| MERFISH RNA / isoform | 0.577 / **0.486** | **0.611** / 0.468 |
| Sprox-seq RNA / protein | **0.611 / 0.557** | 0.498 / 0.483 |
| Melanoma Hi-C RNA / Hi-C | **0.204 / 0.587** | 0.185 / 0.524 |

The GASTON column is the best-loss run of 30 restarts. Saturn's R² rises above its own-axis-only
R² because of the cross pathways.

---

## Install

```bash
conda create -n saturn python=3.10 -y && conda activate saturn
# CPU is the recommended device: the networks are tiny, so they are kernel-launch bound on a GPU
pip install torch --index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/C0nc/Saturn.git && cd Saturn
pip install -e ".[all]"        # gaston-spatial (domains, Poisson fits), glmpca, anndata, mudata
pytest -q tests                # 8 smoke tests, under a minute
```

Optional extras:

| extra | for |
|---|---|
| `[gaston]` | `saturn.domains`, `piecewise.fit_poisson`, the GASTON baseline in `heldout_r2` |
| `[glmpca]` | true Poisson GLM-PCA RNA targets (otherwise Pearson-residual PCA) |
| `snapatac2` (pip, separately) | `preprocess.atac_spectral`, the published ATAC target |

Core dependencies: numpy, scipy, pandas, scikit-learn, matplotlib, torch ≥ 2.0. Tested on Python 3.10,
torch 2.12 and 2.14 (CPU).

> **Name clash.** The distribution is called `saturn-isodepth`, but it installs the import name
> `saturn`. Do not install it into an environment that already has the older research package
> `saturn` 0.1.0 (`SaturnModel`, `GastonMixMM`). Python would resolve `import saturn` to whichever
> comes first.

## Quickstart (simulation, ~1 min on CPU)

```python
import saturn
from saturn.sim import SimConfig, simulate

data = simulate(SimConfig(n_side=30, cross_RA=0.5, cross_AR=0.2, seed=0))
xy = saturn.zscore_coords(data.coords)
res = saturn.fit([data.Y_R, data.Y_A], xy, modalities=["RNA", "ATAC"], seeds=[0, 1], threads=4)
res.axes          # (N, 2) isodepths
res.coupling      # (2, 2): row = modality, column = driving axis
res.r2            # per modality: full / own_axis_only / other_axes_only
res.save("outputs/sim", coords=data.coords)
```

`python examples/quickstart_sim.py` recovers both axes at |Spearman| 1.000.

## Real data in five steps

```python
import numpy as np, saturn
from saturn import domains, annotate, crossterm

# 1. targets: recompute every reduction on exactly the spots you fit
Y_rna, info_r = saturn.rna_glmpc_real(rna.X, K=10)              # GLM-PCA over 3k HVGs
Y_atac, info_a = saturn.atac_spectral(tiles, n_comps=15)         # SnapATAC2, comp 1 (depth) dropped
#   or: saturn.lsi_real(peaks.X, K=15) / methyl_pca / compartment_pca / protein_clr_pca / isoform_usage_pca
xy = saturn.zscore_coords(coords)                                # one common scale keeps the aspect ratio

# 2. fit (pin threads: CPU reduction order changes the basin)
res = saturn.fit([Y_rna, Y_atac], xy, ["RNA", "ATAC"], seeds=[42], threads=8,
                 depth={"RNA": info_r["depth"], "ATAC": info_a["depth"]})
res.axes = saturn.orient(res.axes, marker_scores)                # isodepth sign is arbitrary
res.save("fits/my_sample", coords=coords, barcodes=barcodes)

# 3. domains along each axis + annotation
kR, _ = domains.choose_k(res.axes[:, 0], Y_rna)                  # K at the DP-error elbow
labR = domains.gaston_domains(res.axes[:, 0], Y_rna, n_domains=kR)
table = annotate.annotate_counts(rna.X, rna.var_names, labR, res.axes[:, 0], "RNA",
                                 cats={"cluster": clusters}, conts={"microglia": mg_score})

# 4. which features are functions of the axis (>= 2 sloped domains)
scr, slope, inter, idx = annotate.screen(rna.X, rna.var_names, labR, res.axes[:, 0],
                                         kind="poisson", lib=np.asarray(rna.X.sum(1)).ravel())

# 5. what the cross pathway explains
T, S = crossterm.spot_decomposition(res.model, xy, coords, [Y_rna, Y_atac], 0, 1)
G, prof, genes = crossterm.gene_cross_test(rna.X, rna.var_names, T.rho_RNA, T.rho_X)
```

## Choosing the input reduction for a modality

Saturn never sees raw counts. Each modality enters as a small (N × K) embedding, so **the reduction
decides what the isodepth can find**. A reduction that leaks sequencing depth gives a depth axis; one
with a flat spectrum gives a poor fit. The recommended defaults:

| modality | recommended | K | why / alternatives |
|---|---|---|---|
| RNA counts | `rna_glmpc_real` (Poisson GLM-PCA, depth as offset) | 10 | GASTON's choice; falls back to Pearson-residual PCA |
| ATAC fragments | `atac_spectral` (SnapATAC2 5 kb tiles), component 1 dropped | 14–15 | `lsi_real` for a called peak matrix. Call peaks only post hoc |
| CUT&Tag / histone | `lsi_real` | 10–15 | for very sparse marks (median < ~200 fragments) try fewer components |
| DNA methylation | `methyl_pca(resid, depth=coverage, depth_mode="regress")` | 10 | see below |
| Hi-C | `compartment_pca` on A/B scores | 10 | Fast-Higashi embeddings fit badly (R² 0.12–0.20 vs 0.37): flat spectrum |
| protein / ADT | `protein_clr_pca` (CLR per spot) | ≤ 10 | all proteins; panels are small |
| long-read isoforms | `isoform_usage_pca` (shrunken within-gene usage, CLR) | 10 | models usage, not expression; regress depth |
| TCR / BCR | `tcr_kmer_lsi` with a `mask` | 8 | sequence identity carried no spatial signal at our depth. Clone composition is the alternative |
| anything else continuous (H&E features, mechanics) | centred PCA → `scale_target` | 10 | |

**Methylation: worked example.** The manuscript fits do not use methylation, but the choice was
benchmarked on E11 spatial-DMT (8 reductions × 4 label-free metrics, `paper/` provenance):

| reduction | depth leak ↓ | RNA-cluster 15-NN acc ↑ | Moran's I ↑ |
|---|---|---|---|
| scbs residual PCA (paper default) | 0.82 | 0.52 | 0.61 |
| residual PCA, drop depth PCs | 0.22 | 0.41 | 0.45 |
| **residual PCA, log coverage regressed** (`methyl_pca` default) | **0.07** | 0.47 | 0.51 |
| ALLCools-style β / cell mean | 0.75 | 0.46 | 0.55 |
| M-values | 0.81 | 0.48 | 0.57 |
| 100 kb windows (Amethyst-style) | 0.78 | 0.46 | 0.50 |
| MethylVI | 0.69 | 0.45 | 0.30 |

Coverage alone predicts the RNA clusters at 0.24, against 0.21 for chance. Every reduction that
keeps the coverage component looks better on the cluster test partly *because* of coverage. Use the
coverage-regressed residuals. Methylation-specific traps:
- Coverage is the number of covered regions, **not** Seurat `nCount_DNAm` (that is the sum of
  residuals).
- Do not log-transform fractions.
- If the authors published a methylation PCA with a clear elbow (E11 MPCA: components 1–10 hold 85 %),
  using it is also fine.

**How to choose for a new modality or a new dataset.** Build 2–4 candidate reductions on exactly the
spots you will fit, then score each:

```python
for name, Y in candidates.items():
    print(name, saturn.check_target(Y, depth=depth, coords=xy, labels=rna_clusters))
```

| metric | want | meaning |
|---|---|---|
| `depth_leak` | < ~0.3 | max \|Spearman\| of the top 5 components with log depth. If one component leaks, drop it (`lsi_real(drop_first=False)`) or regress depth |
| `label_knn` | high | predicts labels from *another* modality (e.g. RNA clusters), so biology is kept without circularity |
| `moran` | high | spatial structure in the leading components, which an isodepth can fit |
| `var_share`, `elbow` | a clear elbow | choose K at the elbow. A flat spectrum (no elbow) means no low-dimensional axis; expect a poor fit |

Then break ties with one quick Saturn fit per candidate and compare `heldout_r2`. This is the
deciding criterion, because it measures what the model can actually use. Whatever you choose, keep
**global scaling** (`scale_target(P, "global")`, the default in every builder); never z-score each
component. Recompute the reduction after every change of spot selection.

The complete process, with the decision at every step and the traps that have already cost time,
is in **[docs/analysis_pipeline.md](docs/analysis_pipeline.md)**. A function-by-function reference
is in **[docs/api.md](docs/api.md)**.

## Package layout

```
src/saturn/
  core.py         Saturn model + train_saturn (3 stages: warm-up, cross on, held-out restarts)
  preprocess.py   per-modality targets, global scaling, coordinate scaling
  fit.py          fit (best-of-seeds), SaturnFit, decompose, r2_split, orient, heldout_r2, load_model
  domains.py      GASTON DP domains per axis, K at the DP-error elbow, boundaries, isodepth bins
  piecewise.py    per-domain Poisson / Gaussian / binomial fits behind GASTON's LRT gate
  annotate.py     domain tables, Poisson-z and Welch-t markers, nearest genes, feature screens
  crossterm.py    local gain, gradients, per-gene cross test, cis windows, steepness, own vs cross
  mechanism.py    absorption by cis / TF / composition blocks, drivers, boundary switches
  transition.py   steep-zone transect, sigmoid / bump switch points, bootstrap CIs
  gradient.py     isodepth gradients (autograd), feature gradients, along / across ρ
  cis.py          gene vs cis-peak function pairs, gradient agreement along / across ρ
  tfactivity.py   chromVAR-style TF activity on isodepth bins and per spot
  plotting.py     domain maps and the 5-column function figure
  sim.py          two-modality simulator of the generative model
examples/         quickstart_sim.py
tests/            smoke tests
docs/             analysis_pipeline.md, api.md
skills/saturn/    SKILL.md for Claude Code (copy to ~/.claude/skills/saturn/)
paper/            the original scripts behind the manuscript figures (provenance; hard-coded paths)
```

## Reproducibility

- **Seed and threads together define a fit.** CPU fits depend on the thread count, because reduction
  order changes. Published fits:
  - LPC10 with `SAT_THREADS=8 OMP_NUM_THREADS=8`.
  - Melanoma with `SAT_THREADS=8` and `OMP_NUM_THREADS` unset.

  The same seed at 4 threads gave LPC10 ρ_RNA r = 0.86 against the published axis; R² was nearly
  unchanged.
- `saturn.repro.set_all_seeds` is called inside `train_saturn`. GPU runs set the cuBLAS workspace
  and deterministic algorithms.
- **The manuscript code runs on this package.** Every analysis script behind Figs 1–7 and every figure
  builder imports `saturn` (see `paper/`). Checked against the published outputs (2026-10-21):
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

## License

MIT.
