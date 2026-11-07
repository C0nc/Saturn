# Getting started

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

`python docs/examples/quickstart_sim.py` recovers both axes at |Spearman| 1.000.

## Real data in five steps

```python
import numpy as np, saturn
from saturn import domains, annotate, crossterm

# 1. targets: recompute every reduction on exactly the spots you fit
Y_rna, info_r = saturn.rna_glmpc_real(rna.X, K=10)              # GLM-PCA over 3k HVGs
Y_atac, info_a = saturn.atac_spectral(tiles, n_comps=15)         # SnapATAC2, comp 1 (depth) dropped
#   or: saturn.lsi_real(peaks.X, K=15) / compartment_pca / protein_clr_pca / isoform_usage_pca
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
| Hi-C | `compartment_pca` on A/B scores | 10 | Fast-Higashi embeddings fit badly (R² 0.12–0.20 vs 0.37): flat spectrum |
| protein / ADT | `protein_clr_pca` (CLR per spot) | ≤ 10 | all proteins; panels are small |
| long-read isoforms | `isoform_usage_pca` (shrunken within-gene usage, CLR) | 10 | models usage, not expression; regress depth |
| anything else continuous (H&E features, mechanics) | centred PCA → `scale_target` | 10 | |

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

The complete process, with the decision at every step, is in the [analysis guide](analysis_pipeline.md).
