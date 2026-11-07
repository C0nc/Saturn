# About Saturn

Spatial multi-omics technologies measure several molecular layers at the same tissue locations, but turning these
data into biological insight remains hard. Most current methods learn one integrated embedding of all modalities, which
limits interpretability.

Saturn instead learns **one interpretable isodepth per modality**: a one-dimensional coordinate that describes the
dominant spatial variation of that modality, which maps directly back to its input features. Saturn extends the isodepth
concept of GASTON to spatial multi-omics. It assumes that the observed spatial pattern of each modality is the joint effect
of the isodepths of all modalities: gene expression, for example, is modelled from both the RNA isodepth and the ATAC
isodepth. An isodepth describes only the dominant variation of its modality, so the variation contributed by the other
modality is needed to model the full observation.

## The model

Saturn learns one isodepth per modality and explains each modality by all of them through additive piecewise-linear pathways:

```
Y_m,j(x) = base_m,j + Σ_k P_m←k,j( d_k(x) ) + noise          k over all modalities
```

- `d_m(x)` is the isodepth of modality *m*: a small ReLU network over z-scored coordinates.
- `P_m←m` is the modality's own program, the piecewise-linear function that GASTON uses for gene expression.
- `P_m←k`, with k ≠ m, is the cross pathway: how modality *m* varies along the isodepth of modality *k*.
- With one modality the model reduces to GASTON.

## What one fit returns

- per-modality isodepths ρ_m;
- a coupling matrix: the share of each modality's predicted variance carried by each axis;
- per-spot own and cross terms.

These feed the post-fit analyses: domains, markers, fitted functions, and where the cross term helps.
See the [analysis guide](analysis_pipeline.md).
