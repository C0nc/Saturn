# Saturn: interpretable isodepths for spatial multi-omics

```{image} _static/saturn_overview.png
:alt: Saturn model overview
:width: 520px
:align: center
```

Spatial multi-omics technologies measure several molecular layers at the same tissue locations, but turning these
data into biological insight remains hard. Most current methods learn one integrated embedding of all modalities, which
limits interpretability.

Saturn instead learns **one interpretable isodepth per modality**: a one-dimensional coordinate that describes the
dominant spatial variation of that modality, which maps directly back to its input features. Saturn extends the isodepth
concept of GASTON to spatial multi-omics. It assumes that the observed spatial pattern of each modality is the joint effect
of the isodepths of all modalities: gene expression, for example, is modelled from both the RNA isodepth and the ATAC
isodepth. An isodepth describes only the dominant variation of its modality, so the variation contributed by the other
modality is needed to model the full observation.

## Key applications

- **Explore a dataset without prior knowledge:** per-modality isodepths and spatial domains, from any spatial multi-omics platform.
- **Separate own and cross contributions:** how much of each modality follows its own isodepth and how much follows another modality's.
- **Continue with downstream analysis along the isodepth:** TF activity, spatial gradients of cell-cell communication,
  Granger causality, cell-fate transition probabilities and spatial heterogeneity of isoform usage.

## Getting started

The [getting started](getting_started.md) page fits Saturn on a simulation and then walks through real data in five steps.
The [tutorials](tutorials/index.md) run the whole workflow on three real datasets.

## Datasets in the manuscript

| dataset | modalities |
|---|---|
| Simulation benchmark | RNA + ATAC |
| LPC lesion, mouse brain (DBiT ARC-seq) | RNA + ATAC |
| Melanoma (DBiT) | RNA + ATAC |
| Mouse brain (RT&T-AMP-MERFISH) | RNA + isoform usage |
| Human tonsil (Sprox-seq) | RNA + protein, with protein complexes held out |
| Melanoma (spatial Hi-C-RNA) | RNA + Hi-C |

## Installation

Saturn needs Python 3.9 or later. See [Installation](installation.md).

```bash
git clone https://github.com/C0nc/Saturn.git && cd Saturn
pip install -e ".[all]"
```

## Citation

Manuscript in preparation.

## Important resources

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} About Saturn
:link: about
:link-type: doc
The model and what one fit returns.
:::

:::{grid-item-card} Getting started
:link: getting_started
:link-type: doc
Quickstart, real data, and choosing the input of each modality.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
LPC lesion, MERFISH isoforms and Sprox-seq cell-cell contacts.
:::

:::{grid-item-card} API
:link: api/index
:link-type: doc
Every function, by module.
:::
::::

```{toctree}
:hidden:
:maxdepth: 2

about
installation
getting_started
tutorials/index
analysis_pipeline
api/index
reproducibility
references
```
