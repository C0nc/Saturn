# Tutorials

Each tutorial runs the Saturn workflow on one real dataset, with every step as a `saturn.*` call. The notebooks are stored with
their outputs, and each one compares its fit with the published axes in a visible cell.

::::{grid} 1 1 1 1
:gutter: 2

:::{grid-item-card} LPC lesion: RNA + ATAC
:link: tutorial_lpc_rna_atac
:link-type: doc
DBiT ARC-seq, 3,425 spots. Spot selection, targets, fit, domains, annotation, feature functions, own vs cross axis, TF activity, genes vs cis peaks, cross-term decomposition.
:::

:::{grid-item-card} Mouse brain MERFISH: RNA + isoform usage
:link: tutorial_merfish_isoform
:link-type: doc
RT&T-AMP-MERFISH, 2,721 bins of 100 µm. Isoform usage target, binomial fits along the axis, isoform switches between domains, one cell type across a boundary.
:::

:::{grid-item-card} Human tonsil Sprox-seq: RNA + protein
:link: tutorial_sproxseq_rna_protein
:link-type: doc
Section B1, 3,204 spots. Held-out protein complexes tested against a rotation null, and cell-cell contact complexes.
:::
::::

```{toctree}
:hidden:

tutorial_lpc_rna_atac
tutorial_merfish_isoform
tutorial_sproxseq_rna_protein
```

To rerun a tutorial:

```bash
SATURN_DATA=/path/to/data PYTHON=python bash docs/tutorials/run_tutorial.sh tutorial_lpc_rna_atac /path/to/workdir
```

`docs/tutorials/_src/` holds the source of each notebook.
