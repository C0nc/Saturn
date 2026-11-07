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

## Data

Each tutorial reads every input from one workspace folder, `saturn_workspace/` next to the notebook (or the folder
named by `SATURN_WORKSPACE`). Raw files sit at the top of the folder; the published results that the tutorial compares
against sit in `saturn_workspace/published/`. For example, the LPC tutorial reads

```text
saturn_workspace/
├── arcseq_LPC10S1_rna_adata.h5ad
├── arcseq_LPC10S1_atac_adata.h5ad
├── GSM9249009_07_LPC10S1_atac_fragments.tsv.gz
├── gencode.vM25.primary_chr.annotation.gtf
├── mm10.fa, mm10.fa.fai
├── LPC10_cc_labels.csv, LPC10_coords.npy
└── published/
    ├── targets_lpc10_fov.npz
    ├── lpc10_fov/
    └── ...
```

[`workspace_files.json`](https://github.com/C0nc/Saturn/blob/main/docs/tutorials/workspace_files.json) lists the
files of each tutorial and where each comes from (GEO accession, reference genome, or the figure notebook that made it).

To rerun a tutorial in a working directory that holds `saturn_workspace/`:

```bash
PYTHON=python bash docs/tutorials/run_tutorial.sh tutorial_lpc_rna_atac /path/to/workdir
```

`docs/tutorials/_src/` holds the source of each notebook.
