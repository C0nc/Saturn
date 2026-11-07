# Installation

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
