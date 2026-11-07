#!/usr/bin/env python
"""A/B compartment matrix for Fig 6: scCompartment.hdf5 -> SpHIC117/abcompartment_matrix.csv.

Recovered from SpHIC117/melanoma.ipynb, cell 40 (reads compartment_zscore cell_0..cell_6267 into obsm['X_abcompartment'])
and cell 39 (writes the CSV). The notebook indexed rows by adata.obs_names of melanoma_new_res.h5ad; that index is the
same as HIC/higashi/filelist.txt in order (checked: 6,268 of 6,268), and cell_i is the i-th line of that file, so the
filelist is used here and the undeclared h5ad is not needed. Cell 17 (a bin subset for a clustering) did not feed this
CSV: the published CSV has all 5,359 bins.
Usage: SATURN_ROOT=<workspace> python fig6_ab_matrix.py
"""
import os
import h5py, numpy as np, pandas as pd

ROOT = os.environ["SATURN_ROOT"]; HG = f"{ROOT}/SpHIC117/HIC/higashi"; OUT = f"{ROOT}/SpHIC117/abcompartment_matrix.csv"
assert not os.path.islink(OUT), "would write through a symlink"
bc = [l.strip().split("/")[-1].replace(".contact.tsv", "") for l in open(f"{HG}/filelist.txt") if l.strip()]
with h5py.File(f"{HG}/tmp/scCompartment.hdf5", "r") as f:
    g = f["compartment_zscore"]; X = np.stack([g[f"cell_{i}"][:] for i in range(len(bc))])
pd.DataFrame(X, index=bc, columns=[f"bin_{i}" for i in range(X.shape[1])]).to_csv(OUT)
print(OUT, X.shape)
