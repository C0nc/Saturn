"""Isoform switching along rho -- the analysis Moran's I cannot do.

The field-standard spatial-isoform test is Moran's I on isoform usage (Lebrigand et al.
Slide-seq; the same tables ship with the Spl-ISO-Seq2 Stereo-seq bundle): it asks whether
usage is spatially autocorrelated, and it is DIRECTION-FREE. Only 3-4% of isoforms pass
(249/9007 in CBS1, 427/9670 in CBS2).

rho is an ORDERED coordinate, so it supports a question Moran's I cannot ask: does the
DOMINANT isoform of a gene change as you move along the axis? For each multi-isoform
gene we take usage per rho domain and flag a SWITCH when the argmax isoform differs
between domains, requiring both isoforms to actually lead by a margin.

Usage: python scripts/saturn_isoform_switch.py (import iso_matrix)
"""
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, "<workspace>/Hole/gaston_mix_mm/scripts")
from prep_real2 import H
from lpc_gaston_fig4l import gaston_domains
from domain_annot import R
KDOM, MINREAD, MARGIN = 4, 50, 0.10
strip = lambda s: str(s).split(".")[0]


def iso_matrix(ds):
    """isoform count matrix on the fit's bins, plus the isoform->gene map"""
    cache = f"{R}/{ds}_isoform_counts.npz"
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])), \
               z["isoforms"].astype(str), z["iso_gene"].astype(str)
    ax = pd.read_csv(f"{R}/{ds}/axes.csv", index_col=0)
    keys = [str(b) for b in ax.index]
    import anndata as ad
    D = f"{H}/external_data/rtt_merfish"
    a = ad.read_h5ad(f"{D}/cellxgene_ISO_labels.h5ad")
    a = a[a.obs["sample"].astype(str) == "5"].copy()
    iso = np.array([v.startswith("ENSMUST") for v in a.var_names])
    t2g = pd.read_csv(f"{D}/ensembl_tx2gene_mm.tsv", sep="\t").drop_duplicates("tx").set_index("tx")["gene"]
    xy = a.obs[["global_x", "global_y"]].values.astype(float)
    code = (np.floor(xy / 50.0).astype(np.int64)[:, 0] * 1_000_000 +
            np.floor(xy / 50.0).astype(np.int64)[:, 1])
    kmap = {int(k): i for i, k in enumerate(keys)}
    row = pd.Series(code).map(kmap); ok = row.notna().values
    G = sp.csr_matrix((np.ones(ok.sum()), (row[ok].astype(int).values, np.where(ok)[0])),
                      shape=(len(keys), a.n_obs))
    X = G @ sp.csr_matrix(a.X)[:, np.where(iso)[0]]
    isos = np.array(a.var_names[iso]); ig = t2g.reindex(isos).values
    m = pd.notna(ig)
    X = sp.csr_matrix(X)[:, np.where(m)[0]]; isos = np.asarray(isos)[m]; ig = np.asarray(ig)[m]
    np.savez_compressed(cache, data=X.data, indices=X.indices, indptr=X.indptr,
                        shape=np.array(X.shape), isoforms=isos.astype(object), iso_gene=ig.astype(object))
    return X, isos, ig


