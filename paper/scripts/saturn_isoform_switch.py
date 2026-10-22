"""Isoform switching along rho -- the analysis Moran's I cannot do.

The field-standard spatial-isoform test is Moran's I on isoform usage (Lebrigand et al.
Slide-seq; the same tables ship with the Spl-ISO-Seq2 Stereo-seq bundle): it asks whether
usage is spatially autocorrelated, and it is DIRECTION-FREE. Only 3-4% of isoforms pass
(249/9007 in CBS1, 427/9670 in CBS2).

rho is an ORDERED coordinate, so it supports a question Moran's I cannot ask: does the
DOMINANT isoform of a gene change as you move along the axis? For each multi-isoform
gene we take usage per rho domain and flag a SWITCH when the argmax isoform differs
between domains, requiring both isoforms to actually lead by a margin.

Usage: python scripts/saturn_isoform_switch.py <splisoseq_v2|rtt_merfish_s5>
"""
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
from prep_real2 import H
from lpc_gaston_fig4l import gaston_domains
from domain_annot import R
KDOM, MINREAD, MARGIN = 4, 50, 0.10
LEB = f"{H}/external_data/splisoseq2/leb/Lebrigand_data"
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
    if ds == "splisoseq_v2":
        D = f"{H}/external_data/splisoseq2/visiumhd_data_analysis"
        df = pd.read_csv(f"{D}/V2_ONT/V2_ONT_8um.UMI_filtered.ED4.allinfo.filtered.labeled.gz",
                         sep="\t", header=None, usecols=[1, 3, 11], names=["gene", "bc", "iso"], dtype=str)
        rc = df["bc"].str.extract(r"s_008um_(\d+)_(\d+)").astype(int).values
        bkey = (rc[:, 0] // 8) * 100000 + rc[:, 1] // 8
        has = df["iso"].notna().values
        df = df[has]; bkey = bkey[has]
        kmap = {int(k): i for i, k in enumerate(keys)}
        row = pd.Series(bkey).map(kmap)
        ok = row.notna().values
        icode, isos = pd.factorize(df["iso"][ok])
        X = sp.csr_matrix((np.ones(ok.sum()), (row[ok].astype(int).values, icode)),
                          shape=(len(keys), len(isos)))
        ig = df.loc[ok].groupby(icode)["gene"].first().reindex(range(len(isos))).values
    else:
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


def run(ds):
    OUT = f"{R}/{ds}_isoform_switch"; os.makedirs(OUT, exist_ok=True)
    ax = pd.read_csv(f"{R}/{ds}/axes.csv", index_col=0)
    z = dict(np.load(f"{R}/targets_{ds}.npz", allow_pickle=True))
    X, isos, ig = iso_matrix(ds)
    print(f"   isoform matrix {X.shape}; {len(np.unique(ig))} genes", flush=True)
    q1 = pd.read_csv(f"{LEB}/CBS1/isoform_matrix/res/MoransI_qval_nperm100000_k50_mincells250_mincellspergroup20_imb0.05_seed0.csv", index_col=0)["All"]
    q2 = pd.read_csv(f"{LEB}/CBS2/isoform_matrix/res/MoransI_qval_nperm100000_k50_mincells250_mincellspergroup20_imb0.05_seed0.csv", index_col=0)["All"]
    sig = {strip(i) for i in q1[q1 < 0.05].index} | {strip(i) for i in q2[q2 < 0.05].index}
    rows = []
    for axis in ["RNA", "Isoform"]:
        d = ax[f"d_{axis}"].values; lab = gaston_domains(d, z[f"Y_{axis}"], L_=KDOM)
        U = np.vstack([np.asarray(X[lab == k].sum(0)).ravel() for k in range(KDOM)])   # dom x iso
        for g in np.unique(ig):
            j = np.where(ig == g)[0]
            if len(j) < 2: continue
            sub = U[:, j]
            tot = sub.sum(1)
            if tot.min() < MINREAD: continue
            frac = sub / tot[:, None]
            top = frac.argmax(1)
            lead = np.sort(frac, 1)[:, -1] - np.sort(frac, 1)[:, -2]
            good = lead >= MARGIN
            if good.sum() < 2 or len(np.unique(top[good])) < 2: continue
            k1, k2 = np.where(good)[0][[0, -1]]
            if top[k1] == top[k2]: continue
            rows.append(dict(dataset=ds, axis=axis, gene=g, n_iso=len(j),
                             dom_from=int(k1), dom_to=int(k2),
                             iso_from=isos[j][top[k1]], iso_to=isos[j][top[k2]],
                             frac_from=round(float(frac[k1, top[k1]]), 3),
                             frac_to=round(float(frac[k2, top[k2]]), 3),
                             swing=round(float(abs(frac[k2, top[k2]] - frac[k1, top[k2]])), 3),
                             reads=int(tot.sum()),
                             slideseq_spatial=int(strip(isos[j][top[k1]]) in sig or strip(isos[j][top[k2]]) in sig)))
        print(f"   {ds} d_{axis}: {sum(r['axis']==axis for r in rows)} genes switch dominant isoform along rho",
              flush=True)
    T = pd.DataFrame(rows).sort_values("swing", ascending=False)
    T.to_csv(f"{OUT}/isoform_switches.csv", index=False)
    print(f"-> {OUT}/isoform_switches.csv ({len(T)})")
    return T


if __name__ == "__main__":
    run(sys.argv[1])
