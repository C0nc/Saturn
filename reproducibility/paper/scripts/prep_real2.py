"""Build Saturn targets + interpretation annotations for the real-data suite.

Per dataset -> results/real_saturn/targets_<name>.npz with
    coords, coords_z, barcodes, modalities, Y_<mod>, (mask_<mod>), depth_<mod>,
    ann_<column>   per-spot annotations used to interpret the axes
    mk_<gene>      log-normalised marker expression (from RNA counts)
Reductions: RNA -> GLM-PCs; ATAC -> LSI (auto depth-component drop);
Hi-C A/B scores -> compartment PCA;
protein -> CLR + PCA (all proteins).
"""
import os, sys, re, json, argparse, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, h5py
import hdf5plugin  # noqa: F401
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/saturn_repo/src")
from saturn.preprocess import (scale_target, rna_glmpc_real, lsi_real, compartment_pca,
                               protein_clr_pca, zscore_coords)
OUT = "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/results/real_saturn"
H = "/mnt/HDD2B/Users/yancui"


def lognorm(X, genes, names):
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    lib = np.asarray(X.sum(1)).ravel().clip(1)
    out = {}
    for g in genes:
        if g in names:
            j = names.get_loc(g)
            out[f"mk_{g}"] = np.log1p(np.asarray(X[:, j].todense()).ravel() / lib * 1e4).astype(np.float32)
    return out


def save(name, coords, barcodes, mods, extra):
    np.savez_compressed(f"{OUT}/targets_{name}.npz", coords=coords, coords_z=zscore_coords(coords),
                        barcodes=np.asarray(barcodes), modalities=np.array(mods), **extra)
    print(f"   saved targets_{name}.npz  ({len(barcodes)} spots, modalities {mods})", flush=True)


def is_counts(X):
    s = X[:500].toarray() if sp.issparse(X) else np.asarray(X[:500])
    return np.allclose(s, np.round(s)) and s.min() >= 0


# --------------------------------------------------------------------------- #
def lpc10(sel=False):
    """sel=True: the curated 2,071-spot cortex + lesion-adjacent intermediate subset
    (obs in_cortex_subset; CellCharter K10 {1,5} + lesion-near 2, largest connected
    component; lpc_saturn_with_intermediate.py, refit in lpc10_saturn_refit_v2.py)."""
    import mudata as md
    m = md.read_h5mu(f"{H}/AD_project/LPC10_multiomic_gdrive/LPC10S1_wholebrain_v2.h5mu")
    r, a = m.mod["rna"], m.mod["atac"]
    if sel:
        k = r.obs["in_cortex_subset"].astype(str).isin(["True", "1", "1.0"]).values
        r, a = r[k].copy(), a[k].copy(); print(f"   curated subset: {k.sum()} spots")
    assert (r.obs_names == a.obs_names).all()
    assert is_counts(r.X) and is_counts(a.X), "LPC10: X is not raw counts"
    Yr, ir = rna_glmpc_real(r.X, K=10); Ya, ia = lsi_real(a.X, K=15)
    print(f"   ATAC depth |corr| {np.round(ia['depth_corr'],2)} dropped {ia['dropped']}")
    ex = dict(Y_RNA=Yr, Y_ATAC=Ya, depth_RNA=ir["depth"], depth_ATAC=ia["depth"])
    for c in ["wb_lesion_v2", "wb_lesion_score_v2", "dist_to_lesion_v2", "dist_to_lesion_cortex", "cc_cluster_cortex"]:
        if c in r.obs:
            v = r.obs[c]; ex[f"ann_{c}"] = (v.astype(str).values if v.dtype.name in ("category", "object", "bool")
                                           else v.values.astype(np.float32))
    ex.update(lognorm(r.X, ["Mbp", "Plp1", "Mog", "Cd68", "Aif1", "Trem2", "Cst7", "Lpl", "Gfap", "Snap25", "Spp1"], r.var_names))
    save("lpc10_sel" if sel else "lpc10", np.asarray(r.obsm["spatial"], float), r.obs_names, ["RNA", "ATAC"], ex)


def melanoma_ann():
    """Melanoma targets exist; add interpretation annotations."""
    z = dict(np.load(f"{OUT}/targets_melanoma.npz", allow_pickle=True))
    f = f"{H}/Saturn/paper_figures/figure2/melanoma_tumor_id/per_spot_tumor_call.csv"
    if os.path.exists(f):
        t = pd.read_csv(f, index_col=0)
        t.index = t.index.astype(str)
        bc = pd.Index(z["barcodes"].astype(str))
        for c in t.columns[:4]:
            v = t[c].reindex(bc)
            z[f"ann_{c}"] = v.astype(str).values if v.dtype == object else v.values.astype(np.float32)
        print(f"   tumour-call columns {list(t.columns[:4])}, matched {int(bc.isin(t.index).sum())}/{len(bc)}")
    r = ad.read_h5ad(f"{H}/Hole/gaston_mix_mm/results/melanoma/melanoma_rna.h5ad")
    r = r[z["barcodes"]]
    z.update(lognorm(r.X, ["MITF", "SOX10", "PMEL", "MLANA", "AXL", "PTPRC", "CD74", "COL1A1", "CD3E", "CD68"], r.var_names))
    np.savez_compressed(f"{OUT}/targets_melanoma.npz", **z); print("   melanoma annotations added")


def rtt_merfish(sample=5, bin_um=50.0):
    """RT&T-AMP-MERFISH (Cohen ... Zhuang, Cell 2026; Zenodo 20682863), ISO set: the
    same segmented cells carry ~23k gene features (ISO_Codebook_1/2, gene symbols) and
    ~10k isoform features (ISO_Codebook_3, ENSMUST ids; mapped to genes with Ensembl
    BioMart -> ensembl_tx2gene_mm.tsv). One section (default sample 5), cells pooled in
    bin_um bins (>= 3 cells). Dedicated reductions:
      RNA      gene counts -> GLM-PC
      Isoform  isoform counts -> within-gene usage, shrunk, CLR -> PCA
    Annotations: majority Allen CCF region id, cell-type (cell_label) fractions."""
    from saturn.preprocess import isoform_usage_pca
    D = f"{H}/external_data/rtt_merfish"
    a = ad.read_h5ad(f"{D}/cellxgene_ISO_labels.h5ad")
    a = a[a.obs["sample"].astype(str) == str(sample)].copy()
    iso = np.array([v.startswith("ENSMUST") for v in a.var_names])
    t2g = pd.read_csv(f"{D}/ensembl_tx2gene_mm.tsv", sep="\t").drop_duplicates("tx").set_index("tx")["gene"]
    ig = t2g.reindex(a.var_names[iso]).values
    ok_i = pd.notna(ig)
    xy = a.obs[["global_x", "global_y"]].values.astype(float)
    key = np.floor(xy / bin_um).astype(np.int64); code = key[:, 0] * 1_000_000 + key[:, 1]
    uniq, inv, cnt = np.unique(code, return_inverse=True, return_counts=True)
    kb = np.where(cnt >= 3)[0]
    G = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(uniq), len(inv)))[kb]
    n = np.asarray(G.sum(1)).ravel()
    X = sp.csr_matrix(a.X)
    Xg = G @ X[:, np.where(~iso)[0]]; Xi = G @ X[:, np.where(iso)[0][ok_i]]
    bxy = np.asarray(G @ xy) / n[:, None]
    print(f"   sample {sample}: {a.n_obs} cells -> {len(kb)} bins of {bin_um:.0f} um; "
          f"gene features {int((~iso).sum())}, isoform features {int(ok_i.sum())}")
    Yr, ir = rna_glmpc_real(Xg, K=10)
    dep_i = np.asarray(Xi.sum(1)).ravel()
    Yi, ii = isoform_usage_pca(Xi, ig[ok_i], K=10, depth=dep_i, min_gene_reads=100)
    print(f"   isoform usage: {ii['n_genes']} genes / {ii['n_isoforms']} isoforms usable; median isoform molecules/bin {np.median(dep_i):.0f}")
    ex = dict(Y_RNA=Yr, Y_Isoform=Yi, depth_RNA=ir["depth"], depth_Isoform=dep_i, ann_n_cells=n.astype(np.float32))
    reg = pd.DataFrame({"b": inv, "r": a.obs["annot"].astype(str).values})
    maj = reg.groupby("b")["r"].agg(lambda v: v.value_counts().index[0])
    ex["ann_ccf_region"] = maj.reindex(kb).fillna("NA").astype(str).values
    ct = a.obs["cell_label"].astype(str).values
    for t in pd.Series(ct).value_counts().index[:8]:
        ex[f"ann_frac_{t}"] = (np.asarray(G @ (ct == t).astype(float)).ravel() / n).astype(np.float32)
    gn = a.var_names[~iso]
    ex.update(lognorm(Xg, ["Snap25", "Mbp", "Plp1", "Slc17a7", "Gad1", "Gfap", "Aqp4", "Rorb", "Fezf2"], gn))
    save(f"rtt_merfish_s{sample}", bxy, [str(u) for u in uniq[kb]], ["RNA", "Isoform"], ex)


LPC_TP = {
    "LPC5": dict(rna="claude_code_tutorial_lpc/results/timepoints/arcseq_LPC5S1_rna_adata.h5ad",
                 atac="claude_code_tutorial_lpc/results/timepoints/arcseq_LPC5S1_atac_adata.h5ad",
                 cc_csv="Saturn/paper_figures/figure2/lpc3_GW/timepoint_comparison/LPC5_cc_labels.csv",
                 cc_coords="Saturn/paper_figures/figure2/lpc3_GW/timepoint_comparison/LPC5_coords.npy",
                 lesion="Saturn/paper_figures/figure2/lpc3_GW/timepoint_comparison/wholebrain_lesion_LPC5.npz",
                 col="cc_K10", ids=[1, 8, 5]),
    "LPC10": dict(rna="claude_code_tutorial_lpc/results/timepoints/arcseq_LPC10S1_rna_adata.h5ad",
                  atac="claude_code_tutorial_lpc/results/timepoints/arcseq_LPC10S1_atac_adata.h5ad",
                  cc_csv="Saturn/paper_figures/figure2/lpc3_GW/timepoint_comparison/LPC10_cc_labels.csv",
                  cc_coords="Saturn/paper_figures/figure2/lpc3_GW/timepoint_comparison/LPC10_coords.npy",
                  lesion="Saturn/paper_figures/figure2/lpc3_GW/timepoint_comparison/wholebrain_lesion_LPC10.npz",
                  col="cc_K10", ids=[1, 5, 2]),
    "LPC21": dict(rna="claude_code_tutorial_lpc/results/rna_LPC21S1_matched.h5ad",
                  atac="claude_code_tutorial_lpc/results/atac_LPC21S1_processed_GW.h5ad",
                  cc_csv="Saturn/paper_figures/figure2/lpc3_GW/cellcharter_scvi/cc_labels_per_K.csv",
                  cc_coords="Saturn/paper_figures/figure2/lpc3_GW/cellcharter_scvi/coords.npy",
                  lesion="Saturn/paper_figures/figure2/lpc3_GW/timepoint_comparison/wholebrain_lesion_LPC21.npz",
                  col="cc_K10", ids=[0, 6, 7]),
}


def _read_h5ad_safe(path):
    """anndata chokes on files with a null uns entry (LPC21: uns/log1p/base).
    Fall back to reading X, obs/var names and obsm/spatial with h5py."""
    try:
        return ad.read_h5ad(path)
    except Exception as e:
        print(f"   read_h5ad failed ({type(e).__name__}); h5py fallback", flush=True)
    with h5py.File(path, "r") as f:
        Xg = f["X"]
        if isinstance(Xg, h5py.Dataset):
            X = sp.csr_matrix(Xg[:])
        else:
            enc = Xg.attrs.get("encoding-type", "csr_matrix")
            shp = tuple(Xg.attrs["shape"])
            M = sp.csc_matrix if enc == "csc_matrix" else sp.csr_matrix
            X = M((Xg["data"][:], Xg["indices"][:], Xg["indptr"][:]), shape=shp).tocsr()
        obs = pd.DataFrame(index=[v.decode() if isinstance(v, bytes) else str(v) for v in f["obs/_index"][:]])
        var = pd.DataFrame(index=[v.decode() if isinstance(v, bytes) else str(v) for v in f["var/_index"][:]])
        A = ad.AnnData(X=X, obs=obs, var=var)
        if "obsm/spatial" in f:
            A.obsm["spatial"] = f["obsm/spatial"][:]
    return A


def lpc_tp(tp="LPC10"):
    """LPC timepoint on the USER-BRUSHED subset (lpc_cut_selection.json, saved from the
    cut tool): kept = CellCharter cortex + intermediate domains, minus brushed spots,
    largest 6-NN connected component -- the rule of lpc_saturn_user_cuts_v2.py.
    RNA -> GLM-PCA, ATAC -> LSI (component 1 dropped)."""
    from scipy.spatial import cKDTree
    from scipy.sparse.csgraph import connected_components
    cfg = LPC_TP[tp]
    CUT = f"{H}/Saturn/paper_figures/figure2/lpc3_GW/timepoint_comparison/lpc_cut_selection.json"
    cuts = np.asarray(json.loads(open(CUT).read()).get(tp, []), int)
    r = _read_h5ad_safe(f"{H}/{cfg['rna']}"); a = _read_h5ad_safe(f"{H}/{cfg['atac']}")
    common = np.intersect1d(r.obs_names, a.obs_names)
    r, a = r[common].copy(), a[common].copy()
    xy = np.asarray(r.obsm["spatial"], float)
    cc_lab = pd.read_csv(f"{H}/{cfg['cc_csv']}", index_col=0)[cfg["col"]].astype(int).values
    cc_xy = np.load(f"{H}/{cfg['cc_coords']}")
    nn = cKDTree(cc_xy).query(xy, k=1)[1]
    lab = cc_lab[nn]
    cut = np.zeros(len(cc_xy), bool); cut[cuts[(cuts >= 0) & (cuts < len(cc_xy))]] = True
    keep = np.isin(lab, cfg["ids"]) & ~cut[nn]
    c = xy[keep]
    knn = cKDTree(c).query(c, k=min(7, len(c)))[1]
    kk = knn.shape[1] - 1
    adj = sp.csr_matrix((np.ones(len(c) * kk, np.int8), (np.repeat(np.arange(len(c)), kk), knn[:, 1:].ravel())),
                        shape=(len(c), len(c)))
    ncc, cl = connected_components(adj + adj.T, directed=False)
    big = np.bincount(cl).argmax()
    idx = np.where(keep)[0][cl == big]
    r, a, xy, lab = r[idx].copy(), a[idx].copy(), xy[idx], lab[idx]
    print(f"   {tp}: brushed {len(cuts)}; kept {len(idx)} spots, domains {sorted(set(lab.tolist()))}")
    Yr, ir = rna_glmpc_real(r.X, K=10); Ya, ia = lsi_real(a.X, K=15)
    print(f"   ATAC LSI depth |corr| {np.round(ia['depth_corr'], 2)} dropped {ia['dropped']}")
    ex = dict(Y_RNA=Yr, Y_ATAC=Ya, depth_RNA=ir["depth"], depth_ATAC=ia["depth"],
              ann_domain=lab.astype(str))
    lz = np.load(f"{H}/{cfg['lesion']}", allow_pickle=True)
    lk = [k for k in lz.files if "mask" in k or "lesion" in k]
    if lk:
        lm = np.asarray(lz[lk[0]]).astype(bool)
        if len(lm) == len(cc_xy):
            les_xy = cc_xy[lm]
            if len(les_xy):
                ex["ann_dist_to_lesion"] = cKDTree(les_xy).query(xy, k=1)[0].astype(np.float32)
                ex["ann_in_lesion"] = lm[nn][idx].astype(str)
                print(f"   lesion spots {int(lm.sum())}; kept inside {int(lm[nn][idx].sum())}")
    ex.update(lognorm(r.X, ["Mbp", "Plp1", "Mog", "Cd68", "Aif1", "Trem2", "Cst7", "Lpl", "Gfap", "Snap25", "Spp1"],
                      r.var_names))
    save(f"lpc_{tp.lower()}_cut", xy, r.obs_names, ["RNA", "ATAC"], ex)


INFL_GENES = {
    "DAM": "Trem2 Tyrobp Apoe Cst7 Lpl Itgax Clec7a Cd9 Cd63 Cd74 Ctsb Ctsd Ctsl Ctsz Spp1 Gpnmb Igf1 Fth1 "
           "Lgals3 Axl Lyz2 Cybb Timp2 Csf1 Ccl6".split(),
    "homeostatic": "P2ry12 P2ry13 Cx3cr1 Tmem119 Selplg Siglech".split(),
    "acute_IFN_NFkB": "Tnf Il1b Il6 Ccl2 Ccl3 Ccl4 Cxcl10 Nfkb1 Stat1 Irf7 Isg15 Ifit1 Ifit3 B2m".split(),
    "complement_astro": "C1qa C1qb C1qc C3 C4b Serpina3n Gfap".split(),
}
GTF = "/mnt/HDD2B/Users/yancui/mouse_atlas/gencode.vM25.primary_chr.annotation.gtf"


def _tss(genes):
    """TSS of each gene from the GENCODE vM25 gene records."""
    want = set(genes); out = {}
    with open(GTF) as f:
        for line in f:
            if line[0] == "#":
                continue
            p = line.split("\t")
            if p[2] != "gene":
                continue
            m = re.search(r'gene_name "([^"]+)"', p[8])
            if m and m.group(1) in want:
                out[m.group(1)] = (p[0], int(p[3]) if p[6] == "+" else int(p[4]))
    return out


def lpc_tp_infl(tp="LPC10", win=100_000, prom=2_000, K_rna=5, K_atac=10):
    """LPC timepoint, INFLAMMATION-TARGETED modalities on the brushed subset.

    RNA  = GLM-PCA over an inflammatory gene set only (DAM / homeostatic-microglia /
           acute IFN-NF-kB / complement-astro; Keren-Shaul DAM signature).
    ATAC = LSI over the candidate ENHANCERS of those genes: peaks within +-win of a
           TSS, EXCLUDING promoter-proximal peaks (+-prom), component 1 dropped.
           Innate-immune memory in microglia is carried by permissive marks at
           ENHANCERS while the transcriptome returns to baseline (Neher/Cunningham,
           J Neuroinflammation 2022), so distal peaks are the right readout and
           excluding promoters keeps the ATAC axis from re-reading transcription.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse.csgraph import connected_components
    cfg = LPC_TP[tp]
    CUT = f"{H}/Saturn/paper_figures/figure2/lpc3_GW/timepoint_comparison/lpc_cut_selection.json"
    cuts = np.asarray(json.loads(open(CUT).read()).get(tp, []), int)
    r = _read_h5ad_safe(f"{H}/{cfg['rna']}"); a = _read_h5ad_safe(f"{H}/{cfg['atac']}")
    common = np.intersect1d(r.obs_names, a.obs_names); r, a = r[common].copy(), a[common].copy()
    xy = np.asarray(r.obsm["spatial"], float)
    cc_lab = pd.read_csv(f"{H}/{cfg['cc_csv']}", index_col=0)[cfg["col"]].astype(int).values
    cc_xy = np.load(f"{H}/{cfg['cc_coords']}"); nn = cKDTree(cc_xy).query(xy, k=1)[1]
    cut = np.zeros(len(cc_xy), bool); cut[cuts[(cuts >= 0) & (cuts < len(cc_xy))]] = True
    keep = np.isin(cc_lab[nn], cfg["ids"]) & ~cut[nn]
    c = xy[keep]; knn = cKDTree(c).query(c, k=min(7, len(c)))[1]; kk = knn.shape[1] - 1
    adj = sp.csr_matrix((np.ones(len(c) * kk, np.int8), (np.repeat(np.arange(len(c)), kk), knn[:, 1:].ravel())),
                        shape=(len(c), len(c)))
    _, cl = connected_components(adj + adj.T, directed=False)
    idx = np.where(keep)[0][cl == np.bincount(cl).argmax()]
    r, a, xy, lab = r[idx].copy(), a[idx].copy(), xy[idx], cc_lab[nn][idx]

    genes = [g for grp in INFL_GENES.values() for g in grp]
    present = [g for g in genes if g in set(map(str, r.var_names))]
    Xg = sp.csr_matrix(r.X)[:, [list(r.var_names).index(g) for g in present]]
    Yr, ir = rna_glmpc_real(Xg, K=K_rna, n_hvg=len(present), min_frac=0.0)

    tss = _tss(genes)
    pk = pd.Series(a.var_names).str.extract(r"(chr[\w]+)[:\-_](\d+)[\-_](\d+)")
    pk.columns = ["chr", "s", "e"]; pk["s"] = pk["s"].astype(float); pk["e"] = pk["e"].astype(float)
    mid = (pk["s"] + pk["e"]) / 2
    sel = np.zeros(len(pk), bool); prom_mask = np.zeros(len(pk), bool)
    for g, (ch, t) in tss.items():
        m = (pk["chr"] == ch).values
        d = np.abs(mid.values - t)
        sel |= m & (d <= win)
        prom_mask |= m & (d <= prom)
    enh = sel & ~prom_mask
    Xa = sp.csr_matrix(a.X)[:, np.where(enh)[0]]
    K_atac = int(min(K_atac, max(2, enh.sum() // 50)))
    Ya, ia = lsi_real(Xa, K=K_atac, min_frac=0.0, drop_first=True)
    print(f"   {tp}: {len(idx)} spots | RNA genes {len(present)}/{len(genes)} | "
          f"peaks in +-{win // 1000}kb {int(sel.sum())}, enhancer-only {int(enh.sum())} "
          f"(promoter {int(prom_mask.sum())}) | LSI K={K_atac} depth |corr| {np.round(ia['depth_corr'], 2)}")
    ex = dict(Y_RNA=Yr, Y_ATAC=Ya, depth_RNA=np.asarray(Xg.sum(1)).ravel(), depth_ATAC=ia["depth"],
              ann_domain=lab.astype(str))
    lz = np.load(f"{H}/{cfg['lesion']}", allow_pickle=True)
    lk = [k for k in lz.files if "mask" in k or "lesion" in k]
    if lk:
        lm = np.asarray(lz[lk[0]]).astype(bool)
        if len(lm) == len(cc_xy) and lm.any():
            ex["ann_dist_to_lesion_old"] = cKDTree(cc_xy[lm]).query(xy, k=1)[0].astype(np.float32)
    re_ = f"{H}/Hole/gaston_mix_mm/results/real_saturn/lpc_lesion_reannot/lesion_reannot_{tp}.npz"
    if os.path.exists(re_):
        z2 = np.load(re_, allow_pickle=True)
        core = z2["core"].astype(bool)
        if core.any():
            ex["ann_dist_to_core"] = cKDTree(z2["coords"][core]).query(xy, k=1)[0].astype(np.float32)
    ex.update(lognorm(sp.csr_matrix(r.X), ["Mbp", "Plp1", "Cd68", "Trem2", "Cst7", "Lpl", "Gfap", "C4b", "Apoe",
                                           "P2ry12", "Cx3cr1"], r.var_names))
    save(f"lpc_{tp.lower()}_infl", xy, r.obs_names, ["RNA", "ATAC"], ex)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("names", nargs="+"); a = ap.parse_args()
    for n in a.names:
        print(f"== {n}", flush=True)
        try:
            {"lpc10": lpc10, "melanoma": melanoma_ann, "rtt_merfish_s5": lambda: rtt_merfish(5), "lpc_lpc5_cut": lambda: lpc_tp("LPC5"), "lpc_lpc10_cut": lambda: lpc_tp("LPC10"),
             "lpc_lpc21_cut": lambda: lpc_tp("LPC21"),
             "lpc_lpc5_infl": lambda: lpc_tp_infl("LPC5"), "lpc_lpc10_infl": lambda: lpc_tp_infl("LPC10"),
             "lpc_lpc21_infl": lambda: lpc_tp_infl("LPC21")}[n]()
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"   FAILED {n}: {type(e).__name__}: {e}")
