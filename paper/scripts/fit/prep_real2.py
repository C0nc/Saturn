"""Build Saturn targets + interpretation annotations for the real-data suite.

Per dataset -> results/real_saturn/targets_<name>.npz with
    coords, coords_z, barcodes, modalities, Y_<mod>, (mask_<mod>), depth_<mod>,
    ann_<column>   per-spot annotations used to interpret the axes
    mk_<gene>      log-normalised marker expression (from RNA counts)
Reductions: RNA -> GLM-PCs; ATAC -> LSI (auto depth-component drop);
methylation residuals -> PCA; Hi-C A/B scores -> compartment PCA;
protein -> CLR + PCA (all proteins); TCR CDR3 -> 3-mer TF-IDF-SVD (masked).
"""
import os, sys, re, json, argparse, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, h5py
import hdf5plugin  # noqa: F401
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Saturn/src")
from saturn.preprocess import (scale_target, rna_glmpc_real, lsi_real, methyl_pca, compartment_pca,
                               protein_clr_pca, tcr_kmer_lsi, zscore_coords)
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


def e11_hic():
    from anndata.experimental import read_elem
    with h5py.File(f"{H}/HIC_repo/EM/E11_pseudotime_d8_d10.h5ad", "r") as f:
        X = read_elem(f["layers"]["counts"]); obs = read_elem(f["obs"]); var = read_elem(f["var"])
        coords = np.asarray(read_elem(f["obsm"]["spatial"]), float)
    ab = pd.read_csv(f"{H}/SpHIC113/X_ab.csv", index_col=0)
    # the CSV carries a trailing `cluster_cellcharter_concat` LABEL column next to the
    # 4,813 compartment bins; leaving it in makes it 89% of the target's PC1, i.e. the
    # Hi-C axis fits a cluster indicator instead of compartment structure
    ab = ab[[c for c in ab.columns if str(c).startswith("bin_")]]
    common = [b for b in obs.index if b in ab.index]
    idx = obs.index.get_indexer(common)
    X = sp.csr_matrix(X)[idx]; coords = coords[idx]; obs = obs.iloc[idx]
    Yr, ir = rna_glmpc_real(X, K=10, n_hvg=3000)
    Yh, ih = compartment_pca(ab.loc[common].values, K=10)
    print(f"   Hi-C bins used {ih['n_bins']}; obs cols {list(obs.columns)[:12]}")
    ex = dict(Y_RNA=Yr, Y_HiC=Yh, depth_RNA=ir["depth"])
    for c in obs.columns:
        if any(t in c.lower() for t in ["pseudotime", "dpt", "leiden", "cellcharter", "domain", "celltype", "cell_type"]):
            v = obs[c]; ex[f"ann_{c}"] = (v.astype(str).values if v.dtype.name in ("category", "object")
                                         else v.values.astype(np.float32))
    ex.update(lognorm(X, ["Sox2", "Nes", "Pax6", "Tubb3", "Neurod2", "Dcx", "Hbb-y", "Col1a1", "Foxg1", "Otx2"],
                      pd.Index(var.index)))
    save("e11_hic", coords, common, ["RNA", "HiC"], ex)


def e11_meth(sel=False):
    """sel=True: neural tube = CellCharter K10 clusters {1,3,4,6,9} (hindbrain,
    pan-neural, spinal cord, midbrain, forebrain; saturn_e11_brain_methylvi.py)."""
    D = f"{H}/compare_vcr/results_meth"
    r = ad.read_h5ad(f"{D}/e11_rna.h5ad"); mth = ad.read_h5ad(f"{D}/e11_meth.h5ad")
    assert (r.obs_names == mth.obs_names).all()
    cc = ad.read_h5ad(f"{H}/GSE270498/e11_cellcharter/e11_cc_portable.h5ad", backed="r").obs["cc_K10"].astype(int)
    cc = cc.reindex(r.obs_names)
    if sel:
        k = cc.isin([1, 3, 4, 6, 9]).values
        r, mth = r[k].copy(), mth[k].copy(); cc = cc[k]; print(f"   neural-tube subset: {k.sum()} spots")
    cdf = pd.read_csv(f"{D}/e11_coords.csv")
    num = cdf.select_dtypes("number")
    coords = num.iloc[:, -2:].values.astype(float) if num.shape[1] >= 2 else r.obsm["spatial"]
    if sel:
        coords = coords[k]
    print(f"   coords from e11_coords.csv columns {list(num.columns[-2:])}; rows {len(cdf)} vs spots {r.n_obs}")
    # coverage = # covered regions (nFeature). nCount_DNAm is the SUM OF RESIDUALS, not coverage.
    dm = mth.obs["nFeature_DNAm"].values.astype(float)
    Yr, ir = rna_glmpc_real(r.X, K=10); Ym, im = methyl_pca(mth.X, K=10, depth=dm, depth_mode="regress")
    print(f"   methyl PCs after coverage regression: |corr| with log coverage {np.round(im['depth_corr'], 2)}")
    ex = dict(Y_RNA=Yr, Y_Methyl=Ym, depth_RNA=ir["depth"], depth_Methyl=dm,
              ann_cc_K10=cc.astype(str).values)
    for c in ["seurat_clusters", "DNAm_snn_res.1", "wsnn_res.1"]:
        if c in r.obs: ex[f"ann_{c}"] = r.obs[c].astype(str).values
    ex.update(lognorm(r.X, ["Foxg1", "Otx1", "Six3", "Emx1", "Hbb-y", "Col1a1", "Sox2", "Tubb3", "Pax6", "Myl7"], r.var_names))
    save("e11_meth_sel" if sel else "e11_meth", coords, r.obs_names, ["RNA", "Methyl"], ex)


def tcr_tcrdist(tra, trb, K=8):
    """TCR-dedicated reduction: TCRdist CDR3-beta distance -> kernel PCA
    (classical MDS, CoNGA's TCR landscape), computed in env `tcr` by
    scripts/tcr_tcrdist_helper.py. One TCR per bead (first beta CDR3).
    Returns (emb for TCR+ beads, boolean TCR+ mask over all beads)."""
    import subprocess, tempfile
    first = lambda v: "" if str(v) in ("nan", "None", "") else str(v).split()[0]
    a_ = np.array([first(v) for v in tra], dtype=object); b_ = np.array([first(v) for v in trb], dtype=object)
    # beta-chain only: alpha is detected on 62% of TCR+ beads and beta on 76%, and with
    # both chains the leading kPCs tracked chain PRESENCE (|rho| 0.36-0.43), not sequence.
    # Beads without a beta CDR3 are unobserved for this modality.
    pos = np.array([bool(y) for y in b_])
    a_ = np.array([""] * len(a_), dtype=object)
    tmp = tempfile.mkdtemp(); fin, fout = f"{tmp}/in.npz", f"{tmp}/out.npz"
    np.savez(fin, tra=a_[pos], trb=b_[pos], K=K)
    r = subprocess.run(["/home/yancui/miniconda3/bin/conda", "run", "-n", "tcr", "python",
                        f"{H}/Hole/gaston_mix_mm/scripts/tcr_tcrdist_helper.py", fin, fout],
                       capture_output=True, text=True)
    print("   " + (r.stdout.strip() or r.stderr[-400:]))
    return np.load(fout)["emb"], pos


def rcc_tcr():
    a = ad.read_h5ad(f"{H}/slide_gotags_rcc/rcc_1_a.h5ad")
    tra = a.obs["TRA_cdr3"].astype(str).values; trb = a.obs["TRB_cdr3"].astype(str).values
    Yr, ir = rna_glmpc_real(a.X, K=10)
    E, mk = tcr_tcrdist(tra, trb, K=8)                    # TCR-dedicated: TCRdist -> kPCA
    Yt = np.full((len(mk), E.shape[1]), np.nan, np.float32); Yt[mk] = scale_target(E, "global")
    print(f"   TCR+ spots {int(mk.sum())}")
    ex = dict(Y_RNA=Yr, Y_TCR=Yt, mask_TCR=mk, depth_RNA=ir["depth"], ann_TRB_cdr3=trb, ann_TCRpos=mk.astype(str))
    ex.update(lognorm(a.X, ["CD3E", "CD8A", "CD4", "PTPRC", "CA9", "VIM", "EPCAM", "CD68", "GZMB", "PDCD1"], a.var_names))
    save("rcc_tcr", np.asarray(a.obsm["spatial"], float), a.obs_names, ["RNA", "TCR"], ex)


def rcc_tcr_bin(bin_um=200.0):
    """Slide-GoTags beads are ~single cells (median 3,099 UMI/bead, 25 um nearest
    neighbour), so neighbouring beads are different cell types and the unbinned run
    found no spatially predictable signal for any method (held-out R2 ~0). Pool
    beads into bin_um bins so cell-type COMPOSITION can form gradients; TCR per bin
    = mean TCRdist kernel-PCA embedding of its TCR+ beads."""
    a = ad.read_h5ad(f"{H}/slide_gotags_rcc/rcc_1_a.h5ad")
    xy = np.asarray(a.obsm["spatial"], float)
    code = np.floor(xy / bin_um).astype(np.int64); code = code[:, 0] * 1_000_000 + code[:, 1]
    uniq, inv, cnt = np.unique(code, return_inverse=True, return_counts=True)
    kb = np.where(cnt >= 3)[0]
    G = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(uniq), len(inv)))[kb]
    X = G @ sp.csr_matrix(a.X); bxy = np.asarray(G @ xy) / np.asarray(G.sum(1))
    tra = a.obs["TRA_cdr3"].astype(str).values; trb = a.obs["TRB_cdr3"].astype(str).values
    btra, btrb = [], []
    Gl = G.tolil()
    for i in range(G.shape[0]):
        members = Gl.rows[i]
        btra.append(" ".join(t for t in tra[members] if t not in ("nan", "")))
        btrb.append(" ".join(t for t in trb[members] if t not in ("nan", "")))
    print(f"   {a.n_obs} beads -> {len(kb)} bins of {bin_um:.0f} um (>=3 beads)")
    Yr, ir = rna_glmpc_real(X, K=10)
    E, pos = tcr_tcrdist(tra, trb, K=8)                   # per bead, TCR-dedicated
    Eb = np.zeros((a.n_obs, E.shape[1]), np.float32); Eb[pos] = E
    cnt_pos = np.asarray(G @ pos.astype(float)).ravel()
    mk = cnt_pos > 0
    Yt = np.full((G.shape[0], E.shape[1]), np.nan, np.float32)
    Yt[mk] = scale_target(np.asarray(G @ Eb)[mk] / cnt_pos[mk, None], "global")
    n_tcr = np.asarray(G @ ((tra != "nan") | (trb != "nan")).astype(float)).ravel()
    print(f"   bins with any TCR+ bead: {int(mk.sum())}")
    ex = dict(Y_RNA=Yr, Y_TCR=Yt, mask_TCR=mk, depth_RNA=ir["depth"],
              ann_n_TCRpos_beads=n_tcr.astype(np.float32), ann_n_beads=np.asarray(G.sum(1)).ravel().astype(np.float32))
    ex.update(lognorm(X, ["CD3E", "CD8A", "CD4", "PTPRC", "CA9", "VIM", "EPCAM", "CD68", "GZMB", "PDCD1"], a.var_names))
    save("rcc_tcr_bin", bxy, [str(u) for u in uniq[kb]], ["RNA", "TCR"], ex)


def xenium(bin_um=100.0):
    D = "/mnt/HDD2A/Users/yancui/xenium_rcc_protein"
    r = ad.read_h5ad(f"{D}/RNA.h5ad"); p = ad.read_h5ad(f"{D}/Protein.h5ad")
    assert (r.obs_names == p.obs_names).all()
    xy = np.asarray(r.obsm["spatial"], float)
    key = np.floor(xy / bin_um).astype(np.int64)
    code = key[:, 0] * 1_000_000 + key[:, 1]
    uniq, inv, cnt = np.unique(code, return_inverse=True, return_counts=True)
    keep_bins = np.where(cnt >= 3)[0]
    G = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(uniq), len(inv)))
    G = G[keep_bins]
    Xr = G @ sp.csr_matrix(r.X); Xp = G @ sp.csr_matrix(p.X)
    bxy = np.asarray(G @ xy) / np.asarray(G.sum(1))
    print(f"   {len(xy)} cells -> {len(keep_bins)} bins of {bin_um:.0f} um (>=3 cells)")
    Yr, ir = rna_glmpc_real(Xr, K=10, n_hvg=396, min_frac=0.0)
    Yp, ipr = protein_clr_pca(Xp, K=10)
    ex = dict(Y_RNA=Yr, Y_Protein=Yp, depth_RNA=ir["depth"], depth_Protein=ipr["depth"],
              ann_n_cells=np.asarray(G.sum(1)).ravel().astype(np.float32))
    ctcol = next((c for c in r.obs.columns if "type" in c.lower() or "tacco" in c.lower()), None)
    if ctcol:
        ct = r.obs[ctcol].astype(str).values
        for t in pd.Series(ct).value_counts().index[:8]:
            ex[f"ann_frac_{t}"] = (np.asarray(G @ (ct == t).astype(float)).ravel() / np.asarray(G.sum(1)).ravel()).astype(np.float32)
        print(f"   cell-type fractions from obs['{ctcol}']")
    ex.update(lognorm(Xr, ["CD3D", "PTPRC", "MS4A1", "CD14", "VIM", "PECAM1", "EPCAM", "KRT8", "ACTA2"], r.var_names))
    # the RNA panel lacks CD3E/CD8A/CD68/CA9 -- the protein panel has them: use CLR protein as markers
    Lp = np.log1p(np.asarray(Xp.todense(), float)); Cp = Lp - Lp.mean(1, keepdims=True)
    for prot in ["CD3E", "CD8A", "CD4", "CD68", "CD20", "PanCK", "Vimentin", "CD31", "CD45", "Ki-67", "PD-L1"]:
        if prot in list(p.var_names):
            ex[f"mk_prot_{prot}"] = Cp[:, list(p.var_names).index(prot)].astype(np.float32)
    save("xenium_rcc", bxy, [str(u) for u in uniq[keep_bins]], ["RNA", "Protein"], ex)


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


def redcat(bin_px=50):
    """REDCAT (Li et al., Nat Methods 2026) public source data: 4 co-registered channel
    images from Fig 3 (SRS lipid 2850 / protein 2930 cm-1; two-photon NADH / FAD). The
    CODEX panel and hyperspectral stack are not public. Images are flat-field corrected
    first (tile-median flat field, 500 px tiles; see external_data/redcat/flat_*.npy):
    raw NADH carried a 500 px stitching grid (FFT strength 7.6 -> 1.0).
    Two modalities per bin_px x bin_px bin, log intensities:
      Raman   = mean/sd log lipid, mean/sd log protein, mean log(lipid/protein)
      AutoFl  = mean/sd log NADH,  mean/sd log FAD,     mean log(NADH/FAD) (redox)
    Holes (> 30% pixels below the 2nd percentile of protein) are dropped."""
    D = f"{H}/external_data/redcat"
    ch = {c: np.log(np.clip(np.load(f"{D}/flat_{c}.npy"), 1, None)) for c in ["lipid", "protein", "NADH", "FAD"]}
    Hh, Ww = ch["protein"].shape; ny, nx = Hh // bin_px, Ww // bin_px
    tile = lambda a: a[:ny * bin_px, :nx * bin_px].reshape(ny, bin_px, nx, bin_px).transpose(0, 2, 1, 3).reshape(ny * nx, -1)
    T = {c: tile(v) for c, v in ch.items()}
    dark = tile((ch["protein"] < np.percentile(ch["protein"], 2)).astype(np.float32)).mean(1)
    keep = dark < 0.3
    f = lambda c: np.c_[T[c].mean(1), T[c].std(1)]
    raman = np.c_[f("lipid"), f("protein"), (T["lipid"] - T["protein"]).mean(1)][keep]
    autof = np.c_[f("NADH"), f("FAD"), (T["NADH"] - T["FAD"]).mean(1)][keep]
    def red(X):                                     # z-score features -> PCA (all comps) -> global scale
        Z = (X - X.mean(0)) / X.std(0)
        U, S, _ = np.linalg.svd(Z, full_matrices=False)
        return scale_target(U * S, "global")
    iy, ix = np.divmod(np.arange(ny * nx), nx)
    xy = np.c_[(ix + 0.5) * bin_px, -(iy + 0.5) * bin_px][keep].astype(float)
    print(f"   {ny * nx} bins of {bin_px} px, kept {keep.sum()} (holes dropped {int((~keep).sum())})")
    ex = dict(Y_Raman=red(raman), Y_AutoFl=red(autof),
              mk_lipid=raman[:, 0], mk_protein=raman[:, 2], mk_lipid_protein=raman[:, 4],
              mk_NADH=autof[:, 0], mk_FAD=autof[:, 2], mk_redox_NADH_FAD=autof[:, 4])
    save("redcat", xy, [f"b{i}" for i in np.where(keep)[0]], ["Raman", "AutoFl"], ex)


def splisoseq_v2(bin_8um=8):
    """Spl-ISO-Seq2 (Michielsen et al., Nat Methods 2026) Visium HD sample V2, ONT long
    reads (Zenodo 10.5281/zenodo.19499423, visiumhd_data_analysis). The bundle's
    V2_8um.h5ad holds metadata only, so BOTH modalities come from the long reads,
    each with its dedicated reduction:
      RNA      gene read counts per bin -> GLM-PC (rna_glmpc_real)
      Isoform  isoform-assigned reads -> within-gene usage, shrunk, CLR -> PCA
               (isoform_usage_pca)
    Reads: Spl-IsoQuant allinfo, already filtered to spliced in-tissue reads by the
    authors (col 2 gene, col 4 8-um barcode s_008um_ROW_COL, col 12 isoform).
    Bins: bin_8um x bin_8um native bins (default 8 -> 64 um)."""
    from saturn.preprocess import isoform_usage_pca
    D = f"{H}/external_data/splisoseq2/visiumhd_data_analysis"
    df = pd.read_csv(f"{D}/V2_ONT/V2_ONT_8um.UMI_filtered.ED4.allinfo.filtered.labeled.gz", sep="\t", header=None,
                     usecols=[1, 3, 11], names=["gene", "bc", "iso"], dtype=str)
    rc = df["bc"].str.extract(r"s_008um_(\d+)_(\d+)").astype(int).values
    bkey = (rc[:, 0] // bin_8um) * 100000 + rc[:, 1] // bin_8um
    bins, binv = np.unique(bkey, return_inverse=True)
    gcode, genes = pd.factorize(df["gene"])
    Xg = sp.csr_matrix((np.ones(len(df)), (binv, gcode)), shape=(len(bins), len(genes)))
    has = df["iso"].notna().values
    icode, isos = pd.factorize(df["iso"][has])
    Xi = sp.csr_matrix((np.ones(has.sum()), (binv[has], icode)), shape=(len(bins), len(isos)))
    iso_gene = df.loc[has].groupby(icode)["gene"].first().reindex(range(len(isos))).values
    ok = (np.asarray(Xg.sum(1)).ravel() >= 50) & (np.asarray(Xi.sum(1)).ravel() >= 20)
    print(f"   {len(df):,} reads ({has.mean():.2f} isoform-assigned) -> {len(bins)} bins of {8 * bin_8um} um; "
          f"kept {ok.sum()} with >=50 gene / >=20 isoform reads")
    Xg, Xi, bins = Xg[ok], Xi[ok], bins[ok]
    Yr, ir = rna_glmpc_real(Xg, K=10)
    dep_i = np.asarray(Xi.sum(1)).ravel()
    Yi, ii = isoform_usage_pca(Xi, iso_gene, K=10, depth=dep_i)
    print(f"   isoform usage: {ii['n_genes']} genes / {ii['n_isoforms']} isoforms usable; top {len(ii['isoforms'])} used")
    xy = np.c_[(bins % 100000 + 0.5) * 8 * bin_8um, -(bins // 100000 + 0.5) * 8 * bin_8um].astype(float)
    ex = dict(Y_RNA=Yr, Y_Isoform=Yi, depth_RNA=ir["depth"], depth_Isoform=dep_i)
    # annotations: majority region + cell-type fractions (authors' labels, per 8-um barcode)
    lab = pd.read_csv(f"{D}/V2_ONT/isomatrix/labels.csv", usecols=["0", "region", "first_type"]).set_index("0")
    L = lab.reindex(df["bc"].values)
    kb = pd.Series(bkey).map({b: i for i, b in enumerate(bins)})
    sel = kb.notna().values
    reg = pd.DataFrame({"b": kb[sel].astype(int).values, "r": L["region"].values[sel]}).dropna()
    ex["ann_region"] = reg.groupby("b")["r"].agg(lambda v: v.value_counts().index[0]).reindex(range(len(bins))).fillna("NA").astype(str).values
    ct = pd.DataFrame({"b": kb[sel].astype(int).values, "t": L["first_type"].values[sel]}).dropna()
    frac = pd.crosstab(ct["b"], ct["t"], normalize="index").reindex(range(len(bins))).fillna(0)
    for t in frac.sum().sort_values(ascending=False).index[:8]:
        ex[f"ann_frac_{t}"] = frac[t].values.astype(np.float32)
    # gene markers (Ensembl IDs, version-stripped) + headline isoform usage
    gid = pd.Index([g.split(".")[0] for g in genes])
    lib = np.asarray(Xg.sum(1)).ravel()
    MK = {"Snap25": "ENSMUSG00000027273", "Mbp": "ENSMUSG00000041607", "Plp1": "ENSMUSG00000031425",
          "Slc17a7": "ENSMUSG00000070570", "Gad1": "ENSMUSG00000070880", "Gfap": "ENSMUSG00000020932",
          "Aqp4": "ENSMUSG00000024411"}
    for nm, e in MK.items():
        if e in gid:
            ex[f"mk_{nm}"] = np.log1p(np.asarray(Xg[:, gid.get_loc(e)].todense()).ravel() / lib * 1e4).astype(np.float32)
    igid = np.array([g.split(".")[0] for g in iso_gene])
    for nm, e in {"Snap25": "ENSMUSG00000027273", "Rps24": "ENSMUSG00000025290"}.items():
        cols = np.where(igid == e)[0]
        if len(cols) >= 2:
            sub = np.asarray(Xi[:, cols].todense()); major = np.argmax(sub.sum(0))
            ex[f"mk_iso_{nm}_major_frac"] = ((sub[:, major] + 1) / (sub.sum(1) + 2)).astype(np.float32)
    # sanity: marker genes should be enriched in their expected cell-type labels
    for nm in ["Mbp", "Slc17a7", "Gfap"]:
        e = MK[nm]; m = df["gene"].str.split(".").str[0] == e
        print(f"   {nm}: top labels of its reads {df.loc[m].merge(lab, left_on='bc', right_index=True)['first_type'].value_counts(normalize=True).head(2).round(2).to_dict()}")
    save("splisoseq_v2", xy, [str(b) for b in bins], ["RNA", "Isoform"], ex)


def melanoma_sel():
    """Curated tumour domain: 1,252 in-tissue spots with tumor_mask (CellCharter K7
    clusters scored by melanoma markers + CNV + mel_frac - immune;
    melanoma_tumor_id.py; refit in melanoma_saturn_refit_v2.py)."""
    R = f"{H}/Hole/gaston_mix_mm/results/melanoma"
    t = pd.read_csv(f"{H}/Saturn/paper_figures/figure2/melanoma_tumor_id/per_spot_tumor_call.csv", index_col=0)
    keep = t.index[t["tumor_mask"].astype(bool)]
    r = ad.read_h5ad(f"{R}/melanoma_rna.h5ad"); a = ad.read_h5ad(f"{R}/atac_snap.h5ad")
    keep = [b for b in keep if b in set(r.obs_names) and b in set(a.obs_names)]
    r, a = r[keep].copy(), a[keep].copy()
    print(f"   tumour domain: {len(keep)} spots")
    Yr, ir = rna_glmpc_real(r.X, K=10); Ya, ia = lsi_real(a.X, K=15)
    print(f"   ATAC depth |corr| {np.round(ia['depth_corr'],2)} dropped {ia['dropped']}")
    ex = dict(Y_RNA=Yr, Y_ATAC=Ya, depth_RNA=ir["depth"], depth_ATAC=ia["depth"])
    tt = t.loc[keep]
    for c in ["cnv_burden_v2", "mel_marker", "immune_marker", "stromal_marker", "mel_frac", "immune_frac"]:
        if c in tt: ex[f"ann_{c}"] = tt[c].values.astype(np.float32)
    if "cc_K7" in tt: ex["ann_cc_K7"] = tt["cc_K7"].astype(str).values
    ex.update(lognorm(r.X, ["MITF", "SOX10", "PMEL", "MLANA", "AXL", "PTPRC", "CD74", "COL1A1", "CD3E", "CD68"], r.var_names))
    save("melanoma_sel", np.asarray(r.obsm["spatial"], float), r.obs_names, ["RNA", "ATAC"], ex)


def xenium_he(side="left", bin_um=50.0):
    """Xenium RCC, RNA + H&E (protein kept only as VALIDATION markers). Cells: the
    curated LEFT/RIGHT tumour-nest masks (presel_masks_v2.npz; CellCharter K8 minus
    clusters 1,3, 30 um radius graph, components >= 8,000 cells). Per bin_um bin:
      RNA  summed counts -> GLM-PC
      H&E  mean Phikon ViT-B/16 CLS feature of its cells (stride-112 overlapping-patch
           mean pool, he_features_meowcat) -> standardise -> PCA (foundation-model
           embeddings are reduced by PCA; no Otsu mask, see xenium memory)
      Protein CLR of summed counts: mk_prot_* validation only, not a modality."""
    D = "/mnt/HDD2A/Users/yancui/xenium_rcc_protein"
    msk = np.load(f"{H}/Saturn/paper_figures/figure2/xenium_rcc_saturn_3m_nests/presel_masks_v2.npz")[f"{side}_mask"]
    r = ad.read_h5ad(f"{D}/RNA.h5ad"); p = ad.read_h5ad(f"{D}/Protein.h5ad")
    hf = np.load(f"{D}/he_features_meowcat/he_cell_features_meowcat.npz", allow_pickle=True)
    assert (r.obs_names == p.obs_names).all() and (hf["obs_names"] == np.asarray(r.obs_names)).all()
    cel = np.where(msk & hf["has_he"])[0]
    xy = np.asarray(r.obsm["spatial"], float)[cel]
    key = np.floor(xy / bin_um).astype(np.int64); code = key[:, 0] * 1_000_000 + key[:, 1]
    uniq, inv, cnt = np.unique(code, return_inverse=True, return_counts=True)
    kb = np.where(cnt >= 3)[0]
    G = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(uniq), len(inv)))[kb]
    n = np.asarray(G.sum(1)).ravel()
    Xr = G @ sp.csr_matrix(r.X[cel]); Xp = G @ sp.csr_matrix(p.X[cel])
    Fh = np.asarray(G @ hf["cell_features"][cel]) / n[:, None]
    bxy = np.asarray(G @ xy) / n[:, None]
    print(f"   {side}: {len(cel)} cells -> {len(kb)} bins of {bin_um:.0f} um (>=3 cells)")
    Yr, ir = rna_glmpc_real(Xr, K=10, n_hvg=396, min_frac=0.0)
    Z = (Fh - Fh.mean(0)) / (Fh.std(0) + 1e-8)
    U, S, _ = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    Yh = scale_target(U[:, :10] * S[:10], "global")
    print(f"   H&E PCA: top-10 variance share {float((S[:10]**2).sum() / (S**2).sum()):.2f}")
    ex = dict(Y_RNA=Yr, Y_HE=Yh, depth_RNA=ir["depth"], ann_n_cells=n.astype(np.float32))
    ct = r.obs["TACCO_celltype"].astype(str).values[cel]
    for t in pd.Series(ct).value_counts().index[:8]:
        ex[f"ann_frac_{t}"] = (np.asarray(G @ (ct == t).astype(float)).ravel() / n).astype(np.float32)
    ex.update(lognorm(Xr, ["CD3D", "PTPRC", "MS4A1", "CD14", "VIM", "PECAM1", "EPCAM", "KRT8", "ACTA2"], r.var_names))
    Lp = np.log1p(np.asarray(Xp.todense(), float)); Cp = Lp - Lp.mean(1, keepdims=True)
    for prot in ["CD3E", "CD8A", "CD4", "CD68", "CD163", "CD20", "HLA-DR", "Ki-67", "PD-L1", "E-Cadherin",
                 "Beta-catenin", "CD45", "GranzymeB", "ATP1A1"]:
        if prot in list(p.var_names):
            ex[f"mk_prot_{prot}"] = Cp[:, list(p.var_names).index(prot)].astype(np.float32)
    save(f"xenium_he_{side}", bxy, [str(u) for u in uniq[kb]], ["RNA", "HE"], ex)


def rcc_tcr_clone(bin_um=200.0, pseudo=0.5):
    """RNA + TCR clonal COMPOSITION per bin. Diagnostic (TCR-beta beads): beads of the
    same clone are spatially closer than random TCR+ pairs (median 1,366 vs 2,079 um
    without the dominant clone, p<0.001), but TCRdist similarity beyond identity has
    no spatial signal (Mantel rho 0.03, p=0.37). So the spatial TCR feature is clone
    identity: per bin, counts of each expanded clone (>= 2 beads) + one 'singleton'
    column -> proportions with pseudocount -> CLR -> PCA. Bins without a TCR-beta
    bead are unobserved (mask)."""
    a = ad.read_h5ad(f"{H}/slide_gotags_rcc/rcc_1_a.h5ad")
    xy = np.asarray(a.obsm["spatial"], float)
    code = np.floor(xy / bin_um).astype(np.int64); code = code[:, 0] * 1_000_000 + code[:, 1]
    uniq, inv, cnt = np.unique(code, return_inverse=True, return_counts=True)
    kb = np.where(cnt >= 3)[0]
    G = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(uniq), len(inv)))[kb]
    X = G @ sp.csr_matrix(a.X); bxy = np.asarray(G @ xy) / np.asarray(G.sum(1))
    trb = np.array([("" if str(v) in ("nan", "") else str(v).split()[0]) for v in a.obs["TRB_cdr3"]], dtype=object)
    pos = trb != ""
    vc = pd.Series(trb[pos]).value_counts(); expanded = list(vc.index[vc >= 2])
    lab = np.where(np.isin(trb, expanded), trb, "singleton")
    cols = expanded + ["singleton"]
    B = np.zeros((a.n_obs, len(cols)), np.float32)
    B[np.where(pos)[0], [cols.index(l) for l in lab[pos]]] = 1
    Cb = np.asarray(G @ B); mk = Cb.sum(1) > 0
    P = (Cb[mk] + pseudo) / (Cb[mk].sum(1, keepdims=True) + pseudo * len(cols))
    L = np.log(P); L = L - L.mean(1, keepdims=True)                 # CLR
    L = L - L.mean(0); U, S_, _ = np.linalg.svd(L, full_matrices=False)
    K = min(8, len(cols) - 1)
    Yt = np.full((len(kb), K), np.nan, np.float32); Yt[mk] = scale_target(U[:, :K] * S_[:K], "global")
    Yr, ir = rna_glmpc_real(X, K=10)
    print(f"   {len(kb)} bins; TCR-beta beads {pos.sum()}, {len(expanded)} expanded clones; bins with TCR {mk.sum()}; "
          f"top-{K} CLR-PC variance share {float((S_[:K]**2).sum() / (S_**2).sum()):.2f}")
    ex = dict(Y_RNA=Yr, Y_TCR=Yt, mask_TCR=mk, depth_RNA=ir["depth"],
              ann_n_TCRpos_beads=Cb.sum(1).astype(np.float32), ann_n_beads=np.asarray(G.sum(1)).ravel().astype(np.float32),
              ann_frac_top_clone=np.where(mk, Cb[:, 0] / np.maximum(Cb.sum(1), 1), np.nan).astype(np.float32))
    ex.update(lognorm(X, ["CD3E", "CD8A", "CD4", "PTPRC", "CA9", "VIM", "EPCAM", "CD68", "GZMB", "PDCD1"], a.var_names))
    save("rcc_tcr_clone", bxy, [str(u) for u in uniq[kb]], ["RNA", "TCR"], ex)


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
            {"lpc10": lpc10, "e11_hic": e11_hic, "e11_meth": e11_meth, "rcc_tcr": rcc_tcr,
             "xenium_rcc": xenium, "melanoma": melanoma_ann, "rcc_tcr_bin": rcc_tcr_bin, "redcat": redcat, "splisoseq_v2": splisoseq_v2,
             "lpc10_sel": lambda: lpc10(sel=True), "e11_meth_sel": lambda: e11_meth(sel=True),
             "melanoma_sel": melanoma_sel, "rcc_tcr_clone": rcc_tcr_clone, "rtt_merfish_s5": lambda: rtt_merfish(5), "xenium_he_left": lambda: xenium_he("left"),
             "xenium_he_right": lambda: xenium_he("right"),
             "lpc_lpc5_cut": lambda: lpc_tp("LPC5"), "lpc_lpc10_cut": lambda: lpc_tp("LPC10"),
             "lpc_lpc21_cut": lambda: lpc_tp("LPC21"),
             "lpc_lpc5_infl": lambda: lpc_tp_infl("LPC5"), "lpc_lpc10_infl": lambda: lpc_tp_infl("LPC10"),
             "lpc_lpc21_infl": lambda: lpc_tp_infl("LPC21")}[n]()
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"   FAILED {n}: {type(e).__name__}: {e}")
