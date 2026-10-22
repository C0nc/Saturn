"""Domain annotation + marker screening for datasets whose second modality is CONTINUOUS.

RNA keeps GASTON's Poisson piecewise fit; methylation residuals and Hi-C A/B
compartment scores are signed continuous values, so they use the Gaussian fitter in
piecewise.py -- same segmentation, same LRT/F gate at t = 0.1, different likelihood.

Per dataset it writes:
  <ds>_domains/domain_annotation.csv   composition + markers + gradient counts
  <ds>_domains/domain_maps.*           both segmentations in space
  <ds>_marker_functions/*_screen.csv   every screened feature, slopes, R2
  <ds>_marker_functions/*_top8_functions.*   8 rows x 5 panels per modality

Usage: python scripts/continuous_marker_panels.py <e11_meth_mpca|e11_hic> [n_rows]
"""
import os, re, sys, json
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/saturn_repo/src")
from prep_real2 import H
from lpc_gaston_fig4l import gaston_domains, boundaries, panel_space, L
from lpc_gene_isodepth import gaston_bins
from domain_annot import R, markers, gradient_count, all_tss
from piecewise import fit_gauss
from marker_function_panels import draw as draw_poisson
from saturn.piecewise import fit_poisson
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "svg.fonttype": "none"})

KDOM, T_LRT, NB = 4, 0.1, 25
NTOP_SCREEN = 4000          # continuous features screened, ranked by variance
GTF_MM = f"{H}/mouse_atlas/gencode.vM25.primary_chr.annotation.gtf"
GTF_HS = f"{H}/gencode.v21.annotation.gtf"


def meth_features(bc):
    m = ad.read_h5ad(f"{H}/compare_vcr/results_meth/e11_meth.h5ad")[pd.Index(bc)].copy()
    V = np.asarray(m.X.toarray() if sp.issparse(m.X) else m.X, float)
    names = np.array([str(v) for v in m.var_names])
    coord = pd.Series(names).str.extract(r"(chr[\w]+)[.\-:](\d+)[.\-](\d+)")
    return V, names, coord


def hic_features(bc):
    ab = pd.read_csv(f"{H}/SpHIC113/X_ab.csv", index_col=0)
    ab = ab[[c for c in ab.columns if str(c).startswith("bin_")]].loc[pd.Index(bc)]
    with h5py.File(f"{H}/SpHIC113/HIC/higashi/tmp/scCompartment.hdf5", "r") as f:
        b = f["compartment"]["bin"]
        ch = b["chrom"][:].astype(str); st = b["start"][:]; en = b["end"][:]
    n = min(ab.shape[1], len(ch))
    V = ab.values[:, :n].astype(float)
    names = np.array([f"{ch[i]}:{st[i]}-{en[i]}" for i in range(n)])
    coord = pd.DataFrame({0: ch[:n], 1: st[:n].astype(str), 2: en[:n].astype(str)})
    return V, names, coord


def hic_features_mel(bc):
    """melanoma A/B, read exactly as granger_ab_gene.ipynb does: the CSV is reindexed
    onto the adata barcode order (rows), and bins line up POSITIONALLY with
    scCompartment.hdf5['compartment_zscore']['bin'] (columns)."""
    ab = pd.read_csv(f"{H}/SpHIC117/abcompartment_matrix.csv", index_col=0)
    ab = ab[[c for c in ab.columns if str(c).startswith("bin_")]].reindex(pd.Index(bc))
    assert not ab.isna().any().any(), "spots missing from the melanoma A/B matrix"
    with h5py.File(f"{H}/SpHIC117/HIC/higashi/tmp/scCompartment.hdf5", "r") as f:
        b = f["compartment_zscore"]["bin"]
        ch = b["chrom"][:].astype(str); st = b["start"][:]; en = b["end"][:]
    assert ab.shape[1] == len(ch), f"bin mismatch {ab.shape[1]} vs {len(ch)}"
    names = np.array([f"{ch[i]}:{st[i]}-{en[i]}" for i in range(len(ch))])
    coord = pd.DataFrame({0: ch, 1: st.astype(str), 2: en.astype(str)})
    return ab.values.astype(float), names, coord


def mel_rna(bc):
    """the FULL 10x matrix the fit used, not the h5ad's 2,000-HVG subset"""
    import gzip
    from scipy.io import mmread
    D = f"{H}/SpHIC117/RNA/raw"
    bcs = [l.strip() for l in gzip.open(f"{D}/barcodes.tsv.gz", "rt")]
    feat = pd.read_csv(f"{D}/features.tsv.gz", sep="\t", header=None)
    X = sp.csr_matrix(mmread(f"{D}/matrix.mtx.gz").T)
    i = pd.Index(bcs).get_indexer(pd.Index(bc))
    assert (i >= 0).all()
    return X[i], np.array([str(v) for v in feat[1]])


def xenium_features(bc, bin_um=100.0):
    """protein CLR per 100-um bin, rebuilt the way prep_real2.xenium() bins the cells"""
    D = "/mnt/HDD2A/Users/yancui/xenium_rcc_protein"
    pr = ad.read_h5ad(f"{D}/Protein.h5ad")
    xy = np.asarray(pr.obsm["spatial"], float)
    key = np.floor(xy / bin_um).astype(np.int64)
    code = key[:, 0] * 1_000_000 + key[:, 1]
    uniq, inv = np.unique(code, return_inverse=True)
    G = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(uniq), len(inv)))
    Xp = np.asarray((G @ sp.csr_matrix(pr.X)).todense(), float)
    Lp = np.log1p(Xp); Cp = Lp - Lp.mean(1, keepdims=True)          # CLR, as in the target
    pos = pd.Index([str(u) for u in uniq]).get_indexer(pd.Index(bc))
    return Cp[pos], np.array([str(v) for v in pr.var_names]), None


def xenium_rna(bc, bin_um=100.0):
    D = "/mnt/HDD2A/Users/yancui/xenium_rcc_protein"
    r = ad.read_h5ad(f"{D}/RNA.h5ad")
    xy = np.asarray(r.obsm["spatial"], float)
    key = np.floor(xy / bin_um).astype(np.int64)
    code = key[:, 0] * 1_000_000 + key[:, 1]
    uniq, inv = np.unique(code, return_inverse=True)
    G = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(uniq), len(inv)))
    X = sp.csr_matrix(G @ sp.csr_matrix(r.X))
    pos = pd.Index([str(u) for u in uniq]).get_indexer(pd.Index(bc))
    return X[pos], np.array([str(v) for v in r.var_names])


def sprox_features(bc, sec="A1", tag=""):
    """protein CLR per spot, from the PLA marginals stored beside the target"""
    P = dict(np.load(f"{R}/sproxseq_{sec}{tag}_pla.npz", allow_pickle=True))
    M = P["protein_marginal"].astype(float)
    order = pd.Index([str(b) for b in P["barcodes"]]).get_indexer(pd.Index(bc))
    L = np.log1p(M[order]); C = L - L.mean(1, keepdims=True)
    return C, np.array([str(x) for x in P["proteins"]]), None


def sprox_rna(bc, sec="A1"):
    import gzip
    from scipy.io import mmread
    D = f"{H}/external_data/sproxseq"
    g = {"A1": "GSM9156293", "B1": "GSM9156294", "D1": "GSM9156297"}[sec]
    bcs = [l.strip() for l in gzip.open(f"{D}/{g}_{sec}_barcodes.tsv.gz", "rt")]
    feat = pd.read_csv(f"{D}/{g}_{sec}_features.tsv.gz", sep="\t", header=None)
    X = sp.csr_matrix(mmread(f"{D}/{g}_{sec}_matrix.mtx.gz").T)
    idx = pd.Index(bcs).get_indexer(pd.Index(bc))
    return X[idx], np.array([str(v) for v in feat[1]])


CFG = {
    "e11_meth_mpca": dict(mod="Methyl", feats=meth_features, gtf=GTF_MM,
                          rna=lambda: ad.read_h5ad(f"{H}/compare_vcr/results_meth/e11_rna.h5ad"),
                          cats=["ann_cc_K10", "ann_wsnn_res.1"], conts=["depth_Methyl"],
                          unit="methylation residual"),
    "e11_meth_sel_mpca": dict(mod="Methyl", feats=meth_features, gtf=GTF_MM,
                              rna=lambda: ad.read_h5ad(f"{H}/compare_vcr/results_meth/e11_rna.h5ad"),
                              cats=["ann_cc_K10", "ann_wsnn_res.1"], conts=["depth_Methyl"],
                              unit="methylation residual"),
    # the fit used Fast-Higashi X_fh_pca as the target; the FEATURES screened here are
    # still the A/B compartment bins, so the markers stay genomically interpretable
    "e11_hic_fhpca": dict(mod="HiC", feats=hic_features, gtf=GTF_MM, rna=None,
                          cats=["ann_cluster_cellcharter_concat_11"], conts=[],
                          unit="A/B compartment score"),
    "xenium_rcc": dict(mod="Protein", feats=xenium_features, gtf=None, rna=None,
                       cats=[], conts=["ann_n_cells", "ann_frac_RCC", "ann_frac_T-cell",
                                       "ann_frac_Myeloid", "ann_frac_Fibro", "ann_frac_EC"],
                       unit="protein CLR"),
    "sproxseq_A1_p46": dict(mod="Protein", feats=lambda bc: sprox_features(bc, "A1", "_p46"),
                            gtf=None, rna=None, cats=[],
                            conts=["ann_pla_total", "depth_RNA"], unit="protein CLR"),
    "sproxseq_B1_p46": dict(mod="Protein", feats=lambda bc: sprox_features(bc, "B1", "_p46"),
                            gtf=None, rna=None, cats=[],
                            conts=["ann_pla_total", "depth_RNA"], unit="protein CLR"),
    "sproxseq_D1_p46": dict(mod="Protein", feats=lambda bc: sprox_features(bc, "D1", "_p46"),
                            gtf=None, rna=None, cats=[],
                            conts=["ann_pla_total", "depth_RNA"], unit="protein CLR"),
    "sproxseq_A1": dict(mod="Protein", feats=sprox_features, gtf=None, rna=None,
                        cats=[], conts=["ann_pla_total", "depth_RNA"], unit="protein CLR"),
    "mel_hic_fh": dict(mod="HiC", feats=hic_features_mel, gtf=GTF_HS, rna=None,
                       cats=["ann_cluster_cellcharter_rna_best", "ann_cluster_cellcharter_hic_best",
                             "ann_max_prop_cell_type"],
                       conts=["depth_RNA", "ann_confidence"], unit="A/B compartment z"),
    "e11_hic": dict(mod="HiC", feats=hic_features, gtf=GTF_MM,
                    rna=None, cats=["ann_cluster_cellcharter_concat_11"], conts=[],
                    unit="A/B compartment score"),
}


def rna_matrix(dataset, bc):
    """RNA counts for the dataset's spots (the Hi-C object stores them in layers)"""
    if dataset == "xenium_rcc":
        return xenium_rna(bc)
    if dataset.startswith("sproxseq_"):
        return sprox_rna(bc, dataset.split("_")[1])
    if dataset.startswith("mel_hic"):
        return mel_rna(bc)
    if dataset.startswith("e11_hic"):
        from anndata.experimental import read_elem
        with h5py.File(f"{H}/HIC_repo/EM/E11_pseudotime_d8_d10.h5ad", "r") as f:
            X = read_elem(f["layers"]["counts"]); obs = read_elem(f["obs"]); var = read_elem(f["var"])
        i = obs.index.get_indexer(pd.Index(bc))
        return sp.csr_matrix(X)[i], np.array([str(v) for v in var.index])
    r = CFG[dataset]["rna"]()[pd.Index(bc)].copy()
    return sp.csr_matrix(r.X), np.array([str(v) for v in r.var_names])


def nearest(coord, gtf):
    tch, tpo, tnm = all_tss(gtf)
    ch = coord[0].values.astype(str)
    mid = (coord[1].astype(float).values + coord[2].astype(float).values) / 2
    g = np.full(len(ch), "NA", dtype=object); dd = np.full(len(ch), np.nan)
    for c_ in pd.unique(ch):
        m = tch == c_; sel = np.where(ch == c_)[0]
        if not m.any() or not len(sel):
            continue
        tp_, tn_ = tpo[m], tnm[m]
        o = np.argsort(tp_); tp_, tn_ = tp_[o], tn_[o]
        j = np.clip(np.searchsorted(tp_, mid[sel]), 1, len(tp_) - 1)
        pick = np.where(np.abs(mid[sel] - tp_[j - 1]) <= np.abs(mid[sel] - tp_[j]), j - 1, j)
        g[sel] = tn_[pick]; dd[sel] = mid[sel] - tp_[pick]
    return g, dd


def cont_markers(V, lab, names, k, n=8, min_var=0.0):
    """one-vs-rest markers of a continuous modality: Welch t (saturn.annotate.continuous_markers)"""
    from saturn.annotate import continuous_markers
    return continuous_markers(V, lab, names, k, n=n, min_var=min_var)


def binned_r2_cont(y, d, lab, slope, inter, nb=NB):
    i, _, keep = gaston_bins(d, nb)
    ob = np.array([y[i == b].mean() for b in keep])
    db = np.array([d[i == b].mean() for b in keep])
    lb = np.array([np.bincount(lab[i == b], minlength=lab.max() + 1).argmax() for b in keep])
    pred = inter[lb] + slope[lb] * db
    return float(1 - ((ob - pred) ** 2).sum() / max(((ob - ob.mean()) ** 2).sum(), 1e-12)), db, ob, pred, lb


def run(dataset, n_rows=8, kdom=KDOM):
    cfg = CFG[dataset]; mod = cfg["mod"]
    OD = f"{R}/{dataset}_domains"; OM = f"{R}/{dataset}_marker_functions"
    os.makedirs(OD, exist_ok=True); os.makedirs(OM, exist_ok=True)
    ax = pd.read_csv(f"{R}/{dataset}/axes.csv", index_col=0)
    z = dict(np.load(f"{R}/targets_{dataset}.npz", allow_pickle=True))
    bc = [str(b) for b in ax.index]
    dR, dM = ax["d_RNA"].values, ax[f"d_{mod}"].values
    xy = np.c_[ax.x.values, ax.y.values]
    labR = gaston_domains(dR, z["Y_RNA"], L_=kdom)
    labM = gaston_domains(dM, z[f"Y_{mod}"], L_=kdom)
    bR, bM = boundaries(dR, labR), boundaries(dM, labM)
    print(f"{dataset}: RNA domains {np.bincount(labR).tolist()} | {mod} domains {np.bincount(labM).tolist()}",
          flush=True)

    X, vn = rna_matrix(dataset, bc)
    lib = np.asarray(X.sum(1)).ravel().clip(1)
    V, fn, coord = cfg["feats"](bc)
    if coord is None:
        near_g = np.array(fn, dtype=object); near_d = np.full(len(fn), np.nan)
    else:
        near_g, near_d = nearest(coord, cfg["gtf"])

    # ---- screens -------------------------------------------------------------
    gR = gradient_count(X, lib, labR, dR)
    tot = np.asarray(X.sum(0)).ravel().astype(float)
    cand = np.argsort(-tot)[:1500]; cand = cand[tot[cand] >= 100]
    C = np.asarray(X[:, cand].todense())
    fitG = fit_poisson(C, lib, labR, dR, t=T_LRT)
    r2g = []
    for i in range(len(cand)):
        bi, _, kp = gaston_bins(dR, NB)
        y = np.array([C[bi == b, i].sum() for b in kp]) / np.maximum(
            np.array([lib[bi == b].sum() for b in kp]), 1)
        db = np.array([dR[bi == b].mean() for b in kp])
        lb = np.array([np.bincount(labR[bi == b], minlength=labR.max() + 1).argmax() for b in kp])
        pred = np.exp(fitG[1][i, lb] + fitG[0][i, lb] * db)
        r2g.append(1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
    sg = pd.DataFrame(dict(feature=vn[cand], counts=tot[cand].astype(int),
                           n_sloped_domains=(np.abs(fitG[0]) > 1e-9).sum(1), R2=np.round(r2g, 3),
                           slopes=[np.round(fitG[0][i], 2).tolist() for i in range(len(cand))]))
    sg = sg.sort_values(["n_sloped_domains", "R2"], ascending=False)
    sg.to_csv(f"{OM}/gene_screen.csv", index=False)

    sel = np.argsort(-V.var(0))[:min(NTOP_SCREEN, V.shape[1])]
    slope, inter = fit_gauss(V[:, sel], labM, dM, t=T_LRT)
    r2m = [binned_r2_cont(V[:, j], dM, labM, slope[i], inter[i])[0] for i, j in enumerate(sel)]
    sm = pd.DataFrame(dict(feature=fn[sel], variance=np.round(V[:, sel].var(0), 3),
                           n_sloped_domains=(np.abs(slope) > 1e-12).sum(1), R2=np.round(r2m, 3),
                           nearest_gene=near_g[sel], kb_to_TSS=np.round(near_d[sel] / 1000, 1),
                           slopes=[np.round(s, 3).tolist() for s in slope]))
    sm = sm.sort_values(["n_sloped_domains", "R2"], ascending=False)
    sm.to_csv(f"{OM}/{mod.lower()}_screen.csv", index=False)
    print(f"   RNA: {int((sg.n_sloped_domains >= 2).sum())}/{len(sg)} sloped in >=2 domains | "
          f"{mod}: {int((sm.n_sloped_domains >= 2).sum())}/{len(sm)}", flush=True)

    # ---- domain annotation ---------------------------------------------------
    rows = []
    for m_, lab, d_ in [("RNA", labR, dR), (mod, labM, dM)]:
        gM = (np.abs(slope) > 1e-12).sum(0)
        for k in range(lab.max() + 1):
            ina = lab == k
            row = dict(modality=m_, domain=k, n_spots=int(ina.sum()),
                       isodepth=f"[{d_[ina].min():.2f}, {d_[ina].max():.2f}]")
            for c in cfg["cats"]:
                if c in z:
                    v = z[c].astype(str)
                    row[c.replace("ann_", "")] = json.dumps(
                        {n: round(float((v[ina] == n).mean()), 2) for n in sorted(set(v.tolist()))})
            for c in cfg["conts"]:
                if c in z:
                    row[c] = round(float(np.nanmean(np.asarray(z[c], float)[ina])), 1)
            if m_ == "RNA":
                row["gradient_feats"] = f"{int(gR[k])}/500"
                row["markers"] = "; ".join(f"{g}({lf:+.1f})" for g, lf, _ in
                                           markers(X, lib, lab, vn, k))
            else:
                row["gradient_feats"] = f"{int(gM[k])}/{len(sel)}"
                mk = cont_markers(V[:, sel], lab, fn[sel], k)
                row["markers"] = "; ".join(
                    (f"{f_} ({df:+.2f}, t{t_:+.0f})" if coord is None else
                     f"{near_g[sel][np.where(fn[sel] == f_)[0][0]]}|{f_.split(':')[0]} "
                     f"({df:+.2f}, t{t_:+.0f})") for f_, df, t_ in mk)
            rows.append(row)
    ann = pd.DataFrame(rows)
    ann.to_csv(f"{OD}/domain_annotation.csv", index=False)
    for m_ in ["RNA", mod]:
        sub = ann[ann.modality == m_]
        print(f"\n=== {dataset} {m_} domains ===")
        print(sub.drop(columns=[c for c in sub.columns
                                if c.endswith("_11") or "cc_K10" in c or "wsnn" in c]).to_string(index=False))

    fig, AX = plt.subplots(1, 2, figsize=(11, 4.6))
    for a_, (m_, lab, d_, bn) in zip(AX, [("RNA", labR, dR, bR), (mod, labM, dM, bM)]):
        panel_space(a_, xy, lab, "", None, contour_d=d_, bnds=bn, discrete=True)
        a_.set_title(f"{dataset}: {m_} domains (GASTON DP, K={kdom})\n" +
                     "\n".join(f"D{k} ({int((lab == k).sum())})" for k in range(lab.max() + 1)),
                     fontsize=6, loc="left")
    fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(f"{OD}/domain_maps.{e}", dpi=300 if e == "png" else None)
    plt.close(fig)

    # ---- RNA gene panels (Poisson), same layout as the ATAC datasets ---------
    top_g = sg[sg.n_sloped_domains >= 2].head(n_rows)
    draw_poisson(dataset, "RNA", top_g, X, lib, labR, dR, xy, bR, vn, OM)

    # ---- top-8 panels for the continuous modality ----------------------------
    top = sm[sm.n_sloped_domains >= 2].head(n_rows)
    fig, AX = plt.subplots(len(top), 5, figsize=(18.5, 3.0 * len(top)), squeeze=False)
    for row, (_, rec) in enumerate(top.iterrows()):
        j = int(np.where(fn == rec.feature)[0][0])
        i_ = int(np.where(fn[sel] == rec.feature)[0][0])
        y = V[:, j]
        r2, db, ob, pred, lb = binned_r2_cont(y, dM, labM, slope[i_], inter[i_])
        ttl = rec.feature if coord is None else f"{rec.feature}\n{rec.nearest_gene} {rec.kb_to_TSS:+.0f} kb"
        panel_space(AX[row, 0], xy, dM, f"rho_{mod} (Saturn isodepth)", "coolwarm", contour_d=dM, bnds=bM)
        panel_space(AX[row, 1], xy, labM, f"{mod} domains (K={kdom})", None,
                    contour_d=dM, bnds=bM, discrete=True)
        panel_space(AX[row, 2], xy, y, f"{ttl}\nRAW", "coolwarm", contour_d=dM, bnds=bM)
        panel_space(AX[row, 3], xy, inter[i_][labM] + slope[i_][labM] * dM, f"{ttl}\nFITTED",
                    "coolwarm", contour_d=dM, bnds=bM)
        a_ = AX[row, 4]
        for k in range(labM.max() + 1):
            mk_ = lb == k
            a_.plot(db[mk_], ob[mk_], "o", ms=3, color=f"C{k}")
            if mk_.sum():
                xs = np.linspace(db[mk_].min(), db[mk_].max(), 20)
                a_.plot(xs, inter[i_][k] + slope[i_][k] * xs, color=f"C{k}", lw=1.6)
        a_.set_xlabel(f"isodepth d_{mod}"); a_.set_ylabel(cfg["unit"])
        a_.set_title(f"{rec.feature}: sloped in {int(rec.n_sloped_domains)}/{labM.max() + 1} domains, R2 {r2:.2f}")
        a_.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{dataset}: top {len(top)} {mod} features by multi-domain function "
                 f"(Gaussian piecewise fit, F-gate p<{T_LRT}) -- screened on d_{mod} alone", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / (3.0 * len(top))))
    for e in ("png", "svg"):
        fig.savefig(f"{OM}/{mod.lower()}_top{len(top)}_functions.{e}", dpi=250 if e == "png" else None)
    plt.close(fig)
    print(f"\nwritten {OD} and {OM}", flush=True)
    return ann, sm


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
