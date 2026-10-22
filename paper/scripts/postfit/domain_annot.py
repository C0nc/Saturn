"""Annotate the RNA and ATAC domains of a Saturn fit, the GASTON way.

GASTON annotates the domains its DP segmentation returns by (i) what each domain is
made of -- cluster / cell-type proportions and the anatomy it lands on, (ii) the marker
features specific to it, and (iii) how many features vary CONTINUOUSLY inside it (a
sloped per-domain fit surviving the LRT gate) rather than only jumping at the boundary.

  composition  the dataset's own categorical and continuous annotations
  markers      RNA: top genes by a Poisson enrichment z (one-vs-rest)
               ATAC: top peaks/tiles mapped to their nearest TSS, and the same
                     markers after summing every peak into its nearest gene
  gradient     number of the top-expressed features whose slope in this domain
               survives GASTON's LRT gate (t = 0.1)

Usage: python scripts/domain_annot.py <dataset>      (lpc10_fov | melanoma_whole)
"""
import os, re, sys, json
import numpy as np, pandas as pd, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
from prep_real2 import _read_h5ad_safe, LPC_TP, H
from lpc_gaston_fig4l import gaston_domains, boundaries, panel_space, peak_class
from gaston.segmented_fit import pw_linear_fit
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "svg.fonttype": "none"})

R = f"{H}/Hole/gaston_mix_mm/results/real_saturn"
KDOM, T_LRT, NTOP = 4, 0.1, 500
GTF_MM = f"{H}/mouse_atlas/gencode.vM25.primary_chr.annotation.gtf"
GTF_HS = f"{H}/gencode.v21.annotation.gtf"

CFG = {
    "lpc10_fov": dict(
        gtf=GTF_MM,
        rna=lambda: _read_h5ad_safe(f"{H}/{LPC_TP['LPC10']['rna']}"),
        atac=lambda: _read_h5ad_safe(f"{H}/{LPC_TP['LPC10']['atac']}"),
        cats=["ann_region", "ann_domain"],
        conts=["ann_microglia_z", "ann_myelin_z", "ann_dist_to_lesion"],
    ),
    "melanoma_tissue": dict(
        gtf=GTF_HS,
        rna=lambda: __import__("anndata").read_h5ad(
            f"{H}/Hole/gaston_mix_mm/results/melanoma/melanoma_rna.h5ad"),
        atac=lambda: (__import__("hdf5plugin"), __import__("anndata").read_h5ad(
            f"{H}/Hole/gaston_mix_mm/results/melanoma/atac_snap.h5ad"))[1],
        cats=["ann_cc_K7", "ann_tumor_mask"],
        conts=["ann_cnv_burden_v2", "ann_mel_marker", "ann_immune_marker",
               "ann_stromal_marker", "ann_mel_frac", "ann_immune_frac"],
    ),
    "melanoma_whole": dict(
        gtf=GTF_HS,
        rna=lambda: __import__("anndata").read_h5ad(
            f"{H}/Hole/gaston_mix_mm/results/melanoma/melanoma_rna.h5ad"),
        atac=lambda: (__import__("hdf5plugin"), __import__("anndata").read_h5ad(
            f"{H}/Hole/gaston_mix_mm/results/melanoma/atac_snap.h5ad"))[1],
        cats=["ann_cc_K7", "ann_tumor_mask"],
        conts=["ann_cnv_burden_v2", "ann_mel_marker", "ann_immune_marker",
               "ann_stromal_marker", "ann_mel_frac", "ann_immune_frac"],
    ),
}


def all_tss(gtf):
    """chrom, TSS, name for every gene record in the annotation."""
    ch, po, nm = [], [], []
    with open(gtf) as f:
        for line in f:
            if line[0] == "#":
                continue
            p = line.split("\t", 9)
            if p[2] != "gene":
                continue
            m = re.search(r'gene_name "([^"]+)"', p[8])
            if m:
                ch.append(p[0]); po.append(int(p[3]) if p[6] == "+" else int(p[4])); nm.append(m.group(1))
    return np.array(ch), np.array(po, float), np.array(nm)


def markers(Mat, lib, lab, feats, k, n=8, min_count=100, min_lfc=0.5):
    """one-vs-rest markers of domain k, ranked by a Poisson enrichment z.

    Log2 fold change alone puts sparse predicted genes on top, so features are ranked
    by z = (obs - exp) / sqrt(exp) with exp = total counts x (domain library / total
    library), keeping only features with enough counts and a real fold change.
    """
    ina = lab == k
    tot = np.asarray(Mat.sum(0)).ravel().astype(float)   # uint32 would wrap under -tot
    obs = np.asarray(Mat[ina].sum(0)).ravel()
    exp = tot * (lib[ina].sum() / max(lib.sum(), 1))
    a = obs / max(lib[ina].sum(), 1)
    b = (tot - obs) / max(lib[~ina].sum(), 1)
    lfc = np.log2((a + 1e-9) / (b + 1e-9))
    zsc = (obs - exp) / np.sqrt(np.maximum(exp, 1e-9))
    zsc[(tot < min_count) | (lfc < min_lfc)] = -np.inf
    o = np.argsort(-zsc)[:n]
    return [(feats[i], round(float(lfc[i]), 2), int(tot[i])) for i in o]


def gradient_count(Mat, lib, lab, d, ntop=NTOP):
    """features whose per-domain slope passes GASTON's LRT gate, per domain"""
    tot = np.asarray(Mat.sum(0)).ravel().astype(float)   # uint32 would wrap under -tot
    keep = np.argsort(-tot)[:ntop]
    C = np.asarray(Mat[:, keep].todense())
    rest = np.maximum(lib - C.sum(1), 0)
    fit = pw_linear_fit(np.c_[C, rest], lab, d, None, None, idx_kept=np.arange(len(keep)),
                        umi_threshold=0, t=T_LRT)["all_cell_types"]
    return (np.abs(fit[0]) > 1e-9).sum(0)


def nearest_gene(pkn, gtf):
    """every peak / tile -> (nearest gene name, signed distance to its TSS)"""
    tch, tpo, tnm = all_tss(gtf)
    pk = pd.Series(pkn).str.extract(r"(chr[\w]+)[:\-_](\d+)[\-_](\d+)")
    pk.columns = ["chr", "s", "e"]
    mid = (pk["s"].astype(float) + pk["e"].astype(float)).values / 2
    g = np.full(len(pkn), "NA", dtype=object); dd = np.full(len(pkn), np.nan)
    for c_ in pd.unique(pk["chr"].dropna()):
        m = tch == c_; sel = np.where((pk["chr"] == c_).values)[0]
        if not m.any() or not len(sel):
            continue
        tp_, tn_ = tpo[m], tnm[m]
        o = np.argsort(tp_); tp_, tn_ = tp_[o], tn_[o]
        j = np.clip(np.searchsorted(tp_, mid[sel]), 1, len(tp_) - 1)
        pick = np.where(np.abs(mid[sel] - tp_[j - 1]) <= np.abs(mid[sel] - tp_[j]), j - 1, j)
        g[sel] = tn_[pick]; dd[sel] = mid[sel] - tp_[pick]
    return g, dd


def run(dataset, kdom=KDOM, labels=None):
    cfg = CFG[dataset]
    OUT = f"{R}/{dataset}_domains"; os.makedirs(OUT, exist_ok=True)
    ax = pd.read_csv(f"{R}/{dataset}/axes.csv", index_col=0)
    z = dict(np.load(f"{R}/targets_{dataset}.npz", allow_pickle=True))
    dR, dA = ax["d_RNA"].values, ax["d_ATAC"].values
    xy = np.c_[ax.x.values, ax.y.values]
    labR = gaston_domains(dR, z["Y_RNA"], L_=kdom)
    labA = gaston_domains(dA, z["Y_ATAC"], L_=kdom)
    bR, bA = boundaries(dR, labR), boundaries(dA, labA)

    idx = pd.Index([str(b) for b in ax.index])
    r = cfg["rna"]()[idx].copy(); a = cfg["atac"]()[idx].copy()
    X = sp.csr_matrix(r.X); vn = np.array([str(v) for v in r.var_names])
    lib = np.asarray(X.sum(1)).ravel().clip(1)
    A_ = sp.csr_matrix(a.X); pkn = np.array([str(v) for v in a.var_names])
    alib = np.asarray(A_.sum(1)).ravel().clip(1)

    near_g, near_d = nearest_gene(pkn, cfg["gtf"])
    gname = pd.Index(sorted(set(near_g[near_g != "NA"])))
    col = gname.get_indexer(near_g); keep_p = col >= 0
    Agene = sp.csr_matrix(A_[:, np.where(keep_p)[0]] @
                          sp.csr_matrix((np.ones(keep_p.sum()), (np.arange(keep_p.sum()), col[keep_p])),
                                        shape=(int(keep_p.sum()), len(gname))))
    gname = np.array(gname)
    pos = {p: i for i, p in enumerate(pkn)}

    gR = gradient_count(X, lib, labR, dR)
    gA = gradient_count(A_, alib, labA, dA)

    rows = []
    for mod, lab, d, Mat, libm, feats, grad in [("RNA", labR, dR, X, lib, vn, gR),
                                                ("ATAC", labA, dA, A_, alib, pkn, gA)]:
        dep = np.asarray(z[f"depth_{mod}"], float)
        for k in range(lab.max() + 1):
            ina = lab == k
            mk = markers(Mat, libm, lab, feats, k)
            if mod == "ATAC":
                mk_s = "; ".join(f"{near_g[pos[p]]}({peak_class(near_d[pos[p]])[0]}"
                                 f"{near_d[pos[p]] / 1000:+.0f}kb, {lf:+.1f})" for p, lf, _ in mk)
                gl = "; ".join(f"{g}({lf:+.1f})" for g, lf, _ in
                               markers(Agene, alib, lab, gname, k, min_count=300))
            else:
                mk_s = "; ".join(f"{g}({lf:+.1f})" for g, lf, _ in mk)
                gl = ""
            row = dict(modality=mod, domain=k, label=(labels or {}).get((mod, k), ""),
                       n_spots=int(ina.sum()), isodepth=f"[{d[ina].min():.2f}, {d[ina].max():.2f}]",
                       median_depth=int(np.median(dep[ina])))
            for c in cfg["cats"]:
                if c in z:
                    v = z[c].astype(str)
                    row[c.replace("ann_", "")] = json.dumps(
                        {n: round(float((v[ina] == n).mean()), 2) for n in sorted(set(v.tolist()))})
            for c in cfg["conts"]:
                if c in z:
                    row[c.replace("ann_", "")] = round(float(np.nanmean(np.asarray(z[c], float)[ina])), 2)
            row.update(gradient_feats=f"{int(grad[k])}/{NTOP}", markers=mk_s, gene_level=gl)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/domain_annotation.csv", index=False)
    cats = [c.replace("ann_", "") for c in cfg["cats"]]
    for mod in ["RNA", "ATAC"]:
        sub = df[df.modality == mod]
        print(f"\n=== {dataset} {mod} domains ===")
        print(sub.drop(columns=["modality"] + [c for c in cats if c in sub]).to_string(index=False))
        for c in cats:
            if c in sub:
                print(f"  {c}:", sub[c].tolist())

    fig, AX = plt.subplots(1, 2, figsize=(11, 4.6))
    for a_, (mod, lab, d, bn) in zip(AX, [("RNA", labR, dR, bR), ("ATAC", labA, dA, bA)]):
        panel_space(a_, xy, lab, "", None, contour_d=d, bnds=bn, discrete=True)
        a_.set_title(f"{dataset}: {mod} domains (GASTON DP, K={kdom})\n" +
                     "\n".join(f"D{k} ({int((lab == k).sum())}) {(labels or {}).get((mod, k), '')}"
                               for k in range(lab.max() + 1)), fontsize=6, loc="left")
    fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(f"{OUT}/domain_maps.{e}", dpi=300 if e == "png" else None)
    print("\nwritten", OUT)
    return df


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "melanoma_whole")
