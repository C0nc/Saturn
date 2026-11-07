"""Annotate every RNA and ATAC domain of the LPC10 FOV fit, the GASTON way.

GASTON annotates the domains its DP segmentation returns by (i) what the domain is
made of -- cell-type / cluster proportions and the known anatomy it lands on, (ii) the
marker features that are specific to it, and (iii) how many features vary CONTINUOUSLY
inside it (a sloped per-domain fit that passes the LRT gate) as opposed to jumping at
the boundary. The same three things are computed here:

  composition  region labels (lesion / distal WM / other), CellCharter cluster
               proportions, mean microglia and myelin scores, distance to the lesion
  markers      RNA: top genes by one-vs-rest log2 fold change
               ATAC: top peaks, each mapped to its nearest TSS and proximity class
  gradient     number of the top-expressed features whose slope in this domain
               survives GASTON's LRT gate (t = 0.1) -- i.e. features that form a
               gradient WITHIN the domain rather than only differing between domains
"""
import os, re, sys, json
import numpy as np, pandas as pd, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
from prep_real2 import _read_h5ad_safe, LPC_TP, GTF, H
from lpc_gaston_fig4l import gaston_domains, boundaries, panel_space, peak_class
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/saturn_repo/src")
from saturn import annotate as AN
from saturn.piecewise import fit_poisson
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "svg.fonttype": "none"})

R = f"{H}/Hole/gaston_mix_mm/results/real_saturn"
OUT = f"{R}/lpc10_fov_domains"; os.makedirs(OUT, exist_ok=True)
KDOM, T_LRT, NTOP = 4, 0.1, 500
# final annotation, read off the composition + marker columns below
LABELS = {
    ("RNA", 0): "lesion core + peri-lesional striatum",
    ("RNA", 1): "intact white matter",
    ("RNA", 2): "periventricular / choroid plexus + distal WM",
    ("RNA", 3): "neuronal parenchyma (thalamic)",
    ("ATAC", 0): "myeloid inflammatory chromatin (lesion + distal WM)",
    ("ATAC", 1): "parenchymal background",
    ("ATAC", 2): "transitional band (background -> neuronal)",
    ("ATAC", 3): "neuronal chromatin",
}


def all_tss():
    """chrom, TSS, name for every GENCODE vM25 gene (saturn.annotate.all_tss)"""
    return AN.all_tss(GTF)


def comp(lab, k, values, names):
    """proportion of each label value inside domain k"""
    v = values[lab == k]
    return {n: round(float((v == n).mean()), 2) for n in names}


markers = AN.poisson_markers   # one-vs-rest Poisson enrichment z (log2FC alone puts sparse genes on top)


def gradient_count(Mat, lib, lab, d, ntop=NTOP):
    """features whose per-domain slope passes GASTON's LRT gate, per domain"""
    tot = np.asarray(Mat.sum(0)).ravel().astype(float)   # uint32 would wrap under -tot
    keep = np.argsort(-tot)[:ntop]
    slope, _ = fit_poisson(np.asarray(Mat[:, keep].todense()), lib, lab, d, t=T_LRT)
    return (np.abs(slope) > 1e-9).sum(0), keep, slope


ax = pd.read_csv(f"{R}/lpc10_fov/axes.csv", index_col=0)
z = dict(np.load(f"{R}/targets_lpc10_fov.npz", allow_pickle=True))
dR, dA = ax["d_RNA"].values, ax["d_ATAC"].values
xy = np.c_[ax.x.values, ax.y.values]
labR = gaston_domains(dR, z["Y_RNA"], L_=KDOM)
labA = gaston_domains(dA, z["Y_ATAC"], L_=KDOM)
bR, bA = boundaries(dR, labR), boundaries(dA, labA)
reg = z["ann_region"].astype(str)
cc = z["ann_domain"].astype(str)
cc_names = sorted(set(cc.tolist()))
mg, my, dist = z["ann_microglia_z"].astype(float), z["ann_myelin_z"].astype(float), z["ann_dist_to_lesion"].astype(float)

idx = pd.Index([str(b) for b in ax.index])
r = _read_h5ad_safe(f"{H}/{LPC_TP['LPC10']['rna']}")[idx].copy()
a = _read_h5ad_safe(f"{H}/{LPC_TP['LPC10']['atac']}")[idx].copy()
X = sp.csr_matrix(r.X); vn = np.array([str(v) for v in r.var_names]); lib = np.asarray(X.sum(1)).ravel().clip(1)
A_ = sp.csr_matrix(a.X); pkn = np.array([str(v) for v in a.var_names]); alib = np.asarray(A_.sum(1)).ravel().clip(1)

# peak -> nearest TSS
tch, tpo, tnm = all_tss()
pk = pd.Series(pkn).str.extract(r"(chr[\w]+)[:\-_](\d+)[\-_](\d+)")
pk.columns = ["chr", "s", "e"]
pmid = (pk["s"].astype(float) + pk["e"].astype(float)).values / 2
near_g = np.full(len(pkn), "NA", dtype=object); near_d = np.full(len(pkn), np.nan)
for c_ in pd.unique(pk["chr"].dropna()):
    m = tch == c_; sel = np.where((pk["chr"] == c_).values)[0]
    if not m.any() or not len(sel):
        continue
    tp_, tn_ = tpo[m], tnm[m]
    o = np.argsort(tp_); tp_, tn_ = tp_[o], tn_[o]
    j = np.clip(np.searchsorted(tp_, pmid[sel]), 1, len(tp_) - 1)
    pick = np.where(np.abs(pmid[sel] - tp_[j - 1]) <= np.abs(pmid[sel] - tp_[j]), j - 1, j)
    near_g[sel] = tn_[pick]; near_d[sel] = pmid[sel] - tp_[pick]

# gene-level accessibility: sum every peak into its nearest gene, so the ATAC markers
# read as gene names instead of individual sparse peaks
gname = pd.Index(sorted(set(near_g[near_g != "NA"])))
col = gname.get_indexer(near_g)
keep_p = col >= 0
Agene = sp.csr_matrix(A_[:, np.where(keep_p)[0]] @
                      sp.csr_matrix((np.ones(keep_p.sum()), (np.arange(keep_p.sum()), col[keep_p])),
                                    shape=(int(keep_p.sum()), len(gname))))
gname = np.array(gname)

gR, keepR, slopeR = gradient_count(X, lib, labR, dR)
gA, keepA, slopeA = gradient_count(A_, alib, labA, dA)

rows = []
for mod, lab, d, Mat, libm, feats, grad in [("RNA", labR, dR, X, lib, vn, gR),
                                            ("ATAC", labA, dA, A_, alib, pkn, gA)]:
    for k in range(lab.max() + 1):
        mk = markers(Mat, libm, lab, feats, k)
        if mod == "ATAC":
            mk_s = "; ".join(f"{near_g[np.where(feats == p)[0][0]]}"
                             f"({peak_class(near_d[np.where(feats == p)[0][0]])[0]}"
                             f"{near_d[np.where(feats == p)[0][0]] / 1000:+.0f}kb, {lf:+.1f})"
                             for p, lf, _ in mk)
        else:
            mk_s = "; ".join(f"{g}({lf:+.1f})" for g, lf, _ in mk)
        dep = (z["depth_RNA"] if mod == "RNA" else z["depth_ATAC"]).astype(float)
        rows.append(dict(modality=mod, domain=k, label=LABELS[(mod, k)], n_spots=int((lab == k).sum()),
                         isodepth=f"[{d[lab == k].min():.2f}, {d[lab == k].max():.2f}]",
                         **{f"frac_{n}": comp(lab, k, reg, ["lesion", "distal_WM", "other"])[n]
                            for n in ["lesion", "distal_WM", "other"]},
                         median_depth=int(np.median(dep[lab == k])),
                         microglia_z=round(float(mg[lab == k].mean()), 1),
                         myelin_z=round(float(my[lab == k].mean()), 1),
                         dist_to_lesion=round(float(dist[lab == k].mean()), 0),
                         cellcharter=json.dumps(comp(lab, k, cc, cc_names)),
                         gradient_feats=f"{int(grad[k])}/{NTOP}", markers=mk_s,
                         gene_level=("; ".join(f"{g}({lf:+.1f})" for g, lf, _ in
                                               markers(Agene, alib, lab, gname, k, min_count=300))
                                     if mod == "ATAC" else "")))
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/domain_annotation.csv", index=False)
for mod in ["RNA", "ATAC"]:
    print(f"\n=== {mod} domains ===")
    print(df[df.modality == mod].drop(columns=["modality", "cellcharter"]).to_string(index=False))
    print("  cellcharter:", df[df.modality == mod].cellcharter.tolist())

fig, AX = plt.subplots(1, 2, figsize=(11, 4.6))
for a_, (mod, lab, d, bn) in zip(AX, [("RNA", labR, dR, bR), ("ATAC", labA, dA, bA)]):
    panel_space(a_, xy, lab, "", None, contour_d=d, bnds=bn, discrete=True)
    a_.set_title(f"LPC10 FOV: {mod} domains (GASTON DP, K={KDOM})\n" +
                 "\n".join(f"D{k} ({int((lab == k).sum())}) {LABELS[(mod, k)]}"
                            for k in range(lab.max() + 1)), fontsize=6, loc="left")
fig.tight_layout()
for e in ("png", "svg"):
    fig.savefig(f"{OUT}/domain_maps.{e}", dpi=300 if e == "png" else None)
print("\nwritten", OUT)
