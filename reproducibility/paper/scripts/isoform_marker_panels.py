"""Domain annotation + marker screening for the ISOFORM modality.

Isoform usage is a proportion out of a known denominator -- the gene's reads in that
spot -- so the piecewise fit is BINOMIAL, logit(p) = a_k + b_k * rho per domain, with
the same LRT gate (t = 0.1) GASTON applies to its Poisson fits. Fitting the CLR values
with a Gaussian instead would let an isoform look "sloped" merely because its GENE is
sloped; putting the gene total in as the binomial denominator removes that by
construction, which is the whole point of screening usage rather than expression.

RNA keeps the Poisson fit. Written per dataset:
  <ds>_domains/domain_annotation.csv, domain_maps.*
  <ds>_marker_functions/{gene,isoform}_screen.csv, isoform_top8_functions.*

Usage: python scripts/isoform_marker_panels.py rtt_merfish_s5 [n_rows]
"""
import os, sys, json
import numpy as np, pandas as pd, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "<workspace>/Hole/gaston_mix_mm/scripts")
sys.path.insert(0, "<workspace>/saturn_repo/src")
from prep_real2 import H
from lpc_gaston_fig4l import gaston_domains, boundaries, panel_space
from lpc_gene_isodepth import gaston_bins
from domain_annot import R, markers, gradient_count
from piecewise import fit_binomial
from marker_function_panels import draw as draw_poisson
from saturn.piecewise import fit_poisson
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "svg.fonttype": "none"})

KDOM, T_LRT, NB = 4, 0.1, 25
MIN_GENE_READS, MIN_ISO_READS, NTOP = 200, 50, 800


def load_rtt_merfish(sample=5, bin_um=50.0):
    """RT&T-AMP-MERFISH: gene and isoform feature counts per bin, same binning as prep"""
    import anndata as ad
    D = f"{H}/external_data/rtt_merfish"
    a = ad.read_h5ad(f"{D}/cellxgene_ISO_labels.h5ad")
    a = a[a.obs["sample"].astype(str) == str(sample)].copy()
    iso = np.array([str(v).startswith("ENSMUST") for v in a.var_names])
    t2g = pd.read_csv(f"{D}/ensembl_tx2gene_mm.tsv", sep="\t").drop_duplicates("tx").set_index("tx")["gene"]
    ig = t2g.reindex(a.var_names[iso]).values
    ok_i = pd.notna(ig)
    xy = a.obs[["global_x", "global_y"]].values.astype(float)
    key = np.floor(xy / bin_um).astype(np.int64); code = key[:, 0] * 1_000_000 + key[:, 1]
    uniq, inv, cnt = np.unique(code, return_inverse=True, return_counts=True)
    kb = np.where(cnt >= 3)[0]
    G = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(uniq), len(inv)))[kb]
    X = sp.csr_matrix(a.X)
    Xg = sp.csr_matrix(G @ X[:, np.where(~iso)[0]])
    Xi = sp.csr_matrix(G @ X[:, np.where(iso)[0][ok_i]])
    genes = np.array([str(v) for v in a.var_names[~iso]])
    isos = np.array([str(v) for v in a.var_names[iso][ok_i]])
    return Xg, genes, Xi, isos, np.asarray(ig[ok_i], dtype=object), np.array([str(u) for u in uniq[kb]])


LOADERS = {"rtt_merfish_s5": lambda: load_rtt_merfish(5)}


def ens2sym():
    """Ensembl gene id -> symbol, from the GENCODE vM25 annotation"""
    import re
    out = {}
    with open(f"{H}/mouse_atlas/gencode.vM25.primary_chr.annotation.gtf") as f:
        for line in f:
            if line[0] == "#":
                continue
            p_ = line.split("\t", 9)
            if p_[2] != "gene":
                continue
            gi = re.search(r'gene_id "([^".]+)', p_[8]); gn = re.search(r'gene_name "([^"]+)"', p_[8])
            if gi and gn:
                out[gi.group(1)] = gn.group(1)
    return out


def run(dataset="rtt_merfish_s5", n_rows=8, kdom=KDOM):
    OD = f"{R}/{dataset}_domains"; OM = f"{R}/{dataset}_marker_functions"
    os.makedirs(OD, exist_ok=True); os.makedirs(OM, exist_ok=True)
    ax = pd.read_csv(f"{R}/{dataset}/axes.csv", index_col=0)
    z = dict(np.load(f"{R}/targets_{dataset}.npz", allow_pickle=True))
    dR, dI = ax["d_RNA"].values, ax["d_Isoform"].values
    xy = np.c_[ax.x.values, ax.y.values]
    labR = gaston_domains(dR, z["Y_RNA"], L_=kdom)
    labI = gaston_domains(dI, z["Y_Isoform"], L_=kdom)
    bR, bI = boundaries(dR, labR), boundaries(dI, labI)
    print(f"{dataset}: RNA domains {np.bincount(labR).tolist()} | Isoform domains {np.bincount(labI).tolist()}",
          flush=True)

    Xg, genes, Xi, isos, iso_gene, bins = LOADERS[dataset]()
    sym = ens2sym()
    gsym = np.array([sym.get(str(g).split(".")[0], str(g).split(".")[0]) for g in genes])
    isym = np.array([sym.get(str(g).split(".")[0], str(g).split(".")[0]) for g in iso_gene])
    # MERFISH thresholds are far lower than long-read ones
    global MIN_GENE_READS, MIN_ISO_READS
    if dataset.startswith("rtt_merfish"):
        MIN_GENE_READS, MIN_ISO_READS = 50, 20
    pos = pd.Index([str(b) for b in bins]).get_indexer(pd.Index([str(b) for b in ax.index]))
    assert (pos >= 0).all(), "fitted bins missing from the rebuilt matrices"
    Xg, Xi = Xg[pos], Xi[pos]
    lib = np.asarray(Xg.sum(1)).ravel().clip(1)

    # ---- RNA screen (Poisson, unchanged) --------------------------------------
    gR = gradient_count(Xg, lib, labR, dR)
    tot = np.asarray(Xg.sum(0)).ravel().astype(float)
    cand = np.argsort(-tot)[:1500]; cand = cand[tot[cand] >= 100]
    C = np.asarray(Xg[:, cand].todense())
    fitG = fit_poisson(C, lib, labR, dR, t=T_LRT)
    r2g = []
    bi0, _, kp0 = gaston_bins(dR, NB)
    for i in range(len(cand)):
        y = np.array([C[bi0 == b, i].sum() for b in kp0]) / np.maximum(
            np.array([lib[bi0 == b].sum() for b in kp0]), 1)
        db = np.array([dR[bi0 == b].mean() for b in kp0])
        lb = np.array([np.bincount(labR[bi0 == b], minlength=labR.max() + 1).argmax() for b in kp0])
        pred = np.exp(fitG[1][i, lb] + fitG[0][i, lb] * db)
        r2g.append(1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
    sg = pd.DataFrame(dict(feature=gsym[cand], ensembl=genes[cand], counts=tot[cand].astype(int),
                           n_sloped_domains=(np.abs(fitG[0]) > 1e-9).sum(1), R2=np.round(r2g, 3),
                           slopes=[np.round(fitG[0][i], 2).tolist() for i in range(len(cand))]))
    sg = sg.sort_values(["n_sloped_domains", "R2"], ascending=False)
    sg.to_csv(f"{OM}/gene_screen.csv", index=False)

    # ---- isoform screen (binomial on gene totals) -----------------------------
    gcode, gnames = pd.factorize(pd.Series(iso_gene))
    G = sp.csr_matrix((np.ones(len(gcode)), (np.arange(len(gcode)), gcode)), shape=(len(gcode), len(gnames)))
    gene_tot_per_iso = np.asarray((Xi @ G).todense())[:, gcode]       # spots x isoforms: gene totals
    tot_i = np.asarray(Xi.sum(0)).ravel().astype(float)
    tot_g = np.asarray((Xi @ G).sum(0)).ravel().astype(float)
    n_iso = np.asarray(G.sum(0)).ravel()
    usable = (n_iso[gcode] >= 2) & (tot_g[gcode] >= MIN_GENE_READS) & (tot_i >= MIN_ISO_READS)
    idx = np.where(usable)[0]
    p_glob = tot_i[idx] / np.maximum(tot_g[gcode[idx]], 1)
    S = np.asarray(Xi[:, idx].todense(), float)
    N = gene_tot_per_iso[:, idx]
    v = (S / np.maximum(N, 1) - p_glob).var(0)
    sel = idx[np.argsort(-v)[:NTOP]]
    Ssel = np.asarray(Xi[:, sel].todense(), float); Nsel = gene_tot_per_iso[:, sel]
    print(f"   isoforms usable {len(idx)} (>= {MIN_ISO_READS} reads, gene >= {MIN_GENE_READS}, >= 2 isoforms); "
          f"screening top {len(sel)} by usage variance", flush=True)
    slope, inter = fit_binomial(Ssel, Nsel, labI, dI, t=T_LRT)
    si = pd.DataFrame(dict(isoform=isos[sel], gene=isym[sel], ensembl=iso_gene[sel], reads=tot_i[sel].astype(int),
                           global_usage=np.round(tot_i[sel] / np.maximum(tot_g[gcode[sel]], 1), 3),
                           n_sloped_domains=(np.abs(slope) > 1e-12).sum(1),
                           slopes=[np.round(s, 2).tolist() for s in slope]))
    si["abs_slope"] = [np.abs(s).max() for s in slope]
    si = si.sort_values(["n_sloped_domains", "abs_slope"], ascending=False)
    si.to_csv(f"{OM}/isoform_screen.csv", index=False)
    print(f"   RNA: {int((sg.n_sloped_domains >= 2).sum())}/{len(sg)} sloped in >=2 domains | "
          f"Isoform: {int((si.n_sloped_domains >= 2).sum())}/{len(si)}", flush=True)

    # ---- domain annotation ----------------------------------------------------
    gI = (np.abs(slope) > 1e-12).sum(0)
    rows = []
    for m_, lab, d_ in [("RNA", labR, dR), ("Isoform", labI, dI)]:
        for k in range(lab.max() + 1):
            ina = lab == k
            row = dict(modality=m_, domain=k, n_spots=int(ina.sum()),
                       isodepth=f"[{d_[ina].min():.2f}, {d_[ina].max():.2f}]",
                       median_depth=int(np.median(np.asarray(z[f"depth_{m_}"], float)[ina])))
            if "ann_region" in z:
                v_ = z["ann_region"].astype(str)
                row["region"] = json.dumps({n: round(float((v_[ina] == n).mean()), 2)
                                            for n in sorted(set(v_.tolist()))[:6]})
            if m_ == "RNA":
                row["gradient_feats"] = f"{int(gR[k])}/500"
                row["markers"] = "; ".join(f"{g}({lf:+.1f})" for g, lf, _ in
                                           markers(Xg, lib, lab, gsym, k))
            else:
                row["gradient_feats"] = f"{int(gI[k])}/{len(sel)}"
                u = Ssel / np.maximum(Nsel, 1)
                diff = u[ina].mean(0) - u[~ina].mean(0)
                o = np.argsort(-np.abs(diff))[:8]
                row["markers"] = "; ".join(f"{isym[sel][i]}:{isos[sel][i]}"
                                           f"({diff[i]:+.2f})" for i in o)
            rows.append(row)
    ann = pd.DataFrame(rows); ann.to_csv(f"{OD}/domain_annotation.csv", index=False)
    for m_ in ["RNA", "Isoform"]:
        print(f"\n=== {dataset} {m_} domains ===")
        print(ann[ann.modality == m_].drop(columns=["modality"]).to_string(index=False))

    fig, AX = plt.subplots(1, 2, figsize=(11, 4.6))
    for a_, (m_, lab, d_, bn) in zip(AX, [("RNA", labR, dR, bR), ("Isoform", labI, dI, bI)]):
        panel_space(a_, xy, lab, "", None, contour_d=d_, bnds=bn, discrete=True)
        a_.set_title(f"{dataset}: {m_} domains (GASTON DP, K={kdom})", fontsize=7, loc="left")
    fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(f"{OD}/domain_maps.{e}", dpi=300 if e == "png" else None)
    plt.close(fig)

    # ---- RNA gene panels (Poisson) --------------------------------------------
    draw_poisson(dataset, "RNA", sg[sg.n_sloped_domains >= 2].head(n_rows),
                 Xg, lib, labR, dR, xy, bR, gsym, OM)

    # ---- top isoform usage functions -----------------------------------------
    top = si[si.n_sloped_domains >= 2].head(n_rows)
    fig, AX = plt.subplots(len(top), 5, figsize=(18.5, 3.0 * len(top)), squeeze=False)
    bi, _, kp = gaston_bins(dI, NB)
    for row, (_, rec) in enumerate(top.iterrows()):
        j = int(np.where(isos[sel] == rec.isoform)[0][0])
        s_, n_ = Ssel[:, j], Nsel[:, j]
        u = s_ / np.maximum(n_, 1)
        ub = np.array([s_[bi == b].sum() / max(n_[bi == b].sum(), 1) for b in kp])
        db = np.array([dI[bi == b].mean() for b in kp])
        lb = np.array([np.bincount(labI[bi == b], minlength=labI.max() + 1).argmax() for b in kp])
        ttl = f"{rec.gene}\n{rec.isoform}"
        panel_space(AX[row, 0], xy, dI, "rho_Isoform (Saturn isodepth)", "coolwarm", contour_d=dI, bnds=bI)
        panel_space(AX[row, 1], xy, labI, f"Isoform domains (K={kdom})", None,
                    contour_d=dI, bnds=bI, discrete=True)
        panel_space(AX[row, 2], xy, np.where(n_ > 0, u, np.nan), f"{ttl}\nRAW usage", "coolwarm",
                    contour_d=dI, bnds=bI)
        fit_u = 1 / (1 + np.exp(-(inter[j][labI] + slope[j][labI] * dI)))
        panel_space(AX[row, 3], xy, fit_u, f"{ttl}\nFITTED usage", "coolwarm", contour_d=dI, bnds=bI)
        a_ = AX[row, 4]
        for k in range(labI.max() + 1):
            mk_ = lb == k
            a_.plot(db[mk_], ub[mk_], "o", ms=3, color=f"C{k}")
            if mk_.sum():
                xs = np.linspace(db[mk_].min(), db[mk_].max(), 20)
                a_.plot(xs, 1 / (1 + np.exp(-(inter[j][k] + slope[j][k] * xs))), color=f"C{k}", lw=1.6)
        a_.axhline(rec.global_usage, ls="--", lw=0.8, color="grey")
        a_.set_xlabel("isodepth d_Isoform"); a_.set_ylabel("isoform usage")
        a_.set_ylim(0, 1)
        a_.set_title(f"{rec.isoform}: {int(rec.reads)} reads, global usage {rec.global_usage:.2f} (dashed)\n"
                     f"sloped in {int(rec.n_sloped_domains)}/{labI.max() + 1} domains")
        a_.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{dataset}: top {len(top)} isoforms by multi-domain USAGE function "
                 f"(binomial on gene totals, LRT gate p<{T_LRT})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / (3.0 * len(top))))
    for e in ("png", "svg"):
        fig.savefig(f"{OM}/isoform_top{len(top)}_functions.{e}", dpi=250 if e == "png" else None)
    plt.close(fig)
    print(f"\nwritten {OD} and {OM}", flush=True)
    return ann, si


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "rtt_merfish_s5",
        int(sys.argv[2]) if len(sys.argv) > 2 else 8)
