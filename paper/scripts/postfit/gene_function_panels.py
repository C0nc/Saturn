"""GASTON Fig-3c / 4l panels for the genes whose expression function is MULTI-DOMAIN.

Standard post-fit explanation step for every dataset: after Saturn is fitted, segment
both axes with GASTON's DP, fit every well-expressed gene with GASTON's piecewise
Poisson model along d_RNA, and keep the genes whose slope survives the LRT gate
(t = 0.1) in AT LEAST TWO domains -- those are the genes with a real multi-domain
expression function rather than a single sloped segment. Each of them gets the same
composite panel:

  row 1: rho_RNA | RNA domains | raw expression | fitted expression | expression function
  row 2: rho_ATAC | ATAC domains | best promoter | best proximal | best long-distance

with the domain boundaries drawn in the original spatial coordinates.

Usage: python scripts/gene_function_panels.py <dataset> [n_panels]
"""
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
from prep_real2 import H
from lpc_gaston_fig4l import gaston_domains, boundaries, panel_space, panel_function, peak_class, L
from lpc_gene_isodepth import gaston_bins, pois_fit
from domain_annot import CFG, R, nearest_gene, all_tss
from gaston.segmented_fit import pw_linear_fit
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "svg.fonttype": "none"})

NB_RNA, NB_ATAC, WIN, KDOM, T_LRT = 30, 20, 100_000, 4, 0.1
NCAND, MIN_COUNT = 1500, 200          # genes screened, and the count floor to screen them
CLASSES = ["promoter", "proximal enhancer", "long-distance enhancer"]


def fit_free(cnt, expo, centres, n_grid=10):
    best = None
    for bp in np.quantile(centres, np.linspace(0.2, 0.8, n_grid)):
        f = pois_fit(cnt, expo, centres, bp)
        if best is None or f["R2"] > best["R2"]:
            best = f
    return best


def binned_r2(cnt, lib, d, lab, fit, row, nb):
    """R2 of GASTON's piecewise fit against the binned observed rate"""
    i, c, k = gaston_bins(d, nb)
    y = np.array([cnt[i == b].sum() for b in k]) / np.maximum(
        np.array([lib[i == b].sum() for b in k]), 1)
    db = np.array([d[i == b].mean() for b in k])
    lb = np.array([np.bincount(lab[i == b], minlength=lab.max() + 1).argmax() for b in k])
    pred = np.exp(fit[1][row, lb] + fit[0][row, lb] * db)
    return float(1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))


def run(dataset, n_panels=8, kdom=KDOM):
    cfg = CFG[dataset]
    OUT = f"{R}/{dataset}_gene_functions"; os.makedirs(OUT, exist_ok=True)
    ax = pd.read_csv(f"{R}/{dataset}/axes.csv", index_col=0)
    z = dict(np.load(f"{R}/targets_{dataset}.npz", allow_pickle=True))
    dR, dA = ax["d_RNA"].values, ax["d_ATAC"].values
    xy = np.c_[ax.x.values, ax.y.values]
    labR = gaston_domains(dR, z["Y_RNA"], L_=kdom)
    labA = gaston_domains(dA, z["Y_ATAC"], L_=kdom)
    bR, bA = boundaries(dR, labR), boundaries(dA, labA)
    print(f"{dataset}: RNA domains {np.bincount(labR).tolist()} | ATAC domains {np.bincount(labA).tolist()}",
          flush=True)

    idx = pd.Index([str(b) for b in ax.index])
    r = cfg["rna"]()[idx].copy(); a = cfg["atac"]()[idx].copy()
    X = sp.csr_matrix(r.X); vn = np.array([str(v) for v in r.var_names])
    lib = np.asarray(X.sum(1)).ravel().clip(1)
    A_ = sp.csr_matrix(a.X); pkn = np.array([str(v) for v in a.var_names])
    alib = np.asarray(A_.sum(1)).ravel().clip(1)

    # ---- screen: which genes have a sloped fit in >= 2 domains --------------------
    tot = np.asarray(X.sum(0)).ravel().astype(float)
    cand = np.argsort(-tot)[:NCAND]
    cand = cand[tot[cand] >= MIN_COUNT]
    C = np.asarray(X[:, cand].todense())
    fitG = pw_linear_fit(np.c_[C, np.maximum(lib - C.sum(1), 0)], labR, dR, None, None,
                         idx_kept=np.arange(len(cand)), umi_threshold=0, t=T_LRT)["all_cell_types"]
    nslope = (np.abs(fitG[0]) > 1e-9).sum(1)
    r2 = np.array([binned_r2(C[:, i], lib, dR, labR, fitG, i, NB_RNA) for i in range(len(cand))])
    scr = pd.DataFrame(dict(gene=vn[cand], counts=tot[cand].astype(int), n_sloped_domains=nslope,
                            R2=np.round(r2, 3),
                            slopes=[np.round(fitG[0][i], 2).tolist() for i in range(len(cand))]))
    scr = scr.sort_values(["n_sloped_domains", "R2"], ascending=False)
    scr.to_csv(f"{OUT}/gene_function_screen.csv", index=False)
    multi = scr[scr.n_sloped_domains >= 2]
    print(f"   screened {len(scr)} genes (>= {MIN_COUNT} counts): "
          f"{len(multi)} sloped in >=2 domains, {int((scr.n_sloped_domains >= 3).sum())} in >=3, "
          f"{int((scr.n_sloped_domains >= 4).sum())} in all 4", flush=True)

    # ---- peaks / tiles: nearest TSS, and the per-gene candidate window ------------
    near_g, near_d = nearest_gene(pkn, cfg["gtf"])
    tch, tpo, tnm = all_tss(cfg["gtf"])
    pk = pd.Series(pkn).str.extract(r"(chr[\w]+)[:\-_](\d+)[\-_](\d+)")
    pk.columns = ["chr", "s", "e"]
    pmid = (pk["s"].astype(float) + pk["e"].astype(float)).values / 2
    pstart, pend = pk["s"].astype(float).values, pk["e"].astype(float).values
    pchr = pk["chr"].values
    iA, cA, kA = gaston_bins(dA, NB_ATAC)
    expA = np.array([alib[iA == b].sum() for b in kA])

    rows = []
    for gene in multi.gene.tolist()[:n_panels]:
        gi = int(np.where(vn == gene)[0][0])
        hit = np.where(tnm == gene)[0]
        if not len(hit):
            print(f"   {gene}: no TSS in the annotation -- skipped", flush=True); continue
        ch0, t0 = tch[hit[0]], tpo[hit[0]]
        cj = np.where((pchr == ch0) & (np.abs(pmid - t0) <= WIN))[0]
        bic = {}
        for j_ in cj:
            c_ = np.asarray(A_[:, j_].todense()).ravel().astype(float)
            if c_.sum() < 1:
                continue
            # classify on the distance to the NEAREST EDGE, not the midpoint: a 5 kb
            # tile that CONTAINS the TSS is a promoter tile even though its midpoint
            # can sit 2.5 kb away
            edge = 0.0 if pstart[j_] <= t0 <= pend[j_] else (
                pstart[j_] - t0 if t0 < pstart[j_] else pend[j_] - t0)
            cls = peak_class(edge)
            f = fit_free(np.array([c_[iA == b].sum() for b in kA]), expA, cA)
            if cls not in bic or f["R2"] > bic[cls][0]:
                bic[cls] = (f["R2"], j_, int(c_.sum()), float(pmid[j_] - t0))
        cg = np.asarray(X[:, gi].todense()).ravel().astype(float)
        gfit = pw_linear_fit(np.c_[cg, np.maximum(lib - cg, 0)], labR, dR, None, None,
                             idx_kept=np.array([0]), umi_threshold=0, t=T_LRT)["all_cell_types"]
        sel_j = [bic[c][1] for c in CLASSES if c in bic]
        fitP = None
        if sel_j:
            Cs = np.asarray(A_[:, sel_j].todense()).astype(float)
            fitP = pw_linear_fit(np.c_[Cs, np.maximum(alib - Cs.sum(1), 0)], labA, dA, None, None,
                                 idx_kept=np.arange(len(sel_j)), umi_threshold=0, t=T_LRT)["all_cell_types"]
        nsl = int((np.abs(gfit[0]) > 1e-9).sum())

        fig, AX = plt.subplots(2, 5, figsize=(18.5, 6.8))
        panel_space(AX[0, 0], xy, dR, "rho_RNA (Saturn isodepth)", "coolwarm", contour_d=dR, bnds=bR)
        panel_space(AX[1, 0], xy, dA, "rho_ATAC (Saturn isodepth)", "coolwarm", contour_d=dA, bnds=bA)
        panel_space(AX[0, 1], xy, labR, f"RNA domains (K={kdom})", None, contour_d=dR, bnds=bR, discrete=True)
        panel_space(AX[1, 1], xy, labA, f"ATAC domains (K={kdom})", None, contour_d=dA, bnds=bA, discrete=True)
        panel_space(AX[0, 2], xy, L(cg / lib), f"{gene} RAW expression", "magma", contour_d=dR, bnds=bR)
        panel_space(AX[0, 3], xy, L(np.exp(gfit[1][0, labR] + gfit[0][0, labR] * dR)),
                    f"{gene} FITTED expression", "magma", contour_d=dR, bnds=bR)
        panel_function(AX[0, 4], cg, lib, dR, labR, NB_RNA, "isodepth d_RNA", "log(expr x 1e6 + 1)",
                       f"{gene} expression ({int(cg.sum())} UMI), sloped in {nsl}/{labR.max() + 1} domains",
                       gfit=gfit, grow=0)
        for i, cls in enumerate(CLASSES):
            a_ = AX[1, 2 + i]
            if cls not in bic:
                a_.axis("off"); a_.set_title(f"no {cls} within +-{WIN // 1000} kb"); continue
            r2p, j_, cnt, dist = bic[cls]
            c_ = np.asarray(A_[:, j_].todense()).ravel().astype(float)
            panel_function(a_, c_, alib, dA, labA, NB_ATAC, "isodepth d_ATAC", "log(access x 1e6 + 1)",
                           f"{cls} {dist / 1000:+.1f} kb | {pkn[j_]}\n{cnt} counts, global R2 {r2p:.2f}",
                           gfit=fitP, grow=sel_j.index(j_))
            rows.append(dict(dataset=dataset, gene=gene, n_sloped_domains=nsl, cls=cls, peak=pkn[j_],
                             kb=round(dist / 1000, 1), counts=cnt, R2=round(r2p, 3),
                             slopes=[round(float(fitP[0][sel_j.index(j_), d_]), 2)
                                     for d_ in range(labA.max() + 1)]))
        fig.suptitle(f"{dataset} {gene}: multi-domain expression function (sloped in {nsl} domains) "
                     f"and the best promoter / proximal / long-distance peak", fontsize=9)
        fig.tight_layout()
        for e in ("png", "svg"):
            fig.savefig(f"{OUT}/{gene}_function.{e}", dpi=300 if e == "png" else None)
        plt.close(fig)
        print(f"   wrote {gene}_function ({nsl} sloped domains, R2 {float(multi.set_index('gene').R2[gene]):.2f})",
              flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/panel_peaks.csv", index=False)
    print("written", OUT)
    return scr


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
