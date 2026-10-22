"""Co-fitted cis pairs: gene and a +-1 Mb peak that are BOTH functions of the same rho.

Three questions, in the order they can honestly be asked:

  1. CO-FIT   for each axis, which (gene, cis peak) pairs both have a slope surviving
              the LRT gate in the SAME domain? Only those are comparable at all.
  2. SHAPE    how similar are their fitted functions, in the function-space sense used
              for the Sprox-seq marker/complex matching (curves on a common rho grid,
              z-scored, Pearson)? Calibrated against DISTAL peaks of the same gene, so
              "similar" means more similar than a random peak on the same chromosome.
  3. SPEED    with both features z-scored, slope is SD per unit rho, so the ratio
              |slope_peak| / |slope_gene| inside the shared domain says whether the
              chromatin or the transcript moves faster along the axis.

  4. LAG      the shift d that best aligns the two fitted CURVES over the whole axis.
              This replaces Granger on fitted values, which is NOT valid: two noiseless
              piecewise-linear functions of the same rho are called Granger-causal for
              independent pairs 10.2% of the time (18.2% with 30% noise) against a 5%
              nominal rate -- measured, not assumed. The lag is a descriptive alignment,
              and it is identifiable ONLY because the curves have breakpoints; inside a
              single domain both are straight lines and any shift fits equally well.

Usage: python scripts/saturn_cis_function_pairs.py <dataset> [n_genes]
"""
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/saturn_repo/src")
from prep_real2 import H
from lpc_gaston_fig4l import gaston_domains
from domain_k import kdom
from domain_annot import CFG, R
from saturn import cis as SCIS
from saturn.mechanism import smooth as _smooth

WIN, KDOM, T_LRT, NGRID, MAXPK = SCIS.WIN, 4, 0.1, SCIS.NGRID, SCIS.MAXPK
MAXDIST, SIM_LAG = SCIS.MAXDIST, SCIS.SIM_LAG   # distal peaks per gene; only pairs this similar get a lag
DISTAL_MIN, KNN, MINCNT = SCIS.DISTAL_MIN, 15, 100
# top-N-by-count is dominated by MALAT1 / MT-RNR2 / lincRNAs, so programme genes are
# force-included: on melanoma only 2-3 of each programme reached the top 1500, which made
# the enrichment test vacuous rather than negative.
PANEL = {
  "lpc10_fov": ("Trem2 Tyrobp Csf1r Cx3cr1 P2ry12 Tmem119 Irf8 Spi1 C1qa C1qb C1qc Ctss Hexb Fcrls "
      "Inpp5d Mertk Lgals3bp Apoe Cst7 Itgax Cd68 Aif1 Lyz2 Mpeg1 Laptm5 Ptprc Cd9 Man2b1 Ctsb Ctsd "
      "Ctsl Gpnmb Fth1 Spp1 Marcks Lrmda Ly86 Cd83 Syk Plek Dock8 Zeb2 "
      "Mbp Plp1 Mog Mag Mobp Cnp Cldn11 Sox10 Olig1 Olig2 Mal Ugt8a Aspa Trf Apod Ptgds Opalin Ermn "
      "Gfap Aqp4 Slc1a2 Slc1a3 Aldh1l1 S100b Sox9 Nfia Clu Vim "
      "Snap25 Syt1 Rbfox3 Nrgn Camk2a Slc17a7 Gad1 Gad2 Syp Nefl").split(),
  "melanoma_tissue": ("MITF SOX10 PMEL MLANA TYR DCT TYRP1 PAX3 TFAP2A SOX9 EDNRB GPR143 RAB27A MYO5A "
      "SLC45A2 OCA2 KIT AXL WNT5A ZEB1 SERPINE1 TGFBI FN1 VIM SNAI2 TWIST1 LOXL2 PDGFRB EGFR NRP1 THBS1 "
      "PTPRC CD74 HLA-A HLA-B HLA-DRA HLA-DRB1 STAT1 IRF1 GBP1 GBP2 ISG15 MX1 B2M CXCL9 CXCL10 IFI6 "
      "IFI27 TAP1 PSMB9 CD68 CD3E LYZ C1QA C1QB KRT1 KRT5 KRT10 KRT14 KRT13 SPINK5 SFN S100A8 S100A9 "
      "LTF DSP COL1A1 COL1A2 COL3A1 DCN LUM POSTN FBN1 SPARC MGP TAGLN ACTA2").split()}
zz = SCIS.zz
curves = SCIS.curves
best_lag_many = SCIS.best_lag_many


def smooth(V, xy, k=KNN):
    return _smooth(V, xy, k)


def genes_and_peaks(ds, n_genes, cfg, bc):
    """count matrices on the fit's spots, the gene set (top n_genes by count + programme panel) and TSS table"""
    rna = cfg["rna"](); atac = cfg["atac"]()
    ri = {str(b): i for i, b in enumerate(rna.obs_names)}; ai = {str(b): i for i, b in enumerate(atac.obs_names)}
    Xr = sp.csr_matrix(rna.X)[[ri[b] for b in bc]]; Xa = sp.csr_matrix(atac.X)[[ai[b] for b in bc]]
    gn = np.array([str(g) for g in rna.var_names]); pn = np.array([str(p) for p in atac.var_names])
    tg = np.asarray(Xr.sum(0)).ravel().astype(float); tp = np.asarray(Xa.sum(0)).ravel().astype(float)
    kg = list(np.argsort(-tg)[:n_genes]); gidx = {g: i for i, g in enumerate(gn)}
    extra = [gidx[g] for g in PANEL.get(ds, []) if g in gidx and tg[gidx[g]] >= MINCNT and gidx[g] not in set(kg)]
    print(f"   genes: {n_genes} by count + {len(extra)} programme genes forced in", flush=True)
    return Xr, Xa, gn, pn, tp, np.array(kg + extra), SCIS.tss_table(cfg["gtf"])


def run(ds, n_genes=1500):
    OUT = f"{R}/{ds}_cis_pairs"; os.makedirs(OUT, exist_ok=True)
    cfg = CFG[ds]
    ax = pd.read_csv(f"{R}/{ds}/axes.csv", index_col=0); bc = [str(b) for b in ax.index]
    z = dict(np.load(f"{R}/targets_{ds}.npz", allow_pickle=True))
    xy = np.c_[ax.x.values, ax.y.values]
    Xr, Xa, gn, pn, tp, kg, tss = genes_and_peaks(ds, n_genes, cfg, bc)
    _, pm = SCIS.peak_positions(pn)
    cis, dist = SCIS.cis_pairs(kg, gn, pn, tp, tss, window=WIN, max_peaks=MAXPK, distal_min=DISTAL_MIN,
                               max_distal=MAXDIST)
    need = np.unique(np.concatenate([v for v in list(cis.values()) + list(dist.values()) if len(v)]))
    print(f"   {len(cis)} genes with cis peaks; {len(need)} distinct peaks to fit", flush=True)
    G = SCIS.smoothed_logcpm(Xr, kg, xy, KNN); P = SCIS.smoothed_logcpm(Xa, need, xy, KNN)
    gpos = {v: i for i, v in enumerate(kg)}; ppos = {v: i for i, v in enumerate(need)}

    out = []
    for axis in ["RNA", "ATAC"]:
        d = ax[f"d_{axis}"].values; lab = gaston_domains(d, z[f"Y_{axis}"], L_=kdom(ds, axis))
        pairs = [(gpos[gi], ppos[s_], dict(dataset=ds, axis=axis, kind=tag, gene=gn[gi], peak=pn[s_],
                                           kb=(pm[s_] - tss[gn[gi]][1]) / 1e3))
                 for gi, pk in cis.items() for tag, sel in (("cis", pk), ("distal", dist.get(gi, np.array([], int))))
                 for s_ in sel]
        T = SCIS.function_pairs(G, P, d, lab, pairs, t=T_LRT, n_grid=NGRID, sim_lag=SIM_LAG)
        out.append(T)
        print(f"   {ds} d_{axis}: {int((T.kind == 'cis').sum()) if len(T) else 0} co-fitted cis pairs", flush=True)
    T = pd.concat(out, ignore_index=True); T.to_csv(f"{OUT}/cis_function_pairs.csv", index=False)
    print(f"-> {OUT}/cis_function_pairs.csv ({len(T)} rows)")
    return T


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1500)
