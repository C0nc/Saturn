"""A/B compartment <-> gene expression Granger causality in ISODEPTH space.

granger_ab_gene.ipynb ran this along DPT pseudotime: 50 equal-count windows, a
gene-set score per compartment bin, `grangercausalitytests` at lags 1-5, min p over
lags, p < 0.05.  This is the same biological question asked on the Saturn axes, with
the statistics the rest of this pipeline uses (saturn_granger_rho.py):

  for each axis (rho_RNA, rho_HiC) and each GASTON domain of that axis
    * order spots by rho, cut into NBIN equal-count bins -> a series indexed by rho
    * per compartment bin: mean A/B z-score, and the mean log-normalised expression
      of the >=3 protein-coding genes overlapping it (gene body U TSS+-2kb)
    * DETREND both series against rho inside the domain
    * F-test comp->gene and gene->comp at lags {1,2,3}; min p over lags,
      Bonferroni x (2 directions x n_lags); BH across bins within (series, domain)
    * a circular-shift null on the same pairs calibrates the whole procedure

Differences from the notebook that matter: the notebook takes min p over 5 lags with
no correction and no FDR, so its "p<0.05" counts are not calibrated.  The raw
(trended) series is kept here so the trend inflation stays visible.

A/B is read the notebook's way: CSV reindexed onto the fit's barcode order, bins
positional against scCompartment.hdf5['compartment_zscore']['bin'].

Usage: python scripts/mel_granger_ab.py [dataset] [nbin]
"""
import os, sys
from collections import defaultdict
import numpy as np, pandas as pd, scipy.sparse as sp, h5py
sys.path.insert(0, "<workspace>/Hole/gaston_mix_mm/scripts")
from prep_real2 import H
from lpc_gaston_fig4l import gaston_domains
from domain_annot import R
from saturn_granger_rho import pair_p, bh, detrend, lognorm
from continuous_marker_panels import hic_features_mel, mel_rna

KDOM, NBIN, MIN_GENES, PROMOTER = 4, 30, 3, 2000
BED = f"{H}/protein_coding_genes_human.bed"


def bin_gene_map(names, genes):
    """compartment bin -> indices of the genes overlapping it (body U promoter)"""
    bed = pd.read_csv(BED, sep="\t", header=None,
                      names=["chrom", "start", "end", "gene", "score", "strand"])
    gidx = {g: i for i, g in enumerate(genes)}
    bed = bed[bed.gene.isin(gidx)]
    bed["tss"] = np.where(bed.strand == "+", bed.start, bed.end)
    by_chrom = defaultdict(list)
    for r_ in bed.itertuples():
        by_chrom[r_.chrom].append((r_.start, r_.end, r_.tss - PROMOTER, r_.tss + PROMOTER,
                                   gidx[r_.gene], r_.gene))
    out, gnames = {}, {}
    for j, nm in enumerate(names):
        c_, se = nm.split(":"); s_, e_ = (int(v) for v in se.split("-"))
        hit = {(i, g) for gs, ge, ps, pe, i, g in by_chrom.get(c_, [])
               if (gs < e_ and ge > s_) or (ps < e_ and pe > s_)}
        if len(hit) >= MIN_GENES:
            out[j] = sorted(i for i, _ in hit); gnames[j] = sorted(g for _, g in hit)
    return out, gnames


def run(ds="mel_hic_fh", nbin=NBIN, seed=0, whole=False):
    """whole=True: one series over the entire axis (the notebook's setup, which does
    not split by domain) so the per-domain null is not just a power problem."""
    OUT = f"{R}/{ds}_granger"; os.makedirs(OUT, exist_ok=True)
    ax = pd.read_csv(f"{R}/{ds}/axes.csv", index_col=0); bc = [str(b) for b in ax.index]
    z = dict(np.load(f"{R}/targets_{ds}.npz", allow_pickle=True))

    V, names, _ = hic_features_mel(bc)                       # spots x 5359 A/B bins
    X, genes = mel_rna(bc)
    Xl = lognorm(sp.csr_matrix(X)).tocsc()
    b2g, b2n = bin_gene_map(names, genes)
    keep = sorted(b2g)
    print(f"{ds}: {len(keep)}/{len(names)} compartment bins carry >={MIN_GENES} expressed "
          f"protein-coding genes", flush=True)
    S = np.zeros((len(bc), len(keep)), np.float32)           # gene-set score per bin
    for j, b_ in enumerate(keep):
        S[:, j] = np.asarray(Xl[:, b2g[b_]].mean(1)).ravel()
    A = V[:, keep]

    rng = np.random.default_rng(seed)
    rows = []
    for axis in ["RNA", "HiC"]:
        d = ax[f"d_{axis}"].values
        lab = (np.zeros(len(d), int) if whole else gaston_domains(d, z[f"Y_{axis}"], L_=KDOM))
        for k in range(1 if whole else KDOM):
            m = np.where(lab == k)[0]
            if len(m) < nbin * 3: continue
            o = m[np.argsort(d[m])]
            edges = np.array_split(o, nbin)
            rb = np.array([d[e].mean() for e in edges])
            Ab = np.vstack([A[e].mean(0) for e in edges])
            Sb = np.vstack([S[e].mean(0) for e in edges])
            Ad, Sd = detrend(Ab, rb), detrend(Sb, rb)
            n0 = len(rows)
            for tag, Aa, Ss in (("detrended", Ad, Sd), ("raw", Ab, Sb)):
                for j, b_ in enumerate(keep):
                    x, y = Aa[:, j], Ss[:, j]
                    if x.std() < 1e-9 or y.std() < 1e-9: continue
                    pf, pr, bl = pair_p(y, x)                 # comp->gene, gene->comp
                    p0f, p0r, _ = pair_p(y, np.roll(x, rng.integers(5, nbin - 5)))
                    c_, se = names[b_].split(":")
                    rows.append(dict(axis=axis, domain=k, series=tag, bin=names[b_],
                                     chrom=c_, start=int(se.split("-")[0]),
                                     n_genes=len(b2g[b_]), genes=",".join(b2n[b_][:8]),
                                     pcc=float(np.corrcoef(x, y)[0, 1]),
                                     p_comp2gene=pf, p_gene2comp=pr, best_lag=bl,
                                     p_null_c2g=p0f, p_null_g2c=p0r))
            print(f"   d_{axis} domain {k}: {len(m)} spots -> {nbin} bins, "
                  f"{(len(rows) - n0) // 2} compartment bins tested", flush=True)

    T = pd.DataFrame(rows)
    for c_ in ["p_comp2gene", "p_gene2comp", "p_null_c2g", "p_null_g2c"]:
        T["q" + c_[1:]] = np.nan
        for _, g_ in T.groupby(["axis", "series", "domain"]):
            T.loc[g_.index, "q" + c_[1:]] = bh(g_[c_].values)
    f = f"{OUT}/granger_ab_gene_rho_nb{nbin}" + ("_whole" if whole else "") + ".csv"
    T.to_csv(f, index=False)

    print(f"\n-> {f}  ({len(T)} rows)\n")
    D = T[T.series == "detrended"]
    rep = D.groupby(["axis", "domain"]).apply(lambda g: pd.Series({
        "n_bins": len(g),
        "c2g_p<.05": f"{(g.p_comp2gene < .05).mean():.1%}",
        "null_p<.05": f"{(g.p_null_c2g < .05).mean():.1%}",
        "c2g_q<.1": int((g.q_comp2gene < .1).sum()),
        "g2c_q<.1": int((g.q_gene2comp < .1).sum()),
        "null_q<.1": int((g.q_null_c2g < .1).sum())}))
    print(rep.to_string())
    print("\nraw (trended) series, same columns:")
    Rw = T[T.series == "raw"]
    print(Rw.groupby(["axis", "domain"]).apply(lambda g: pd.Series({
        "c2g_p<.05": f"{(g.p_comp2gene < .05).mean():.1%}",
        "null_p<.05": f"{(g.p_null_c2g < .05).mean():.1%}"})).to_string())
    return T


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "mel_hic_fh",
        int(sys.argv[2]) if len(sys.argv) > 2 else NBIN, whole="--whole" in sys.argv)
