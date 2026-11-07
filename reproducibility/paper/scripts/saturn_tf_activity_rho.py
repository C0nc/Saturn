"""Isodepth TF activity: SEACells metacells replaced by rho bins of the Saturn fit.

SEACells' recipe is aggregate-then-chromVAR: build metacells, sum ATAC inside each,
score JASPAR motifs against peaks, and take a chromVAR-style deviation z per metacell.
Here the aggregation unit is a bin of the Saturn isodepth instead of a SEACells metacell,
so TF activity becomes a function of rho rather than a per-metacell value:

  for each axis (rho_RNA, rho_ATAC)
    * order spots by rho, cut into equal-count bins   <- the "isodepth metacells"
    * sum ATAC inside each bin, depth-normalise, log
    * chromVAR-style deviation z per bin per TF (motif-hit peaks vs matched random sets)
    * fit z as a piecewise linear function of rho with GASTON's domains, per domain

Motif scanning is vectorised (one-hot x PWM, both strands) so it also handles the 5 kb
tiles of the melanoma fit, not only narrow called peaks. The scan is cached per dataset
and reused across both axes.

Usage: python scripts/saturn_tf_activity_rho.py <dataset> [nbin]
"""
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, "<workspace>/Hole/gaston_mix_mm/scripts")
sys.path.insert(0, "<workspace>/saturn_repo/src")
from prep_real2 import H
from lpc_gaston_fig4l import gaston_domains
from domain_k import kdom
from domain_annot import CFG, R
from piecewise import fit_gauss
from saturn import tfactivity as TA

NBIN, KDOM, T_LRT, TOPK, NPERM, QTHR = 60, 4, 0.1, 12000, 50, 0.99
FASTA = {"lpc10_fov": f"{H}/scdori_run/mouse_genome/mm10.fa",
         "melanoma_whole": f"{H}/scdori_run/human_genome/hg38.fa",
         "melanoma_tissue": f"{H}/scdori_run/human_genome/hg38.fa"}
PANEL = {
    "lpc10_fov": {
        "OL":        ["SOX10", "OLIG1", "OLIG2", "MYRF", "NKX2-2", "ZEB2"],
        "astrocyte": ["SOX9", "NFIA", "NFIB", "HES5", "LHX2", "ID3"],
        "microglia": ["IRF8", "MAFB", "MEF2C", "CEBPA", "SPI1", "KLF4"],
        "stress":    ["ATF4", "DDIT3", "STAT3", "NFKB1"],
        "neuron":    ["NEUROD1", "NEUROG2", "POU3F2", "POU3F3"]},
    "melanoma_whole": {
        "melanocytic": ["MITF", "SOX10", "TFAP2A", "PAX3"],
        "invasive":    ["JUN", "FOSL1", "TEAD1", "TEAD4", "ZEB1", "SNAI2"],
        "interferon":  ["IRF1", "IRF8", "STAT1", "STAT2", "NFKB1", "RELA"],
        "myeloid":     ["SPI1", "CEBPA", "CEBPB", "MAFB"],
        "lymphoid":    ["TCF7", "LEF1", "EOMES", "TBX21"]},
}
PANEL["melanoma_tissue"] = PANEL["melanoma_whole"]
B2I = TA.B2I


onehot = TA.onehot


scan = TA.scan


def motif_hits(ds, peaks, keep):
    cache = f"{R}/{ds}_tf_activity/motif_hits.npz"
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if len(z["keep"]) == len(keep) and (z["keep"] == keep).all():
            print("   motif scan: cached", flush=True)
            return z["hits"], list(z["tfs"].astype(str)), z["valid"]
    tfs = [t for sub in PANEL[ds].values() for t in sub]
    pwm = TA.jaspar_pwms(tfs)
    print(f"   JASPAR: {len(pwm)}/{len(tfs)} PWMs", flush=True)
    seqs, valid = TA.peak_sequences(FASTA[ds], peaks, keep)
    print(f"   {len(seqs)} sequences, scan width {min(int(np.percentile([len(s) for s in seqs], 95)), 5000)}", flush=True)
    Hh, tfn = TA.scan_motifs(seqs, pwm, q=QTHR)
    os.makedirs(f"{R}/{ds}_tf_activity", exist_ok=True)
    np.savez_compressed(cache, hits=Hh, tfs=np.array(tfn, object),
                        valid=np.array(valid), keep=np.array(keep))
    return Hh, tfn, np.array(valid)


def peak_gc(ds, peaks, valid):
    """GC content of each retained peak, cached (needed for chromVAR's matched background)"""
    f = f"{R}/{ds}_tf_activity/peak_gc.npz"
    if os.path.exists(f):
        z = np.load(f)
        if len(z["valid"]) == len(valid) and (z["valid"] == valid).all():
            return z["gc"]
    gc = TA.peak_gc(FASTA[ds], peaks, valid)
    np.savez_compressed(f, gc=gc, valid=np.asarray(valid))
    return gc


background_sets = TA.background_sets


deviations_matched = TA.deviations_matched


def deviations(Ab, hits, seed=0):
    """chromVAR-style z: motif-hit mean minus background, vs matched random sets"""
    rng = np.random.default_rng(seed); n, P = Ab.shape
    Z = np.full((n, hits.shape[1]), np.nan)
    for j in range(hits.shape[1]):
        h = np.where(hits[:, j])[0]; o = np.where(~hits[:, j])[0]
        if len(h) < 20 or len(o) < 20: continue
        obs = Ab[:, h].mean(1) - Ab[:, o].mean(1)
        pm = np.empty((NPERM, n))
        for p in range(NPERM):
            r = rng.choice(P, len(h), replace=False)
            mask = np.ones(P, bool); mask[r] = False
            pm[p] = Ab[:, r].mean(1) - Ab[:, mask].mean(1)
        Z[:, j] = (obs - pm.mean(0)) / (pm.std(0) + 1e-12)
    return Z


def run(ds, nbin=NBIN):
    OUT = f"{R}/{ds}_tf_activity"; os.makedirs(OUT, exist_ok=True)
    cfg = CFG[ds]
    ax = pd.read_csv(f"{R}/{ds}/axes.csv", index_col=0); bc = [str(b) for b in ax.index]
    z = dict(np.load(f"{R}/targets_{ds}.npz", allow_pickle=True))
    atac = cfg["atac"](); rna = cfg["rna"]()
    ai = {str(b): i for i, b in enumerate(atac.obs_names)}
    ri = {str(b): i for i, b in enumerate(rna.obs_names)}
    A = sp.csr_matrix(atac.X)[[ai[b] for b in bc]]
    Rm = sp.csr_matrix(rna.X)[[ri[b] for b in bc]]
    peaks = np.array([str(p) for p in atac.var_names])
    genes = np.array([str(g) for g in rna.var_names])
    tot = np.asarray(A.sum(0)).ravel().astype(float)
    keep = np.argsort(-tot)[:TOPK]; keep.sort()
    hits, tfn, valid = motif_hits(ds, peaks, keep)
    Asub = A[:, valid]
    gc = peak_gc(ds, peaks, valid)
    acc = np.asarray(Asub.sum(0)).ravel().astype(float)
    BG = background_sets(gc, acc, hits)
    print(f"   GC/accessibility-matched background: {sum(v is not None for v in BG.values())}"
          f"/{len(BG)} TFs, 50 sets each", flush=True)

    allfit, allbin = [], []
    for axis in ["RNA", "ATAC"]:
        d = ax[f"d_{axis}"].values
        lab = gaston_domains(d, z[f"Y_{axis}"], L_=kdom(ds, axis))
        groups, db, lb = TA.rho_bins(d, lab, nbin)
        Ab = TA.log_accessibility(np.vstack([np.asarray(Asub[g].sum(0)).ravel() for g in groups]))
        Z = deviations_matched(Ab, hits, BG)
        sl, it = fit_gauss(Z, lb, db, t=T_LRT)
        for j, t in enumerate(tfn):
            y = Z[:, j]
            if not np.isfinite(y).all(): continue
            pred = it[j][lb] + sl[j][lb] * db
            r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
            gi = np.where(genes == t.capitalize())[0] if ds == "lpc10_fov" else np.where(genes == t)[0]
            expr = (np.array([np.asarray(Rm[g][:, gi].sum()).ravel()[0] for g in groups])
                    if len(gi) else np.full(nbin, np.nan))
            allfit.append(dict(dataset=ds, axis=axis, tf=t,
                               lineage=[k for k, v in PANEL[ds].items() if t in v][0],
                               R2=round(float(r2), 3),
                               n_sloped=int((np.abs(sl[j]) > 1e-12).sum()),
                               slopes=[round(float(s), 3) for s in sl[j]],
                               z_range=round(float(np.ptp(y)), 2)))
            for b in range(nbin):
                allbin.append(dict(axis=axis, tf=t, bin=b, rho=db[b], domain=int(lb[b]),
                                   z=y[b], fitted=pred[b], tf_expr=expr[b]))
        print(f"   {ds} d_{axis}: {nbin} isodepth bins, {len(tfn)} TFs fitted", flush=True)
    F = pd.DataFrame(allfit); B = pd.DataFrame(allbin)
    F.to_csv(f"{OUT}/tf_fits.csv", index=False); B.to_csv(f"{OUT}/tf_bins.csv", index=False)
    print(f"-> {OUT}/tf_fits.csv, tf_bins.csv")
    return F, B


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else NBIN)
