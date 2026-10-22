"""TF activity as a function of a Saturn axis (paper Fig 2 f-g): chromVAR on isodepth bins and per spot.

SEACells' recipe is aggregate-then-chromVAR. Here the aggregation unit is a bin of the isodepth:
spots ordered by rho, cut into equal-count bins, ATAC summed per bin, depth-normalised, log; a chromVAR-style
deviation z per bin per TF against GC- and accessibility-matched background peak sets; then z is fitted as a
per-domain piecewise-linear function of rho (Gaussian fitter, LRT gate).

  scan_motifs        max PWM log-odds score per peak (both strands, vectorised one-hot x PWM); a peak is a hit
                     for a TF if its score is in the top (1 - q) of all peaks (q = 0.99)
  jaspar_pwms        log-odds PWMs from JASPAR2022 (pyjaspar)
  peak_sequences / peak_gc   sequences and GC content from a FASTA (pyfaidx)
  background_sets    chromVAR's matched background: per motif-hit peak, peaks of similar GC AND accessibility
                     (uniform random sets put a constant negative offset on every z)
  deviations_matched z of the motif set vs its matched sets
  deviations_fast    per-spot z with ONE permutation set shared by all TFs
  rho_binned_activity   the whole rho-bin pipeline for one axis -> fits + per-bin table
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import piecewise as pw

NBIN, QTHR, NBG = 60, 0.99, 50
B2I = {c: i for i, c in enumerate("ACGT")}


def onehot(seqs, W):
    X = np.zeros((len(seqs), W, 4), np.float32)
    for i, s in enumerate(seqs):
        a = np.frombuffer(s[:W].encode(), np.uint8)
        for c, j in B2I.items():
            X[i, :len(a), j] = (a == ord(c))
    return X


def scan(X, pssm):
    """max PWM score per sequence over both strands"""
    L = pssm.shape[0]; W = X.shape[1]
    if W <= L:
        return np.full(X.shape[0], -np.inf, np.float32)
    best = np.full(X.shape[0], -np.inf, np.float32)
    for mat in (pssm, pssm[::-1, ::-1]):
        sc = np.zeros((X.shape[0], W - L + 1), np.float32)
        for l in range(L):
            sc += X[:, l:l + W - L + 1, :] @ mat[l]
        best = np.maximum(best, sc.max(1))
    return best


def jaspar_pwms(tfs, release="JASPAR2022"):
    """{tf: (L, 4) log-odds PWM, pseudocount 0.5}; tries the name as given, then Title case"""
    from pyjaspar import jaspardb
    jdb = jaspardb(release=release); pwm = {}
    for t in tfs:
        m = jdb.fetch_motifs_by_name(t) or jdb.fetch_motifs_by_name(t.title())
        if m:
            mo = m[0]; mo.pseudocounts = 0.5
            pwm[t] = np.array([[mo.pwm.log_odds()[b][i] for b in "ACGT"] for i in range(mo.length)], np.float32)
    return pwm


def peak_sequences(fasta, peak_names, idx, min_len=30):
    """upper-case sequences (N -> A) of peaks idx ('chr:start-end'); returns (seqs, kept idx)"""
    from pyfaidx import Fasta
    fa = Fasta(fasta); ok = set(fa.keys()); seqs, valid = [], []
    for i in idx:
        c, s_ = peak_names[i].split(":"); s0, e0 = (int(v) for v in s_.split("-"))
        if c not in ok:
            continue
        try:
            s = str(fa[c][max(0, s0 - 1):e0]).upper().replace("N", "A")
        except Exception:
            continue
        if len(s) < min_len:
            continue
        seqs.append(s); valid.append(i)
    return seqs, np.array(valid)


def scan_motifs(seqs, pwm: dict, q=QTHR, chunk=1000):
    """(peaks x TFs) bool hits and the TF order. Scan width = 95th pct of lengths, capped at 5 kb."""
    W = min(int(np.percentile([len(s) for s in seqs], 95)), 5000)
    tfn = list(pwm); sc = np.full((len(seqs), len(tfn)), -np.inf, np.float32)
    for a in range(0, len(seqs), chunk):
        X = onehot(seqs[a:a + chunk], W)
        for j, t in enumerate(tfn):
            sc[a:a + chunk, j] = scan(X, pwm[t])
    H = np.zeros(sc.shape, bool)
    for j in range(len(tfn)):
        v = sc[:, j][np.isfinite(sc[:, j])]
        if len(v) > 100:
            H[:, j] = sc[:, j] >= np.quantile(v, q)
    return H, tfn


def peak_gc(fasta, peak_names, valid):
    from pyfaidx import Fasta
    fa = Fasta(fasta); gc = np.full(len(valid), np.nan, np.float32)
    for n, i in enumerate(valid):
        c, s_ = peak_names[i].split(":"); s0, e0 = (int(v) for v in s_.split("-"))
        try:
            q = str(fa[c][max(0, s0 - 1):e0]).upper()
            if len(q):
                gc[n] = (q.count("G") + q.count("C")) / len(q)
        except Exception:
            pass
    gc[~np.isfinite(gc)] = np.nanmedian(gc)
    return gc


def background_sets(gc, acc, hits, nbg=NBG, K=100, seed=0):
    """{tf column: (nbg, n_hits) peak indices} matched on GC and log accessibility; None if < 20 hits"""
    from scipy.spatial import cKDTree
    zc = lambda v: (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)
    Fx = np.c_[zc(gc), zc(np.log1p(acc))]; tree = cKDTree(Fx); rng = np.random.default_rng(seed); out = {}
    for j in range(hits.shape[1]):
        h = np.where(hits[:, j])[0]
        if len(h) < 20:
            out[j] = None; continue
        _, nn = tree.query(Fx[h], k=min(K, len(Fx)))
        out[j] = nn[np.arange(len(h))[None, :].repeat(nbg, 0), rng.integers(0, nn.shape[1], (nbg, len(h)))]
    return out


def deviations_matched(M, hits, BG):
    """(rows x TFs) z of the motif-set mean against its matched sets. M rows = bins or spots (log accessibility)."""
    Z = np.full((M.shape[0], hits.shape[1]), np.nan, np.float32)
    for j, bg in BG.items():
        if bg is None:
            continue
        h = np.where(hits[:, j])[0]
        obs = M[:, h].mean(1)
        nul = np.column_stack([M[:, bg[b]].mean(1) for b in range(bg.shape[0])])
        Z[:, j] = (obs - nul.mean(1)) / (nul.std(1) + 1e-12)
    return Z


def deviations_fast(M, hits, nperm=100, seed=0):
    """chromVAR z with one permutation set shared by every TF (hit sets of equal size)"""
    rng = np.random.default_rng(seed); n, P = M.shape; S = M.sum(1); k = int(np.median(hits.sum(0)))
    Rm = np.zeros((P, nperm), np.float32)
    for p in range(nperm):
        Rm[rng.choice(P, k, replace=False), p] = 1.0
    PS = M @ Rm; perms = PS / k - (S[:, None] - PS) / (P - k)
    pm, ps = perms.mean(1), perms.std(1) + 1e-12
    Z = np.full((n, hits.shape[1]), np.nan, np.float32)
    for j in range(hits.shape[1]):
        h = hits[:, j]
        if h.sum() < 20 or (~h).sum() < 20:
            continue
        Z[:, j] = (M[:, h].mean(1) - M[:, ~h].mean(1) - pm) / ps
    return Z


def log_accessibility(A, dtype=np.float64):
    """per-row depth-normalised log1p (scaled to the median depth)"""
    A = np.asarray(A.todense() if hasattr(A, "todense") else A, float)
    dep = np.maximum(A.sum(1, keepdims=True), 1)
    return np.log1p(A / dep * np.median(dep)).astype(dtype)


def rho_bins(d, lab, n_bins=NBIN):
    """equal-count bins along d: (groups of spot indices, bin mean rho, bin majority domain)"""
    o = np.argsort(d); groups = np.array_split(o, n_bins)
    db = np.array([d[g].mean() for g in groups])
    lb = np.array([np.bincount(lab[g], minlength=lab.max() + 1).argmax() for g in groups])
    return groups, db, lb


def rho_binned_activity(A, d, lab, hits, tfs, BG, n_bins=NBIN, t=pw.T_LRT):
    """chromVAR z per rho bin for each TF, fitted per domain. A: spots x motif-scanned peaks counts.
    Returns (fits DataFrame: tf, R2, n_sloped, slopes, z_range; bins DataFrame; groups)."""
    groups, db, lb = rho_bins(d, lab, n_bins)
    Ag = np.vstack([np.asarray(A[g].sum(0)).ravel() for g in groups])
    Z = deviations_matched(log_accessibility(Ag), hits, BG)
    sl, it = pw.fit_gauss(Z, lb, db, t=t)
    fits, bins = [], []
    for j, tf in enumerate(tfs):
        y = Z[:, j]
        if not np.isfinite(y).all():
            continue
        pred = it[j][lb] + sl[j][lb] * db
        r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
        fits.append(dict(tf=tf, R2=round(float(r2), 3), n_sloped=int((np.abs(sl[j]) > 1e-12).sum()),
                         slopes=[round(float(s), 3) for s in sl[j]], z_range=round(float(np.ptp(y)), 2)))
        for b in range(len(groups)):
            bins.append(dict(tf=tf, bin=b, rho=db[b], domain=int(lb[b]), z=y[b], fitted=pred[b]))
    return pd.DataFrame(fits), pd.DataFrame(bins), groups
