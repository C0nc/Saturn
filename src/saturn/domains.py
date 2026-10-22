"""GASTON-style domains along one Saturn isodepth.

A domain is a contiguous interval of ONE axis, found by GASTON's dynamic-programming
segmented regression of that modality's targets on its isodepth (each modality is segmented
on its own axis with its own targets). Domains belong to the axis: every feature read off that
axis shares its breakpoints.

Needs GASTON (`pip install gaston-spatial`).
"""
from __future__ import annotations
import numpy as np

KDOM = 4             # domains per axis (published fits)
NBUCKET = 50         # isodepth buckets for the DP (GASTON default; 20 for sparse ATAC panels)
MIN_DOM_FRAC = 0.01  # domains smaller than this share of spots are DP tail artifacts -> merged


def gaston_domains(d, Y, n_domains: int = KDOM, n_buckets: int = NBUCKET, min_frac: float = MIN_DOM_FRAC,
                   verbose: bool = False) -> np.ndarray:
    """Integer domain label per spot (0 = low end of the axis).

    d (N,) isodepth of one modality; Y (N, K) that modality's Saturn targets."""
    from gaston.dp_related import dp_bucketized, find_segments_from_dp
    d = np.asarray(d, float)
    ends = np.linspace(d.min(), d.max() + 0.01, n_buckets + 1)
    err, segmap = dp_bucketized(np.asarray(Y).T, ends, n_domains, xcoords=d)
    segs = find_segments_from_dp(err, segmap, n_domains)
    binlab = np.digitize(d, ends)
    lab = np.zeros(len(d), int)
    for c, seg in enumerate(segs):
        for s in seg:
            lab[binlab == s + 1] = c
    # merge tail-artifact domains (e.g. LPC21 ATAC put 4 of 3157 spots in one domain)
    while True:
        cnt = np.bincount(lab, minlength=lab.max() + 1)
        small = [k for k in range(len(cnt)) if 0 < cnt[k] < min_frac * len(d)]
        if not small:
            break
        k = small[0]; nbr = k - 1 if k > 0 else k + 1
        if verbose:
            print(f"merging domain {k} ({cnt[k]} spots) into {nbr}", flush=True)
        lab[lab == k] = nbr
        _, lab = np.unique(lab, return_inverse=True)
    return lab


def boundaries(d, lab) -> list:
    """isodepth values where the domain label changes (for contour lines on maps)"""
    o = np.argsort(d); dd, ll = np.asarray(d)[o], np.asarray(lab)[o]
    return [0.5 * (dd[i] + dd[i + 1]) for i in range(len(dd) - 1) if ll[i] != ll[i + 1]]


def isodepth_bins(d, n_bins: int):
    """equal-width bins along the isodepth, as gaston.bin_data.
    Returns (bin index per spot, centres of non-empty bins, ids of non-empty bins)."""
    d = np.asarray(d, float)
    edges = np.linspace(d.min() - 1e-6, d.max() + 1e-6, n_bins + 1)
    idx = np.clip(np.digitize(d, edges) - 1, 0, n_bins - 1)
    keep = np.array([(idx == b).any() for b in range(n_bins)])
    return idx, (0.5 * (edges[:-1] + edges[1:]))[keep], np.where(keep)[0]
