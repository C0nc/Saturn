"""Annotate the domains of a Saturn axis and screen features that are functions of it.

GASTON annotates the domains its DP segmentation returns by (i) what each domain is made of
(cluster / cell-type composition, the anatomy it lands on), (ii) the markers specific to it,
and (iii) how many features vary CONTINUOUSLY inside it (a per-domain slope that survives the
likelihood-ratio gate) rather than only jumping at the boundary. This module does all three for
any modality:

  markers         counts: Poisson enrichment z, one-vs-rest  (`poisson_markers`)
                  continuous: Welch t, one-vs-rest            (`continuous_markers`)
  genomic context ATAC peaks / tiles / Hi-C bins -> nearest TSS (`nearest_gene`, `peak_class`)
                  and peaks summed into their nearest gene      (`peaks_to_genes`)
  gradient        per-domain sloped features                    (`gradient_count`)
  screen          features sloped in >= 2 domains, ranked by binned R2 (`screen`)
  table           everything above per domain, plus n spots, isodepth range and MEDIAN DEPTH
                  (`domain_table`). Median depth per domain is the cheapest check that a domain
                  is biology rather than coverage.
"""
from __future__ import annotations
import json, re
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import piecewise as pw

PROM, PROXIMAL = 2_000, 20_000   # promoter |d| <= 2 kb; proximal enhancer <= 20 kb; else long-distance


# ------------------------------------------------------------------ markers
def poisson_markers(Mat, lib, lab, feats, k, n=8, min_count=100, min_lfc=0.5):
    """one-vs-rest markers of domain k for a count modality, ranked by Poisson enrichment
    z = (obs - exp) / sqrt(exp), exp = total counts x (domain library / total library).

    Log2 fold change alone puts sparse predicted genes (Gm*, RP11-*) on top; z with a count floor
    and a real fold change does not. Returns [(feature, log2FC, total count)]."""
    ina = np.asarray(lab) == k
    tot = np.asarray(Mat.sum(0)).ravel().astype(float)   # uint32 would wrap under -tot
    obs = np.asarray(Mat[ina].sum(0)).ravel()
    exp = tot * (lib[ina].sum() / max(lib.sum(), 1))
    a = obs / max(lib[ina].sum(), 1)
    b = (tot - obs) / max(lib[~ina].sum(), 1)
    lfc = np.log2((a + 1e-9) / (b + 1e-9))
    z = (obs - exp) / np.sqrt(np.maximum(exp, 1e-9))
    z[(tot < min_count) | (lfc < min_lfc)] = -np.inf
    o = np.argsort(-z)[:n]
    return [(feats[i], round(float(lfc[i]), 2), int(tot[i])) for i in o]


def continuous_markers(V, lab, names, k, n=8, min_var=0.0):
    """one-vs-rest markers of a continuous modality: Welch t on the domain contrast.
    Returns [(feature, mean difference, t)] ranked by |t|."""
    ina = np.asarray(lab) == k
    a, b = V[ina], V[~ina]
    diff = a.mean(0) - b.mean(0)
    se = np.sqrt(a.var(0, ddof=1) / max(len(a), 2) + b.var(0, ddof=1) / max(len(b), 2))
    t = diff / np.maximum(se, 1e-9)
    t[V.var(0) <= min_var] = 0
    o = np.argsort(-np.abs(t))[:n]
    return [(names[i], round(float(diff[i]), 2), round(float(t[i]), 1)) for i in o]


# ------------------------------------------------------------------ genomic context
def all_tss(gtf):
    """chrom, TSS, gene name for every `gene` record of a GTF"""
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


def gene_lengths(gtf) -> dict:
    """gene body length (bp) per gene name, first record wins"""
    L = {}
    for line in open(gtf):
        if line[0] == "#":
            continue
        p = line.split("\t", 9)
        if p[2] != "gene":
            continue
        m = re.search(r'gene_name "([^"]+)"', p[8])
        if m and m.group(1) not in L:
            L[m.group(1)] = int(p[4]) - int(p[3])
    return L


def parse_regions(names):
    """'chr1:100-200' / 'chr1-100-200' / 'chr1_100_200' -> DataFrame chr, start, end"""
    pk = pd.Series([str(v) for v in names]).str.extract(r"(chr[\w]+)[:\-_](\d+)[\-_](\d+)")
    pk.columns = ["chr", "start", "end"]
    pk[["start", "end"]] = pk[["start", "end"]].astype(float)
    return pk


def nearest_gene(region_names, gtf):
    """every peak / tile / bin -> (nearest gene, signed distance from its midpoint to the TSS)"""
    tch, tpo, tnm = all_tss(gtf)
    pk = parse_regions(region_names)
    mid = ((pk["start"] + pk["end"]) / 2).values
    g = np.full(len(pk), "NA", dtype=object); dd = np.full(len(pk), np.nan)
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


def peak_class(dist, half_width: float = 0.0):
    """promoter / proximal enhancer / long-distance enhancer. Pass the region's half width so the
    distance is measured to the region EDGE: a 5 kb tile containing the TSS has its midpoint up to
    2.5 kb away and would otherwise fail a 2 kb promoter cutoff."""
    ad = max(abs(dist) - half_width, 0)
    return "promoter" if ad <= PROM else ("proximal enhancer" if ad <= PROXIMAL else "long-distance enhancer")


def peaks_to_genes(A, near_g):
    """sum every peak's counts into its nearest gene -> (spots x genes csr, gene names)"""
    near_g = np.asarray(near_g)
    names = pd.Index(sorted(set(near_g[near_g != "NA"])))
    col = names.get_indexer(near_g); keep = col >= 0
    P = sp.csr_matrix((np.ones(keep.sum()), (np.arange(keep.sum()), col[keep])), shape=(int(keep.sum()), len(names)))
    return sp.csr_matrix(sp.csr_matrix(A)[:, np.where(keep)[0]] @ P), np.array(names)


# ------------------------------------------------------------------ gradients and screens
def gradient_count(Mat, lib, lab, d, n_top=500, t=pw.T_LRT):
    """per domain: how many of the n_top most-counted features keep a slope (Poisson, LRT gate)"""
    tot = np.asarray(Mat.sum(0)).ravel().astype(float)
    keep = np.argsort(-tot)[:n_top]
    slope, _ = pw.fit_poisson(sp.csr_matrix(Mat)[:, keep], lib, lab, d, t=t)
    return (np.abs(slope) > 1e-9).sum(0)


def screen(values, names, lab, d, kind="poisson", lib=None, trials=None, n_top=1500, min_count=100,
           n_bins=25, t=pw.T_LRT):
    """Which features are functions of the axis. Fits every candidate per domain with the
    modality's likelihood, then ranks by (# sloped domains, binned R2).

    kind  poisson   values = counts (spots x features), lib = library per spot
          gauss     values = continuous matrix
          binomial  values = successes, trials = denominators (spots x features or spots)
    Keep features with n_sloped_domains >= 2 (a real multi-domain function) for panels, and
    report N(>=2) / N screened with every figure."""
    names = np.asarray(names)
    if kind == "poisson":
        tot = np.asarray(values.sum(0)).ravel().astype(float)
        cand = np.argsort(-tot)[:n_top]; cand = cand[tot[cand] >= min_count]
        C = np.asarray(sp.csr_matrix(values)[:, cand].todense(), float)
        slope, inter = pw.fit_poisson(C, lib, lab, d, t=t)
        r2 = [pw.binned_r2(C[:, i], d, lab, slope[i], inter[i], n_bins, lib=lib, link="log")[0] for i in range(len(cand))]
        extra = dict(counts=tot[cand].astype(int))
    elif kind == "gauss":
        V = np.asarray(values, float)
        cand = np.argsort(-V.var(0))[:min(n_top, V.shape[1])]
        slope, inter = pw.fit_gauss(V[:, cand], lab, d, t=t)
        r2 = [pw.binned_r2(V[:, j], d, lab, slope[i], inter[i], n_bins)[0] for i, j in enumerate(cand)]
        extra = dict(variance=np.round(V[:, cand].var(0), 4))
    elif kind == "binomial":
        S = np.asarray(values, float)
        Nt = np.asarray(trials, float)
        Nt = np.repeat(Nt[:, None], S.shape[1], 1) if Nt.ndim == 1 else Nt
        cand = np.argsort(-S.sum(0))[:min(n_top, S.shape[1])]
        slope, inter = pw.fit_binomial(S[:, cand], Nt[:, cand], lab, d, t=t)
        r2 = []
        for i, j in enumerate(cand):
            r2.append(pw.binned_r2(S[:, j], d, lab, slope[i], inter[i], n_bins, lib=Nt[:, j], link="logit")[0])
        extra = dict(successes=S[:, cand].sum(0).astype(int))
    else:
        raise ValueError(kind)
    df = pd.DataFrame(dict(feature=names[cand], **extra, n_sloped_domains=pw.n_sloped(slope),
                           R2=np.round(r2, 3), slopes=[np.round(s, 3).tolist() for s in slope]))
    df["_i"] = cand
    return df.sort_values(["n_sloped_domains", "R2"], ascending=False).reset_index(drop=True), slope, inter, cand


def domain_table(lab, d, modality, depth=None, cats=None, conts=None, markers=None, gradient=None, labels=None):
    """One row per domain: n spots, isodepth range, median depth, composition of each categorical
    annotation (JSON of fractions), mean of each continuous annotation, sloped-feature count and
    marker string. cats / conts: dict name -> per-spot array. markers: dict domain -> str."""
    rows = []
    for k in range(int(lab.max()) + 1):
        ina = lab == k
        row = dict(modality=modality, domain=k, label=(labels or {}).get(k, ""), n_spots=int(ina.sum()),
                   isodepth=f"[{d[ina].min():.2f}, {d[ina].max():.2f}]")
        if depth is not None:
            row["median_depth"] = float(np.median(np.asarray(depth, float)[ina]))
        for c, v in (cats or {}).items():
            v = np.asarray(v).astype(str)
            row[c] = json.dumps({n: round(float((v[ina] == n).mean()), 2) for n in sorted(set(v.tolist()))})
        for c, v in (conts or {}).items():
            row[c] = round(float(np.nanmean(np.asarray(v, float)[ina])), 3)
        if gradient is not None:
            row["gradient_feats"] = int(gradient[k])
        if markers is not None:
            row["markers"] = markers.get(k, "")
        rows.append(row)
    return pd.DataFrame(rows)


def annotate_counts(Mat, feats, lab, d, modality, depth=None, cats=None, conts=None, n_top_gradient=500,
                    gtf=None, labels=None):
    """Domain table for a count modality (RNA genes or ATAC peaks/tiles). With `gtf`, ATAC markers are
    written as nearest-gene(class, kb to TSS, log2FC) and a gene-level marker column is added."""
    Mat = sp.csr_matrix(Mat); feats = np.asarray(feats); lib = np.asarray(Mat.sum(1)).ravel().clip(1)
    grad = gradient_count(Mat, lib, lab, d, n_top=n_top_gradient)
    mk, gl = {}, {}
    if gtf is not None:
        near_g, near_d = nearest_gene(feats, gtf)
        pos = {p: i for i, p in enumerate(feats)}
        Ag, gname = peaks_to_genes(Mat, near_g)
    for k in range(int(lab.max()) + 1):
        m = poisson_markers(Mat, lib, lab, feats, k)
        if gtf is None:
            mk[k] = "; ".join(f"{g}({lf:+.1f})" for g, lf, _ in m)
        else:
            mk[k] = "; ".join(f"{near_g[pos[p]]}({peak_class(near_d[pos[p]])[0]}{near_d[pos[p]] / 1000:+.0f}kb, {lf:+.1f})"
                              for p, lf, _ in m)
            gl[k] = "; ".join(f"{g}({lf:+.1f})" for g, lf, _ in poisson_markers(Ag, lib, lab, gname, k, min_count=300))
    T = domain_table(lab, d, modality, depth if depth is not None else lib, cats, conts, mk, grad, labels)
    if gl:
        T["gene_level"] = [gl[k] for k in T.domain]
    return T


def annotate_continuous(V, names, lab, d, modality, depth=None, cats=None, conts=None, n_top=4000, labels=None):
    """Domain table for a continuous modality (methylation, Hi-C, protein CLR ...): Welch-t markers and
    Gaussian per-domain sloped-feature counts over the n_top most variable features."""
    V = np.asarray(V, float); names = np.asarray(names)
    sel = np.argsort(-V.var(0))[:min(n_top, V.shape[1])]
    slope, _ = pw.fit_gauss(V[:, sel], lab, d)
    grad = (np.abs(slope) > 1e-12).sum(0)
    mk = {k: "; ".join(f"{f} ({df:+.2f}, t{t:+.0f})" for f, df, t in continuous_markers(V[:, sel], lab, names[sel], k))
          for k in range(int(lab.max()) + 1)}
    return domain_table(lab, d, modality, depth, cats, conts, mk, grad, labels)
