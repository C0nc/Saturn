"""Per-figure inputs for supp_domains.py: each figure's own load() and domain names, so the supplementary pages show
exactly the domains, names and colours of the main figures. One function per figure returns a supp_domains.Spec."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp

H = "/mnt/HDD2B/Users/yancui"; PF = f"{H}/Saturn/published_figures"
R = Path(f"{H}/Hole/gaston_mix_mm/results/real_saturn")
for p in (f"{PF}/_style", f"{H}/saturn_repo/src", f"{PF}/fig2", f"{PF}/fig4", f"{H}/Hole/gaston_mix_mm/scripts"):
    sys.path.insert(0, p)
from supp_domains import Mod, Spec
from saturn.annotate import nearest_gene
GTF = dict(mouse=f"{H}/mouse_atlas/gencode.vM25.primary_chr.annotation.gtf", human=f"{H}/gencode.v21.annotation.gtf")


def _import(fig, mod):
    sys.path.insert(0, f"{PF}/{fig}")
    return __import__(mod)


def coding_tss(gtf):
    """chrom, TSS, name of every protein-coding gene (GTF gene_type / gene_biotype)"""
    import re
    ch, po, nm = [], [], []
    with open(gtf) as f:
        for line in f:
            if line[0] == "#":
                continue
            p = line.split("\t", 9)
            if p[2] != "gene" or not re.search(r'gene_(?:type|biotype) "protein_coding"', p[8]):
                continue
            m = re.search(r'gene_name "([^"]+)"', p[8])
            if m:
                ch.append(p[0]); po.append(int(p[3]) if p[6] == "+" else int(p[4])); nm.append(m.group(1))
    return np.array(ch), np.array(po, float), np.array(nm)


def region_labels(names, gtf):
    """peak / bin -> 'nearest protein-coding gene (+12 kb)'"""
    from saturn.annotate import parse_regions
    tch, tpo, tnm = coding_tss(gtf)
    pk = parse_regions(names); mid = ((pk["start"] + pk["end"]) / 2).values; out = {}
    for c_ in pd.unique(pk["chr"].dropna()):
        m = tch == c_; sel = np.where((pk["chr"] == c_).values)[0]
        if not m.any():
            continue
        o = np.argsort(tpo[m]); tp_, tn_ = tpo[m][o], tnm[m][o]
        j = np.clip(np.searchsorted(tp_, mid[sel]), 1, len(tp_) - 1)
        pick = np.where(np.abs(mid[sel] - tp_[j - 1]) <= np.abs(mid[sel] - tp_[j]), j - 1, j)
        for s_, p_ in zip(sel, pick):
            out[names[s_]] = f"{tn_[p_]} ({(mid[s_] - tp_[p_]) / 1e3:+.0f} kb)"
    return out


def named(feats):
    """RNA markers: skip unnamed / mitochondrial / ribosomal / predicted genes and MALAT1 (no annotation value)"""
    return ~pd.Series(np.asarray(feats, str)).str.match(
        r"^(MT-|mt-|ENSG|ENSMUSG|LINC|RP[LS]\d|Rp[ls]\d|Gm\d|MALAT1|Malat1)|.*Rik$|.*-AS\d$").values


def _cols(dcol, D, m):
    c = dcol(D, m); return {k: c[k] for k in range(len(c))}


def _targets(ds):
    return dict(np.load(R / f"targets_{ds}.npz", allow_pickle=True))


def _counts_mod(D, m, z, unit, lab_fn=None, **kw):
    if m == "RNA":
        kw.setdefault("keep", named(D["names"][m]))
    return Mod(name=m, d=np.asarray(D["d"][m], float), lab=np.asarray(D["lab"][m]), names=D["names_dom"][m],
               colors=kw.pop("colors"), kind="poisson", feats=np.asarray(D["names"][m]), X=D["X"][m].tocsc(), lib=D["lib"][m],
               Y=z[f"Y_{m}"], depth=D["lib"][m], depth_label=f"median {m} counts / spot", label=lab_fn, unit=unit, **kw)


# --------------------------------------------------------------------------- Fig 2 LPC lesion
def fig2():
    F = _import("fig2", "fig2_lpc"); D = F.load(); z = _targets(F.DS)
    import saturn_cross_mechanism as MC
    pk = region_labels(list(D["names"]["ATAC"]), GTF["mouse"])
    notes = {"RNA": "named from independent annotations: lesion core = most CODEX-lesion spots; white matter = highest "
                    "myelin score; the remaining, microglia-poor domain = thalamic & periventricular",
             "ATAC": "inflammatory chromatin = highest microglia score (lesion + distal WM focus); the others are named "
                     "by their rho distance from it: parenchymal, then transitional / thalamic"}
    mods = [_counts_mod(D, "RNA", z, "log CP10k", colors=_cols(F.dcol, D, "RNA")),
            _counts_mod(D, "ATAC", z, "log CP10k", lab_fn=lambda f: pk.get(f, f), colors=_cols(F.dcol, D, "ATAC"), min_count=200)]
    return Spec(fig="fig2", title="Fig 2 · LPC lesion (LPC10 FOV)", xy=D["xy"], mods=mods, s=1.1,
                comp=MC.composition("lpc10", D["bc"]), comp_label="cell2location composition", notes=notes)


# --------------------------------------------------------------------------- Fig 3 melanoma
def fig3():
    F = _import("fig3", "fig3_melanoma"); D = F.load(); z = _targets(F.DS)
    from paper_style import DOMAIN
    F.dcol = lambda D, m: [DOMAIN[D["names_dom"][m][k]] for k in range(len(D["names_dom"][m]))]
    import saturn_cross_mechanism as MC
    pk = region_labels(list(D["names"]["ATAC"]), GTF["human"])
    notes = {"RNA": "named from marker programmes: epidermis = KRT10-high; IFN-stromal = STAT1 + COL1A1-high; "
                    "melanocytic tumour = PMEL-high; the remaining domain = AP-1 stress (checked JUN-high)",
             "ATAC": "immune-accessible = highest immune-marker score; metabolic tumour = highest accessibility at the "
                     "PHGDH tile (chr1:119.67 Mb); the remaining domain = mixed tumour-stroma"}
    mods = [_counts_mod(D, "RNA", z, "log CP10k", colors=_cols(F.dcol, D, "RNA")),
            _counts_mod(D, "ATAC", z, "log CP10k", lab_fn=lambda f: pk.get(f, f), colors=_cols(F.dcol, D, "ATAC"), min_count=200)]
    return Spec(fig="fig3", title="Fig 3 · melanoma (DBiT RNA + ATAC)", xy=D["xy"], mods=mods, s=2.6,
                comp=MC.composition("melanoma", D["bc"]), comp_label="Starfysh composition", notes=notes)


# --------------------------------------------------------------------------- Fig 4 MERFISH isoform
def fig4():
    F = _import("fig4", "fig4_isoform_sv"); D = F.load(); B, C = D["B"], D["C"]
    z = B["z"]; dc = lambda m: {k: F.DOMAIN[F.NAMES[m][k]] for k in F.NAMES[m]}
    rna = Mod(name="RNA", d=np.asarray(D["d"]["RNA"], float), lab=np.asarray(D["lab"]["RNA"]), names=F.NAMES["RNA"],
              colors=dc("RNA"), kind="poisson", feats=np.asarray(B["genes"]), X=B["Xg"].tocsc(), lib=B["lib"], Y=z["Y_RNA"],
              depth=B["lib"], depth_label="median transcripts / bin", unit="log CP10k", keep=named(B["genes"]))
    # isoform usage: successes = isoform counts, trials = the gene's counts over its isoforms
    S = B["Xi"].tocsc(); genes = C["iso_gene"]; ug, gi = np.unique(genes, return_inverse=True)
    A = sp.csr_matrix((np.ones(len(genes)), (np.arange(len(genes)), gi)), shape=(len(genes), len(ug)))
    Ng = (S @ A).tocsc(); N = Ng[:, gi]
    tot = np.asarray(S.sum(0)).ravel(); multi = np.bincount(gi)[gi] >= 2
    lab_iso = dict(zip(C["iso"], [f"{g} {i[-6:]}" for g, i in zip(genes, C["iso"])]))
    iso = Mod(name="Isoform", d=np.asarray(D["d"]["Isoform"], float), lab=np.asarray(D["lab"]["Isoform"]),
              names=F.NAMES["Isoform"], colors=dc("Isoform"), kind="binom", feats=np.asarray(C["iso"]), X=S, N=N,
              Y=z["Y_Isoform"], depth=B["depth"], depth_label="median isoform transcripts / bin", unit="usage",
              label=lambda f: lab_iso.get(f, f), keep=multi & (tot >= 200))
    notes = {"RNA": "named from MERFISH cell-type labels per domain (top 3 types, checked by assertion): thalamus / "
                    "hypothalamus, white matter & hippocampus, ventricle / choroid plexus, cortex",
             "Isoform": "subcortical & white matter = oligodendrocyte-led with thalamus; amygdala / olfactory cortex = "
                        "Cortex Subplate & Amygdalar Areas; cortex = L2/3 IT-led"}
    comp = B["ct_frac"].copy(); comp.index = np.arange(len(comp))
    return Spec(fig="fig4", title="Fig 4 · brain MERFISH (RNA + isoform usage)", xy=D["xy"], mods=[rna, iso], s=1.0,
                comp=comp, comp_label="MERFISH cell-type composition", notes=notes)


# --------------------------------------------------------------------------- Fig 5 Sprox-seq
def fig5():
    F = _import("fig5", "fig5_B1"); D = F.load(); z = _targets(F.FIT)
    comp = pd.get_dummies(D["sp"].compartment).astype(float); comp.columns = [F.COMP_NAME.get(c, c) for c in comp.columns]
    notes = {"RNA": "named from marker-gene compartments (crypt epithelium, IgA plasma, mantle & T zone, germinal centre) "
                    "enriched in each domain",
             "Protein": "named from the enriched proteins: CD3 / PD-1 / CD35 (T / Tfh-FDC), IgD (naive mantle B), CD9 + CD38"}
    mods = [_counts_mod(D, "RNA", z, "log CP10k", colors=_cols(F.dcol, D, "RNA")),
            _counts_mod(D, "Protein", z, "log CP10k", lab_fn=F.nice, colors=_cols(F.dcol, D, "Protein"), min_count=50,
                        min_lfc=0.15, keep=~pd.Series(D["names"]["Protein"]).str.match(r"^(mIgG|HIgG)").values)]
    return Spec(fig="fig5", title="Fig 5 · tonsil Sprox-seq B1 (RNA + protein)", xy=D["xy"], mods=mods, s=1.4,
                comp=comp, comp_label="marker-gene compartment", notes=notes)


# --------------------------------------------------------------------------- Fig 6 melanoma Hi-C
def fig6():
    F = _import("fig6", "fig6_melanoma_hic"); D = F.load(); z = _targets(F.DS)
    from continuous_marker_panels import hic_features_mel
    bc = list(pd.read_csv(R / F.DS / "axes.csv", index_col=0).index.astype(str))
    V, bins, _ = hic_features_mel(bc); bl = region_labels(list(bins), GTF["human"])
    rna = Mod(name="RNA", d=np.asarray(D["d"]["RNA"], float), lab=np.asarray(D["lab"]), names=D["names_dom"],
              colors={k: c for k, c in enumerate(F.dcol(D, "RNA"))}, kind="poisson", feats=D["genes"], X=D["X"].tocsc(),
              lib=D["libr"], Y=z["Y_RNA"], depth=D["libr"], depth_label="median RNA UMI / spot", unit="log CP10k",
              keep=named(D["genes"]))
    hic = Mod(name="HiC", d=np.asarray(D["d"]["HiC"], float), lab=np.asarray(D["labH"]), names=D["names_H"],
              colors={k: c for k, c in enumerate(F.dcol(D, "HiC"))}, kind="gauss", feats=np.asarray(bins), X=V, Y=z["Y_HiC"],
              depth=D["reads"], depth_label="median log Hi-C reads / spot", unit="A/B compartment (z)",
              label=lambda f: bl.get(f, f))
    ct = [k for k in z if k.startswith("mk_ct_")]
    comp = pd.DataFrame({k[6:].replace("_", " "): z[k].astype(float) for k in ct})
    notes = {"RNA": "each domain assigned the marker programme it is most enriched for (z across domains, one-to-one "
                    "matching); every assignment checked enriched. The melanocytic-tumour domain has the highest melanocytic score "
                    "(DCT, OCA2, PMEL) but is also keratin-high (adjacent epidermis), so keratins lead its one-vs-rest markers",
             "HiC": "elbow K = 2: the domain with the higher tumour-programme score (DCT, PMEL, MLANA, TYR, FRMD5, DPP6, "
                    "KCNMA1) = tumour, the other = non-tumour"}
    return Spec(fig="fig6", title="Fig 6 · melanoma spatial Hi-C (RNA + A/B compartments)", xy=D["xy"], mods=[rna, hic],
                s=1.0, comp=comp, comp_label="cell-type proportion", notes=notes)
