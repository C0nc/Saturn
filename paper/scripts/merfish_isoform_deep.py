"""MERFISH isoform usage: is the spatial heterogeneity cell-type composition, or regulation?

RT&T-AMP-MERFISH (Cohen ... Zhuang, Cell 2026), section 5, the Saturn fit `rtt_merfish_s5`
(50 um bins). Every segmented cell is mapped to its fit bin, so it carries rho_RNA,
rho_Isoform and both domain labels, plus its own cell-type label.

(1) COMPOSITION vs WITHIN-TYPE, for every isoform switch found on the bins
    (saturn_isoform_switch.py: dominant isoform changes between two domains).
    Usage of the switched-to isoform in domain d is U_d = sum_t w_td u_td, with w_td the share
    of the gene's isoform reads from cell type t and u_td that type's usage. The change splits
    EXACTLY (Kitagawa/Oaxaca, averaged weights):
        composition  sum_t (w_t,to - w_t,from) * (u_t,to + u_t,from)/2
        within-type  sum_t (w_t,to + w_t,from)/2 * (u_t,to - u_t,from)
    Types absent from one domain carry their pooled usage (they add composition only).
    Within-type change: 1,000 cell bootstraps inside each (type, domain) -> 95% CI.
    Per type with >= MINR reads in both domains: the same bootstrap for u_t,to - u_t,from.

(2) MECHANISM of each switched isoform pair from GENCODE vM25: alternative TSS (5' ends
    > 100 bp apart), alternative 3' end, both, or internal only; and coding change (one isoform
    not protein_coding). Positive control: for genes whose measured protein-coding isoforms
    differ in 3'UTR length by >= 500 bp, is the LONGEST-UTR isoform used more in neurons than
    in glia (sign test + Wilcoxon over genes)?

(3) USAGE vs EXPRESSION per multi-isoform gene, on each axis: largest domain-to-domain usage
    swing of any isoform vs |log2 fold change| of total gene expression across the same domains.
    Computed pooled AND within astrocytes / oligodendrocytes only (composition-free).

Usage: python scripts/merfish_isoform_deep.py
"""
import os, re, sys, json
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad
from scipy.stats import wilcoxon, binomtest
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_real2 import H
from lpc_gaston_fig4l import gaston_domains

DS, SAMPLE, BIN = "rtt_merfish_s5", "5", 50.0
R = f"{H}/Hole/gaston_mix_mm/results/real_saturn"
D = f"{H}/external_data/rtt_merfish"
GTF = f"{H}/mouse_atlas/gencode.vM25.primary_chr.annotation.gtf"
OUT = f"{R}/{DS}_isoform_deep"
MINR, NBOOT, TSS_BP, UTR_BP = 20, 1000, 100, 500
GLIA = {"Oligodendrocytes", "NFOLs", "Astrocytes", "Microglia", "OPCs"}
NONNEURAL = {"Endothelial", "Choroid Plexus", "Ependymal", "Pericytes", "VLMCs", "SMCs", "Other"}
strip = lambda s: str(s).split(".")[0]


def load_cells():
    a = ad.read_h5ad(f"{D}/cellxgene_ISO_labels.h5ad")
    a = a[a.obs["sample"].astype(str) == SAMPLE].copy()
    iso = np.array([v.startswith("ENSMUST") for v in a.var_names])
    t2g = pd.read_csv(f"{D}/ensembl_tx2gene_mm.tsv", sep="\t").drop_duplicates("tx").set_index("tx")["gene"]
    ig = t2g.reindex(a.var_names[iso]).values; ok = pd.notna(ig)
    xy = a.obs[["global_x", "global_y"]].values.astype(float)
    key = np.floor(xy / BIN).astype(np.int64); code = (key[:, 0] * 1_000_000 + key[:, 1]).astype(str)
    X = sp.csr_matrix(a.X)
    Xi = X[:, np.where(iso)[0][ok]].tocsc(); Xg = X[:, np.where(~iso)[0]].tocsc()
    return dict(bin=code, ct=a.obs["cell_label"].astype(str).values, Xi=Xi, iso=np.array(a.var_names[iso][ok]),
                iso_gene=np.array(ig[ok], str), Xg=Xg, genes=np.array(a.var_names[~iso]))


def cell_domains(C):
    ax = pd.read_csv(f"{R}/{DS}/axes.csv", index_col=0); ax.index = ax.index.astype(str)
    z = dict(np.load(f"{R}/targets_{DS}.npz", allow_pickle=True))
    lab = {m: pd.Series(gaston_domains(ax[f"d_{m}"].values, z[f"Y_{m}"], L_=4), index=ax.index) for m in ("RNA", "Isoform")}
    keep = pd.Index(C["bin"]).isin(ax.index)
    dom = {m: np.full(len(C["bin"]), -1) for m in lab}
    for m in lab: dom[m][keep] = lab[m].reindex(pd.Index(C["bin"][keep])).values
    return dom, keep


def usage_counts(C, gene, iso_to):
    cols = np.where(C["iso_gene"] == gene)[0]
    tot = np.asarray(C["Xi"][:, cols].sum(1)).ravel()
    a = np.asarray(C["Xi"][:, np.where(C["iso"] == iso_to)[0][0]].todense()).ravel()
    return a, tot


def decompose(a, tot, ct, dom, d_from, d_to, rng):
    """Kitagawa split of U_to - U_from into composition + within-type; bootstrap within-type CI"""
    sel = {d: dom == d for d in (d_from, d_to)}
    types = sorted(set(ct[sel[d_from] | sel[d_to]]))

    def stats(idx_f, idx_t):
        rows = {}
        pooled = {}
        for t in types:
            af = a[idx_f][ct[idx_f] == t].sum(); nf = tot[idx_f][ct[idx_f] == t].sum()
            at = a[idx_t][ct[idx_t] == t].sum(); nt = tot[idx_t][ct[idx_t] == t].sum()
            rows[t] = (af, nf, at, nt)
            pooled[t] = (af + at) / max(nf + nt, 1)
        Nf = sum(r[1] for r in rows.values()); Nt = sum(r[3] for r in rows.values())
        comp = within = 0.0
        for t, (af, nf, at, nt) in rows.items():
            uf = af / nf if nf > 0 else pooled[t]; ut = at / nt if nt > 0 else pooled[t]
            wf, wt = nf / max(Nf, 1), nt / max(Nt, 1)
            comp += (wt - wf) * (ut + uf) / 2; within += (wt + wf) / 2 * (ut - uf)
        Uf = sum(r[0] for r in rows.values()) / max(Nf, 1); Ut = sum(r[2] for r in rows.values()) / max(Nt, 1)
        return Ut - Uf, comp, within, rows
    idx_f, idx_t = np.where(sel[d_from])[0], np.where(sel[d_to])[0]
    delta, comp, within, rows = stats(idx_f, idx_t)
    # bootstrap cells inside each (type, domain) stratum
    strata_f = {t: idx_f[ct[idx_f] == t] for t in types}; strata_t = {t: idx_t[ct[idx_t] == t] for t in types}
    bw, per_type = [], {t: [] for t in types}
    for _ in range(NBOOT):
        bf = np.concatenate([rng.choice(v, len(v)) for v in strata_f.values() if len(v)])
        bt = np.concatenate([rng.choice(v, len(v)) for v in strata_t.values() if len(v)])
        _, _, w_, r_ = stats(bf, bt); bw.append(w_)
        for t, (af, nf, at, nt) in r_.items():
            if nf >= MINR and nt >= MINR: per_type[t].append(at / nt - af / nf)
    tt = []
    for t, (af, nf, at, nt) in rows.items():
        if nf >= MINR and nt >= MINR and len(per_type[t]) > 50:
            v = np.array(per_type[t]); p = 2 * min((v <= 0).mean(), (v >= 0).mean())
            tt.append(dict(type=t, reads_from=int(nf), reads_to=int(nt), du=at / nt - af / nf,
                           ci_lo=np.percentile(v, 2.5), ci_hi=np.percentile(v, 97.5), p=max(p, 1 / NBOOT)))
    return dict(delta=delta, comp=comp, within=within, within_ci=(np.percentile(bw, 2.5), np.percentile(bw, 97.5))), tt


def gtf_transcripts(ids):
    ids = set(ids); T = {}
    pat_id = re.compile(r'transcript_id "([^"]+)"'); pat_ty = re.compile(r'transcript_type "([^"]+)"')
    with open(GTF) as f:
        for line in f:
            if line[0] == "#": continue
            p = line.split("\t", 9)
            if p[2] not in ("transcript", "exon", "CDS"): continue
            m = pat_id.search(p[8])
            if not m or strip(m.group(1)) not in ids: continue
            t = strip(m.group(1)); s, e = int(p[3]), int(p[4])
            r = T.setdefault(t, dict(strand=p[6], exons=[], cds=[], type=None))
            if p[2] == "transcript": r["start"], r["end"], r["type"] = s, e, pat_ty.search(p[8]).group(1)
            elif p[2] == "exon": r["exons"].append((s, e))
            else: r["cds"].append((s, e))
    for t, r in T.items():
        plus = r["strand"] == "+"
        r["tss"] = r["start"] if plus else r["end"]; r["tes"] = r["end"] if plus else r["start"]
        if r["cds"]:
            stop = max(e for _, e in r["cds"]) if plus else min(s for s, _ in r["cds"])
            r["utr3"] = sum(max(0, e - max(s, stop + 1) + 1) for s, e in r["exons"]) if plus else \
                        sum(max(0, min(e, stop - 1) - s + 1) for s, e in r["exons"])
        else:
            r["utr3"] = np.nan
    return T


def mechanism(T, a, b):
    if a not in T or b not in T: return dict(mechanism="unannotated", coding_change=np.nan)
    A, B = T[a], T[b]
    dtss, dtes = abs(A["tss"] - B["tss"]) > TSS_BP, abs(A["tes"] - B["tes"]) > TSS_BP
    mech = "alt TSS + 3' end" if dtss and dtes else "alt TSS" if dtss else "alt 3' end" if dtes else "internal splicing"
    coding = {A["type"], B["type"]}
    return dict(mechanism=mech, coding_change=("protein_coding" in coding and len(coding) > 1),
                type_from=A["type"], type_to=B["type"], utr3_from=A["utr3"], utr3_to=B["utr3"])


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    C = load_cells(); dom, keep = cell_domains(C)
    print(f"{DS}: {len(C['bin'])} cells, {keep.sum()} in fitted bins; {len(C['iso'])} isoforms", flush=True)
    S = pd.read_csv(f"{R}/{DS}_isoform_switch/isoform_switches.csv")
    rows, trows = [], []
    for _, s in S.iterrows():
        a, tot = usage_counts(C, s.gene, s.iso_to)
        m = keep & (tot > 0)
        dec, tt = decompose(a[m], tot[m], C["ct"][m], dom[s.axis][m], int(s.dom_from), int(s.dom_to), rng)
        rows.append(dict(axis=s.axis, gene=s.gene, dom_from=int(s.dom_from), dom_to=int(s.dom_to), iso_from=s.iso_from,
                         iso_to=s.iso_to, delta=dec["delta"], composition=dec["comp"], within=dec["within"],
                         within_lo=dec["within_ci"][0], within_hi=dec["within_ci"][1],
                         within_frac=dec["within"] / dec["delta"] if abs(dec["delta"]) > 1e-9 else np.nan))
        for r in tt: trows.append(dict(axis=s.axis, gene=s.gene, dom_from=int(s.dom_from), dom_to=int(s.dom_to), iso_to=s.iso_to, **r))
    Dm = pd.DataFrame(rows); Tt = pd.DataFrame(trows)
    Dm["within_sig"] = (np.sign(Dm.within_lo) == np.sign(Dm.within_hi)) & (np.sign(Dm.within_lo) == np.sign(Dm.delta))
    if len(Tt):
        Tt["q"] = multipletests(Tt.p, method="fdr_bh")[1]
        Tt["same_direction"] = np.sign(Tt.du) == np.sign(Tt.merge(Dm[["axis", "gene", "dom_from", "dom_to", "delta"]],
                                                                   on=["axis", "gene", "dom_from", "dom_to"], how="left").delta.values)
    Dm.to_csv(f"{OUT}/switch_decomposition.csv", index=False); Tt.to_csv(f"{OUT}/switch_within_type.csv", index=False)
    print(f"\n(1) {len(Dm)} switches: median within-type share {Dm.within_frac.median():.2f}; "
          f"within-type change significant and same-direction in {int(Dm.within_sig.sum())}/{len(Dm)}", flush=True)
    if len(Tt):
        ok = Tt[(Tt.q < 0.05) & Tt.same_direction]
        print(f"    type-level: {len(ok)}/{len(Tt)} (type, switch) cases significant (BH q<0.05) in the switch direction; "
              f"{ok.gene.nunique()} genes; by type: {ok.type.value_counts().head(8).to_dict()}", flush=True)

    # (2) mechanism + neuronal 3'UTR control
    T = gtf_transcripts(set(map(strip, C["iso"])))
    Mrows = [dict(axis=s.axis, gene=s.gene, **mechanism(T, strip(s.iso_from), strip(s.iso_to))) for _, s in S.iterrows()]
    Mm = pd.DataFrame(Mrows); Mm.to_csv(f"{OUT}/switch_mechanism.csv", index=False)
    print(f"\n(2) switch mechanism: {Mm.mechanism.value_counts().to_dict()} | coding change {int(Mm.coding_change.fillna(False).sum())}/{len(Mm)}", flush=True)
    ct = C["ct"]; neu = keep & ~np.isin(ct, list(GLIA | NONNEURAL)); gli = keep & np.isin(ct, list(GLIA))
    res = []
    for g in np.unique(C["iso_gene"]):
        cols = np.where(C["iso_gene"] == g)[0]
        pcs = [c for c in cols if strip(C["iso"][c]) in T and T[strip(C["iso"][c])]["type"] == "protein_coding"
               and np.isfinite(T[strip(C["iso"][c])]["utr3"])]
        if len(pcs) < 2: continue
        u = np.array([T[strip(C["iso"][c])]["utr3"] for c in pcs])
        if u.max() - u.min() < UTR_BP: continue
        L = pcs[int(np.argmax(u))]
        tn = np.asarray(C["Xi"][neu][:, cols].sum()).item(); tg = np.asarray(C["Xi"][gli][:, cols].sum()).item()
        if tn < 100 or tg < 100: continue
        un = C["Xi"][neu][:, L].sum() / tn; ug = C["Xi"][gli][:, L].sum() / tg
        res.append(dict(gene=g, utr_long=u.max(), utr_short=u.min(), usage_long_neuron=un, usage_long_glia=ug))
    U = pd.DataFrame(res); U["diff"] = U.usage_long_neuron - U.usage_long_glia; U.to_csv(f"{OUT}/utr_neuron_vs_glia.csv", index=False)
    k = int((U["diff"] > 0).sum())
    print(f"    3'UTR control: {len(U)} genes; longest-UTR isoform higher in neurons in {k}/{len(U)} "
          f"(sign p {binomtest(k, len(U), 0.5, alternative='greater').pvalue:.2g}); median diff {U['diff'].median():+.3f}; "
          f"Wilcoxon p {wilcoxon(U['diff'], alternative='greater').pvalue:.2g}", flush=True)

    # (3) usage vs expression
    grows = []
    gi = {g: i for i, g in enumerate(C["genes"])}
    for m in ("RNA", "Isoform"):
        for scope, mask in (("pooled", keep), ("astrocytes", keep & (ct == "Astrocytes")), ("oligodendrocytes", keep & (ct == "Oligodendrocytes"))):
            dm = dom[m]
            libg = np.asarray(C["Xg"].sum(1)).ravel()
            for g in np.unique(C["iso_gene"]):
                cols = np.where(C["iso_gene"] == g)[0]
                if len(cols) < 2 or g not in gi: continue
                U_, E_, ok = [], [], True
                for k_ in range(4):
                    s_ = mask & (dm == k_)
                    n = np.asarray(C["Xi"][s_][:, cols].sum(0)).ravel()
                    if n.sum() < 50 or libg[s_].sum() == 0: ok = False; break
                    U_.append(n / n.sum()); E_.append(C["Xg"][s_][:, gi[g]].sum() / libg[s_].sum() * 1e4)
                if not ok: continue
                U_ = np.array(U_); E_ = np.array(E_)
                swing = float((U_.max(0) - U_.min(0)).max())
                lfc = float(np.log2((E_.max() + 1e-3) / (E_.min() + 1e-3)))
                grows.append(dict(axis=m, scope=scope, gene=g, n_iso=len(cols), usage_swing=swing, expr_log2fc=lfc))
    G = pd.DataFrame(grows); G.to_csv(f"{OUT}/usage_vs_expression.csv", index=False)
    print("\n(3) usage vs expression (swing >= 0.2 = usage changes; |log2FC| < 0.5 = expression flat):")
    for (m, sc), s in G.groupby(["axis", "scope"]):
        pure = ((s.usage_swing >= 0.2) & (s.expr_log2fc < 0.5)).sum(); both = ((s.usage_swing >= 0.2) & (s.expr_log2fc >= 0.5)).sum()
        print(f"   rho_{m:8s} {sc:16s} genes {len(s):4d} | usage changes {int((s.usage_swing >= 0.2).sum()):3d} "
              f"(expression flat {int(pure)}, both move {int(both)}) | corr(swing, |log2FC|) {np.corrcoef(s.usage_swing, s.expr_log2fc)[0, 1]:+.2f}")
    json.dump(dict(n_switches=len(Dm)), open(f"{OUT}/summary.json", "w"))


if __name__ == "__main__":
    main()
