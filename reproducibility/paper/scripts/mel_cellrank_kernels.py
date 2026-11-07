"""CellRank 2 on melanoma (Fig 3 fit `melanoma_tissue`): tumour-stage transitions from
a DPT baseline kernel, the rho_RNA kernel, the rho_ATAC kernel, and their fusion.

Orientation: every time-like coordinate increases with the melanocytic tumour programme
(ann_mel_marker): non-tumour -> melanocytic tumour. Both axes track it with similar strength
(Spearman rho_RNA -0.23, rho_ATAC +0.35, sign-corrected). CNV burden was tried first and orders
rho_RNA (+0.03) and DPT (+0.02) not at all; on rho_ATAC CNV runs AGAINST the melanocytic programme,
so '--orient ann_cnv_burden_v2' gives the opposite, dedifferentiation reading.
(saturn_cellrank_rho.py oriented towards the immune score.)

Kernels (each 0.8 * PseudotimeKernel + 0.2 * ConnectivityKernel, CellRank's usual regulariser):
  DPT        scanpy diffusion pseudotime on the RNA graph, root = lowest-CNV spot     (baseline)
  rho_RNA    Saturn rho_RNA on the RNA graph   (GLM-PCA, the reduction Saturn was fitted on)
  rho_ATAC   Saturn rho_ATAC on the ATAC graph (5 kb-tile spectral)
  fusion_w   w * k_rhoRNA + (1 - w) * k_rhoATAC for w in WS   (CellRank's two-kernel combination;
             w=1 and w=0 reproduce the single-axis kernels)

Per kernel, GPCCA: Schur -> N_STATES macrostates -> 2 terminal states (top_n) -> fate
probabilities. Tumour fate = the terminal state whose fate probability correlates most with ORIENT.
Reported:
  terminal states       size, Fig 3 domain composition, CNV / melanocytic / immune / stromal z
  tumour fate           correlation with CNV; agreement with every other kernel
  fate entropy          per spot (normalised); AUC of entropy for the gradient transition zones
                        (melanoma_tissue_gradient/spots.csv, either axis) vs all other spots
  drivers (genes)       g.compute_lineage_drivers on log-normalised HVGs for the tumour fate
  drivers (ATAC)        Spearman of tumour-fate probability with ATAC tile accessibility
                        (TF-IDF, tiles open in >= 5% of spots); top positive
Outputs results/real_saturn/melanoma_tissue_cellrank_kernels/
Usage: python scripts/mel_cellrank_kernels.py
"""
import json, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, scanpy as sc
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "<workspace>/Saturn/published_figures/fig3")
from cellrank.kernels import PseudotimeKernel, ConnectivityKernel
from cellrank.estimators import GPCCA

H = "<workspace>"
R = f"{H}/Hole/gaston_mix_mm/results/real_saturn"
OUT = f"{R}/melanoma_tissue_cellrank_kernels" + ("" if "--orient" not in sys.argv else "_" + sys.argv[sys.argv.index("--orient") + 1])
ORIENT = sys.argv[sys.argv.index("--orient") + 1] if "--orient" in sys.argv else "ann_mel_marker"
N_STATES, K, WS, SEED = 4, 15, (0.25, 0.5, 0.75), 0
ZS = ["ann_cnv_burden_v2", "ann_mel_marker", "ann_immune_marker", "ann_stromal_marker"]


def build():
    import fig3_melanoma as F3
    D = F3.load(); names = F3.label_domains(D); z = D["z"]; bc = D["bc"]
    a = ad.AnnData(sp.csr_matrix(D["X"]["RNA"]).astype(np.float32), obs=pd.DataFrame(index=bc),
                   var=pd.DataFrame(index=D["names"]["RNA"]))
    a.var_names_make_unique(); a.obsm["spatial"] = np.asarray(D["xy"], float)
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=3000); a.raw = a
    for m, key in (("RNA", "rna"), ("ATAC", "atac")):
        a.obsm[f"X_{key}"] = np.asarray(z[f"Y_{m}"], float)
        sc.pp.neighbors(a, n_neighbors=K, use_rep=f"X_{key}", key_added=key, random_state=SEED)
    a.obsp["connectivities"] = a.obsp["rna_connectivities"]; a.obsp["distances"] = a.obsp["rna_distances"]
    a.uns["neighbors"] = a.uns["rna"]
    sc.tl.leiden(a, resolution=.6, key_added="leiden", neighbors_key="rna", random_state=SEED)
    cnv = np.asarray(z[ORIENT], float); a.obs["cnv"] = cnv          # 'cnv' = the orientation score
    for c in ZS: a.obs[c] = np.asarray(z[c], float)
    for m in ("RNA", "ATAC"):
        v = np.asarray(D["d"][m], float); s = np.sign(spearmanr(v, cnv)[0]) or 1.0; v = s * v
        a.obs[f"rho_{m}"] = (v - v.min()) / np.ptp(v)
        a.obs[f"domain_{m}"] = [names[m][k] for k in D["lab"][m]]
        print(f"rho_{m}: sign {s:+.0f}, Spearman with {ORIENT} {spearmanr(a.obs[f'rho_{m}'], cnv)[0]:+.2f}", flush=True)
    sc.tl.diffmap(a, neighbors_key="rna", random_state=SEED)
    a.uns["iroot"] = int(np.argmin(cnv)); sc.tl.dpt(a, neighbors_key="rna")
    print(f"DPT (root = lowest-{ORIENT} spot): Spearman with {ORIENT} {spearmanr(a.obs.dpt_pseudotime, cnv)[0]:+.2f}", flush=True)
    return a, D


def kernel(a, time_key, graph):
    pk = PseudotimeKernel(a, time_key=time_key, conn_key=f"{graph}_connectivities").compute_transition_matrix()
    ck = ConnectivityKernel(a, conn_key=f"{graph}_connectivities").compute_transition_matrix()
    return 0.8 * pk + 0.2 * ck


def atac_features(D, bc):
    X = sp.csr_matrix(D["X"]["ATAC"]); B = (X > 0).astype(np.float32)
    keep = np.asarray(B.mean(0)).ravel() >= 0.05
    B = B[:, keep]; nm = np.asarray(D["names"]["ATAC"])[keep]
    tf = sp.diags(1 / np.maximum(np.asarray(B.sum(1)).ravel(), 1)) @ B
    idf = np.log1p(B.shape[0] / np.maximum(np.asarray(B.sum(0)).ravel(), 1))
    return np.asarray((tf @ sp.diags(idf)).todense()), nm


def rank_corr(M, v):
    Rm = np.apply_along_axis(rankdata, 0, M); rv = rankdata(v)
    Rm = (Rm - Rm.mean(0)) / Rm.std(0).clip(1e-9); rv = (rv - rv.mean()) / rv.std()
    return (Rm * rv[:, None]).mean(0)


def run():
    os.makedirs(OUT, exist_ok=True)
    a, D = build()
    grad = pd.read_csv(f"{R}/melanoma_tissue_gradient/spots.csv", index_col=0).loc[a.obs_names]
    zone = (grad.transition_RNA | grad.transition_ATAC).values.astype(bool)
    A, anm = atac_features(D, list(a.obs_names))
    ks = {"DPT": kernel(a, "dpt_pseudotime", "rna"), "rho_RNA": kernel(a, "rho_RNA", "rna"),
          "rho_ATAC": kernel(a, "rho_ATAC", "atac")}
    for w in WS: ks[f"fusion_w{w:.2f}"] = w * ks["rho_RNA"] + (1 - w) * ks["rho_ATAC"]
    fates, rows, drivers = {}, [], {}
    for name, kern in ks.items():
        kern = kern.compute_transition_matrix() if hasattr(kern, "compute_transition_matrix") else kern
        g = GPCCA(kern)
        g.compute_schur(n_components=N_STATES + 4)
        g.compute_macrostates(n_states=N_STATES, cluster_key="leiden")
        g.predict_terminal_states(method="top_n", n_states=2)
        g.compute_fate_probabilities()
        fp = g.fate_probabilities; F = pd.DataFrame(np.asarray(fp), index=a.obs_names, columns=list(fp.names))
        cc = {c: spearmanr(F[c], a.obs.cnv)[0] for c in F.columns}; tum = max(cc, key=cc.get)
        P = F.values.clip(1e-12); ent = -(P * np.log(P)).sum(1) / np.log(P.shape[1])
        a.obs[f"tumour_fate_{name}"] = F[tum].values; a.obs[f"fate_entropy_{name}"] = ent
        a.obs[f"macro_{name}"] = g.macrostates.astype(str).values
        term = g.terminal_states
        tinfo = {}
        for t in F.columns:
            mk = (term == t).values
            tinfo[t] = dict(n=int(mk.sum()),
                            domain_RNA=pd.Series(a.obs.domain_RNA[mk]).value_counts().head(2).to_dict(),
                            domain_ATAC=pd.Series(a.obs.domain_ATAC[mk]).value_counts().head(2).to_dict(),
                            **{c.replace("ann_", ""): round(float((a.obs[c][mk].median() - a.obs[c].mean()) / a.obs[c].std()), 2) for c in ZS})
        try:
            dr = g.compute_lineage_drivers(lineages=[tum], use_raw=False)
            gtop = list(dr.sort_values(f"{tum}_corr", ascending=False).index[:15])
        except Exception as e:
            gtop = [f"driver error: {type(e).__name__}"]
        r = rank_corr(A, F[tum].values); atop = list(anm[np.argsort(-r)[:10]])
        drivers[name] = dict(genes=gtop, atac=atop)
        auc = roc_auc_score(zone, ent)
        rows.append(dict(kernel=name, tumour_terminal=tum, tumour_fate_vs_orient=round(float(cc[tum]), 3),
                         entropy_auc_gradient_zones=round(float(auc), 3),
                         entropy_zone_median=round(float(np.median(ent[zone])), 3),
                         entropy_rest_median=round(float(np.median(ent[~zone])), 3), terminals=json.dumps(tinfo)))
        F.to_csv(f"{OUT}/fate_{name}.csv"); fates[name] = F[tum].values
        print(f"\n[{name}] terminal states {list(F.columns)}; tumour fate = '{tum}' (Spearman with {ORIENT} {cc[tum]:+.2f})", flush=True)
        for t, v in tinfo.items(): print(f"    {t}: {v}")
        print(f"    fate entropy: gradient zones {np.median(ent[zone]):.2f} vs rest {np.median(ent[~zone]):.2f} (AUC {auc:.2f})")
        print(f"    driver genes: {gtop[:10]}")
        print(f"    top ATAC tiles: {atop[:5]}", flush=True)
    names = list(fates)
    agree = pd.DataFrame([[spearmanr(fates[x], fates[y])[0] for y in names] for x in names], index=names, columns=names)
    print("\ntumour-fate agreement between kernels (Spearman):\n", agree.round(2).to_string())
    jac = {}
    for x in names:
        for y in names:
            if x < y:
                gx, gy = set(drivers[x]["genes"]), set(drivers[y]["genes"]); jac[f"{x}|{y}"] = round(len(gx & gy) / max(len(gx | gy), 1), 2)
    pd.DataFrame(rows).to_csv(f"{OUT}/kernels_summary.csv", index=False); agree.to_csv(f"{OUT}/fate_agreement.csv")
    json.dump(dict(drivers=drivers, driver_gene_jaccard=jac), open(f"{OUT}/drivers.json", "w"), indent=1)
    a.obs.to_csv(f"{OUT}/obs.csv"); print(f"-> {OUT}")


if __name__ == "__main__":
    run()
