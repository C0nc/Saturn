"""Why the ATAC cross term explains an RNA gene, and why it helps where rho_ATAC is steep (Fig 7 g/h follow-up).

Datasets: the two RNA+ATAC figure fits (lpc10, melanoma). Genes = cross-explained genes (FDR 5%, dR2 >= 0.01).
  absorb     for each gene: y ~ B(rho_RNA) + K  vs  + K + B(rho_ATAC), K = one candidate source of the cross signal:
               cis    the gene's own +-50 kb accessibility (hinge basis of it)
               TF     top-6 PCs of per-spot chromVAR TF activity (23/24 panel TFs)
               comp   top-6 PCs of cell-type composition (cell2location 26 types / Starfysh 28 states)
               all    cis + TF + comp
               null   the TF block or the composition block spatially rotated (5 rotations each, mean) = what any smooth 6-column block absorbs
             absorbed share = 1 - dR2_ATAC|K / dR2_ATAC. All blocks 15-NN smoothed; equal width (6 columns).
  drivers    per gene: correlation of its cross profile with each cell type and each TF (top 3 of each)
  boundary   per spot: local gradient magnitude (25-NN linear fit, standardised feature) of every cell type and TF;
             each feature's change in the rho_ATAC-steep quintile vs the rho_RNA-steep quintile (log2 ratio);
             and the gain trend split into its two parts (|grad rho_ATAC| up, |grad rho_RNA| down)
-> results/real_saturn/saturn_cross_term/<key>_mech_{genes.csv, drivers.csv, boundary.csv, summary.json}
Usage: python scripts/saturn_cross_mechanism.py [lpc10 melanoma]
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import saturn_cross_term as SC

R, OUT = SC.R, SC.OUT
DR2_MIN, NPC, KS, KG, NROT = 0.01, 6, 15, 25, 5


def composition(key, bc):
    if key == "lpc10":
        import mudata as md
        m = md.read_h5mu(f"{R}/lpc10_fov_mudata/LPC10_lesion_FOV.h5mu"); C = m.obsm["c2l_proportions"]
        C.index = m.obs_names.astype(str)
    else:
        C = pd.read_csv(f"{SC.H}/melanoma_multiseek/data/starfysh_composition.csv", index_col=0); C.index = C.index.astype(str)
    return C.reindex(bc)


def tf_activity(key, bc):
    z = np.load(f"{R}/{SC.FITS[key]['ds']}_tf_activity/tf_z_per_spot.npz", allow_pickle=True)
    return pd.DataFrame(z["z"], index=z["barcodes"].astype(str), columns=z["tfs"].astype(str)).reindex(bc)


def smooth(V, xy, k=KS):
    nn = cKDTree(xy).query(xy, k=k)[1]; return V[nn].mean(1)


def pcs(V, n=NPC):
    V = (V - V.mean(0)) / (V.std(0) + 1e-9); U, S, _ = np.linalg.svd(V, full_matrices=False); return U[:, :n] * S[:n]


def local_grad(V, xy, k=KG):
    """|gradient| of each standardised column by a local linear fit on the k nearest spots (coords scaled by std)"""
    V = (V - V.mean(0)) / (V.std(0) + 1e-9); X = (xy - xy.mean(0)) / xy.std(0)
    nn = cKDTree(X).query(X, k=k)[1]; out = np.empty(V.shape)
    for i in range(len(X)):
        A = X[nn[i]] - X[nn[i]].mean(0); b = V[nn[i]] - V[nn[i]].mean(0)
        g = np.linalg.lstsq(A, b, rcond=None)[0]; out[i] = np.linalg.norm(g, axis=0)
    return out


def r2(A, Y):
    c = np.linalg.lstsq(A, Y, rcond=None)[0]
    return 1 - ((Y - A @ c) ** 2).sum(0) / (((Y - Y.mean(0)) ** 2).sum(0) + 1e-12)


def dR2(base, Y, Bx):
    return r2(np.c_[base, Bx], Y) - r2(base, Y)


def run(key):
    T = pd.read_csv(f"{OUT}/{key}_spots.csv", index_col=0); T.index = T.index.astype(str); bc = list(T.index)
    xy = T[["x", "y"]].values; N = len(T)
    G = pd.read_csv(f"{OUT}/{key}_genes.csv"); hit = G[(G.q < 0.05) & (G.dR2_cross >= DR2_MIN)].gene.values
    Xm, genes = SC.rna(key, bc); col = np.isin(genes, hit) & ~pd.Series(genes).duplicated().values
    names = genes[col]; Y = SC.logcpm(Xm, col)
    C, Z = composition(key, bc), tf_activity(key, bc)
    ok = C.notna().all(1).values & Z.notna().all(1).values
    print(f"[{key}] {len(names)} cross genes, {ok.sum()}/{N} spots with composition + TF", flush=True)
    T, xy, Y, C, Z = T[ok], xy[ok], Y[ok], C[ok], Z[ok]; bc = list(T.index); N = len(T)
    Cs, Zs = smooth(C.values, xy), smooth(Z.values, xy)
    Bo, Bx = SC.hinge(T.rho_RNA.values), SC.hinge(T.rho_X.values); base = np.c_[np.ones(N), Bo]
    Kc, Kz = pcs(Cs), pcs(Zs)
    d0 = dR2(base, Y, Bx)
    res = dict(gene=names, dR2_ATAC=d0, dR2_given_TF=dR2(np.c_[base, Kz], Y, Bx), dR2_given_comp=dR2(np.c_[base, Kc], Y, Bx))
    rng = np.random.default_rng(0); nul = []
    for _ in range(NROT):
        nul += [dR2(np.c_[base, SC.rotate(xy, K, rng)], Y, Bx) for K in (Kz, Kc)]
    res["dR2_given_null"] = np.mean(nul, 0)
    # cis: the gene's own window, per gene
    keep, W, _ = SC.cis_test(key, bc, names, np.zeros((N, len(names))), np.ones(len(names), bool))
    gi = {g: i for i, g in enumerate(names)}; cis_d, all_d = np.full(len(names), np.nan), np.full(len(names), np.nan)
    Ws = smooth(W, xy)
    for j, g in enumerate(keep):
        i = gi[g]; Kw = SC.hinge(Ws[:, j])[:, :NPC]
        cis_d[i] = dR2(np.c_[base, Kw], Y[:, [i]], Bx)[0]
        all_d[i] = dR2(np.c_[base, Kw, Kz, Kc], Y[:, [i]], Bx)[0]
    res["dR2_given_cis"], res["dR2_given_all"] = cis_d, all_d
    D = pd.DataFrame(res)
    for b in ("cis", "TF", "comp", "all", "null"):
        D[f"absorbed_{b}"] = 1 - D[f"dR2_given_{b}"] / D.dR2_ATAC
    D.to_csv(f"{OUT}/{key}_mech_genes.csv", index=False)

    # drivers: cross profile of each gene vs each cell type / TF
    _, _, _, prof = SC.fit_split(Y, T.rho_RNA.values, T.rho_X.values)
    rc = np.corrcoef(np.c_[prof, Cs].T)[:len(names), len(names):]; rz = np.corrcoef(np.c_[prof, Zs].T)[:len(names), len(names):]
    rows = []
    for i, g in enumerate(names):
        oc, oz = np.argsort(-np.abs(rc[i]))[:3], np.argsort(-np.abs(rz[i]))[:3]
        rows.append(dict(gene=g, **{f"type{j+1}": f"{C.columns[o]} ({rc[i, o]:+.2f})" for j, o in enumerate(oc)},
                         **{f"tf{j+1}": f"{Z.columns[o]} ({rz[i, o]:+.2f})" for j, o in enumerate(oz)}))
    pd.DataFrame(rows).to_csv(f"{OUT}/{key}_mech_drivers.csv", index=False)

    # boundary: what changes where rho_ATAC is steep
    lg = np.log((T.grad_X / T.grad_X.median()) / (T.grad_RNA / T.grad_RNA.median())).values
    top, bot = lg >= np.quantile(lg, 0.8), lg <= np.quantile(lg, 0.2)
    feats = pd.concat([pd.DataFrame(Cs, columns=[f"type:{c}" for c in C.columns]),
                       pd.DataFrame(Zs, columns=[f"TF:{c}" for c in Z.columns])], axis=1)
    Gm = local_grad(feats.values, xy)
    Bd = pd.DataFrame(dict(feature=feats.columns, grad_ATACsteep=Gm[top].mean(0), grad_RNAsteep=Gm[bot].mean(0),
                           grad_all=Gm.mean(0), level_ATACsteep=feats.values[top].mean(0) / (feats.values.mean(0) + 1e-9)))
    Bd["log2_steep_ratio"] = np.log2(Bd.grad_ATACsteep / Bd.grad_RNAsteep)
    Bd["rho_grad_vs_steepness"] = [spearmanr(Gm[:, j], lg)[0] for j in range(Gm.shape[1])]
    Bd.sort_values("log2_steep_ratio", ascending=False).to_csv(f"{OUT}/{key}_mech_boundary.csv", index=False)
    g = T.gain_RNA.values
    S = dict(n_genes=len(names), n_spots=N, n_cis=len(keep),
             median_absorbed={b: float(np.nanmedian(D[f"absorbed_{b}"])) for b in ("cis", "TF", "comp", "all", "null")},
             gain_vs_loggradATAC=float(spearmanr(np.log(T.grad_X), g)[0]),
             gain_vs_neg_loggradRNA=float(spearmanr(-np.log(T.grad_RNA), g)[0]),
             gradRNA_in_steep_rel=float(T.grad_RNA[top].mean() / T.grad_RNA.mean()),
             gradATAC_in_steep_rel=float(T.grad_X[top].mean() / T.grad_X.mean()))
    json.dump(S, open(f"{OUT}/{key}_mech_summary.json", "w"), indent=1)
    print(json.dumps(S, indent=1)); print(Bd.sort_values("log2_steep_ratio", ascending=False).head(8).round(3).to_string())


if __name__ == "__main__":
    for k in (sys.argv[1:] or ["lpc10", "melanoma"]):
        run(k)
