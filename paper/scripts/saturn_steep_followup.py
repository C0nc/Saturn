"""What the rho_ATAC-steep zones are (Fig 7h, j) and how far the steepness reading goes (Fig S8).

Steepness ratio per spot = log((|grad rho_ATAC| / median) / (|grad rho_RNA| / median)) (saturn.crossterm.steepness).
ATAC-steep zone = top 20 % of the ratio; RNA-steep zone = bottom 20 %.

  switch    per cell type (composition) and TF (chromVAR activity): boundary sharpness = |local gradient| of the
            smoothed standardised feature (8-NN smooth, 25-NN linear fit); log2 mean in the ATAC-steep zone over
            the RNA-steep zone; two-sided p against 200 rotations of the zone masks       -> {key}_zone_switch.csv
  edge      steepness-gain Spearman keeping spots > c spots from the edge of the fitted area (rotation p)
  controls  the same correlation with log RNA depth, log ATAC depth and edge distance partialled out (ranks)
  data      model |grad rho| vs the local gradient of the raw Saturn targets (all components), and the two ratios
  codex     LPC only: CODEX protein boundary strength (same section, never used in the fit) vs |grad rho_ATAC|,
            |grad rho_RNA| and the ratio (rotation p)                                         -> lpc10_codex.csv
Output: results/real_saturn/saturn_cross_term/steep_followup/ ; summary followup.json
Run after saturn_cross_term.py and saturn_cross_mechanism.py, with OMP_NUM_THREADS unset.
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/saturn_repo/src")       # Saturn package (saturn_repo)
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
from saturn import crossterm as CT, transition as TR
from saturn.mechanism import smooth
import saturn_cross_term as SC, saturn_cross_mechanism as MC

H = "/mnt/HDD2B/Users/yancui"; R = f"{H}/Hole/gaston_mix_mm/results/real_saturn"; X7 = f"{R}/saturn_cross_term"
OUT = f"{X7}/steep_followup"; NROT = 200; Q = 0.2
CODEX = f"{H}/catalign_lpc10/results/LPC10S1_wm/LPC10S1_codex_per_spot.h5ad"
CODEX_GROUPS = {"cortical neuron layers": ["NeuN", "SATB2", "Tbr1", "Ctip2", "CUX2"], "myelin / OL": ["MBP", "Mog", "Olig2", "APC"],
                "lesion myeloid (CD11c)": ["CD11c"], "astrocyte (GFAP)": ["GFAP"]}


def gmag(v, xy):
    return np.hypot(*TR.grad_vec(v, xy, k=25).T)


def boundary(F, xy):
    """|local gradient| of each smoothed, standardised column"""
    Z = smooth(((F - F.mean()) / (F.std() + 1e-12)).values, xy, k=8)
    return pd.DataFrame({c: gmag(Z[:, j], xy) for j, c in enumerate(F.columns)}, index=F.index)


def rot_p(obs, null):
    return float((1 + (np.abs(null) >= abs(obs)).sum()) / (len(null) + 1))


def edge_dist(xy):
    tr = cKDTree(xy); sp = np.median(tr.query(xy, k=2)[0][:, 1])
    nb = np.array([len(v) - 1 for v in tr.query_ball_point(xy, 1.5 * sp)])
    return cKDTree(xy[nb < 8]).query(xy)[0] / sp


def depths(key, bc):
    """log RNA UMI and log ATAC fragment depth per spot"""
    from domain_annot import CFG
    X, _ = SC.rna(key, bc); A = CFG[SC.FITS[key]["ds"]]["atac"](); A.obs_names = A.obs_names.astype(str)
    return np.log(np.asarray(X.sum(1)).ravel()), np.log(np.asarray(A[pd.Index(bc)].X.sum(1)).ravel())


def run(key, rng):
    T = pd.read_csv(f"{X7}/{key}_spots.csv", index_col=0); bc = list(T.index.astype(str)); xy = T[["x", "y"]].values
    s = CT.steepness(T); g = T.gain_RNA.values; top, bot = s >= np.quantile(s, 1 - Q), s <= np.quantile(s, Q)
    S = dict(n=len(T), n_zone=int(top.sum()))
    # ---- what switches in the ATAC-steep zone ---------------------------------------------------------------
    rows = []
    for kind, F in (("type", MC.composition(key, bc)), ("TF", MC.tf_activity(key, bc))):
        F = F.loc[:, F.std() > 0].fillna(0); B = boundary(F, xy)
        con = np.log2(B[top].mean() / B[bot].mean())
        null = np.array([np.log2(B[CT.rotate(xy, top, rng)].mean() / B[CT.rotate(xy, bot, rng)].mean()).values for _ in range(NROT)])
        for j, c in enumerate(F.columns):
            rows.append(dict(kind=kind, feature=c, log2_sharpness=float(con[c]), p_rot=rot_p(con[c], null[:, j])))
    W = pd.DataFrame(rows).sort_values("log2_sharpness", ascending=False); W.to_csv(f"{OUT}/{key}_zone_switch.csv", index=False)
    # ---- edge ----------------------------------------------------------------------------------------------
    d = edge_dist(xy); S["edge"] = dict(frac_zone_within3=float((d[top] <= 3).mean()), frac_all_within3=float((d <= 3).mean()))
    for c in (0, 3, 6, 10):
        m = d > c
        if m.sum() < 30:
            continue
        obs = spearmanr(s[m], g[m]).statistic
        null = np.array([spearmanr(s[m], CT.rotate(xy[m], g[m], rng)).statistic for _ in range(NROT)])
        S["edge"][f"interior>{c}"] = dict(n=int(m.sum()), rho=float(obs), p_rot=rot_p(obs, null))
    # ---- depth / edge partial ---------------------------------------------------------------------------------
    dR, dA = depths(key, bc); cov = [dR, dA, d]
    def res(a):
        for c in cov:
            a = CT.rank_resid(a, c)
        return a
    S["controls"] = dict(rho_raw=float(spearmanr(s, g).statistic), rho_partial=float(spearmanr(res(s), res(g)).statistic),
                         partialled=["log RNA depth", "log ATAC depth", "edge distance"])
    # ---- model vs raw data ----------------------------------------------------------------------------------
    z = dict(np.load(f"{R}/targets_{SC.FITS[key]['ds']}.npz", allow_pickle=True)); o = pd.Index([str(b) for b in z["barcodes"]]).get_indexer(bc)
    def gm(M):
        M = smooth((M - M.mean(0)) / (M.std(0) + 1e-9), xy, k=8); return np.sqrt(sum(gmag(M[:, j], xy) ** 2 for j in range(M.shape[1])))
    gA, gR = gm(z["Y_ATAC"][o]), gm(z["Y_RNA"][o]); sd = np.log((gA / np.median(gA)) / (gR / np.median(gR)))
    S["data"] = dict(gradATAC=float(spearmanr(T.grad_X, gA).statistic), gradRNA=float(spearmanr(T.grad_RNA, gR).statistic),
                     ratio=float(spearmanr(s, sd).statistic))
    S["gain_by_quintile"] = pd.Series(g).groupby(pd.qcut(pd.Series(s).rank(method="first"), 5, labels=False).values).mean().round(4).tolist()
    # ---- CODEX (LPC) -----------------------------------------------------------------------------------------
    if key == "lpc10":
        import anndata as ad
        C = ad.read_h5ad(CODEX)[bc]; P = pd.DataFrame(np.log1p(np.asarray(C.X, float)), index=bc, columns=C.var_names)
        B = boundary(P.drop(columns="DAPI"), xy)
        rows = []
        for grp, cs in CODEX_GROUPS.items():
            b = B[cs].mean(1).values
            for nm, v in (("grad_RNA", np.log(T.grad_RNA.values)), ("grad_ATAC", np.log(T.grad_X.values)), ("ratio", s)):
                obs = spearmanr(v, b).statistic; null = np.array([spearmanr(v, CT.rotate(xy, b, rng)).statistic for _ in range(NROT)])
                rows.append(dict(boundary=grp, axis=nm, rho=float(obs), p_rot=rot_p(obs, null)))
        pd.DataFrame(rows).to_csv(f"{OUT}/lpc10_codex.csv", index=False)
    return S


def main():
    os.makedirs(OUT, exist_ok=True); rng = np.random.default_rng(0); res = {}
    for k in sys.argv[1:] or ["lpc10", "melanoma"]:
        res[k] = run(k, rng); print(k, json.dumps(res[k], indent=1), flush=True)
    json.dump(res, open(f"{OUT}/followup.json", "w"), indent=1)


if __name__ == "__main__":
    main()
