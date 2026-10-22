"""Fig 5 (B1) tables: both Saturn axes of the depth-regressed p46 fit, their domains, and the held-out complex layer.

Section B1 (fit sproxseq_B1_p46_dr). Axes oriented as in sproxseq_B1_rna_axis.py: rho_RNA increases from the crypt
epithelium into the lymphoid interior, rho_Protein from the T zone to the B follicle.

  domains     GASTON DP on each axis; K = elbow of the DP error curve (domain_k_elbow.kneedle), registered as fig5
  proteins    29 targets: A/B probe counts summed (protein_marginal), for Poisson fitted functions
  complexes   isotype-corrected specificity (sproxseq_D1_crypt_niche.spec) per equal-count bin of each axis;
              slope over 8 bins, two-sided p against 200 rotations of that axis field; 30-bin profiles for curves
-> results/real_saturn/sproxseq_B1_fig5/{spots.csv, domains.json, proteins.npz, complexes_<axis>.csv, profiles_<axis>.csv,
   bins_<axis>.csv, summary.json}
Usage: python scripts/sproxseq_B1_fig5_prep.py
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sproxseq_ccc import rotate, R, CCC
from sproxseq_D1_crypt_niche import layers, spec
from lpc_gaston_fig4l import gaston_domains
from domain_k_elbow import curve, kneedle

SEC, FIT = "B1", "sproxseq_B1_p46_dr"; OUT = f"{R}/sproxseq_B1_fig5"; NROT, NB_TEST, NB_PROF = 200, 8, 30


def ebins(t, nb):
    return np.eye(nb)[np.minimum((np.argsort(np.argsort(t)) * nb) // len(t), nb - 1)]


def slope(S, nb):
    c = np.arange(nb) - (nb - 1) / 2
    return (S * c[:, None]).sum(0) / (c ** 2).sum()


def main():
    os.makedirs(OUT, exist_ok=True)
    sp = pd.read_csv(f"{R}/sproxseq_B1_rna_axis/spots.csv", index_col=0); bc = list(sp.index)
    ax = pd.read_csv(f"{R}/{FIT}/axes.csv", index_col=0).loc[bc]
    z = dict(np.load(f"{R}/targets_{FIT}.npz", allow_pickle=True))
    zi = pd.Index(pd.read_csv(f"{R}/{FIT}/axes.csv", index_col=0).index.astype(str)).get_indexer(bc)
    xy = sp[["x", "y"]].values.astype(float); out = {}
    for m, t in (("RNA", sp.t_RNA.values), ("Protein", sp.t_Prot.values)):
        d = ax[f"d_{m}"].values; d = d if np.corrcoef(d, t)[0, 1] > 0 else -d          # crypt -> interior / T -> follicle
        Y = np.asarray(z[f"Y_{m}"], float)[zi]
        K = kneedle(curve(d, Y)); lab = gaston_domains(d, Y, L_=K)
        o = np.argsort([np.median(d[lab == k]) for k in range(lab.max() + 1)]); lab = np.argsort(o)[lab]   # 0 = low end
        sp[f"rho_{m}"] = d; sp[f"dom_{m}"] = lab; out[m] = dict(K=int(K), sizes=np.bincount(lab).tolist())
        comp = {int(k): sp.compartment[lab == k].value_counts(normalize=True).round(2).head(4).to_dict() for k in range(K)}
        out[m]["composition"] = comp
    # proteins: A/B probes summed per target
    P = np.load(f"{R}/sproxseq_B1_p46_pla.npz", allow_pickle=True)
    pi = {b: i for i, b in enumerate(P["barcodes"].astype(str))}; ix = [pi[b] for b in bc]
    probes = P["proteins"].astype(str); tg = sorted({p.split("_")[0] for p in probes})
    M = P["protein_marginal"][ix].astype(float)
    Pc = np.column_stack([M[:, [p.split("_")[0] == t for p in probes]].sum(1) for t in tg])
    np.savez_compressed(f"{OUT}/proteins.npz", counts=Pc, targets=np.array(tg), barcodes=np.array(bc))
    # complexes along both axes
    pairs, O, E, IO, IE, tgc = layers(SEC, bc); name = [f"{a}:{b}" for a, b in pairs]
    ccc = {f"{a}:{b}": k for k, v in CCC.items() for a, b in v}
    for m in ("RNA", "Protein"):
        d = sp[f"rho_{m}"].values
        S = spec(ebins(d, NB_TEST), pairs, O, E, IO, IE, tgc); obs = slope(S, NB_TEST)
        rng = np.random.default_rng(0)
        null = np.array([slope(spec(ebins(rotate(xy, d[:, None], rng)[:, 0], NB_TEST), pairs, O, E, IO, IE, tgc), NB_TEST)
                         for _ in range(NROT)])
        p = ((np.abs(null) >= np.abs(obs)[None]).sum(0) + 1) / (NROT + 1)
        Hp = ebins(d, NB_PROF); Sp = spec(Hp, pairs, O, E, IO, IE, tgc)
        pd.DataFrame(Sp, columns=name).to_csv(f"{OUT}/profiles_{m}.csv")
        g = np.argmax(Hp, 1)
        pd.DataFrame(dict(rho=[d[g == b].mean() for b in range(NB_PROF)],
                          domain=[np.bincount(sp[f"dom_{m}"].values[g == b]).argmax() for b in range(NB_PROF)])).to_csv(f"{OUT}/bins_{m}.csv")
        T = pd.DataFrame(dict(slope=obs, p=p, reads=O.sum(0), ccc=[ccc.get(n, "") for n in name]), index=name)
        T.to_csv(f"{OUT}/complexes_{m}.csv")
        out[m].update(n_complexes=len(name), frac_p05=float((p < 0.05).mean()),
                      ccc={n: dict(slope=round(float(T.loc[n, "slope"]), 4), p=float(T.loc[n, "p"])) for n in name if n in ccc})
        print(f"{m}: K {out[m]['K']} {out[m]['sizes']} | complexes with slope p<0.05 {out[m]['frac_p05']:.1%} (chance 5%)", flush=True)
        for n, v in out[m]["ccc"].items(): print(f"   {ccc[n]:14s} {n:16s} slope {v['slope']:+.3f} p {v['p']:.3f}")
    sp.to_csv(f"{OUT}/spots.csv"); json.dump(out, open(f"{OUT}/summary.json", "w"), indent=1)
    np.savez_compressed(f"{OUT}/layers.npz", O=O, E=E, IO=IO, IE=IE, pairs=np.array(name), tg=np.array(tgc))
    print("->", OUT)


if __name__ == "__main__":
    main()
