"""Three-way cis consistency of gene expression, A/B compartment and TAD insulation along the Fig 6 Saturn axes
(the Hi-C analogue of the LPC cis gene-peak concordance, Fig 2h).

Unit = one 500 kb bin (cis window: its >= 3 protein-coding genes, its A/B value, its insulation) in one domain of one
axis. Per spot: gene score = mean log-expression of the bin's genes; A/B and insulation as in cache_series.npz.
  fit       each of the three series as a per-domain linear function of rho (saturn piecewise.fit_gauss, F-test gate
            t = 0.1) -> slope sign per domain, 0 if not sloped
  pairs     for every pair (gene-A/B, gene-insulation, A/B-insulation): among units where both are sloped, the fraction
            with the SAME sign
  three-way among units where all three are sloped, the share of each sign pattern (gene, A/B, insulation)
  control   the bin's gene score paired with the chromatin of a random genome-wide bin outside its cis window
            (not the same chromosome within 1 Mb; 20 draws); cis effect = observed - control mean, p = share of draws >=
            observed (one-sided, +1 rule). --ctrl distal: random bin on the same chromosome > 1 Mb away instead
-> results/real_saturn/<DS>_insulation_granger/cis_threeway_{pairs,patterns}.csv, cis_threeway_units.csv
   (--ctrl distal -> *_distal.csv)
Usage: python scripts/mel_hic_cis_threeway.py [--ds mel_hic_abins_dr50_51_rnadr] [--ctrl random|distal]
"""
import os, sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domain_annot import R
from piecewise import fit_gauss
from lpc_gaston_fig4l import gaston_domains
from domain_k_elbow import curve, kneedle

DS = sys.argv[sys.argv.index("--ds") + 1] if "--ds" in sys.argv else "mel_hic_abins_dr50_51_rnadr"
G = f"{R}/{DS}_insulation_granger"; NDRAW, MINDIST, T = 20, 1_000_000, 0.1
CTRL = sys.argv[sys.argv.index("--ctrl") + 1] if "--ctrl" in sys.argv else "random"
SUF = "" if CTRL == "random" else f"_{CTRL}"
PAIRS = [("gene", "AB"), ("gene", "IN"), ("AB", "IN")]


def signs(V, lab, d):
    sl, _ = fit_gauss(V, lab, d, t=T); return np.sign(sl)                        # features x domains


def stats(Sg, Sa, Si):
    out = {}
    for a, b in PAIRS:
        X, Y = {"gene": Sg, "AB": Sa, "IN": Si}[a], {"gene": Sg, "AB": Sa, "IN": Si}[b]
        m = (X != 0) & (Y != 0); out[f"{a}-{b}"] = (float((X[m] == Y[m]).mean()) if m.any() else np.nan, int(m.sum()))
    m = (Sg != 0) & (Sa != 0) & (Si != 0)
    pats = {f"{pg:+d}{pa:+d}{pi:+d}": float(((Sg == pg) & (Sa == pa) & (Si == pi) & m).sum() / max(m.sum(), 1))
            for pg, pa, pi in itertools.product((1, -1), repeat=3)}
    out["n3"] = int(m.sum())
    # consistency independent of the arbitrary signs: gene and A/B agree AND insulation opposes them
    out["g=AB,IN opp"] = float((((Sg == Sa) & (Si == -Sg)) & m).sum() / max(m.sum(), 1))
    return out, pats


def main():
    C = np.load(f"{G}/cache_series.npz", allow_pickle=True); bc = list(C["bc"].astype(str))
    ax = pd.read_csv(f"{R}/{DS}/axes.csv", index_col=0).loc[bc]; z = dict(np.load(f"{R}/targets_{DS}.npz", allow_pickle=True))
    S, AB, IN = (np.asarray(C[k], float) for k in ("S", "AB", "IN")); names = C["names"].astype(str)
    ch = np.array([n.split(":")[0] for n in names]); st = np.array([int(n.split(":")[1].split("-")[0]) for n in names])
    rng = np.random.default_rng(0); prs, pts, units = [], [], []
    for a in ("RNA", "HiC"):
        d = ax[f"d_{a}"].values; Y = z[f"Y_{a}"]; lab = gaston_domains(d, Y, L_=kneedle(curve(d, Y))); K = lab.max() + 1
        Sg, Sa, Si = signs(S, lab, d), signs(AB, lab, d), signs(IN, lab, d)
        pool = [np.where((ch != ch[j]) | (np.abs(st - st[j]) > MINDIST))[0] if CTRL == "random"
                else np.where((ch == ch[j]) & (np.abs(st - st[j]) > MINDIST))[0] for j in range(len(names))]
        dist = [np.array([rng.choice(pool[j]) for j in range(len(names))]) for _ in range(NDRAW)]
        for k in list(range(K)) + ["all"]:
            sel = (slice(None), [k]) if k != "all" else (slice(None), slice(None))
            o, p = stats(Sg[sel], Sa[sel], Si[sel])
            ctrl = [stats(Sg[sel], Sa[dj][sel], Si[dj][sel]) for dj in dist]
            for key in [f"{x}-{y}" for x, y in PAIRS]:
                cv = np.array([c[0][key][0] for c in ctrl])
                prs.append(dict(axis=a, domain=k, pair=key, n=o[key][1], same_sign=o[key][0], control=float(np.nanmean(cv)),
                                ctrl_sd=float(np.nanstd(cv)), p=float((1 + (cv >= o[key][0]).sum()) / (NDRAW + 1))))
            cv = np.array([c[0]["g=AB,IN opp"] for c in ctrl])
            prs.append(dict(axis=a, domain=k, pair="three-way (gene=A/B, insulation opposite)", n=o["n3"],
                            same_sign=o["g=AB,IN opp"], control=float(cv.mean()), ctrl_sd=float(cv.std()),
                            p=float((1 + (cv >= o["g=AB,IN opp"]).sum()) / (NDRAW + 1))))
            for pat, v in p.items():
                cv = np.array([c[1][pat] for c in ctrl])
                pts.append(dict(axis=a, domain=k, pattern=pat, share=v, control=float(cv.mean()), n3=o["n3"]))
        for j in range(len(names)):
            for k in range(K):
                units.append(dict(axis=a, domain=k, bin=names[j], genes=str(C["genes"][j]), gene=Sg[j, k], AB=Sa[j, k], IN=Si[j, k]))
        print(a, pd.DataFrame(prs)[lambda x: (x.axis == a) & (x.domain == "all")][["pair", "n", "same_sign", "control", "p"]].round(3).to_string(index=False), flush=True)
    pd.DataFrame(prs).to_csv(f"{G}/cis_threeway_pairs{SUF}.csv", index=False)
    pd.DataFrame(pts).to_csv(f"{G}/cis_threeway_patterns{SUF}.csv", index=False)
    pd.DataFrame(units).to_csv(f"{G}/cis_threeway_units{SUF}.csv", index=False)
    print("->", G)


if __name__ == "__main__":
    main()
