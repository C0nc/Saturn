"""Saturn targets for Sprox-seq (Wang, ... Tay, Cell 2026; GSE304749), human tonsil.

Sprox-seq measures three layers on the same Visium spot: mRNA, surface proteins, and
PROTEIN COMPLEXES via proximity ligation (PLA products between antibody pairs). The
design here keeps the third layer out of the fit on purpose:

  INPUT       RNA      Visium counts -> GLM-PCA
              Protein  per-protein PLA marginal (every product containing that
                       protein) -> CLR -> PCA, the ADT convention
  VALIDATION  the 1,678 PLA products themselves -- protein complexes and the
              cell-cell interactions they mediate (VLA-4/VCAM1, LFA1/ICAM1). Never
              seen by the model, so agreement with the fitted axes is a real test
              rather than a restatement of the input.

A PLA row is named `<probeX>:<probeY>` where a probe is `<protein>` or `<protein>_A`
/`_B`. Isotype controls (mIgG1/2a/2b, HIgG) are dropped from the protein modality and
kept as QC.

Usage: python scripts/sproxseq_prep.py [A1|B1|D1]
"""
import gzip, os, re, sys
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.io import mmread
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/saturn_repo/src")
from prep_real2 import lognorm, save, H
from saturn.preprocess import rna_glmpc_real, protein_clr_pca

D = f"{H}/external_data/sproxseq"
GSM_RNA = {"A1": "GSM9156293", "B1": "GSM9156294", "D1": "GSM9156297"}
GSM_PLA = {"A1": "GSM9156300", "B1": "GSM9156303", "D1": "GSM9156305"}
CTRL = {"mIgG1", "mIgG2a", "mIgG2b", "HIgG"}
MARKERS = ["MS4A1", "CD3E", "CD19", "AICDA", "BCL6", "CR2", "FDCSP", "CXCL13", "MKI67",
           "CD27", "PRDM1", "VCAM1", "ITGA4", "ICAM1"]


def probe_protein(tok):
    return re.sub(r"_[AB]$", "", tok)


def load_pla(sec):
    df = pd.read_csv(f"{D}/{GSM_PLA[sec]}_{sec}-PLA_count_matrix.txt.gz", sep="\t", index_col=0)
    pairs = [(probe_protein(i.split(":")[0]), probe_protein(i.split(":")[1])) for i in df.index]
    return df, pairs


def run(sec="A1", probe_level=False):
    """probe_level=True keeps all 46 probe tokens (CD29_A and CD29_B separate, isotype
    controls included) -- the authors' own namespace, for modelling the whole protein
    structure of the tissue. False collapses to 29 targets (counts_A + counts_B)."""
    # ---- RNA: standard Visium triplet ---------------------------------------
    g = GSM_RNA[sec]
    bcs = [l.strip() for l in gzip.open(f"{D}/{g}_{sec}_barcodes.tsv.gz", "rt")]
    feat = pd.read_csv(f"{D}/{g}_{sec}_features.tsv.gz", sep="\t", header=None)
    X = sp.csr_matrix(mmread(f"{D}/{g}_{sec}_matrix.mtx.gz").T)          # spots x genes
    genes = np.array([str(v) for v in feat[1]])
    posf = f"{D}/{g}_{sec}_tissue_positions.csv.gz"
    pos = pd.read_csv(posf)
    if pos.columns[0].lower().startswith("barcode"):
        pos = pos.set_index(pos.columns[0])
    else:
        pos = pd.read_csv(posf, header=None,
                          names=["bc", "in_tissue", "row", "col", "py", "px"]).set_index("bc")
    pos.columns = [c.lower() for c in pos.columns]
    ycol = [c for c in pos.columns if "row_in" in c or c == "py"][0]
    xcol = [c for c in pos.columns if "col_in" in c or c == "px"][0]
    intis = [c for c in pos.columns if "in_tissue" in c][0]

    # ---- PLA: protein marginals + the interaction layer ----------------------
    pla, pairs = load_pla(sec)
    common = [b for b in bcs if b in set(pla.columns)]
    common = [b for b in common if b in pos.index and int(pos.loc[b, intis]) == 1]
    print(f"{sec}: Visium {len(bcs)} spots, PLA {pla.shape[1]} spots, in-tissue overlap {len(common)}",
          flush=True)
    Xr = X[[bcs.index(b) for b in common]]
    Pl = pla[common]
    if probe_level:
        raw = [(i.split(":")[0], i.split(":")[1]) for i in pla.index]
        prots = sorted({t for ab in raw for t in ab})
        M = np.zeros((len(common), len(prots)))
        for j, p in enumerate(prots):                       # counts_A + counts_B, as the authors do
            M[:, j] = (Pl.values[[i for i, (a, _) in enumerate(raw) if a == p]].sum(0)
                       + Pl.values[[i for i, (_, b) in enumerate(raw) if b == p]].sum(0))
    else:
        prots = sorted({p for ab in pairs for p in ab} - CTRL)
        M = np.zeros((len(common), len(prots)))
        for j, p in enumerate(prots):
            sel = [i for i, (a, b) in enumerate(pairs) if a == p or b == p]
            M[:, j] = Pl.values[sel].sum(0)
    print(f"   protein features {len(prots)} ({'probe level, controls kept' if probe_level else 'targets, controls dropped'}); "
          f"PLA products {len(pairs)}; median PLA per spot {np.median(Pl.values.sum(0)):.0f}",
          flush=True)

    Yr, ir = rna_glmpc_real(Xr, K=10)
    Yp, ip = protein_clr_pca(sp.csr_matrix(M), K=10)
    coords = np.c_[pos.loc[common, xcol].values.astype(float),
                   -pos.loc[common, ycol].values.astype(float)]
    ex = dict(Y_RNA=Yr, Y_Protein=Yp, depth_RNA=ir["depth"], depth_Protein=ip["depth"],
              ann_pla_total=Pl.values.sum(0).astype(np.float32))
    # protein levels as annotations (CLR), for reading the axes
    L = np.log1p(M); Cp = L - L.mean(1, keepdims=True)
    for j, p in enumerate(prots):
        ex[f"mk_prot_{p}"] = Cp[:, j].astype(np.float32)
    ex.update(lognorm(Xr, MARKERS, pd.Index(genes)))
    save(f"sproxseq_{sec}" + ("_p46" if probe_level else ""), coords, common, ["RNA", "Protein"], ex)

    # the validation layer, stored next to the target but NOT part of it
    np.savez_compressed(f"{H}/Hole/gaston_mix_mm/results/real_saturn/sproxseq_{sec}"
                        + ("_p46" if probe_level else "") + "_pla.npz",
                        counts=Pl.values.T.astype(np.float32),          # spots x products
                        products=np.array([f"{a}:{b}" for a, b in pairs], dtype=object),
                        prot_a=np.array([a for a, _ in pairs], dtype=object),
                        prot_b=np.array([b for _, b in pairs], dtype=object),
                        barcodes=np.array(common, dtype=object),
                        protein_marginal=M.astype(np.float32),
                        proteins=np.array(prots, dtype=object))
    print(f"   validation layer -> sproxseq_{sec}_pla.npz", flush=True)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "A1", "--probe" in sys.argv)
