"""Fig 6 Hi-C target from the two chromatin layers themselves: A/B compartment + TAD insulation (500 kb bins).

Replaces the Fast-Higashi embedding of mel_hic_fh_dr50_51 (same 6,108 spots, chip rows 50-51 dropped; RNA target
unchanged). Both layers are the per-spot series already used post hoc in Fig 6 (cache_series.npz: A/B compartment
z-score and Higashi single-spot insulation, 2,558 bins with >= 3 genes).

  1 depth     every bin regressed on a cubic in log Hi-C reads (Higashi read_count_all.npy; single-spot insulation
              follows reads at r 0.96, so without this the axis would be a coverage axis)
  2 scale     each bin z-scored; each layer weighted 1/sqrt(n bins) so A/B and insulation contribute equally
  3 reduce    PCA of [A/B | insulation] -> 10 PCs -> Saturn global scaling (as every other target)
-> results/real_saturn/targets_mel_hic_abins_dr50_51.npz (+ depth_HiC = real log Hi-C reads)
  --keep-depth  skip step 1 (-> targets_mel_hic_abins_dr50_51_depth.npz): Hi-C depth differs between tumour and the
                non-tumour band, so regressing it also removes part of that contrast; both fits are kept for comparison
  --rna-dr      also regress log RNA depth out of the RNA GLM-PCs (-> ..._rnadr)
Usage: python scripts/mel_hic_abins_prep.py [--keep-depth] [--rna-dr]
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "<workspace>/saturn_repo/src")
from saturn.preprocess import scale_target

R = "<workspace>/Hole/gaston_mix_mm/results/real_saturn"; HG = "<workspace>/SpHIC117/HIC/higashi"
SRC = "mel_hic_fh_dr50_51"
KEEP_DEPTH = "--keep-depth" in sys.argv          # depth is partly biology here (non-tumour band ~0.4x Hi-C reads)
RNA_DR = "--rna-dr" in sys.argv                  # also regress log RNA depth out of the RNA target
DST = "mel_hic_abins_dr50_51" + ("_depth" if KEEP_DEPTH else "") + ("_rnadr" if RNA_DR else "")


def resid(M, r):
    z = (r - r.mean()) / r.std(); X = np.c_[np.ones_like(z), z, z ** 2, z ** 3]
    return M - X @ np.linalg.lstsq(X, M, rcond=None)[0]


def main():
    z = dict(np.load(f"{R}/targets_{SRC}.npz", allow_pickle=True)); bc = z["barcodes"].astype(str)
    C = np.load(f"{R}/{SRC}_insulation_granger/cache_series.npz", allow_pickle=True)
    assert list(C["bc"].astype(str)) == list(bc)
    fl = [l.strip().split("/")[-1].split(".")[0] for l in open(f"{HG}/filelist.txt")]
    rc = np.load(f"{HG}/read_count_all.npy")[pd.Index(fl).get_indexer(pd.Index(bc))].astype(float)   # log reads
    blocks = []
    for k in ("AB", "IN"):
        M = np.asarray(C[k], float)
        if not KEEP_DEPTH: M = resid(M, rc)
        sd = M.std(0); M = M[:, sd > 1e-9]
        M = (M - M.mean(0)) / M.std(0); blocks.append(M / np.sqrt(M.shape[1]))
        print(f"{k}: {M.shape[1]} bins after depth regression", flush=True)
    X = np.hstack(blocks); X -= X.mean(0)
    U, S, _ = np.linalg.svd(X, full_matrices=False); P = (U * S)[:, :10]
    ev = S ** 2 / (S ** 2).sum()
    print(f"10 PCs = {100 * ev[:10].sum():.1f}% of variance; |r| with log reads "
          f"{[round(abs(np.corrcoef(P[:, i], rc)[0, 1]), 2) for i in range(5)]}", flush=True)
    z["Y_HiC"] = scale_target(P, "global").astype(np.float32); z["depth_HiC"] = rc.astype(np.float32)
    if RNA_DR:                                     # without it the RNA axis falls onto depth (Spearman 0.54) once the
        ld = np.log(np.asarray(z["depth_RNA"], float).clip(1))   # epidermis -> tumour gradient is taken by Hi-C
        z["Y_RNA"] = scale_target(resid(np.asarray(z["Y_RNA"], float), ld), "global").astype(np.float32)
        print("RNA target: cubic in log RNA depth regressed out", flush=True)
    np.savez_compressed(f"{R}/targets_{DST}.npz", **z); print("->", f"{R}/targets_{DST}.npz")


if __name__ == "__main__":
    main()
